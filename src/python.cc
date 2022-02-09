// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <boost/functional/hash.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "ipc_message.h"
#include "message_queue.h"
#include "pb_env.h"
#include "pb_map.h"
#include "pb_metric_reporter.h"
#include "pb_utils.h"
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#define LOG_IF_EXCEPTION(X)                                     \
  do {                                                          \
    try {                                                       \
      (X);                                                      \
    }                                                           \
    catch (const PythonBackendException& pb_exception) {        \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what()); \
    }                                                           \
  } while (false)

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X)      \
  do {                                                                      \
    TRITONSERVER_Error* raasnie_err__ = (X);                                \
    if (raasnie_err__ != nullptr) {                                         \
      for (size_t ridx = 0; ridx < RESPONSES_COUNT; ++ridx) {               \
        if ((*RESPONSES)[ridx] != nullptr) {                                \
          LOG_IF_ERROR(                                                     \
              TRITONBACKEND_ResponseSend(                                   \
                  (*RESPONSES)[ridx], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  raasnie_err__),                                           \
              "failed to send error response");                             \
          (*RESPONSES)[ridx] = nullptr;                                     \
        }                                                                   \
      }                                                                     \
      TRITONSERVER_ErrorDelete(raasnie_err__);                              \
      return;                                                               \
    }                                                                       \
  } while (false)


#define RESPOND_ALL_AND_RETURN_IF_EXCEPTION(RESPONSES, RESPONSES_COUNT, X)  \
  do {                                                                      \
    try {                                                                   \
      (X);                                                                  \
    }                                                                       \
    catch (const PythonBackendException& exception) {                       \
      TRITONSERVER_Error* raarie_err__ = TRITONSERVER_ErrorNew(             \
          TRITONSERVER_ERROR_INTERNAL, exception.what());                   \
      for (size_t ridx = 0; ridx < RESPONSES_COUNT; ++ridx) {               \
        if ((*RESPONSES)[ridx] != nullptr) {                                \
          LOG_IF_ERROR(                                                     \
              TRITONBACKEND_ResponseSend(                                   \
                  (*RESPONSES)[ridx], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  raarie_err__),                                            \
              "failed to send error response");                             \
          (*RESPONSES)[ridx] = nullptr;                                     \
        }                                                                   \
      }                                                                     \
      TRITONSERVER_ErrorDelete(raarie_err__);                               \
      return;                                                               \
    }                                                                       \
  } while (false)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                      \
  do {                                                                   \
    if ((*RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                   \
      if (err__ != nullptr) {                                            \
        LOG_IF_ERROR(                                                    \
            TRITONBACKEND_ResponseSend(                                  \
                (*RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                  \
            "failed to send error response");                            \
        (*RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                 \
      }                                                                  \
    }                                                                    \
  } while (false)

#define GUARDED_RESPOND_IF_EXCEPTION(RESPONSES, IDX, X)                  \
  do {                                                                   \
    if ((*RESPONSES)[IDX] != nullptr) {                                  \
      try {                                                              \
        (X);                                                             \
      }                                                                  \
      catch (const PythonBackendException& pb_exception) {               \
        TRITONSERVER_Error* err__ = TRITONSERVER_ErrorNew(               \
            TRITONSERVER_ERROR_INTERNAL, pb_exception.what());           \
        LOG_IF_ERROR(                                                    \
            TRITONBACKEND_ResponseSend(                                  \
                (*RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                  \
            "failed to send error response");                            \
        (*RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                 \
      }                                                                  \
    }                                                                    \
  } while (false)

#define RETURN_IF_EXCEPTION(X)                                 \
  do {                                                         \
    try {                                                      \
      (X);                                                     \
    }                                                          \
    catch (const PythonBackendException& pb_exception) {       \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew( \
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());   \
      return rarie_err__;                                      \
    }                                                          \
  } while (false)

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

struct BackendState {
  std::string python_lib;
  int64_t shm_default_byte_size;
  int64_t shm_growth_byte_size;
  int64_t stub_timeout_seconds;
  int64_t shm_message_queue_size;
  std::atomic<int> number_of_instance_inits;
  std::string shared_memory_region_prefix;
  std::unique_ptr<EnvironmentManager> env_manager;
};

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get backend state
  BackendState* StateForBackend() { return backend_state_; }

  // Get the Python execution environment
  std::string PythonExecutionEnv() { return python_execution_env_; }

  // Force CPU only tensors
  bool ForceCPUOnlyInputTensors() { return force_cpu_only_input_tensors_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  BackendState* backend_state_;
  std::string python_execution_env_;
  bool force_cpu_only_input_tensors_;
};


class ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance);

  TRITONBACKEND_Model* triton_model_;
  bi::interprocess_mutex* health_mutex_;
  std::unique_ptr<MessageQueue> stub_message_queue_;
  std::unique_ptr<MessageQueue> parent_message_queue_;
  std::string model_path_;
  std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>
      ipc_control_;
  bi::managed_external_buffer::handle_t ipc_control_offset_;
  std::vector<std::future<void>> bls_futures_;
  std::vector<TRITONSERVER_InferenceResponse*> bls_inference_responses_;
  std::mutex bls_responses_mutex_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  std::string shm_region_name_;
  off_t shm_reset_offset_;

  // Stub process pid
  pid_t stub_pid_;

  // Parent process pid
  pid_t parent_pid_;
  bool initialized_;

  // Path to python execution environment
  std::string path_to_libpython_;
  std::string path_to_activate_;

 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  ~ModelInstanceState();

  // Create the stub process.
  TRITONSERVER_Error* SetupStubProcess();
  TRITONSERVER_Error* SendMessageToStub(off_t message);

  // Checks whether the stub process is live
  bool IsStubProcessAlive();

  // Get a message from the stub process
  TRITONSERVER_Error* ReceiveMessageFromStub(off_t& message);

  // Get a message from the stub process
  void SendMessageAndReceiveResponse(
      off_t message, off_t& response, bool& restart,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Responds to all the requests with an error message.
  void RespondErrorToAllRequests(
      const char* message,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Kill stub process
  void KillStubProcess();

  // Start stub process
  TRITONSERVER_Error* StartStubProcess();
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance), stub_pid_(0),
      initialized_(false)
{
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;
}

void
ModelInstanceState::KillStubProcess()
{
  kill(stub_pid_, SIGKILL);
  int status;
  waitpid(stub_pid_, &status, 0);
  stub_pid_ = 0;
}

void
ModelInstanceState::SendMessageAndReceiveResponse(
    bi::managed_external_buffer::handle_t message,
    bi::managed_external_buffer::handle_t& response, bool& restart,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  auto error = SendMessageToStub(message);
  if (error != nullptr) {
    restart = true;
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  bi::managed_external_buffer::handle_t response_message;
  error = ReceiveMessageFromStub(response_message);
  if (error != nullptr) {
    restart = true;
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  response = response_message;
}

TRITONSERVER_Error*
ModelInstanceState::SendMessageToStub(
    bi::managed_external_buffer::handle_t message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

      // Check if lock has been acquired.
      if (lock) {
        ipc_control_->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    stub_message_queue_->Push(
        message, timeout_miliseconds /* duration ms */, success);

    if (!success && !IsStubProcessAlive()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Stub process is not healthy.");
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ReceiveMessageFromStub(
    bi::managed_external_buffer::handle_t& message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

      // Check if lock has been acquired.
      if (lock) {
        ipc_control_->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    message = parent_message_queue_->Pop(
        timeout_miliseconds /* duration ms */, success);

    if (!success && !IsStubProcessAlive()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Stub process is not healthy.");
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::RespondErrorToAllRequests(
    const char* message,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    if ((*responses)[r] == nullptr)
      continue;

    std::string err_message =
        std::string(
            "Failed to process the request(s) for model instance '" + Name() +
            "', message: ") +
        message;

    TRITONSERVER_Error* err =
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message.c_str());
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");

    (*responses)[r] = nullptr;
    TRITONSERVER_ErrorDelete(err);
  }
}

bool
ModelInstanceState::IsStubProcessAlive()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::seconds(1);
  bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

  // Check if lock has been acquired.
  if (lock) {
    return ipc_control_->stub_health;
  } else {
    // If It failed to obtain the lock, it means that the stub has been
    // stuck or exited while holding the health mutex lock.
    return false;
  }
}

TRITONSERVER_Error*
ModelInstanceState::StartStubProcess()
{
  new (&(ipc_control_->stub_health_mutex)) bi::interprocess_mutex;
  stub_message_queue_->ResetSemaphores();
  parent_message_queue_->ResetSemaphores();

  std::string kind = TRITONSERVER_InstanceGroupKindString(kind_);

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int64_t shm_growth_size =
      model_state->StateForBackend()->shm_growth_byte_size;
  int64_t shm_default_size =
      model_state->StateForBackend()->shm_default_byte_size;
  const char* model_path = model_state->RepositoryPath().c_str();

  initialized_ = false;

  pid_t pid = fork();
  if (pid < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Failed to fork the stub process.");
  }

  // Stub process
  if (pid == 0) {
    const char* stub_args[4];
    stub_args[0] = "bash";
    stub_args[1] = "-c";
    stub_args[3] = nullptr;  // Last argument must be nullptr

    // Default Python backend stub
    std::string python_backend_stub =
        model_state->StateForBackend()->python_lib +
        "/triton_python_backend_stub";

    // Path to alternative Python backend stub
    std::string model_python_backend_stub =
        std::string(model_path) + "/triton_python_backend_stub";

    if (FileExists(model_python_backend_stub)) {
      python_backend_stub = model_python_backend_stub;
    }

    std::string bash_argument;

    // This shared memory variable indicates whether the stub process should
    // revert the LD_LIBRARY_PATH changes to avoid shared library issues in
    // executables and libraries.
    ipc_control_->uses_env = false;
    if (model_state->PythonExecutionEnv() != "") {
      std::stringstream ss;

      // Need to properly set the LD_LIBRARY_PATH so that Python environments
      // using different python versions load properly.
      ss << "source " << path_to_activate_
         << " && exec env LD_LIBRARY_PATH=" << path_to_libpython_
         << ":$LD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
         << " " << shm_region_name_ << " " << shm_default_size << " "
         << shm_growth_size << " " << parent_pid_ << " "
         << model_state->StateForBackend()->python_lib << " "
         << ipc_control_offset_ << " " << Name();
      ipc_control_->uses_env = true;
      bash_argument = ss.str();
    } else {
      std::stringstream ss;
      ss << " exec " << python_backend_stub << " " << model_path_ << " "
         << shm_region_name_ << " " << shm_default_size << " "
         << shm_growth_size << " " << parent_pid_ << " "
         << model_state->StateForBackend()->python_lib << " "
         << ipc_control_offset_ << " " << Name();
      bash_argument = ss.str();
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Starting Python backend stub: ") + bash_argument)
            .c_str());

    stub_args[2] = bash_argument.c_str();

    int stub_status_code =
        system((python_backend_stub + "> /dev/null 2>&1").c_str());

    // If running stub process without any arguments returns any status code,
    // other than 1, it can indicate a permission issue as a result of
    // downloading the stub process from a cloud object storage service.
    if (WEXITSTATUS(stub_status_code) != 1) {
      // Give the execute permission for the triton_python_backend_stub to the
      // owner.
      int error = chmod(python_backend_stub.c_str(), S_IXUSR);
      if (error != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("Failed to give execute permission to "
                         "triton_python_backend_stub in ") +
             python_backend_stub + " " + Name() +
             " Error No.: " + std::to_string(error))
                .c_str());
      }
    }

    if (execvp("bash", (char**)stub_args) != 0) {
      std::stringstream ss;
      ss << "Failed to run python backend stub. Errno = " << errno << '\n'
         << "Python backend stub path: " << python_backend_stub << '\n'
         << "Shared Memory Region Name: " << shm_region_name_ << '\n'
         << "Shared Memory Default Byte Size: " << shm_default_size << '\n'
         << "Shared Memory Growth Byte Size: " << shm_growth_size << '\n';
      std::string log_message = ss.str();
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, log_message.c_str());

      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to initialize model instance ") + Name())
              .c_str());
    }

  } else {
    stub_pid_ = pid;
    triton::common::TritonJson::WriteBuffer buffer;
    Model()->ModelConfig().Write(&buffer);

    std::unordered_map<std::string, std::string> initialize_map = {
        {"model_config", buffer.MutableContents()},
        {"model_instance_kind", TRITONSERVER_InstanceGroupKindString(kind_)},
        {"model_instance_name", name_},
        {"model_instance_device_id", std::to_string(device_id_)},
        {"model_repository", model_state->RepositoryPath()},
        {"model_version", std::to_string(model_state->Version())},
        {"model_name", model_state->Name()}};

    std::unique_ptr<IPCMessage> initialize_message =
        IPCMessage::Create(shm_pool_, false /* inline_response */);
    initialize_message->Command() = PYTHONSTUB_InitializeRequest;

    std::unique_ptr<PbMap> pb_map = PbMap::Create(shm_pool_, initialize_map);
    bi::managed_external_buffer::handle_t initialize_map_offset =
        pb_map->ShmOffset();

    // The stub process will be the owner of the map.
    pb_map->Release();

    initialize_message->Args() = initialize_map_offset;
    stub_message_queue_->Push(initialize_message->ShmOffset());

    // The stub process will be the owner of the message.
    initialize_message->Release();

    std::unique_ptr<IPCMessage> initialize_response_message =
        IPCMessage::LoadFromSharedMemory(
            shm_pool_, parent_message_queue_->Pop());

    if (initialize_response_message->Command() !=
        PYTHONSTUB_InitializeResponse) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string(
               "Received unexpected resposne from Python backend stub: ") +
           name_)
              .c_str());
    }

    auto initialize_response = (shm_pool_->Load<InitializeResponseShm>(
                                    initialize_response_message->Args()))
                                   .data_;

    if (initialize_response->response_has_error) {
      if (initialize_response->response_is_error_set) {
        std::unique_ptr<PbString> error_message =
            PbString::LoadFromSharedMemory(
                shm_pool_, initialize_response->response_error);
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, error_message->String());
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("Initialize() failed for ") + model_state->Name())
                .c_str());
      }
    }

    initialized_ = true;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::SetupStubProcess()
{
  std::string kind = TRITONSERVER_InstanceGroupKindString(kind_);
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  // Increase the stub process count to avoid shared memory region name
  // collision
  model_state->StateForBackend()->number_of_instance_inits++;
  shm_region_name_ =
      model_state->StateForBackend()->shared_memory_region_prefix +
      std::to_string(model_state->StateForBackend()->number_of_instance_inits);
  int64_t shm_default_size =
      model_state->StateForBackend()->shm_default_byte_size;

  try {
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, shm_default_size, true /* create */);
  }
  catch (const PythonBackendException& pb_exception) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
  }

  AllocatedSharedMemory<IPCControlShm> ipc_control =
      shm_pool_->Construct<IPCControlShm>();
  ipc_control_ = std::move(ipc_control.data_);
  ipc_control_offset_ = ipc_control.handle_;

  uint64_t model_version = model_state->Version();
  const char* model_path = model_state->RepositoryPath().c_str();

  std::stringstream ss;
  std::string artifact_name;
  RETURN_IF_ERROR(model_state->ModelConfig().MemberAsString(
      "default_model_filename", &artifact_name));
  ss << model_path << "/" << model_version << "/";

  if (artifact_name.size() > 0) {
    ss << artifact_name;
  } else {
    // Default artifact name.
    ss << "model.py";
  }

  model_path_ = ss.str();
  struct stat buffer;

  // Check if model.py exists
  if (stat(model_path_.c_str(), &buffer) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("model.py does not exist in the model repository path: " + model_path_)
            .c_str());
  }

  // Path to the extracted Python env
  std::string python_execution_env = "";
  if (model_state->PythonExecutionEnv() != "") {
    try {
      python_execution_env =
          model_state->StateForBackend()->env_manager->ExtractIfNotExtracted(
              model_state->PythonExecutionEnv());
    }
    catch (PythonBackendException& pb_exception) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
    }

    path_to_activate_ = python_execution_env + "/bin/activate";
    path_to_libpython_ = python_execution_env + "/lib";
    if (python_execution_env.length() > 0 && !FileExists(path_to_activate_)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Path ") + path_to_activate_ +
           " does not exist. The Python environment should contain an "
           "'activate' script.")
              .c_str());
    }
  }

  parent_pid_ = getpid();

  auto message_queue_size =
      model_state->StateForBackend()->shm_message_queue_size;

  RETURN_IF_EXCEPTION(
      stub_message_queue_ =
          MessageQueue::Create(shm_pool_, message_queue_size));
  RETURN_IF_EXCEPTION(
      parent_message_queue_ =
          MessageQueue::Create(shm_pool_, message_queue_size));
  ipc_control_->parent_message_queue = parent_message_queue_->ShmOffset();
  ipc_control_->stub_message_queue = stub_message_queue_->ShmOffset();

  RETURN_IF_ERROR(StartStubProcess());

  return nullptr;
}

ModelInstanceState::~ModelInstanceState()
{
  if (initialized_) {
    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      ipc_control_->stub_health = false;
    }

    // Sleep 1 second so that the child process has a chance to change the
    // health variable
    sleep(1);

    bool healthy = false;
    bool force_kill = false;
    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      healthy = ipc_control_->stub_health;
    }

    if (healthy) {
      // Finalize command does not have any arguments.
      std::unique_ptr<IPCMessage> ipc_message =
          IPCMessage::Create(shm_pool_, false /* inline_response */);

      ipc_message->Command() = PYTHONSTUB_FinalizeRequest;
      stub_message_queue_->Push(ipc_message->ShmOffset());
      parent_message_queue_->Pop();

      stub_message_queue_.reset();
      parent_message_queue_.reset();

    } else {
      force_kill = true;
    }

    int status;
    if (force_kill) {
      kill(stub_pid_, SIGKILL);
    }
    waitpid(stub_pid_, &status, 0);
  }
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));

  const char* path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
  python_execution_env_ = "";
  force_cpu_only_input_tensors_ = true;

  void* bstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  backend_state_ = reinterpret_cast<BackendState*>(bstate);
  triton::common::TritonJson::Value params;
  if (model_config_.Find("parameters", &params)) {
    // Skip the EXECUTION_ENV_PATH variable if it doesn't exist.
    TRITONSERVER_Error* error =
        GetParameterValue(params, "EXECUTION_ENV_PATH", &python_execution_env_);
    if (error == nullptr) {
      std::string relative_path_keyword = "$$TRITON_MODEL_DIRECTORY";
      size_t relative_path_loc =
          python_execution_env_.find(relative_path_keyword);
      if (relative_path_loc != std::string::npos) {
        python_execution_env_.replace(
            relative_path_loc, relative_path_loc + relative_path_keyword.size(),
            path);
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Using Python execution env ") + python_execution_env_)
              .c_str());
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }

    // Skip the FORCE_CPU_ONLY_INPUT_TENSORS variable if it doesn't exits.
    std::string force_cpu_only_input_tensor;
    error = nullptr;
    error = GetParameterValue(
        params, "FORCE_CPU_ONLY_INPUT_TENSORS", &force_cpu_only_input_tensor);
    if (error == nullptr) {
      if (force_cpu_only_input_tensor == "yes") {
        force_cpu_only_input_tensors_ = true;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Forcing CPU only input tensors.")).c_str());
      } else if (force_cpu_only_input_tensor == "no") {
        force_cpu_only_input_tensors_ = false;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Input tensors can be both in CPU and GPU. "
                         "FORCE_CPU_ONLY_INPUT_TENSORS is off."))
                .c_str());
      } else {
        throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Incorrect value for FORCE_CPU_ONLY_INPUT_TENSORS: ") +
             force_cpu_only_input_tensor + "'")
                .c_str()));
      }
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }
  }

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + Name() + "'")
            .c_str()));
  }
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  // Check backend version to ensure compatibility
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState> backend_state(new BackendState());
  triton::common::TritonJson::Value cmdline;
  backend_state->shm_default_byte_size = 64 * 1024 * 1024;  // 64 MBs
  backend_state->shm_growth_byte_size = 64 * 1024 * 1024;   // 64 MBs
  backend_state->stub_timeout_seconds = 30;
  backend_state->shm_message_queue_size = 1000;
  backend_state->number_of_instance_inits = 0;
  backend_state->shared_memory_region_prefix =
      "triton_python_backend_shm_region_";

  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value shm_growth_size;
    std::string shm_growth_byte_size;
    if (cmdline.Find("shm-growth-byte-size", &shm_growth_size)) {
      RETURN_IF_ERROR(shm_growth_size.AsString(&shm_growth_byte_size));
      try {
        backend_state->shm_growth_byte_size = std::stol(shm_growth_byte_size);
        if (backend_state->shm_growth_byte_size <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-growth-byte-size") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_default_size;
    std::string shm_default_byte_size;
    if (cmdline.Find("shm-default-byte-size", &shm_default_size)) {
      RETURN_IF_ERROR(shm_default_size.AsString(&shm_default_byte_size));
      try {
        backend_state->shm_default_byte_size = std::stol(shm_default_byte_size);
        // Shared memory default byte size can't be less than 4 MBs.
        if (backend_state->shm_default_byte_size < 4 * 1024 * 1024) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-default-byte-size") +
               " can't be smaller than 4 MiBs")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_region_prefix;
    std::string shm_region_prefix_str;
    if (cmdline.Find("shm-region-prefix-name", &shm_region_prefix)) {
      RETURN_IF_ERROR(shm_region_prefix.AsString(&shm_region_prefix_str));
      // Shared memory default byte size can't be less than 4 MBs.
      if (shm_region_prefix_str.size() == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("shm-region-prefix-name") +
             " must at least contain one character.")
                .c_str());
      }
      backend_state->shared_memory_region_prefix = shm_region_prefix_str;
    }

    triton::common::TritonJson::Value shm_message_queue_size;
    std::string shm_message_queue_size_str;
    if (cmdline.Find("shm_message_queue_size", &shm_message_queue_size)) {
      RETURN_IF_ERROR(
          shm_message_queue_size.AsString(&shm_message_queue_size_str));
      try {
        backend_state->shm_message_queue_size =
            std::stol(shm_message_queue_size_str);
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value stub_timeout_seconds;
    std::string stub_timeout_string_seconds;
    if (cmdline.Find("stub-timeout-seconds", &stub_timeout_seconds)) {
      RETURN_IF_ERROR(
          stub_timeout_seconds.AsString(&stub_timeout_string_seconds));
      try {
        backend_state->stub_timeout_seconds =
            std::stol(stub_timeout_string_seconds);
        if (backend_state->stub_timeout_seconds <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("stub-timeout-seconds") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Shared memory configuration is shm-default-byte-size=") +
       std::to_string(backend_state->shm_default_byte_size) +
       ",shm-growth-byte-size=" +
       std::to_string(backend_state->shm_growth_byte_size) +
       ",stub-timeout-seconds=" +
       std::to_string(backend_state->stub_timeout_seconds))
          .c_str());

  // Use BackendArtifacts to determine the location of Python files
  const char* location;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  backend_state->python_lib = location;
  backend_state->env_manager = std::make_unique<EnvironmentManager>();

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release();
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: Start");
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: End");
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_IF_ERROR(instance_state->SetupStubProcess());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}  // extern "C"
}}}  // namespace triton::backend::python
