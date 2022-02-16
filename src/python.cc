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
#include "infer_request.h"
#include "infer_response.h"
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

  TRITONSERVER_Error* GetInputTensor(
      const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
      TRITONBACKEND_Request* request,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses);

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      bool& restart, std::atomic<bool>& cleanup);
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
  health_mutex_ = &(ipc_control_->stub_health_mutex);
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

    auto initialize_response =
        std::move((shm_pool_->Load<InitializeResponseShm>(
                      initialize_response_message->Args())))
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


TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
    TRITONBACKEND_Request* request,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
  const char* input_name;
  // Load iidx'th input name
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInputName(request, input_idx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;

  RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
      in, HostPolicyName().c_str(), &input_name, &input_dtype, &input_shape,
      &input_dims_count, &input_byte_size, &input_buffer_count));

  BackendInputCollector collector(
      &request, 1, responses.get(), Model()->TritonMemoryManager(),
      false /* pinned_enable */, CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  bool cpu_only_tensors = model_state->ForceCPUOnlyInputTensors();
  if (input_dtype == TRITONSERVER_TYPE_BYTES) {
    cpu_only_tensors = true;
  }

#ifdef TRITON_ENABLE_GPU
  CUDADriverAPI& cuda_driver_api = CUDADriverAPI::getInstance();
  // If CUDA driver API is not available, the input tensors will be moved to
  // CPU.
  if (!cuda_driver_api.IsAvailable()) {
    cpu_only_tensors = true;
  }
#endif

  TRITONSERVER_MemoryType src_memory_type;
  int64_t src_memory_type_id;
  size_t src_byte_size;
  const void* src_ptr;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
      in, 0 /* input buffer index */, &src_ptr, &src_byte_size,
      &src_memory_type, &src_memory_type_id));

// If TRITON_ENABLE_GPU is false, we need to copy the tensors
// to the CPU.
#ifndef TRITON_ENABLE_GPU
  cpu_only_tensors = true;
#endif  // TRITON_ENABLE_GPU

  if (cpu_only_tensors || src_memory_type != TRITONSERVER_MEMORY_GPU) {
    input_tensor = std::make_unique<PbTensor>(
        std::string(input_name),
        std::vector<int64_t>(input_shape, input_shape + input_dims_count),
        input_dtype, TRITONSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */, nullptr /* buffer ptr*/, input_byte_size,
        nullptr /* DLManagedTensor */);
    RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(shm_pool_));
    char* input_buffer = reinterpret_cast<char*>(input_tensor->DataPtr());
    collector.ProcessTensor(
        input_name, input_buffer, input_byte_size,
        TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
  } else {
#ifdef TRITON_ENABLE_GPU
    // Retreiving GPU input tensors
    const void* buffer = nullptr;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = {{TRITONSERVER_MEMORY_GPU, src_memory_type_id}};
    RETURN_IF_ERROR(collector.ProcessTensor(
        input_name, nullptr, 0, alloc_perference,
        reinterpret_cast<const char**>(&buffer), &input_byte_size,
        &src_memory_type, &src_memory_type_id));
    input_tensor = std::make_unique<PbTensor>(
        std::string(input_name),
        std::vector<int64_t>(input_shape, input_shape + input_dims_count),
        input_dtype, src_memory_type, src_memory_type_id,
        const_cast<void*>(buffer), input_byte_size,
        nullptr /* DLManagedTensor */);
    RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(shm_pool_));
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Python backend does not support GPU tensors.");
#endif  // TRITON_ENABLE_GPU
  }

  return nullptr;
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    bool& restart, std::atomic<bool>& cleanup)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int max_batch_size = model_state->MaxBatchSize();
  std::string name = model_state->Name();

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.

  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Python backend for '" + name + "'")
                  .c_str()));
      return;
    }
  }

  // We take the responsibility of the responses.
  std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses(
      new std::vector<TRITONBACKEND_Response*>());
  responses->reserve(request_count);
  PbMetricReporter reporter(
      TritonModelInstance(), requests, request_count, responses);
  reporter.SetExecStartNs(exec_start_ns);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses->emplace_back(response);
    } else {
      responses->emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_RETURN_IF_ERROR(responses, request_count, err);
      }
    } else {
      ++total_batch_size;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                name + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
  }

  std::unique_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(shm_pool_, false /*inline_response*/);
  ipc_message->Command() = PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest;

  AllocatedSharedMemory<RequestBatch> request_batch;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      request_batch = shm_pool_->Construct<RequestBatch>());

  request_batch.data_->batch_size = request_count;
  ipc_message->Args() = request_batch.handle_;

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t> requests_shm;

  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      requests_shm =
          shm_pool_->ConstructMany<bi::managed_external_buffer::handle_t>(
              request_count));
  request_batch.data_->requests = requests_shm.handle_;

  std::vector<std::unique_ptr<InferRequest>> pb_inference_requests;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_input_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    uint32_t requested_output_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    std::vector<std::shared_ptr<PbTensor>> pb_input_tensors;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      std::shared_ptr<PbTensor> pb_input_tensor;

      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          GetInputTensor(iidx, pb_input_tensor, request, responses));
      pb_input_tensors.emplace_back(std::move(pb_input_tensor));
    }

    std::vector<std::string> requested_output_names;
    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));
      requested_output_names.emplace_back(requested_output_name);
    }

    // request id
    const char* id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count, TRITONBACKEND_RequestId(request, &id));

    uint64_t correlation_id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    uint32_t flags;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count, TRITONBACKEND_RequestFlags(request, &flags));

    std::unique_ptr<InferRequest> infer_request =
        std::make_unique<InferRequest>(
            id, correlation_id, pb_input_tensors, requested_output_names,
            model_state->Name(), model_state->Version(), flags);

    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        responses, request_count, infer_request->SaveToSharedMemory(shm_pool_));
    pb_inference_requests.emplace_back(std::move(infer_request));
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  reporter.SetComputeStartNs(compute_start_ns);

  // This means that the stub process has exited and Python
  // backend failed to restart the stub process.
  if (stub_pid_ == 0) {
    const char* error_message = "The stub process has exited unexpectedly.";
    RespondErrorToAllRequests(
        error_message, responses, requests, request_count);
    return;
  }

  // Release the ownership of all the shared memory objects.
  ipc_message->Release();
  for (auto& pb_infer_request : pb_inference_requests) {
    pb_infer_request->Release();
  }
  request_batch.data_.release();
  requests_shm.data_.release();

  bi::managed_external_buffer::handle_t response_message;
  SendMessageAndReceiveResponse(
      ipc_message->ShmOffset(), response_message, restart, responses, requests,
      request_count);
  if (restart) {
    return;
  }

  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      ipc_message =
          IPCMessage::LoadFromSharedMemory(shm_pool_, response_message));

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  reporter.SetComputeEndNs(compute_end_ns);

  // Parsing the request response
  AllocatedSharedMemory<ResponseBatch> response_batch;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      response_batch = shm_pool_->Load<ResponseBatch>(ipc_message->Args()));

  // If inference fails, release all the requests and send an error response.
  // If inference fails at this stage, it usually indicates a bug in the model
  // code
  if (response_batch.data_->has_error) {
    if (response_batch.data_->is_error_set) {
      std::unique_ptr<PbString> error_message_shm;
      RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
          responses, request_count,
          error_message_shm = PbString::LoadFromSharedMemory(
              shm_pool_, response_batch.data_->error));
      RespondErrorToAllRequests(
          error_message_shm->String(), responses, requests, request_count);
    } else {
      const char* error_message =
          "Failed to fetch the error in response batch.";
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    }
    return;
  }

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      response_shm_offset;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      response_shm_offset =
          shm_pool_->Load<bi::managed_external_buffer::handle_t>(
              response_batch.data_->responses));

  // bool has_gpu_output = false;

  // // The vector that stores the tensor pairs for the tensors that the stub
  // // provides in GPU but the output buffer provided by Triton is in CPU.
  // std::vector<std::pair<std::shared_ptr<PbTensor>, std::pair<void*,
  // uint32_t>>>
  //     tensor_buffer_pairs;

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = (*responses)[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    std::unique_ptr<InferResponse> infer_response;
    try {
      infer_response = InferResponse::LoadFromSharedMemory(
          shm_pool_, (response_shm_offset.data_.get())[r]);
      if (infer_response->HasError()) {
        // if (infer_response->IsErrorMessageSet()) {
        TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            infer_response->Error()->Message().c_str());

        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed sending response");
        TRITONSERVER_ErrorDelete(err);
        // } else {
        //   const char* err_string = "Failed to process response.";
        //   TRITONSERVER_Error* err =
        //       TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_string);

        //   LOG_IF_ERROR(
        //       TRITONBACKEND_ResponseSend(
        //           (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        //           err),
        //       "failed sending response");
        //   TRITONSERVER_ErrorDelete(err);
        // }

        (*responses)[r] = nullptr;

        // If has_error is true, we do not look at the response even if the
        // response is set.
        continue;
      }
    }
    catch (const PythonBackendException& pb_exception) {
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");
      TRITONSERVER_ErrorDelete(err);
      (*responses)[r] = nullptr;
      continue;
    }

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    bool cuda_copy = false;
    std::set<std::string> requested_output_names;
    for (size_t j = 0; j < requested_output_count; ++j) {
      const char* output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, j, &output_name));
      requested_output_names.insert(output_name);
    }

    for (auto& output_tensor : infer_response->OutputTensors()) {
      if (requested_output_names.find(output_tensor->Name()) ==
          requested_output_names.end()) {
        continue;
      }

      TRITONSERVER_MemoryType src_memory_type = output_tensor->MemoryType();
      int64_t src_memory_type_id = output_tensor->MemoryTypeId();

      TRITONSERVER_MemoryType actual_memory_type = src_memory_type;
      int64_t actual_memory_type_id = src_memory_type_id;

      // if (actual_memory_type == TRITONSERVER_MEMORY_GPU)
      // has_gpu_output = true;

      TRITONBACKEND_Output* response_output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &response_output, output_tensor->Name().c_str(),
              static_cast<TRITONSERVER_DataType>(output_tensor->TritonDtype()),
              output_tensor->Dims().data(), output_tensor->Dims().size()));

      void* buffer;
      bool cuda_used = false;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              response_output, &buffer, output_tensor->ByteSize(),
              &actual_memory_type, &actual_memory_type_id));

      //       if (src_memory_type == TRITONSERVER_MEMORY_GPU &&
      //           actual_memory_type == TRITONSERVER_MEMORY_GPU) {
      // #ifdef TRITON_ENABLE_GPU
      //         cudaSetDevice(output_tensor->MemoryTypeId());
      //         cudaIpcMemHandle_t cuda_ipc_mem_handle;
      //         cudaError_t err = cudaIpcGetMemHandle(&cuda_ipc_mem_handle,
      //         buffer);
      //         output_tensor->SetCudaIpcMemHandle(&cuda_ipc_mem_handle);
      //         output_tensor->SetMemoryType(actual_memory_type);
      //         output_tensor->SetMemoryTypeId(actual_memory_type_id);

      //         if (err != cudaSuccess) {
      //           GUARDED_RESPOND_IF_ERROR(
      //               responses, r,
      //               TRITONSERVER_ErrorNew(
      //                   TRITONSERVER_ERROR_INTERNAL,
      //                   std::string(
      //                       "failed to get cuda ipc handle: " +
      //                       std::string(cudaGetErrorString(err)))
      //                       .c_str()));
      //         } else {
      //           output_tensor->SetDataPtr(buffer);
      //           output_tensor->RawDataShm()->offset =
      //               output_tensor->GetGPUPointerOffset();
      //         }
      // #endif
      //       }

      // if (src_memory_type == TRITONSERVER_MEMORY_GPU &&
      //     (actual_memory_type == TRITONSERVER_MEMORY_CPU ||
      //      actual_memory_type == TRITONSERVER_MEMORY_CPU_PINNED)) {
      //   tensor_buffer_pairs.push_back({output_tensor, {buffer, r}});

      //   // Set the memory type to CPU in shared memory. The stubs notices the
      //   // change in the memory type and should copy the input tensors to
      //   CPU. output_tensor->RawDataShm()->memory_type =
      //   TRITONSERVER_MEMORY_CPU; output_tensor->RawDataShm()->memory_type_id
      //   = actual_memory_type_id;
      // }

      if (src_memory_type != TRITONSERVER_MEMORY_GPU) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            CopyBuffer(
                "Failed to copy the output tensor to buffer.", src_memory_type,
                src_memory_type_id, actual_memory_type, actual_memory_type_id,
                output_tensor->ByteSize(), output_tensor->DataPtr(), buffer,
                CudaStream(), &cuda_used));
      }
      cuda_copy |= cuda_used;
    }
#ifdef TRITON_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
    }
#endif  // TRITON_ENABLE_GPU
  }

  // // If the output tensor is in GPU, there will be a second round trip
  // // required for filling the GPU buffers provided by the main process.
  // if (has_gpu_output) {
  //   ipc_message->Command() =
  //   PYTHONSTUB_CommandType::PYTHONSTUB_LoadGPUBuffers;
  //   SendMessageAndReceiveResponse(
  //       ipc_message->SharedMemoryOffset(), response_message, restart,
  //       responses, requests, 0);

  //   bool cuda_copy = false;
  //   for (auto& tensor_buffer_pair : tensor_buffer_pairs) {
  //     bool cuda_used = false;
  //     auto& tensor = tensor_buffer_pair.first;

  //     // Reload the tensor from shared memory so that the memory data is
  //     // updated.
  //     std::shared_ptr<std::mutex> cuda_ipc_mutex;
  //     std::shared_ptr<PbTensor> reloaded_tensor =
  //         PbTensor::LoadFromSharedMemory(
  //             shm_pool_, tensor->ShmOffset(),
  //             cuda_ipc_mutex /* cuda_ipc_open_mutex */,
  //             cuda_ipc_mutex /* cuda_ipc_close_mutex */);
  //     auto& buffer = tensor_buffer_pair.second.first;
  //     auto& response_index = tensor_buffer_pair.second.second;
  //     GUARDED_RESPOND_IF_ERROR(
  //         responses, response_index,
  //         CopyBuffer(
  //             "Failed to copy the output tensor to buffer.",
  //             TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
  //             reloaded_tensor->ByteSize(), reloaded_tensor->GetDataPtr(),
  //             buffer, CudaStream(), &cuda_used));
  //     cuda_copy |= cuda_used;
  //   }
  // #ifdef TRITON_ENABLE_GPU
  // if (cuda_copy) {
  //   cudaStreamSynchronize(stream_);
  // }
  // #endif  // TRITON_ENABLE_GPU
  // }

  for (uint32_t r = 0; r < request_count; ++r) {
    // If error happens at this stage, we can only log it
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_ResponseSend(
            (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  reporter.SetExecEndNs(exec_end_ns);
  reporter.SetBatchStatistics(total_batch_size);

  return;
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
      ipc_message->Release();
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

  // First destroy the IPCControl. This makes sure that IPCControl is destroyed
  // before the shared memory manager goes out of scope.
  ipc_control_.reset();
  stub_message_queue_.reset();
  parent_message_queue_.reset();
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
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  // If restart is equal to true, it indicates that the stub process is
  // unhealthy and needs a restart.
  bool restart = false;
  std::atomic<bool> cleanup{false};
  instance_state->ProcessRequests(requests, request_count, restart, cleanup);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       instance_state->Name() + " released " + std::to_string(request_count) +
       " requests")
          .c_str());

  if (restart) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "Stub process is unhealthy and it will be restarted.");
    instance_state->KillStubProcess();
    LOG_IF_ERROR(
        instance_state->StartStubProcess(),
        "Failed to restart the stub process.");
  }

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
