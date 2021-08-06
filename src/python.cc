// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "infer_request.h"
#include "infer_response.h"
#include "pb_env.h"
#include "pb_main_utils.h"
#include "pb_tensor.h"
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

#ifdef TRITON_ENABLE_GPU_TENSORS
#include <cuda.h>
#endif  // TRITON_ENABLE_GPU_TENSORS

#define LOG_IF_EXCEPTION(X)                                     \
  do {                                                          \
    try {                                                       \
      (X);                                                      \
    }                                                           \
    catch (const PythonBackendException& pb_exception) {        \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what()); \
    }                                                           \
  } while (false)


#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return nullptr;                                                  \
    }                                                                  \
  } while (false)

#define RESPOND_ALL_AND_RETURN_IF_EXCEPTION(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                     \
    try {                                                                  \
      (X);                                                                 \
    }                                                                      \
    catch (const PythonBackendException& exception) {                      \
      TRITONSERVER_Error* raarie_err__ = TRITONSERVER_ErrorNew(            \
          TRITONSERVER_ERROR_INTERNAL, exception.what());                  \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__);     \
      return nullptr;                                                      \
    }                                                                      \
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

#define RESPOND_AND_RETURN_IF_EXCEPTION(REQUEST, X)                     \
  do {                                                                  \
    try {                                                               \
      (X);                                                              \
    }                                                                   \
    catch (const PythonBackendException& exception) {                   \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew(          \
          TRITONSERVER_ERROR_INTERNAL, exception.what());               \
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

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_EXCEPTION(RESPONSES, IDX, X)                 \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      try {                                                             \
        (X);                                                            \
      }                                                                 \
      catch (const PythonBackendException& pb_exception) {              \
        TRITONSERVER_Error* err__ = TRITONSERVER_ErrorNew(              \
            TRITONSERVER_ERROR_INTERNAL, pb_exception.what());          \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
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
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  off_t ipc_control_offset_;
  std::unique_ptr<bi::scoped_lock<bi::interprocess_mutex>> parent_lock_;
  std::string model_path_;
  IPCMessage* ipc_message_;
  IPCControl* ipc_control_;
  std::vector<TRITONSERVER_InferenceResponse*> bls_inference_responses_;
  std::unique_ptr<SharedMemory> shm_pool_;
  off_t shm_reset_offset_;
  std::unique_ptr<RequestExecutor> request_executor_;
  std::vector<std::unique_ptr<InferResponse>> infer_responses_;

  // Stub process pid
  pid_t stub_pid_;

  // Parent process pid
  pid_t parent_pid_;
  bool initialized_;

  // Path to python execution environment
  std::string path_to_libpython_;
  std::string path_to_activate_;

#ifdef TRITON_ENABLE_GPU_TENSORS
  std::unordered_map<
      std::array<char, sizeof(cudaIpcMemHandle_t)>, void*,
      boost::hash<std::array<char, sizeof(cudaIpcMemHandle_t)>>>
      gpu_tensors_map_;
#endif  // TRITON_ENABLE_GPU_TENSORS
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  ~ModelInstanceState();

  // Load Triton inputs to the appropriate Protobufs
  TRITONSERVER_Error* GetInputTensor(
      const uint32_t input_idx, Tensor* input_tensor_shm,
      std::shared_ptr<PbTensor>& input_tensor, TRITONBACKEND_Request* request,
      std::vector<TRITONBACKEND_Response*>& responses);

  TRITONSERVER_Error* ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Create the stub process.
  TRITONSERVER_Error* SetupStubProcess();
  void CleanupBLSResponses();

  // Notifies the stub process on the new request.  Returns false if the parent
  // process fails to acquire the lock.
  bool NotifyStub();

  // Checks whether the stub process is live
  bool IsStubProcessAlive();

  // Wait for stub notification
  bool WaitForStubNotification();

  // Responds to all the requests with an error message.
  void RespondErrorToAllRequests(
      const char* message, std::vector<TRITONBACKEND_Response*>& responses,
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Kill stub process
  void KillStubProcess();

  // Start stub process
  TRITONSERVER_Error* StartStubProcess();

  // Reset the shared memory offset
  void ResetSharedMemoryOffset();

  // Notify the stub process and wait for the response.
  bool NotifyStubAndWait(bool& restart);
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance), stub_pid_(0),
      initialized_(false)
{
  request_executor_ =
      std::make_unique<RequestExecutor>(model_state->TritonServer());
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
  return nullptr;  // success
}

bool
ModelInstanceState::NotifyStub()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(1000);
  bi::scoped_lock<bi::interprocess_mutex> lock(*stub_mutex_, timeout);

  if (lock) {
    stub_cond_->notify_one();
    return true;
  } else {
    return false;
  }
}

void
ModelInstanceState::KillStubProcess()
{
  kill(stub_pid_, SIGKILL);
  int status;
  waitpid(stub_pid_, &status, 0);
  stub_pid_ = 0;
}

bool
ModelInstanceState::WaitForStubNotification()
{
  uint64_t timeout_miliseconds = 1000;
  boost::posix_time::ptime timeout =
      boost::get_system_time() +
      boost::posix_time::milliseconds(timeout_miliseconds);

  {
    bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

    // Check if lock has been acquired.
    if (lock) {
      ipc_control_->stub_health = false;
    } else {
      // If It failed to obtain the lock, it means that the stub has been
      // stuck or exited while holding the health mutex lock.
      return false;
    }
  }

  timeout = boost::get_system_time() +
            boost::posix_time::milliseconds(timeout_miliseconds);
  while (!parent_cond_->timed_wait(*parent_lock_, timeout)) {
    if (!IsStubProcessAlive()) {
      return false;
    }

    timeout = boost::get_system_time() +
              boost::posix_time::milliseconds(timeout_miliseconds);
  }
  return true;
}

void
ModelInstanceState::RespondErrorToAllRequests(
    const char* message, std::vector<TRITONBACKEND_Response*>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    if (responses[r] == nullptr)
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
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");

    responses[r] = nullptr;
    TRITONSERVER_ErrorDelete(err);
  }
}

void
ModelInstanceState::ResetSharedMemoryOffset()
{
  shm_pool_->SetOffset(shm_reset_offset_);
}

bool
ModelInstanceState::NotifyStubAndWait(bool& restart)
{
  // Notify stub process and wait for the
  bool failed = !NotifyStub() || !WaitForStubNotification();

  // Should we restart the stub process if we fail to notify the stub process?
  if (failed && restart) {
    KillStubProcess();
    const char* error_message = "The stub process has exited unexpectedly.";
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error_message);
    TRITONSERVER_Error* err = StartStubProcess();
    if (err == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO, "Stub process successfully restarted.");

      // Sucessfully restarted the stub process.
      restart = true;
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string(
               "Stub process failed to restart. Your future requests to "
               "model ") +
           name_ + " will fail. Error: " + TRITONSERVER_ErrorMessage(err))
              .c_str());

      // Failed to restart the stub process.
      restart = false;
    }
  }

  return !failed;
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int max_batch_size = model_state->MaxBatchSize();
  std::string name = model_state->Name();

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
      return nullptr;
    }

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
        RequestsRespondWithError(requests, request_count, err);
        return nullptr;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return nullptr;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                name + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return nullptr;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  ipc_message_->command = PYTHONSTUB_CommandType::PYTHONSTUB_Execute;
  ExecuteArgs* exec_args;
  off_t exec_args_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&exec_args, sizeof(ExecuteArgs), exec_args_offset));
  ipc_message_->args = exec_args_offset;

  RequestBatch* request_batch;
  off_t request_batch_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&request_batch, sizeof(RequestBatch), request_batch_offset));
  exec_args->request_batch = request_batch_offset;
  request_batch->batch_size = request_count;

  Request* requests_shm;
  off_t requests_shm_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&requests_shm, sizeof(Request) * request_count,
      requests_shm_offset));
  request_batch->requests = requests_shm_offset;

  // We take the responsibilty of the responses.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response.");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    Request* python_infer_request = &requests_shm[r];
    uint32_t requested_input_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    uint32_t requested_output_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
    python_infer_request->requested_output_count = requested_output_count;

    Tensor* input_tensors;
    off_t input_tensors_offset;

    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        &responses, request_count,
        shm_pool_->Map(
            (char**)&input_tensors, sizeof(Tensor) * requested_input_count,
            input_tensors_offset));
    python_infer_request->inputs = input_tensors_offset;

    std::vector<std::shared_ptr<PbTensor>> pb_input_tensors;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      Tensor* input_tensor = &input_tensors[iidx];
      std::shared_ptr<PbTensor> pb_input_tensor;

      RESPOND_ALL_AND_RETURN_IF_ERROR(
          &responses, request_count,
          GetInputTensor(
              iidx, input_tensor, pb_input_tensor, request, responses));
      pb_input_tensors.emplace_back(std::move(pb_input_tensor));
    }

    std::vector<std::string> requested_output_names;
    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          &responses, request_count,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));
      requested_output_names.emplace_back(requested_output_name);
    }

    // request id
    const char* id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count, TRITONBACKEND_RequestId(request, &id));

    uint64_t correlation_id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    InferRequest infer_request(
        id, correlation_id, pb_input_tensors, requested_output_names,
        model_state->Name(), model_state->Version());
    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        &responses, request_count,
        infer_request.SaveToSharedMemory(shm_pool_, python_infer_request));
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // This means that the stub process has exited and Python
  // backend failed to restart the stub process.
  if (stub_pid_ == 0) {
    const char* error_message = "The stub process has exited unexpectedly.";
    RespondErrorToAllRequests(
        error_message, responses, requests, request_count);

    return nullptr;
  }

  // If parent fails to notify the stub or the stub fails to notify the
  // parent in a timely manner, kill the stub process and restart the
  // stub process.
  bool restart = true;
  if (!NotifyStubAndWait(restart)) {
    RespondErrorToAllRequests(
        "The stub process has exited unexpectedly.", responses, requests,
        request_count);

    return nullptr;
  }

  // If the stub command is no longer PYTHONSTUB_InferExecRequest, it indicates
  // that inference request exeuction has finished.
  while (ipc_message_->stub_command ==
         PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest) {
    bool is_response_batch_set = false;
    ResponseBatch* response_batch;
    TRITONSERVER_InferenceResponse* inference_response = nullptr;
    try {
      ExecuteArgs* exec_args;
      shm_pool_->MapOffset((char**)&exec_args, ipc_message_->stub_args);
      RequestBatch* request_batch;
      shm_pool_->MapOffset((char**)&request_batch, exec_args->request_batch);
      shm_pool_->Map(
          (char**)&response_batch, sizeof(ResponseBatch),
          exec_args->response_batch);
      response_batch->batch_size = 1;
      response_batch->has_error = false;
      response_batch->is_error_set = false;
      response_batch->cleanup = false;
      is_response_batch_set = true;

      if (request_batch->batch_size == 1) {
        std::unique_ptr<InferRequest> infer_request;
        infer_request = InferRequest::LoadFromSharedMemory(
            shm_pool_, request_batch->requests);
        std::unique_ptr<InferResponse> infer_response;
        infer_response = request_executor_->Infer(
            infer_request, shm_pool_, &inference_response);

        if (inference_response != nullptr)
          bls_inference_responses_.push_back(inference_response);

        Response* response;
        shm_pool_->Map(
            (char**)&response, sizeof(Response), response_batch->responses);

        infer_response->SaveToSharedMemory(
            shm_pool_, response, false /* copy */);
      }
    }
    catch (const PythonBackendException& pb_exception) {
      if (is_response_batch_set) {
        response_batch->has_error = true;
        off_t string_offset = 0;
        LOG_IF_EXCEPTION(SaveStringToSharedMemory(
            shm_pool_, string_offset, pb_exception.what()));
        if (string_offset != 0) {
          response_batch->is_error_set = true;
          response_batch->error = string_offset;
        }
      } else {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what());
      }
    }

    bool restart = true;
    if (!NotifyStubAndWait(restart)) {
      RespondErrorToAllRequests(
          "The stub process has exited unexpectedly.", responses, requests,
          request_count);

      return nullptr;
    }
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Parsing the request response
  ResponseBatch* response_batch;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      &responses, request_count,
      shm_pool_->MapOffset((char**)&response_batch, exec_args->response_batch));

  // If inference fails, release all the requests and send an error response. If
  // inference fails at this stage, it usually indicates a bug in the model code
  if (response_batch->has_error) {
    if (response_batch->is_error_set) {
      char* error_message;
      RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
          &responses, request_count,
          LoadStringFromSharedMemory(
              shm_pool_, response_batch->error, error_message));
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    } else {
      const char* error_message =
          "Failed to fetch the error in response batch.";
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    }

    return nullptr;
  }

  Response* responses_shm;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      &responses, request_count,
      shm_pool_->MapOffset((char**)&responses_shm, response_batch->responses));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    std::unique_ptr<InferResponse> infer_response;
    try {
      infer_response = InferResponse::LoadFromSharedMemory(
          shm_pool_, response_batch->responses + sizeof(Response) * r);
      if (infer_response->HasError()) {
        if (infer_response->IsErrorMessageSet()) {
          TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              infer_response->Error()->Message().c_str());

          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed sending response");
          TRITONSERVER_ErrorDelete(err);
        } else {
          const char* err_string = "Failed to process response.";
          TRITONSERVER_Error* err =
              TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_string);

          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed sending response");
          TRITONSERVER_ErrorDelete(err);
        }

        responses[r] = nullptr;

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
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");
      TRITONSERVER_ErrorDelete(err);
      responses[r] = nullptr;
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

#ifdef TRITON_ENABLE_GPU_TENSORS
      if (actual_memory_type == TRITONSERVER_MEMORY_GPU &&
          output_tensor->IsReused()) {
        std::array<char, sizeof(cudaIpcMemHandle_t)> cuda_handle;
        char* cuda_handle_ptr =
            reinterpret_cast<char*>(output_tensor->CudaIpcMemHandle());
        std::copy(
            cuda_handle_ptr, cuda_handle_ptr + cuda_handle.size(),
            cuda_handle.begin());
        std::unordered_map<
            std::array<char, sizeof(cudaIpcMemHandle_t)>, void*>::const_iterator
            reused_gpu_tensor = gpu_tensors_map_.find(cuda_handle);


        // If the tensor is reused, it must be in the GPU tensors map.
        if (reused_gpu_tensor == gpu_tensors_map_.end()) {
          GUARDED_RESPOND_IF_EXCEPTION(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  "Tensor is reused but cannot be found."));
        } else {
          output_tensor->SetDataPtr(reused_gpu_tensor->second);
        }
      }
#endif
      TRITONBACKEND_Output* response_output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &response_output, output_tensor->Name().c_str(),
              static_cast<TRITONSERVER_DataType>(output_tensor->TritonDtype()),
              output_tensor->Dims().data(), output_tensor->Dims().size()));

      void* buffer;
      bool cuda_used;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              response_output, &buffer, output_tensor->ByteSize(),
              &actual_memory_type, &actual_memory_type_id));

      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          CopyBuffer(
              "Failed to copy the output tensor to buffer.", src_memory_type,
              src_memory_type_id, actual_memory_type, actual_memory_type_id,
              output_tensor->ByteSize(), output_tensor->GetDataPtr(), buffer,
              CudaStream(), &cuda_used));
      cuda_copy |= cuda_used;
    }
#ifdef TRITON_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
    }
#endif  // TRITON_ENABLE_GPU

    // If error happens at this stage, we can only log it
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       Name() + " released " + std::to_string(request_count) + " requests")
          .c_str());

  if (response_batch->cleanup) {
    ipc_message_->command = PYTHONSTUB_CommandType::PYTHONSTUB_TensorCleanup;
    // If parent fails to notify the stub or the stub fails to notify the
    // parent in a timely manner, kill the stub process and restart the
    // stub process.

    bool restart = true;
    bool success = NotifyStubAndWait(restart);
    if (!success) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          "Python backend stub exited unexpectedly while performing tensor "
          "cleanup.");
    }
  }

  return nullptr;
}

void
ModelInstanceState::CleanupBLSResponses()
{
  for (auto& bls_inference_response : bls_inference_responses_) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(bls_inference_response),
        " failed to release BLS inference response.");
  }

  bls_inference_responses_.clear();
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
  new (stub_mutex_) bi::interprocess_mutex;
  new (health_mutex_) bi::interprocess_mutex;
  new (parent_mutex_) bi::interprocess_mutex;
  new (stub_cond_) bi::interprocess_condition;
  new (parent_cond_) bi::interprocess_condition;
  parent_lock_ =
      std::make_unique<bi::scoped_lock<bi::interprocess_mutex>>(*parent_mutex_);

  std::string kind = TRITONSERVER_InstanceGroupKindString(kind_);
  std::string shm_region_name =
      std::string("/") + Name() + "_" + kind + "_" + std::to_string(device_id_);

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

    // This shared memory variable indicates whether the
    // stub process should revert the LD_LIBRARY_PATH changes to avoid
    // shared library issues in executables and libraries.
    ipc_control_->uses_env = false;
    if (model_state->PythonExecutionEnv() != "") {
      std::stringstream ss;
      ss << "source " << path_to_activate_
         << " && exec env LD_LIBRARY_PATH=" << path_to_libpython_
         << ":$LD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
         << " " << shm_region_name << " " << shm_default_size << " "
         << shm_growth_size << " " << parent_pid_ << " "
         << model_state->StateForBackend()->python_lib << " "
         << ipc_control_offset_ << " " << Name();
      ipc_control_->uses_env = true;
      // Need to properly set the LD_LIBRARY_PATH so that Python environments
      // using different python versions load properly.
      bash_argument = ss.str();
    } else {
      std::stringstream ss;
      ss << " exec " << python_backend_stub << " " << model_path_ << " "
         << shm_region_name << " " << shm_default_size << " " << shm_growth_size
         << " " << parent_pid_ << " "
         << model_state->StateForBackend()->python_lib << " "
         << ipc_control_offset_ << " " << Name();
      bash_argument = ss.str();
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Starting Python backend stub: ") + bash_argument)
            .c_str());

    stub_args[2] = bash_argument.c_str();
    if (execvp("bash", (char**)stub_args) == -1) {
      std::stringstream ss;
      ss << "Failed to run python backend stub. Errno = " << errno << '\n'
         << "Python backend stub path: " << python_backend_stub << '\n'
         << "Shared Memory Region Name: " << shm_region_name << '\n'
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
    // Pre initialization step.
    if (!WaitForStubNotification()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Timed out occurred while waiting for the stub process. "
                       "Failed to initialize model instance ") +
           Name())
              .c_str());
    }

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

    ipc_message_->command = PYTHONSTUB_CommandType::PYTHONSTUB_Initialize;

    InitializeArgs* initialize_args;
    RETURN_IF_EXCEPTION(shm_pool_->Map(
        (char**)&initialize_args, sizeof(InitializeArgs), ipc_message_->args));
    RETURN_IF_EXCEPTION(SaveMapToSharedMemory(
        shm_pool_, initialize_args->args, initialize_map));

    bool restart = false;
    if (!NotifyStubAndWait(restart)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to initialize stub, stub process exited "
                       "unexpectedly: ") +
           name_)
              .c_str());
    }

    if (initialize_args->response_has_error) {
      if (initialize_args->response_is_error_set) {
        char* err_message;
        RETURN_IF_EXCEPTION(LoadStringFromSharedMemory(
            shm_pool_, initialize_args->response_error, err_message));
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message);
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
  std::string shm_region_name =
      std::string("/") + Name() + "_" + kind + "_" + std::to_string(device_id_);

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int64_t shm_growth_size =
      model_state->StateForBackend()->shm_growth_byte_size;
  int64_t shm_default_size =
      model_state->StateForBackend()->shm_default_byte_size;

  try {
    shm_pool_ = std::make_unique<SharedMemory>(
        shm_region_name, shm_default_size, shm_growth_size,
        true /* truncate */);
  }
  catch (const PythonBackendException& pb_exception) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
  }

  IPCControl* ipc_control;
  shm_pool_->Map((char**)&ipc_control, sizeof(IPCControl), ipc_control_offset_);
  ipc_control_ = ipc_control;

  // Stub mutex and CV
  bi::interprocess_mutex* stub_mutex;
  off_t stub_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&stub_mutex, sizeof(bi::interprocess_mutex), stub_mutex_offset));
  ipc_control_->stub_mutex = stub_mutex_offset;

  bi::interprocess_condition* stub_cv;
  off_t stub_cv_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&stub_cv, sizeof(bi::interprocess_condition), stub_cv_offset));
  ipc_control_->stub_cond = stub_cv_offset;

  stub_cond_ = stub_cv;
  stub_mutex_ = stub_mutex;

  // Parent Mutex and CV
  bi::interprocess_mutex* parent_mutex;
  off_t parent_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&parent_mutex, sizeof(bi::interprocess_mutex),
      parent_mutex_offset));
  ipc_control_->parent_mutex = parent_mutex_offset;

  bi::interprocess_condition* parent_cv;
  off_t parent_cv_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&parent_cv, sizeof(bi::interprocess_condition),
      parent_cv_offset));
  ipc_control_->parent_cond = parent_cv_offset;

  bi::interprocess_mutex* health_mutex;
  off_t health_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&health_mutex, sizeof(bi::interprocess_mutex),
      health_mutex_offset));
  ipc_control_->stub_health_mutex = health_mutex_offset;

  parent_cond_ = parent_cv;
  parent_mutex_ = parent_mutex;
  health_mutex_ = health_mutex;

  off_t ipc_offset;
  RETURN_IF_EXCEPTION(
      shm_pool_->Map((char**)&ipc_message_, sizeof(IPCMessage), ipc_offset));
  ipc_control_->ipc_message = ipc_offset;

  // Offset that must be used for resetting the shared memory usage.
  shm_reset_offset_ = ipc_offset + sizeof(IPCMessage);

  uint64_t model_version = model_state->Version();
  const char* model_path = model_state->RepositoryPath().c_str();

  std::stringstream ss;
  // Use <path>/version/model.py as the model location
  ss << model_path << "/" << model_version << "/model.py";
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
      ipc_message_->command = PYTHONSTUB_CommandType::PYTHONSTUB_Finalize;

      bool restart = false;
      if (!NotifyStubAndWait(restart))
        force_kill = true;
    } else {
      force_kill = true;
    }

    int status;
    if (force_kill) {
      kill(stub_pid_, SIGKILL);
    }
    waitpid(stub_pid_, &status, 0);
  }

  // Destory the lock before deletion of shared memory is triggered.
  parent_lock_.reset(nullptr);
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, Tensor* input_tensor_shm,
    std::shared_ptr<PbTensor>& input_tensor, TRITONBACKEND_Request* request,
    std::vector<TRITONBACKEND_Response*>& responses)
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
      &request, 1, &responses, Model()->TritonMemoryManager(),
      false /* pinned_enable */, CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  bool cpu_only_tensors = model_state->ForceCPUOnlyInputTensors();
  if (input_dtype == TRITONSERVER_TYPE_BYTES) {
    cpu_only_tensors = true;
  }
  TRITONSERVER_MemoryType src_memory_type;
  int64_t src_memory_type_id;
  size_t src_byte_size;
  const void* src_ptr;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
      in, 0 /* input buffer index */, &src_ptr, &src_byte_size,
      &src_memory_type, &src_memory_type_id));

// If TRITON_ENABLE_GPU_TENSORS is false, we need to copy the tensors
// to the CPU.
#ifndef TRITON_ENABLE_GPU_TENSORS
  cpu_only_tensors = true;
#endif  // TRITON_ENABLE_GPU_TENSORS

  if (cpu_only_tensors || src_memory_type != TRITONSERVER_MEMORY_GPU) {
    input_tensor = std::make_unique<PbTensor>(
        std::string(input_name),
        std::vector<int64_t>(input_shape, input_shape + input_dims_count),
        input_dtype, TRITONSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */, nullptr /* buffer ptr*/, input_byte_size,
        nullptr /* DLManagedTensor */);
    RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
        shm_pool_, input_tensor_shm, false /* copy */));
    char* input_buffer = reinterpret_cast<char*>(input_tensor->GetDataPtr());
    collector.ProcessTensor(
        input_name, input_buffer, input_byte_size,
        TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
  } else {
#ifdef TRITON_ENABLE_GPU_TENSORS
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
    RETURN_IF_EXCEPTION(
        input_tensor->SaveToSharedMemory(shm_pool_, input_tensor_shm));
    std::array<char, CUDA_IPC_HANDLE_SIZE> cuda_mem_handle_array;
    char* cuda_handle =
        reinterpret_cast<char*>(input_tensor->CudaIpcMemHandle());
    std::copy(
        cuda_handle, cuda_handle + cuda_mem_handle_array.size(),
        cuda_mem_handle_array.begin());
    gpu_tensors_map_.insert(
        {cuda_mem_handle_array,
         reinterpret_cast<void*>(input_tensor->GetGPUStartAddress())});
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Python backend does not support GPU tensors.");
#endif  // TRITON_ENABLE_GPU_TENSORS
  }

  return nullptr;
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
  TRITONSERVER_Error* err =
      instance_state->ProcessRequests(requests, request_count);
  instance_state->CleanupBLSResponses();

  // We should return the shared memory offset before returning from this
  // function. Otherwise there will be shared memory leaks if there is an
  // error when processing the requests
  instance_state->ResetSharedMemoryOffset();
  if (err != nullptr) {
    return err;
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
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
