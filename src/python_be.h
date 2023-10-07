// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
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
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "infer_request.h"
#include "infer_response.h"
#include "ipc_message.h"
#include "memory_manager.h"
#include "message_queue.h"
#include "metric.h"
#include "metric_family.h"
#include "pb_env.h"
#include "pb_map.h"
#include "pb_metric_reporter.h"
#include "pb_utils.h"
#include "request_executor.h"
#include "scoped_defer.h"
#include "shm_manager.h"
#include "stub_launcher.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/common/nvtx.h"
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
  int64_t thread_pool_size;
  std::unique_ptr<EnvironmentManager> env_manager;
  std::string runtime_modeldir;
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

  // Is decoupled API being used.
  bool IsDecoupled() { return decoupled_; }

  // Returns the value in the `runtime_modeldir_` field
  std::string RuntimeModelDir() { return runtime_modeldir_; }

  // Launch auto-complete stub process.
  TRITONSERVER_Error* LaunchAutoCompleteStubProcess();

  // Validate Model Configuration
  TRITONSERVER_Error* ValidateModelConfig();

  // Auto-complete stub
  std::unique_ptr<StubLauncher>& Stub() { return auto_complete_stub_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  BackendState* backend_state_;
  std::string python_execution_env_;
  bool force_cpu_only_input_tensors_;
  bool decoupled_;
  std::string runtime_modeldir_;
  std::unique_ptr<StubLauncher> auto_complete_stub_;
};

class ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance);

  TRITONBACKEND_Model* triton_model_;
  std::unique_ptr<StubLauncher> model_instance_stub_;
  std::vector<intptr_t> closed_requests_;
  std::mutex closed_requests_mutex_;

  std::thread stub_to_parent_queue_monitor_;
  bool stub_to_parent_thread_;
  // Decoupled monitor thread
  std::thread decoupled_monitor_;
  bool decoupled_thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<IPCMessage> received_message_;
  std::vector<std::future<void>> futures_;
  std::unique_ptr<boost::asio::thread_pool> thread_pool_;
  std::unordered_map<void*, std::shared_ptr<InferPayload>> infer_payload_;
  std::unique_ptr<RequestExecutor> request_executor_;

 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  ~ModelInstanceState();

  // Launch stub process.
  TRITONSERVER_Error* LaunchStubProcess();

  TRITONSERVER_Error* SendMessageToStub(off_t message);
  void ResponseSendDecoupled(std::shared_ptr<IPCMessage> response_send_message);

  // Checks whether the stub process is live
  bool IsStubProcessAlive();

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

  // In the decoupled mode, the parent message queue is monitored only by this
  // function during the execute phase. No other thread should pop any message
  // from the message queue in the decoupled mode.
  void DecoupledMessageQueueMonitor();

  // This function is executed on a separate thread and monitors the queue for
  // message sent from stub to parent process.
  void StubToParentMQMonitor();

  // Process the log request.
  void ProcessLogRequest(const std::unique_ptr<IPCMessage>& message);

  // Convert TRITONBACKEND_Input to Python backend tensors.
  TRITONSERVER_Error* GetInputTensor(
      const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
      TRITONBACKEND_Request* request,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses);

  // Process all the requests obtained from Triton.
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      bool& restart);

  // Process all the requests in the decoupled mode.
  TRITONSERVER_Error* ProcessRequestsDecoupled(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<std::unique_ptr<InferRequest>>& pb_infer_requests,
      PbMetricReporter& pb_metric_reporter);

  bool ExistsInClosedRequests(intptr_t closed_request);

  // Execute a BLS Request
  void ExecuteBLSRequest(
      std::shared_ptr<IPCMessage> ipc_message, const bool is_stream);

  // Cleanup BLS responses
  void CleanupBLSResponses();

  // Wait for BLS requests to complete
  void WaitForBLSRequestsToFinish();

  // Check the incoming requests for errors
  TRITONSERVER_Error* CheckIncomingRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      size_t& total_batch_size);

  // Set error for response send message
  void SetErrorForResponseSendMessage(
      ResponseSendMessage* response_send_message,
      std::shared_ptr<TRITONSERVER_Error*> error,
      std::unique_ptr<PbString>& error_message);

  TRITONSERVER_Error* SaveRequestsToSharedMemory(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<std::unique_ptr<InferRequest>>& pb_inference_requests,
      AllocatedSharedMemory<char>& request_batch,
      std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses);

  // Model instance stub
  std::unique_ptr<StubLauncher>& Stub() { return model_instance_stub_; }

  // Stop the stub_to_parent_queue_monitor thread
  void TerminateMonitor();

  // Start the stub_to_parent_queue_monitor thread
  void StartMonitor();

  // Send bls decoupled response to the stub process
  void SendBLSDecoupledResponse(std::unique_ptr<InferResponse> infer_response);

  // Prepare the response batch object
  void PrepareResponseBatch(
      ResponseBatch** response_batch,
      AllocatedSharedMemory<char>& response_batch_shm,
      std::unique_ptr<IPCMessage>* ipc_message,
      bi::managed_external_buffer::handle_t** response_handle);

  // Prepare the response handle
  void PrepareResponseHandle(
      std::unique_ptr<InferResponse>* infer_response,
      bi::managed_external_buffer::handle_t* response_handle);

  // Process the bls decoupled cleanup request
  void ProcessBLSCleanupRequest(const std::unique_ptr<IPCMessage>& message);

  // Process request cancellation query
  void ProcessIsRequestCancelled(const std::unique_ptr<IPCMessage>& message);

  // Process a message. The function 'request_handler' is invoked
  // to handle the request. T should be either 'MetricFamily', 'Metric' or
  // 'ModelLoader', and MessageType should be either 'MetricFamilyMessage',
  // 'MetricMessage' or 'ModelLoaderMessage'.
  template <typename T, typename MessageType>
  void ProcessMessage(
      const std::unique_ptr<IPCMessage>& message,
      std::function<void(std::unique_ptr<T>&, MessageType*)> request_handler);

  // Process a metric family request
  void ProcessMetricFamilyRequest(const std::unique_ptr<IPCMessage>& message);

  // Process a metric request
  void ProcessMetricRequest(const std::unique_ptr<IPCMessage>& message);

  // Process a model control request
  void ProcessModelControlRequest(const std::unique_ptr<IPCMessage>& message);
};
}}}  // namespace triton::backend::python
