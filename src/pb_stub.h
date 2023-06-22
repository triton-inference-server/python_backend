// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "infer_request.h"
#include "infer_response.h"
#include "ipc_message.h"
#include "message_queue.h"
#include "metric.h"
#include "metric_family.h"
#include "pb_log.h"
#include "pb_response_iterator.h"
#include "pb_utils.h"


namespace bi = boost::interprocess;
namespace py = pybind11;
using namespace pybind11::literals;

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif

namespace triton { namespace backend { namespace python {

#define LOG_IF_EXCEPTION(X)                              \
  do {                                                   \
    try {                                                \
      (X);                                               \
    }                                                    \
    catch (const PythonBackendException& pb_exception) { \
      LOG_INFO << pb_exception.what();                   \
    }                                                    \
  } while (false)

#define LOG_EXCEPTION(E)  \
  do {                    \
    LOG_INFO << E.what(); \
  } while (false)

/// Macros that use current filename and line number.
#define LOG_INFO LOG_FL(__FILE__, __LINE__, LogLevel::INFO)
#define LOG_WARN LOG_FL(__FILE__, __LINE__, LogLevel::WARNING)
#define LOG_ERROR LOG_FL(__FILE__, __LINE__, LogLevel::ERROR)
#define LOG_VERBOSE LOG_FL(__FILE__, __LINE__, LogLevel::VERBOSE)

class Logger {
 public:
  Logger() { backend_logging_active_ = false; };
  ~Logger() { log_instance_.reset(); };
  /// Python client log function
  static void Log(const std::string& message, LogLevel level = LogLevel::INFO);

  /// Python client log info function
  static void LogInfo(const std::string& message);

  /// Python client warning function
  static void LogWarn(const std::string& message);

  /// Python client log error function
  static void LogError(const std::string& message);

  /// Python client log verbose function
  static void LogVerbose(const std::string& message);

  /// Internal log function
  void Log(
      const std::string& filename, uint32_t lineno, LogLevel level,
      const std::string& message);

  /// Log format helper function
  const std::string LeadingLogChar(const LogLevel& level);

  /// Set PYBE Logging Status
  void SetBackendLoggingActive(bool status);

  /// Get PYBE Logging Status
  bool BackendLoggingActive();

  /// Singleton Getter Function
  static std::unique_ptr<Logger>& GetOrCreateInstance();

  DISALLOW_COPY_AND_ASSIGN(Logger);

  /// Flush the log.
  void Flush() { std::cerr << std::flush; }

 private:
  static std::unique_ptr<Logger> log_instance_;
  bool backend_logging_active_;
};

class LogMessage {
 public:
  /// Create a log message, stripping the path down to the filename only
  LogMessage(const char* file, int line, LogLevel level) : level_(level)
  {
    std::string path(file);
    size_t pos = path.rfind('/');
    if (pos != std::string::npos) {
      path = path.substr(pos + 1, std::string::npos);
    }
    file_ = path;
    line_ = static_cast<uint32_t>(line);
  }
  /// Log message to console or send to backend (see Logger::Log for details)
  ~LogMessage()
  {
    Logger::GetOrCreateInstance()->Log(file_, line_, level_, stream_.str());
  }

  std::stringstream& stream() { return stream_; }

 private:
  std::stringstream stream_;
  std::string file_;
  uint32_t line_;
  LogLevel level_;
};

#define LOG_FL(FN, LN, LVL) LogMessage((char*)(FN), LN, LVL).stream()

// The payload for the stub_to_parent message queue. This struct serves as a
// wrapper for different types of messages so that they can be sent through the
// same buffer.
struct UtilsMessagePayload {
  UtilsMessagePayload(
      const PYTHONSTUB_CommandType& command_type, void* utils_message_ptr)
      : command_type(command_type), utils_message_ptr(utils_message_ptr)
  {
  }
  PYTHONSTUB_CommandType command_type;
  void* utils_message_ptr;
};

class Stub {
 public:
  Stub() : stub_to_parent_thread_(false), parent_to_stub_thread_(false){};
  static std::unique_ptr<Stub>& GetOrCreateInstance();

  /// Instantiate a new Python backend Stub.
  void Instantiate(
      int64_t shm_growth_size, int64_t shm_default_size,
      const std::string& shm_region_name, const std::string& model_path,
      const std::string& model_version, const std::string& triton_install_path,
      bi::managed_external_buffer::handle_t ipc_control_handle,
      const std::string& model_instance_name);

  /// Get the health of the stub process.
  bool& Health();

  /// Get the shared memory manager.
  std::unique_ptr<SharedMemoryManager>& SharedMemory();

  /// Run a single command from the shared memory.
  bool RunCommand();

  /// Setup for the stub process
  py::module StubSetup();

  /// Set the model configuration for auto-complete
  void AutoCompleteModelConfig(
      bi::managed_external_buffer::handle_t string_handle,
      std::string* auto_complete_config);

  /// Initialize the user's Python code.
  void Initialize(bi::managed_external_buffer::handle_t map_handle);

  /// Send a message to the parent process.
  void SendIPCMessage(std::unique_ptr<IPCMessage>& ipc_message);

  /// Send a utils message to the parent process.
  void SendIPCUtilsMessage(std::unique_ptr<IPCMessage>& ipc_message);

  /// Receive a message from the parent process.
  std::unique_ptr<IPCMessage> PopMessage();

  /// Update the health variable in the stub process.
  void UpdateHealth();

  /// Finalize and terminate the stub process
  void Finalize();

  /// Load all the requests from shared memory
  py::list LoadRequestsFromSharedMemory(RequestBatch* request_batch_shm_ptr);

  /// Execute a batch of requests.
  void ProcessRequests(RequestBatch* request_batch_shm_ptr);

  void ProcessRequestsDecoupled(RequestBatch* request_batch_shm_ptr);

  /// Get the memory manager message queue
  std::unique_ptr<MessageQueue<uint64_t>>& MemoryManagerQueue();

  /// Get the shared memory pool
  std::unique_ptr<SharedMemoryManager>& ShmPool() { return shm_pool_; }

  void ProcessResponse(InferResponse* response);
  void LoadGPUBuffers(std::unique_ptr<IPCMessage>& ipc_message);
  bool IsDecoupled();
  ~Stub();

  /// Start stub to parent message handler process
  void LaunchStubToParentQueueMonitor();

  /// End stub to parent message handler process
  void TerminateStubToParentQueueMonitor();

  /// Add client log to queue
  void EnqueueLogRequest(std::unique_ptr<PbLog>& log_ptr);

  /// Thread process
  void ServiceStubToParentRequests();

  /// Send client log to the python backend
  void SendLogMessage(std::unique_ptr<UtilsMessagePayload>& utils_msg_payload);

  /// Check if stub to parent message handler is running
  bool StubToParentServiceActive();

  /// Start parent to stub message handler process
  void LaunchParentToStubQueueMonitor();

  /// End parent to stub message handler process
  void TerminateParentToStubQueueMonitor();

  /// Check if parent to stub message handler is running
  bool ParentToStubServiceActive();

  /// Thread process
  void ParentToStubMQMonitor();

  /// Get the ResponseIterator object associated with the infer response
  std::shared_ptr<ResponseIterator> GetResponseIterator(
      std::shared_ptr<InferResponse> infer_response);

  /// Send the id to the python backend for object cleanup
  void SendCleanupId(std::unique_ptr<UtilsMessagePayload>& utils_msg_payload);

  /// Add cleanup id to queue
  void EnqueueCleanupId(void* id);

  /// Is the stub initialized
  bool IsInitialized();

  /// Is the stub in the finalize stage
  bool IsFinalizing();

  /// Helper function to enqueue a utils message to the stub to parent message
  /// buffer
  void EnqueueUtilsMessage(
      std::unique_ptr<UtilsMessagePayload> utils_msg_payload);

  /// Send the custom metrics message to the python backend
  void SendCustomMetricsMessage(
      CustomMetricsMessage** custom_metrics_msg,
      PYTHONSTUB_CommandType command_type,
      bi::managed_external_buffer::handle_t handle);

  /// Helper function to prepare the custom metrics message
  void PrepareCustomMetricsMessage(
      AllocatedSharedMemory<CustomMetricsMessage>& custom_metrics_msg_shm,
      CustomMetricsMessage** custom_metrics_msg);

  /// Helper function to retrieve a proxy stream for dlpack synchronization
  /// for provided device
  cudaStream_t GetProxyStream(const int& device_id);

 private:
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  std::string model_path_;
  std::string model_version_;
  std::string name_;
  std::string triton_install_path_;
  IPCControlShm* ipc_control_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
  py::object model_instance_;
  py::object deserialize_bytes_;
  py::object serialize_bytes_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      stub_message_queue_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      parent_message_queue_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      stub_to_parent_mq_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      parent_to_stub_mq_;
  std::unique_ptr<MessageQueue<uint64_t>> memory_manager_message_queue_;
  bool initialized_;
  bool finalizing_;
  static std::unique_ptr<Stub> stub_instance_;
  std::vector<std::shared_ptr<PbTensor>> gpu_tensors_;
  std::queue<std::unique_ptr<UtilsMessagePayload>> stub_to_parent_buffer_;
  std::thread stub_to_parent_queue_monitor_;
  bool stub_to_parent_thread_;
  std::mutex stub_to_parent_message_mu_;
  std::condition_variable stub_to_parent_message_cv_;
  std::thread parent_to_stub_queue_monitor_;
  bool parent_to_stub_thread_;
  std::mutex response_iterator_map_mu_;
  std::unordered_map<void*, std::shared_ptr<ResponseIterator>>
      response_iterator_map_;
  std::mutex dlpack_proxy_stream_pool_mu_;
  std::unordered_map<int, cudaStream_t> dlpack_proxy_stream_pool_;
};
}}}  // namespace triton::backend::python
