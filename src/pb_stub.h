// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "infer_request.h"
#include "infer_response.h"
#include "ipc_message.h"
#include "message_queue.h"
#include "pb_log.h"
#include "pb_utils.h"


namespace bi = boost::interprocess;
namespace py = pybind11;
using namespace pybind11::literals;

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
  Logger(){};
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

  /// Singleton Getter function
  static Logger& GetOrCreateInstance()
  {
    static Logger instance;
    return instance;
  }

  DISALLOW_COPY_AND_ASSIGN(Logger);

  /// Flush the log.
  void Flush() { std::cerr << std::flush; }
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
    Logger::GetOrCreateInstance().Log(file_, line_, level_, stream_.str());
  }

  std::stringstream& stream() { return stream_; }

 private:
  std::stringstream stream_;
  std::string file_;
  uint32_t line_;
  LogLevel level_;
};

#define LOG_FL(FN, LN, LVL) LogMessage((char*)(FN), LN, LVL).stream()

class Stub {
 public:
  Stub(){};
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

  /// Send a log message to the parent process.
  void SendIPCLogMessage(std::unique_ptr<IPCMessage>& ipc_message);

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

  void ProcessResponse(InferResponse* response);
  void LoadGPUBuffers(std::unique_ptr<IPCMessage>& ipc_message);
  bool IsDecoupled();
  ~Stub();

  /// Start client log handler process
  void LaunchLogRequestThread();

  /// End client log handler process
  void TerminateLogRequestThread();

  /// Add client log to queue
  void EnqueueLogRequest(std::unique_ptr<PbLog>& log_ptr);

  /// Thread process
  void ServiceLogRequests();

  /// Send client log to the python backend
  void SendLogMessage(std::unique_ptr<PbLog>& log_send_message);

  /// Check if log handler is running
  bool LogServiceActive();

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
      log_message_queue_;
  std::unique_ptr<MessageQueue<uint64_t>> memory_manager_message_queue_;
  bool initialized_;
  static std::unique_ptr<Stub> stub_instance_;
  std::vector<std::shared_ptr<PbTensor>> gpu_tensors_;
  std::queue<std::unique_ptr<PbLog>> log_request_buffer_;
  std::thread log_monitor_;
  bool log_thread_;
  std::mutex log_message_mutex_;
  std::condition_variable log_message_cv_;
};
}}}  // namespace triton::backend::python
