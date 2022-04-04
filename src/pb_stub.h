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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include "infer_request.h"
#include "infer_response.h"
#include "ipc_message.h"
#include "message_queue.h"
#include "pb_utils.h"

#pragma once

namespace bi = boost::interprocess;
namespace py = pybind11;
using namespace pybind11::literals;

namespace triton { namespace backend { namespace python {

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

  /// Initialize the user's Python code.
  void Initialize(bi::managed_external_buffer::handle_t map_handle);

  /// Send a message to the parent process.
  void SendIPCMessage(std::unique_ptr<IPCMessage>& ipc_message);

  /// Receive a message from the parent process.
  std::unique_ptr<IPCMessage> PopMessage();

  /// Update the health variable in the stub process.
  void UpdateHealth();

  /// Finalize and terminate the stub process
  void Finalize();

  /// Execute a batch of requests.
  py::list Execute(
      RequestBatch* request_batch_shm_ptr,
      ResponseBatch* response_batch_shm_ptr,
      bi::managed_external_buffer::handle_t* responses_shm_handle);

  /// Get the memory manager message queue
  std::unique_ptr<MessageQueue<uint64_t>>& MemoryManagerQueue();

  void ProcessResponse(InferResponse* response);
  void LoadGPUBuffers(std::unique_ptr<IPCMessage>& ipc_message);
  ~Stub();

 private:
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  std::string model_path_;
  std::string model_version_;
  std::string model_instance_name_;
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
  std::unique_ptr<MessageQueue<uint64_t>> memory_manager_message_queue_;
  std::mutex tensors_to_remove_mutex_;
  std::vector<std::unique_ptr<IPCMessage>> messages_;
  std::mutex messages_mutex_;
  std::condition_variable messages_cv_;
  bool initialized_;
  static std::unique_ptr<Stub> stub_instance_;
  std::vector<std::shared_ptr<PbTensor>> gpu_tensors_;
};
}}}  // namespace triton::backend::python
