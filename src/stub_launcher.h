// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/functional/hash.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <future>
#include <sstream>
#include <string>
#include <vector>
#include "ipc_message.h"
#include "memory_manager.h"
#include "message_queue.h"
#include "pb_utils.h"
#include "python.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

class StubLauncher {
 public:
  StubLauncher(
      ModelState* model_state, const std::string stub_process_kind,
      const std::string name);
  StubLauncher(
      ModelState* model_state, const std::string stub_process_kind,
      const std::string name, const int32_t device_id, const std::string kind);

  // Initialize stub process
  TRITONSERVER_Error* Initialize();

  // Stub process setup
  TRITONSERVER_Error* Setup(
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          stub_message_queue,
      bi::interprocess_mutex** health_mutex,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          parent_message_queue,
      std::unique_ptr<MemoryManager>* memory_manager,
      std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>*
          ipc_control,
      bi::managed_external_buffer::handle_t* ipc_control_handle,
      std::unique_ptr<SharedMemoryManager>* shm_pool,
      std::unique_ptr<boost::asio::thread_pool>* thread_pool);

  // Launch stub process
  TRITONSERVER_Error* Launch(
      pid_t* stub_pid,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          stub_message_queue,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          parent_message_queue,
      std::unique_ptr<MemoryManager>* memory_manager,
      std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>*
          ipc_control,
      bi::managed_external_buffer::handle_t* ipc_control_handle,
      std::unique_ptr<SharedMemoryManager>* shm_pool);

  // Model stub process for auto-complete
  void ModelStubProcess(
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          stub_message_queue,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          parent_message_queue,
      std::unique_ptr<SharedMemoryManager>* shm_pool);

  // Model Instance stub process
  TRITONSERVER_Error* InstanceStubProcess(
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          stub_message_queue,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          parent_message_queue,
      std::unique_ptr<SharedMemoryManager>* shm_pool);

  // Destruct Stub process
  void Destruct(
      pid_t* stub_pid_,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          stub_message_queue,
      bi::interprocess_mutex** health_mutex,
      std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>*
          parent_message_queue,
      std::unique_ptr<MemoryManager>* memory_manager,
      std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>*
          ipc_control,
      std::unique_ptr<SharedMemoryManager>* shm_pool,
      std::thread* decoupled_monitor,
      std::unique_ptr<boost::asio::thread_pool>* thread_pool,
      std::vector<std::future<void>>* futures);

  // Fix string to json format
  void FixStringToJsonFormat(std::string* str);

 private:
  ModelState* model_state_;
  const std::string stub_process_kind_;
  const std::string name_;
  const int32_t device_id_;
  const std::string kind_;

  // Parent process pid
  pid_t parent_pid_;

  bool initialized_;
  std::string shm_region_name_;
  std::string model_path_;

  // Path to python execution environment
  std::string path_to_libpython_;
  std::string path_to_activate_;
};
}}}  // namespace triton::backend::python
