// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "memory_manager.h"


namespace triton { namespace backend { namespace python {


#ifdef TRITON_ENABLE_GPU
GPUMemoryRecord::GPUMemoryRecord(void* ptr)
{
  ptr_ = ptr;
  release_callback_ = [](void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to free the allocated cuda memory. error: ") +
           cudaGetErrorString(err))
              .c_str());
    }
  };
}

void*
GPUMemoryRecord::MemoryId()
{
  return ptr_;
}

const std::function<void(void*)>&
GPUMemoryRecord::ReleaseCallback()
{
  return release_callback_;
}
#endif

MemoryManager::MemoryManager(
    std::unique_ptr<MessageQueue<uint64_t>>&& memory_message_queue)
{
  message_queue_ = std::move(memory_message_queue);
  record_count_ = 0;
  thread_ = std::thread(&MemoryManager::QueueMonitorThread, this);
}

uint64_t
MemoryManager::AddRecord(std::unique_ptr<MemoryRecord>&& memory_record)
{
  std::lock_guard<std::mutex> lock{mu_};

  record_count_++;
  records_.emplace(record_count_, std::move(memory_record));

  return record_count_;
}

void
MemoryManager::QueueMonitorThread()
{
  while (true) {
    uint64_t memory = message_queue_->Pop();
    if (memory == 0) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock{mu_};
      auto it = records_.find(memory);
      if (it == records_.end()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            "Unexpected memory index received for deallocation.");
        continue;
      }

      // Call the release callback.
      it->second->ReleaseCallback()(it->second->MemoryId());
      records_.erase(it);

      // Reset the record_count_ when the number of records is zero.
      if (records_.size() == 0)
        record_count_ = 0;
    }
  }
}

MemoryManager::~MemoryManager()
{
  // Push a dummy 0 message that will trigger the destruction of the background
  // thread.
  message_queue_->Push(0);
  thread_.join();
}

}}};  // namespace triton::backend::python
