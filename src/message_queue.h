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

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <cstddef>
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace bi = boost::interprocess;

/// Struct holding the represenation of a message queue inside the shared
/// memory.
/// \param size Total size of the message queue.
/// \param mutex Handle of the mutex variable protecting index.
/// \param index Used element index.
/// \param sem_empty Semaphore object counting the number of empty buffer slots.
/// \param sem_full Semaphore object counting the number of used buffer slots.
struct MessageQueueShm {
  bi::interprocess_semaphore sem_empty{0};
  bi::interprocess_semaphore sem_full{0};
  bi::interprocess_mutex mutex;
  std::size_t size;
  bi::managed_external_buffer::handle_t buffer;
  int head;
  int tail;
};

class MessageQueue {
 public:
  /// Create a new MessageQueue in the shared memory.
  static std::unique_ptr<MessageQueue> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      uint32_t message_queue_size);

  /// Load an already existing message queue from the shared memory.
  static std::unique_ptr<MessageQueue> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t message_queue_handle);

  /// Push a message inside the message queue.
  /// \param message The shared memory handle of the message.
  void Push(bi::managed_external_buffer::handle_t message);
  void Push(
      bi::managed_external_buffer::handle_t message, int const& duration,
      bool& success);

  /// Pop a message from the message queue. This call will block until there
  /// is a message inside the message queue to return.
  /// \return the handle of the new message.
  bi::managed_external_buffer::handle_t Pop();
  bi::managed_external_buffer::handle_t Pop(int const& duration, bool& success);

  /// Resets the semaphores for the message queue. This function is useful for
  /// when the stub process may have exited unexpectedly and the semaphores need
  /// to be restarted so that the message queue is in a proper state.
  void ResetSemaphores();

  /// Get the shared memory handle of MessageQueue
  bi::managed_external_buffer::handle_t ShmHandle();

  /// Release the ownership of this object in shared memory.
  void Release();

 private:
  std::size_t& Size() { return mq_shm_ptr_->size; }
  const bi::interprocess_mutex& Mutex() { return mq_shm_ptr_->mutex; }
  bi::interprocess_mutex* MutexMutable() { return &(mq_shm_ptr_->mutex); }
  int& Head() { return mq_shm_ptr_->head; }
  int& Tail() { return mq_shm_ptr_->tail; }
  bi::managed_external_buffer::handle_t* Buffer() { return mq_buffer_shm_ptr_; }
  const bi::interprocess_semaphore& SemEmpty()
  {
    return mq_shm_ptr_->sem_empty;
  }
  bi::interprocess_semaphore* SemEmptyMutable()
  {
    return &(mq_shm_ptr_->sem_empty);
  }
  const bi::interprocess_semaphore& SemFull() { return mq_shm_ptr_->sem_full; }
  bi::interprocess_semaphore* SemFullMutable()
  {
    return &(mq_shm_ptr_->sem_full);
  }

  void HeadIncrement();
  void TailIncrement();

  AllocatedSharedMemory<MessageQueueShm> mq_shm_;
  AllocatedSharedMemory<bi::managed_external_buffer::handle_t> mq_buffer_shm_;

  MessageQueueShm* mq_shm_ptr_;
  bi::managed_external_buffer::handle_t* mq_buffer_shm_ptr_;
  bi::managed_external_buffer::handle_t mq_handle_;

  /// Create/load a Message queue.
  /// \param mq_shm Message queue representation in shared memory.
  MessageQueue(
      AllocatedSharedMemory<MessageQueueShm>& mq_shm,
      AllocatedSharedMemory<bi::managed_external_buffer::handle_t>&
          mq_buffer_shm);
};
}}}  // namespace triton::backend::python
