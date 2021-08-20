// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ipc_message.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
namespace bi = boost::interprocess;

/// Struct holding the represenation of a message queue inside the shared
/// memory.
/// \param size Total size of the message queue.
/// \param mutex Offset of the mutex variable protecting index.
/// \param index Used element index.
/// \param sem_empty Semaphore object counting the number of empty buffer slots.
/// \param sem_full Semaphore object counting the number of used buffer slots.
struct MessageQueueShm {
  std::size_t size;
  off_t buffer;
  off_t mutex;
  int index;
  off_t sem_empty;
  off_t sem_full;
};

class MessageQueue {
  std::size_t* size_;
  off_t* buffer_;
  bi::interprocess_mutex* mutex_;
  int* index_;
  bi::interprocess_semaphore* sem_empty_;
  bi::interprocess_semaphore* sem_full_;
  off_t shm_struct_;

 public:
  /// Create a Message queue.
  /// \param shm_pool Shared memory pool
  /// \param number_of_messages Maximum number of messages that the
  /// message queue can hold.
  MessageQueue(
      std::unique_ptr<SharedMemory>& shm_pool, std::size_t number_of_messages);
  MessageQueue() {}

  /// Push a message inside the message queue.
  /// \param message The shared memory offset of the message.
  void Push(off_t message);
  void Push(off_t message, int const& duration, bool& success);

  /// Pop a message from the message queue. This call will block until there
  /// is a message inside the message queue to return. \return the offset of
  /// the new message.
  off_t Pop();
  off_t Pop(int const& duration, bool& success);

  off_t ShmOffset();
  static std::unique_ptr<MessageQueue> LoadFromSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, off_t message_queue_offset);

  /// Resets the semaphores for the message queue. This function is useful for
  /// when the stub process may have exited unexpectedly and the semaphores need
  /// to be restarted so that the message queue is in a proper state.
  void ResetSemaphores();
};
}}}  // namespace triton::backend::python
