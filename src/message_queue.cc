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

#include "message_queue.h"

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <iostream>
#include "ipc_message.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {
MessageQueue::MessageQueue(
    std::unique_ptr<SharedMemory>& shm_pool, std::size_t number_of_messages)
{
  MessageQueueShm* message_queue_shm;
  shm_pool->Map(
      (char**)&message_queue_shm, sizeof(MessageQueueShm), shm_struct_);

  message_queue_shm->size = number_of_messages;
  size_ = &(message_queue_shm->size);

  message_queue_shm->index = 0;
  index_ = &(message_queue_shm->index);

  shm_pool->Map(
      (char**)&sem_full_, sizeof(bi::interprocess_semaphore),
      message_queue_shm->sem_full);
  shm_pool->Map(
      (char**)&sem_empty_, sizeof(bi::interprocess_semaphore),
      message_queue_shm->sem_empty);

  new (sem_full_) bi::interprocess_semaphore(0);
  new (sem_empty_) bi::interprocess_semaphore(number_of_messages);

  shm_pool->Map(
      (char**)&mutex_, sizeof(bi::interprocess_mutex),
      message_queue_shm->mutex);
  new (mutex_) bi::interprocess_mutex;

  shm_pool->Map(
      (char**)&buffer_, sizeof(off_t) * number_of_messages,
      message_queue_shm->buffer);
}

void
MessageQueue::Push(off_t message, int const& duration, bool& success)
{
  boost::system_time timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(duration);

  while (true) {
    try {
      if (!sem_empty_->timed_wait(timeout)) {
        success = false;
        return;
      } else {
        break;
      }
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    timeout =
        boost::get_system_time() + boost::posix_time::milliseconds(duration);
    bi::scoped_lock<bi::interprocess_mutex> lock{*mutex_, timeout};
    if (!lock) {
      sem_empty_->post();
      success = false;
      return;
    }
    success = true;

    buffer_[*index_] = message;
    (*index_)++;
  }
  sem_full_->post();
}

void
MessageQueue::Push(off_t message)
{
  while (true) {
    try {
      sem_empty_->wait();
      break;
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{*mutex_};
    buffer_[*index_] = message;
    (*index_)++;
  }
  sem_full_->post();
}

off_t
MessageQueue::ShmOffset()
{
  return shm_struct_;
}

off_t
MessageQueue::Pop()
{
  off_t message;

  while (true) {
    try {
      sem_full_->wait();
      break;
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{*mutex_};
    message = buffer_[*index_ - 1];
    (*index_)--;
  }
  sem_empty_->post();

  return message;
}

off_t
MessageQueue::Pop(int const& duration, bool& success)
{
  off_t message = 0;
  boost::system_time timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(duration);

  while (true) {
    try {
      if (!sem_full_->timed_wait(timeout)) {
        success = false;
        return message;
      } else {
        break;
      }
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    timeout =
        boost::get_system_time() + boost::posix_time::milliseconds(duration);
    bi::scoped_lock<bi::interprocess_mutex> lock{*mutex_, timeout};
    if (!lock) {
      sem_full_->post();
      success = false;
      return message;
    }
    success = true;

    message = buffer_[*index_ - 1];
    (*index_)--;
  }
  sem_empty_->post();

  return message;
}

void
MessageQueue::ResetSemaphores()
{
  new (sem_full_) bi::interprocess_semaphore(0);
  new (sem_empty_) bi::interprocess_semaphore(*size_);
  new (mutex_) bi::interprocess_mutex;
}

std::unique_ptr<MessageQueue>
MessageQueue::LoadFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t message_queue_offset)
{
  std::unique_ptr<MessageQueue> message_queue =
      std::make_unique<MessageQueue>();
  MessageQueueShm* message_queue_shm;
  shm_pool->MapOffset((char**)&message_queue_shm, message_queue_offset);
  message_queue->size_ = &(message_queue_shm->size);
  message_queue->index_ = &(message_queue_shm->index);

  shm_pool->MapOffset((char**)&message_queue->mutex_, message_queue_shm->mutex);
  shm_pool->MapOffset(
      (char**)&message_queue->sem_full_, message_queue_shm->sem_full);
  shm_pool->MapOffset(
      (char**)&message_queue->sem_empty_, message_queue_shm->sem_empty);
  shm_pool->MapOffset(
      (char**)&message_queue->buffer_, message_queue_shm->buffer);
  message_queue->shm_struct_ = message_queue_offset;
  return message_queue;
}

}}}  // namespace triton::backend::python
