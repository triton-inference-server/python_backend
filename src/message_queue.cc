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

namespace triton { namespace backend { namespace python {

std::unique_ptr<MessageQueue>
MessageQueue::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool, uint32_t message_queue_size)
{
  AllocatedSharedMemory<MessageQueueShm> mq_shm =
      shm_pool->Construct<MessageQueueShm>();
  mq_shm.data_->size = message_queue_size;

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t> mq_buffer_shm =
      shm_pool->Construct<bi::managed_external_buffer::handle_t>(
          message_queue_size);
  mq_shm.data_->buffer = mq_buffer_shm.handle_;
  mq_shm.data_->index = 0;

  new (&(mq_shm.data_->mutex)) bi::interprocess_mutex{};
  new (&(mq_shm.data_->sem_empty))
      bi::interprocess_semaphore{message_queue_size};
  new (&(mq_shm.data_->sem_full)) bi::interprocess_semaphore{0};

  return std::unique_ptr<MessageQueue>(new MessageQueue(mq_shm, mq_buffer_shm));
}

MessageQueue::MessageQueue(
    AllocatedSharedMemory<MessageQueueShm>& mq_shm,
    AllocatedSharedMemory<bi::managed_external_buffer::handle_t>& mq_buffer_shm)
    : mq_shm_(std::move(mq_shm)), mq_buffer_shm_(std::move(mq_buffer_shm))
{
  mq_buffer_shm_ptr_ = mq_buffer_shm_.data_.get();
  mq_shm_ptr_ = mq_shm_.data_.get();
  mq_handle_ = mq_shm_.handle_;
}

void
MessageQueue::Push(
    bi::managed_external_buffer::handle_t message, int const& duration,
    bool& success)
{
  boost::system_time timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(duration);

  while (true) {
    try {
      if (!SemEmptyMutable()->timed_wait(timeout)) {
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
    bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable(), timeout};
    if (!lock) {
      SemEmptyMutable()->post();
      success = false;
      return;
    }
    success = true;

    Buffer()[Index()] = message;
    Index()++;
  }
  SemFullMutable()->post();
}

void
MessageQueue::Push(bi::managed_external_buffer::handle_t message)
{
  while (true) {
    try {
      SemEmptyMutable()->wait();
      break;
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable()};
    Buffer()[Index()] = message;
    Index()++;
  }
  SemFullMutable()->post();
}

bi::managed_external_buffer::handle_t
MessageQueue::Pop()
{
  off_t message;

  while (true) {
    try {
      SemFullMutable()->wait();
      break;
    }
    catch (bi::interprocess_exception& ex) {
    }
  }

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable()};
    message = Buffer()[Index() - 1];
    Index()--;
  }
  SemEmptyMutable()->post();

  return message;
}

bi::managed_external_buffer::handle_t
MessageQueue::Pop(int const& duration, bool& success)
{
  off_t message = 0;
  boost::system_time timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(duration);

  while (true) {
    try {
      if (!SemFullMutable()->timed_wait(timeout)) {
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
    bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable(), timeout};
    if (!lock) {
      SemFullMutable()->post();
      success = false;
      return message;
    }
    success = true;

    message = Buffer()[Index() - 1];
    Index()--;
  }
  SemEmptyMutable()->post();

  return message;
}

void
MessageQueue::ResetSemaphores()
{
  new (SemFullMutable()) bi::interprocess_semaphore(0);
  new (SemEmptyMutable()) bi::interprocess_semaphore(Size());
  new (MutexMutable()) bi::interprocess_mutex;
}

std::unique_ptr<MessageQueue>
MessageQueue::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t message_queue_offset)
{
  AllocatedSharedMemory<MessageQueueShm> mq_shm =
      shm_pool->Load<MessageQueueShm>(message_queue_offset);
  AllocatedSharedMemory<bi::managed_external_buffer::handle_t> mq_shm_buffer =
      shm_pool->Load<bi::managed_external_buffer::handle_t>(
          mq_shm.data_->buffer);

  return std::unique_ptr<MessageQueue>(new MessageQueue(mq_shm, mq_shm_buffer));
}

bi::managed_external_buffer::handle_t
MessageQueue::ShmOffset()
{
  return mq_handle_;
}

void
MessageQueue::Release()
{
  if (mq_shm_.data_ != nullptr) {
    mq_shm_.data_.release();
  }

  if (mq_buffer_shm_.data_ != nullptr) {
    mq_buffer_shm_.data_.release();
  }
}
}}}  // namespace triton::backend::python
