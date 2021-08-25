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

#include "ipc_message.h"

#include <memory>

namespace triton { namespace backend { namespace python {

std::unique_ptr<IPCMessage>
IPCMessage::LoadFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t message_offset)
{
  std::unique_ptr<IPCMessage> ipc_message = std::make_unique<IPCMessage>();
  ipc_message->shm_offset_ = message_offset;
  shm_pool->MapOffset((char**)&ipc_message->ipc_message_shm_, message_offset);

  if (ipc_message->ipc_message_shm_->inline_response) {
    shm_pool->MapOffset(
        (char**)&ipc_message->response_mutex_,
        ipc_message->ipc_message_shm_->response_mutex);
    shm_pool->MapOffset(
        (char**)&ipc_message->response_cond_,
        ipc_message->ipc_message_shm_->response_cond);
  }

  return ipc_message;
}

PYTHONSTUB_CommandType&
IPCMessage::Command()
{
  return ipc_message_shm_->command;
}

off_t&
IPCMessage::Args()
{
  return ipc_message_shm_->args;
}

bool&
IPCMessage::InlineResponse()
{
  return ipc_message_shm_->inline_response;
}

bi::interprocess_condition*
IPCMessage::ResponseCondition()
{
  return response_cond_;
}

bi::interprocess_mutex*
IPCMessage::ResponseMutex()
{
  return response_mutex_;
}

off_t&
IPCMessage::RequestOffset()
{
  return this->ipc_message_shm_->request_offset;
}

}}};  // namespace triton::backend::python
