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

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include "shm_manager.h"


namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

typedef enum PYTHONSTUB_commandtype_enum {
  PYTHONSTUB_ExecuteRequest,
  PYTHONSTUB_ExecuteResposne,
  PYTHONSTUB_InitializeRequest,
  PYTHONSTUB_InitializeResponse,
  PYTHONSTUB_FinalizeRequest,
  PYTHONSTUB_FinalizeResponse,
  PYTHONSTUB_LoadGPUBuffers,
  PYTHONSTUB_InferExecRequest,
  PYTHONSTUB_InferExecResponse
} PYTHONSTUB_CommandType;

///
/// Shared memory representation of IPCMessage
///
/// \param command determines the IPC command that is going to be passed.
/// \param args determines the shared memory offset for the input parameters.
/// \param is_response determines whether this is a response of another IPC
/// message. If this parameter is set, it must provide the offset of the
/// corresponding request in \param request_offset.
/// \param request_offset determines the request offset.
struct IPCMessageShm {
  PYTHONSTUB_CommandType command;
  off_t args;
  bool inline_response = false;
  off_t request_offset;
  off_t response_mutex;
  off_t response_cond;
};

class IPCMessage {
  struct IPCMessageShm* ipc_message_shm_;
  off_t shm_offset_;
  bi::interprocess_mutex* response_mutex_;
  bi::interprocess_condition* response_cond_;

 public:
  IPCMessage() {}
  IPCMessage(
      const std::unique_ptr<SharedMemory>& shm_pool, bool inline_response)
  {
    shm_pool->Map(
        (char**)&ipc_message_shm_, sizeof(IPCMessageShm), shm_offset_);

    ipc_message_shm_->inline_response = inline_response;
    if (inline_response) {
      shm_pool->Map(
          (char**)&response_mutex_, sizeof(bi::interprocess_mutex) + 15,
          ipc_message_shm_->response_mutex);
      shm_pool->Map(
          (char**)&response_cond_, sizeof(bi::interprocess_condition) + 15,
          ipc_message_shm_->response_cond);

      void* ptr_a = reinterpret_cast<void*>(
          ((uintptr_t)response_cond_ + 15) & ~(uintptr_t)0x0F);
      ipc_message_shm_->response_cond += ((char*)ptr_a - (char*)response_cond_);
      void* ptr_b = reinterpret_cast<void*>(
          ((uintptr_t)response_mutex_ + 15) & ~(uintptr_t)0x0F);
      ipc_message_shm_->response_mutex +=
          ((char*)ptr_b - (char*)response_mutex_);
      response_cond_ = reinterpret_cast<bi::interprocess_condition*>(ptr_a);
      response_mutex_ = reinterpret_cast<bi::interprocess_mutex*>(ptr_b);

      new (response_cond_) bi::interprocess_condition;
      new (response_mutex_) bi::interprocess_mutex;
    }
  }

  off_t SharedMemoryOffset() { return shm_offset_; }

  static std::unique_ptr<IPCMessage> LoadFromSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, off_t message_offset);
  PYTHONSTUB_CommandType& Command();
  bool& InlineResponse();
  off_t& RequestOffset();
  bi::interprocess_condition* ResponseCondition();
  bi::interprocess_mutex* ResponseMutex();
  off_t& Args();
};

}}};  // namespace triton::backend::python
