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

#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

typedef enum PYTHONSTUB_commandtype_enum {
  PYTHONSTUB_ExecuteRequest,
  PYTHONSTUB_ExecuteResposne,
  PYTHONSTUB_InitializeRequest,
  PYTHONSTUB_InitializeResponse,
  PYTHONSTUB_FinalizeRequest,
  PYTHONSTUB_FinalizeResponse,
  PYTHONSTUB_TensorCleanup,
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
  bool is_resposne = false;
  off_t request_offset;
};

class IPCMessage {
  struct IPCMessageShm* ipc_message_shm_;
  off_t shm_offset_;

 public:
  IPCMessage() {}
  IPCMessage(const std::unique_ptr<SharedMemory>& shm_pool)
  {
    shm_pool->Map(
        (char**)&ipc_message_shm_, sizeof(IPCMessageShm), shm_offset_);
  }

  off_t SharedMemoryOffset() { return shm_offset_; }

  static std::unique_ptr<IPCMessage> LoadFromSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, off_t message_offset);
  PYTHONSTUB_CommandType& Command();
  bool& IsResponse();
  off_t& RequestOffset();
  off_t& Args();
};

}}};  // namespace triton::backend::python
