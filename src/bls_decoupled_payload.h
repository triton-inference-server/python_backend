// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class Stub;

//
// Sharing the info about triton server inference requests
// invoked by this BLS decoupled inference request
//
struct BLSDecoupledInferRequestShm {
    intptr_t request_address;
    intptr_t infer_payload_id;
};
  
class BLSDecoupledInferRequestPayload {
public:
    BLSDecoupledInferRequestPayload(intptr_t request_address, intptr_t infer_payload_id):
    request_address_(request_address), infer_payload_id_(infer_payload_id)
    {
    };

    bi::managed_external_buffer::handle_t BLSDecoupledInferRequestShmHandle();

    /// Save a inference Request invoked with BLS related information to shared memory.
    /// Currently, only the inference request address from the python backend will be included.
    /// Can only be called from the python backend process.
    /// \param shm_pool Shared memory pool
    void SaveBLSDecoupledInferRequestPayloadToSharedMemory(
        std::unique_ptr<SharedMemoryManager>& shm_pool);

    static std::unique_ptr<BLSDecoupledInferRequestPayload> LoadBLSDecoupledInferRequestPayloadFromSharedMemory(
        std::unique_ptr<SharedMemoryManager>& shm_pool,
        bi::managed_external_buffer::handle_t shm_handle);

private:
    intptr_t request_address_;
    intptr_t infer_payload_id_;
    AllocatedSharedMemory<char> bls_decoupled_infer_request_shm_;
    BLSDecoupledInferRequestShm* bls_decoupled_infer_request_shm_ptr_;
};

}}}
