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

#include "bls_decoupled_payload.h"

namespace triton { namespace backend { namespace python {

bi::managed_external_buffer::handle_t
BLSDecoupledInferRequestPayload::BLSDecoupledInferRequestShmHandle()
{
  return bls_decoupled_infer_request_shm_.handle_;
}

void 
BLSDecoupledInferRequestPayload::SaveBLSDecoupledInferRequestPayloadToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool) 
{
  bls_decoupled_infer_request_shm_ = 
    shm_pool->Construct<char>(sizeof(BLSDecoupledInferRequestShm));
  bls_decoupled_infer_request_shm_ptr_ =
      reinterpret_cast<BLSDecoupledInferRequestShm*>(bls_decoupled_infer_request_shm_.data_.get());
  bls_decoupled_infer_request_shm_ptr_->request_address = request_address_;
  bls_decoupled_infer_request_shm_ptr_->infer_payload_id = infer_payload_id_;
}

std::unique_ptr<BLSDecoupledInferRequestPayload> 
BLSDecoupledInferRequestPayload::LoadBLSDecoupledInferRequestPayloadFromSharedMemory(
  std::unique_ptr<SharedMemoryManager>& shm_pool,
  bi::managed_external_buffer::handle_t shm_handle
)
{
  AllocatedSharedMemory<BLSDecoupledInferRequestShm> bls_inference_request_shm =
    shm_pool->Load<BLSDecoupledInferRequestShm>(shm_handle);
  BLSDecoupledInferRequestShm* bls_infer_request_shm_ptr =
    reinterpret_cast<BLSDecoupledInferRequestShm*>(bls_inference_request_shm.data_.get());
  
  return std::unique_ptr<BLSDecoupledInferRequestPayload>(new BLSDecoupledInferRequestPayload(
    bls_infer_request_shm_ptr->request_address, bls_infer_request_shm_ptr->infer_payload_id
  ));
}

}}}
