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

#include "pb_error.h"
#include "pb_tensor.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

struct ResponseShm {
  uint32_t outputs_size;
  bi::managed_external_buffer::handle_t error;
  bool has_error;
  // Indicates whether this error has a message or not.
  bool is_error_set;
};

class InferResponse {
 public:
  InferResponse(
      const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError> error = nullptr);
  std::vector<std::shared_ptr<PbTensor>>& OutputTensors();
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu = true);
  static std::unique_ptr<InferResponse> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t response_handle,
      bool open_cuda_handle);
  bool HasError();
  std::shared_ptr<PbError>& Error();
  bi::managed_external_buffer::handle_t ShmHandle();

  // Disallow copying the inference response object.
  DISALLOW_COPY_AND_ASSIGN(InferResponse);

 private:
  InferResponse(
      AllocatedSharedMemory<char>& response_shm,
      std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError>& pb_error);
  std::vector<std::shared_ptr<PbTensor>> output_tensors_;
  std::shared_ptr<PbError> error_;
  bi::managed_external_buffer::handle_t shm_handle_;
  AllocatedSharedMemory<char> response_shm_;
};
}}}  // namespace triton::backend::python
