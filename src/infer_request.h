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

#include <string>
#include "infer_response.h"
#include "pb_tensor.h"

namespace triton { namespace backend { namespace python {
class InferRequest {
  std::string request_id_;
  uint64_t correlation_id_;
  std::vector<std::shared_ptr<PbTensor>> inputs_;
  std::vector<std::string> requested_output_names_;
  std::string model_name_;
  int64_t model_version_;

 public:
  InferRequest(
      const std::string& request_id, uint64_t correlation_id,
      const std::vector<std::shared_ptr<PbTensor>>& inputs,
      const std::vector<std::string>& requested_output_names,
      const std::string& model_name, const int64_t model_version);

  const std::vector<std::shared_ptr<PbTensor>>& Inputs();
  const std::string& RequestId();
  uint64_t CorrelationId();
  const std::string& ModelName();
  int64_t ModelVersion();
  const std::vector<std::string>& RequestedOutputNames();

  /// Save an Inference Request to shared memory.
  /// \param shm_pool Shared memory pool to save the inference request.
  /// \param request_shm A pointer to a location in shared memory with enough
  /// space to save the inference request.
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, Request* request_shm);

  /// Create an Inference Request object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param request_offset Shared memory offset of the request.
  static std::unique_ptr<InferRequest> LoadFromSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, off_t request_offset);
#ifdef TRITON_PB_STUB
  std::unique_ptr<InferResponse> Exec();
#endif
};
}}};  // namespace triton::backend::python
