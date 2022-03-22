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

#include <string>
#include "infer_response.h"
#include "pb_tensor.h"

namespace triton { namespace backend { namespace python {

//
// Inference Request
//
struct InferRequestShm {
  uint64_t correlation_id;
  uint32_t input_count;
  uint32_t requested_output_count;
  int64_t model_version;
  uint32_t flags;
};

class InferRequest {
 public:
  InferRequest(
      const std::string& request_id, uint64_t correlation_id,
      const std::vector<std::shared_ptr<PbTensor>>& inputs,
      const std::vector<std::string>& requested_output_names,
      const std::string& model_name, const int64_t model_version,
      const uint32_t flags = 0);

  const std::vector<std::shared_ptr<PbTensor>>& Inputs();
  const std::string& RequestId();
  uint64_t CorrelationId();
  const std::string& ModelName();
  int64_t ModelVersion();
  uint32_t Flags();
  void SetFlags(uint32_t flags);
  const std::vector<std::string>& RequestedOutputNames();
  bi::managed_external_buffer::handle_t ShmHandle();

#ifdef TRITON_PB_STUB
  std::unique_ptr<InferResponse> Exec();
#endif

  /// Save an Inference Request to shared memory.
  /// \param shm_pool Shared memory pool to save the inference request.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create an Inference Request object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param request_handle Shared memory handle of the request.
  /// \param open_cuda_handle Determines if the tensor in the infer request
  /// object is a GPU tensor, to call the cudaIpcOpenMemHandle to obtain the
  /// tensor or not.
  /// \return Returns the infer request in the specified request_handle
  /// location.
  static std::unique_ptr<InferRequest> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t request_handle,
      bool open_cuda_handle);

  /// Disallow copying the inference request object.
  DISALLOW_COPY_AND_ASSIGN(InferRequest);

  ~InferRequest() {}

 private:
  InferRequest(
      AllocatedSharedMemory<char>& infer_request_shm,
      std::unique_ptr<PbString>& request_id_shm,
      std::vector<std::unique_ptr<PbString>>& requested_output_names_shm,
      std::unique_ptr<PbString>& model_name_shm,
      std::vector<std::shared_ptr<PbTensor>>& input_tensors);

  std::string request_id_;
  uint64_t correlation_id_;
  std::vector<std::shared_ptr<PbTensor>> inputs_;
  std::vector<std::string> requested_output_names_;
  std::string model_name_;
  int64_t model_version_;
  uint32_t flags_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<char> infer_request_shm_;
  InferRequestShm* infer_request_shm_ptr_;

  std::unique_ptr<PbString> request_id_shm_;
  std::vector<std::unique_ptr<PbString>> requested_output_names_shm_;
  std::unique_ptr<PbString> model_name_shm_;
  bi::managed_external_buffer::handle_t* output_names_handle_shm_ptr_;
  bi::managed_external_buffer::handle_t* input_tensors_handle_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
};
}}};  // namespace triton::backend::python
