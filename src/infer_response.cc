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

#include "infer_response.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

namespace triton { namespace backend { namespace python {

InferResponse::InferResponse(
    const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
    std::shared_ptr<PbError> error)
    : output_tensors_(std::move(output_tensors)), error_(error)
{
}

std::vector<std::shared_ptr<PbTensor>>&
InferResponse::OutputTensors()
{
  return output_tensors_;
}

bool
InferResponse::HasError()
{
  return error_.get() != nullptr;
}

void
InferResponse::SaveToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  size_t output_tensor_length = output_tensors_.size();
  response_shm_ = shm_pool->Construct<ResponseShm>();
  response_shm_.data_->has_error = false;
  response_shm_.data_->is_error_set = false;

  // Only save the output tensors to shared memory when the inference response
  // doesn't have error.
  if (HasError()) {
    response_shm_.data_->has_error = true;
    Error()->SaveToSharedMemory(shm_pool);

    response_shm_.data_->is_error_set = true;
    response_shm_.data_->error = Error()->ShmOffset();
    response_shm_.data_->outputs_size = 0;
  } else {
    tensor_offset_shm_ =
        shm_pool->ConstructMany<bi::managed_external_buffer::handle_t>(
            output_tensor_length);
    response_shm_.data_->outputs = tensor_offset_shm_.handle_;
    response_shm_.data_->outputs_size = output_tensor_length;

    size_t j = 0;
    for (auto& output_tensor : output_tensors_) {
      output_tensor->SaveToSharedMemory(shm_pool);
      (tensor_offset_shm_.data_.get())[j] = output_tensor->ShmOffset();
      j++;
    }
  }
}

bi::managed_external_buffer::handle_t
InferResponse::ShmOffset()
{
  return shm_offset_;
}

std::unique_ptr<InferResponse>
InferResponse::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t response_offset)
{
  AllocatedSharedMemory<ResponseShm> response_shm =
      shm_pool->Load<ResponseShm>(response_offset);
  uint32_t requested_output_count = response_shm.data_->outputs_size;

  std::shared_ptr<PbError> pb_error;
  std::vector<std::shared_ptr<PbTensor>> output_tensors;
  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      tensor_offset_shm;

  // If the error field is set, do not load output tensors from shared memory.
  if (response_shm.data_->has_error && response_shm.data_->is_error_set) {
    pb_error =
        PbError::LoadFromSharedMemory(shm_pool, response_shm.data_->error);
  } else if (
      response_shm.data_->has_error && !response_shm.data_->is_error_set) {
    pb_error = std::make_shared<PbError>("");
  } else {
    tensor_offset_shm = shm_pool->Load<bi::managed_external_buffer::handle_t>(
        response_shm.data_->outputs);
    for (size_t idx = 0; idx < requested_output_count; ++idx) {
      std::shared_ptr<PbTensor> pb_tensor = PbTensor::LoadFromSharedMemory(
          shm_pool, (tensor_offset_shm.data_.get())[idx]);
      output_tensors.emplace_back(std::move(pb_tensor));
    }
  }

  return std::unique_ptr<InferResponse>(new InferResponse(
      response_shm, output_tensors, pb_error, tensor_offset_shm));
}

InferResponse::InferResponse(
    AllocatedSharedMemory<ResponseShm>& response_shm,
    std::vector<std::shared_ptr<PbTensor>>& output_tensors,
    std::shared_ptr<PbError>& pb_error,
    AllocatedSharedMemory<bi::managed_external_buffer::handle_t>&
        tensor_offset_shm)
{
  response_shm_ = std::move(response_shm);
  output_tensors_ = std::move(output_tensors);
  error_ = std::move(pb_error);
  tensor_offset_shm_ = std::move(tensor_offset_shm);
}

std::shared_ptr<PbError>&
InferResponse::Error()
{
  return error_;
}

}}}  // namespace triton::backend::python
