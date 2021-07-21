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
    py::object error)
    : output_tensors_(std::move(output_tensors)), error_(error)
{
}

std::vector<std::shared_ptr<PbTensor>>&
InferResponse::OutputTensors()
{
  return output_tensors_;
}

#ifdef TRITON_PB_STUB
bool
InferResponse::HasError()
{
  return !error_.is(py::none());
}
#endif

void
InferResponse::SaveToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Response* response_shm)
{
  size_t output_tensor_length = output_tensors_.size();

  Tensor* output_tensors_shm;
  off_t output_tensors_offset;
  shm_pool->Map(
      (char**)&output_tensors_shm, sizeof(Tensor) * output_tensor_length,
      output_tensors_offset);
  response_shm->outputs = output_tensors_offset;
  response_shm->outputs_size = output_tensor_length;

  size_t j = 0;
  for (auto& output_tensor : output_tensors_) {
    Tensor* output_tensor_shm = &output_tensors_shm[j];
    output_tensor->SaveToSharedMemory(
        shm_pool, output_tensor_shm);
    j++;
  }
}

py::object
InferResponse::Error()
{
  return error_;
}

}}}  // namespace triton::backend::python
