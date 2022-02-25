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

#include "infer_request.h"
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "pb_utils.h"
#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

InferRequest::InferRequest(
    const std::string& request_id, uint64_t correlation_id,
    const std::vector<std::shared_ptr<PbTensor>>& inputs,
    const std::vector<std::string>& requested_output_names,
    const std::string& model_name, const int64_t model_version,
    const uint32_t flags)
    : request_id_(request_id), correlation_id_(correlation_id), inputs_(inputs),
      requested_output_names_(requested_output_names), model_name_(model_name),
      model_version_(model_version), flags_(flags)
{
}

const std::vector<std::shared_ptr<PbTensor>>&
InferRequest::Inputs()
{
  return inputs_;
}

const std::string&
InferRequest::RequestId()
{
  return request_id_;
}

uint64_t
InferRequest::CorrelationId()
{
  return correlation_id_;
}

const std::vector<std::string>&
InferRequest::RequestedOutputNames()
{
  return requested_output_names_;
}

const std::string&
InferRequest::ModelName()
{
  return model_name_;
}

int64_t
InferRequest::ModelVersion()
{
  return model_version_;
}

uint32_t
InferRequest::Flags()
{
  return flags_;
}

void
InferRequest::SetFlags(uint32_t flags)
{
  flags_ = flags;
}

bi::managed_external_buffer::handle_t
InferRequest::ShmHandle()
{
  return shm_handle_;
}

void
InferRequest::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<char> infer_request_shm = shm_pool->Construct<char>(
      sizeof(InferRequestShm) +
      (RequestedOutputNames().size() *
       sizeof(bi::managed_external_buffer::handle_t)) +
      (Inputs().size() * sizeof(bi::managed_external_buffer::handle_t)) +
      PbString::ShmStructSize(ModelName()) +
      PbString::ShmStructSize(RequestId()));

  infer_request_shm_ptr_ =
      reinterpret_cast<InferRequestShm*>(infer_request_shm.data_.get());
  infer_request_shm_ptr_->correlation_id = CorrelationId();
  infer_request_shm_ptr_->input_count = Inputs().size();
  infer_request_shm_ptr_->model_version = model_version_;
  infer_request_shm_ptr_->requested_output_count =
      RequestedOutputNames().size();

  output_names_handle_shm_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm));

  // [FIXME] Make this part of the memory layout too.
  size_t i = 0;
  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  for (auto& requested_output_name : requested_output_names_) {
    std::unique_ptr<PbString> requested_output_name_shm =
        PbString::Create(shm_pool, requested_output_name);
    output_names_handle_shm_ptr_[i] = requested_output_name_shm->ShmHandle();
    requested_output_names_shm.emplace_back(
        std::move(requested_output_name_shm));
    i++;
  }

  input_tensors_handle_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(output_names_handle_shm_ptr_) +
          sizeof(bi::managed_external_buffer::handle_t) *
              RequestedOutputNames().size());
  i = 0;
  for (auto& input : Inputs()) {
    input_tensors_handle_ptr_[i] = input->ShmHandle();
    i++;
  }

  size_t model_name_offset =
      sizeof(InferRequestShm) +
      (RequestedOutputNames().size() *
       sizeof(bi::managed_external_buffer::handle_t)) +
      (Inputs().size() * sizeof(bi::managed_external_buffer::handle_t));

  std::unique_ptr<PbString> model_name_shm = PbString::Create(
      ModelName(),
      reinterpret_cast<char*>(infer_request_shm_ptr_) + model_name_offset,
      infer_request_shm.handle_ + model_name_offset);

  size_t request_id_offset =
      model_name_offset + PbString::ShmStructSize(ModelName());
  std::unique_ptr<PbString> request_id_shm = PbString::Create(
      RequestId(),
      reinterpret_cast<char*>(infer_request_shm_ptr_) + request_id_offset,
      infer_request_shm.handle_ + request_id_offset);

  // Save the references to shared memory.
  infer_request_shm_ = std::move(infer_request_shm);
  request_id_shm_ = std::move(request_id_shm);
  model_name_shm_ = std::move(model_name_shm);
  shm_handle_ = infer_request_shm_.handle_;
  requested_output_names_shm_ = std::move(requested_output_names_shm);
}

std::unique_ptr<InferRequest>
InferRequest::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t request_handle)
{
  AllocatedSharedMemory<char> infer_request_shm =
      shm_pool->Load<char>(request_handle);
  InferRequestShm* infer_request_shm_ptr =
      reinterpret_cast<InferRequestShm*>(infer_request_shm.data_.get());

  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  uint32_t requested_output_count =
      infer_request_shm_ptr->requested_output_count;

  bi::managed_external_buffer::handle_t* output_names_handle_shm_ptr =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          (reinterpret_cast<char*>(infer_request_shm_ptr) +
           sizeof(InferRequestShm)));

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    std::unique_ptr<PbString> pb_string = PbString::LoadFromSharedMemory(
        shm_pool, output_names_handle_shm_ptr[output_idx]);
    requested_output_names_shm.emplace_back(std::move(pb_string));
  }

  bi::managed_external_buffer::handle_t* input_names_handle_shm_ptr =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          (reinterpret_cast<char*>(infer_request_shm_ptr) +
           sizeof(InferRequestShm) +
           (infer_request_shm_ptr->requested_output_count *
            sizeof(bi::managed_external_buffer::handle_t))));

  std::vector<std::shared_ptr<PbTensor>> input_tensors;
  for (size_t input_idx = 0; input_idx < infer_request_shm_ptr->input_count;
       ++input_idx) {
    std::shared_ptr<PbTensor> input_tensor = PbTensor::LoadFromSharedMemory(
        shm_pool, input_names_handle_shm_ptr[input_idx]);
    input_tensors.emplace_back(std::move(input_tensor));
  }

  size_t model_name_offset =
      sizeof(InferRequestShm) +
      (requested_output_count * sizeof(bi::managed_external_buffer::handle_t)) +
      (infer_request_shm_ptr->input_count *
       sizeof(bi::managed_external_buffer::handle_t));

  std::unique_ptr<PbString> model_name_shm = PbString::LoadFromSharedMemory(
      request_handle + model_name_offset,
      reinterpret_cast<char*>(infer_request_shm_ptr) + model_name_offset);

  size_t request_id_offset = model_name_offset + model_name_shm->Size();
  std::unique_ptr<PbString> request_id_shm = PbString::LoadFromSharedMemory(
      request_handle + request_id_offset,
      reinterpret_cast<char*>(infer_request_shm_ptr) + request_id_offset);

  return std::unique_ptr<InferRequest>(new InferRequest(
      infer_request_shm, request_id_shm, requested_output_names_shm,
      model_name_shm, input_tensors));
}

InferRequest::InferRequest(
    AllocatedSharedMemory<char>& infer_request_shm,
    std::unique_ptr<PbString>& request_id_shm,
    std::vector<std::unique_ptr<PbString>>& requested_output_names_shm,
    std::unique_ptr<PbString>& model_name_shm,
    std::vector<std::shared_ptr<PbTensor>>& input_tensors)
    : infer_request_shm_(std::move(infer_request_shm)),
      request_id_shm_(std::move(request_id_shm)),
      requested_output_names_shm_(std::move(requested_output_names_shm)),
      model_name_shm_(std::move(model_name_shm))
{
  infer_request_shm_ptr_ =
      reinterpret_cast<InferRequestShm*>(infer_request_shm_.data_.get());
  output_names_handle_shm_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm));
  input_tensors_handle_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm) +
          sizeof(bi::managed_external_buffer::handle_t) *
              infer_request_shm_ptr_->requested_output_count);
  inputs_ = std::move(input_tensors);

  std::vector<std::string> requested_output_names;
  for (size_t output_idx = 0;
       output_idx < infer_request_shm_ptr_->requested_output_count;
       ++output_idx) {
    auto& pb_string = requested_output_names_shm_[output_idx];
    requested_output_names.emplace_back(pb_string->String());
  }

  request_id_ = request_id_shm_->String();
  requested_output_names_ = std::move(requested_output_names);
  model_name_ = model_name_shm_->String();
  flags_ = infer_request_shm_ptr_->flags;
  model_version_ = infer_request_shm_ptr_->model_version;
  correlation_id_ = infer_request_shm_ptr_->correlation_id;
}

}}}  // namespace triton::backend::python
