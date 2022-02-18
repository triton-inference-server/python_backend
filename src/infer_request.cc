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
InferRequest::ShmOffset()
{
  return shm_offset_;
}

void
InferRequest::Release()
{
  infer_request_shm_.data_.release();
  request_id_shm_->Release();
  for (auto& requested_output_shm : requested_output_names_shm_) {
    requested_output_shm->Release();
  }

  for (auto& input : inputs_) {
    input->Release();
  }

  model_name_shm_->Release();
  output_names_handle_shm_.data_.release();
  input_tensors_handle_.data_.release();
}

void
InferRequest::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<InferRequestShm> infer_request_shm =
      shm_pool->Construct<InferRequestShm>();

  infer_request_shm.data_->correlation_id = CorrelationId();

  std::unique_ptr<PbString> request_id_shm =
      PbString::Create(shm_pool, RequestId());

  infer_request_shm.data_->id = request_id_shm->ShmOffset();
  infer_request_shm.data_->requested_output_count =
      RequestedOutputNames().size();

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      output_names_handle =
          shm_pool->Construct<bi::managed_external_buffer::handle_t>(
              RequestedOutputNames().size());
  infer_request_shm.data_->requested_output_names = output_names_handle.handle_;

  size_t i = 0;
  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  for (auto& requested_output_name : requested_output_names_) {
    std::unique_ptr<PbString> requested_output_name_shm =
        PbString::Create(shm_pool, requested_output_name);
    (output_names_handle.data_.get())[i] =
        requested_output_name_shm->ShmOffset();
    requested_output_names_shm.emplace_back(
        std::move(requested_output_name_shm));
    i++;
  }

  infer_request_shm.data_->input_count = Inputs().size();
  infer_request_shm.data_->model_version = model_version_;

  std::unique_ptr<PbString> model_name_shm =
      PbString::Create(shm_pool, ModelName());
  infer_request_shm.data_->model_name = model_name_shm->ShmOffset();

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      input_tensors_handle =
          shm_pool->Construct<bi::managed_external_buffer::handle_t>(
              Inputs().size());

  infer_request_shm.data_->inputs = input_tensors_handle.handle_;
  i = 0;
  for (auto& input : Inputs()) {
    (input_tensors_handle.data_.get())[i] = input->ShmOffset();
    i++;
  }


  // Save the references to shared memory.
  infer_request_shm_ = std::move(infer_request_shm);
  infer_request_shm_ptr_ = infer_request_shm.data_.get();
  request_id_shm_ = std::move(request_id_shm);
  output_names_handle_shm_ = std::move(output_names_handle);
  requested_output_names_shm_ = std::move(requested_output_names_shm);
  input_tensors_handle_ = std::move(input_tensors_handle);
  input_tensors_handle_ptr_ = input_tensors_handle.data_.get();
  model_name_shm_ = std::move(model_name_shm);
  shm_offset_ = infer_request_shm_.handle_;
}

std::unique_ptr<InferRequest>
InferRequest::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t request_offset)
{
  AllocatedSharedMemory<InferRequestShm> infer_request_shm =
      shm_pool->Load<InferRequestShm>(request_offset);
  std::unique_ptr<PbString> request_id_shm =
      PbString::LoadFromSharedMemory(shm_pool, infer_request_shm.data_->id);

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      input_tensors_handle =
          shm_pool->Load<bi::managed_external_buffer::handle_t>(
              infer_request_shm.data_->inputs);

  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      output_names_handle =
          shm_pool->Load<bi::managed_external_buffer::handle_t>(
              infer_request_shm.data_->requested_output_names);

  std::vector<std::unique_ptr<PbString>> requested_output_names_shm;
  uint32_t requested_output_count =
      infer_request_shm.data_->requested_output_count;

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    std::unique_ptr<PbString> pb_string = PbString::LoadFromSharedMemory(
        shm_pool, (output_names_handle.data_.get())[output_idx]);

    requested_output_names_shm.emplace_back(std::move(pb_string));
  }

  std::vector<std::shared_ptr<PbTensor>> input_tensors;
  for (size_t input_idx = 0; input_idx < infer_request_shm.data_->input_count;
       ++input_idx) {
    std::shared_ptr<PbTensor> input_tensor = PbTensor::LoadFromSharedMemory(
        shm_pool, (input_tensors_handle.data_.get())[input_idx]);
    input_tensors.emplace_back(std::move(input_tensor));
  }

  std::unique_ptr<PbString> model_name_shm = PbString::LoadFromSharedMemory(
      shm_pool, infer_request_shm.data_->model_name);

  return std::unique_ptr<InferRequest>(new InferRequest(
      infer_request_shm, request_id_shm, requested_output_names_shm,
      model_name_shm, output_names_handle, input_tensors_handle,
      input_tensors));
}

InferRequest::InferRequest(
    AllocatedSharedMemory<InferRequestShm>& infer_request_shm,
    std::unique_ptr<PbString>& request_id_shm,
    std::vector<std::unique_ptr<PbString>>& requested_output_names_shm,
    std::unique_ptr<PbString>& model_name_shm,
    AllocatedSharedMemory<bi::managed_external_buffer::handle_t>&
        output_names_handle_shm,
    AllocatedSharedMemory<bi::managed_external_buffer::handle_t>&
        input_tensors_handle,
    std::vector<std::shared_ptr<PbTensor>>& input_tensors)
    : infer_request_shm_(std::move(infer_request_shm)),
      request_id_shm_(std::move(request_id_shm)),
      requested_output_names_shm_(std::move(requested_output_names_shm)),
      model_name_shm_(std::move(model_name_shm)),
      output_names_handle_shm_(std::move(output_names_handle_shm)),
      input_tensors_handle_(std::move(input_tensors_handle))

{
  infer_request_shm_ptr_ = infer_request_shm_.data_.get();
  output_names_handle_shm_ptr_ = output_names_handle_shm_.data_.get();
  input_tensors_handle_ptr_ = input_tensors_handle_.data_.get();
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
