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
#include "scoped_defer.h"
#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

InferRequest::InferRequest(
    const std::string& request_id, uint64_t correlation_id,
    const std::vector<std::shared_ptr<PbTensor>>& inputs,
    const std::set<std::string>& requested_output_names,
    const std::string& model_name, const int64_t model_version,
    const uint32_t flags, const intptr_t response_factory_address,
    const intptr_t request_address)
    : request_id_(request_id), correlation_id_(correlation_id), inputs_(inputs),
      requested_output_names_(requested_output_names), model_name_(model_name),
      model_version_(model_version), flags_(flags),
      response_factory_address_(response_factory_address),
      request_address_(request_address)
{
  for (auto& input : inputs) {
    if (!input) {
      throw PythonBackendException(
          "Input tensor for request with id '" + request_id +
          "' and model name '" + model_name + "' should not be empty.");
    }
  }

  for (auto& requested_output_name : requested_output_names) {
    if (requested_output_name == "") {
      throw PythonBackendException(
          "Requested output name for request with id '" + request_id +
          "' and model name '" + model_name + "' should not be empty.");
    }
  }

  inputs_ = inputs;
  requested_output_names_ = requested_output_names;
#ifdef TRITON_PB_STUB
  response_sender_ = std::make_shared<ResponseSender>(
      request_address_, response_factory_address_,
      Stub::GetOrCreateInstance()->SharedMemory());
#endif
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

const std::set<std::string>&
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

intptr_t
InferRequest::RequestAddress()
{
  return request_address_;
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
  infer_request_shm_ptr_->flags = Flags();
  infer_request_shm_ptr_->address = request_address_;
  infer_request_shm_ptr_->response_factory_address = response_factory_address_;

  output_names_handle_shm_ptr_ =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(infer_request_shm_ptr_) +
          sizeof(InferRequestShm));

  // [FIXME] This could also be a part of the single allocated memory for this
  // object.
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
    bi::managed_external_buffer::handle_t request_handle, bool open_cuda_handle)
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
        shm_pool, input_names_handle_shm_ptr[input_idx], open_cuda_handle);
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

  std::set<std::string> requested_output_names;
  for (size_t output_idx = 0;
       output_idx < infer_request_shm_ptr_->requested_output_count;
       ++output_idx) {
    auto& pb_string = requested_output_names_shm_[output_idx];
    requested_output_names.emplace(pb_string->String());
  }

  request_id_ = request_id_shm_->String();
  requested_output_names_ = std::move(requested_output_names);
  model_name_ = model_name_shm_->String();
  flags_ = infer_request_shm_ptr_->flags;
  model_version_ = infer_request_shm_ptr_->model_version;
  correlation_id_ = infer_request_shm_ptr_->correlation_id;
  request_address_ = infer_request_shm_ptr_->address;
  response_factory_address_ = infer_request_shm_ptr_->response_factory_address;

#ifdef TRITON_PB_STUB
  response_sender_ = std::make_shared<ResponseSender>(
      request_address_, response_factory_address_,
      Stub::GetOrCreateInstance()->SharedMemory());
#endif
}

#ifndef TRITON_PB_STUB
TRITONSERVER_Error*
InferRequest::DeleteResponseFactory()
{
  TRITONBACKEND_ResponseFactory* response_factory =
      reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
          response_factory_address_);
  TRITONSERVER_Error* error =
      TRITONBACKEND_ResponseFactoryDelete(response_factory);

  return error;
}
#endif

#ifdef TRITON_PB_STUB
std::shared_ptr<ResponseSender>
InferRequest::GetResponseSender()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  if (!stub->IsDecoupled()) {
    throw PythonBackendException(
        "'get_response_sender' function must be called only when the model is "
        "using the decoupled transaction policy.");
  }

  return response_sender_;
}


std::shared_ptr<InferResponse>
InferRequest::Exec()
{
  ResponseBatch* response_batch = nullptr;
  bool responses_is_set = false;
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  std::unique_ptr<SharedMemoryManager>& shm_pool = stub->SharedMemory();
  bi::managed_external_buffer::handle_t* response_handle = nullptr;

  PythonBackendException pb_exception(std::string{});
  std::unique_ptr<IPCMessage> ipc_message;

  AllocatedSharedMemory<char> request_batch;
  ScopedDefer data_load_complete([&ipc_message] {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    ipc_message->ResponseCondition()->notify_all();
  });

  try {
    py::gil_scoped_release release;
    ipc_message = IPCMessage::Create(shm_pool, true /* inline_response */);
    bool has_exception = false;
    PythonBackendException pb_exception(std::string{});

    ipc_message->Command() =
        PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest;

    request_batch = shm_pool->Construct<char>(
        sizeof(RequestBatch) + sizeof(bi::managed_external_buffer::handle_t));

    RequestBatch* request_batch_shm_ptr =
        reinterpret_cast<RequestBatch*>(request_batch.data_.get());
    request_batch_shm_ptr->batch_size = 1;
    ipc_message->Args() = request_batch.handle_;

    bi::managed_external_buffer::handle_t* requests_shm =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            request_batch.data_.get() + sizeof(RequestBatch));
    request_batch_shm_ptr->batch_size = 1;

    bool has_gpu_tensor = false;
    size_t i = 0;
    for (auto& input_tensor : inputs_) {
      input_tensor->SaveToSharedMemory(shm_pool, false /* copy_gpu */);
      if (!input_tensor->IsCPU()) {
        has_gpu_tensor = true;
      }
      ++i;
    }

    SaveToSharedMemory(shm_pool);

    // Save the shared memory offset of the request.
    *requests_shm = ShmHandle();

    // Send the BLS request to the parent process and wait for the response.
    {
      bi::scoped_lock<bi::interprocess_mutex> lock{
          *(ipc_message->ResponseMutex())};
      stub->SendIPCMessage(ipc_message);
      ipc_message->ResponseCondition()->wait(lock);
    }

    // Additional round trip required for asking the stub process
    // to fill in the GPU tensor buffers
    if (has_gpu_tensor) {
      AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
          gpu_buffers_handle =
              shm_pool->Load<bi::managed_external_buffer::handle_t>(
                  request_batch_shm_ptr->gpu_buffers_handle);
      try {
#ifdef TRITON_ENABLE_GPU
        size_t i = 0;
        for (auto& input_tensor : this->Inputs()) {
          if (!input_tensor->IsCPU()) {
            std::unique_ptr<PbMemory> dst_buffer =
                PbMemory::LoadFromSharedMemory(
                    shm_pool, (gpu_buffers_handle.data_.get())[i],
                    true /* open cuda handle */);
            PbMemory::CopyBuffer(dst_buffer, input_tensor->Memory());
            ++i;
          }
        }
#endif  // TRITON_ENABLE_GPU
      }
      catch (const PythonBackendException& exception) {
        // We need to catch the exception here. Otherwise, we will not notify
        // the main process and it will wait for the response forever.
        pb_exception = exception;
        has_exception = true;
      }

      {
        bi::scoped_lock<bi::interprocess_mutex> lock{
            *(ipc_message->ResponseMutex())};
        ipc_message->ResponseCondition()->notify_all();
        ipc_message->ResponseCondition()->wait(lock);
      }
    }

    // The exception will be thrown after the message was sent to the main
    // process.
    if (has_exception) {
      throw pb_exception;
    }

    // Get the response for the current message.
    std::unique_ptr<IPCMessage> bls_response = IPCMessage::LoadFromSharedMemory(
        shm_pool, ipc_message->ResponseHandle());

    AllocatedSharedMemory<char> response_batch_shm =
        shm_pool->Load<char>(bls_response->Args());
    response_batch =
        reinterpret_cast<ResponseBatch*>(response_batch_shm.data_.get());
    response_handle = reinterpret_cast<bi::managed_external_buffer::handle_t*>(
        response_batch_shm.data_.get() + sizeof(ResponseBatch));

    responses_is_set = true;
    if (response_batch->has_error) {
      if (response_batch->is_error_set) {
        std::unique_ptr<PbString> pb_string =
            PbString::LoadFromSharedMemory(shm_pool, response_batch->error);
        return std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(pb_string->String()));
      } else {
        return std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(
                "An error occurred while performing BLS request."));
      }
    }
  }
  catch (const PythonBackendException& pb_exception) {
    return std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{},
        std::make_shared<PbError>(pb_exception.what()));
  }

  if (responses_is_set) {
    std::unique_ptr<InferResponse> infer_response =
        InferResponse::LoadFromSharedMemory(
            shm_pool, *response_handle, true /* open cuda handle */);
    auto& memory_manager_message_queue = stub->MemoryManagerQueue();

    for (auto& output_tensor : infer_response->OutputTensors()) {
      if (!output_tensor->IsCPU()) {
        uint64_t memory_release_id = output_tensor->Memory()->MemoryReleaseId();
        output_tensor->Memory()->SetMemoryReleaseCallback(
            [&memory_manager_message_queue, memory_release_id]() {
              memory_manager_message_queue->Push(memory_release_id);
            });
      }
    }

    return infer_response;
  } else {
    return std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{},
        std::make_shared<PbError>(
            "An error occurred while performing BLS request."));
  }
}

#endif

}}}  // namespace triton::backend::python
