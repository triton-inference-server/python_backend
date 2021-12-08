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
#include "infer_response.h"
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

InferRequest::InferRequest(
    const std::string& request_id, uint64_t correlation_id,
    const std::vector<std::shared_ptr<PbTensor>>& inputs,
    const std::vector<std::string>& requested_output_names,
    const std::string& model_name, const int64_t model_version)
    : request_id_(request_id), correlation_id_(correlation_id), inputs_(inputs),
      requested_output_names_(requested_output_names), model_name_(model_name),
      model_version_(model_version)
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

void
InferRequest::SaveToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Request* request_shm)
{
  request_shm->correlation_id = this->CorrelationId();
  off_t id_offset;
  SaveStringToSharedMemory(shm_pool, id_offset, this->RequestId().c_str());
  request_shm->id = id_offset;
  request_shm->requested_output_count = this->RequestedOutputNames().size();
  off_t requested_output_names_offset;
  off_t* requested_output_names;
  shm_pool->Map(
      (char**)&requested_output_names,
      sizeof(off_t) * request_shm->requested_output_count,
      requested_output_names_offset);

  request_shm->requested_output_names = requested_output_names_offset;
  size_t i = 0;
  for (auto& requested_output_name : requested_output_names_) {
    SaveStringToSharedMemory(
        shm_pool, requested_output_names[i], requested_output_name.c_str());
    i++;
  }

  request_shm->requested_input_count = this->Inputs().size();
  request_shm->model_version = this->model_version_;
  SaveStringToSharedMemory(
      shm_pool, request_shm->model_name, this->model_name_.c_str());
}

std::unique_ptr<InferRequest>
InferRequest::LoadFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t request_offset,
    std::shared_ptr<std::mutex>& cuda_ipc_open_mutex,
    std::shared_ptr<std::mutex>& cuda_ipc_close_mutex)
{
  Request* request;
  shm_pool->MapOffset((char**)&request, request_offset);

  char* id = nullptr;
  LoadStringFromSharedMemory(shm_pool, request->id, id);

  uint32_t requested_input_count = request->requested_input_count;

  std::vector<std::shared_ptr<PbTensor>> py_input_tensors;
  for (size_t input_idx = 0; input_idx < requested_input_count; ++input_idx) {
    std::shared_ptr<PbTensor> pb_input_tensor = PbTensor::LoadFromSharedMemory(
        shm_pool, request->inputs + sizeof(Tensor) * input_idx,
        cuda_ipc_open_mutex, cuda_ipc_close_mutex);
    py_input_tensors.emplace_back(std::move(pb_input_tensor));
  }

  std::vector<std::string> requested_output_names;
  uint32_t requested_output_count = request->requested_output_count;
  off_t* output_names;
  shm_pool->MapOffset((char**)&output_names, request->requested_output_names);

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    char* output_name = nullptr;
    LoadStringFromSharedMemory(shm_pool, output_names[output_idx], output_name);
    requested_output_names.emplace_back(output_name);
  }

  char* model_name;
  LoadStringFromSharedMemory(shm_pool, request->model_name, model_name);
  return std::make_unique<InferRequest>(
      id, request->correlation_id, std::move(py_input_tensors),
      requested_output_names, model_name, request->model_version);
}

#ifdef TRITON_PB_STUB
std::unique_ptr<InferResponse>
InferRequest::Exec()
{
  ResponseBatch* response_batch = nullptr;
  bool responses_is_set = false;
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  std::unique_ptr<SharedMemory>& shm_pool = stub->GetSharedMemory();

  try {
    py::gil_scoped_release release;
    std::unique_ptr<IPCMessage> ipc_message =
        std::make_unique<IPCMessage>(shm_pool, true /* inline_response */);
    bool has_exception = false;
    PythonBackendException pb_exception(std::string{});

    ipc_message->Command() =
        PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest;

    RequestBatch* request_batch;
    shm_pool->Map(
        (char**)&request_batch, sizeof(RequestBatch), ipc_message->Args());
    request_batch->batch_size = 1;

    Request* request;
    shm_pool->Map((char**)&request, sizeof(Request), request_batch->requests);

    request->requested_input_count = this->Inputs().size();
    Tensor* tensors;
    bool has_gpu_tensor = false;
    shm_pool->Map(
        (char**)&tensors, sizeof(Tensor) * request->requested_input_count,
        request->inputs);

    size_t i = 0;
    for (auto& input_tensor : inputs_) {
      input_tensor->SaveToSharedMemory(
          shm_pool, &tensors[i], true /* copy_cpu */, false /* copy_gpu */);
      if (!input_tensor->IsCPU()) {
        has_gpu_tensor = true;
      }
      ++i;
    }

    SaveToSharedMemory(shm_pool, request);
    {
      bi::scoped_lock<bi::interprocess_mutex> lock{
          *(ipc_message->ResponseMutex())};
      stub->SendIPCMessage(ipc_message);
      ipc_message->ResponseCondition()->wait(lock);
    }

    if (has_gpu_tensor) {
      try {
        for (auto& input_tensor : this->Inputs()) {
          if (!input_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
            input_tensor->SetCudaIpcMutexes(
                stub->CudaIpcOpenMutex(), stub->CudaIpcCloseMutex());
            input_tensor->LoadGPUData(shm_pool);
#endif  // TRITON_ENABLE_GPU
          }
        }
      }
      catch (const PythonBackendException& exception) {
        // We need to catch the exception here. Otherwise, we will not notify
        // the main process and it will wait for the resposne forever.
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
        shm_pool, ipc_message->RequestOffset());
    shm_pool->MapOffset((char**)&response_batch, bls_response->Args());
    responses_is_set = true;

    if (response_batch->has_error) {
      if (response_batch->is_error_set) {
        char* err_string;
        LoadStringFromSharedMemory(shm_pool, response_batch->error, err_string);
        return std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(err_string));
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
            shm_pool, response_batch->responses, stub->CudaIpcOpenMutex(),
            stub->CudaIpcCloseMutex());

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
