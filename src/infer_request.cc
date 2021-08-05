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
    std::unique_ptr<SharedMemory>& shm_pool, off_t request_offset)
{
  Request* request;
  shm_pool->MapOffset((char**)&request, request_offset);

  char* id = nullptr;
  LoadStringFromSharedMemory(shm_pool, request->id, id);

  uint32_t requested_input_count = request->requested_input_count;

  std::vector<std::shared_ptr<PbTensor>> py_input_tensors;
  for (size_t input_idx = 0; input_idx < requested_input_count; ++input_idx) {
    std::shared_ptr<PbTensor> pb_input_tensor = PbTensor::LoadFromSharedMemory(
        shm_pool, request->inputs + sizeof(Tensor) * input_idx);
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
  IPCMessage* ipc_message = stub->GetIPCMessage();
  try {
    ipc_message->stub_command = PYTHONSTUB_CommandType::PYTHONSTUB_Execute;

    ExecuteArgs* exec_args;
    shm_pool->Map(
        (char**)&exec_args, sizeof(ExecuteArgs), ipc_message->stub_args);

    RequestBatch* request_batch;
    shm_pool->Map(
        (char**)&request_batch, sizeof(RequestBatch), exec_args->request_batch);
    request_batch->batch_size = 1;

    Request* request;
    shm_pool->Map((char**)&request, sizeof(Request), request_batch->requests);

    request->requested_input_count = this->Inputs().size();
    Tensor* tensors;
    shm_pool->Map(
        (char**)&tensors, sizeof(Tensor) * request->requested_input_count,
        request->inputs);

    // TODO: Custom handling for GPU
    size_t i = 0;
    for (auto& input_tensor : this->Inputs()) {
      input_tensor->SaveToSharedMemory(shm_pool, &tensors[i]);
      i += 1;
    }
    this->SaveToSharedMemory(shm_pool, request);

    ipc_message->stub_command =
        PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest;
    stub->NotifyParent();
    stub->WaitForNotification();

    ipc_message->stub_command = PYTHONSTUB_CommandType::PYTHONSTUB_Execute;

    shm_pool->MapOffset((char**)&response_batch, exec_args->response_batch);
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
    return InferResponse::LoadFromSharedMemory(
        shm_pool, response_batch->responses);
  } else {
    return std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{},
        std::make_shared<PbError>(
            "An error occurred while performing BLS request."));
  }
}
#endif

}}}  // namespace triton::backend::python
