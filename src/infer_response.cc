// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>

#include "scoped_defer.h"


namespace triton { namespace backend { namespace python {

InferResponse::InferResponse(
    const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
    std::shared_ptr<PbError> error, const bool is_last_response, void* id)
    : error_(error), is_last_response_(is_last_response), id_(id)
{
  for (auto& output : output_tensors) {
    if (!output) {
      throw PythonBackendException(
          "Output tensor for inference response should not be empty.");
    }
  }

  output_tensors_ = output_tensors;
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
    std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu)
{
  size_t output_tensor_length = output_tensors_.size();
  if (HasError()) {
    response_shm_ = shm_pool->Construct<char>(sizeof(ResponseShm));
  } else {
    response_shm_ = shm_pool->Construct<char>(
        sizeof(ResponseShm) +
        output_tensor_length * sizeof(bi::managed_external_buffer::handle_t));
  }

  ResponseShm* response_shm_ptr =
      reinterpret_cast<ResponseShm*>(response_shm_.data_.get());
  response_shm_ptr->has_error = false;
  response_shm_ptr->is_error_set = false;
  shm_handle_ = response_shm_.handle_;
  response_shm_ptr->is_last_response = is_last_response_;

  // Only save the output tensors to shared memory when the inference response
  // doesn't have error.
  if (HasError()) {
    response_shm_ptr->has_error = true;
    Error()->SaveToSharedMemory(shm_pool);

    response_shm_ptr->is_error_set = true;
    response_shm_ptr->error = Error()->ShmHandle();
    response_shm_ptr->outputs_size = 0;
  } else {
    bi::managed_external_buffer::handle_t* tensor_handle_shm_ptr =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            response_shm_.data_.get() + sizeof(ResponseShm));
    response_shm_ptr->outputs_size = output_tensor_length;

    size_t j = 0;
    for (auto& output_tensor : output_tensors_) {
      output_tensor->SaveToSharedMemory(shm_pool, copy_gpu);
      tensor_handle_shm_ptr[j] = output_tensor->ShmHandle();
      j++;
    }
    response_shm_ptr->id = id_;
  }
}

bi::managed_external_buffer::handle_t
InferResponse::ShmHandle()
{
  return shm_handle_;
}

void
InferResponse::PruneOutputTensors(
    const std::set<std::string>& requested_output_names)
{
  for (auto it = output_tensors_.begin(); it != output_tensors_.end();) {
    if (requested_output_names.find((*it)->Name()) ==
        requested_output_names.end()) {
      it = output_tensors_.erase(it);
    } else {
      it++;
    }
  }
}

std::unique_ptr<InferResponse>
InferResponse::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t response_handle,
    bool open_cuda_handle)
{
  AllocatedSharedMemory<char> response_shm =
      shm_pool->Load<char>(response_handle);
  ResponseShm* response_shm_ptr =
      reinterpret_cast<ResponseShm*>(response_shm.data_.get());
  uint32_t requested_output_count = response_shm_ptr->outputs_size;

  std::shared_ptr<PbError> pb_error;
  std::vector<std::shared_ptr<PbTensor>> output_tensors;

  // If the error field is set, do not load output tensors from shared memory.
  if (response_shm_ptr->has_error && response_shm_ptr->is_error_set) {
    pb_error = PbError::LoadFromSharedMemory(shm_pool, response_shm_ptr->error);
  } else if (response_shm_ptr->has_error && !response_shm_ptr->is_error_set) {
    pb_error =
        std::make_shared<PbError>("Failed to retrieve the response error.");
  } else {
    bi::managed_external_buffer::handle_t* tensor_handle_shm =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            response_shm.data_.get() + sizeof(ResponseShm));
#ifdef TRITON_PB_STUB
    // Need to acquire the GIL to avoid hangs.
    py::gil_scoped_acquire acquire;
#endif
    for (size_t idx = 0; idx < requested_output_count; ++idx) {
      std::shared_ptr<PbTensor> pb_tensor = PbTensor::LoadFromSharedMemory(
          shm_pool, tensor_handle_shm[idx], open_cuda_handle);
      output_tensors.emplace_back(std::move(pb_tensor));
    }
  }

  return std::unique_ptr<InferResponse>(new InferResponse(
      response_shm, output_tensors, pb_error,
      response_shm_ptr->is_last_response, response_shm_ptr->id));
}

InferResponse::InferResponse(
    AllocatedSharedMemory<char>& response_shm,
    std::vector<std::shared_ptr<PbTensor>>& output_tensors,
    std::shared_ptr<PbError>& pb_error, const bool is_last_response, void* id)
{
  response_shm_ = std::move(response_shm);
  output_tensors_ = std::move(output_tensors);
  error_ = std::move(pb_error);
  shm_handle_ = response_shm_.handle_;
  id_ = id;
  is_last_response_ = is_last_response;
}

std::shared_ptr<PbError>&
InferResponse::Error()
{
  return error_;
}

void*
InferResponse::Id()
{
  return id_;
}

bool
InferResponse::IsLastResponse()
{
  return is_last_response_;
}

#ifndef TRITON_PB_STUB
void
InferResponse::Send(
    TRITONBACKEND_Response* response, void* cuda_stream,
    bool& requires_deferred_callback, const uint32_t flags,
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    GPUBuffersHelper& gpu_buffer_helper,
    std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>& output_buffers,
    const std::set<std::string>& requested_output_names)
{
  std::shared_ptr<TRITONSERVER_Error*> response_error =
      WrapTritonErrorInSharedPtr(nullptr);
  std::unique_ptr<ScopedDefer> response_error_handling;
  requires_deferred_callback = false;

  // This lambda expression will be called when this function exits, if the
  // inference response doesn't have any GPU tensors. Otherwise, it will be
  // called when the object is destructed or DeferredSendCallback is called.
  response_error_handling =
      std::make_unique<ScopedDefer>([response, response_error, flags] {
        if (response != nullptr) {
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(response, flags, *response_error),
              "failed to send the response.");
        }
      });

  // Moves the response sending callback so that it is not called until the stub
  // process fills in the GPU buffers.
  ScopedDefer deferred_task([this, &requires_deferred_callback,
                             &response_error_handling, &gpu_buffer_helper,
                             response_error, &shm_pool] {
    if (*response_error != nullptr) {
      gpu_buffer_helper.SetError(
          shm_pool, TRITONSERVER_ErrorMessage(*response_error));
    }
    if (requires_deferred_callback) {
      deferred_send_callback_ = std::move(response_error_handling);
    }
  });

  if (HasError()) {
    *response_error =
        TRITONSERVER_ErrorNew(Error()->Code(), Error()->Message().c_str());
    return;
  }

  bool cuda_copy = false;

  for (auto& output_tensor : OutputTensors()) {
    // FIXME: for decoupled models we will skip the requested output names.
    TRITONSERVER_MemoryType src_memory_type = output_tensor->MemoryType();
    int64_t src_memory_type_id = output_tensor->MemoryTypeId();

    TRITONSERVER_MemoryType actual_memory_type = src_memory_type;
    int64_t actual_memory_type_id = src_memory_type_id;

    if (actual_memory_type == TRITONSERVER_MEMORY_GPU) {
      requires_deferred_callback = true;
    }

    TRITONBACKEND_Output* response_output;
    SET_ERROR_AND_RETURN(
        response_error,
        TRITONBACKEND_ResponseOutput(
            response, &response_output, output_tensor->Name().c_str(),
            static_cast<TRITONSERVER_DataType>(output_tensor->TritonDtype()),
            output_tensor->Dims().data(), output_tensor->Dims().size()));

    void* buffer;
    SET_ERROR_AND_RETURN(
        response_error, TRITONBACKEND_OutputBuffer(
                            response_output, &buffer, output_tensor->ByteSize(),
                            &actual_memory_type, &actual_memory_type_id));

    bool cuda_used = false;
    TRITONSERVER_BufferAttributes* output_buffer_attributes;
    SET_ERROR_AND_RETURN(
        response_error, TRITONBACKEND_OutputBufferAttributes(
                            response_output, &output_buffer_attributes));

    std::unique_ptr<PbMemory> output_buffer;
    if (src_memory_type == TRITONSERVER_MEMORY_GPU &&
        actual_memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
      cudaIpcMemHandle_t* cuda_ipc_mem_handle_p;
      SET_ERROR_AND_RETURN(
          response_error,
          TRITONSERVER_BufferAttributesCudaIpcHandle(
              output_buffer_attributes,
              reinterpret_cast<void**>(&cuda_ipc_mem_handle_p)));

      if (cuda_ipc_mem_handle_p != nullptr) {
        SET_ERROR_AND_RETURN_IF_EXCEPTION(
            response_error,
            output_buffer = PbMemory::Create(
                shm_pool, actual_memory_type, actual_memory_type_id,
                output_tensor->ByteSize(), reinterpret_cast<char*>(buffer),
                false /* copy_gpu */));
        output_buffer->SetCudaIpcHandle(cuda_ipc_mem_handle_p);
      } else {
        SET_ERROR_AND_RETURN_IF_EXCEPTION(
            response_error,
            output_buffer = PbMemory::Create(
                shm_pool, actual_memory_type, actual_memory_type_id,
                output_tensor->ByteSize(), reinterpret_cast<char*>(buffer),
                true /* copy_gpu */));
      }
      gpu_buffer_helper.AddBuffer(output_buffer->ShmHandle());
      output_buffers.push_back({std::move(output_buffer), buffer});
#endif
    }

    // When we requested a GPU buffer but received a CPU buffer.
    if (src_memory_type == TRITONSERVER_MEMORY_GPU &&
        (actual_memory_type == TRITONSERVER_MEMORY_CPU ||
         actual_memory_type == TRITONSERVER_MEMORY_CPU_PINNED)) {
      SET_ERROR_AND_RETURN_IF_EXCEPTION(
          response_error,
          output_buffer = PbMemory::Create(
              shm_pool, actual_memory_type, actual_memory_type_id,
              output_tensor->ByteSize(), nullptr /* data ptr */));

      gpu_buffer_helper.AddBuffer(output_buffer->ShmHandle());
      output_buffers.push_back({std::move(output_buffer), buffer});
    }

    if (src_memory_type != TRITONSERVER_MEMORY_GPU) {
      SET_ERROR_AND_RETURN(
          response_error,
          CopyBuffer(
              "Failed to copy the output tensor to buffer.", src_memory_type,
              src_memory_type_id, actual_memory_type, actual_memory_type_id,
              output_tensor->ByteSize(), output_tensor->DataPtr(), buffer,
              reinterpret_cast<cudaStream_t>(cuda_stream), &cuda_used));
    }

    cuda_copy |= cuda_used;
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  }
#endif  // TRITON_ENABLE_GPU
}
#endif

#ifndef TRITON_PB_STUB
void
InferResponse::DeferredSendCallback()
{
  deferred_send_callback_.reset();
}
#endif

}}}  // namespace triton::backend::python
