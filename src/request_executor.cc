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

#include "request_executor.h"

#include <future>
#include "pb_utils.h"
#include "scoped_defer.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

TRITONSERVER_Error*
CreateTritonErrorFromException(const PythonBackendException& pb_exception)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  std::unique_ptr<SharedMemoryManager> shm_pool(
      reinterpret_cast<SharedMemoryManager*>(userp));

  ScopedDefer _([&shm_pool] { shm_pool.release(); });
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
  } else {
    switch (*actual_memory_type) {
      case TRITONSERVER_MEMORY_CPU:
#ifndef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU:
#endif
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
        try {
          std::unique_ptr<PbMemory> pb_memory = PbMemory::Create(
              shm_pool, *actual_memory_type, *actual_memory_type_id, byte_size,
              nullptr /* data */, false /* copy_gpu */);
          *buffer = pb_memory->DataPtr();
          *buffer_userp = reinterpret_cast<void*>(pb_memory.get());
          pb_memory.release();
        }
        catch (const PythonBackendException& pb_exception) {
          TRITONSERVER_Error* err =
              CreateTritonErrorFromException(pb_exception);
          return err;
        }

      } break;
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to set current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaMalloc(buffer, byte_size);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }
#endif
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;  // Success
}

RequestExecutor::RequestExecutor(
    std::unique_ptr<SharedMemoryManager>& shm_pool, TRITONSERVER_Server* server)
    : server_(server), shm_pool_(shm_pool)
{
  TRITONSERVER_ResponseAllocator* allocator;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */));
  response_allocator_ = allocator;
}

std::unique_ptr<InferResponse>
RequestExecutor::Infer(
    const std::shared_ptr<InferRequest>& infer_request,
    TRITONSERVER_InferenceResponse** triton_response)
{
  std::unique_ptr<InferResponse> infer_response;
  bool is_ready = false;
  const char* model_name = infer_request->ModelName().c_str();
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  TRITONSERVER_InferenceResponse* response = nullptr;

  // This variable indicates whether the InferenceRequest should be deleted as a
  // part of the catch block or it will be automatically deleted using the
  // InferResponseComplete callback.
  bool delete_inference_request = true;

  try {
    int64_t model_version = infer_request->ModelVersion();
    THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
        server_, model_name, model_version, &is_ready));

    if (!is_ready) {
      throw PythonBackendException(
          (std::string("Failed for execute the inference request. Model '") +
           model_name + "' is not ready.")
              .c_str());
    }

    // Inference
    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestNew(
        &irequest, server_, model_name, model_version));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetId(
        irequest, infer_request->RequestId().c_str()));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetCorrelationId(
        irequest, infer_request->CorrelationId()));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetFlags(
        irequest, infer_request->Flags()));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete, nullptr /* request_release_userp */));

    for (auto& infer_input : infer_request->Inputs()) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddInput(
          irequest, infer_input->Name().c_str(),
          static_cast<TRITONSERVER_DataType>(infer_input->TritonDtype()),
          infer_input->Dims().data(), infer_input->Dims().size()));

      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
          irequest, infer_input->Name().c_str(), infer_input->DataPtr(),
          infer_input->ByteSize(), infer_input->MemoryType(),
          infer_input->MemoryTypeId()));
    }

    for (auto& requested_output_name : infer_request->RequestedOutputNames()) {
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, requested_output_name.c_str()));
    }

    {
      auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
      std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest, response_allocator_, shm_pool_.get(), InferResponseComplete,
          reinterpret_cast<void*>(p)));

      THROW_IF_TRITON_ERROR(TRITONSERVER_ServerInferAsync(
          server_, irequest, nullptr /* trace */));

      // Wait for the inference to complete.
      response = completed.get();
      *triton_response = response;
      delete_inference_request = false;
      THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseError(response));

      uint32_t output_count;
      THROW_IF_TRITON_ERROR(
          TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

      std::vector<std::shared_ptr<PbTensor>> output_tensors;
      for (uint32_t idx = 0; idx < output_count; ++idx) {
        const char* cname;
        TRITONSERVER_DataType datatype;
        const int64_t* shape;
        uint64_t dim_count;
        const void* base;
        size_t byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        void* userp;

        THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseOutput(
            response, idx, &cname, &datatype, &shape, &dim_count, &base,
            &byte_size, &memory_type, &memory_type_id, &userp));
        std::string sname = cname;
        std::vector<int64_t> dims_vector{shape, shape + dim_count};

        // userp is only set for the CPU tensors
        if (memory_type != TRITONSERVER_MEMORY_GPU) {
          if (byte_size != 0) {
            std::shared_ptr<PbTensor> pb_tensor = std::make_shared<PbTensor>(
                sname, dims_vector, datatype, memory_type, memory_type_id,
                const_cast<void*>(base), byte_size,
                nullptr /* DLManagedTensor */);

            // Load the data so that it is deallocated automatically.
            std::unique_ptr<PbMemory> pb_memory(
                reinterpret_cast<PbMemory*>(userp));
            pb_tensor->SetMemory(std::move(pb_memory));
            output_tensors.push_back(pb_tensor);
          } else {
            output_tensors.push_back(std::make_shared<PbTensor>(
                sname, dims_vector, datatype, memory_type, memory_type_id,
                const_cast<void*>(base), byte_size,
                nullptr /* DLManagedTensor */));
          }
        } else {
          output_tensors.push_back(std::make_shared<PbTensor>(
              sname, dims_vector, datatype, memory_type, memory_type_id,
              const_cast<void*>(base), byte_size,
              nullptr /* DLManagedTensor */));
        }
      }

      std::shared_ptr<PbError> pb_error;
      infer_response =
          std::make_unique<InferResponse>(output_tensors, pb_error);
    }
  }
  catch (const PythonBackendException& pb_exception) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceResponseDelete(response),
          "Failed to delete inference resposne.");

      *triton_response = nullptr;
    }

    if (delete_inference_request) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "Failed to delete inference request.");
    }

    std::shared_ptr<PbError> pb_error =
        std::make_shared<PbError>(pb_exception.what());
    infer_response = std::make_unique<InferResponse>(
        std::vector<std::shared_ptr<PbTensor>>{}, pb_error);
  }

  return infer_response;
}

RequestExecutor::~RequestExecutor()
{
  if (response_allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(response_allocator_),
        "Failed to delete allocator.");
  }
}
}}};  // namespace triton::backend::python
