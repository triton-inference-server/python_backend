// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <future>
#include "triton/core/tritonserver.h"

#include "pb_main_utils.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {
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
CreateTritonErrorFromException(const PythonBackendException& pb_exception)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // We reuse the request so we don't delete it here.
}

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  SharedMemory* shm_pool = reinterpret_cast<SharedMemory*>(userp);

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
  } else {
    switch (*actual_memory_type) {
      case TRITONSERVER_MEMORY_CPU:
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        off_t tensor_offset;
        try {
          shm_pool->Map((char**)buffer, byte_size, tensor_offset);
        }
        catch (const PythonBackendException& pb_exception) {
          return CreateTritonErrorFromException(pb_exception);
        }
        *buffer_userp = new off_t(tensor_offset);
      } break;

      case TRITONSERVER_MEMORY_GPU: {
        // throw PythonBackendException("Not supported!");
        // auto err = cudaSetDevice(*actual_memory_type_id);
        // if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
        //     (err != cudaErrorInsufficientDriver)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "GPU tensors are not supported in BLS");
      }

        // err = cudaMalloc(&allocated_ptr, byte_size);
        // if (err != cudaSuccess) {
        //   return TRITONSERVER_ErrorNew(
        //       TRITONSERVER_ERROR_INTERNAL,
        //       std::string(
        //           "cudaMalloc failed: " +
        //           std::string(cudaGetErrorString(err))) .c_str());
        // }
        // break;
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
  //   std::string* name = nullptr;
  //   if (buffer_userp != nullptr) {
  //     name = reinterpret_cast<std::string*>(buffer_userp);
  //   } else {
  //     name = new std::string("<unknown>");
  //   }

  //   std::cout << "Releasing buffer " << buffer << " of size " << byte_size
  //             << " in " << TRITONSERVER_MemoryTypeString(memory_type)
  //             << " for result '" << *name << "'" << std::endl;
  //   switch (memory_type) {
  //     case TRITONSERVER_MEMORY_CPU:
  //       free(buffer);
  //       break;
  // #ifdef TRITON_ENABLE_GPU
  //     case TRITONSERVER_MEMORY_CPU_PINNED: {
  //       auto err = cudaSetDevice(memory_type_id);
  //       if (err == cudaSuccess) {
  //         err = cudaFreeHost(buffer);
  //       }
  //       if (err != cudaSuccess) {
  //         std::cerr << "error: failed to cudaFree " << buffer << ": "
  //                   << cudaGetErrorString(err) << std::endl;
  //       }
  //       break;
  //     }
  //     case TRITONSERVER_MEMORY_GPU: {
  //       auto err = cudaSetDevice(memory_type_id);
  //       if (err == cudaSuccess) {
  //         err = cudaFree(buffer);
  //       }
  //       if (err != cudaSuccess) {
  //         std::cerr << "error: failed to cudaFree " << buffer << ": "
  //                   << cudaGetErrorString(err) << std::endl;
  //       }
  //       break;
  //     }
  // #endif  // TRITON_ENABLE_GPU
  //     default:
  //       std::cerr << "error: unexpected buffer allocated in CUDA managed
  //       memory"
  //                 << std::endl;
  //       break;
  //   }

  //   delete name;

  return nullptr;  // Success
}

std::unique_ptr<InferResponse>
ExecuteInferRequest(
    TRITONSERVER_Server* server,
    const std::unique_ptr<InferRequest>& infer_request,
    const std::unique_ptr<SharedMemory>& shm_pool)
{
  bool is_ready = false;
  const char* model_name = infer_request->ModelName().c_str();
  int64_t model_version = infer_request->ModelVersion();
  THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
      server, model_name, model_version, &is_ready));

  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */));

  // Inference
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestNew(
      &irequest, server, model_name, model_version));

  THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetId(
      irequest, infer_request->RequestId().c_str()));

  THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      irequest, InferRequestComplete, nullptr /* request_release_userp */));

  for (auto& infer_input : infer_request->Inputs()) {
    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddInput(
        irequest, infer_input->Name().c_str(),
        static_cast<TRITONSERVER_DataType>(infer_input->TritonDtype()),
        infer_input->Dims().data(), infer_input->Dims().size()));

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest, infer_input->Name().c_str(), infer_input->GetDataPtr(),
        infer_input->ByteSize(), infer_input->MemoryType(),
        infer_input->MemoryTypeId()));
  }

  for (auto& requested_output_name : infer_request->RequestedOutputNames()) {
    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        irequest, requested_output_name.c_str()));
  }

  std::unique_ptr<InferResponse> infer_response;

  {
    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest, allocator, shm_pool.get(), InferResponseComplete,
        reinterpret_cast<void*>(p)));

    THROW_IF_TRITON_ERROR(
        TRITONSERVER_ServerInferAsync(server, irequest, nullptr /* trace */));

    // Wait for the inference to complete.
    TRITONSERVER_InferenceResponse* response = completed.get();
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
      output_tensors.push_back(std::make_shared<PbTensor>(
          sname, dims_vector, datatype, memory_type, memory_type_id,
          const_cast<void*>(base), byte_size, nullptr /* DLManagedTensor */));
    }

    std::shared_ptr<PbError> pb_error;
    infer_response = std::make_unique<InferResponse>(output_tensors, pb_error);

    THROW_IF_TRITON_ERROR(TRITONSERVER_InferenceResponseDelete(response));
  }

  return infer_response;
}

}}};  // namespace triton::backend::python
