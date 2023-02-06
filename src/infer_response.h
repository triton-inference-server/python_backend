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

#pragma once

#include <future>
#include "pb_error.h"
#include "pb_tensor.h"
#include "pb_utils.h"
#include "scoped_defer.h"

namespace triton { namespace backend { namespace python {

struct ResponseShm {
  uint32_t outputs_size;
  bi::managed_external_buffer::handle_t error;
  bool has_error;
  // Indicates whether this error has a message or not.
  bool is_error_set;
};

#define SET_ERROR_AND_RETURN(E, X)           \
  do {                                       \
    TRITONSERVER_Error* raasnie_err__ = (X); \
    if (raasnie_err__ != nullptr) {          \
      *E = raasnie_err__;                    \
      return E;                              \
    }                                        \
  } while (false)

#define SET_ERROR_AND_RETURN_IF_EXCEPTION(E, X)                \
  do {                                                         \
    try {                                                      \
      (X);                                                     \
    }                                                          \
    catch (const PythonBackendException& pb_exception) {       \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew( \
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());   \
      *E = rarie_err__;                                        \
      return E;                                                \
    }                                                          \
  } while (false)

class InferResponse {
 public:
  InferResponse(
      const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError> error = nullptr);
  InferResponse(
      const std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::promise<std::unique_ptr<InferResponse>>* promise,
      std::shared_ptr<PbError> error = nullptr);
  std::vector<std::shared_ptr<PbTensor>>& OutputTensors();
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu = true);
  static std::unique_ptr<InferResponse> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t response_handle,
      bool open_cuda_handle);
  bool HasError();
  std::shared_ptr<PbError>& Error();
  bi::managed_external_buffer::handle_t ShmHandle();
  void PruneOutputTensors(const std::set<std::string>& requested_output_names);
  std::unique_ptr<std::future<std::unique_ptr<InferResponse>>>
  GetNextResponse();
  void SetNextResponseHandle(
      bi::managed_external_buffer::handle_t next_response_handle);
  bi::managed_external_buffer::handle_t NextResponseHandle();

#ifndef TRITON_PB_STUB
  /// Send an inference response. If the response has a GPU tensor, sending the
  /// response needs to be done in two step. The boolean
  /// 'requires_deferred_callback' indicates whether DeferredSendCallback method
  /// should be called or not.
  std::shared_ptr<TRITONSERVER_Error*> Send(
      TRITONBACKEND_ResponseFactory* response_factory, void* cuda_stream,
      bool& requires_deferred_callback, const uint32_t flags,
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>& output_buffers,
      const std::set<std::string>& requested_output_names = {},
      TRITONBACKEND_Response* response = nullptr);

  void DeferredSendCallback();
#endif

  // Disallow copying the inference response object.
  DISALLOW_COPY_AND_ASSIGN(InferResponse);

 private:
  InferResponse(
      AllocatedSharedMemory<char>& response_shm,
      std::vector<std::shared_ptr<PbTensor>>& output_tensors,
      std::shared_ptr<PbError>& pb_error);
  std::vector<std::shared_ptr<PbTensor>> output_tensors_;
  std::shared_ptr<PbError> error_;
  bi::managed_external_buffer::handle_t shm_handle_;
  AllocatedSharedMemory<char> response_shm_;
  std::vector<std::pair<std::unique_ptr<PbMemory>, void*>> gpu_output_buffers_;
  std::unique_ptr<ScopedDefer> deferred_send_callback_;

  std::unique_ptr<std::future<std::unique_ptr<InferResponse>>>
      next_response_future_;
  bi::managed_external_buffer::handle_t next_response_handle_;
};

#ifdef TRITON_PB_STUB
class ResponseGenerator {
 public:
  ResponseGenerator(
      const std::vector<std::shared_ptr<InferResponse>>& responses);

  std::shared_ptr<InferResponse> Next();
  int Length();
  std::vector<std::shared_ptr<InferResponse>>::iterator Begin();
  std::vector<std::shared_ptr<InferResponse>>::iterator End();

 private:
  std::vector<std::shared_ptr<InferResponse>> responses_;
  size_t index_;
};
#endif

}}}  // namespace triton::backend::python
