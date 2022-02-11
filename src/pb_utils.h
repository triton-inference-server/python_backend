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

#pragma once

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#endif  // TRITON_ENABLE_GPU
#include <pthread.h>
#include <climits>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

#define STUB_SET_RESPONSE_ERROR_IF_ERROR(SHM_POOL, RESPONSE, R, X) \
  do {                                                             \
    try {                                                          \
      (X);                                                         \
    }                                                              \
    catch (cont PythonBackendException & pb_exception) {           \
      off_t string_offset__;                                       \
      try {                                                        \
        SaveStringToSharedMemory(                                  \
            SHM_POOL, string_offset__, pb_exception.what());       \
        RESPONSE->has_error = true;                                \
        RESPONSE->error = string_offset__;                         \
        if (R)                                                     \
          return;                                                  \
      }                                                            \
      catch (cont PythonBackendException & pb2_exception) {        \
        printf(TRITONSERVER_ErrorMessage(pb_exception.what()));    \
        printf(                                                    \
            TRITONSERVER_LOG_ERROR,                                \
            TRITONSERVER_ErrorMessage(pb2_exception.what()));      \
      }                                                            \
    }                                                              \
    while (false)

#define THROW_IF_TRITON_ERROR(X)                                          \
  do {                                                                    \
    TRITONSERVER_Error* tie_err__ = (X);                                  \
    if (tie_err__ != nullptr) {                                           \
      throw PythonBackendException(TRITONSERVER_ErrorMessage(tie_err__)); \
    }                                                                     \
  } while (false)

#define THROW_IF_CUDA_ERROR(X)                          \
  do {                                                  \
    cudaError_t cuda_err__ = (X);                       \
    if (cuda_err__ != cudaSuccess) {                    \
      throw PythonBackendException(                     \
          std::string(cudaGetErrorString(cuda_err__))); \
    }                                                   \
  } while (false)

#define THROW_IF_ERROR(MSG, X)           \
  do {                                   \
    int return__ = (X);                  \
    if (return__ != 0) {                 \
      throw PythonBackendException(MSG); \
    }                                    \
  } while (false)


#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete;
#define DISALLOW_ASSIGN(TypeName) void operator=(const TypeName&) = delete;
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName)                  \
  DISALLOW_ASSIGN(TypeName)

struct InitializeResponseShm {
  // Indicates whether the response has an error or not.
  bool response_has_error;
  // Indicates whether the response error is set or not.
  bool response_is_error_set;
  // Contains the error message.
  bi::managed_external_buffer::handle_t response_error;
};

// Control data structure for the communication between the Python stub and the
// main stub.
struct IPCControlShm {
  bool stub_health;
  bool parent_health;
  bool uses_env;
  bi::interprocess_mutex parent_health_mutex;
  bi::interprocess_mutex stub_health_mutex;
  bi::managed_external_buffer::handle_t stub_message_queue;
  bi::managed_external_buffer::handle_t parent_message_queue;
};

struct ResponseShm {
  // Offset for Tensor output.
  off_t outputs;
  uint32_t outputs_size;
  off_t error;
  bool has_error;
  // Indicates whether this error has a message or not.
  bool is_error_set;
};

struct ResponseBatch {
  // Offset for response object.
  off_t responses;
  uint32_t batch_size;
  off_t error;
  bool has_error;
  // Indicates whether an additional call to stub is required for the clean up
  // of the resources.
  bool cleanup;
  // Indicates whether this error has a message or not.
  bool is_error_set;
};

struct RequestBatch {
  // Offset for request object.
  off_t requests;
  uint32_t batch_size;
};

//
// PythonBackendException
//
// Exception thrown if error occurs in PythonBackend.
//
struct PythonBackendException : std::exception {
  PythonBackendException(std::string message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

#ifdef TRITON_ENABLE_GPU
class CUDADriverAPI {
 public:
  static CUDADriverAPI& getInstance()
  {
    static CUDADriverAPI instance;
    return instance;
  }

 private:
  void* dl_open_handle_ = nullptr;
  CUresult (*cu_pointer_get_attribute_fn_)(
      CUdeviceptr*, CUpointer_attribute, CUdeviceptr) = nullptr;
  CUresult (*cu_get_error_string_fn_)(CUresult, const char**) = nullptr;
  CUDADriverAPI();
  ~CUDADriverAPI() noexcept(false);

 public:
  CUDADriverAPI(CUDADriverAPI const&) = delete;
  void operator=(CUDADriverAPI const&) = delete;
  bool IsAvailable();
  void PointerGetAttribute(
      CUdeviceptr* start_address, CUpointer_attribute attr,
      CUdeviceptr device_ptr);
};
#endif  // TRITON_ENABLE_GPU

}}}  // namespace triton::backend::python
