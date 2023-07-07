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

#include "pb_utils.h"

#include <archive.h>
#include <archive_entry.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "scoped_defer.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace triton { namespace backend { namespace python {

#ifdef TRITON_ENABLE_GPU

CUDAHandler::CUDAHandler()
{
  dl_open_handle_ = dlopen("libcuda.so", RTLD_LAZY);

  // If libcuda.so is successfully opened, it must be able to find
  // "cuPointerGetAttribute", "cuGetErrorString", and
  // "cuDevicePrimaryCtxGetState" symbols.
  if (dl_open_handle_ != nullptr) {
    void* cu_pointer_get_attribute_fn =
        dlsym(dl_open_handle_, "cuPointerGetAttribute");
    if (cu_pointer_get_attribute_fn == nullptr) {
      throw PythonBackendException(
          std::string("Failed to dlsym 'cuPointerGetAttribute'. Error: ") +
          dlerror());
    }
    *((void**)&cu_pointer_get_attribute_fn_) = cu_pointer_get_attribute_fn;

    void* cu_get_error_string_fn = dlsym(dl_open_handle_, "cuGetErrorString");
    if (cu_get_error_string_fn == nullptr) {
      throw PythonBackendException(
          std::string("Failed to dlsym 'cuGetErrorString'. Error: ") +
          dlerror());
    }
    *((void**)&cu_get_error_string_fn_) = cu_get_error_string_fn;

    void* cu_init_fn = dlsym(dl_open_handle_, "cuInit");
    if (cu_init_fn == nullptr) {
      throw PythonBackendException(
          std::string("Failed to dlsym 'cuInit'. Error: ") + dlerror());
    }
    *((void**)&cu_init_fn_) = cu_init_fn;

    void* cu_device_primary_ctx_get_state_fn =
        dlsym(dl_open_handle_, "cuDevicePrimaryCtxGetState");
    if (cu_device_primary_ctx_get_state_fn == nullptr) {
      throw PythonBackendException(
          std::string("Failed to dlsym 'cuDevicePrimaryCtxGetState'. Error: ") +
          dlerror());
    }
    *((void**)&cu_device_primary_ctx_get_state_fn_) =
        cu_device_primary_ctx_get_state_fn;

    // Initialize the driver API.
    CUresult cuda_err = (*cu_init_fn_)(0 /* flags */);
    if (cuda_err != CUDA_SUCCESS) {
      const char* error_string;
      (*cu_get_error_string_fn_)(cuda_err, &error_string);
      error_str_ = std::string("failed to call cuInit: ") + error_string;
      int status = dlclose(dl_open_handle_);
      if (status != 0) {
        throw PythonBackendException("Failed to close the libcuda handle.");
      }
      dl_open_handle_ = nullptr;
    }
  }
}

void
CUDAHandler::PointerGetAttribute(
    CUdeviceptr* start_address, CUpointer_attribute attribute,
    CUdeviceptr dev_ptr)
{
  CUresult cuda_err =
      (*cu_pointer_get_attribute_fn_)(start_address, attribute, dev_ptr);
  if (cuda_err != CUDA_SUCCESS) {
    const char* error_string;
    (*cu_get_error_string_fn_)(cuda_err, &error_string);
    throw PythonBackendException(
        std::string(
            "failed to get cuda pointer device attribute: " +
            std::string(error_string))
            .c_str());
  }
}

bool
CUDAHandler::IsAvailable()
{
  return dl_open_handle_ != nullptr;
}

void
CUDAHandler::OpenCudaHandle(
    int64_t memory_type_id, cudaIpcMemHandle_t* cuda_mem_handle,
    void** data_ptr)
{
  std::lock_guard<std::mutex> guard{mu_};
  ScopedSetDevice scoped_set_device(memory_type_id);

  cudaError_t err = cudaIpcOpenMemHandle(
      data_ptr, *cuda_mem_handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    throw PythonBackendException(
        std::string("Failed to open the cudaIpcHandle. error: ") +
        cudaGetErrorString(err));
  }
}

void
CUDAHandler::CloseCudaHandle(int64_t memory_type_id, void* data_ptr)
{
  std::lock_guard<std::mutex> guard{mu_};
  int current_device;

  // Save the previous device
  cudaError_t err = cudaGetDevice(&current_device);
  if (err != cudaSuccess) {
    throw PythonBackendException(
        std::string("Failed to get the current CUDA device. error: ") +
        cudaGetErrorString(err));
  }

  // Restore the previous device before returning from the function.
  ScopedSetDevice scoped_set_device(memory_type_id);
  err = cudaIpcCloseMemHandle(data_ptr);
  if (err != cudaSuccess) {
    throw PythonBackendException(
        std::string("Failed to close the cudaIpcHandle. error: ") +
        cudaGetErrorString(err));
  }
}

bool
CUDAHandler::HasPrimaryContext(int device)
{
  unsigned int ctx_flags;
  int ctx_is_active = 0;
  CUresult cuda_err = (*cu_device_primary_ctx_get_state_fn_)(
      device, &ctx_flags, &ctx_is_active);
  if (cuda_err != CUDA_SUCCESS) {
    const char* error_string;
    (*cu_get_error_string_fn_)(cuda_err, &error_string);
    throw PythonBackendException(
        std::string(
            "failed to get primary context state: " + std::string(error_string))
            .c_str());
  }

  return ctx_is_active == 1;
}

void
CUDAHandler::MaybeSetDevice(int device)
{
  if (HasPrimaryContext(device)) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      throw PythonBackendException(
          std::string("Failed to set the CUDA device to ") +
          std::to_string(device) + ". error: " + cudaGetErrorString(err));
    }
  }
}


CUDAHandler::~CUDAHandler() noexcept(false)
{
  if (dl_open_handle_ != nullptr) {
    int status = dlclose(dl_open_handle_);
    if (status != 0) {
      throw PythonBackendException("Failed to close the libcuda handle.");
    }
  }
}

ScopedSetDevice::ScopedSetDevice(int device)
{
  device_ = device;
  THROW_IF_CUDA_ERROR(cudaGetDevice(&current_device_));

  if (current_device_ != device_) {
    THROW_IF_CUDA_ERROR(cudaSetDevice(device_));
  }
}

ScopedSetDevice::~ScopedSetDevice()
{
  if (current_device_ != device_) {
    CUDAHandler& cuda_handler = CUDAHandler::getInstance();
    cuda_handler.MaybeSetDevice(current_device_);
  }
}
#endif

#ifndef TRITON_PB_STUB
std::shared_ptr<TRITONSERVER_Error*>
WrapTritonErrorInSharedPtr(TRITONSERVER_Error* error)
{
  std::shared_ptr<TRITONSERVER_Error*> response_error(
      new TRITONSERVER_Error*, [](TRITONSERVER_Error** error) {
        if (error != nullptr && *error != nullptr) {
          TRITONSERVER_ErrorDelete(*error);
        }

        if (error != nullptr) {
          delete error;
        }
      });
  *response_error = error;
  return response_error;
}
#endif
}}}  // namespace triton::backend::python
