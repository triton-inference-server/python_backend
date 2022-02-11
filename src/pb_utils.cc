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

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace triton { namespace backend { namespace python {

#ifdef TRITON_ENABLE_GPU

CUDADriverAPI::CUDADriverAPI()
{
  dl_open_handle_ = dlopen("libcuda.so", RTLD_LAZY);

  // If libcuda.so is succesfully opened, it must be able to find
  // "cuPointerGetAttribute" and "cuGetErrorString" symbols.
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
  }
}

void
CUDADriverAPI::PointerGetAttribute(
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
CUDADriverAPI::IsAvailable()
{
  return dl_open_handle_ != nullptr;
}

CUDADriverAPI::~CUDADriverAPI() noexcept(false)
{
  if (dl_open_handle_ != nullptr) {
    int status = dlclose(dl_open_handle_);
    if (status != 0) {
      throw PythonBackendException("Failed to close the libcuda handle.");
    }
  }
}
#endif

}}}  // namespace triton::backend::python
