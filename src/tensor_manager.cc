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

#include "tensor_manager.h"

#include <algorithm>
#include <iostream>
#include <mutex>

namespace triton { namespace backend { namespace python {

void
TensorManager::InsertOpenedMemHandle(
    void* dev_ptr, cudaIpcMemHandle_t* cuda_handle_ptr)
{
  std::lock_guard<std::mutex> guard{mutex_};

  std::array<char, sizeof(cudaIpcMemHandle_t)> cuda_handle;
  char* lcuda_handle_ptr = reinterpret_cast<char*>(cuda_handle_ptr);
  std::copy(
      lcuda_handle_ptr, lcuda_handle_ptr + cuda_handle.size(),
      cuda_handle.begin());
  auto cuda_handles_itr =
      std::find(cuda_handles_.begin(), cuda_handles_.end(), cuda_handle);
  if (cuda_handles_itr != cuda_handles_.cend()) {
    return;
  }

  cuda_handles_.emplace_back(std::move(cuda_handle));
  device_ptrs_.push_back(dev_ptr);
}

void*
TensorManager::FindCudaIpcMemHandle(cudaIpcMemHandle_t* cuda_handle_ptr)
{
  std::lock_guard<std::mutex> guard{mutex_};
  std::array<char, sizeof(cudaIpcMemHandle_t)> cuda_handle;
  char* lcuda_handle_ptr = reinterpret_cast<char*>(cuda_handle_ptr);
  std::copy(
      lcuda_handle_ptr, lcuda_handle_ptr + cuda_handle.size(),
      cuda_handle.begin());
  auto cuda_handles_itr =
      std::find(cuda_handles_.begin(), cuda_handles_.end(), cuda_handle);
  if (cuda_handles_itr == cuda_handles_.cend()) {
    return nullptr;
  }
  size_t index = std::distance(cuda_handles_.begin(), cuda_handles_itr);
  return device_ptrs_[index];
}

void
TensorManager::Clear()
{
  std::lock_guard<std::mutex> guard{mutex_};
  cuda_handles_.clear();
  device_ptrs_.clear();
}

cudaIpcMemHandle_t*
TensorManager::FindDevicePointer(void* device_ptr)
{
  std::lock_guard<std::mutex> guard{mutex_};
  auto device_ptr_itr =
      std::find(device_ptrs_.begin(), device_ptrs_.end(), device_ptr);
  
  if (device_ptr_itr == device_ptrs_.cend()) {
    return nullptr;
  }

  size_t index = std::distance(device_ptrs_.begin(), device_ptr_itr);
  return reinterpret_cast<cudaIpcMemHandle_t*>(cuda_handles_[index].data());
}
}}}  // namespace triton::backend::python
