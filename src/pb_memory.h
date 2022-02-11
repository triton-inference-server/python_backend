// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_utils.h"
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend { namespace python {

//
// Represents a memory object in shared memory.
//
struct MemoryShm {
  // The shared memory offset of the data. For device pointers this will contain
  // the offset to the cudaMemHandle_t object.
  bi::managed_external_buffer::handle_t memory_ptr;

  // If the memory type is a GPU pointer, the offset of the GPU pointer from the
  // base address. For CPU memory type this field contains garbage data.
  uint64_t gpu_pointer_offset;

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  uint64_t byte_size;
};

class PbMemory {
 public:
  static std::unique_ptr<PbMemory> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      uint64_t byte_size, char* data);
  static std::unique_ptr<PbMemory> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t memory_offset);
  static std::unique_ptr<PbMemory> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      std::unique_ptr<BackendMemory>& backend_memory);

  bi::managed_external_buffer::handle_t ShmOffset();
  void Release();

 private:
  AllocatedSharedMemory<MemoryShm> memory_shm_;
  MemoryShm* memory_shm_ptr_;

  AllocatedSharedMemory<char> memory_data_shm_;
  char* memory_data_shm_ptr_;

  // Refers to the pointer that can hold the data. For CPU pointers this will be
  // the same as memory_data_shm_ptr_.
  char* data_ptr_;

  bi::managed_external_buffer::handle_t memory_shm_handle_;
  bool opened_cuda_ipc_handle_;

  /// Calculate the pointer offest from the base address.
  /// \return The offset of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  uint64_t GetGPUPointerOffset();

  /// Get the GPU start address.
  /// \return The start address of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  void* GetGPUStartAddress();

  PbMemory(
      AllocatedSharedMemory<MemoryShm>& memory_shm,
      AllocatedSharedMemory<char>& memory_data_shm, char* data,
      bool opened_cuda_ipc_handle);
};
}}}  // namespace triton::backend::python
