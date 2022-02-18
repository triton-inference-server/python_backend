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

#include "pb_memory.h"

namespace triton { namespace backend { namespace python {

std::unique_ptr<PbMemory>
PbMemory::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    uint64_t byte_size, char* data)
{
  AllocatedSharedMemory<MemoryShm> memory_shm =
      shm_pool->Construct<MemoryShm>();

  AllocatedSharedMemory<char> memory_data_shm;
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    memory_data_shm = shm_pool->Construct<char>(sizeof(cudaIpcMemHandle_t));
    if (data != nullptr) {
      THROW_IF_CUDA_ERROR(cudaSetDevice(memory_type_id));
      THROW_IF_CUDA_ERROR(cudaIpcGetMemHandle(
          reinterpret_cast<cudaIpcMemHandle_t*>(memory_data_shm.data_.get()),
          data));
    }
  }
#endif
  else {
    memory_data_shm = shm_pool->Construct<char>(byte_size);
    if (data != nullptr) {
      std::copy(data, data + byte_size, memory_data_shm.data_.get());
    }
    data = memory_data_shm.data_.get();
  }

  memory_shm.data_->byte_size = byte_size;
  memory_shm.data_->memory_ptr = memory_data_shm.handle_;
  memory_shm.data_->memory_type_id = memory_type_id;
  memory_shm.data_->memory_type = memory_type;

  std::unique_ptr<PbMemory> pb_memory(new PbMemory(
      memory_shm, memory_data_shm, data, false /* opened_cuda_ipc_handle */));

#ifdef TRITON_ENABLE_GPU
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    pb_memory->memory_shm_.data_->gpu_pointer_offset =
        pb_memory->GetGPUPointerOffset();
  }
#endif

  return pb_memory;
}

std::unique_ptr<PbMemory>
PbMemory::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<MemoryShm> memory_shm =
      shm_pool->Load<MemoryShm>(handle);

  AllocatedSharedMemory<char> memory_data_shm =
      shm_pool->Load<char>(memory_shm.data_->memory_ptr);

  char* data_ptr;
  bool opened_cuda_ipc_handle = false;
  if (memory_shm.data_->memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaIpcMemHandle_t* cuda_handle =
        reinterpret_cast<cudaIpcMemHandle_t*>(memory_data_shm.data_.get());

    // The pointer opened by the cudaIpcOpenMemHandle will refer to the base
    // address. We need to manually correct the offset.
    void* data_ptr_base;
    THROW_IF_CUDA_ERROR(cudaSetDevice(memory_shm.data_->memory_type_id));
    THROW_IF_CUDA_ERROR(cudaIpcOpenMemHandle(
        &data_ptr_base, *cuda_handle, cudaIpcMemLazyEnablePeerAccess));

    data_ptr =
        (reinterpret_cast<char*>(data_ptr_base) +
         memory_shm.data_->gpu_pointer_offset);
    opened_cuda_ipc_handle = true;
#endif
  }

  return std::unique_ptr<PbMemory>(new PbMemory(
      memory_shm, memory_data_shm, data_ptr,
      opened_cuda_ipc_handle /* opened_cuda_ipc_handle */));
}

PbMemory::PbMemory(
    AllocatedSharedMemory<MemoryShm>& memory_shm,
    AllocatedSharedMemory<char>& memory_data_shm, char* data,
    bool opened_cuda_ipc_handle)
    : memory_shm_(std::move(memory_shm)),
      memory_data_shm_(std::move(memory_data_shm)), data_ptr_(data),
      opened_cuda_ipc_handle_(opened_cuda_ipc_handle)
{
  memory_shm_ptr_ = memory_shm_.data_.get();
  memory_data_shm_ptr_ = memory_data_shm_.data_.get();
  memory_shm_handle_ = memory_shm_.handle_;
}

bi::managed_external_buffer::handle_t
PbMemory::ShmOffset()
{
  return memory_shm_handle_;
}

void
PbMemory::Release()
{
  memory_shm_.data_.release();
  memory_data_shm_.data_.release();
}

void*
PbMemory::GetGPUStartAddress()
{
  if (memory_shm_ptr_->memory_type == TRITONSERVER_MEMORY_GPU) {
    CUDADriverAPI& driver_api = CUDADriverAPI::getInstance();
    CUdeviceptr start_address;

    driver_api.PointerGetAttribute(
        &start_address, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        reinterpret_cast<CUdeviceptr>(data_ptr_));

    return reinterpret_cast<void*>(start_address);
  }

  throw PythonBackendException(
      "Calling GetGPUStartAddress function on CPU memory.");
}

uint64_t
PbMemory::GetGPUPointerOffset()
{
  uint64_t offset;
  if (memory_shm_ptr_->memory_type == TRITONSERVER_MEMORY_GPU) {
    offset = data_ptr_ - reinterpret_cast<char*>(GetGPUStartAddress());
  } else {
    throw PythonBackendException(
        "Calling GetGPUPointerOffset function on CPU tensor.");
  }
  return offset;
}

TRITONSERVER_MemoryType
PbMemory::MemoryType() const
{
  return memory_shm_ptr_->memory_type;
}

int64_t
PbMemory::MemoryTypeId() const
{
  return memory_shm_ptr_->memory_type_id;
}

uint64_t
PbMemory::ByteSize() const
{
  return memory_shm_ptr_->byte_size;
}

char*
PbMemory::DataPtr() const
{
  return data_ptr_;
}

}}}  // namespace triton::backend::python
