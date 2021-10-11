// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
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
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#include <dlpack/dlpack.h>

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include <string>
#include "pb_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

typedef enum PYTHONBACKEND_tensortype_enum {
  PYTHONBACKEND_RAW,
  PYTHONBACKEND_NUMPY,
  PYTHONBACKEND_DLPACK
} PYTHONBACKEND_TensorType;

// PbTensor class is the representation of Triton tensors
// inside Python backend.
class PbTensor {
 private:
  std::string name_;
#ifdef TRITON_PB_STUB
  py::array numpy_array_;
  // Storing the serialized version of the numpy array
  py::array numpy_array_serialized_;
#endif
  TRITONSERVER_DataType dtype_;
  void* memory_ptr_;
  int64_t memory_type_id_;
  std::vector<int64_t> dims_;
  TRITONSERVER_MemoryType memory_type_;
  PYTHONBACKEND_TensorType tensor_type_;
  uint64_t byte_size_;
  DLManagedTensor* dl_managed_tensor_;
  Tensor* tensor_shm_;
  RawData* raw_data_shm_;

#ifdef TRITON_ENABLE_GPU
  bool is_cuda_handle_set_;
  cudaIpcMemHandle_t* cuda_ipc_mem_handle_ = nullptr;
#ifndef TRITON_PB_STUB
  std::unique_ptr<BackendMemory> backend_memory_;
#endif  // TRITON_PB_STUB
#endif  // TRITON_ENABLE_GPU
  // bool is_reused_ = false;
  uint64_t device_ptr_offset_ = 0;
  bool destruct_cuda_ipc_mem_handle_ = false;
  off_t raw_shm_offset_ = 0;
  off_t shm_offset_ = 0;

 public:
#ifdef TRITON_PB_STUB
  /// Create a PbTensor using a numpy array
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  PbTensor(const std::string& name, py::object numpy_array);

  /// Create a PbTensor using a numpy array. This constructor is used for types
  /// that are not natively available in C++ such as float16. This constructor
  /// will fix the type of the NumPy array to match the Triton dtype.
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  /// \param dtype The triton dtype
  PbTensor(
      const std::string& name, py::object numpy_array,
      TRITONSERVER_DataType dtype);
#endif

  /// Create a PbTensor from raw pointer. This constructor is used for
  /// interfacing with DLPack tensors.
  /// \param name The name of the tensor
  /// \param dims Tensor dimensions
  /// \param dtype Triton dtype
  /// \param memory_type The memory type of the tensor
  /// \param memory_type_id The memory type_id of the tensor
  /// \param memory_ptr Pointer to the location of the data. Data must be
  /// contiguous and in C order.
  /// \param byte_size Total number of bytes that the tensor uses.
  /// \param shm_offset The shared memory offset of the device pointer.
  PbTensor(
      const std::string& name, const std::vector<int64_t>& dims,
      TRITONSERVER_DataType dtype, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, void* memory_ptr, uint64_t byte_size,
      DLManagedTensor* dl_managed_tensor = nullptr, off_t shm_offset = 0);

  // Copying tensor objects is not allowed.
  PbTensor(const PbTensor& other) = delete;
  PbTensor& operator=(const PbTensor& other) = delete;

#ifdef TRITON_PB_STUB
  /// Construct a Python backend tensor using a DLPack
  /// capsule.
  /// \param dlpack source dlpack tensor
  /// \param name name of the tensor
  static std::shared_ptr<PbTensor> FromDLPack(
      const std::string& name, const py::capsule& dlpack);

  /// Construct a Python backend tensor using a NumPy object.
  /// \param numpy_array Numpy array
  /// \param name name of the tensor
  static std::shared_ptr<PbTensor> FromNumpy(
      const std::string& name, py::object numpy_array);

  /// Get a PyCapsule object containing the DLPack representation of the tensor.
  /// \return Capsule object containing pointer to a DLPack object.
  py::capsule ToDLPack();
#endif

  /// Get the name of the tensor
  /// \return name of the tensor.
  const std::string& Name() const;
  static std::shared_ptr<PbTensor> LoadFromSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, off_t tensor_offset);
#ifdef TRITON_ENABLE_GPU

  /// Get the GPU start address.
  /// \return The start address of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  void* GetGPUStartAddress();

  /// Get the cuda IPC handle corresponding to this tensor.
  /// \return The cudaIpcMemHandle
  cudaIpcMemHandle_t* CudaIpcMemHandle();

  /// Set the cuda IPC handle corresponding to this tensor.
  /// \param cuda_ipc_mem_handle CUDA ipc mem handle.
  void SetCudaIpcMemHandle(cudaIpcMemHandle_t* cuda_ipc_mem_handle)
  {
    cuda_ipc_mem_handle_ = cuda_ipc_mem_handle;
  }

  /// Get the GPU pointer offset.
  /// \return The offset of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  uint64_t GetGPUPointerOffset();
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_PB_STUB
  /// Get NumPy representation of the tensor.
  /// \throw If the tensor is stored in GPU, an exception is thrown
  /// \return NumPy representation of the Tensor
  const py::array& AsNumpy() const;
#endif

#ifndef TRITON_PB_STUB
  /// Set the backend memory object that holds the data for GPU tensors.
  /// \param backend_memory Backend memory.
  void SetBackendMemory(
      std::unique_ptr<BackendMemory> backend_memory,
      std::unique_ptr<SharedMemory>& shm_pool);
#endif

  /// Save tensor inside shared memory.
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemory>& shm_pool, Tensor* tensor_shm,
      bool copy_cpu, bool copy_gpu);

  /// Get the triton dtype
  /// \return Triton dtype
  int TritonDtype() const;

  /// This function will be automatically called by the stub when the tensor is
  /// no longer required.
  void DeleteDLPack();

  /// Shared memory offset of the raw pointer.
  off_t RawShmOffset();

  /// Shared memory offset of the tensor.
  off_t ShmOffset() {
    return shm_offset_;
  }

  /// Get the type of the tensor
  /// \return Type of the tensor.
  PYTHONBACKEND_TensorType TensorType() const;

  /// Tells whether the Tensor is stored in CPU or not.
  /// \return A boolean value indicating whether the tensor is stored in CPU
  /// or not.
  bool IsCPU() const;

  /// Get the total byte size of the tensor.
  uint64_t ByteSize() const;

  /// Get the triton memory type of the Tensor.
  /// \return the memory type of the tensor.
  TRITONSERVER_MemoryType MemoryType() const;

  /// Get the dimensions of the tensor
  /// \return A vector containing the tensor dimensions.
  const std::vector<int64_t>& Dims() const;

  /// Get the data pointer.
  /// \return The location to the memory where the data is stored.
  void* GetDataPtr() const;

  /// Set the underlying pointer to use. This must be only used when the tensor
  /// is being reused.
  void SetDataPtr(void* ptr);

  /// After the GPU tensor buffer is provided, copy the data to the output
  /// buffers.
  void LoadGPUData(std::unique_ptr<SharedMemory>& shm_pool, std::mutex& gpu_load_mutex);
  void CopyToCPU(std::unique_ptr<SharedMemory>& shm_pool);

  Tensor* SharedMemoryObject() { return tensor_shm_; }


  RawData* RawDataShm() { return raw_data_shm_; }

  /// Get the memory type id.
  /// \return The memory type id of the tensor.
  int64_t MemoryTypeId() const;

  PbTensor();

  /// Destructor
  ~PbTensor() noexcept(false);
};
}}}  // namespace triton::backend::python
