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

#include <dlpack/dlpack.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace py = pybind11;

namespace triton { namespace backend { namespace python {

typedef enum PYTHONBACKEND_tensortype_enum {
  PYTHONBACKEND_RAW,
  PYTHONBACKEND_NUMPY,
  PYTHONBACKEND_DLPACK
} PYTHONBACKEND_TensorType;

// PbTensor class is the representation of Triton tensors
// inside Python backend.
class PbTensor {
  std::string name_;
  py::array numpy_array_;
  int dtype_;
  void* memory_ptr_;
  int64_t memory_type_id_;
  std::vector<int64_t> dims_;
  TRITONSERVER_MemoryType memory_type_;
  PYTHONBACKEND_TensorType tensor_type_;
  uint64_t byte_size_;
  DLManagedTensor* dl_managed_tensor_;

 public:
  /// Create a PbTensor using a numpy array
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  PbTensor(std::string name, py::object numpy_array);

  /// Create a PbTensor using a numpy array. This constructor is used for types
  /// that are not natively available in C++ such as float16. This constructor
  /// will fix the type of the NumPy array to match the Triton dtype.
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  /// \param dtype The triton dtype
  PbTensor(std::string name, py::object numpy_array, int dtype);

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
  PbTensor(
      std::string name, std::vector<int64_t> dims, int dtype,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      void* memory_ptr, uint64_t byte_size,
      DLManagedTensor* dl_managed_tensor = nullptr);

  py::capsule ToDLPack();
  static std::unique_ptr<PbTensor> FromDLPack(
      std::string name, py::capsule dlpack);
  const std::string& Name();
  py::array& AsNumpy();
  int TritonDtype();

  void DeleteDLPack();
  PYTHONBACKEND_TensorType TensorType();

  /// Tells whether the Tensor is stored in CPU or not.
  /// \return A boolean value indicating whether the tensor is stored in CPU
  /// or not.
  bool IsCPU();

  /// Get the total byte size of the tensor.
  uint64_t ByteSize();

  /// Get the triton memory type of the Tensor.
  /// \return the memory type of the tensor.
  TRITONSERVER_MemoryType MemoryType();

  /// Get the dimensions of the tensor
  /// \return A vector containing the tensor dimensions.
  const std::vector<int64_t>& Dims();

  /// Get the data pointer.
  /// \return The location to the memory where the data is stored.
  void* GetDataPtr();

  /// Get the memory type id.
  /// \return The memory type id of the tensor.
  int64_t MemoryTypeId();
};
}}}  // namespace triton::backend::python
