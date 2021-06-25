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

#include "pb_stub_utils.h"
#include "pb_tensor.h"
#include "pb_utils.h"

namespace py = pybind11;

namespace triton { namespace backend { namespace python {
PbTensor::PbTensor(std::string name, py::object numpy_array)
    : name_(name), numpy_array_(numpy_array)
{
  dtype_ = numpy_to_triton_type(numpy_array.attr("dtype"));
  tensor_type_ = PYTHONBACKEND_NUMPY;
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;
}

PbTensor::PbTensor(std::string name, py::object numpy_array, int dtype)
    : name_(name)
{
  if (numpy_to_triton_type(numpy_array.attr("dtype")) != dtype) {
    numpy_array = numpy_array.attr("view")(triton_to_numpy_type(dtype));
  }
  numpy_array_ = numpy_array;
  tensor_type_ = PYTHONBACKEND_NUMPY;
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;
}

PbTensor::PbTensor(
    std::string name, std::vector<int64_t> dims, int dtype,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    void* memory_ptr, uint64_t byte_size, DLManagedTensor* dl_managed_tensor)
{
  name_ = name;
  memory_ptr_ = memory_ptr;
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  dtype_ = dtype;
  dims_ = dims;
  numpy_array_ = py::none();
  byte_size_ = byte_size;
  dl_managed_tensor_ = dl_managed_tensor;

  if (dl_managed_tensor != nullptr) {
    tensor_type_ = PYTHONBACKEND_DLPACK;
  } else {
    tensor_type_ = PYTHONBACKEND_RAW;
  }
}

bool
PbTensor::IsCPU()
{
  if (tensor_type_ == PYTHONBACKEND_NUMPY ||
      ((tensor_type_ == PYTHONBACKEND_RAW ||
        tensor_type_ == PYTHONBACKEND_DLPACK) &&
       memory_type_ == TRITONSERVER_MEMORY_CPU)) {
    return true;
  } else {
    return false;
  }
}

TRITONSERVER_MemoryType
PbTensor::MemoryType()
{
  return memory_type_;
}

int64_t
PbTensor::MemoryTypeId()
{
  return memory_type_id_;
}

uint64_t
PbTensor::ByteSize()
{
  return byte_size_;
}

const std::vector<int64_t>&
PbTensor::Dims()
{
  return dims_;
}

PYTHONBACKEND_TensorType
PbTensor::TensorType()
{
  return tensor_type_;
}

void
delete_unused_dltensor(PyObject* dlp)
{
  if (PyCapsule_IsValid(dlp, "dltensor")) {
    DLManagedTensor* dl_managed_tensor =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(dlp, "dltensor"));
    free(dl_managed_tensor);
  }
}

py::capsule
PbTensor::ToDLPack()
{
  DLManagedTensor* dlpack_tensor = new DLManagedTensor;
  dlpack_tensor->dl_tensor.ndim = dims_.size();
  dlpack_tensor->dl_tensor.byte_offset = 0;
  dlpack_tensor->dl_tensor.data = const_cast<void*>(memory_ptr_);
  dlpack_tensor->dl_tensor.shape = dims_.data();
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->deleter = [](DLManagedTensor* m) {};
  dlpack_tensor->dl_tensor.device.device_id = memory_type_id_;
  dlpack_tensor->dl_tensor.dtype = convert_triton_to_dlpack_type(dtype_);

  switch (memory_type_) {
    case TRITONSERVER_MEMORY_GPU:
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
      break;
    case TRITONSERVER_MEMORY_CPU:
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDAHost;
      break;
  }

  return py::capsule(
      static_cast<void*>(dlpack_tensor), "dltensor", &delete_unused_dltensor);
}

void
PbTensor::DeleteDLPack()
{
  if (dl_managed_tensor_ != nullptr) {
    dl_managed_tensor_->deleter(dl_managed_tensor_);
    dl_managed_tensor_ = nullptr;
  }
}

std::unique_ptr<PbTensor>
PbTensor::FromDLPack(std::string name, py::capsule dlpack_tensor)
{
  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(dlpack_tensor.get_pointer());

  // TODO: Make sure that the tensor is contiguous
  void* memory_ptr = dl_managed_tensor->dl_tensor.data;
  std::vector<int64_t> dims(
      dl_managed_tensor->dl_tensor.shape,
      dl_managed_tensor->dl_tensor.shape + dl_managed_tensor->dl_tensor.ndim);

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  if (dl_managed_tensor->dl_tensor.device.device_type ==
      DLDeviceType::kDLCUDA) {
    memory_type = TRITONSERVER_MEMORY_GPU;
    memory_type_id = dl_managed_tensor->dl_tensor.device.device_id;
  } else if (
      dl_managed_tensor->dl_tensor.device.device_type == DLDeviceType::kDLCPU) {
    memory_type = TRITONSERVER_MEMORY_CPU;
    memory_type_id = 0;
  } else {
    // TODO unsupported.
  }

  TRITONSERVER_DataType dtype =
      convert_dlpack_to_triton_type(dl_managed_tensor->dl_tensor.dtype);

  // Calculate tensor size.
  uint64_t size = 1;
  for (auto& dim : dims) {
    size *= dim;
  }
  size *= (dl_managed_tensor->dl_tensor.dtype.bits + 7) / 8;

  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dlpack");
  return std::make_unique<PbTensor>(
      name, dims, static_cast<int>(dtype), memory_type, memory_type_id,
      memory_ptr, size, dl_managed_tensor);
}


const std::string&
PbTensor::Name()
{
  return name_;
}

py::array&
PbTensor::AsNumpy()
{
  if ((tensor_type_ == PYTHONBACKEND_RAW &&
       memory_type_ == TRITONSERVER_MEMORY_CPU) ||
      (tensor_type_ == PYTHONBACKEND_DLPACK &&
       memory_type_ == TRITONSERVER_MEMORY_CPU)) {
    if (numpy_array_.equal(py::none())) {
      numpy_array_ = py::array(
          py::dtype(py::format_descriptor<float>::format()), byte_size_,
          (void*)memory_ptr_);
      // numpy_array_ =
      //     numpy_array_.attr("view")(triton_to_numpy_type(dtype_));
    }
  } else if (tensor_type_ == PYTHONBACKEND_NUMPY) {
    return numpy_array_;
  } else {
    std::cout << "error saw " << std::endl;
    throw PythonBackendException("GPU Tensors cannot be converted to NumPy");
  }

  return numpy_array_;
}

int
PbTensor::TritonDtype()
{
  return dtype_;
}

void*
PbTensor::GetDataPtr()
{
  return static_cast<void*>(memory_ptr_);
}
}}}  // namespace triton::backend::python
