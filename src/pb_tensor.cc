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

#include <cuda.h>
#ifdef TRITON_PB_STUB
#include "pb_stub_utils.h"
namespace py = pybind11;
#endif
#include "pb_tensor.h"
#include "pb_utils.h"


namespace triton { namespace backend { namespace python {

#ifdef TRITON_PB_STUB
PbTensor::PbTensor(const std::string& name, py::object numpy_array)
    : name_(name)
{
  dtype_ = numpy_to_triton_type(numpy_array.attr("dtype"));
  tensor_type_ = PYTHONBACKEND_NUMPY;
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;

  bool is_contiguous =
      numpy_array.attr("data").attr("c_contiguous").cast<bool>();
  if (!is_contiguous) {
    py::module numpy = py::module::import("numpy");
    numpy_array = numpy.attr("ascontiguousarray")(numpy_array);
  }
  numpy_array_ = numpy_array;
  memory_ptr_ = numpy_array_.request().ptr;

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
  byte_size_ = numpy_array_.nbytes();
}

PbTensor::PbTensor(
    const std::string& name, py::object numpy_array,
    TRITONSERVER_DataType dtype)
    : name_(name)
{
  if (numpy_to_triton_type(numpy_array.attr("dtype")) != dtype) {
    numpy_array = numpy_array.attr("view")(triton_to_numpy_type(dtype));
  }
  bool is_contiguous =
      numpy_array.attr("data").attr("c_contiguous").cast<bool>();
  if (!is_contiguous) {
    py::module numpy = py::module::import("numpy");
    numpy_array = numpy.attr("ascontiguousarray")(numpy_array);
  }
  numpy_array_ = numpy_array;
  tensor_type_ = PYTHONBACKEND_NUMPY;
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  memory_ptr_ = numpy_array_.request().ptr;
  dtype_ = dtype;

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
  byte_size_ = numpy_array_.nbytes();
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;
}
#endif

PbTensor::PbTensor(
    const std::string& name, const std::vector<int64_t>& dims,
    TRITONSERVER_DataType dtype, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* memory_ptr, uint64_t byte_size,
    DLManagedTensor* dl_managed_tensor)
{
  name_ = name;
  memory_ptr_ = memory_ptr;
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  dtype_ = dtype;
  dims_ = dims;

#ifdef TRITON_PB_STUB
  if (memory_type_ == TRITONSERVER_MEMORY_CPU ||
      memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED) {
    py::object numpy_array =
        py::array(triton_to_pybind_dtype(dtype_), dims_, (void*)memory_ptr_);
    numpy_array_ = numpy_array.attr("view")(triton_to_numpy_type(dtype_));
  } else {
    numpy_array_ = py::none();
  }
#endif

  byte_size_ = byte_size;
  dl_managed_tensor_ = dl_managed_tensor;

  if (dl_managed_tensor != nullptr) {
    tensor_type_ = PYTHONBACKEND_DLPACK;
  } else {
    tensor_type_ = PYTHONBACKEND_RAW;
  }
}

void
PbTensor::SetReusedGPUTensorName(const std::string& reused_gpu_tensor_name)
{
  reused_gpu_tensor_name_ = reused_gpu_tensor_name;
}

#ifdef TRITON_PB_STUB
std::shared_ptr<PbTensor>
PbTensor::FromNumpy(const std::string& name, py::object numpy_array)
{
  return std::make_shared<PbTensor>(name, numpy_array);
}
#endif

bool
PbTensor::IsCPU() const
{
  if (tensor_type_ == PYTHONBACKEND_NUMPY ||
      ((tensor_type_ == PYTHONBACKEND_RAW ||
        tensor_type_ == PYTHONBACKEND_DLPACK) &&
       (memory_type_ == TRITONSERVER_MEMORY_CPU ||
        memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED))) {
    return true;
  } else {
    return false;
  }
}

TRITONSERVER_MemoryType
PbTensor::MemoryType() const
{
  return memory_type_;
}

int64_t
PbTensor::MemoryTypeId() const
{
  return memory_type_id_;
}

uint64_t
PbTensor::ByteSize() const
{
  return byte_size_;
}

const std::vector<int64_t>&
PbTensor::Dims() const
{
  return dims_;
}

PYTHONBACKEND_TensorType
PbTensor::TensorType() const
{
  return tensor_type_;
}

#ifdef TRITON_PB_STUB
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
  dlpack_tensor->dl_tensor.data = memory_ptr_;
  dlpack_tensor->dl_tensor.shape = &dims_[0];
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->deleter = [](DLManagedTensor* m) {};
  dlpack_tensor->dl_tensor.device.device_id = memory_type_id_;
  dlpack_tensor->dl_tensor.dtype = triton_to_dlpack_type(dtype_);

  if (dtype_ == TRITONSERVER_TYPE_BYTES) {
    throw PythonBackendException(
        "DLPack does not have support for string tensors.");
  }

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
#endif

void
PbTensor::DeleteDLPack()
{
  if (dl_managed_tensor_ != nullptr) {
    std::cout << "Tensor deallocated" << std::endl;
    dl_managed_tensor_->deleter(dl_managed_tensor_);
    dl_managed_tensor_ = nullptr;
  }
}

std::shared_ptr<PbTensor>
PbTensor::LoadFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t tensor_offset)
{
  Tensor* tensor_shm;
  shm_pool->MapOffset((char**)&tensor_shm, tensor_offset);

  char* name;
  LoadStringFromSharedMemory(shm_pool, tensor_shm->name, name);
  std::string name_str = name;

  size_t dims_count = tensor_shm->dims_count;
  RawData* raw_data;
  shm_pool->MapOffset((char**)&raw_data, tensor_shm->raw_data);

  int64_t* dims;
  shm_pool->MapOffset((char**)&dims, tensor_shm->dims);

  std::string reused_gpu_tensor_name;
  char* data = nullptr;
  if (raw_data->memory_type == TRITONSERVER_MEMORY_CPU) {
    shm_pool->MapOffset((char**)&data, raw_data->memory_ptr);
  } else if (raw_data->memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (!tensor_shm->is_reused) {
      cudaIpcMemHandle_t* cuda_mem_handle;
      shm_pool->MapOffset((char**)&cuda_mem_handle, raw_data->memory_ptr);
      cudaSetDevice(raw_data->memory_type_id);

      cudaError_t err = cudaIpcOpenMemHandle(
          (void**)&data, *cuda_mem_handle, cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        throw PythonBackendException(std::string(
                                         "failed to open cuda ipc handle: " +
                                         std::string(cudaGetErrorString(err)))
                                         .c_str());
      }
    } else {
      char* reused_gpu_tensor;
      LoadStringFromSharedMemory(
          shm_pool, tensor_shm->reused_tensor_name, reused_gpu_tensor);
      reused_gpu_tensor_name = reused_gpu_tensor;
    }

    // Adjust the offset. cudaIpcOpenMemHandle will map the base address of a
    // GPU pointer and the offset is not preserved when transferring the pointer
    // using cudaIpcMemHandle.
    data = data + raw_data->offset;
#endif
  }

  std::shared_ptr<PbTensor> pb_tensor = std::make_shared<PbTensor>(
      name, std::vector<int64_t>(dims, dims + dims_count), tensor_shm->dtype,
      raw_data->memory_type, raw_data->memory_type_id, data,
      raw_data->byte_size, nullptr /**/);

#ifdef TRITON_ENABLE_GPU
  if (reused_gpu_tensor_name != "") {
    pb_tensor->SetReusedGPUTensorName(reused_gpu_tensor_name);
  }
#endif

  return pb_tensor;
}

#ifdef TRITON_PB_STUB
std::shared_ptr<PbTensor>
PbTensor::FromDLPack(const std::string& name, const py::capsule& dlpack_tensor)
{
  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(dlpack_tensor.get_pointer());

  void* memory_ptr = dl_managed_tensor->dl_tensor.data;
  memory_ptr = reinterpret_cast<char*>(memory_ptr) +
               dl_managed_tensor->dl_tensor.byte_offset;

  int64_t* strides = dl_managed_tensor->dl_tensor.strides;

  int ndim = dl_managed_tensor->dl_tensor.ndim;
  std::vector<int64_t> dims(
      dl_managed_tensor->dl_tensor.shape,
      dl_managed_tensor->dl_tensor.shape + ndim);

  // Check if the input is contiguous and in C order
  if (strides != nullptr) {
    int64_t calculated_stride{1};
    bool is_contiguous_c_order = true;
    for (size_t i = 1; i < dims.size(); i++) {
      if (strides[ndim - i] != calculated_stride) {
        is_contiguous_c_order = false;
        break;
      }

      calculated_stride *= dims[ndim - i];
    }

    if (!is_contiguous_c_order) {
      throw PythonBackendException(
          "DLPack tensor is not contiguous. Only contiguous DLPack "
          "tensors that are stored in C-Order are supported.");
    }
  }

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  switch (dl_managed_tensor->dl_tensor.device.device_type) {
    case DLDeviceType::kDLCUDA:
      memory_type = TRITONSERVER_MEMORY_GPU;
      memory_type_id = dl_managed_tensor->dl_tensor.device.device_id;
      break;
    case DLDeviceType::kDLCPU:
      memory_type = TRITONSERVER_MEMORY_CPU;
      memory_type_id = 0;
      break;
    case DLDeviceType::kDLCUDAHost:
      memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
      memory_type_id = 0;
      break;
    default:
      throw PythonBackendException(
          "DLDevice type " +
          std::to_string(dl_managed_tensor->dl_tensor.device.device_type) +
          " is not support by Python backend.");
      break;
  }

  TRITONSERVER_DataType dtype =
      dlpack_to_triton_type(dl_managed_tensor->dl_tensor.dtype);

  // Calculate tensor size.
  uint64_t byte_size = 1;
  for (auto& dim : dims) {
    byte_size *= dim;
  }
  byte_size *= (dl_managed_tensor->dl_tensor.dtype.bits + 7) / 8;

  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dlpack");
  return std::make_unique<PbTensor>(
      name, dims, dtype, memory_type, memory_type_id, memory_ptr, byte_size,
      dl_managed_tensor);
}
#endif

PbTensor::~PbTensor()
{
  DeleteDLPack();
}

const std::string&
PbTensor::Name() const
{
  return name_;
}

#ifdef TRITON_PB_STUB
const py::array&
PbTensor::AsNumpy() const
{
  if (this->IsCPU()) {
    return numpy_array_;
  } else {
    throw PythonBackendException(
        "Tensor is stored in GPU and cannot be converted to NumPy.");
  }

  return numpy_array_;
}
#endif

void*
PbTensor::GetGPUStartAddress()
{
  if (!this->IsCPU()) {
    CUdeviceptr start_address;
    CUresult cuda_err = cuPointerGetAttribute(
        &start_address, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        (CUdeviceptr)this->GetDataPtr());
    if (cuda_err != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(cuda_err, &error_string);
      throw PythonBackendException(
          std::string(
              "failed to get cuda pointer device attribute: " +
              std::string(error_string))
              .c_str());
    }

    return reinterpret_cast<void*>(start_address);
  }

  throw PythonBackendException(
      "Calling GetGPUStartAddress function on a CPU tensor.");
}

void
PbTensor::SaveToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Tensor* tensor_shm, bool copy)
{
  const std::string& tensor_name = this->Name();
  TRITONSERVER_DataType dtype_triton =
      static_cast<TRITONSERVER_DataType>(this->TritonDtype());
  tensor_shm->is_reused = false;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;

  if (this->IsCPU()) {
    size_t dims_count = dims_.size();
    memory_type = TRITONSERVER_MEMORY_CPU;
    memory_type_id = 0;

    char* data_in_shm;
    char* data_ptr;

    // Custom handling for type bytes.
    // if (dtype_triton == TRITONSERVER_TYPE_BYTES) {
    //   py::object serialized_bytes_or_none = serialize_bytes(numpy_array);
    //   if (serialize_bytes.is_none()) {
    //     const char* err_message = "An error happened during
    //     serialization."; LOG_INFO << err_message;
    //     SetErrorForResponse(response_shm, err_message);
    //     break;
    //   }

    //   py::bytes serialized_bytes = serialized_bytes_or_none;
    //   data_ptr = PyBytes_AsString(serialized_bytes.ptr());
    //   byte_size = PyBytes_Size(serialized_bytes.ptr());
    // } else {
    data_ptr = static_cast<char*>(memory_ptr_);
    // }

    uint64_t* ptr_offset;
    SaveTensorToSharedMemory(
        shm_pool, tensor_shm, data_in_shm, memory_type, memory_type_id,
        byte_size_, tensor_name.c_str(), dims_.data(), dims_count, dtype_triton,
        &ptr_offset);
    *ptr_offset = 0;

    // TODO: We can remove this memcpy if the numpy object
    // is already in shared memory.
    if (copy) {
      std::copy(data_ptr, data_ptr + byte_size_, data_in_shm);
    } else {
      memory_ptr_ = reinterpret_cast<void*>(data_in_shm);
    }
  } else {
#ifdef TRITON_ENABLE_GPU
    char* cuda_handle;
    uint64_t* ptr_offset;
    SaveTensorToSharedMemory(
        shm_pool, tensor_shm, cuda_handle, this->MemoryType(),
        this->MemoryTypeId(), this->ByteSize(), tensor_name.c_str(),
        this->Dims().data(), this->Dims().size(), dtype_triton, &ptr_offset);
    char* d_ptr = reinterpret_cast<char*>(this->GetDataPtr());
    *ptr_offset = GetDevicePointerOffset(d_ptr);
    if (reused_gpu_tensor_name_ == "") {
      cudaSetDevice(this->MemoryTypeId());
      cudaError_t err = cudaIpcGetMemHandle(
          reinterpret_cast<cudaIpcMemHandle_t*>(cuda_handle),
          this->GetDataPtr());
      if (err != cudaSuccess) {
        throw PythonBackendException(std::string(
                                         "failed to get cuda ipc handle: " +
                                         std::string(cudaGetErrorString(err)))
                                         .c_str());
      }
    } else {
      tensor_shm->is_reused = true;
      RawData* raw_data;
      shm_pool->MapOffset((char**)&raw_data, tensor_shm->raw_data);
      raw_data->offset = *ptr_offset;
      off_t reused_tensor_name_offset;
      SaveStringToSharedMemory(
          shm_pool, reused_tensor_name_offset, reused_gpu_tensor_name_.c_str());
      tensor_shm->reused_tensor_name = reused_tensor_name_offset;
    }
    void* start_address = this->GetGPUStartAddress();
    *ptr_offset = reinterpret_cast<char*>(this->GetDataPtr()) -
                  reinterpret_cast<char*>(start_address);
  }
#else
    throw PythonBackendException(
        "Python backend was not built with GPU tensor support");
#endif
}
void
PbTensor::SetDataPtr(void* ptr)
{
  memory_ptr_ = ptr;
}

bool
PbTensor::IsReused()
{
  return reused_gpu_tensor_name_ != "";
}

const std::string&
PbTensor::ReusedGPUTensorName()
{
  return reused_gpu_tensor_name_;
}

int
PbTensor::TritonDtype() const
{
  return dtype_;
}

void*
PbTensor::GetDataPtr() const
{
  return memory_ptr_;
}
}}}  // namespace triton::backend::python
