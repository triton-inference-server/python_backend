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

#ifdef TRITON_ENABLE_GPU_TENSORS
#include <cuda.h>
#endif  // TRITON_ENABLE_GPU_TENSORS

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

  if (dtype_ == TRITONSERVER_TYPE_BYTES) {
    py::module triton_pb_utils =
        py::module::import("triton_python_backend_utils");
    numpy_array_serialized_ =
        triton_pb_utils.attr("serialize_byte_tensor")(numpy_array);
    memory_ptr_ = numpy_array_serialized_.request().ptr;
    byte_size_ = numpy_array_serialized_.nbytes();
  } else {
    memory_ptr_ = numpy_array_.request().ptr;
    byte_size_ = numpy_array_.nbytes();
  }

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
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

  if (dtype == TRITONSERVER_TYPE_BYTES) {
    py::module triton_pb_utils =
        py::module::import("triton_python_backend_utils");
    numpy_array_serialized_ =
        triton_pb_utils.attr("serialize_byte_tensor")(numpy_array);
    memory_ptr_ = numpy_array_serialized_.request().ptr;
    byte_size_ = numpy_array_serialized_.nbytes();

  } else {
    memory_ptr_ = numpy_array_.request().ptr;
    byte_size_ = numpy_array_.nbytes();
  }
  tensor_type_ = PYTHONBACKEND_NUMPY;
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  dtype_ = dtype;

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;
}
#endif  // TRITON_PB_STUB

PbTensor::PbTensor(
    const std::string& name, const std::vector<int64_t>& dims,
    TRITONSERVER_DataType dtype, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* memory_ptr, uint64_t byte_size,
    DLManagedTensor* dl_managed_tensor, off_t shm_offset)
{
  name_ = name;
  memory_ptr_ = memory_ptr;
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  dtype_ = dtype;
  dims_ = dims;
  shm_offset_ = shm_offset;

#ifdef TRITON_PB_STUB
  if (memory_type_ == TRITONSERVER_MEMORY_CPU ||
      memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED) {
    if (dtype != TRITONSERVER_TYPE_BYTES) {
      py::object numpy_array =
          py::array(triton_to_pybind_dtype(dtype_), dims_, (void*)memory_ptr_);
      numpy_array_ = numpy_array.attr("view")(triton_to_numpy_type(dtype_));
    } else {
      py::object numpy_array = py::array(
          triton_to_pybind_dtype(TRITONSERVER_TYPE_UINT8), {byte_size},
          (void*)memory_ptr_);
      py::module triton_pb_utils =
          py::module::import("triton_python_backend_utils");
      numpy_array_ =
          triton_pb_utils.attr("deserialize_bytes_tensor")(numpy_array)
              .attr("reshape")(dims);
    }
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

std::shared_ptr<PbTensor>
PbTensor::FromNumpy(const std::string& name, py::object numpy_array)
{
  return std::make_shared<PbTensor>(name, numpy_array);
}

py::capsule
PbTensor::ToDLPack()
{
  if (dtype_ == TRITONSERVER_TYPE_BYTES) {
    throw PythonBackendException(
        "DLPack does not have support for string tensors.");
  }

  DLManagedTensor* dlpack_tensor = new DLManagedTensor;
  dlpack_tensor->dl_tensor.ndim = dims_.size();
  dlpack_tensor->dl_tensor.byte_offset = 0;
  dlpack_tensor->dl_tensor.data = memory_ptr_;
  dlpack_tensor->dl_tensor.shape = &dims_[0];
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->deleter = [](DLManagedTensor* m) {};
  dlpack_tensor->dl_tensor.device.device_id = memory_type_id_;
  dlpack_tensor->dl_tensor.dtype = triton_to_dlpack_type(dtype_);

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
#endif  // TRITON_PB_STUB

void
PbTensor::DeleteDLPack()
{
  if (dl_managed_tensor_ != nullptr) {
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
  std::shared_ptr<PbTensor> pb_tensor;

  char* data = nullptr;
  if (raw_data->memory_type == TRITONSERVER_MEMORY_CPU) {
    shm_pool->MapOffset((char**)&data, raw_data->memory_ptr);
    pb_tensor = std::make_shared<PbTensor>(
        name, std::vector<int64_t>(dims, dims + dims_count), tensor_shm->dtype,
        raw_data->memory_type, raw_data->memory_type_id, data,
        raw_data->byte_size, nullptr /* DLManaged Tensor */);
  } else if (raw_data->memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU_TENSORS
    cudaIpcMemHandle_t* cuda_ipc_mem_handle;
    shm_pool->MapOffset((char**)&cuda_ipc_mem_handle, raw_data->memory_ptr);
    if (!tensor_shm->is_reused) {
      cudaSetDevice(raw_data->memory_type_id);

      cudaError_t err = cudaIpcOpenMemHandle(
          (void**)&data, *cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        throw PythonBackendException(std::string(
                                         "failed to open cuda ipc handle: " +
                                         std::string(cudaGetErrorString(err)))
                                         .c_str());
      }
      // Adjust the offset. cudaIpcOpenMemHandle will map the base address of a
      // GPU pointer and the offset is not preserved when transferring the
      // pointer using cudaIpcMemHandle.
      data = data + raw_data->offset;
      pb_tensor = std::make_shared<PbTensor>(
          name, std::vector<int64_t>(dims, dims + dims_count),
          tensor_shm->dtype, raw_data->memory_type, raw_data->memory_type_id,
          data, raw_data->byte_size, nullptr /* DLManaged Tensor */);
      pb_tensor->cuda_ipc_mem_handle_ = cuda_ipc_mem_handle;
    } else {
      pb_tensor = std::make_shared<PbTensor>(
          name, std::vector<int64_t>(dims, dims + dims_count),
          tensor_shm->dtype, raw_data->memory_type, raw_data->memory_type_id,
          data, raw_data->byte_size, nullptr /* DLManaged Tensor */);
      pb_tensor->reused_tensor_offset_ = raw_data->offset;
      pb_tensor->cuda_ipc_mem_handle_ = cuda_ipc_mem_handle;
      pb_tensor->is_reused_ = true;
    }
#else
    throw PythonBackendException("GPU Tensor is not supported.");
#endif  // TRITON_ENABLE_GPU_TENSORS
  }

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
#endif  // TRITON_PB_STUB

PbTensor::~PbTensor() noexcept(false)
{
#ifdef TRITON_ENABLE_GPU_TENSORS
  if (!IsCPU() && cuda_ipc_mem_handle_ != nullptr &&
      destruct_cuda_ipc_mem_handle_) {
    cudaError_t err = cudaIpcCloseMemHandle(GetGPUStartAddress());
    cuda_ipc_mem_handle_ = nullptr;
    if (err != cudaSuccess) {
      throw PythonBackendException(std::string(
                                       "failed to close cuda ipc handle: " +
                                       std::string(cudaGetErrorString(err)))
                                       .c_str());
    }
  }
#endif  // TRITON_ENABLE_GPU_TENSORS
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
#endif  // TRITON_PB_STUB

#ifdef TRITON_ENABLE_GPU_TENSORS
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
PbTensor::SetReusedIpcHandle(cudaIpcMemHandle_t* cuda_ipc_mem_handle)
{
  destruct_cuda_ipc_mem_handle_ = false;
  cuda_ipc_mem_handle_ = cuda_ipc_mem_handle;
  is_reused_ = true;
}

cudaIpcMemHandle_t*
PbTensor::CudaIpcMemHandle()
{
  return cuda_ipc_mem_handle_;
}
#endif  // TRITON_ENABLE_GPU_TENSORS

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

    data_ptr = static_cast<char*>(memory_ptr_);

    uint64_t* ptr_offset;
    SaveTensorToSharedMemory(
        shm_pool, tensor_shm, data_in_shm, memory_type, memory_type_id,
        byte_size_, tensor_name.c_str(), dims_.data(), dims_count, dtype_triton,
        &ptr_offset, shm_offset_);
    *ptr_offset = 0;

    // TODO: We can remove this memcpy if the numpy object
    // is already in shared memory.
    if (copy) {
      std::copy(data_ptr, data_ptr + byte_size_, data_in_shm);
    } else {
      memory_ptr_ = reinterpret_cast<void*>(data_in_shm);
    }
  } else {
#ifdef TRITON_ENABLE_GPU_TENSORS
    char* cuda_handle;
    uint64_t* ptr_offset;
    SaveTensorToSharedMemory(
        shm_pool, tensor_shm, cuda_handle, this->MemoryType(),
        this->MemoryTypeId(), this->ByteSize(), tensor_name.c_str(),
        this->Dims().data(), this->Dims().size(), dtype_triton, &ptr_offset);
    char* d_ptr = reinterpret_cast<char*>(this->GetDataPtr());
    *ptr_offset = GetDevicePointerOffset(d_ptr);
    if (!IsReused()) {
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
      cuda_ipc_mem_handle_ = reinterpret_cast<cudaIpcMemHandle_t*>(cuda_handle);
    } else {
      tensor_shm->is_reused = true;
      RawData* raw_data;
      shm_pool->MapOffset((char**)&raw_data, tensor_shm->raw_data);
      raw_data->offset = *ptr_offset;
      *(reinterpret_cast<cudaIpcMemHandle_t*>(cuda_handle)) =
          *CudaIpcMemHandle();
    }
    void* start_address = this->GetGPUStartAddress();
    *ptr_offset = reinterpret_cast<char*>(this->GetDataPtr()) -
                  reinterpret_cast<char*>(start_address);
#else
    throw PythonBackendException("GPU tensors are not supported.");
#endif  // TRITON_ENABLE_GPU_TENSORS
  }
}

void
PbTensor::SetDataPtr(void* ptr)
{
  memory_ptr_ = reinterpret_cast<void*>(
      (reinterpret_cast<char*>(ptr) + reused_tensor_offset_));
}

bool
PbTensor::IsReused()
{
  return is_reused_;
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
