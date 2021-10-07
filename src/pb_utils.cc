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
#include "shm_manager.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace triton { namespace backend { namespace python {

#define THROW_IF_ERROR(MSG, X)           \
  do {                                   \
    int return__ = (X);                  \
    if (return__ != 0) {                 \
      throw PythonBackendException(MSG); \
    }                                    \
  } while (false)

void
LoadStringFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset, char*& str)
{
  String* string;
  shm_pool->MapOffset((char**)&string, shm_offset);
  shm_pool->MapOffset((char**)&str, string->data);
}

void
SaveStringToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& shm_offset, const char* str)
{
  String* string_shm;
  shm_pool->Map((char**)&string_shm, sizeof(String), shm_offset);
  string_shm->length = strlen(str) + 1;

  char* string_data;
  off_t str_data_offset;
  shm_pool->Map((char**)&string_data, string_shm->length, str_data_offset);
  string_shm->data = str_data_offset;
  strcpy(string_data, str);
}

void
print_gpu_data(void* d_ptr, size_t byte_size, const char* message)
{
  size_t test_size = byte_size / sizeof(float) + 1;
  float test_data[test_size];

  cudaError_t err =
      cudaMemcpy(test_data, d_ptr, byte_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw PythonBackendException(
        std::string(
            "failed to copy data: " + std::string(cudaGetErrorString(err)))
            .c_str());
  }

  std::cout << message << std::endl;
  for (size_t i = 0; i < test_size; ++i) {
    std::cout << test_data[i] << " ";
  }
  std::cout << std::endl;
}

void
print_cuda_ipc_handle(cudaIpcMemHandle_t* cuda_ipc, const char* message)
{
  std::cout << message << std::endl;
  char* cuda_handle = (char*)cuda_ipc;
  for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
    std::cout << unsigned(cuda_handle[i]) << " ";
  }

  std::cout << std::endl;
}

void
SaveRawDataToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& raw_data_offset,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size, uint64_t** offset,
    off_t raw_ptr_offset)
{
  // raw data
  RawData* raw_data;
  shm_pool->Map((char**)&raw_data, sizeof(RawData), raw_data_offset);

  raw_data->memory_type = memory_type;
  raw_data->memory_type_id = memory_type_id;
  raw_data->byte_size = byte_size;
  *offset = &(raw_data->offset);
  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    // If the raw_ptr_offset is not equal to zero, the user has provided
    // the offset for the raw ptr.
    if (raw_ptr_offset == 0) {
      off_t buffer_offset;
      shm_pool->Map((char**)&raw_data_ptr, byte_size, buffer_offset);
      raw_data->memory_ptr = buffer_offset;
    } else {
      raw_data->memory_ptr = raw_ptr_offset;
    }
  }

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    off_t buffer_offset;
    shm_pool->Map(
        (char**)&raw_data_ptr, sizeof(cudaIpcMemHandle_t), buffer_offset);
    raw_data->memory_ptr = buffer_offset;

#else
    throw PythonBackendException(
        "Python backend does not support GPU tensors.");
#endif  // TRITON_ENABLE_GPU
  }
}

void
SaveMapToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& shm_offset,
    const std::unordered_map<std::string, std::string>& map)
{
  Dict* dict;
  shm_pool->Map((char**)&dict, sizeof(Dict), shm_offset);
  dict->length = map.size();

  Pair* pairs;
  shm_pool->Map((char**)&pairs, sizeof(Pair) * map.size(), dict->values);

  size_t i = 0;
  for (const auto& pair : map) {
    SaveStringToSharedMemory(shm_pool, pairs[i].key, pair.first.c_str());
    SaveStringToSharedMemory(shm_pool, pairs[i].value, pair.second.c_str());
    i += 1;
  }
}

void
LoadMapFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset,
    std::unordered_map<std::string, std::string>& map)
{
  Dict* dict;
  shm_pool->MapOffset((char**)&dict, shm_offset);

  Pair* pairs;
  shm_pool->MapOffset((char**)&pairs, dict->values);
  for (size_t i = 0; i < dict->length; i++) {
    char* key;
    LoadStringFromSharedMemory(shm_pool, pairs[i].key, key);

    char* value;
    LoadStringFromSharedMemory(shm_pool, pairs[i].value, value);
    map.emplace(std::make_pair(key, value));
  }
}

void
SaveTensorToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Tensor* tensor,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, uint64_t byte_size, const char* name,
    const int64_t* dims, size_t dims_count, TRITONSERVER_DataType dtype,
    uint64_t** offset_ptr, off_t raw_ptr_offset)
{
  off_t raw_data_offset;
  // Raw Data
  SaveRawDataToSharedMemory(
      shm_pool, raw_data_offset, raw_data_ptr, memory_type, memory_type_id,
      byte_size, offset_ptr, raw_ptr_offset);
  tensor->raw_data = raw_data_offset;

  // name
  off_t name_offset;
  SaveStringToSharedMemory(shm_pool, name_offset, name);
  tensor->name = name_offset;

  // input dtype
  tensor->dtype = dtype;

  // input dims
  int64_t* tensor_dims;
  tensor->dims_count = dims_count;
  off_t tensor_dims_offset;
  shm_pool->Map(
      (char**)&tensor_dims, sizeof(int64_t) * dims_count, tensor_dims_offset);
  tensor->dims = tensor_dims_offset;

  for (size_t j = 0; j < dims_count; ++j) {
    tensor_dims[j] = dims[j];
  }
}

void
CopySingleArchiveEntry(archive* input_archive, archive* output_archive)
{
  const void* buff;
  size_t size;
#if ARCHIVE_VERSION_NUMBER >= 3000000
  int64_t offset;
#else
  off_t offset;
#endif

  for (;;) {
    int return_status;
    return_status =
        archive_read_data_block(input_archive, &buff, &size, &offset);
    if (return_status == ARCHIVE_EOF)
      break;
    if (return_status != ARCHIVE_OK)
      throw PythonBackendException(
          "archive_read_data_block() failed with error code = " +
          std::to_string(return_status));

    return_status =
        archive_write_data_block(output_archive, buff, size, offset);
    if (return_status != ARCHIVE_OK) {
      throw PythonBackendException(
          "archive_write_data_block() failed with error code = " +
          std::to_string(return_status) + ", error message is " +
          archive_error_string(output_archive));
    }
  }
}

void
ExtractTarFile(std::string& archive_path, std::string& dst_path)
{
  char current_directory[PATH_MAX];
  if (getcwd(current_directory, PATH_MAX) == nullptr) {
    throw PythonBackendException(
        (std::string("Failed to get the current working directory. Error: ") +
         std::strerror(errno)));
  }
  if (chdir(dst_path.c_str()) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + dst_path +
         " Error: " + std::strerror(errno))
            .c_str());
  }

  struct archive_entry* entry;
  int flags = ARCHIVE_EXTRACT_TIME;

  struct archive* input_archive = archive_read_new();
  struct archive* output_archive = archive_write_disk_new();
  archive_write_disk_set_options(output_archive, flags);

  archive_read_support_filter_gzip(input_archive);
  archive_read_support_format_tar(input_archive);

  if (archive_path.size() == 0) {
    throw PythonBackendException("The archive path is empty.");
  }

  THROW_IF_ERROR(
      "archive_read_open_filename() failed.",
      archive_read_open_filename(
          input_archive, archive_path.c_str(), 10240 /* block_size */));

  while (true) {
    int read_status = archive_read_next_header(input_archive, &entry);
    if (read_status == ARCHIVE_EOF)
      break;
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(
          std::string("archive_read_next_header() failed with error code = ") +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(input_archive));
    }

    read_status = archive_write_header(output_archive, entry);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_header() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }

    CopySingleArchiveEntry(input_archive, output_archive);

    read_status = archive_write_finish_entry(output_archive);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_finish_entry() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }
  }

  archive_read_close(input_archive);
  archive_read_free(input_archive);

  archive_write_close(output_archive);
  archive_write_free(output_archive);

  // Revert the directory change.
  if (chdir(current_directory) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + current_directory)
            .c_str());
  }
}

bool
FileExists(std::string& path)
{
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

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

class RunBeforeReturn {
  std::function<void()> run_before_return_fn_;

 public:
  RunBeforeReturn(const std::function<void()>& run_before_return_fn)
      : run_before_return_fn_(run_before_return_fn)
  {
  }

  ~RunBeforeReturn() { run_before_return_fn_(); }
};

}}}  // namespace triton::backend::python
