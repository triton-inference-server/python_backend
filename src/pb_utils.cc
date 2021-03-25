// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <cstring>

#include "pb_utils.h"
#include "shm_manager.h"
//#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace python {
void CreateIPCMutex(pthread_mutex_t** mutex)
{
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(*mutex, &mattr);
    pthread_mutexattr_destroy(&mattr);
}

void CreateIPCCondVariable(pthread_cond_t** cv)
{
    pthread_condattr_t cattr;
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(*cv, &cattr);
    pthread_condattr_destroy(&cattr);
}

TRITONSERVER_Error* LoadStringFromSharedMemory(std::unique_ptr<SharedMemory> &shm_pool, off_t shm_offset, char *&str)
{
    String *string;
    RETURN_IF_ERROR(shm_pool->MapOffset((char **) &string, sizeof(String), shm_offset));
    RETURN_IF_ERROR(shm_pool->MapOffset((char **) &str, string->length, string->data));

    return nullptr; // success
}

TRITONSERVER_Error* SaveStringToSharedMemory(std::unique_ptr<SharedMemory> &shm_pool, off_t &shm_offset, const char *str) {
  String *string_shm;
  RETURN_IF_ERROR(shm_pool->Map((char **) &string_shm, sizeof(String), shm_offset));
  string_shm->length = sizeof(str);

  char *string_data;
  off_t str_data_offset;
  RETURN_IF_ERROR(shm_pool->Map((char **) &string_data, sizeof(str), str_data_offset));
  string_shm->data = str_data_offset;
  strcpy(string_data, str);

  return nullptr; // success
}

// TRITONSERVER_Error* SaveTensorToSharedMemory(std::unique_ptr<SharedMemory> &shm_pool, off_t &tensor_shm_offset, const Tensor& tensor)
// {
//     return nullptr; // success
// }

TRITONSERVER_Error* SaveRawDataToSharedMemory(
    std::unique_ptr<SharedMemory> &shm_pool,
    off_t &raw_data_offset,
    char *&raw_data_ptr,
    TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size)
{
  // raw data
  RawData *raw_data;
  RETURN_IF_ERROR(shm_pool->Map((char **) &raw_data, sizeof(RawData), raw_data_offset));

  raw_data->memory_type = memory_type;
  raw_data->memory_type_id = memory_type_id;
  raw_data->byte_size = byte_size;

  off_t buffer_offset;
  RETURN_IF_ERROR(shm_pool->Map((char **) &raw_data_ptr, byte_size, buffer_offset));
  raw_data->memory_ptr = buffer_offset;

  return nullptr; // success
}

TRITONSERVER_Error* SaveTensorToSharedMemory(
    std::unique_ptr<SharedMemory> &shm_pool,
    Tensor* tensor,
    char *&raw_data_ptr,
    TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size, const char *name, const int64_t* dims, size_t dims_count, TRITONSERVER_DataType dtype)
{

  // Raw Data
  off_t raw_data_offset;
  RETURN_IF_ERROR(SaveRawDataToSharedMemory(shm_pool, raw_data_offset, raw_data_ptr, memory_type, memory_type_id, byte_size));
  tensor->raw_data = raw_data_offset;

  // name
  off_t name_offset;
  RETURN_IF_ERROR(SaveStringToSharedMemory(shm_pool, name_offset, name));
  tensor->name = name_offset;

  // input dtype
  tensor->dtype = dtype;

  // input dims
  int64_t *tensor_dims;
  tensor->dims_count = dims_count;
  off_t tensor_dims_offset;
  RETURN_IF_ERROR(shm_pool->Map((char **) &tensor_dims, sizeof(int64_t) * dims_count, tensor_dims_offset));
  tensor->dims = tensor_dims_offset;

  for (size_t j = 0; j < dims_count; ++j) {
    tensor_dims[j] = dims[j];
  }

  return nullptr; // success
}

// template<typename T>
// void AppendToTensorArray(py::list &py_tensors, const char* name, uint64_t byte_size, char *data_ptr)
// {
//     py::array_t<T, py::array::c_style | py::array::forcecast> array(byte_size);
// 
//     py::buffer_info buff = array.request();
//     T *ptr = static_cast<T *>(buff.ptr);
// }

}}}  // namespace triton::backend::python