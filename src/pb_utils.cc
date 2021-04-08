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

#include "pb_utils.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {
void
CreateIPCMutex(pthread_mutex_t** mutex)
{
  pthread_mutexattr_t mattr;
  pthread_mutexattr_init(&mattr);
  pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(*mutex, &mattr);
  pthread_mutexattr_destroy(&mattr);
}

void
CreateIPCCondVariable(pthread_cond_t** cv)
{
  pthread_condattr_t cattr;
  pthread_condattr_init(&cattr);
  pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
  pthread_cond_init(*cv, &cattr);
  pthread_condattr_destroy(&cattr);
}

void
LoadStringFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset, char*& str)
{
  String* string;
  shm_pool->MapOffset((char**)&string, sizeof(String), shm_offset);
  shm_pool->MapOffset((char**)&str, string->length, string->data);
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
SaveRawDataToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& raw_data_offset,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size)
{
  // raw data
  RawData* raw_data;
  shm_pool->Map((char**)&raw_data, sizeof(RawData), raw_data_offset);

  raw_data->memory_type = memory_type;
  raw_data->memory_type_id = memory_type_id;
  raw_data->byte_size = byte_size;

  off_t buffer_offset;
  shm_pool->Map((char**)&raw_data_ptr, byte_size, buffer_offset);
  raw_data->memory_ptr = buffer_offset;
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
  shm_pool->MapOffset((char**)&dict, sizeof(Dict), shm_offset);

  Pair* pairs;
  shm_pool->MapOffset(
      (char**)&pairs, sizeof(Pair) * dict->length, dict->values);
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
    int memory_type_id, uint64_t byte_size, const char* name,
    const int64_t* dims, size_t dims_count, TRITONSERVER_DataType dtype)
{
  // Raw Data
  off_t raw_data_offset;
  SaveRawDataToSharedMemory(
      shm_pool, raw_data_offset, raw_data_ptr, memory_type, memory_type_id,
      byte_size);
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

}}}  // namespace triton::backend::python
