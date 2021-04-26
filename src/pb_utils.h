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

#pragma once

#include <pthread.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

#define STUB_SET_RESPONSE_ERROR_IF_ERROR(SHM_POOL, RESPONSE, R, X)        \
  do {                                                                    \
    try {                                                                 \
      (X);                                                                \
    }                                                                     \
    catch (cont PythonBackendException & pb_exception) {                  \
      off_t string_offset__;                                              \
      try {                                                               \
        SaveStringToSharedMemory(                                         \
            SHM_POOL, string_offset__, pb_exception->err_.error_message); \
        RESPONSE->has_error = true;                                       \
        RESPONSE->error = string_offset__;                                \
        if (R)                                                            \
          return;                                                         \
      }                                                                   \
      catch (cont PythonBackendException & pb2_exception) {               \
        printf(TRITONSERVER_ErrorMessage(                                 \
            pb_exception->err_.error_message.c_str()));                   \
        printf(                                                           \
            TRITONSERVER_LOG_ERROR,                                       \
            TRITONSERVER_ErrorMessage(                                    \
                pb2_exception->err_.error_message.c_str()));              \
      }                                                                   \
    }                                                                     \
    while (false)

// Create a conditional variable.
void CreateIPCCondVariable(pthread_cond_t** cv);

// Create a mutex that is shared between different processes.
void CreateIPCMutex(pthread_mutex_t** mutex);

//
// Represents a raw data
//
struct RawData {
  off_t memory_ptr;
  TRITONSERVER_MemoryType memory_type;
  int memory_type_id;
  uint64_t byte_size;
};

//
// Represents a Tensor object that will be passed to Python code.
//
struct Tensor {
  off_t raw_data;  // Offset for raw data field.
  off_t name;      // Offset for name field.
  TRITONSERVER_DataType dtype;
  off_t dims;  // Shared memory offset for the dimensions.
  size_t dims_count;
};

struct String {
  off_t data;
  size_t length;
};

//
// Inference Request
//
struct Request {
  off_t id;  // Offset for the id field.
  uint64_t correlation_id;
  off_t inputs;  // Offset for input field.
  uint32_t requested_input_count;
  off_t requested_output_names;  // Offset for the requested output names
  uint32_t requested_output_count;
};

struct Response {
  off_t outputs;  // Offset for Tensor output.
  uint32_t outputs_size;
  off_t error;
  bool has_error;
  bool is_error_set;  // Indicates whether this error has a message or not.
};

struct ResponseBatch {
  off_t responses;  // Offset for response object.
  uint32_t batch_size;
  off_t error;
  bool has_error;
  bool is_error_set;  // Indicates whether this error has a message or not.
};

struct RequestBatch {
  off_t requests;  // Offset for request object.
  uint32_t batch_size;
};

struct IPCMessage {
  // request points to a RequestBatch struct.
  off_t request_batch;

  // response points to a ResponseBatch struct.
  off_t response_batch;
};

// Representing a key value pair
struct Pair {
  off_t key;
  off_t value;
};

struct Dict {
  uint32_t length;
  // Values point to the location where there are `length` pairs.
  off_t values;
};

//
// PythonBackendError
//
struct PythonBackendError {
  std::string error_message;
};

//
// PythonBackendException
//
// Exception thrown if error occurs in PythonBackend.
//
struct PythonBackendException {
  PythonBackendException(std::unique_ptr<PythonBackendError> err)
      : err_(std::move(err))
  {
  }
  std::unique_ptr<PythonBackendError> err_;
};

void SaveMapToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& shm_offset,
    const std::unordered_map<std::string, std::string>& map);

void LoadMapFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset,
    std::unordered_map<std::string, std::string>& map);

void SaveStringToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& shm_offset,
    const char* str);
void LoadStringFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset, char*& str);

void LoadRawDataFromSharedLibrary(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& tensor_shm_offset,
    const Tensor& tensor);
void SaveRawDataToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& raw_data_offset,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size);

void SaveTensorToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Tensor* tensor,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size, const char* name,
    const int64_t* dims, size_t dims_count, TRITONSERVER_DataType dtype);
void LoadTensorFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t tensor_shm_offset,
    Tensor& tensor);

}}}  // namespace triton::backend::python
