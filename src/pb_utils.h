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

#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

#include <pthread.h>
#include <memory>
#include <string>
#include <vector>

namespace triton { namespace backend { namespace python {


#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, MUTEX, X) \
  do {                                                                        \
    TRITONSERVER_Error* raarie_err__ = (X);                                   \
    if (raarie_err__ != nullptr) {                                            \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__);        \
      pthread_mutex_unlock(MUTEX);                                            \
      return nullptr;                                                         \
    }                                                                         \
  } while (false)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define STUB_SET_RESPONSE_ERROR_IF_ERROR(SHM_POOL, RESPONSE, R, X)             \
  do {                                                                         \
    TRITONSERVER_Error* err__ = (X);                                           \
    if (err__ != nullptr) {                                                    \
      const char* err_message__ = TRITONSERVER_ErrorMessage(err__);            \
      off_t string_offset__;                                                   \
      TRITONSERVER_Error* err_string__;                                        \
      err_string__ =                                                           \
          SaveStringToSharedMemory(SHM_POOL, string_offset__, err_message__);  \
      if (err_string__ != nullptr) {                                           \
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err__)); \
        LOG_MESSAGE(                                                           \
            TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err_string__));  \
      } else {                                                                 \
        RESPONSE->has_error = true;                                            \
        RESPONSE->error = string_offset__;                                     \
      }                                                                        \
                                                                               \
      if (R)                                                                   \
        return;                                                                \
    }                                                                          \
  } while (false)

#define RETURN_AND_UNLOCK_MUTEX_IF_ERROR(MUTEX, X) \
  do {                                             \
    TRITONSERVER_Error* rie_err__ = (X);           \
    if (rie_err__ != nullptr) {                    \
      pthread_mutex_unlock(MUTEX);                 \
      return rie_err__;                            \
    }                                              \
  } while (false)


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
};

struct ResponseBatch {
  off_t responses;  // Offset for response object.
  uint32_t batch_size;
  off_t error;
  bool has_error;
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

TRITONSERVER_Error* SaveStringToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& shm_offset,
    const char* str);
TRITONSERVER_Error* LoadStringFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t shm_offset, char*& str);

TRITONSERVER_Error* LoadRawDataFromSharedLibrary(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& tensor_shm_offset,
    const Tensor& tensor);
TRITONSERVER_Error* SaveRawDataToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t& raw_data_offset,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size);

TRITONSERVER_Error* SaveTensorToSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, Tensor* tensor,
    char*& raw_data_ptr, TRITONSERVER_MemoryType memory_type,
    int memory_type_id, uint64_t byte_size, const char* name,
    const int64_t* dims, size_t dims_count, TRITONSERVER_DataType dtype);
TRITONSERVER_Error* LoadTensorFromSharedMemory(
    std::unique_ptr<SharedMemory>& shm_pool, off_t tensor_shm_offset,
    Tensor& tensor);
}}}  // namespace triton::backend::python
