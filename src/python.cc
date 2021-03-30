// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <pthread.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "pb_utils.h"
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace py = pybind11;

namespace triton { namespace backend { namespace python {

struct BackendState {
  std::string python_lib;
  std::string python_runtime;
};

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get backend state
  BackendState* StateForBackend() { return backend_state_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  BackendState* backend_state_;
};

class ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance);

  TRITONBACKEND_Model* triton_model_;
  pthread_mutex_t* child_mutex_;
  pthread_cond_t* child_cond_;
  pthread_mutex_t* parent_mutex_;
  pthread_cond_t* parent_cond_;
  std::string model_path_;
  IPCMessage* ipc_message_;
  std::unique_ptr<SharedMemory> shm_pool_;
  // Child process pid
  pid_t pid_;

 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  ~ModelInstanceState();

  // Load Triton inputs to the appropriate Protobufs
  TRITONSERVER_Error* GetInputTensor(
      const uint32_t input_idx, Tensor* input_tensor,
      TRITONBACKEND_Request* request,
      std::vector<TRITONBACKEND_Response*>& responses);

  TRITONSERVER_Error* ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Child process loop
  void WaitForMessageCallback();

  // Create the child process.
  TRITONSERVER_Error* SetupChildProcess();

  // Notifies the child process on the new request
  void NotifyChild();

  // Notifies the process on the new request
  void NotifyParent();

  // Process a single request in the child process.
  void ChildProcessRequest(
      Request* request, ResponseBatch* response_batch,
      py::object& infer_request, py::object& PyRequest, py::object& PyTensor);

  // Process a single response in the child process.
  void ChildProcessResponse(
      Response* response_shm, ResponseBatch* response_batch,
      py::handle response);
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance)
{
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

void
ModelInstanceState::NotifyChild()
{
  pthread_mutex_lock(child_mutex_);
  pthread_cond_signal(child_cond_);
  pthread_mutex_unlock(child_mutex_);
}

void
ModelInstanceState::NotifyParent()
{
  pthread_mutex_lock(parent_mutex_);
  pthread_cond_signal(parent_cond_);
  pthread_mutex_unlock(parent_mutex_);
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Create Python inference requests
  RequestBatch* request_batch;
  off_t request_batch_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&request_batch, sizeof(RequestBatch), request_batch_offset));

  ipc_message_->request_batch = request_batch_offset;
  request_batch->batch_size = request_count;

  Request* requests_shm;
  off_t requests_shm_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&requests_shm, sizeof(Request) * request_count,
      requests_shm_offset));
  request_batch->requests = requests_shm_offset;

  // We take the responsiblity of the responses.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    Request* python_infer_request = &requests_shm[r];
    uint32_t requested_input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    python_infer_request->requested_input_count = requested_input_count;

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
    python_infer_request->requested_output_count = requested_output_count;

    Tensor* input_tensors;
    off_t input_tensors_offset;

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        shm_pool_->Map(
            (char**)&input_tensors, sizeof(Tensor) * requested_input_count,
            input_tensors_offset));
    python_infer_request->inputs = input_tensors_offset;

    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      Tensor* input_tensor = &input_tensors[iidx];

      GUARDED_RESPOND_IF_ERROR(
          responses, r, GetInputTensor(iidx, input_tensor, request, responses));
    }

    off_t* requested_output_names;
    off_t requested_output_names_offset;

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        shm_pool_->Map(
            (char**)&requested_output_names,
            sizeof(off_t) * requested_output_count,
            requested_output_names_offset));
    python_infer_request->requested_output_names =
        requested_output_names_offset;

    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));

      // output name
      off_t output_name_offset;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          SaveStringToSharedMemory(
              shm_pool_, output_name_offset, requested_output_name));
      requested_output_names[iidx] = output_name_offset;
    }

    // request id
    const char* id;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &id));

    off_t id_offset;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, SaveStringToSharedMemory(shm_pool_, id_offset, id));
    python_infer_request->id = id_offset;

    char* id_test;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        LoadStringFromSharedMemory(shm_pool_, id_offset, id_test));

    uint64_t correlation_id;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));
    python_infer_request->correlation_id = correlation_id;
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  pthread_mutex_lock(parent_mutex_);
  NotifyChild();

  // Wait for child notification
  pthread_cond_wait(parent_cond_, parent_mutex_);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Parsing the request response
  ResponseBatch* response_batch;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      &responses, request_count, parent_mutex_,
      shm_pool_->MapOffset(
          (char**)&response_batch, sizeof(ResponseBatch),
          ipc_message_->response_batch));

  // If inference fails, release all the requests and send an error response If
  // inference fails at this stage, it usually indicates a bug in the model code
  if (response_batch->has_error) {
    char* error_message;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count, parent_mutex_,
        LoadStringFromSharedMemory(
            shm_pool_, response_batch->error, error_message));
    for (uint32_t r = 0; r < request_count; ++r) {
      if (responses[r] == nullptr)
        continue;

      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to process the requests, message: ") +
           error_message)
              .c_str());
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");

      responses[r] = nullptr;
      TRITONSERVER_ErrorDelete(err);
    }

    pthread_mutex_unlock(parent_mutex_);
    return nullptr;
  }

  Response* responses_shm;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      &responses, request_count, parent_mutex_,
      shm_pool_->MapOffset(
          (char**)&responses_shm, sizeof(Response) * response_batch->batch_size,
          response_batch->responses));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    // Get response r
    Response* response_shm = &responses_shm[r];

    if (response_shm->has_error) {
      char* err_string;

      TRITONSERVER_Error* err_load_string;
      err_load_string = LoadStringFromSharedMemory(
          shm_pool_, response_shm->error, err_string);

      if (err_load_string == nullptr) {
        TRITONSERVER_Error* err =
            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_string);
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed sending response");
        TRITONSERVER_ErrorDelete(err);
      } else {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                err_load_string),
            "failed sending response");
      }
      responses[r] = nullptr;

      // If has_error is true, we do not look at the response even if the
      // response is set.
      continue;
    }

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    Tensor* output_tensors;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        shm_pool_->MapOffset(
            (char**)&output_tensors, sizeof(Tensor) * requested_output_count,
            response_shm->outputs));

    for (size_t j = 0; j < requested_output_count; ++j) {
      Tensor* output_tensor = &output_tensors[j];

      TRITONBACKEND_Output* triton_output;
      TRITONSERVER_DataType triton_dt = output_tensor->dtype;

      size_t dims_count = output_tensor->dims_count;

      int64_t* dims;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&dims, sizeof(int64_t) * dims_count,
              output_tensor->dims));

      char* name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          LoadStringFromSharedMemory(shm_pool_, output_tensor->name, name));

      RawData* raw_data;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&raw_data, sizeof(RawData), output_tensor->raw_data));

      char* data;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&data, raw_data->byte_size, raw_data->memory_ptr));

      // Prepare output buffers.
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &triton_output, name, triton_dt, dims, dims_count));

      uint64_t output_byte_size = raw_data->byte_size;

      // Custom handling for TRITONSERVER_TYPE_BYTES
      if (triton_dt == TRITONSERVER_TYPE_BYTES) {
        //
      } else {
      }

      void* output_buffer;

      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              triton_output, &output_buffer, output_byte_size,
              &output_memory_type, &output_memory_type_id));

      if ((responses[r] == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "can't create response in GPU memory."));
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory.")
                .c_str());
        continue;
      }

      // Copy Python output to Triton output buffers
      std::copy(data, data + output_byte_size, (char*)output_buffer);
    }

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, (std::string("Request ") + std::to_string(r) +
                                   ": failed to create output response")
                                      .c_str());
      continue;
    }

    // If error happens at this stage, we can only log it
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed sending response");
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), 1, exec_start_ns, compute_start_ns,
          compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       Name() + " released " + std::to_string(request_count) + " requests")
          .c_str());


  // Update the shared memory offset so that we can reuse the shared memory
  shm_pool_->SetOffset(request_batch_offset);

  pthread_mutex_unlock(parent_mutex_);
  return nullptr;
}

void
ModelInstanceState::ChildProcessResponse(
    Response* response_shm, ResponseBatch* response_batch, py::handle response)
{
  // Initialize has_error to false
  response_shm->has_error = false;

  py::bool_ py_has_error = response.attr("has_error")();
  bool has_error = py_has_error;

  if (has_error) {
    response_shm->has_error = true;
    off_t err_string_offset;
    py::str py_string_err = py::str(response.attr("error")());
    std::string response_error = py_string_err;
    LOG_IF_ERROR(
        SaveStringToSharedMemory(
            shm_pool_, err_string_offset, response_error.c_str()),
        "Failed to create string in shared memory");
    response_shm->error = err_string_offset;

    // Skip the response value if it has error
    return;
  }

  py::list output_tensors = response.attr("output_tensors")();
  size_t output_tensor_length = py::len(output_tensors);

  size_t j = 0;
  Tensor* output_tensors_shm;
  off_t output_tensors_offset;
  STUB_SET_RESPONSE_ERROR_IF_ERROR(
      shm_pool_, response_batch, true /* return */,
      shm_pool_->Map(
          (char**)&output_tensors_shm, sizeof(Tensor) * output_tensor_length,
          output_tensors_offset));
  response_shm->outputs = output_tensors_offset;
  response_shm->outputs_size = output_tensor_length;

  for (auto& output_tensor : output_tensors) {
    Tensor* output_tensor_shm = &output_tensors_shm[j];
    py::str name = output_tensor.attr("name")();
    std::string output_name = name;

    py::array numpy_array = output_tensor.attr("as_numpy")();
    py::buffer_info buffer = numpy_array.request();

    // TODO: If the data_ptr is in shared memory the copy is not required.
    char* data_ptr = static_cast<char*>(buffer.ptr);

    char* data_in_shm;
    const TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    const int memory_type_id = 0;

    size_t dims_count = numpy_array.ndim();
    int64_t dims[dims_count];

    // TODO: Fix serialization/deserialization when using TYPE_BYTES
    const ssize_t byte_size = numpy_array.nbytes();

    const ssize_t* numpy_shape = numpy_array.shape();
    for (size_t i = 0; i < dims_count; i++) {
      dims[i] = numpy_shape[i];
    }

    // TODO: Extract dtype from the users code.
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        SaveTensorToSharedMemory(
            shm_pool_, output_tensor_shm, data_in_shm, memory_type,
            memory_type_id, byte_size, output_name.c_str(), dims, dims_count,
            TRITONSERVER_TYPE_INT8));

    // TODO: We can remove this memcpy if the lifecycle of the numpy object
    // is correct.
    memcpy(data_in_shm, data_ptr, byte_size);
    j += 1;
  }
}

void
ModelInstanceState::ChildProcessRequest(
    Request* request, ResponseBatch* response_batch, py::object& infer_request,
    py::object& PyRequest, py::object& PyTensor)
{
  char* id = nullptr;
  STUB_SET_RESPONSE_ERROR_IF_ERROR(
      shm_pool_, response_batch, true /* return */,
      LoadStringFromSharedMemory(shm_pool_, request->id, id));

  uint32_t requested_input_count = request->requested_input_count;
  Tensor* input_tensors;
  STUB_SET_RESPONSE_ERROR_IF_ERROR(
      shm_pool_, response_batch, true /* return */,
      shm_pool_->MapOffset(
          (char**)&input_tensors, sizeof(Tensor) * requested_input_count,
          request->inputs));

  py::list py_input_tensors;
  for (size_t input_idx = 0; input_idx < requested_input_count; ++input_idx) {
    Tensor* input_tensor = &input_tensors[input_idx];

    char* name = nullptr;

    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        LoadStringFromSharedMemory(shm_pool_, input_tensor->name, name));

    RawData* raw_data;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        shm_pool_->MapOffset(
            (char**)&raw_data, sizeof(RawData), input_tensor->raw_data));

    char* data;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        shm_pool_->MapOffset(
            (char**)&data, raw_data->byte_size, raw_data->memory_ptr));

    size_t dims_count = input_tensor->dims_count;

    int64_t* dims;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        shm_pool_->MapOffset(
            (char**)&dims, sizeof(int64_t) * dims_count, input_tensor->dims));

    TRITONSERVER_DataType dtype = input_tensor->dtype;
    std::vector<int64_t> shape{dims, dims + dims_count};

    switch (dtype) {
      case TRITONSERVER_TYPE_BOOL:
        break;
      case TRITONSERVER_TYPE_UINT8:
        //            py::array_t<uint8_t, py::array::c_style |
        //            py::array::forcecast> array(raw_data->byte_size);
        break;
      case TRITONSERVER_TYPE_UINT16:
        //            py::array_t<uint16_t, py::array::c_style |
        //            py::array::forcecast> array(raw_data->byte_size);
        break;
      case TRITONSERVER_TYPE_UINT32:
        //            py::array_t<uint32_t, py::array::c_style |
        //            py::array::forcecast> array(raw_data->byte_size);
        break;
      case TRITONSERVER_TYPE_UINT64:
        //            py::array_t<uint64_t, py::array::c_style |
        //            py::array::forcecast> array(raw_data->byte_size);
        break;
      case TRITONSERVER_TYPE_INT8: {
        auto dtype =
            pybind11::dtype(pybind11::format_descriptor<int8_t>::format());
        py::array numpy_array(dtype, shape, (void*)data);
        py::object py_input_tensor = PyTensor(name, numpy_array);
        py_input_tensors.append(py_input_tensor);
        break;
      }
      case TRITONSERVER_TYPE_INT16:
        //            py::array_t<int16_t, py::array::c_style |
        //            py::array::forcecast> array;
        break;
      case TRITONSERVER_TYPE_INT32:
        //            py::array_t<int32_t, py::array::c_style |
        //            py::array::forcecast> array;
        break;
      case TRITONSERVER_TYPE_INT64:
        //            py::array_t<int32_t, py::array::c_style |
        //            py::array::forcecast> array;
        break;
      case TRITONSERVER_TYPE_FP32:
        //            py::array_t<int32_t, py::array::c_style |
        //            py::array::forcecast> array;
        break;
      case TRITONSERVER_TYPE_FP64:
        //            py::array_t<int32_t, py::array::c_style |
        //            py::array::forcecast> array;
        break;
      default:
        break;
    }
  }

  py::list py_requested_output_names;

  uint32_t requested_output_count = request->requested_output_count;
  off_t* output_names;
  STUB_SET_RESPONSE_ERROR_IF_ERROR(
      shm_pool_, response_batch, true /* return */,
      shm_pool_->MapOffset(
          (char**)&output_names, sizeof(off_t) * requested_output_count,
          request->requested_output_names));

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    char* output_name = nullptr;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, true /* return */,
        LoadStringFromSharedMemory(
            shm_pool_, output_names[output_idx], output_name));
    py_requested_output_names.append(output_name);
  }

  infer_request = PyRequest(
      py_input_tensors, id, request->correlation_id, py_requested_output_names);
}

void
ModelInstanceState::WaitForMessageCallback()
{
  py::scoped_interpreter guard{};
  py::object PyRequest;
  py::object PyTensor;
  py::object model_instance;

  ResponseBatch* response_batch;
  off_t response_batch_offset;
  auto err = shm_pool_->Map(
      (char**)&response_batch, sizeof(Response), response_batch_offset);
  ipc_message_->response_batch = response_batch_offset;
  response_batch->has_error = false;

  // If there is an error in this stage, we can only log it.
  if (err != nullptr) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
    NotifyParent();
    return;
  }

  try {
    ModelState* model_state = reinterpret_cast<ModelState*>(Model());
    std::string python_lib = model_state->StateForBackend()->python_lib;
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, model_path_);
    sys.attr("path").attr("insert")(0, python_lib);
    py::module python_backend_utils =
        py::module::import("triton_python_backend_utils");
    py::object TritonPythonModel =
        py::module::import("model").attr("TritonPythonModel");
    PyRequest = python_backend_utils.attr("InferenceRequest");
    PyTensor = python_backend_utils.attr("Tensor");

    triton::common::TritonJson::WriteBuffer buffer;
    Model()->ModelConfig().Write(&buffer);

    model_instance = TritonPythonModel();
    py::dict model_config_params;

    // Call initialize if exists.
    if (py::hasattr(model_instance, "initialize")) {
      model_config_params["model_config"] = buffer.MutableContents();
      model_config_params["model_instance_kind"] =
          TRITONSERVER_InstanceGroupKindString(kind_);
      model_config_params["model_instance_name"] = name_;
      model_config_params["model_instance_device_id"] =
          std::to_string(device_id_);
      model_config_params["model_repository"] = model_state->RepositoryPath();
      model_config_params["model_version"] =
          std::to_string(model_state->Version());
      model_config_params["model_name"] = model_state->Name();

      model_instance.attr("initialize")(model_config_params);
    }
  }
  catch (const py::error_already_set& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());

    off_t err_string_offset;
    LOG_IF_ERROR(
        SaveStringToSharedMemory(shm_pool_, err_string_offset, e.what()),
        "Failed to create string in shared memory");
    response_batch->has_error = true;
    response_batch->error = err_string_offset;
    NotifyParent();
    return;
  }

  for (;; pthread_mutex_unlock(child_mutex_)) {
    pthread_mutex_lock(child_mutex_);
    NotifyParent();

    pthread_cond_wait(child_cond_, child_mutex_);

    // Reset the value for has_error
    response_batch->has_error = false;

    RequestBatch* request_batch;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, false /* return */,
        shm_pool_->MapOffset(
            (char**)&request_batch, sizeof(RequestBatch),
            ipc_message_->request_batch));
    if (response_batch->has_error) {
      continue;
    }
    uint32_t batch_size = request_batch->batch_size;

    Request* requests;
    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, false /* return */,
        shm_pool_->MapOffset(
            (char**)&requests, sizeof(Request) * batch_size,
            request_batch->requests));
    if (response_batch->has_error) {
      continue;
    }

    py::list py_request_list;
    for (size_t i = 0; i < batch_size; i++) {
      Request* request = &requests[i];
      py::object infer_request;
      ChildProcessRequest(
          request, response_batch, infer_request, PyRequest, PyTensor);
      py_request_list.append(infer_request);
    }

    // If there was an error in one of the requests, skip the processing of this
    // batch.
    if (response_batch->has_error) {
      continue;
    }

    py::list responses;

    if (!py::hasattr(model_instance, "execute")) {
      std::string message =
          "Python model " + Name() + " does not implement `execute` method.";
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, message.c_str());

      off_t err_string_offset;
      LOG_IF_ERROR(
          SaveStringToSharedMemory(
              shm_pool_, err_string_offset, message.c_str()),
          "Failed to create string in shared memory");
      response_batch->has_error = true;
      response_batch->error = err_string_offset;

      continue;
    }

    // Execute Response
    try {
      responses = model_instance.attr("execute")(py_request_list);
    }
    catch (const py::error_already_set& e) {
      off_t err_string_offset;
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
      LOG_IF_ERROR(
          SaveStringToSharedMemory(shm_pool_, err_string_offset, e.what()),
          "Failed to create string in shared memory");
      response_batch->has_error = true;
      response_batch->error = err_string_offset;

      continue;
    }

    Response* responses_shm;
    off_t responses_shm_offset;
    size_t response_size = py::len(responses);

    STUB_SET_RESPONSE_ERROR_IF_ERROR(
        shm_pool_, response_batch, false /* return */,
        shm_pool_->Map(
            (char**)&responses_shm, sizeof(Response) * response_size,
            responses_shm_offset));
    if (response_batch->has_error) {
      continue;
    }
    response_batch->responses = responses_shm_offset;
    response_batch->batch_size = response_size;

    size_t i = 0;
    for (auto& response : responses) {
      Response* response_shm = &responses_shm[i];
      ChildProcessResponse(response_shm, response_batch, response);
      i += 1;
    }
  }

  // Call finalize if exists.
  if (py::hasattr(model_instance, "finalize")) {
    try {
      model_instance.attr("finalize")();
    }
    catch (const py::error_already_set& e) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    }
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetupChildProcess()
{
  std::string shm_region_name = std::string("/") + Name();
  shm_pool_ = std::make_unique<SharedMemory>(shm_region_name);

  // Child Mutex and CV
  pthread_mutex_t* child_mutex;
  off_t child_mutex_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&child_mutex, sizeof(pthread_mutex_t), child_mutex_offset));
  CreateIPCMutex(&child_mutex);

  pthread_cond_t* child_cv;
  off_t child_cv_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&child_cv, sizeof(pthread_cond_t), child_cv_offset));
  CreateIPCCondVariable(&child_cv);

  child_cond_ = child_cv;
  child_mutex_ = child_mutex;

  // Parent Mutex and CV
  pthread_mutex_t* parent_mutex;
  off_t parent_mutex_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&parent_mutex, sizeof(pthread_mutex_t), parent_mutex_offset));
  CreateIPCMutex(&parent_mutex);

  pthread_cond_t* parent_cv;
  off_t parent_cv_offset;
  RETURN_IF_ERROR(shm_pool_->Map(
      (char**)&parent_cv, sizeof(pthread_cond_t), parent_cv_offset));
  CreateIPCCondVariable(&parent_cv);

  parent_cond_ = parent_cv;
  parent_mutex_ = parent_mutex;

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  off_t ipc_offset;
  RETURN_IF_ERROR(
      shm_pool_->Map((char**)&ipc_message_, sizeof(IPCMessage), ipc_offset));

  uint64_t model_version = model_state->Version();
  const char* model_path = model_state->RepositoryPath().c_str();

  std::stringstream ss;
  // Use <path>/version/model.py as the model location
  ss << model_path << "/" << model_version;
  model_path_ = ss.str();

  pid_t pid = fork();

  if (pid < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Failed to fork the child process.");
  }

  // Child process
  if (pid == 0) {
    sleep(1);  // A small delay to allow parent process to reach the mutex.
    WaitForMessageCallback();

    // We will exit the process on return because we don't want to return a
    // response two times.
    exit(0);

    return nullptr;
  } else {
    pid_ = pid;
    pthread_mutex_lock(parent_mutex_);
    pthread_cond_wait(parent_cond_, parent_mutex_);
    pthread_mutex_unlock(parent_mutex_);

    ResponseBatch* response_batch;
    RETURN_IF_ERROR(shm_pool_->MapOffset(
        (char**)&response_batch, sizeof(RequestBatch),
        ipc_message_->response_batch));

    if (response_batch->has_error) {
      char* err_message;
      RETURN_IF_ERROR(LoadStringFromSharedMemory(
          shm_pool_, response_batch->error, err_message));
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message);
    }

    return nullptr;
  }
}

ModelInstanceState::~ModelInstanceState()
{
  int status;

  // Terminate the child process.
  kill(pid_, SIGTERM);
  waitpid(pid_, &status, 0);
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, Tensor* input_tensor,
    TRITONBACKEND_Request* request,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  const char* input_name;
  // Load iidx'th input name
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInputName(request, input_idx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;

  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_InputProperties(
                   in, &input_name, &input_dtype, &input_shape,
                   &input_dims_count, &input_byte_size, &input_buffer_count));

  // We need to create a new collector for every request because python backend
  // sends each request individually to the python model
  BackendInputCollector collector(
      &request, 1, &responses, Model()->TritonMemoryManager(),
      false /* pinned_enable */, CudaStream());


  const TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  const int memory_type_id = 0;

  char* input_buffer;
  RESPOND_AND_RETURN_IF_ERROR(
      request, SaveTensorToSharedMemory(
                   shm_pool_, input_tensor, input_buffer, memory_type,
                   memory_type_id, input_byte_size, input_name, input_shape,
                   input_dims_count, input_dtype));

  // Load raw data into input_tensor raw data.
  // FIXME: Avoid the copy to CPU Memory when
  // the data is in GPU.
  collector.ProcessTensor(
      input_name, input_buffer, input_byte_size, memory_type, memory_type_id);

  return nullptr;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));

  const char* path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));

  void* bstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  backend_state_ = reinterpret_cast<BackendState*>(bstate);

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + Name() + "'")
            .c_str()));
  }
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  // Check backend version to ensure compatibility
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor);
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState> backend_state(new BackendState());
  triton::common::TritonJson::Value cmdline;
  backend_state->python_runtime = "python3";

  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value python_runtime;
    if (cmdline.Find("python-runtime", &python_runtime)) {
      RETURN_IF_ERROR(python_runtime.AsString(&backend_state->python_runtime));
    }
  }

  // Use BackendArtifacts to determine the location of Python files
  const char* location;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  backend_state->python_lib = location;

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release();
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: Start");
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: End");
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());
  RETURN_IF_ERROR(instance_state->SetupChildProcess());

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  ModelState* model_state =
      reinterpret_cast<ModelState*>(instance_state->Model());
  int max_batch_size = model_state->MaxBatchSize();
  std::string name = model_state->Name();


  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.

  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Python backend for '" + name + "'")
                  .c_str()));
      return nullptr;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return nullptr;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return nullptr;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                name + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return nullptr;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  RETURN_IF_ERROR(instance_state->ProcessRequests(requests, request_count));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::python
