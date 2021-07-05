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
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <atomic>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>
#include "pb_tensor.h"
#include "pb_utils.h"
#include "shm_manager.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

namespace py = pybind11;
using namespace pybind11::literals;
namespace bi = boost::interprocess;

namespace triton { namespace backend { namespace python {

#define LOG_IF_EXCEPTION(X)                              \
  do {                                                   \
    try {                                                \
      (X);                                               \
    }                                                    \
    catch (const PythonBackendException& pb_exception) { \
      LOG_INFO << pb_exception.what();                   \
    }                                                    \
  } while (false)

#define LOG_EXCEPTION(E)  \
  do {                    \
    LOG_INFO << E.what(); \
  } while (false)

// Macros that use current filename and line number.
#define LOG_INFO LOG_INFO_FL(__FILE__, __LINE__)

class Logger {
 public:
  // Log a message.
  void Log(const std::string& msg) { std::cerr << msg << std::endl; }

  // Flush the log.
  void Flush() { std::cerr << std::flush; }
};

Logger gLogger_;
class LogMessage {
 public:
  LogMessage(const char* file, int line)
  {
    std::string path(file);
    size_t pos = path.rfind('/');
    if (pos != std::string::npos) {
      path = path.substr(pos + 1, std::string::npos);
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    stream_ << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1)
            << std::setw(2) << tm_time.tm_mday << " " << std::setw(2)
            << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
            << std::setw(2) << tm_time.tm_sec << "." << std::setw(6)
            << tv.tv_usec << ' ' << static_cast<uint32_t>(getpid()) << ' '
            << path << ':' << line << "] ";
  }

  ~LogMessage() { gLogger_.Log(stream_.str()); }

  std::stringstream& stream() { return stream_; }

 private:
  std::stringstream stream_;
};

#define LOG_INFO_FL(FN, LN) LogMessage((char*)(FN), LN).stream()

void
SignalHandler(int signum)
{
  // Skip the SIGINT
}

std::atomic<bool> sigterm_received{false};

void
SigtermHandler(int signum)
{
  sigterm_received = true;
}

class Stub {
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  bi::scoped_lock<bi::interprocess_mutex> stub_lock_;
  std::string model_path_;
  IPCMessage* ipc_message_;
  std::unique_ptr<SharedMemory> shm_pool_;
  py::object PyRequest_;
  py::object model_instance_;
  py::object deserialize_bytes_;
  py::object serialize_bytes_;
  ResponseBatch* response_batch_;
  std::vector<PbTensor*> tensors_to_remove_;
  bool require_cleanup_;

#ifdef TRITON_ENABLE_GPU
  std::unordered_map<void*, std::string> gpu_tensors_map_;
#endif

 public:
  Stub(
      int64_t shm_growth_size, int64_t shm_default_size,
      std::string& shm_region_name, std::string& model_path)
  {
    try {
      model_path_ = model_path;
      stub_mutex_ = nullptr;
      stub_cond_ = nullptr;
      parent_mutex_ = nullptr;
      parent_cond_ = nullptr;
      health_mutex_ = nullptr;

      // This boolean indicates whether a cleanup is required or not.
      require_cleanup_ = false;

      shm_pool_ = std::make_unique<SharedMemory>(
          shm_region_name, shm_default_size, shm_growth_size);

      // Stub mutex and CV
      bi::interprocess_mutex* stub_mutex;
      off_t stub_mutex_offset;
      shm_pool_->Map(
          (char**)&stub_mutex, sizeof(bi::interprocess_mutex),
          stub_mutex_offset);

      bi::interprocess_condition* stub_cv;
      off_t stub_cv_offset;
      shm_pool_->Map(
          (char**)&stub_cv, sizeof(bi::interprocess_condition), stub_cv_offset);

      stub_cond_ = stub_cv;
      stub_mutex_ = stub_mutex;

      // Parent Mutex and CV
      bi::interprocess_mutex* parent_mutex;
      off_t parent_mutex_offset;
      shm_pool_->Map(
          (char**)&parent_mutex, sizeof(bi::interprocess_mutex),
          parent_mutex_offset);

      bi::interprocess_condition* parent_cv;
      off_t parent_cv_offset;
      shm_pool_->Map(
          (char**)&parent_cv, sizeof(bi::interprocess_condition),
          parent_cv_offset);

      bi::interprocess_mutex* health_mutex;
      off_t health_mutex_offset;
      shm_pool_->Map(
          (char**)&health_mutex, sizeof(bi::interprocess_mutex),
          health_mutex_offset);

      health_mutex_ = health_mutex;
      parent_mutex_ = parent_mutex;
      parent_cond_ = parent_cv;

      IPCMessage* ipc_message;
      off_t ipc_offset;
      shm_pool_->Map((char**)&ipc_message, sizeof(IPCMessage), ipc_offset);

      off_t response_batch_offset;
      shm_pool_->Map(
          (char**)&response_batch_, sizeof(Response), response_batch_offset);
      ipc_message->response_batch = response_batch_offset;
      response_batch_->has_error = false;
      ipc_message_ = ipc_message;

      stub_lock_ = bi::scoped_lock<bi::interprocess_mutex>(*stub_mutex_);
      NotifyParent();
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_INFO << pb_exception.what() << std::endl;
      exit(1);
    }
  }

  void NotifyParent()
  {
    if (parent_mutex_ == nullptr || parent_cond_ == nullptr) {
      LOG_INFO << "Parent process mutex and conditional variable is not "
                  "initialized. "
               << "Exiting..";
      exit(1);
    }

    bi::scoped_lock<bi::interprocess_mutex> lk(*parent_mutex_);
    parent_cond_->notify_one();
  }

  bool& Health() { return ipc_message_->health; }

  std::unique_ptr<SharedMemory>& GetSharedMemory() { return shm_pool_; }

  void SetErrorForResponse(Response* response, const char* err_message)
  {
    off_t err_string_offset = 0;
    response->is_error_set = false;
    response->has_error = true;
    LOG_IF_EXCEPTION(
        SaveStringToSharedMemory(shm_pool_, err_string_offset, err_message));

    if (err_string_offset != 0) {
      response->error = err_string_offset;
      response->is_error_set = true;
    }
  }

  void SetErrorForResponseBatch(const char* err_message)
  {
    off_t err_string_offset = 0;
    response_batch_->is_error_set = false;
    response_batch_->has_error = true;
    LOG_IF_EXCEPTION(
        SaveStringToSharedMemory(shm_pool_, err_string_offset, err_message));

    if (err_string_offset != 0) {
      response_batch_->error = err_string_offset;
      response_batch_->is_error_set = true;
    }
  }

  void ProcessResponse(
      Response* response_shm, ResponseBatch* response_batch,
      py::handle response, py::object& serialize_bytes)
  {
    // Initialize has_error to false
    response_shm->has_error = false;

    py::bool_ py_has_error = response.attr("has_error")();
    bool has_error = py_has_error;

    if (has_error) {
      py::str py_string_err = py::str(response.attr("error")());
      std::string response_error = py_string_err;
      SetErrorForResponse(response_shm, response_error.c_str());

      // Skip the response value when the response has error.
      return;
    }

    py::list output_tensors = response.attr("output_tensors")();
    size_t output_tensor_length = py::len(output_tensors);

    size_t j = 0;
    Tensor* output_tensors_shm;
    off_t output_tensors_offset;
    shm_pool_->Map(
        (char**)&output_tensors_shm, sizeof(Tensor) * output_tensor_length,
        output_tensors_offset);
    response_shm->outputs = output_tensors_offset;
    response_shm->outputs_size = output_tensor_length;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;

    for (auto& output_tensor : output_tensors) {
      PbTensor* output_tensor_pb = output_tensor.cast<PbTensor*>();
      if (output_tensor_pb->TensorType() == PYTHONBACKEND_DLPACK) {
        require_cleanup_ = true;
        tensors_to_remove_.push_back(output_tensor_pb);
      }
      Tensor* output_tensor_shm = &output_tensors_shm[j];
      output_tensor_shm->is_reused = false;
      std::string output_name = output_tensor_pb->Name();
      TRITONSERVER_DataType dtype_triton =
          static_cast<TRITONSERVER_DataType>(output_tensor_pb->TritonDtype());
      memory_type = TRITONSERVER_MEMORY_CPU;
      memory_type_id = 0;

      if (output_tensor_pb->IsCPU()) {
        py::array numpy_array = output_tensor_pb->AsNumpy();
        py::buffer_info buffer = numpy_array.request();
        size_t dims_count = numpy_array.ndim();
        memory_type = TRITONSERVER_MEMORY_CPU;
        memory_type_id = 0;

        char* data_in_shm;
        char* data_ptr;

        int64_t dims[dims_count];
        ssize_t byte_size;

        // Custom handling for type bytes.
        if (dtype_triton == TRITONSERVER_TYPE_BYTES) {
          py::object serialized_bytes_or_none = serialize_bytes(numpy_array);
          if (serialize_bytes.is_none()) {
            const char* err_message = "An error happened during serialization.";
            LOG_INFO << err_message;
            SetErrorForResponse(response_shm, err_message);
            break;
          }

          py::bytes serialized_bytes = serialized_bytes_or_none;
          data_ptr = PyBytes_AsString(serialized_bytes.ptr());
          byte_size = PyBytes_Size(serialized_bytes.ptr());
        } else {
          data_ptr = static_cast<char*>(buffer.ptr);
          byte_size = numpy_array.nbytes();
        }

        const ssize_t* numpy_shape = numpy_array.shape();
        for (size_t i = 0; i < dims_count; i++) {
          dims[i] = numpy_shape[i];
        }

        uint64_t* ptr_offset;
        SaveTensorToSharedMemory(
            shm_pool_, output_tensor_shm, data_in_shm, memory_type,
            memory_type_id, byte_size, output_name.c_str(), dims, dims_count,
            dtype_triton, &ptr_offset);
        *ptr_offset = 0;

        // TODO: We can remove this memcpy if the numpy object
        // is already in shared memory.
        std::copy(data_ptr, data_ptr + byte_size, data_in_shm);
      } else {
#ifdef TRITON_ENABLE_GPU
        char* cuda_handle;
        uint64_t* ptr_offset;
        SaveTensorToSharedMemory(
            shm_pool_, output_tensor_shm, cuda_handle,
            output_tensor_pb->MemoryType(), output_tensor_pb->MemoryTypeId(),
            output_tensor_pb->ByteSize(), output_name.c_str(),
            output_tensor_pb->Dims().data(), output_tensor_pb->Dims().size(),
            dtype_triton, &ptr_offset);

        std::unordered_map<void*, std::string>::const_iterator
            reused_gpu_tensor =
                gpu_tensors_map_.find(output_tensor_pb->GetDataPtr());
        if (reused_gpu_tensor == gpu_tensors_map_.end()) {
          cudaSetDevice(output_tensor_pb->MemoryTypeId());
          cudaError_t err = cudaIpcGetMemHandle(
              reinterpret_cast<cudaIpcMemHandle_t*>(cuda_handle),
              output_tensor_pb->GetDataPtr());
          if (err != cudaSuccess) {
            throw PythonBackendException(
                std::string(
                    "failed to get cuda ipc handle: " +
                    std::string(cudaGetErrorString(err)))
                    .c_str());
          }
        } else {
          output_tensor_shm->is_reused = true;
          off_t reused_tensor_name_offset;
          SaveStringToSharedMemory(
              shm_pool_, reused_tensor_name_offset,
              reused_gpu_tensor->second.c_str());
          output_tensor_shm->reused_tensor_name = reused_tensor_name_offset;
        }

        CUdeviceptr start_address;
        CUresult cuda_err = cuPointerGetAttribute(
            &start_address, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            (CUdeviceptr)output_tensor_pb->GetDataPtr());
        if (cuda_err != CUDA_SUCCESS) {
          const char* error_string;
          cuGetErrorString(cuda_err, &error_string);
          throw PythonBackendException(
              std::string(
                  "failed to get cuda pointer device attribute: " +
                  std::string(error_string))
                  .c_str());
        }
        *ptr_offset = reinterpret_cast<char*>(output_tensor_pb->GetDataPtr()) -
                      reinterpret_cast<char*>(start_address);
#else
        throw PythonBackendException(
            "Python backend was not built with GPU tensor support");
#endif
      }
      j += 1;
    }
  }

  void ProcessRequest(
      Request* request, ResponseBatch* response_batch,
      py::object& infer_request, py::object& PyRequest,
      py::object& deserialize_bytes)
  {
    char* id = nullptr;
    LoadStringFromSharedMemory(shm_pool_, request->id, id);

    uint32_t requested_input_count = request->requested_input_count;
    Tensor* input_tensors;
    shm_pool_->MapOffset(
        (char**)&input_tensors, sizeof(Tensor) * requested_input_count,
        request->inputs);

    py::list py_input_tensors;
    for (size_t input_idx = 0; input_idx < requested_input_count; ++input_idx) {
      Tensor* input_tensor = &input_tensors[input_idx];

      char* name = nullptr;
      LoadStringFromSharedMemory(shm_pool_, input_tensor->name, name);
      size_t dims_count = input_tensor->dims_count;

      RawData* raw_data;
      shm_pool_->MapOffset(
          (char**)&raw_data, sizeof(RawData), input_tensor->raw_data);

      int64_t* dims;
      shm_pool_->MapOffset(
          (char**)&dims, sizeof(int64_t) * dims_count, input_tensor->dims);

      TRITONSERVER_DataType dtype = input_tensor->dtype;
      std::vector<int64_t> shape{dims, dims + dims_count};

      char* data;
      if (raw_data->memory_type == TRITONSERVER_MEMORY_CPU) {
        shm_pool_->MapOffset(
            (char**)&data, raw_data->byte_size, raw_data->memory_ptr);

        py::dtype dtype_numpy;
        switch (dtype) {
          case TRITONSERVER_TYPE_BOOL:
            dtype_numpy = py::dtype(py::format_descriptor<bool>::format());
            break;
          case TRITONSERVER_TYPE_UINT8:
            dtype_numpy = py::dtype(py::format_descriptor<uint8_t>::format());
            break;
          case TRITONSERVER_TYPE_UINT16:
            dtype_numpy = py::dtype(py::format_descriptor<uint16_t>::format());
            break;
          case TRITONSERVER_TYPE_UINT32:
            dtype_numpy = py::dtype(py::format_descriptor<uint32_t>::format());
            break;
          case TRITONSERVER_TYPE_UINT64:
            dtype_numpy = py::dtype(py::format_descriptor<uint64_t>::format());
            break;
          case TRITONSERVER_TYPE_INT8:
            dtype_numpy = py::dtype(py::format_descriptor<int8_t>::format());
            break;
          case TRITONSERVER_TYPE_INT16:
            dtype_numpy = py::dtype(py::format_descriptor<int16_t>::format());
            break;
          case TRITONSERVER_TYPE_INT32:
            dtype_numpy = py::dtype(py::format_descriptor<int32_t>::format());
            break;
          case TRITONSERVER_TYPE_INT64:
            dtype_numpy = py::dtype(py::format_descriptor<int64_t>::format());
            break;
          case TRITONSERVER_TYPE_FP16:
            // Will be reinterpreted in the python code.
            dtype_numpy = py::dtype(py::format_descriptor<uint16_t>::format());
            break;
          case TRITONSERVER_TYPE_FP32:
            dtype_numpy = py::dtype(py::format_descriptor<float>::format());
            break;
          case TRITONSERVER_TYPE_FP64:
            dtype_numpy = py::dtype(py::format_descriptor<double>::format());
            break;
          case TRITONSERVER_TYPE_BYTES:
            // Will be reinterpreted in the python code.
            dtype_numpy = py::dtype(py::format_descriptor<uint8_t>::format());
            break;
          default:
            break;
        }
        std::string name_str(name);

        try {
          // Custom handling for bytes
          if (dtype == TRITONSERVER_TYPE_BYTES) {
            py::array numpy_array(
                dtype_numpy, {raw_data->byte_size}, (void*)data);
            py::list dims = py::cast(shape);

            py::object deserialized =
                deserialize_bytes(numpy_array).attr("reshape")(dims);

            PbTensor py_input_tensor(name_str, deserialized, dtype);
            py_input_tensors.append(py_input_tensor);
          } else {
            py::array numpy_array(dtype_numpy, shape, (void*)data);
            PbTensor py_input_tensor(name_str, numpy_array, dtype);
            py_input_tensors.append(py_input_tensor);
          }
        }
        catch (const py::error_already_set& e) {
          LOG_INFO << e.what();
          throw PythonBackendException(e.what());
          return;
        }
      } else if (raw_data->memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
        cudaIpcMemHandle_t* cuda_mem_handle;
        cudaSetDevice(raw_data->memory_type_id);
        shm_pool_->MapOffset(
            (char**)&cuda_mem_handle, sizeof(cudaIpcMemHandle_t),
            raw_data->memory_ptr);

        cudaError_t err = cudaIpcOpenMemHandle(
            (void**)&data, *cuda_mem_handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
          throw PythonBackendException(std::string(
                                           "failed to open cuda ipc handle: " +
                                           std::string(cudaGetErrorString(err)))
                                           .c_str());
        }

        // Clean up is required to remove the GPU tensors
        require_cleanup_ = true;
        gpu_tensors_map_.insert({static_cast<void*>(data), name});

        // Adjust the device pointer with the offset. cudaIpcOpenMemHandle will
        // map the base address only and the offset needs to be manually
        // adjusted.
        data = data + raw_data->offset;

        PbTensor py_input_tensor = PbTensor(
            name, shape, dtype, raw_data->memory_type, raw_data->memory_type_id,
            static_cast<void*>(data), raw_data->byte_size, nullptr);

        py_input_tensors.append(py_input_tensor);
#else
        throw PythonBackendException(
            "Python backend was not compiled with GPU tensor support.");
#endif
      }
    }

    py::list py_requested_output_names;
    uint32_t requested_output_count = request->requested_output_count;
    off_t* output_names;
    shm_pool_->MapOffset(
        (char**)&output_names, sizeof(off_t) * requested_output_count,
        request->requested_output_names);

    for (size_t output_idx = 0; output_idx < requested_output_count;
         ++output_idx) {
      char* output_name = nullptr;
      LoadStringFromSharedMemory(
          shm_pool_, output_names[output_idx], output_name);
      py_requested_output_names.append(output_name);
    }

    infer_request = PyRequest(
        py_input_tensors, id, request->correlation_id,
        py_requested_output_names);
  }

  void SetResponseFromException(const PythonBackendException& pb_exception)
  {
    SetErrorForResponseBatch(pb_exception.what());
  }


  int Execute()
  {
    // Reset the value for has_error
    response_batch_->has_error = false;

    // Reset the cleanup value
    response_batch_->cleanup = false;
    require_cleanup_ = false;

    RequestBatch* request_batch;
    try {
      shm_pool_->MapOffset(
          (char**)&request_batch, sizeof(RequestBatch),
          ipc_message_->request_batch);
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_EXCEPTION(pb_exception);
      SetResponseFromException(pb_exception);
      return 0;
    }
    uint32_t batch_size = request_batch->batch_size;

    // An empty batch size indicates termination
    if (batch_size == 0) {
      return 1;
    }

    Request* requests;
    try {
      shm_pool_->MapOffset(
          (char**)&requests, sizeof(Request) * batch_size,
          request_batch->requests);
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_EXCEPTION(pb_exception);
      SetResponseFromException(pb_exception);
      return 0;
    }

    py::list py_request_list;
    for (size_t i = 0; i < batch_size; i++) {
      Request* request = &requests[i];
      py::object infer_request;
      try {
        ProcessRequest(
            request, response_batch_, infer_request, PyRequest_,
            deserialize_bytes_);
      }
      catch (const PythonBackendException& pb_exception) {
        LOG_EXCEPTION(pb_exception);
        SetResponseFromException(pb_exception);
        return 0;
      }
      py_request_list.append(infer_request);
    }

    py::list responses;

    if (!py::hasattr(model_instance_, "execute")) {
      std::string message = "Python model " + model_path_ +
                            " does not implement `execute` method.";
      LOG_INFO << message;
      SetErrorForResponseBatch(message.c_str());

      return 0;
    }

    // Execute Response
    try {
      responses = model_instance_.attr("execute")(py_request_list);
    }
    catch (const py::error_already_set& e) {
      LOG_INFO << e.what();
      SetErrorForResponseBatch(e.what());

      return 0;
    }

    Response* responses_shm;
    off_t responses_shm_offset;
    size_t response_size = py::len(responses);

    try {
      shm_pool_->Map(
          (char**)&responses_shm, sizeof(Response) * response_size,
          responses_shm_offset);
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_EXCEPTION(pb_exception);
      SetResponseFromException(pb_exception);
      return 0;
    }
    response_batch_->responses = responses_shm_offset;
    response_batch_->batch_size = response_size;

    size_t i = 0;
    for (auto& response : responses) {
      Response* response_shm = &responses_shm[i];
      try {
        ProcessResponse(
            response_shm, response_batch_, response, serialize_bytes_);
      }
      catch (const PythonBackendException& pb_exception) {
        LOG_EXCEPTION(pb_exception);
        SetErrorForResponse(response_shm, pb_exception.what());
      }
      i += 1;
    }

    // When require_cleanup_ is true it needs one additional step for cleaning
    // up the tensors on the client side.
    if (require_cleanup_) {
      response_batch_->cleanup = true;
      NotifyParent();

      // Exit if it has received a SIGTERM signal.
      if (WaitForNotification()) {
        LOG_INFO << "Received SIGTERM: exiting.";
        return 1;
      }

      for (PbTensor*& pb_tensor : tensors_to_remove_) {
        pb_tensor->DeleteDLPack();
      }

      tensors_to_remove_.clear();

#ifdef TRITON_ENABLE_GPU
      for (const auto& gpu_tensor_pair : gpu_tensors_map_) {
        cudaError_t err = cudaIpcCloseMemHandle(gpu_tensor_pair.first);
        if (err != cudaSuccess) {
          LOG_INFO << std::string(
              "failed to close CUDA IPC handle:" +
              std::string(cudaGetErrorString(err)));
        }
      }
      gpu_tensors_map_.clear();
#endif
    }

    return 0;
  }

  void Initialize(std::string& model_version, std::string triton_install_path)
  {
    try {
      try {
        py::module sys = py::module::import("sys");

        std::string model_name =
            model_path_.substr(model_path_.find_last_of("/") + 1);
        std::string model_path_parent =
            model_path_.substr(0, model_path_.find_last_of("/"));
        std::string model_path_parent_parent =
            model_path_parent.substr(0, model_path_parent.find_last_of("/"));
        std::string python_backend_folder = triton_install_path;
        sys.attr("path").attr("append")(model_path_parent);
        sys.attr("path").attr("append")(model_path_parent_parent);
        sys.attr("path").attr("append")(python_backend_folder);

        py::module python_backend_utils =
            py::module::import("triton_python_backend_utils");
        py::module c_python_backend_utils =
            py::module::import("c_python_backend_utils");
        py::setattr(
            python_backend_utils, "Tensor",
            c_python_backend_utils.attr("Tensor"));

        py::object TritonPythonModel =
            py::module::import((model_version + std::string(".model")).c_str())
                .attr("TritonPythonModel");
        PyRequest_ = python_backend_utils.attr("InferenceRequest");
        deserialize_bytes_ =
            python_backend_utils.attr("deserialize_bytes_tensor");
        serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");
        model_instance_ = TritonPythonModel();

        std::unordered_map<std::string, std::string> map;
        LoadMapFromSharedMemory(shm_pool_, ipc_message_->request_batch, map);
        py::dict model_config_params;

        for (const auto& pair : map) {
          model_config_params[pair.first.c_str()] = pair.second;
        }
        // Call initialize if exists.
        if (py::hasattr(model_instance_, "initialize")) {
          model_instance_.attr("initialize")(model_config_params);
        }
      }

      catch (const py::error_already_set& e) {
        LOG_INFO << e.what();
        SetErrorForResponseBatch(e.what());

        NotifyParent();
        exit(1);
      }
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_INFO << "Failed to initialize Python stub: " << pb_exception.what();
      NotifyParent();
      exit(1);
    }
  }

  void UpdateHealth()
  {
    bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
    ipc_message_->health = true;
  }

  void Finalize()
  {
    // Call finalize if exists.
    if (py::hasattr(model_instance_, "finalize")) {
      try {
        model_instance_.attr("finalize")();
      }
      catch (const py::error_already_set& e) {
        LOG_INFO << e.what();
      }
    }
  }

  // Wait for notification from the server. Returns true if the parent process
  // has received a SIGTERM, and false otherwise.
  bool WaitForNotification()
  {
    boost::posix_time::ptime timeout;
    do {
      timeout =
          boost::get_system_time() + boost::posix_time::milliseconds(1000);
    } while (!stub_cond_->timed_wait(stub_lock_, timeout) != 0 &&
             !sigterm_received);
    return sigterm_received;
  }
};

PYBIND11_EMBEDDED_MODULE(c_python_backend_utils, module)
{
  py::class_<PbTensor>(module, "Tensor")
      .def(py::init<std::string&, py::array&>())
      .def("name", &PbTensor::Name)
      .def("as_numpy", &PbTensor::AsNumpy)
      .def("triton_dtype", &PbTensor::TritonDtype)
      .def("to_dlpack", &PbTensor::ToDLPack)
      .def("is_cpu", &PbTensor::IsCPU)
      .def("from_dlpack", &PbTensor::FromDLPack);

  py::register_exception<PythonBackendException>(
      module, "PythonBackendException");
}

extern "C" {

int
main(int argc, char** argv)
{
  if (argc < 7) {
    LOG_INFO << "Expected 7 arguments, found " << argc << " arguments.";
    exit(1);
  }
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SigtermHandler);

  // Path to model.py
  std::string model_path = argv[1];
  std::string shm_region_name = argv[2];
  int64_t shm_default_size = std::stoi(argv[3]);

  std::vector<std::string> model_path_tokens;

  // Find the package name from model path.
  size_t prev = 0, pos = 0;
  do {
    pos = model_path.find("/", prev);
    if (pos == std::string::npos)
      pos = model_path.length();
    std::string token = model_path.substr(prev, pos - prev);
    if (!token.empty())
      model_path_tokens.push_back(token);
    prev = pos + 1;
  } while (pos < model_path.length() && prev < model_path.length());

  if (model_path_tokens.size() < 2) {
    LOG_INFO << "Model path does not look right: " << model_path;
    exit(1);
  }
  std::string model_version = model_path_tokens[model_path_tokens.size() - 2];
  int64_t shm_growth_size = std::stoi(argv[4]);
  pid_t parent_pid = std::stoi(argv[5]);
  std::string triton_install_path = argv[6];

  std::unique_ptr<Stub> stub;
  try {
    stub = std::make_unique<Stub>(
        shm_growth_size, shm_default_size, shm_region_name, model_path);
  }
  catch (const PythonBackendException& pb_exception) {
    LOG_INFO << "Failed to preinitialize Python stub: " << pb_exception.what();
    exit(1);
  }

  // Exit if it has received a SIGTERM signal.
  if (stub->WaitForNotification()) {
    LOG_INFO << "Received SIGTERM: exiting.";
    exit(1);
  }

  // Start the Python Interpreter
  py::scoped_interpreter guard{};
<<<<<<< HEAD

=======
>>>>>>> Fix deadlock when init takes long
  std::atomic<bool> non_graceful_exit = {false};
  std::atomic<bool> background_thread_running = {true};
  std::thread background_thread(
      [&parent_pid, &background_thread_running, &stub, &non_graceful_exit] {
        while (background_thread_running) {
          // Every 300ms set the health variable to true. This variable is in
          // shared memory and will be set to false by the parent process.
          // The parent process expects that the stub process sets this
          // variable to true within 1 second.
          sleep(0.3);

          stub->UpdateHealth();
          if (sigterm_received) {
            background_thread_running = false;
          }

          if (kill(parent_pid, 0) != 0) {
            // Destroy Stub
            stub.reset();
            LOG_INFO << "Non-graceful termination detected. ";
            background_thread_running = false;
            non_graceful_exit = true;
            sigterm_received = true;
          }
        }
      });

  // Initialize needs to be called after the background health thread
  // has started. Otherwise, if the initialize takes long time, the
  // main process wrongly assumes that the stub process has crashed.
  stub->Initialize(model_version, argv[6] /* triton install path */);

  stub->Initialize(model_version, argv[6] /* triton install path */);
  // Wait for messages from the parent process
  while (true) {
    stub->NotifyParent();
    if (stub->WaitForNotification()) {
      break;
    }

    int stop = stub->Execute();
    if (stop)
      break;
  }

  if (!non_graceful_exit) {
    stub->Finalize();
    stub->NotifyParent();
  }

  background_thread_running = false;
  background_thread.join();
  return 0;
}
}
}}}  // namespace triton::backend::python
