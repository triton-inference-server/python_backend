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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>
#include "infer_request.h"
#include "infer_response.h"
#include "pb_tensor.h"
#include "pb_utils.h"
#include "shm_manager.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "pb_stub.h"

namespace py = pybind11;
using namespace pybind11::literals;
namespace bi = boost::interprocess;

namespace triton { namespace backend {
namespace python {

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
std::atomic<bool> non_graceful_exit = {false};


void
SignalHandler(int signum)
{
  // Skip the SIGINT and SIGTERM
}

void
Stub::Instantiate(
    int64_t shm_growth_size, int64_t shm_default_size,
    std::string& shm_region_name, const std::string& model_path,
    const std::string& model_version, const std::string& triton_install_path,
    off_t ipc_control_offset)
{
  model_path_ = model_path;
  model_version_ = model_version;
  triton_install_path_ = triton_install_path;

  stub_mutex_ = nullptr;
  stub_cond_ = nullptr;
  parent_mutex_ = nullptr;
  parent_cond_ = nullptr;
  health_mutex_ = nullptr;

  initialized_ = false;
  try {
    // This boolean indicates whether a cleanup is required or not.
    require_cleanup_ = false;

    shm_pool_ = std::make_unique<SharedMemory>(
        shm_region_name, shm_default_size, shm_growth_size);

    // Stub mutex and CV
    shm_pool_->MapOffset((char**)&ipc_control_, ipc_control_offset);
    shm_pool_->MapOffset((char**)&stub_mutex_, ipc_control_->stub_mutex);
    shm_pool_->MapOffset((char**)&stub_cond_, ipc_control_->stub_cond);

    // Parent Mutex and CV
    shm_pool_->MapOffset((char**)&parent_mutex_, ipc_control_->parent_mutex);
    shm_pool_->MapOffset((char**)&parent_cond_, ipc_control_->parent_cond);

    bi::interprocess_mutex* health_mutex;
    shm_pool_->MapOffset(
        (char**)&health_mutex, ipc_control_->stub_health_mutex);
    health_mutex_ = health_mutex;

    shm_pool_->MapOffset((char**)&ipc_message_, ipc_control_->ipc_message);
    stub_lock_ =
        std::make_unique<bi::scoped_lock<bi::interprocess_mutex>>(*stub_mutex_);

    // If the Python model is using an execution environment, we need to
    // remove the first part of the LD_LIBRARY_PATH before the colon (i.e.
    // <Python Shared Lib>:$OLD_LD_LIBRARY_PATH). The <Python Shared Lib>
    // section was added before launching the stub process and it may
    // interfere with the shared library resolution of other executable and
    // binaries.
    if (ipc_control_->uses_env) {
      char* ld_library_path = std::getenv("LD_LIBRARY_PATH");

      if (ld_library_path != nullptr) {
        std::string ld_library_path_str = ld_library_path;
        // If we use an Execute Environment, the path must contain a colon.
        size_t find_pos = ld_library_path_str.find(':');
        if (find_pos == std::string::npos) {
          throw PythonBackendException(
              "LD_LIBRARY_PATH must contain a colon when passing an "
              "execution environment.");
        }
        ld_library_path_str = ld_library_path_str.substr(find_pos + 1);
        int status = setenv(
            "LD_LIBRARY_PATH", const_cast<char*>(ld_library_path_str.c_str()),
            1 /* overwrite */);
        if (status != 0) {
          throw PythonBackendException(
              "Failed to correct the LD_LIBRARY_PATH environment in the "
              "Python backend stub.");
        }
      } else {
        throw PythonBackendException(
            "When using an execution environment, LD_LIBRARY_PATH variable "
            "cannot be empty.");
      }
    }
  }
  catch (const PythonBackendException& pb_exception) {
    LOG_INFO << pb_exception.what() << std::endl;
    exit(1);
  }
}
catch (const PythonBackendException& pb_exception)
{
  LOG_INFO << pb_exception.what() << std::endl;
  exit(1);
}
}  // namespace python

void
Stub::NotifyParent()
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

bool&
Stub::Health()
{
  return ipc_control_->stub_health;
}

std::unique_ptr<SharedMemory>&
Stub::GetSharedMemory()
{
  return shm_pool_;
}

void
Stub::SetErrorForResponse(Response* response, const char* err_message)
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

void
Stub::SetErrorForResponseBatch(
    ResponseBatch* response_batch, const char* err_message)
{
  off_t err_string_offset = 0;
  response_batch->is_error_set = false;
  response_batch->has_error = true;
  LOG_IF_EXCEPTION(
      SaveStringToSharedMemory(shm_pool_, err_string_offset, err_message));

  if (err_string_offset != 0) {
    response_batch->error = err_string_offset;
    response_batch->is_error_set = true;
  }
}

void
Stub::ProcessResponse(
    Response* response_shm, ResponseBatch* response_batch,
    InferResponse* response, py::object& serialize_bytes)
{
  // Initialize has_error to false
  response_shm->has_error = false;

  bool has_error = response->HasError();

  if (has_error) {
    py::str py_string_err = response->Error();
    std::string response_error = py_string_err;
    SetErrorForResponse(response_shm, response_error.c_str());

    // Skip the response value when the response has error.
    return;
  }

  const std::vector<std::shared_ptr<PbTensor>>& output_tensors =
      response->OutputTensors();
  size_t output_tensor_length = output_tensors.size();

  size_t j = 0;
  Tensor* output_tensors_shm;
  off_t output_tensors_offset;
  shm_pool_->Map(
      (char**)&output_tensors_shm, sizeof(Tensor) * output_tensor_length,
      output_tensors_offset);
  response_shm->outputs = output_tensors_offset;
  response_shm->outputs_size = output_tensor_length;

  for (auto& output_tensor : output_tensors) {
    if (output_tensor->TensorType() == PYTHONBACKEND_DLPACK) {
      require_cleanup_ = true;
      tensors_to_remove_.push_back(output_tensor);
    }

    Tensor* output_tensor_shm = &output_tensors_shm[j];
    if (output_tensor->IsCPU()) {
      output_tensor->SaveToSharedMemory(
          this->GetSharedMemory(), output_tensor_shm);
    } else {
      char* d_ptr = reinterpret_cast<char*>(output_tensor->GetDataPtr());
      size_t ptr_offset = GetDevicePointerOffset(d_ptr);
      char* base_addr = d_ptr - ptr_offset;
      std::unordered_map<void*, std::string>::const_iterator reused_gpu_tensor =
          gpu_tensors_map_.find(base_addr);
      if (reused_gpu_tensor == gpu_tensors_map_.end()) {
        output_tensor->SaveToSharedMemory(
            this->GetSharedMemory(), output_tensor_shm);
      } else {
        output_tensor->SaveReusedGPUTensorToSharedMemory(
            this->GetSharedMemory(), output_tensor_shm,
            reused_gpu_tensor->second);
      }
    }

    j += 1;
  }
}

InferRequest
Stub::ProcessRequest(
    Request* request, ResponseBatch* response_batch,
    py::object& deserialize_bytes)
{
  char* id = nullptr;
  LoadStringFromSharedMemory(shm_pool_, request->id, id);

  uint32_t requested_input_count = request->requested_input_count;
  Tensor* input_tensors;
  shm_pool_->MapOffset((char**)&input_tensors, request->inputs);

  std::vector<std::shared_ptr<PbTensor>> py_input_tensors;
  for (size_t input_idx = 0; input_idx < requested_input_count; ++input_idx) {
    Tensor* input_tensor = &input_tensors[input_idx];

    char* name = nullptr;
    LoadStringFromSharedMemory(shm_pool_, input_tensor->name, name);
    size_t dims_count = input_tensor->dims_count;

    RawData* raw_data;
    shm_pool_->MapOffset((char**)&raw_data, input_tensor->raw_data);

    int64_t* dims;
    shm_pool_->MapOffset((char**)&dims, input_tensor->dims);

    TRITONSERVER_DataType dtype = input_tensor->dtype;
    std::vector<int64_t> shape{dims, dims + dims_count};

    char* data;
    if (raw_data->memory_type == TRITONSERVER_MEMORY_CPU) {
      shm_pool_->MapOffset((char**)&data, raw_data->memory_ptr);

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

          py_input_tensors.emplace_back(
              std::make_shared<PbTensor>(name_str, deserialized, dtype));
        } else {
          py::array numpy_array(dtype_numpy, shape, (void*)data);
          py_input_tensors.emplace_back(
              std::make_shared<PbTensor>(name_str, numpy_array, dtype));
        }
      }
      catch (const py::error_already_set& e) {
        LOG_INFO << e.what();
        throw PythonBackendException(e.what());
      }
    } else if (raw_data->memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
      cudaIpcMemHandle_t* cuda_mem_handle;
      cudaSetDevice(raw_data->memory_type_id);
      shm_pool_->MapOffset((char**)&cuda_mem_handle, raw_data->memory_ptr);

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

      size_t ptr_offset = GetDevicePointerOffset(static_cast<void*>(data));
      gpu_tensors_map_.insert(
          {reinterpret_cast<void*>(static_cast<char*>(data) - ptr_offset),
           name});

      // Adjust the device pointer with the offset. cudaIpcOpenMemHandle will
      // map the base address only and the offset needs to be manually
      // adjusted.
      data = data + raw_data->offset;

      py_input_tensors.emplace_back(std::make_shared<PbTensor>(
          name, shape, dtype, raw_data->memory_type, raw_data->memory_type_id,
          static_cast<void*>(data), raw_data->byte_size, nullptr));
#else
      throw PythonBackendException(
          "Python backend was not compiled with GPU tensor support.");
#endif
    }
  }

  std::vector<std::string> requested_output_names;
  uint32_t requested_output_count = request->requested_output_count;
  off_t* output_names;
  shm_pool_->MapOffset((char**)&output_names, request->requested_output_names);

  for (size_t output_idx = 0; output_idx < requested_output_count;
       ++output_idx) {
    char* output_name = nullptr;
    LoadStringFromSharedMemory(
        shm_pool_, output_names[output_idx], output_name);
    requested_output_names.emplace_back(output_name);
  }

  char* model_name;
  LoadStringFromSharedMemory(shm_pool_, request->model_name, model_name);

  return InferRequest(
      id, request->correlation_id, py_input_tensors, requested_output_names,
      model_name, request->model_version);
}

void
Stub::SetResponseFromException(
    ResponseBatch* response_batch, const PythonBackendException& pb_exception)
{
  SetErrorForResponseBatch(response_batch, pb_exception.what());
}

bool
Stub::RunCommand()
{
  switch (ipc_message_->command) {
    case PYTHONSTUB_CommandType::PYTHONSTUB_Initialize: {
      InitializeArgs* initialize_args;
      bool has_exception = false;
      std::string error_string;
      shm_pool_->MapOffset((char**)&initialize_args, ipc_message_->args);
      initialize_args->response_has_error = false;
      try {
        Initialize(initialize_args);
      }
      catch (const PythonBackendException& pb_exception) {
        has_exception = true;
        error_string = pb_exception.what();
      }
      catch (const py::error_already_set& error) {
        has_exception = true;
        error_string = error.what();
      }

      if (has_exception) {
        LOG_INFO << "Failed to initialize Python stub: " << error_string;
        initialize_args->response_has_error = true;
        initialize_args->response_is_error_set = false;
        off_t err_string_offset;
        LOG_IF_EXCEPTION(SaveStringToSharedMemory(
            shm_pool_, err_string_offset, error_string.c_str()));
        if (err_string_offset != 0) {
          initialize_args->response_is_error_set = true;
          initialize_args->response_error = err_string_offset;
        }
        return true;
      }
    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_Execute: {
      ExecuteArgs* execute_args;
      bool has_exception = false;
      std::string error_string;

      shm_pool_->MapOffset((char**)&execute_args, ipc_message_->args);
      ResponseBatch* response_batch;
      shm_pool_->Map(
          (char**)&response_batch, sizeof(ResponseBatch),
          execute_args->response_batch);
      response_batch->has_error = false;
      try {
        Execute(execute_args, response_batch);
      }
      catch (const PythonBackendException& pb_exception) {
        has_exception = true;
        error_string = pb_exception.what();
      }
      catch (const py::error_already_set& error) {
        has_exception = true;
        error_string = error.what();
      }

      if (has_exception) {
        LOG_INFO << "Failed to execute request batch: " << error_string;
        response_batch->has_error = true;
        response_batch->is_error_set = false;
        off_t err_string_offset;
        LOG_IF_EXCEPTION(SaveStringToSharedMemory(
            shm_pool_, err_string_offset, error_string.c_str()));
        if (err_string_offset != 0) {
          response_batch->is_error_set = true;
          response_batch->error = err_string_offset;
        }
      }
    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_Finalize:
      return true;
    case PYTHONSTUB_CommandType::PYTHONSTUB_TensorCleanup:
      Cleanup();
      break;
    default:
      break;
  }

  return false;
}

void
Stub::Execute(ExecuteArgs* execute_args, ResponseBatch* response_batch)
{
  response_batch->cleanup = false;
  require_cleanup_ = false;

  RequestBatch* request_batch;
  shm_pool_->MapOffset((char**)&request_batch, execute_args->request_batch);
  uint32_t batch_size = request_batch->batch_size;

  if (batch_size == 0) {
    return;
  }

  Request* requests;
  shm_pool_->MapOffset((char**)&requests, request_batch->requests);

  py::list py_request_list;
  for (size_t i = 0; i < batch_size; i++) {
    Request* request = &requests[i];
    py_request_list.append(
        ProcessRequest(request, response_batch, deserialize_bytes_));
  }

  if (!py::hasattr(model_instance_, "execute")) {
    std::string message =
        "Python model " + model_path_ + " does not implement `execute` method.";
    throw PythonBackendException(message);
  }
  py::object request_list = py_request_list;

  // Execute Response
  py::list responses = model_instance_.attr("execute")(request_list);

  Response* responses_shm;
  off_t responses_shm_offset;
  size_t response_size = py::len(responses);

  // If the number of request objects do not match the number of resposne
  // objects throw an error.
  if (response_size != batch_size) {
    throw PythonBackendException(
        "Number of InferenceResponse objects do not match the number of "
        "InferenceRequest objects.");
  }

  shm_pool_->Map(
      (char**)&responses_shm, sizeof(Response) * response_size,
      responses_shm_offset);
  response_batch->responses = responses_shm_offset;
  response_batch->batch_size = response_size;

  size_t i = 0;
  for (auto& response : responses) {
    InferResponse* infer_response = response.cast<InferResponse*>();
    Response* response_shm = &responses_shm[i];
    ProcessResponse(
        response_shm, response_batch, infer_response, serialize_bytes_);
    i += 1;
  }
}

void
Stub::Initialize(InitializeArgs* initialize_args)
{
  py::module sys = py::module::import("sys");

  std::string model_name =
      model_path_.substr(model_path_.find_last_of("/") + 1);
  std::string model_path_parent =
      model_path_.substr(0, model_path_.find_last_of("/"));
  std::string model_path_parent_parent =
      model_path_parent.substr(0, model_path_parent.find_last_of("/"));
  std::string python_backend_folder = triton_install_path_;
  sys.attr("path").attr("append")(model_path_parent);
  sys.attr("path").attr("append")(model_path_parent_parent);
  sys.attr("path").attr("append")(python_backend_folder);

  py::module python_backend_utils =
      py::module::import("triton_python_backend_utils");
  py::module c_python_backend_utils =
      py::module::import("c_python_backend_utils");
  py::setattr(
      python_backend_utils, "Tensor", c_python_backend_utils.attr("Tensor"));
  py::setattr(
      python_backend_utils, "InferenceRequest",
      c_python_backend_utils.attr("InferenceRequest"));
  py::setattr(
      python_backend_utils, "InferenceResponse",
      c_python_backend_utils.attr("InferenceResponse"));

  py::object TritonPythonModel =
      py::module::import((model_version_ + std::string(".model")).c_str())
          .attr("TritonPythonModel");
  deserialize_bytes_ = python_backend_utils.attr("deserialize_bytes_tensor");
  serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");
  model_instance_ = TritonPythonModel();

  std::unordered_map<std::string, std::string> map;
  LoadMapFromSharedMemory(shm_pool_, initialize_args->args, map);
  py::dict model_config_params;

  for (const auto& pair : map) {
    model_config_params[pair.first.c_str()] = pair.second;
  }

  // Call initialize if exists.
  if (py::hasattr(model_instance_, "initialize")) {
    model_instance_.attr("initialize")(model_config_params);
  }

  initialized_ = true;
}

void
Stub::UpdateHealth()
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
  ipc_control_->stub_health = true;
}

void
Stub::Cleanup()
{
  // Deleting the tensors should automatically trigger the destructor.
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

void
Stub::Finalize()
{
  // Call finalize if exists.
  if (initialized_ && py::hasattr(model_instance_, "finalize")) {
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
bool
Stub::WaitForNotification()
{
  boost::posix_time::ptime timeout;
  do {
    timeout = boost::get_system_time() + boost::posix_time::milliseconds(1000);
  } while (!stub_cond_->timed_wait(*stub_lock_, timeout) != 0 &&
           !non_graceful_exit);
  return non_graceful_exit;
}

Stub::~Stub()
{
  // stub_lock_ must be destroyed before the shared memory is deconstructed.
  // Otherwise, the shared memory will be destructed first and lead to
  // segfault.
  stub_lock_.reset();
}

std::unique_ptr<Stub> Stub::stub_instance_;

std::unique_ptr<Stub>&
Stub::GetOrCreateInstance()
{
  if (stub_instance_.get() == nullptr) {
    Stub::stub_instance_ = std::make_unique<Stub>();
  }

  return Stub::stub_instance_;
}

PYBIND11_EMBEDDED_MODULE(c_python_backend_utils, module)
{
  py::class_<PbTensor, std::shared_ptr<PbTensor>>(module, "Tensor")
      .def(py::init(&PbTensor::FromNumpy))
      .def("name", &PbTensor::Name)
      .def("as_numpy", &PbTensor::AsNumpy)
      .def("triton_dtype", &PbTensor::TritonDtype)
      .def("to_dlpack", &PbTensor::ToDLPack)
      .def("is_cpu", &PbTensor::IsCPU)
      .def("from_dlpack", &PbTensor::FromDLPack);

  py::class_<InferRequest>(module, "InferenceRequest")
      .def(py::init<
           const std::string&, uint64_t,
           const std::vector<std::shared_ptr<PbTensor>>&,
           const std::vector<std::string>&, const std::string&,
           const int64_t>())
      .def("inputs", &InferRequest::Inputs)
      .def("request_id", &InferRequest::RequestId)
      .def("correlation_id", &InferRequest::CorrelationId)
      // .def("exec", &InferRequest::Exec)
      .def("requested_output_names", &InferRequest::RequestedOutputNames);

  py::class_<InferResponse>(module, "InferenceResponse")
      .def(
          py::init<const std::vector<std::shared_ptr<PbTensor>>&, py::object>(),
          py::arg("output_tensors"), py::arg("error") = py::none())
      .def(
          "output_tensors", &InferResponse::OutputTensors,
          py::return_value_policy::reference)
      .def("has_error", &InferResponse::HasError)
      .def("error", &InferResponse::Error);

  py::register_exception<PythonBackendException>(
      module, "PythonBackendException");
}

extern "C" {

int
main(int argc, char** argv)
{
  if (argc < 8) {
    LOG_INFO << "Expected 8 arguments, found " << argc << " arguments.";
    exit(1);
  }
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

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
  std::string triton_install_path = argv[6];

  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  try {
    stub->Instantiate(
        shm_growth_size, shm_default_size, shm_region_name, model_path,
        model_version, argv[6] /* triton install path */,
        std::stoi(argv[7]) /* IPCControl offset */);
  }
  catch (const PythonBackendException& pb_exception) {
    LOG_INFO << "Failed to preinitialize Python stub: " << pb_exception.what();
    exit(1);
  }

  // Start the Python Interpreter
  py::scoped_interpreter guard{};
  pid_t parent_pid = std::stoi(argv[5]);

  std::atomic<bool> background_thread_running = {true};
  std::thread background_thread =
      std::thread([&parent_pid, &background_thread_running, &stub] {
        while (background_thread_running) {
          // Every 300ms set the health variable to true. This variable is in
          // shared memory and will be set to false by the parent process.
          // The parent process expects that the stub process sets this variable
          // to true within 1 second.
          sleep(0.3);

          stub->UpdateHealth();

          if (kill(parent_pid, 0) != 0) {
            // Destroy Stub
            LOG_INFO << "Non-graceful termination detected. ";
            background_thread_running = false;
            non_graceful_exit = true;

            // Destroy stub and exit.
            stub.reset();
            exit(1);
          }
        }
      });

  // This is the only place where NotifyParent() and WaitForNotification() are
  // allowed to be called. The stub process will always keep listening for new
  // notifications from the parent process. After the notification is received
  // the stub process will run the appropriate comamnd and wait for new
  // notifications.
  bool finalize = false;
  while (true) {
    stub->NotifyParent();
    if (finalize) {
      stub->Finalize();
      background_thread_running = false;
      background_thread.join();
      break;
    }

    if (stub->WaitForNotification()) {
      background_thread_running = false;
      background_thread.join();
      break;
    }

    finalize = stub->RunCommand();
  }

  // Stub must be destroyed before the py::scoped_interpreter goes out of scope.
  // The reason is that stub object has some attributes that are Python objects.
  // If the scoped_interpreter is destroyed before the stub object, this process
  // will no longer hold the GIL lock and destruction of the stub will result in
  // segfault.
  stub.reset();

  return 0;
}
}
}}  // namespace triton::backend
}  // namespace triton::backend::python
