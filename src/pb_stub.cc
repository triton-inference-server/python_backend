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

#include "pb_stub.h"

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
#include "message_queue.h"
#include "pb_tensor.h"
#include "pb_utils.h"
#include "shm_manager.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

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
std::atomic<bool> non_graceful_exit = {false};


void
SignalHandler(int signum)
{
  // Skip the SIGINT and SIGTERM
}

void
Stub::Instantiate(
    int64_t shm_growth_size, int64_t shm_default_size,
    const std::string& shm_region_name, const std::string& model_path,
    const std::string& model_version, const std::string& triton_install_path,
    off_t ipc_control_offset, const std::string& model_instance_name)
{
  model_path_ = model_path;
  model_version_ = model_version;
  triton_install_path_ = triton_install_path;
  model_instance_name_ = model_instance_name;
  health_mutex_ = nullptr;
  initialized_ = false;
  cuda_ipc_open_mutex_ = std::make_shared<std::mutex>();
  cuda_ipc_close_mutex_ = std::make_shared<std::mutex>();

  try {
    shm_pool_ = std::make_unique<SharedMemory>(
        shm_region_name, shm_default_size, shm_growth_size);

    shm_pool_->MapOffset((char**)&ipc_control_, ipc_control_offset);

    bi::interprocess_mutex* health_mutex;
    shm_pool_->MapOffset(
        (char**)&health_mutex, ipc_control_->stub_health_mutex);
    health_mutex_ = health_mutex;

    stub_message_queue_ = MessageQueue::LoadFromSharedMemory(
        shm_pool_, ipc_control_->stub_message_queue);
    parent_message_queue_ = MessageQueue::LoadFromSharedMemory(
        shm_pool_, ipc_control_->parent_message_queue);

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
    InferResponse* response)
{
  // Initialize has_error to false
  response_shm->has_error = false;

  bool has_error = response->HasError();

  if (has_error) {
    response_shm->has_error = true;
    py::str py_string_err = response->Error()->Message();
    std::string response_error = py_string_err;
    SetErrorForResponse(response_shm, response_error.c_str());

    // Skip the response value when the response has error.
    return;
  }

  std::vector<std::shared_ptr<PbTensor>>& output_tensors =
      response->OutputTensors();
  for (auto& output_tensor : output_tensors) {
    if (!output_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
      AddToTensorsToRemove(output_tensor);
#else
      throw PythonBackendException("GPU tensors is not supported.");
#endif
    }
  }
  response->SaveToSharedMemory(
      shm_pool_, response_shm, true /* copy_cpu */, false /* copy_gpu */);
}

void
Stub::AddToTensorsToRemove(std::shared_ptr<PbTensor> tensor)
{
  std::lock_guard<std::mutex> guard{tensors_to_remove_mutex_};
  output_gpu_tensors_.push_back(tensor);
}

std::shared_ptr<InferRequest>
Stub::ProcessRequest(off_t request_offset, ResponseBatch* response_batch)
{
  std::shared_ptr<InferRequest> infer_request =
      InferRequest::LoadFromSharedMemory(
          shm_pool_, request_offset, cuda_ipc_open_mutex_,
          cuda_ipc_close_mutex_);

  for (auto& tensor : infer_request->Inputs()) {
    if (!tensor->IsCPU())
      input_gpu_tensors_.push_back(tensor);
  }

  return infer_request;
}

void
Stub::SetResponseFromException(
    ResponseBatch* response_batch, const PythonBackendException& pb_exception)
{
  SetErrorForResponseBatch(response_batch, pb_exception.what());
}

std::unique_ptr<IPCMessage>
Stub::PopMessage()
{
  bool success = false;
  std::unique_ptr<IPCMessage> ipc_message;
  off_t message;
  while (!success) {
    message = stub_message_queue_->Pop(1000, success);
  }

  ipc_message = IPCMessage::LoadFromSharedMemory(shm_pool_, message);

  return ipc_message;
}

std::shared_ptr<std::mutex>&
Stub::CudaIpcCloseMutex()
{
  return cuda_ipc_close_mutex_;
}

std::shared_ptr<std::mutex>&
Stub::CudaIpcOpenMutex()
{
  return cuda_ipc_open_mutex_;
}

bool
Stub::RunCommand()
{
  std::unique_ptr<IPCMessage> ipc_message = this->PopMessage();

  switch (ipc_message->Command()) {
    case PYTHONSTUB_CommandType::PYTHONSTUB_InitializeRequest: {
      bool has_exception = false;
      std::string error_string;

      std::unique_ptr<IPCMessage> initialize_response_msg =
          std::make_unique<IPCMessage>(shm_pool_, false);
      initialize_response_msg->Command() = PYTHONSTUB_InitializeResponse;

      InitializeResponse* initialize_response;
      shm_pool_->Map(
          (char**)&initialize_response, sizeof(InitializeResponse),
          initialize_response_msg->Args());
      initialize_response->response_has_error = false;
      initialize_response->response_is_error_set = false;

      try {
        Initialize(ipc_message->Args());
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
        initialize_response->response_has_error = true;
        initialize_response->response_is_error_set = false;
        off_t err_string_offset;
        LOG_IF_EXCEPTION(SaveStringToSharedMemory(
            shm_pool_, err_string_offset, error_string.c_str()));
        if (err_string_offset != 0) {
          initialize_response->response_is_error_set = true;
          initialize_response->response_error = err_string_offset;
        }

        this->SendIPCMessage(initialize_response_msg);
        return true;
      }

      this->SendIPCMessage(initialize_response_msg);
    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest: {
      RequestBatch* request_batch;
      bool has_exception = false;
      std::string error_string;

      std::unique_ptr<IPCMessage> execute_response =
          std::make_unique<IPCMessage>(shm_pool_, false /* Inline response */);
      execute_response->Command() = PYTHONSTUB_ExecuteResposne;

      shm_pool_->MapOffset((char**)&request_batch, ipc_message->Args());
      ResponseBatch* response_batch;
      shm_pool_->Map(
          (char**)&response_batch, sizeof(ResponseBatch),
          execute_response->Args());
      response_batch->has_error = false;
      try {
        Execute(request_batch, response_batch);
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
        std::string err_message =
            std::string(
                "Failed to process the request(s) for model '" +
                model_instance_name_ + "', message: ") +
            error_string;
        LOG_INFO << err_message.c_str();
        response_batch->has_error = true;
        response_batch->is_error_set = false;
        off_t err_string_offset = 0;
        LOG_IF_EXCEPTION(SaveStringToSharedMemory(
            shm_pool_, err_string_offset, error_string.c_str()));
        if (err_string_offset != 0) {
          response_batch->is_error_set = true;
          response_batch->error = err_string_offset;
        }
      }
      this->SendIPCMessage(execute_response);
    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_FinalizeRequest:
      ipc_message->Command() = PYTHONSTUB_FinalizeResponse;
      this->SendIPCMessage(ipc_message);
      return true;
    case PYTHONSTUB_CommandType::PYTHONSTUB_LoadGPUBuffers:
      try {
        LoadGPUBuffers();
      }
      catch (const PythonBackendException& pb_exception) {
        LOG_INFO << "An error occurred while trying to load GPU buffers in the "
                    "Python backend stub: "
                 << pb_exception.what() << std::endl;
      }

      ipc_message->Command() = PYTHONSTUB_LoadGPUBuffers;
      this->SendIPCMessage(ipc_message);
      break;
    default:
      break;
  }

  return false;
}

void
Stub::Execute(RequestBatch* request_batch, ResponseBatch* response_batch)
{
  uint32_t batch_size = request_batch->batch_size;

  if (batch_size == 0) {
    return;
  }

  py::list py_request_list;
  for (size_t i = 0; i < batch_size; i++) {
    off_t request_offset = request_batch->requests + i * sizeof(Request);
    py_request_list.append(ProcessRequest(request_offset, response_batch));
  }

  if (!py::hasattr(model_instance_, "execute")) {
    std::string message =
        "Python model " + model_path_ + " does not implement `execute` method.";
    throw PythonBackendException(message);
  }
  py::object request_list = py_request_list;
  py::module asyncio = py::module::import("asyncio");

  // Execute Response
  py::object execute_return = model_instance_.attr("execute")(request_list);
  py::list responses;
  bool is_coroutine = asyncio.attr("iscoroutine")(execute_return).cast<bool>();

  if (is_coroutine) {
    responses = asyncio.attr("run")(execute_return);
  } else {
    responses = execute_return;
  }

  Response* responses_shm;
  off_t responses_shm_offset;
  size_t response_size = py::len(responses);

  // If the number of request objects do not match the number of resposne
  // objects throw an error.
  if (response_size != batch_size) {
    std::string err =
        "Number of InferenceResponse objects do not match the number of "
        "InferenceRequest objects. InferenceRequest(s) size is:" +
        std::to_string(batch_size) +
        ", and InferenceResponse(s) size is:" + std::to_string(response_size) +
        "\n";
    throw PythonBackendException(err);
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
    ProcessResponse(response_shm, response_batch, infer_response);
    i += 1;
  }
}

void
Stub::Initialize(off_t map_offset)
{
  py::module sys = py::module_::import("sys");

  std::string model_name =
      model_path_.substr(model_path_.find_last_of("/") + 1);

  // Model name without the .py extension
  auto dotpy_pos = model_name.find_last_of(".py");
  if (dotpy_pos == std::string::npos || dotpy_pos != model_name.size() - 1) {
    throw PythonBackendException(
        "Model name must end with '.py'. Model name is \"" + model_name +
        "\".");
  }

  // The position of last character of the string that is searched for is
  // returned by 'find_last_of'. Need to manually adjust the position.
  std::string model_name_trimmed = model_name.substr(0, dotpy_pos - 2);
  std::string model_path_parent =
      model_path_.substr(0, model_path_.find_last_of("/"));
  std::string model_path_parent_parent =
      model_path_parent.substr(0, model_path_parent.find_last_of("/"));
  std::string python_backend_folder = triton_install_path_;
  sys.attr("path").attr("append")(model_path_parent);
  sys.attr("path").attr("append")(model_path_parent_parent);
  sys.attr("path").attr("append")(python_backend_folder);

  py::module python_backend_utils =
      py::module_::import("triton_python_backend_utils");
  py::module c_python_backend_utils =
      py::module_::import("c_python_backend_utils");
  py::setattr(
      python_backend_utils, "Tensor", c_python_backend_utils.attr("Tensor"));
  py::setattr(
      python_backend_utils, "InferenceRequest",
      c_python_backend_utils.attr("InferenceRequest"));
  py::setattr(
      python_backend_utils, "InferenceResponse",
      c_python_backend_utils.attr("InferenceResponse"));
  py::setattr(
      python_backend_utils, "TritonError",
      c_python_backend_utils.attr("TritonError"));
  py::setattr(
      python_backend_utils, "TritonModelException",
      c_python_backend_utils.attr("TritonModelException"));

  py::object TritonPythonModel =
      py::module_::import(
          (std::string(model_version_) + "." + model_name_trimmed).c_str())
          .attr("TritonPythonModel");
  deserialize_bytes_ = python_backend_utils.attr("deserialize_bytes_tensor");
  serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");
  model_instance_ = TritonPythonModel();

  std::unordered_map<std::string, std::string> map;
  LoadMapFromSharedMemory(shm_pool_, map_offset, map);
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
Stub::LoadGPUBuffers()
{
  std::lock_guard<std::mutex> guard{tensors_to_remove_mutex_};
#ifdef TRITON_ENABLE_GPU
  for (auto& tensor : output_gpu_tensors_) {
    if (tensor->RawDataShm()->memory_type == TRITONSERVER_MEMORY_GPU) {
      tensor->SetCudaIpcMutexes(CudaIpcOpenMutex(), CudaIpcCloseMutex());
      tensor->LoadGPUData(shm_pool_);
    } else {
      tensor->CopyToCPU(shm_pool_);
    }
  }
#endif  // TRITON_ENABLE_GPU

  output_gpu_tensors_.clear();
  input_gpu_tensors_.clear();
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

void
Stub::SendIPCMessage(std::unique_ptr<IPCMessage>& ipc_message)
{
  bool success = false;
  while (!success) {
    parent_message_queue_->Push(
        ipc_message->SharedMemoryOffset(), 1000, success);
  }
}

Stub::~Stub()
{
  // stub_lock_ must be destroyed before the shared memory is deconstructed.
  // Otherwise, the shared memory will be destructed first and lead to
  // segfault.
  stub_lock_.reset();
  stub_message_queue_.reset();
  parent_message_queue_.reset();
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

  py::class_<InferRequest, std::shared_ptr<InferRequest>>(
      module, "InferenceRequest")
      .def(
          py::init<
              const std::string&, uint64_t,
              const std::vector<std::shared_ptr<PbTensor>>&,
              const std::vector<std::string>&, const std::string&,
              const int64_t>(),
          py::arg("request_id") = "", py::arg("correlation_id") = 0,
          py::arg("inputs"), py::arg("requested_output_names"),
          py::arg("model_name"), py::arg("model_version") = -1)
      .def(
          "inputs", &InferRequest::Inputs,
          py::return_value_policy::reference_internal)
      .def("request_id", &InferRequest::RequestId)
      .def("correlation_id", &InferRequest::CorrelationId)
      .def("exec", &InferRequest::Exec)
      .def(
          "async_exec",
          [](std::shared_ptr<InferRequest>& infer_request) {
            py::object loop =
                py::module_::import("asyncio").attr("get_running_loop")();
            py::cpp_function callback = [infer_request]() {
              auto response = infer_request->Exec();
              return response;
            };
            py::object future =
                loop.attr("run_in_executor")(py::none(), callback);
            return future;
          })
      .def(
          "requested_output_names", &InferRequest::RequestedOutputNames,
          py::return_value_policy::reference_internal);

  py::class_<InferResponse>(module, "InferenceResponse")
      .def(
          py::init<
              const std::vector<std::shared_ptr<PbTensor>>&,
              std::shared_ptr<PbError>>(),
          py::arg("output_tensors"), py::arg("error") = nullptr)
      .def(
          "output_tensors", &InferResponse::OutputTensors,
          py::return_value_policy::reference)
      .def("has_error", &InferResponse::HasError)
      .def("error", &InferResponse::Error);

  py::class_<PbError, std::shared_ptr<PbError>>(module, "TritonError")
      .def(py::init<std::string>())
      .def("message", &PbError::Message);

  py::register_exception<PythonBackendException>(
      module, "TritonModelException");
}

extern "C" {

int
main(int argc, char** argv)
{
  if (argc < 9) {
    LOG_INFO << "Expected 9 arguments, found " << argc << " arguments.";
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
  std::string model_instance_name = argv[8];

  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  try {
    stub->Instantiate(
        shm_growth_size, shm_default_size, shm_region_name, model_path,
        model_version, argv[6] /* triton install path */,
        std::stoi(argv[7]) /* IPCControl offset */, model_instance_name);
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
          // The parent process expects that the stub process sets this
          // variable to true within 1 second.
          std::this_thread::sleep_for(std::chrono::milliseconds(300));

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
    if (finalize) {
      stub->Finalize();
      background_thread_running = false;
      background_thread.join();
      break;
    }

    finalize = stub->RunCommand();
  }

  // Stub must be destroyed before the py::scoped_interpreter goes out of
  // scope. The reason is that stub object has some attributes that are Python
  // objects. If the scoped_interpreter is destroyed before the stub object,
  // this process will no longer hold the GIL lock and destruction of the stub
  // will result in segfault.
  stub.reset();

  return 0;
}
}
}}}  // namespace triton::backend::python
