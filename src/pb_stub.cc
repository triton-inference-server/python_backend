// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "infer_response.h"
#include "pb_error.h"
#include "pb_map.h"
#include "pb_string.h"
#include "scoped_defer.h"
#include "shm_manager.h"
#include "triton/common/nvtx.h"


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
    bi::managed_external_buffer::handle_t ipc_control_handle,
    const std::string& model_instance_name)
{
  model_path_ = model_path;
  model_version_ = model_version;
  triton_install_path_ = triton_install_path;
  model_instance_name_ = model_instance_name;
  health_mutex_ = nullptr;
  initialized_ = false;

  try {
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name, shm_default_size, shm_growth_size, false /* create */);

    AllocatedSharedMemory<IPCControlShm> ipc_control =
        shm_pool_->Load<IPCControlShm>(ipc_control_handle);
    ipc_control_ = ipc_control.data_.get();

    health_mutex_ = &(ipc_control_->stub_health_mutex);

    stub_message_queue_ = MessageQueue<bi::managed_external_buffer::handle_t>::
        LoadFromSharedMemory(shm_pool_, ipc_control_->stub_message_queue);

    parent_message_queue_ =
        MessageQueue<bi::managed_external_buffer::handle_t>::
            LoadFromSharedMemory(shm_pool_, ipc_control_->parent_message_queue);

    memory_manager_message_queue_ =
        MessageQueue<uint64_t>::LoadFromSharedMemory(
            shm_pool_, ipc_control_->memory_manager_message_queue);

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

std::unique_ptr<MessageQueue<uint64_t>>&
Stub::MemoryManagerQueue()
{
  return memory_manager_message_queue_;
}

bool&
Stub::Health()
{
  return ipc_control_->stub_health;
}

std::unique_ptr<SharedMemoryManager>&
Stub::SharedMemory()
{
  return shm_pool_;
}

std::unique_ptr<IPCMessage>
Stub::PopMessage()
{
  bool success = false;
  std::unique_ptr<IPCMessage> ipc_message;
  bi::managed_external_buffer::handle_t message;
  while (!success) {
    message = stub_message_queue_->Pop(1000, success);
  }

  ipc_message = IPCMessage::LoadFromSharedMemory(shm_pool_, message);

  return ipc_message;
}

bool
Stub::RunCommand()
{
  NVTX_RANGE(nvtx_, "RunCommand " + model_instance_name_);
  std::unique_ptr<IPCMessage> ipc_message = this->PopMessage();
  switch (ipc_message->Command()) {
    case PYTHONSTUB_CommandType::PYTHONSTUB_InitializeRequest: {
      bool has_exception = false;
      std::string error_string;

      std::unique_ptr<IPCMessage> initialize_response_msg =
          IPCMessage::Create(shm_pool_, false /* inline_response */);
      initialize_response_msg->Command() = PYTHONSTUB_InitializeResponse;
      std::unique_ptr<PbString> error_string_shm;
      AllocatedSharedMemory<InitializeResponseShm> initialize_response =
          shm_pool_->Construct<InitializeResponseShm>();

      // The initialization is done in three steps. First the main process sends
      // a message to the stub process asking to begin to initilize the Python
      // model. After that is finished stub process sends a message to the
      // parent process that the initialization is finished.  Finally, the
      // parent process sends a message to the stub process asking the stub
      // process to release any objects it has held in shared memory.
      ScopedDefer receive_initialize_finalize(
          [this] { stub_message_queue_->Pop(); });
      ScopedDefer _([this, &initialize_response_msg] {
        SendIPCMessage(initialize_response_msg);
      });

      initialize_response.data_->response_has_error = false;
      initialize_response.data_->response_is_error_set = false;
      initialize_response_msg->Args() = initialize_response.handle_;

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
        // Do not delete the region. The region will be deleted by the parent
        // process.
        shm_pool_->SetDeleteRegion(false);
        LOG_INFO << "Failed to initialize Python stub: " << error_string;
        initialize_response.data_->response_has_error = true;
        initialize_response.data_->response_is_error_set = false;

        LOG_IF_EXCEPTION(
            error_string_shm = PbString::Create(shm_pool_, error_string));
        if (error_string_shm != nullptr) {
          initialize_response.data_->response_is_error_set = true;
          initialize_response.data_->response_error =
              error_string_shm->ShmHandle();
        }

        return true;  // Terminate the stub process.
      }

    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest: {
      bool has_exception = false;
      std::string error_string;

      std::unique_ptr<IPCMessage> execute_response =
          IPCMessage::Create(shm_pool_, false /* Inline response */);
      execute_response->Command() = PYTHONSTUB_ExecuteResposne;

      AllocatedSharedMemory<char> request_batch =
          shm_pool_->Load<char>(ipc_message->Args());
      RequestBatch* request_batch_shm_ptr =
          reinterpret_cast<RequestBatch*>(request_batch.data_.get());
      AllocatedSharedMemory<char> response_batch = shm_pool_->Construct<char>(
          request_batch_shm_ptr->batch_size *
              sizeof(bi::managed_external_buffer::handle_t) +
          sizeof(ResponseBatch));
      ResponseBatch* response_batch_shm_ptr =
          reinterpret_cast<ResponseBatch*>(response_batch.data_.get());

      std::unique_ptr<PbString> error_string_shm;
      py::list inference_responses;

      bi::managed_external_buffer::handle_t* responses_shm_handle =
          reinterpret_cast<bi::managed_external_buffer::handle_t*>(
              response_batch.data_.get() + sizeof(ResponseBatch));

      execute_response->Args() = response_batch.handle_;

      ScopedDefer execute_finalize([this] { stub_message_queue_->Pop(); });
      ScopedDefer _(
          [this, &execute_response] { SendIPCMessage(execute_response); });

      response_batch_shm_ptr->has_error = false;
      response_batch_shm_ptr->is_error_set = false;
      try {
        inference_responses = Execute(
            request_batch_shm_ptr, response_batch_shm_ptr,
            responses_shm_handle);
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
        error_string_shm = PbString::Create(shm_pool_, error_string);
        response_batch_shm_ptr->has_error = true;
        response_batch_shm_ptr->is_error_set = true;
        response_batch_shm_ptr->error = error_string_shm->ShmHandle();
      }

    } break;
    case PYTHONSTUB_CommandType::PYTHONSTUB_FinalizeRequest:
      ipc_message->Command() = PYTHONSTUB_FinalizeResponse;
      SendIPCMessage(ipc_message);
      return true;  // Terminate the stub process
    case PYTHONSTUB_CommandType::PYTHONSTUB_LoadGPUBuffers:
      try {
        LoadGPUBuffers(ipc_message);
      }
      catch (const PythonBackendException& pb_exception) {
        LOG_INFO << "An error occurred while trying to load GPU buffers in the "
                    "Python backend stub: "
                 << pb_exception.what() << std::endl;
      }

      break;
    default:
      break;
  }

  return false;
}

void
Stub::Initialize(bi::managed_external_buffer::handle_t map_handle)
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
      python_backend_utils, "TritonError",
      c_python_backend_utils.attr("TritonError"));
  py::setattr(
      python_backend_utils, "TritonModelException",
      c_python_backend_utils.attr("TritonModelException"));
  py::setattr(
      python_backend_utils, "Tensor", c_python_backend_utils.attr("Tensor"));
  py::setattr(
      python_backend_utils, "InferenceRequest",
      c_python_backend_utils.attr("InferenceRequest"));
  py::setattr(
      python_backend_utils, "InferenceResponse",
      c_python_backend_utils.attr("InferenceResponse"));
  c_python_backend_utils.attr("shared_memory") = py::cast(shm_pool_.get());

  py::object TritonPythonModel =
      py::module_::import(
          (std::string(model_version_) + "." + model_name_trimmed).c_str())
          .attr("TritonPythonModel");
  deserialize_bytes_ = python_backend_utils.attr("deserialize_bytes_tensor");
  serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");
  model_instance_ = TritonPythonModel();

  std::unordered_map<std::string, std::string> map;
  std::unique_ptr<PbMap> pb_map_shm =
      PbMap::LoadFromSharedMemory(shm_pool_, map_handle);

  // Get the unordered_map representation of the map in shared memory.
  map = pb_map_shm->UnorderedMap();

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
Stub::ProcessResponse(InferResponse* response)
{
  response->SaveToSharedMemory(shm_pool_, false /* copy_gpu */);

  for (auto& output_tensor : response->OutputTensors()) {
    if (!output_tensor->IsCPU()) {
      gpu_tensors_.push_back(output_tensor);
    }
  }
}

void
Stub::LoadGPUBuffers(std::unique_ptr<IPCMessage>& ipc_message)
{
  AllocatedSharedMemory<char> gpu_buffers_handle =
      shm_pool_->Load<char>(ipc_message->Args());

  uint64_t* gpu_buffer_count =
      reinterpret_cast<uint64_t*>(gpu_buffers_handle.data_.get());
  bi::managed_external_buffer::handle_t* gpu_buffers_handle_shm =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          gpu_buffers_handle.data_.get() + sizeof(uint64_t));

  if (gpu_tensors_.size() != *gpu_buffer_count) {
    LOG_INFO
        << (std::string(
                "GPU buffers size does not match the provided buffers: ") +
            std::to_string(gpu_tensors_.size()) +
            " != " + std::to_string(*gpu_buffer_count));
    return;
  }

  // We need to hold the cpu_buffers until the main process makes a copy from
  // them.
  std::vector<std::unique_ptr<PbMemory>> cpu_buffers;
  std::vector<std::unique_ptr<PbMemory>> dst_buffers;

  bool has_cpu_buffer = false;
  for (size_t i = 0; i < gpu_tensors_.size(); i++) {
    std::unique_ptr<PbMemory> dst_buffer = PbMemory::LoadFromSharedMemory(
        shm_pool_, gpu_buffers_handle_shm[i], true /* open_cuda_handle */);
    if (dst_buffer->MemoryType() == TRITONSERVER_MEMORY_CPU) {
      has_cpu_buffer = true;
    }
    dst_buffers.emplace_back(std::move(dst_buffer));
  }

  // Pop a dummy message from the stub message queue indicating that the parent
  // has finished copying the tensors.
  ScopedDefer _([this, has_cpu_buffer] {
    if (has_cpu_buffer) {
      stub_message_queue_->Pop();
    }
  });

  ScopedDefer load_gpu_buffer_response(
      [this, has_cpu_buffer] { parent_message_queue_->Push(1000); });

  for (size_t i = 0; i < gpu_tensors_.size(); i++) {
    std::shared_ptr<PbTensor>& src_buffer = gpu_tensors_[i];

    // If the memory type is CPU, the buffer is empty and we need to create
    // a buffer.
    if (dst_buffers[i]->MemoryType() == TRITONSERVER_MEMORY_CPU) {
      dst_buffers[i] = PbMemory::Create(
          shm_pool_, dst_buffers[i]->MemoryType(),
          dst_buffers[i]->MemoryTypeId(), src_buffer->ByteSize(),
          nullptr /* buffer */);

      // Update the handle so that the main process can load it.
      gpu_buffers_handle_shm[i] = dst_buffers[i]->ShmHandle();
    }

    PbMemory::CopyBuffer(dst_buffers[i], src_buffer->Memory());

    if (dst_buffers[i]->MemoryType() == TRITONSERVER_MEMORY_CPU) {
      cpu_buffers.push_back(std::move(dst_buffers[i]));
    }
  }

  gpu_tensors_.clear();
}

py::list
Stub::Execute(
    RequestBatch* request_batch_shm_ptr, ResponseBatch* response_batch_shm_ptr,
    bi::managed_external_buffer::handle_t* responses_shm_handle)
{
  uint32_t batch_size = request_batch_shm_ptr->batch_size;
  py::list responses;

  if (batch_size == 0) {
    return responses;
  }

  py::list py_request_list;
  bi::managed_external_buffer::handle_t* request_shm_handle =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(request_batch_shm_ptr) +
          sizeof(RequestBatch));

  for (size_t i = 0; i < batch_size; i++) {
    std::shared_ptr<InferRequest> infer_request =
        InferRequest::LoadFromSharedMemory(
            shm_pool_, request_shm_handle[i], true /* open_cuda_handle */);
    py_request_list.append(std::move(infer_request));
  }

  if (!py::hasattr(model_instance_, "execute")) {
    std::string message =
        "Python model " + model_path_ + " does not implement `execute` method.";
    throw PythonBackendException(message);
  }

  py::object request_list = py_request_list;
  py::module asyncio = py::module::import("asyncio");

  // Execute Response
  py::object execute_return;
  py::object responses_obj;
  bool is_coroutine;

  {
    NVTX_RANGE(nvtx_, "PyExecute " + model_instance_name_);
    execute_return = model_instance_.attr("execute")(request_list);
    is_coroutine = asyncio.attr("iscoroutine")(execute_return).cast<bool>();
  }

  if (is_coroutine) {
    responses_obj = asyncio.attr("run")(execute_return);
  } else {
    responses_obj = execute_return;
  }

  // Check the return type of execute function.
  if (!py::isinstance<py::list>(responses_obj)) {
    std::string str = py::str(execute_return.get_type());
    throw PythonBackendException(
        std::string("Expected a list in the execute return, found type '") +
        str + "'.");
  }

  responses = responses_obj;
  size_t response_size = py::len(responses);

  // If the number of request objects do not match the number of
  // resposne objects throw an error.
  if (response_size != batch_size) {
    std::string err =
        "Number of InferenceResponse objects do not match the number "
        "of "
        "InferenceRequest objects. InferenceRequest(s) size is:" +
        std::to_string(batch_size) +
        ", and InferenceResponse(s) size is:" + std::to_string(response_size) +
        "\n";
    throw PythonBackendException(err);
  }
  for (auto& response : responses) {
    // Check the return type of execute function.
    if (!py::isinstance<InferResponse>(response)) {
      std::string str = py::str(response.get_type());
      throw PythonBackendException(
          std::string("Expected an 'InferenceResponse' object in the execute "
                      "function return list, found type '") +
          str + "'.");
    }
  }

  response_batch_shm_ptr->batch_size = response_size;

  std::vector<std::shared_ptr<PbTensor>> gpu_tensors;

  size_t i = 0;
  for (auto& response : responses) {
    InferResponse* infer_response = response.cast<InferResponse*>();
    ProcessResponse(infer_response);
    for (auto output_tensor : infer_response->OutputTensors()) {
      if (!output_tensor->IsCPU()) {
        gpu_tensors.push_back(output_tensor);
      }
    }
    responses_shm_handle[i] = infer_response->ShmHandle();
    i += 1;
  }

  return responses;
}

void
Stub::UpdateHealth()
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
  ipc_control_->stub_health = true;
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
    parent_message_queue_->Push(ipc_message->ShmHandle(), 1000, success);
  }
}

Stub::~Stub()
{
  model_instance_ = py::none();
  stub_message_queue_.reset();
  parent_message_queue_.reset();
  memory_manager_message_queue_.reset();
}

std::unique_ptr<Stub> Stub::stub_instance_;

std::unique_ptr<Stub>&
Stub::GetOrCreateInstance()
{
  if (Stub::stub_instance_.get() == nullptr) {
    Stub::stub_instance_ = std::make_unique<Stub>();
  }

  return Stub::stub_instance_;
}

PYBIND11_EMBEDDED_MODULE(c_python_backend_utils, module)
{
  py::class_<PbError, std::shared_ptr<PbError>>(module, "TritonError")
      .def(py::init<std::string>())
      .def("message", &PbError::Message);

  py::class_<InferRequest, std::shared_ptr<InferRequest>>(
      module, "InferenceRequest")
      .def(
          py::init<
              const std::string&, uint64_t,
              const std::vector<std::shared_ptr<PbTensor>>&,
              const std::vector<std::string>&, const std::string&,
              const int64_t, const uint32_t>(),
          py::arg("request_id") = "", py::arg("correlation_id") = 0,
          py::arg("inputs"), py::arg("requested_output_names"),
          py::arg("model_name"), py::arg("model_version") = -1,
          py::arg("flags") = 0)
      .def(
          "inputs", &InferRequest::Inputs,
          py::return_value_policy::reference_internal)
      .def("request_id", &InferRequest::RequestId)
      .def("correlation_id", &InferRequest::CorrelationId)
      .def("flags", &InferRequest::Flags)
      .def("set_flags", &InferRequest::SetFlags)
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

  py::class_<PbTensor, std::shared_ptr<PbTensor>>(module, "Tensor")
      .def(py::init(&PbTensor::FromNumpy))
      .def("name", &PbTensor::Name)
      .def("as_numpy", &PbTensor::AsNumpy)
      .def("triton_dtype", &PbTensor::TritonDtype)
      .def("to_dlpack", &PbTensor::ToDLPack)
      .def("is_cpu", &PbTensor::IsCPU)
      .def("from_dlpack", &PbTensor::FromDLPack);

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

  // This class is not part of the public API for Python backend. This is only
  // used for internal testing purposes.
  py::class_<SharedMemoryManager>(module, "SharedMemory")
      .def("free_memory", &SharedMemoryManager::FreeMemory);

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
        std::stoi(argv[7]) /* IPCControl handle */, model_instance_name);
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

  // The stub process will always keep listening for new notifications from the
  // parent process. After the notification is received the stub process will
  // run the appropriate comamnd and wait for new notifications.
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
