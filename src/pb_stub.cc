// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <regex>
#include <thread>
#include <unordered_map>

#include "model_loader.h"
#include "pb_error.h"
#include "pb_map.h"
#include "pb_preferred_memory.h"
#include "pb_response_iterator.h"
#include "pb_string.h"
#include "pb_utils.h"
#include "response_sender.h"
#include "scoped_defer.h"
#include "shm_manager.h"
#include "triton/common/nvtx.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace py = pybind11;
using namespace pybind11::literals;
namespace bi = boost::interprocess;
#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif

namespace triton { namespace backend { namespace python {

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
    const std::string& name, const std::string& python_runtime_model)
{
  model_context_.Init(
      model_path, python_runtime_model, triton_install_path, model_version);
  name_ = name;
  health_mutex_ = nullptr;
  initialized_ = false;
  finalizing_ = false;
  stub_to_parent_thread_ = false;
  parent_to_stub_thread_ = false;

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

    stub_to_parent_mq_ = MessageQueue<bi::managed_external_buffer::handle_t>::
        LoadFromSharedMemory(shm_pool_, ipc_control_->stub_to_parent_mq);

    parent_to_stub_mq_ = MessageQueue<bi::managed_external_buffer::handle_t>::
        LoadFromSharedMemory(shm_pool_, ipc_control_->parent_to_stub_mq);

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
Stub::IsDecoupled()
{
  return ipc_control_->decoupled;
}

bool
Stub::RunCommand()
{
  NVTX_RANGE(nvtx_, "RunCommand " + name_);
  std::unique_ptr<IPCMessage> ipc_message;
  {
    // Release the GIL lock when waiting for new message. Without this line, the
    // other threads in the user's Python model cannot make progress if they
    // give up GIL.
    py::gil_scoped_release release;
    ipc_message = this->PopMessage();
  }
  switch (ipc_message->Command()) {
    case PYTHONSTUB_CommandType::PYTHONSTUB_AutoCompleteRequest: {
      // Only run this case when auto complete was requested by
      // Triton core.
      bool has_exception = false;
      std::string error_string;
      std::string auto_complete_config;

      std::unique_ptr<IPCMessage> auto_complete_response_msg =
          IPCMessage::Create(shm_pool_, false /* inline_response */);
      auto_complete_response_msg->Command() = PYTHONSTUB_AutoCompleteResponse;
      std::unique_ptr<PbString> error_string_shm;
      std::unique_ptr<PbString> auto_complete_config_shm;
      AllocatedSharedMemory<AutoCompleteResponseShm> auto_complete_response =
          shm_pool_->Construct<AutoCompleteResponseShm>();

      ScopedDefer receive_autocomplete_finalize(
          [this] { stub_message_queue_->Pop(); });
      ScopedDefer _([this, &auto_complete_response_msg] {
        SendIPCMessage(auto_complete_response_msg);
      });

      auto_complete_response.data_->response_has_error = false;
      auto_complete_response.data_->response_is_error_set = false;
      auto_complete_response.data_->response_has_model_config = false;
      auto_complete_response_msg->Args() = auto_complete_response.handle_;

      try {
        AutoCompleteModelConfig(ipc_message->Args(), &auto_complete_config);
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
        LOG_INFO << "Failed to initialize Python stub for auto-complete: "
                 << error_string;
        auto_complete_response.data_->response_has_error = true;
        auto_complete_response.data_->response_is_error_set = false;

        LOG_IF_EXCEPTION(
            error_string_shm = PbString::Create(shm_pool_, error_string));
        if (error_string_shm != nullptr) {
          auto_complete_response.data_->response_is_error_set = true;
          auto_complete_response.data_->response_error =
              error_string_shm->ShmHandle();
        }

        return true;  // Terminate the stub process.
      } else {
        LOG_IF_EXCEPTION(
            auto_complete_config_shm =
                PbString::Create(shm_pool_, auto_complete_config));
        if (auto_complete_config_shm != nullptr) {
          auto_complete_response.data_->response_has_model_config = true;
          auto_complete_response.data_->response_model_config =
              auto_complete_config_shm->ShmHandle();
        }
      }
    } break;
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
      // a message to the stub process asking to begin to initialize the Python
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
      AllocatedSharedMemory<char> request_batch =
          shm_pool_->Load<char>(ipc_message->Args());
      RequestBatch* request_batch_shm_ptr =
          reinterpret_cast<RequestBatch*>(request_batch.data_.get());
      if (!ipc_control_->decoupled) {
        ProcessRequests(request_batch_shm_ptr);
      } else {
        ProcessRequestsDecoupled(request_batch_shm_ptr);
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
        LOG_ERROR
            << "An error occurred while trying to load GPU buffers in the "
               "Python backend stub: "
            << pb_exception.what() << std::endl;
      }

      break;
    default:
      break;
  }

  return false;
}

py::module
Stub::StubSetup()
{
  py::module sys = py::module_::import("sys");

  model_context_.StubSetup(sys);

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
  py::setattr(
      python_backend_utils, "Logger", c_python_backend_utils.attr("Logger"));
  py::setattr(
      python_backend_utils, "PreferredMemory",
      c_python_backend_utils.attr("PreferredMemory"));
  py::setattr(
      python_backend_utils, "TRITONSERVER_MEMORY_GPU",
      c_python_backend_utils.attr("TRITONSERVER_MEMORY_GPU"));
  py::setattr(
      python_backend_utils, "TRITONSERVER_MEMORY_CPU",
      c_python_backend_utils.attr("TRITONSERVER_MEMORY_CPU"));
  py::setattr(
      python_backend_utils, "MetricFamily",
      c_python_backend_utils.attr("MetricFamily"));
  py::setattr(
      python_backend_utils, "load_model",
      c_python_backend_utils.attr("load_model"));
  py::setattr(
      python_backend_utils, "unload_model",
      c_python_backend_utils.attr("unload_model"));
  py::setattr(
      python_backend_utils, "is_model_ready",
      c_python_backend_utils.attr("is_model_ready"));

  c_python_backend_utils.attr("shared_memory") = py::cast(shm_pool_.get());

  deserialize_bytes_ = python_backend_utils.attr("deserialize_bytes_tensor");
  serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");

  return sys;
}

void
Stub::AutoCompleteModelConfig(
    bi::managed_external_buffer::handle_t string_handle,
    std::string* auto_complete_config)
{
  py::module sys = StubSetup();

  std::unique_ptr<PbString> pb_string_shm =
      PbString::LoadFromSharedMemory(shm_pool_, string_handle);

  py::module python_backend_utils =
      py::module_::import("triton_python_backend_utils");
  py::object model_config =
      python_backend_utils.attr("ModelConfig")(pb_string_shm->String());
  python_backend_utils.def(
      "get_model_dir",
      []() {
        std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
        return stub->GetModelDir();
      },
      py::return_value_policy::reference);

  if (py::hasattr(sys.attr("TritonPythonModel"), "auto_complete_config")) {
    model_config = sys.attr("TritonPythonModel")
                       .attr("auto_complete_config")(model_config);
  }

  if (!py::isinstance(model_config, python_backend_utils.attr("ModelConfig"))) {
    throw PythonBackendException(
        "auto_complete_config function in model '" + name_ +
        "' must return a valid pb.ModelConfig object.");
  }
  py::module json = py::module_::import("json");
  (*auto_complete_config) = std::string(
      py::str(json.attr("dumps")(model_config.attr("_model_config"))));
}

void
Stub::Initialize(bi::managed_external_buffer::handle_t map_handle)
{
  py::module sys = StubSetup();

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

  py::object TritonPythonModel = sys.attr("TritonPythonModel");
  deserialize_bytes_ = python_backend_utils.attr("deserialize_bytes_tensor");
  serialize_bytes_ = python_backend_utils.attr("serialize_byte_tensor");
  python_backend_utils.def(
      "get_model_dir",
      []() {
        std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
        return stub->GetModelDir();
      },
      py::return_value_policy::reference);
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

  LaunchStubToParentQueueMonitor();
  LaunchParentToStubQueueMonitor();

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
  ScopedDefer load_gpu_buffer_response([this] {
    // LoadGPUBuffers must let the parent process know when loading the
    // buffers have been finished.
    parent_message_queue_->Push(DUMMY_MESSAGE);
    gpu_tensors_.clear();
  });

  AllocatedSharedMemory<GPUBuffersShm> gpu_buffers_handle =
      shm_pool_->Load<GPUBuffersShm>(ipc_message->Args());

  if (!gpu_buffers_handle.data_->success) {
    std::unique_ptr<PbString> error = PbString::LoadFromSharedMemory(
        shm_pool_, gpu_buffers_handle.data_->error);
    throw PythonBackendException(
        "Failed to load GPU buffers: " + error->String());
  }

  uint64_t gpu_buffer_count = gpu_buffers_handle.data_->buffer_count;
  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      gpu_buffers_handle_shm =
          shm_pool_->Load<bi::managed_external_buffer::handle_t>(
              gpu_buffers_handle.data_->buffers);

  if (gpu_tensors_.size() != gpu_buffer_count) {
    throw PythonBackendException(
        std::string("GPU buffers size does not match the provided buffers: ") +
        std::to_string(gpu_tensors_.size()) +
        " != " + std::to_string(gpu_buffer_count));
  }

  std::vector<std::unique_ptr<PbMemory>> dst_buffers;
  for (size_t i = 0; i < gpu_tensors_.size(); i++) {
    std::unique_ptr<PbMemory> dst_buffer = PbMemory::LoadFromSharedMemory(
        shm_pool_, gpu_buffers_handle_shm.data_.get()[i],
        true /* open_cuda_handle */);
    dst_buffers.emplace_back(std::move(dst_buffer));
  }

  for (size_t i = 0; i < gpu_tensors_.size(); i++) {
    std::shared_ptr<PbTensor>& src_buffer = gpu_tensors_[i];
    PbMemory::CopyBuffer(dst_buffers[i], src_buffer->Memory());
  }
}

py::list
Stub::LoadRequestsFromSharedMemory(RequestBatch* request_batch_shm_ptr)
{
  uint32_t batch_size = request_batch_shm_ptr->batch_size;
  py::list py_request_list;

  if (batch_size == 0) {
    return py_request_list;
  }

  bi::managed_external_buffer::handle_t* request_shm_handle =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          reinterpret_cast<char*>(request_batch_shm_ptr) +
          sizeof(RequestBatch));

  for (size_t i = 0; i < batch_size; i++) {
    std::shared_ptr<InferRequest> infer_request =
        InferRequest::LoadFromSharedMemory(
            shm_pool_, request_shm_handle[i], true /* open_cuda_handle */);
    py_request_list.append(infer_request);
  }

  return py_request_list;
}

void
Stub::ProcessRequestsDecoupled(RequestBatch* request_batch_shm_ptr)
{
  py::list py_request_list =
      LoadRequestsFromSharedMemory(request_batch_shm_ptr);
  std::unique_ptr<IPCMessage> execute_response =
      IPCMessage::Create(shm_pool_, false /* Inline response */);
  execute_response->Command() = PYTHONSTUB_ExecuteResponse;

  AllocatedSharedMemory<ResponseBatch> response_batch =
      shm_pool_->Construct<ResponseBatch>();
  ResponseBatch* response_batch_shm_ptr =
      reinterpret_cast<ResponseBatch*>(response_batch.data_.get());
  execute_response->Args() = response_batch.handle_;
  bool has_exception = false;
  std::string error_string;
  std::unique_ptr<PbString> error_string_shm;

  ScopedDefer execute_finalize([this] { stub_message_queue_->Pop(); });
  ScopedDefer _(
      [this, &execute_response] { SendIPCMessage(execute_response); });

  try {
    response_batch_shm_ptr->has_error = false;
    response_batch_shm_ptr->is_error_set = false;

    if (!py::hasattr(model_instance_, "execute")) {
      std::string message = "Python model " + model_context_.PythonModelPath() +
                            " does not implement `execute` method.";
      throw PythonBackendException(message);
    }

    {
      NVTX_RANGE(nvtx_, "PyExecute " + name_);

      py::object execute_return =
          model_instance_.attr("execute")(py_request_list);
      if (!py::isinstance<py::none>(execute_return)) {
        throw PythonBackendException(
            "Python model '" + name_ +
            "' is using the decoupled mode and the execute function must "
            "return None.");
      }
    }
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
            "Failed to process the request(s) for model '" + name_ +
            "', message: ") +
        error_string;
    LOG_INFO << err_message.c_str();
    response_batch_shm_ptr->has_error = true;
    error_string_shm = PbString::Create(shm_pool_, error_string);
    response_batch_shm_ptr->error = error_string_shm->ShmHandle();
    response_batch_shm_ptr->is_error_set = true;
  }
}

void
Stub::ProcessRequests(RequestBatch* request_batch_shm_ptr)
{
  std::unique_ptr<IPCMessage> execute_response =
      IPCMessage::Create(shm_pool_, false /* Inline response */);
  execute_response->Command() = PYTHONSTUB_ExecuteResponse;

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

  py::list responses;

  // Notifying the stub should be after responses.
  ScopedDefer execute_finalize([this] { stub_message_queue_->Pop(); });
  ScopedDefer _(
      [this, &execute_response] { SendIPCMessage(execute_response); });

  execute_response->Args() = response_batch.handle_;

  bool has_exception = false;
  std::string error_string;
  try {
    response_batch_shm_ptr->has_error = false;
    response_batch_shm_ptr->is_error_set = false;

    uint32_t batch_size = request_batch_shm_ptr->batch_size;

    if (batch_size == 0) {
      return;
    }

    py::list py_request_list =
        LoadRequestsFromSharedMemory(request_batch_shm_ptr);

    if (!py::hasattr(model_instance_, "execute")) {
      std::string message = "Python model " + model_context_.PythonModelPath() +
                            " does not implement `execute` method.";
      throw PythonBackendException(message);
    }

    py::object request_list = py_request_list;
    py::module asyncio = py::module::import("asyncio");

    // Execute Response
    py::object execute_return;
    py::object responses_obj;
    bool is_coroutine;

    {
      NVTX_RANGE(nvtx_, "PyExecute " + name_);
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
    // response objects throw an error.
    if (response_size != batch_size) {
      std::string err =
          "Number of InferenceResponse objects do not match the number "
          "of "
          "InferenceRequest objects. InferenceRequest(s) size is:" +
          std::to_string(batch_size) + ", and InferenceResponse(s) size is:" +
          std::to_string(response_size) + "\n";
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

    for (size_t i = 0; i < batch_size; i++) {
      InferResponse* infer_response = responses[i].cast<InferResponse*>();
      InferRequest* infer_request = py_request_list[i].cast<InferRequest*>();
      infer_response->PruneOutputTensors(infer_request->RequestedOutputNames());

      ProcessResponse(infer_response);
      responses_shm_handle[i] = infer_response->ShmHandle();
    }
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
            "Failed to process the request(s) for model '" + name_ +
            "', message: ") +
        error_string;
    error_string_shm = PbString::Create(shm_pool_, error_string);
    response_batch_shm_ptr->has_error = true;
    response_batch_shm_ptr->is_error_set = true;
    response_batch_shm_ptr->error = error_string_shm->ShmHandle();
  }
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
  finalizing_ = true;
  // Call finalize if exists.
  if (initialized_ && py::hasattr(model_instance_, "finalize")) {
    try {
      model_instance_.attr("finalize")();
    }
    catch (const py::error_already_set& e) {
      LOG_INFO << e.what();
    }
  }
#ifdef TRITON_ENABLE_GPU
  // We also need to destroy created proxy CUDA streams for dlpack, if any
  std::lock_guard<std::mutex> lock(dlpack_proxy_stream_pool_mu_);
  for (auto& entry : dlpack_proxy_stream_pool_) {
    // We don't need to switch device to destroy a stream
    // https://stackoverflow.com/questions/64663943/how-to-destroy-a-stream-that-was-created-on-a-specific-device
    cudaError_t err = cudaStreamDestroy(entry.second);
    if (err != cudaSuccess) {
      LOG_ERROR
          << "Failed to destroy dlpack CUDA proxy stream on device with id " +
                 std::to_string(entry.first);
    }
  }
#endif
}

void
Stub::SendIPCMessage(std::unique_ptr<IPCMessage>& ipc_message)
{
  bool success = false;
  while (!success) {
    parent_message_queue_->Push(ipc_message->ShmHandle(), 1000, success);
  }
}

void
Stub::SendIPCUtilsMessage(std::unique_ptr<IPCMessage>& ipc_message)
{
  bool success = false;
  while (!success) {
    stub_to_parent_mq_->Push(ipc_message->ShmHandle(), 1000, success);
  }
}

Stub::~Stub()
{
  {
    py::gil_scoped_acquire acquire;
    model_instance_ = py::none();
  }
  stub_instance_.reset();
  stub_message_queue_.reset();
  parent_message_queue_.reset();
  stub_to_parent_mq_.reset();
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

void
Stub::LaunchStubToParentQueueMonitor()
{
  stub_to_parent_thread_ = true;
  stub_to_parent_queue_monitor_ =
      std::thread(&Stub::ServiceStubToParentRequests, this);
  Logger::GetOrCreateInstance()->SetBackendLoggingActive(true);
}

void
Stub::TerminateStubToParentQueueMonitor()
{
  Logger::GetOrCreateInstance()->SetBackendLoggingActive(false);
  {
    std::lock_guard<std::mutex> guard{stub_to_parent_message_mu_};
    // Push a dummy message to signal the thread to terminate.
    stub_to_parent_buffer_.push(DUMMY_MESSAGE);
  }
  stub_to_parent_message_cv_.notify_one();
  stub_to_parent_queue_monitor_.join();
}

void
Stub::EnqueueLogRequest(std::unique_ptr<PbLog>& log_ptr)
{
  std::unique_ptr<UtilsMessagePayload> utils_msg_payload =
      std::make_unique<UtilsMessagePayload>(
          PYTHONSTUB_LogRequest, reinterpret_cast<void*>(log_ptr.release()));
  EnqueueUtilsMessage(std::move(utils_msg_payload));
}

void
Stub::ServiceStubToParentRequests()
{
  while (stub_to_parent_thread_) {
    std::unique_lock<std::mutex> guard{stub_to_parent_message_mu_};
    while (stub_to_parent_buffer_.empty()) {
      stub_to_parent_message_cv_.wait(guard);
    }
    // On exit, will send messages to the parent process until
    // DUMMY_MESSAGE is reached
    std::unique_ptr<UtilsMessagePayload> utils_msg_payload =
        std::move(stub_to_parent_buffer_.front());
    if (utils_msg_payload == DUMMY_MESSAGE) {
      stub_to_parent_buffer_.pop();
      break;
    } else {
      stub_to_parent_buffer_.pop();
      if (utils_msg_payload->command_type == PYTHONSTUB_LogRequest) {
        SendLogMessage(utils_msg_payload);
      } else if (utils_msg_payload->command_type == PYTHONSTUB_CleanupRequest) {
        SendCleanupId(utils_msg_payload);
      } else if (
          utils_msg_payload->command_type == PYTHONSTUB_IsRequestCancelled) {
        SendIsCancelled(utils_msg_payload);
      } else {
        std::cerr << "Error when sending message via stub_to_parent message "
                     "buffer - unknown command\n";
      }
    }
  }
}

void
Stub::SendLogMessage(std::unique_ptr<UtilsMessagePayload>& utils_msg_payload)
{
  std::unique_ptr<PbLog> log_send_message = std::unique_ptr<PbLog>(
      reinterpret_cast<PbLog*>(utils_msg_payload->utils_message_ptr));

  std::unique_ptr<PbLogShm> log_request_shm = PbLogShm::Create(
      shm_pool_, log_send_message->Filename(), log_send_message->Line(),
      log_send_message->Message(), log_send_message->Level());
  LogSendMessage* send_message_payload = log_request_shm->LogMessage();
  send_message_payload->waiting_on_stub = false;
  std::unique_ptr<IPCMessage> log_request_msg =
      IPCMessage::Create(shm_pool_, false /* inline_response */);
  log_request_msg->Args() = log_request_shm->ShmHandle();
  log_request_msg->Command() = PYTHONSTUB_LogRequest;
  ScopedDefer _([send_message_payload] {
    {
      bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
      send_message_payload->waiting_on_stub = false;
      send_message_payload->cv.notify_all();
    }
  });

  {
    // Send a message to be caught by the log monitor thread in python_be.cc
    bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
    SendIPCUtilsMessage(log_request_msg);
    while (!send_message_payload->waiting_on_stub) {
      send_message_payload->cv.wait(guard);
    }
  }
}

void
Stub::SendCleanupId(std::unique_ptr<UtilsMessagePayload>& utils_msg_payload)
{
  void* id = utils_msg_payload->utils_message_ptr;
  {
    std::lock_guard<std::mutex> lock(response_iterator_map_mu_);
    response_iterator_map_.erase(id);
  }

  std::unique_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(shm_pool_, true /* inline_response */);
  ipc_message->Command() = PYTHONSTUB_CleanupRequest;
  AllocatedSharedMemory<char> cleanup_request_message =
      shm_pool_->Construct<char>(
          sizeof(CleanupMessage) +
          sizeof(bi::managed_external_buffer::handle_t));
  CleanupMessage* cleanup_message_ptr =
      reinterpret_cast<CleanupMessage*>(cleanup_request_message.data_.get());
  cleanup_message_ptr->id = id;
  cleanup_message_ptr->waiting_on_stub = false;
  ipc_message->Args() = cleanup_request_message.handle_;

  {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    SendIPCUtilsMessage(ipc_message);
    while (!cleanup_message_ptr->waiting_on_stub) {
      ipc_message->ResponseCondition()->wait(lock);
    }
  }
}

void
Stub::EnqueueCleanupId(void* id)
{
  if (id != nullptr) {
    std::unique_ptr<UtilsMessagePayload> utils_msg_payload =
        std::make_unique<UtilsMessagePayload>(PYTHONSTUB_CleanupRequest, id);
    EnqueueUtilsMessage(std::move(utils_msg_payload));
  }
}

void
Stub::EnqueueIsCancelled(PbCancel* pb_cancel)
{
  std::unique_ptr<UtilsMessagePayload> utils_msg_payload =
      std::make_unique<UtilsMessagePayload>(
          PYTHONSTUB_IsRequestCancelled, reinterpret_cast<void*>(pb_cancel));
  EnqueueUtilsMessage(std::move(utils_msg_payload));
}

void
Stub::SendIsCancelled(std::unique_ptr<UtilsMessagePayload>& utils_msg_payload)
{
  PbCancel* pb_cancel =
      reinterpret_cast<PbCancel*>(utils_msg_payload->utils_message_ptr);
  pb_cancel->SaveToSharedMemory(shm_pool_);

  IsCancelledMessage* message_payload = pb_cancel->ShmPayload();
  std::unique_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(shm_pool_, false /* inline_response */);
  ipc_message->Command() = utils_msg_payload->command_type;
  ipc_message->Args() = pb_cancel->ShmHandle();

  bool is_cancelled = false;
  {
    bi::scoped_lock<bi::interprocess_mutex> lk(message_payload->mu);

    SendIPCUtilsMessage(ipc_message);
    while (!message_payload->waiting_on_stub) {
      message_payload->cv.wait(lk);
    }

    is_cancelled = message_payload->is_cancelled;
    message_payload->waiting_on_stub = false;
    message_payload->cv.notify_all();
  }
  pb_cancel->ReportIsCancelled(is_cancelled);
}

bool
Stub::StubToParentServiceActive()
{
  return stub_to_parent_thread_;
}

void
Stub::LaunchParentToStubQueueMonitor()
{
  parent_to_stub_thread_ = true;
  parent_to_stub_queue_monitor_ =
      std::thread(&Stub::ParentToStubMQMonitor, this);
}

void
Stub::TerminateParentToStubQueueMonitor()
{
  if (parent_to_stub_thread_) {
    parent_to_stub_thread_ = false;
    // Push a dummy message to signal the thread to terminate.
    parent_to_stub_mq_->Push(DUMMY_MESSAGE);
    parent_to_stub_queue_monitor_.join();
  }
}

void
Stub::ParentToStubMQMonitor()
{
  while (parent_to_stub_thread_) {
    bi::managed_external_buffer::handle_t handle = parent_to_stub_mq_->Pop();
    if (handle == DUMMY_MESSAGE) {
      break;
    }

    std::unique_ptr<IPCMessage> ipc_message;
    ResponseBatch* response_batch = nullptr;
    bi::managed_external_buffer::handle_t* response_handle = nullptr;
    std::unique_ptr<InferResponse> infer_response;
    bool responses_is_set = false;
    PythonBackendException pb_exception(std::string{});

    try {
      ipc_message = IPCMessage::LoadFromSharedMemory(shm_pool_, handle);
      AllocatedSharedMemory<char> response_batch_shm =
          shm_pool_->Load<char>(ipc_message->Args());
      response_batch =
          reinterpret_cast<ResponseBatch*>(response_batch_shm.data_.get());
      response_handle =
          reinterpret_cast<bi::managed_external_buffer::handle_t*>(
              response_batch_shm.data_.get() + sizeof(ResponseBatch));
      responses_is_set = true;

      if (response_batch->has_error) {
        if (response_batch->is_error_set) {
          std::unique_ptr<PbString> pb_string =
              PbString::LoadFromSharedMemory(shm_pool_, response_batch->error);
          infer_response = std::make_unique<InferResponse>(
              std::vector<std::shared_ptr<PbTensor>>{},
              std::make_shared<PbError>(pb_string->String()));
        } else {
          infer_response = std::make_unique<InferResponse>(
              std::vector<std::shared_ptr<PbTensor>>{},
              std::make_shared<PbError>(
                  "An error occurred while performing BLS request."));
        }
      }

      if (responses_is_set) {
        infer_response = InferResponse::LoadFromSharedMemory(
            shm_pool_, *response_handle, true /* open cuda handle */);

        for (auto& output_tensor : infer_response->OutputTensors()) {
          if (!output_tensor->IsCPU()) {
            uint64_t memory_release_id =
                output_tensor->Memory()->MemoryReleaseId();
            output_tensor->Memory()->SetMemoryReleaseCallback(
                [this, memory_release_id]() {
                  this->MemoryManagerQueue()->Push(memory_release_id);
                });
          }
        }
      } else {
        infer_response = std::make_unique<InferResponse>(
            std::vector<std::shared_ptr<PbTensor>>{},
            std::make_shared<PbError>(
                "An error occurred while performing BLS request."));
      }
    }
    catch (const PythonBackendException& pb_exception) {
      infer_response = std::make_unique<InferResponse>(
          std::vector<std::shared_ptr<PbTensor>>{},
          std::make_shared<PbError>(pb_exception.what()));
    }

    {
      std::lock_guard<std::mutex> lock(response_iterator_map_mu_);
      if (response_iterator_map_.find(infer_response->Id()) !=
          response_iterator_map_.end()) {
        response_iterator_map_[infer_response->Id()]->EnqueueResponse(
            std::move(infer_response));
      } else {
        auto response_iterator =
            std::make_shared<ResponseIterator>(std::move(infer_response));
        response_iterator_map_.insert(
            std::pair<void*, std::shared_ptr<ResponseIterator>>(
                response_iterator->Id(), response_iterator));
      }
    }

    {
      bi::scoped_lock<bi::interprocess_mutex> lock{
          *(ipc_message->ResponseMutex())};
      response_batch->waiting_on_stub = true;
      ipc_message->ResponseCondition()->notify_all();
    }
  }
}

bool
Stub::ParentToStubServiceActive()
{
  return parent_to_stub_thread_;
}

std::shared_ptr<ResponseIterator>
Stub::GetResponseIterator(std::shared_ptr<InferResponse> infer_response)
{
  std::lock_guard<std::mutex> lock(response_iterator_map_mu_);
  if (response_iterator_map_.find(infer_response->Id()) !=
      response_iterator_map_.end()) {
    // Need to re-construct the 'ResponseIterator' and update the
    // 'response_iterator_map_' to make sure the 'ResponseIterator' object has
    // the correct first response.
    auto response_iterator = std::make_shared<ResponseIterator>(infer_response);
    std::vector<std::shared_ptr<InferResponse>> existing_responses =
        response_iterator_map_[infer_response->Id()]->GetExistingResponses();
    for (auto& response : existing_responses) {
      response_iterator->EnqueueResponse(response);
    }

    response_iterator_map_[infer_response->Id()] = response_iterator;
  } else {
    auto response_iterator = std::make_shared<ResponseIterator>(infer_response);
    response_iterator_map_.insert(
        std::pair<void*, std::shared_ptr<ResponseIterator>>(
            response_iterator->Id(), response_iterator));
  }

  return response_iterator_map_[infer_response->Id()];
}

bool
Stub::IsInitialized()
{
  return initialized_;
}

bool
Stub::IsFinalizing()
{
  return finalizing_;
}

void
Stub::EnqueueUtilsMessage(
    std::unique_ptr<UtilsMessagePayload> utils_msg_payload)
{
  {
    std::lock_guard<std::mutex> guard{stub_to_parent_message_mu_};
    stub_to_parent_buffer_.push(std::move(utils_msg_payload));
  }
  stub_to_parent_message_cv_.notify_one();
}

cudaStream_t
Stub::GetProxyStream(const int& device_id)
{
#ifdef TRITON_ENABLE_GPU
  std::lock_guard<std::mutex> lock(dlpack_proxy_stream_pool_mu_);
  if (dlpack_proxy_stream_pool_.find(device_id) ==
      dlpack_proxy_stream_pool_.end()) {
    cudaStream_t new_proxy_stream;
    cudaError_t err = cudaStreamCreate(&new_proxy_stream);
    if (err == cudaSuccess) {
      dlpack_proxy_stream_pool_.emplace(device_id, new_proxy_stream);
      return new_proxy_stream;
    } else {
      throw PythonBackendException(
          "Failed to create a CUDA stream for a DLPack call.");
    }
  }
  return dlpack_proxy_stream_pool_[device_id];
#else
  return nullptr;
#endif
}

std::unique_ptr<Logger> Logger::log_instance_;

std::unique_ptr<Logger>&
Logger::GetOrCreateInstance()
{
  if (Logger::log_instance_.get() == nullptr) {
    Logger::log_instance_ = std::make_unique<Logger>();
  }

  return Logger::log_instance_;
}

// Bound function, called from the python client
void
Logger::Log(const std::string& message, LogLevel level)
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  py::object frame = py::module_::import("inspect").attr("currentframe");
  py::object caller_frame = frame();
  py::object info = py::module_::import("inspect").attr("getframeinfo");
  py::object caller_info = info(caller_frame);
  py::object filename_python = caller_info.attr("filename");
  std::string filename = filename_python.cast<std::string>();
  py::object lineno = caller_info.attr("lineno");
  uint32_t line = lineno.cast<uint32_t>();

  if (!stub->StubToParentServiceActive()) {
    Logger::GetOrCreateInstance()->Log(filename, line, level, message);
  } else {
    std::unique_ptr<PbLog> log_msg(new PbLog(filename, line, message, level));
    stub->EnqueueLogRequest(log_msg);
  }
}

// Called internally (.e.g. LOG_ERROR << "Error"; )
void
Logger::Log(
    const std::string& filename, uint32_t lineno, LogLevel level,
    const std::string& message)
{
  // If the log monitor service is not active yet, format
  // and pass messages to cerr
  if (!BackendLoggingActive()) {
    std::string path(filename);
    size_t pos = path.rfind('/');
    if (pos != std::string::npos) {
      path = path.substr(pos + 1, std::string::npos);
    }
    std::stringstream ss;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    ss << LeadingLogChar(level) << std::setfill('0') << std::setw(2)
       << (tm_time.tm_mon + 1) << std::setw(2) << tm_time.tm_mday << " "
       << std::setw(2) << tm_time.tm_hour << ':' << std::setw(2)
       << tm_time.tm_min << ':' << std::setw(2) << tm_time.tm_sec << "."
       << std::setw(6) << tv.tv_usec << ' ' << static_cast<uint32_t>(getpid())
       << ' ' << path << ':' << lineno << "] ";
    std::cerr << ss.str() << " " << message << std::endl;
  } else {
    // Ensure we do not create a stub instance before it has initialized
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    std::unique_ptr<PbLog> log_msg(new PbLog(filename, lineno, message, level));
    stub->EnqueueLogRequest(log_msg);
  }
}

void
Logger::LogInfo(const std::string& message)
{
  Logger::Log(message, LogLevel::INFO);
}

void
Logger::LogWarn(const std::string& message)
{
  Logger::Log(message, LogLevel::WARNING);
}

void
Logger::LogError(const std::string& message)
{
  Logger::Log(message, LogLevel::ERROR);
}

void
Logger::LogVerbose(const std::string& message)
{
  Logger::Log(message, LogLevel::VERBOSE);
}

const std::string
Logger::LeadingLogChar(const LogLevel& level)
{
  switch (level) {
    case LogLevel::WARNING:
      return "W";
    case LogLevel::ERROR:
      return "E";
    case LogLevel::INFO:
    case LogLevel::VERBOSE:
    default:
      return "I";
  }
}

void
Logger::SetBackendLoggingActive(bool status)
{
  backend_logging_active_ = status;
}

bool
Logger::BackendLoggingActive()
{
  return backend_logging_active_;
}

PYBIND11_EMBEDDED_MODULE(c_python_backend_utils, module)
{
  py::class_<PbError, std::shared_ptr<PbError>> triton_error(
      module, "TritonError");
  py::enum_<TRITONSERVER_Error_Code>(triton_error, "__ErrorCode")
      .value("UNKNOWN", TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNKNOWN)
      .value("INTERNAL", TRITONSERVER_Error_Code::TRITONSERVER_ERROR_INTERNAL)
      .value("NOT_FOUND", TRITONSERVER_Error_Code::TRITONSERVER_ERROR_NOT_FOUND)
      .value(
          "INVALID_ARG",
          TRITONSERVER_Error_Code::TRITONSERVER_ERROR_INVALID_ARG)
      .value(
          "UNAVAILABLE",
          TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNAVAILABLE)
      .value(
          "UNSUPPORTED",
          TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNSUPPORTED)
      .value(
          "ALREADY_EXISTS",
          TRITONSERVER_Error_Code::TRITONSERVER_ERROR_ALREADY_EXISTS)
      .value("CANCELLED", TRITONSERVER_Error_Code::TRITONSERVER_ERROR_CANCELLED)
      .export_values();
  triton_error.def_property_readonly_static(
      "UNKNOWN",
      [](py::object /* self */) { return TRITONSERVER_ERROR_UNKNOWN; });
  triton_error.def_property_readonly_static(
      "INTERNAL",
      [](py::object /* self */) { return TRITONSERVER_ERROR_INTERNAL; });
  triton_error.def_property_readonly_static(
      "NOT_FOUND",
      [](py::object /* self */) { return TRITONSERVER_ERROR_NOT_FOUND; });
  triton_error.def_property_readonly_static(
      "INVALID_ARG",
      [](py::object /* self */) { return TRITONSERVER_ERROR_INVALID_ARG; });
  triton_error.def_property_readonly_static(
      "UNAVAILABLE",
      [](py::object /* self */) { return TRITONSERVER_ERROR_UNAVAILABLE; });
  triton_error.def_property_readonly_static(
      "UNSUPPORTED",
      [](py::object /* self */) { return TRITONSERVER_ERROR_UNSUPPORTED; });
  triton_error.def_property_readonly_static(
      "ALREADY_EXISTS",
      [](py::object /* self */) { return TRITONSERVER_ERROR_ALREADY_EXISTS; });
  triton_error.def_property_readonly_static(
      "CANCELLED",
      [](py::object /* self */) { return TRITONSERVER_ERROR_CANCELLED; });
  triton_error.def(
      py::init<const std::string&, TRITONSERVER_Error_Code>(),
      py::arg("message").none(false),
      py::arg("code").none(false) = TRITONSERVER_ERROR_INTERNAL);
  triton_error.def("code", &PbError::Code);
  triton_error.def("message", &PbError::Message);

  py::class_<PreferredMemory, std::shared_ptr<PreferredMemory>>(
      module, "PreferredMemory")
      .def(
          py::init<const PreferredMemory::MemoryType&, const int64_t&>(),
          py::arg("preferred_memory_type").none(false),
          py::arg("preferred_device_id").none(false) = 0);

  py::enum_<PreferredMemory::MemoryType>(module, "MemoryType")
      .value("TRITONSERVER_MEMORY_GPU", PreferredMemory::MemoryType::GPU)
      .value("TRITONSERVER_MEMORY_CPU", PreferredMemory::MemoryType::CPU)
      .export_values();

  py::class_<InferenceTrace, std::shared_ptr<InferenceTrace>>(
      module, "InferenceTrace");

  py::class_<InferRequest, std::shared_ptr<InferRequest>>(
      module, "InferenceRequest")
      .def(
          py::init([](const std::string& request_id, uint64_t correlation_id,
                      const std::vector<std::shared_ptr<PbTensor>>& inputs,
                      const std::vector<std::string>& requested_output_names,
                      const std::string& model_name,
                      const int64_t model_version, const uint32_t flags,
                      const int32_t timeout,
                      const PreferredMemory& preferred_memory,
                      const InferenceTrace& trace) {
            std::set<std::string> requested_outputs;
            for (auto& requested_output_name : requested_output_names) {
              requested_outputs.emplace(requested_output_name);
            }
            // FIXME: InferenceRequest parameters are not supported in BLS now.
            return std::make_shared<InferRequest>(
                request_id, correlation_id, inputs, requested_outputs,
                model_name, model_version, "" /*parameters*/, flags, timeout,
                0 /*response_factory_address*/, 0 /*request_address*/,
                preferred_memory, trace);
          }),
          py::arg("request_id").none(false) = "",
          py::arg("correlation_id").none(false) = 0,
          py::arg("inputs").none(false),
          py::arg("requested_output_names").none(false),
          py::arg("model_name").none(false),
          py::arg("model_version").none(false) = -1,
          py::arg("flags").none(false) = 0, py::arg("timeout").none(false) = 0,
          py::arg("preferred_memory").none(false) =
              PreferredMemory(PreferredMemory::DEFAULT, 0),
          py::arg("trace").none(false) = InferenceTrace())
      .def(
          "inputs", &InferRequest::Inputs,
          py::return_value_policy::reference_internal)
      .def("request_id", &InferRequest::RequestId)
      .def("correlation_id", &InferRequest::CorrelationId)
      .def("flags", &InferRequest::Flags)
      .def("set_flags", &InferRequest::SetFlags)
      .def("timeout", &InferRequest::Timeout)
      .def("parameters", &InferRequest::Parameters)
      .def("trace", &InferRequest::Trace)
      .def(
          "exec",
          [](std::shared_ptr<InferRequest>& infer_request,
             const bool decoupled) {
            std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
            std::shared_ptr<InferResponse> response =
                infer_request->Exec(decoupled);
            py::object response_object;
            if (decoupled) {
              auto response_iterator = stub->GetResponseIterator(response);
              response_object = py::cast(response_iterator);
            } else {
              response_object = py::cast(response);
            }

            return response_object;
          },
          py::arg("decoupled").none(false) = false)
      .def(
          "async_exec",
          [](std::shared_ptr<InferRequest>& infer_request,
             const bool decoupled) {
            std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
            if (stub->IsDecoupled()) {
              throw PythonBackendException(
                  "Async BLS request execution is not support in the decoupled "
                  "API.");
            }
            py::object loop =
                py::module_::import("asyncio").attr("get_running_loop")();
            py::cpp_function callback = [&stub, infer_request, decoupled]() {
              std::shared_ptr<InferResponse> response =
                  infer_request->Exec(decoupled);
              py::object response_object;
              if (decoupled) {
                auto response_iterator = stub->GetResponseIterator(response);
                response_object = py::cast(response_iterator);
              } else {
                response_object = py::cast(response);
              }

              return response_object;
            };
            py::object future =
                loop.attr("run_in_executor")(py::none(), callback);
            return future;
          },
          py::arg("decoupled").none(false) = false)
      .def(
          "requested_output_names", &InferRequest::RequestedOutputNames,
          py::return_value_policy::reference_internal)
      .def("get_response_sender", &InferRequest::GetResponseSender)
      .def("is_cancelled", &InferRequest::IsCancelled);

  py::class_<PbTensor, std::shared_ptr<PbTensor>>(module, "Tensor")
      .def(py::init(&PbTensor::FromNumpy))
      .def("name", &PbTensor::Name)
      // The reference_internal is added to make sure that the NumPy object has
      // the same lifetime as the tensor object. This means even when the NumPy
      // object is only in scope, the tensor object is not deallocated from
      // shared memory to make sure the NumPy object is still valid.
      .def(
          "as_numpy", &PbTensor::AsNumpy,
          py::return_value_policy::reference_internal)
      .def("triton_dtype", &PbTensor::TritonDtype)
      .def("to_dlpack", &PbTensor::ToDLPack)
      .def("is_cpu", &PbTensor::IsCPU)
      .def("shape", &PbTensor::Dims)
      .def("from_dlpack", &PbTensor::FromDLPack)
      .def("__dlpack__", &PbTensor::DLPack, py::arg("stream") = py::none())
      .def("__dlpack_device__", &PbTensor::DLPackDevice);

  py::class_<InferResponse, std::shared_ptr<InferResponse>>(
      module, "InferenceResponse")
      .def(
          py::init<
              const std::vector<std::shared_ptr<PbTensor>>&,
              std::shared_ptr<PbError>>(),
          py::arg("output_tensors") = py::list(),
          py::arg("error") = static_cast<std::shared_ptr<PbError>>(nullptr))
      .def(
          "output_tensors", &InferResponse::OutputTensors,
          py::return_value_policy::reference)
      .def("has_error", &InferResponse::HasError)
      .def("error", &InferResponse::Error);

  py::class_<ResponseSender, std::shared_ptr<ResponseSender>>(
      module, "InferenceResponseSender")
      .def(
          "send", &ResponseSender::Send, py::arg("response") = nullptr,
          py::arg("flags") = 0)
      .def("is_cancelled", &ResponseSender::IsCancelled);

  py::class_<ResponseIterator, std::shared_ptr<ResponseIterator>>(
      module, "ResponseIterator")
      .def(py::init<const std::shared_ptr<InferResponse>&>())
      .def(
          "__iter__",
          [](ResponseIterator& it) -> ResponseIterator& {
            it.Iter();
            return it;
          })
      .def("__next__", &ResponseIterator::Next);

  py::class_<Logger> logger(module, "Logger");
  py::enum_<LogLevel>(logger, "LogLevel")
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR)
      .value("VERBOSE", LogLevel::VERBOSE)
      .export_values();
  logger.def_static(
      "log", py::overload_cast<const std::string&, LogLevel>(&Logger::Log),
      py::arg("message"), py::arg("level") = LogLevel::INFO);
  logger.def_static("log_info", &Logger::LogInfo, py::arg("message"));
  logger.def_static("log_warn", &Logger::LogWarn, py::arg("message"));
  logger.def_static("log_error", &Logger::LogError, py::arg("message"));
  logger.def_static("log_verbose", &Logger::LogVerbose, py::arg("message"));

  py::class_<Metric, std::shared_ptr<Metric>>(module, "Metric")
      .def("increment", &Metric::SendIncrementRequest)
      .def("set", &Metric::SendSetValueRequest)
      .def("value", &Metric::SendGetValueRequest);

  py::enum_<MetricKind>(module, "MetricKind")
      .value("COUNTER", MetricKind::COUNTER)
      .value("GAUGE", MetricKind::GAUGE)
      .export_values();

  py::class_<MetricFamily, std::shared_ptr<MetricFamily>>(
      module, "MetricFamily")
      .def(
          py::init(&MetricFamily::CreateMetricFamily),
          py::arg("name").none(false), py::arg("description").none(false),
          py::arg("kind").none(false))
      .def(
          "Metric", &MetricFamily::CreateMetric,
          py::arg("labels").none(true) = py::none());
  module.attr("MetricFamily").attr("COUNTER") = MetricKind::COUNTER;
  module.attr("MetricFamily").attr("GAUGE") = MetricKind::GAUGE;

  module.def(
      "load_model", &LoadModel, py::arg("model_name").none(false),
      py::arg("config").none(false) = "",
      py::arg("files").none(true) = py::none());
  module.def(
      "unload_model", &UnloadModel, py::arg("model_name").none(false),
      py::arg("unload_dependents").none(false) = false);
  module.def(
      "is_model_ready", &IsModelReady, py::arg("model_name").none(false),
      py::arg("model_version").none(false) = "");

  // This class is not part of the public API for Python backend. This is only
  // used for internal testing purposes.
  py::class_<SharedMemoryManager>(module, "SharedMemory")
      .def("free_memory", &SharedMemoryManager::FreeMemory);

  py::register_exception<PythonBackendException>(
      module, "TritonModelException");
}


void
ModelContext::Init(
    const std::string& model_path, const std::string& runtime_modeldir,
    const std::string& triton_install_path, const std::string& model_version)
{
  type_ = ModelType::DEFAULT;
  if (runtime_modeldir != "DEFAULT") {
    // For python based backends, existence of `model.py` in the corresponding
    // backend folder happens on the core side, so we can omit this check here.
    python_model_path_ = runtime_modeldir + "/model.py";
    type_ = ModelType::BACKEND;
  } else {
    python_model_path_ = model_path;
    // Check if model file exists in this path.
    struct stat buffer;
    if (stat(python_model_path_.c_str(), &buffer) != 0) {
      throw PythonBackendException(
          ("Python model file not found in \'" + model_path + "\'"));
    }
  }

  model_dir_ = model_path.substr(0, model_path.find_last_of("\\/"));
  python_backend_folder_ = triton_install_path;
  model_version_ = model_version;
  runtime_modeldir_ = runtime_modeldir;
}

void
ModelContext::StubSetup(py::module& sys)
{
  std::string model_name =
      python_model_path_.substr(python_model_path_.find_last_of("/") + 1);

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

  if (type_ == ModelType::DEFAULT) {
    std::string model_path_parent =
        python_model_path_.substr(0, python_model_path_.find_last_of("/"));
    std::string model_path_parent_parent =
        model_path_parent.substr(0, model_path_parent.find_last_of("/"));
    sys.attr("path").attr("append")(model_path_parent);
    sys.attr("path").attr("append")(model_path_parent_parent);
    sys.attr("path").attr("append")(python_backend_folder_);
    sys = py::module_::import(
        (std::string(model_version_) + "." + model_name_trimmed).c_str());
  } else {
    std::string model_path_parent =
        python_model_path_.substr(0, python_model_path_.find_last_of("/"));
    std::string backend_model_dir(model_path_parent);
    sys.attr("path").attr("append")(backend_model_dir);
    sys.attr("path").attr("append")(python_backend_folder_);
    sys = py::module_::import(model_name_trimmed.c_str());
  }
}


extern "C" {

int
main(int argc, char** argv)
{
  std::unique_ptr<Logger>& logger = Logger::GetOrCreateInstance();
  if (argc < 9) {
    LOG_INFO << "Expected 9 arguments, found " << argc << " arguments.";
    logger.reset();
    exit(1);
  }
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Path to model
  std::string model_path = argv[1];
  std::string shm_region_name = argv[2];
  int64_t shm_default_size = std::stol(argv[3]);

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
    logger.reset();
    exit(1);
  }
  std::string model_version = model_path_tokens[model_path_tokens.size() - 2];
  int64_t shm_growth_size = std::stol(argv[4]);
  std::string triton_install_path = argv[6];
  std::string name = argv[8];
  std::string runtime_modeldir = argv[9];

  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  try {
    stub->Instantiate(
        shm_growth_size, shm_default_size, shm_region_name, model_path,
        model_version, argv[6] /* triton install path */,
        std::stoi(argv[7]) /* IPCControl handle */, name, runtime_modeldir);
  }
  catch (const PythonBackendException& pb_exception) {
    LOG_INFO << "Failed to preinitialize Python stub: " << pb_exception.what();
    logger.reset();
    exit(1);
  }

  // Start the Python Interpreter
  py::scoped_interpreter guard{};
  pid_t parent_pid = std::stoi(argv[5]);

  std::atomic<bool> background_thread_running = {true};
  std::thread background_thread =
      std::thread([&parent_pid, &background_thread_running, &stub, &logger] {
        // Send a dummy message after the stub process is launched to notify the
        // parent process that the health thread has started.
        std::unique_ptr<IPCMessage> ipc_message = IPCMessage::Create(
            stub->SharedMemory(), false /* inline_response */);
        stub->SendIPCMessage(ipc_message);

        while (background_thread_running) {
          // Every 300ms set the health variable to true. This variable is in
          // shared memory and will be set to false by the parent process.
          // The parent process expects that the stub process sets this
          // variable to true within 1 second.
          std::this_thread::sleep_for(std::chrono::milliseconds(300));

          stub->UpdateHealth();

          if (kill(parent_pid, 0) != 0) {
            // When unhealthy, we should stop attempting to send
            // messages to the backend ASAP.
            if (stub->StubToParentServiceActive()) {
              stub->TerminateStubToParentQueueMonitor();
            }
            if (stub->ParentToStubServiceActive()) {
              stub->TerminateParentToStubQueueMonitor();
            }
            // Destroy Stub
            LOG_INFO << "Non-graceful termination detected. ";
            background_thread_running = false;
            non_graceful_exit = true;

            // Destroy stub and exit.
            logger.reset();
            stub.reset();
            exit(1);
          }
        }
      });

  // The stub process will always keep listening for new notifications from the
  // parent process. After the notification is received the stub process will
  // run the appropriate command and wait for new notifications.
  bool finalize = false;
  while (true) {
    if (finalize) {
      stub->Finalize();
      // Need check or may receive not joinable error
      if (stub->StubToParentServiceActive()) {
        stub->TerminateStubToParentQueueMonitor();
      }
      if (stub->ParentToStubServiceActive()) {
        stub->TerminateParentToStubQueueMonitor();
      }
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
  logger.reset();
  stub.reset();

  return 0;
}
}
}}}  // namespace triton::backend::python
