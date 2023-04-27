// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "stub_launcher.h"
#include "python_be.h"

namespace triton { namespace backend { namespace python {

StubLauncher::StubLauncher(const std::string stub_process_kind)
    : parent_pid_(0), stub_pid_(0), is_initialized_(false),
      stub_process_kind_(stub_process_kind), model_instance_name_(""),
      device_id_(0), kind_("")

{
}

StubLauncher::StubLauncher(
    const std::string stub_process_kind, const std::string model_instance_name,
    const int32_t device_id, const std::string kind)
    : parent_pid_(0), stub_pid_(0), is_initialized_(false),
      stub_process_kind_(stub_process_kind),
      model_instance_name_(model_instance_name), device_id_(device_id),
      kind_(kind)
{
}

TRITONSERVER_Error*
StubLauncher::Initialize(ModelState* model_state)
{
  model_name_ = model_state->Name();
  shm_default_byte_size_ =
      model_state->StateForBackend()->shm_default_byte_size;
  shm_growth_byte_size_ = model_state->StateForBackend()->shm_growth_byte_size;
  shm_message_queue_size_ =
      model_state->StateForBackend()->shm_message_queue_size;
  python_execution_env_ = model_state->PythonExecutionEnv();
  python_lib_ = model_state->StateForBackend()->python_lib;
  model_state->ModelConfig().Write(&model_config_buffer_);
  is_decoupled_ = model_state->IsDecoupled();
  model_repository_path_ = model_state->RepositoryPath();

  // Atomically increase and read the stub process count to avoid shared memory
  // region name collision
  int num_init = ++model_state->StateForBackend()->number_of_instance_inits;
  shm_region_name_ =
      model_state->StateForBackend()->shared_memory_region_prefix +
      std::to_string(num_init);

  model_version_ = model_state->Version();

  std::stringstream ss;
  std::string artifact_name;
  RETURN_IF_ERROR(model_state->ModelConfig().MemberAsString(
      "default_model_filename", &artifact_name));
  ss << model_repository_path_ << "/" << model_version_ << "/";

  if (artifact_name.size() > 0) {
    ss << artifact_name;
  } else {
    // Default artifact name.
    ss << "model.py";
  }

  model_path_ = ss.str();
  struct stat buffer;

  // Check if model.py exists
  if (stat(model_path_.c_str(), &buffer) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("model.py does not exist in the model repository path: " + model_path_)
            .c_str());
  }

  // Path to the extracted Python env
  std::string python_execution_env = "";
  if (python_execution_env_ != "") {
    try {
      python_execution_env =
          model_state->StateForBackend()->env_manager->ExtractIfNotExtracted(
              python_execution_env_);
    }
    catch (PythonBackendException& pb_exception) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
    }

    path_to_activate_ = python_execution_env + "/bin/activate";
    path_to_libpython_ = python_execution_env + "/lib";
    if (python_execution_env.length() > 0 && !FileExists(path_to_activate_)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Path " + path_to_activate_ +
           " does not exist. The Python environment should contain an "
           "'activate' script.")
              .c_str());
    }
  }

  parent_pid_ = getpid();

  return nullptr;
}

TRITONSERVER_Error*
StubLauncher::Setup()
{
  // Destruct any in-use shared memory object before starting the stub process.
  ipc_control_ = nullptr;
  stub_message_queue_ = nullptr;
  parent_message_queue_ = nullptr;
  stub_to_parent_mq_ = nullptr;
  parent_to_stub_mq_ = nullptr;
  memory_manager_ = nullptr;

  try {
    // It is necessary for restart to make sure that the previous shared memory
    // pool is destructed before the new pool is created.
    shm_pool_ = nullptr;
    shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, shm_default_byte_size_, shm_growth_byte_size_,
        true /* create */);
  }
  catch (const PythonBackendException& pb_exception) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
  }

  AllocatedSharedMemory<IPCControlShm> current_ipc_control =
      shm_pool_->Construct<IPCControlShm>();
  ipc_control_ = std::move(current_ipc_control.data_);
  ipc_control_handle_ = current_ipc_control.handle_;

  RETURN_IF_EXCEPTION(
      stub_message_queue_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
  RETURN_IF_EXCEPTION(
      parent_message_queue_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
  RETURN_IF_EXCEPTION(
      stub_to_parent_mq_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
  RETURN_IF_EXCEPTION(
      parent_to_stub_mq_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));

  std::unique_ptr<MessageQueue<intptr_t>> memory_manager_message_queue;
  RETURN_IF_EXCEPTION(
      memory_manager_message_queue =
          MessageQueue<intptr_t>::Create(shm_pool_, shm_message_queue_size_));

  memory_manager_message_queue->ResetSemaphores();
  ipc_control_->memory_manager_message_queue =
      memory_manager_message_queue->ShmHandle();
  ipc_control_->decoupled = is_decoupled_;

  memory_manager_ =
      std::make_unique<MemoryManager>(std::move(memory_manager_message_queue));
  ipc_control_->parent_message_queue = parent_message_queue_->ShmHandle();
  ipc_control_->stub_to_parent_mq = stub_to_parent_mq_->ShmHandle();
  ipc_control_->stub_message_queue = stub_message_queue_->ShmHandle();
  ipc_control_->parent_to_stub_mq = parent_to_stub_mq_->ShmHandle();

  new (&(ipc_control_->stub_health_mutex)) bi::interprocess_mutex;
  health_mutex_ = &(ipc_control_->stub_health_mutex);

  stub_message_queue_->ResetSemaphores();
  parent_message_queue_->ResetSemaphores();
  stub_to_parent_mq_->ResetSemaphores();
  parent_to_stub_mq_->ResetSemaphores();

  is_initialized_ = false;

  return nullptr;
}

TRITONSERVER_Error*
StubLauncher::Launch()
{
  std::string stub_name;
  if (stub_process_kind_ == "AUTOCOMPLETE_STUB") {
    stub_name = model_name_;
  } else {
    stub_name = model_instance_name_;
  }

  const char* stub_args[4];
  stub_args[0] = "bash";
  stub_args[1] = "-c";
  stub_args[3] = nullptr;  // Last argument must be nullptr

  // Default Python backend stub
  std::string python_backend_stub = python_lib_ + "/triton_python_backend_stub";

  // Path to alternative Python backend stub
  std::string model_python_backend_stub =
      std::string(model_repository_path_) + "/triton_python_backend_stub";

  if (FileExists(model_python_backend_stub)) {
    python_backend_stub = model_python_backend_stub;
  }

  std::string bash_argument;

  // This shared memory variable indicates whether the stub process should
  // revert the LD_LIBRARY_PATH changes to avoid shared library issues in
  // executables and libraries.
  ipc_control_->uses_env = false;
  if (python_execution_env_ != "") {
    std::stringstream ss;

    // Need to properly set the LD_LIBRARY_PATH so that Python environments
    // using different python versions load properly.
    ss << "source " << path_to_activate_
       << " && exec env LD_LIBRARY_PATH=" << path_to_libpython_
       << ":$LD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
       << " " << shm_region_name_ << " " << shm_default_byte_size_ << " "
       << shm_growth_byte_size_ << " " << parent_pid_ << " " << python_lib_
       << " " << ipc_control_handle_ << " " << stub_name;
    ipc_control_->uses_env = true;
    bash_argument = ss.str();
  } else {
    std::stringstream ss;
    ss << " exec " << python_backend_stub << " " << model_path_ << " "
       << shm_region_name_ << " " << shm_default_byte_size_ << " "
       << shm_growth_byte_size_ << " " << parent_pid_ << " " << python_lib_
       << " " << ipc_control_handle_ << " " << stub_name;
    bash_argument = ss.str();
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Starting Python backend stub: ") + bash_argument).c_str());

  stub_args[2] = bash_argument.c_str();

  int stub_status_code =
      system((python_backend_stub + "> /dev/null 2>&1").c_str());

  // If running stub process without any arguments returns any status code,
  // other than 1, it can indicate a permission issue as a result of
  // downloading the stub process from a cloud object storage service.
  if (WEXITSTATUS(stub_status_code) != 1) {
    // Give the execute permission for the triton_python_backend_stub to the
    // owner.
    int error = chmod(python_backend_stub.c_str(), S_IXUSR);
    if (error != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to give execute permission to "
                       "triton_python_backend_stub in ") +
           python_backend_stub + " " + stub_name +
           " Error No.: " + std::to_string(error))
              .c_str());
    }
  }

  pid_t pid = fork();
  if (pid < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to fork the stub process for auto-complete.");
  }
  if (pid == 0) {
    // Replace this child process with the new stub process.
    execvp("bash", (char**)stub_args);
    // execvp() never return if succeeded. Otherwise, an error has occured.
    std::stringstream ss;
    ss << "Failed to run python backend stub. Errno = " << errno << '\n'
       << "Python backend stub path: " << python_backend_stub << '\n'
       << "Shared Memory Region Name: " << shm_region_name_ << '\n'
       << "Shared Memory Default Byte Size: " << shm_default_byte_size_ << '\n'
       << "Shared Memory Growth Byte Size: " << shm_growth_byte_size_ << '\n';
    // Print the error message directly because the underlying mutexes in
    // LOG_MESSAGE() could be forked when it is locked by other thread(s).
    std::cerr << '\n' << ss.str() << '\n';
    // Terminate the child execution immediately to avoid any issues.
    _Exit(1);
  } else {
    ScopedDefer _([&] {
      // Push a dummy message to the message queue so that the stub
      // process is notified that it can release the object stored in
      // shared memory.
      stub_message_queue_->Push(DUMMY_MESSAGE);

      // If the model is not initialized, wait for the stub process to exit.
      if (!is_initialized_) {
        int status;
        stub_message_queue_.reset();
        parent_message_queue_.reset();
        memory_manager_.reset();
        waitpid(stub_pid_, &status, 0);
      }
    });

    stub_pid_ = pid;

    // The stub process would send two messages to the parent process during the
    // initialization.
    // 1. When the stub process's health monitoring thread has started.
    // 2. When the initialization is fully completed and the Python model is
    // loaded.
    //
    // The reason it is broken into two steps is that creation of the health
    // monitoring thread may take longer which can make the server process think
    // that the stub process is unhealthy and return early. Waiting until the
    // health thread is spawn would make sure would prevent this issue.
    parent_message_queue_->Pop();

    if (stub_process_kind_ == "AUTOCOMPLETE_STUB") {
      try {
        AutocompleteStubProcess();
      }
      catch (const PythonBackendException& ex) {
        // Need to kill the stub process first
        kill(stub_pid_, SIGKILL);
        int status;
        waitpid(stub_pid_, &status, 0);
        stub_pid_ = 0;
        throw BackendModelException(
            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, ex.what()));
      }
    } else if (stub_process_kind_ == "MODEL_INSTANCE_STUB") {
      RETURN_IF_ERROR(ModelInstanceStubProcess());
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Unknown stub_process_kind: ") + stub_process_kind_)
              .c_str());
    }

    is_initialized_ = true;
  }

  return nullptr;
}

void
StubLauncher::AutocompleteStubProcess()
{
  std::string model_config = model_config_buffer_.MutableContents();

  std::unique_ptr<IPCMessage> auto_complete_message =
      IPCMessage::Create(shm_pool_, false /* inline_response */);
  auto_complete_message->Command() = PYTHONSTUB_AutoCompleteRequest;

  std::unique_ptr<PbString> pb_string =
      PbString::Create(shm_pool_, model_config);
  bi::managed_external_buffer::handle_t string_handle = pb_string->ShmHandle();

  auto_complete_message->Args() = string_handle;
  stub_message_queue_->Push(auto_complete_message->ShmHandle());

  std::unique_ptr<IPCMessage> auto_complete_response_message =
      IPCMessage::LoadFromSharedMemory(shm_pool_, parent_message_queue_->Pop());

  if (auto_complete_response_message->Command() !=
      PYTHONSTUB_AutoCompleteResponse) {
    throw PythonBackendException(
        "Received unexpected response from Python backend stub: " +
        model_name_);
  }

  auto auto_complete_response =
      std::move((shm_pool_->Load<AutoCompleteResponseShm>(
                    auto_complete_response_message->Args())))
          .data_;

  if (auto_complete_response->response_has_error) {
    if (auto_complete_response->response_is_error_set) {
      std::unique_ptr<PbString> error_message = PbString::LoadFromSharedMemory(
          shm_pool_, auto_complete_response->response_error);
      throw PythonBackendException(error_message->String());
    } else {
      throw PythonBackendException("Auto-complete failed for " + model_name_);
    }
  }

  if (auto_complete_response->response_has_model_config) {
    std::unique_ptr<PbString> auto_complete_config =
        PbString::LoadFromSharedMemory(
            shm_pool_, auto_complete_response->response_model_config);
    std::string auto_complete_config_string = auto_complete_config->String();
    if (!auto_complete_config_string.empty()) {
      TRITONSERVER_Error* err =
          auto_complete_config_.Parse(auto_complete_config_string);
      if (err != nullptr) {
        throw PythonBackendException("Failed to parse auto-complete JSON.");
      }
    }
  }
}

TRITONSERVER_Error*
StubLauncher::ModelInstanceStubProcess()
{
  std::unordered_map<std::string, std::string> initialize_map = {
      {"model_config", model_config_buffer_.MutableContents()},
      {"model_instance_kind", kind_},
      {"model_instance_name", model_instance_name_},
      {"model_instance_device_id", std::to_string(device_id_)},
      {"model_repository", model_repository_path_},
      {"model_version", std::to_string(model_version_)},
      {"model_name", model_name_}};

  std::unique_ptr<IPCMessage> initialize_message =
      IPCMessage::Create(shm_pool_, false /* inline_response */);
  initialize_message->Command() = PYTHONSTUB_InitializeRequest;

  std::unique_ptr<PbMap> pb_map = PbMap::Create(shm_pool_, initialize_map);
  bi::managed_external_buffer::handle_t initialize_map_handle =
      pb_map->ShmHandle();

  initialize_message->Args() = initialize_map_handle;
  stub_message_queue_->Push(initialize_message->ShmHandle());

  bi::managed_external_buffer::handle_t message;
  RETURN_IF_ERROR(ReceiveMessageFromStub(message));

  std::unique_ptr<IPCMessage> initialize_response_message =
      IPCMessage::LoadFromSharedMemory(shm_pool_, message);

  if (initialize_response_message->Command() != PYTHONSTUB_InitializeResponse) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string(
             "Received unexpected response from Python backend stub: ") +
         model_instance_name_)
            .c_str());
  }

  auto initialize_response =
      std::move((shm_pool_->Load<InitializeResponseShm>(
                    initialize_response_message->Args())))
          .data_;

  if (initialize_response->response_has_error) {
    if (initialize_response->response_is_error_set) {
      std::unique_ptr<PbString> error_message = PbString::LoadFromSharedMemory(
          shm_pool_, initialize_response->response_error);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, error_message->String().c_str());
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Launch stub process failed for ") + model_name_)
              .c_str());
    }
  }

  return nullptr;
}

void
StubLauncher::UpdateHealth()
{
  is_healthy_ = false;
  if (is_initialized_) {
    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      ipc_control_->stub_health = false;
    }

    // Sleep 1 second so that the child process has a chance to change the
    // health variable
    sleep(1);

    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      is_healthy_ = ipc_control_->stub_health;
    }
  }
}

void
StubLauncher::TerminateStub()
{
  if (is_initialized_) {
    bool force_kill = false;
    if (is_healthy_) {
      // Finalize command does not have any arguments.
      std::unique_ptr<IPCMessage> ipc_message =
          IPCMessage::Create(shm_pool_, false /* inline_response */);

      ipc_message->Command() = PYTHONSTUB_FinalizeRequest;
      stub_message_queue_->Push(ipc_message->ShmHandle());
      parent_message_queue_->Pop();

      stub_message_queue_.reset();
      parent_message_queue_.reset();
      memory_manager_.reset();
    } else {
      force_kill = true;
    }

    int status;
    if (force_kill) {
      kill(stub_pid_, SIGKILL);
    }
    waitpid(stub_pid_, &status, 0);
  }

  // First destroy the IPCControl. This makes sure that IPCControl is
  // destroyed before the shared memory manager goes out of scope.
  ipc_control_.reset();
  stub_message_queue_.reset();
  parent_message_queue_.reset();
  memory_manager_.reset();
}

void
StubLauncher::ClearQueues()
{
  stub_to_parent_mq_.reset();
  parent_to_stub_mq_.reset();
}

void
StubLauncher::KillStubProcess()
{
  kill(stub_pid_, SIGKILL);
  int status;
  waitpid(stub_pid_, &status, 0);
  stub_pid_ = 0;
}

TRITONSERVER_Error*
StubLauncher::ReceiveMessageFromStub(
    bi::managed_external_buffer::handle_t& message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

      // Check if lock has been acquired.
      if (lock) {
        ipc_control_->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    message = parent_message_queue_->Pop(
        timeout_miliseconds /* duration ms */, success);

    bool is_stub_alive = false;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() + boost::posix_time::seconds(1);
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);
      if (lock) {
        is_stub_alive = ipc_control_->stub_health;
      } else {
        // If It failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        is_stub_alive = false;
      }
    }

    if (!success && !is_stub_alive) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Stub process '") + model_instance_name_ +
           "' is not healthy.")
              .c_str());
    }
  }

  return nullptr;  // success
}
}}};  // namespace triton::backend::python
