// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "python_be.h"

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

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
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::CheckIncomingRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    size_t& total_batch_size)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int max_batch_size = model_state->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "null request given to Python backend for '" + Name() + "'")
              .c_str());
    }
  }

  for (size_t i = 0; i < request_count; i++) {
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
        return err;
      }
    } else {
      ++total_batch_size;
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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "batch size " + std::to_string(total_batch_size) + " for '" +
            Name() + "', max allowed is " + std::to_string(max_batch_size))
            .c_str());
  }

  return nullptr;  // success
}

bool
ModelInstanceState::ExistsInClosedRequests(intptr_t closed_request)
{
  std::lock_guard<std::mutex> guard{closed_requests_mutex_};
  return std::find(
             closed_requests_.begin(), closed_requests_.end(),
             closed_request) != closed_requests_.end();
}

void
ModelInstanceState::SetErrorForResponseSendMessage(
    ResponseSendMessage* response_send_message,
    std::shared_ptr<TRITONSERVER_Error*> error,
    std::unique_ptr<PbString>& error_message)
{
  if (error && *error != nullptr) {
    response_send_message->has_error = true;
    LOG_IF_EXCEPTION(
        error_message = PbString::Create(
            Stub()->ShmPool(), TRITONSERVER_ErrorMessage(*error)));
    response_send_message->error = error_message->ShmHandle();
    response_send_message->is_error_set = true;
  }
}

void
ModelInstanceState::SendMessageAndReceiveResponse(
    bi::managed_external_buffer::handle_t message,
    bi::managed_external_buffer::handle_t& response, bool& restart,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  auto error = SendMessageToStub(message);
  if (error != nullptr) {
    restart = true;
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  bi::managed_external_buffer::handle_t response_message;
  error = ReceiveMessageFromStub(response_message);
  if (error != nullptr) {
    restart = true;
    RespondErrorToAllRequests(
        TRITONSERVER_ErrorMessage(error), responses, requests, request_count);

    return;
  }

  response = response_message;
}

TRITONSERVER_Error*
ModelInstanceState::SendMessageToStub(
    bi::managed_external_buffer::handle_t message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(
          *(Stub()->HealthMutex()), timeout);

      // Check if lock has been acquired.
      if (lock) {
        Stub()->IpcControl()->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    Stub()->StubMessageQueue()->Push(
        message, timeout_miliseconds /* duration ms */, success);

    if (!success && !IsStubProcessAlive()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Stub process is not healthy.");
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ReceiveMessageFromStub(
    bi::managed_external_buffer::handle_t& message)
{
  bool success = false;
  while (!success) {
    uint64_t timeout_miliseconds = 1000;
    {
      boost::posix_time::ptime timeout =
          boost::get_system_time() +
          boost::posix_time::milliseconds(timeout_miliseconds);

      bi::scoped_lock<bi::interprocess_mutex> lock(
          *Stub()->HealthMutex(), timeout);

      // Check if lock has been acquired.
      if (lock) {
        Stub()->IpcControl()->stub_health = false;
      } else {
        // If it failed to obtain the lock, it means that the stub has been
        // stuck or exited while holding the health mutex lock.
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to obtain the health mutex.");
      }
    }

    message = Stub()->ParentMessageQueue()->Pop(
        timeout_miliseconds /* duration ms */, success);

    if (!success && !IsStubProcessAlive()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "Stub process is not healthy.");
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::RespondErrorToAllRequests(
    const char* message,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    if ((*responses)[r] == nullptr)
      continue;

    std::string err_message =
        std::string(
            "Failed to process the request(s) for model instance '" + Name() +
            "', message: ") +
        message;

    TRITONSERVER_Error* err =
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message.c_str());
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");

    (*responses)[r] = nullptr;
    TRITONSERVER_ErrorDelete(err);
  }
}

void
ModelInstanceState::WaitForBLSRequestsToFinish()
{
  futures_.clear();
}

bool
ModelInstanceState::IsStubProcessAlive()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::seconds(1);
  bi::scoped_lock<bi::interprocess_mutex> lock(*Stub()->HealthMutex(), timeout);

  // Check if lock has been acquired.
  if (lock) {
    return Stub()->IpcControl()->stub_health;
  } else {
    // If It failed to obtain the lock, it means that the stub has been
    // stuck or exited while holding the health mutex lock.
    return false;
  }
}

TRITONSERVER_Error*
ModelInstanceState::SaveRequestsToSharedMemory(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<std::unique_ptr<InferRequest>>& pb_inference_requests,
    AllocatedSharedMemory<char>& request_batch,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
  // Clear any existing items in the requests vector
  pb_inference_requests.clear();

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  RETURN_IF_EXCEPTION(
      request_batch = Stub()->ShmPool()->Construct<char>(
          sizeof(RequestBatch) +
          request_count * sizeof(bi::managed_external_buffer::handle_t)));

  RequestBatch* request_batch_shm_ptr =
      reinterpret_cast<RequestBatch*>(request_batch.data_.get());
  request_batch_shm_ptr->batch_size = request_count;

  bi::managed_external_buffer::handle_t* requests_shm =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          request_batch.data_.get() + sizeof(RequestBatch));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_input_count = 0;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    uint32_t requested_output_count = 0;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    std::vector<std::shared_ptr<PbTensor>> pb_input_tensors;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      std::shared_ptr<PbTensor> pb_input_tensor;

      RETURN_IF_ERROR(
          GetInputTensor(iidx, pb_input_tensor, request, responses));
      pb_input_tensors.emplace_back(std::move(pb_input_tensor));
    }

    std::set<std::string> requested_output_names;
    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(
          request, iidx, &requested_output_name));
      requested_output_names.emplace(requested_output_name);
    }

    // request id
    const char* id;
    RETURN_IF_ERROR(TRITONBACKEND_RequestId(request, &id));

    uint64_t correlation_id;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    uint32_t flags;
    RETURN_IF_ERROR(TRITONBACKEND_RequestFlags(request, &flags));

    std::unique_ptr<InferRequest> infer_request;
    if (model_state->IsDecoupled()) {
      TRITONBACKEND_ResponseFactory* factory_ptr;
      RETURN_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request));
      infer_request = std::make_unique<InferRequest>(
          id, correlation_id, pb_input_tensors, requested_output_names,
          model_state->Name(), model_state->Version(), flags,
          reinterpret_cast<intptr_t>(factory_ptr),
          reinterpret_cast<intptr_t>(request));
    } else {
      infer_request = std::make_unique<InferRequest>(
          id, correlation_id, pb_input_tensors, requested_output_names,
          model_state->Name(), model_state->Version(), flags, 0,
          reinterpret_cast<intptr_t>(request));
    }

    RETURN_IF_EXCEPTION(infer_request->SaveToSharedMemory(Stub()->ShmPool()));
    requests_shm[r] = infer_request->ShmHandle();
    pb_inference_requests.emplace_back(std::move(infer_request));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::LaunchStubProcess()
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  Stub() = std::make_unique<StubLauncher>(
      "MODEL_INSTANCE_STUB", Name(), DeviceId(),
      TRITONSERVER_InstanceGroupKindString(Kind()));
  RETURN_IF_ERROR(Stub()->Initialize(model_state));
  RETURN_IF_ERROR(Stub()->Launch());

  thread_pool_ = std::make_unique<boost::asio::thread_pool>(
      model_state->StateForBackend()->thread_pool_size);

  if (model_state->IsDecoupled()) {
    decoupled_thread_ = true;
    decoupled_monitor_ =
        std::thread(&ModelInstanceState::DecoupledMessageQueueMonitor, this);
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
    TRITONBACKEND_Request* request,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
  NVTX_RANGE(nvtx_, "GetInputTensor " + Name());
  const char* input_name;
  // Load iidx'th input name
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInputName(request, input_idx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;

  RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
      in, HostPolicyName().c_str(), &input_name, &input_dtype, &input_shape,
      &input_dims_count, &input_byte_size, &input_buffer_count));

  // Only use input collector when a response array is provided.
  std::unique_ptr<BackendInputCollector> collector;
  if (responses) {
    collector = std::make_unique<BackendInputCollector>(
        &request, 1, responses.get(), Model()->TritonMemoryManager(),
        false /* pinned_enable */, CudaStream(), nullptr, nullptr, 0,
        HostPolicyName().c_str());
  }

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  bool cpu_only_tensors = model_state->ForceCPUOnlyInputTensors();

  if (input_dtype == TRITONSERVER_TYPE_BYTES) {
    cpu_only_tensors = true;
  }

#ifdef TRITON_ENABLE_GPU
  CUDAHandler& cuda_handler = CUDAHandler::getInstance();
  // If CUDA driver API is not available, the input tensors will be moved to
  // CPU.
  if (!cuda_handler.IsAvailable()) {
    cpu_only_tensors = true;
  }
#endif

  TRITONSERVER_MemoryType src_memory_type;
  int64_t src_memory_type_id;
  size_t src_byte_size;
  const void* src_ptr;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
      in, 0 /* input buffer index */, &src_ptr, &src_byte_size,
      &src_memory_type, &src_memory_type_id));

// If TRITON_ENABLE_GPU is false, we need to copy the tensors
// to the CPU.
#ifndef TRITON_ENABLE_GPU
  cpu_only_tensors = true;
#endif  // TRITON_ENABLE_GPU

  if (cpu_only_tensors || src_memory_type != TRITONSERVER_MEMORY_GPU) {
    input_tensor = std::make_shared<PbTensor>(
        std::string(input_name),
        std::vector<int64_t>(input_shape, input_shape + input_dims_count),
        input_dtype, TRITONSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */, nullptr /* buffer ptr*/, input_byte_size,
        nullptr /* DLManagedTensor */);
    RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
        Stub()->ShmPool(), false /* copy_gpu */));
    char* input_buffer = reinterpret_cast<char*>(input_tensor->DataPtr());

    if (collector) {
      collector->ProcessTensor(
          input_name, input_buffer, input_byte_size,
          TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
    } else {
      size_t byte_size = input_byte_size;
      RETURN_IF_ERROR(backend::ReadInputTensor(
          request, input_name, input_buffer, &byte_size));
    }
  } else {
#ifdef TRITON_ENABLE_GPU

    // Retreiving GPU input tensors
    const void* buffer = nullptr;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = {{TRITONSERVER_MEMORY_GPU, src_memory_type_id}};

    // collector is used in the non-decoupled mode.
    if (collector) {
      RETURN_IF_ERROR(collector->ProcessTensor(
          input_name, nullptr, 0, alloc_perference,
          reinterpret_cast<const char**>(&buffer), &input_byte_size,
          &src_memory_type, &src_memory_type_id));
      // If the tensor is using the cuda shared memory, we need to extract the
      // handle that was used to create the device pointer. This is because of a
      // limitation in the legacy CUDA IPC API that doesn't allow getting the
      // handle of an exported pointer. If the cuda handle exists, it indicates
      // that the cuda shared memory was used and the input is in a single
      // buffer.
      // [FIXME] For the case where the input is in cuda shared memory and uses
      // multiple input buffers this needs to be changed.
      TRITONSERVER_BufferAttributes* buffer_attributes;

      // This value is not used.
      const void* buffer_p;
      RETURN_IF_ERROR(TRITONBACKEND_InputBufferAttributes(
          in, 0, &buffer_p, &buffer_attributes));

      input_tensor = std::make_shared<PbTensor>(
          std::string(input_name),
          std::vector<int64_t>(input_shape, input_shape + input_dims_count),
          input_dtype, src_memory_type, src_memory_type_id,
          const_cast<void*>(buffer), input_byte_size,
          nullptr /* DLManagedTensor */);

      cudaIpcMemHandle_t* cuda_ipc_handle;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesCudaIpcHandle(
          buffer_attributes, reinterpret_cast<void**>(&cuda_ipc_handle)));
      if (cuda_ipc_handle != nullptr) {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), false /* copy_gpu */));
        RETURN_IF_EXCEPTION(
            input_tensor->Memory()->SetCudaIpcHandle(cuda_ipc_handle));
      } else {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), true /* copy_gpu */));
      }
    } else {
      void* dev_ptr;
      RETURN_IF_CUDA_ERROR(
          cudaMalloc(&dev_ptr, input_byte_size), TRITONSERVER_ERROR_INTERNAL,
          std::string("Failed to allocated CUDA memory"));

      size_t byte_size = input_byte_size;

      bool cuda_used = false;
      RETURN_IF_ERROR(backend::ReadInputTensor(
          request, input_name, reinterpret_cast<char*>(dev_ptr), &byte_size,
          TRITONSERVER_MEMORY_GPU, src_memory_type_id, CudaStream(),
          &cuda_used));

      if (cuda_used) {
#ifdef TRITON_ENABLE_GPU
        cudaStreamSynchronize(stream_);
#endif
      }

      input_tensor = std::make_shared<PbTensor>(
          std::string(input_name),
          std::vector<int64_t>(input_shape, input_shape + input_dims_count),
          input_dtype, src_memory_type, src_memory_type_id,
          const_cast<void*>(dev_ptr), input_byte_size,
          nullptr /* DLManagedTensor */);

      RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
          Stub()->ShmPool(), true /* copy_gpu */));

      std::unique_ptr<MemoryRecord> gpu_memory_record =
          std::make_unique<GPUMemoryRecord>(input_tensor->Memory()->DataPtr());
      uint64_t memory_release_id =
          Stub()->GetMemoryManager()->AddRecord(std::move(gpu_memory_record));
      input_tensor->Memory()->SetMemoryReleaseId(memory_release_id);
    }
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Python backend does not support GPU tensors.");
#endif  // TRITON_ENABLE_GPU
  }

  return nullptr;
}

void
ModelInstanceState::ExecuteBLSRequest(std::shared_ptr<IPCMessage> ipc_message)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  auto request_executor = std::make_unique<RequestExecutor>(
      Stub()->ShmPool(), model_state->TritonServer());
  bool is_response_batch_set = false;
  std::unique_ptr<InferResponse> infer_response;
  ResponseBatch* response_batch;
  TRITONSERVER_InferenceResponse* inference_response = nullptr;
  std::unique_ptr<PbString> pb_error_message;
  std::unique_ptr<IPCMessage> bls_response;
  AllocatedSharedMemory<char> response_batch_shm;
  try {
    bls_response =
        IPCMessage::Create(Stub()->ShmPool(), false /* inline_response */);

    AllocatedSharedMemory<char> request_batch =
        Stub()->ShmPool()->Load<char>(ipc_message->Args());
    RequestBatch* request_batch_shm_ptr =
        reinterpret_cast<RequestBatch*>(request_batch.data_.get());

    bls_response->Command() = PYTHONSTUB_InferExecResponse;
    ipc_message->ResponseHandle() = bls_response->ShmHandle();

    // The response batch of the handle will contain a ResponseBatch
    response_batch_shm = Stub()->ShmPool()->Construct<char>(
        sizeof(ResponseBatch) + sizeof(bi::managed_external_buffer::handle_t));
    response_batch =
        reinterpret_cast<ResponseBatch*>(response_batch_shm.data_.get());
    bi::managed_external_buffer::handle_t* response_handle =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            response_batch_shm.data_.get() + sizeof(ResponseBatch));
    bls_response->Args() = response_batch_shm.handle_;

    response_batch->batch_size = 1;
    response_batch->has_error = false;
    response_batch->is_error_set = false;
    response_batch->cleanup = false;
    is_response_batch_set = true;
    bool has_gpu_tensor = false;

    PythonBackendException pb_exception(std::string{});

    uint32_t gpu_buffers_count = 0;
    if (request_batch_shm_ptr->batch_size == 1) {
      std::shared_ptr<InferRequest> infer_request;
      bi::managed_external_buffer::handle_t* request_handle =
          reinterpret_cast<bi::managed_external_buffer::handle_t*>(
              request_batch.data_.get() + sizeof(RequestBatch));
      infer_request = InferRequest::LoadFromSharedMemory(
          Stub()->ShmPool(), *request_handle, false /* open_cuda_handle */);

      // If the BLS inputs are in GPU an additional round trip between the
      // stub process and the main process is required. The reason is that we
      // need to first allocate the GPU memory from the memory pool and then
      // ask the stub process to fill in those allocated buffers.
      try {
        for (auto& input_tensor : infer_request->Inputs()) {
          if (!input_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
            gpu_buffers_count++;
            BackendMemory* backend_memory;
            std::unique_ptr<BackendMemory> lbackend_memory;
            has_gpu_tensor = true;
            TRITONSERVER_Error* error = BackendMemory::Create(
                Model()->TritonMemoryManager(),
                {BackendMemory::AllocationType::GPU_POOL,
                 BackendMemory::AllocationType::GPU},
                input_tensor->MemoryTypeId(), input_tensor->ByteSize(),
                &backend_memory);
            if (error != nullptr) {
              LOG_MESSAGE(
                  TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(error));
              break;
            }
            lbackend_memory.reset(backend_memory);
            input_tensor->SetMemory(std::move(PbMemory::Create(
                Stub()->ShmPool(), std::move(lbackend_memory))));
#endif  // TRITON_ENABLE_GPU
          }
        }
      }
      catch (const PythonBackendException& exception) {
        pb_exception = exception;
      }
      AllocatedSharedMemory<bi::managed_external_buffer::handle_t> gpu_handles;

      // Wait for the extra round trip to complete. The stub process will fill
      // in the data for the GPU tensors. If there is an error, the extra round
      // trip must be still completed, otherwise the stub process will always be
      // waiting for a message from the parent process.
      if (has_gpu_tensor) {
        try {
          gpu_handles = Stub()
                            ->ShmPool()
                            ->Construct<bi::managed_external_buffer::handle_t>(
                                gpu_buffers_count);
          request_batch_shm_ptr->gpu_buffers_count = gpu_buffers_count;
          request_batch_shm_ptr->gpu_buffers_handle = gpu_handles.handle_;
          size_t i = 0;
          for (auto& input_tensor : infer_request->Inputs()) {
            if (!input_tensor->IsCPU()) {
              gpu_handles.data_.get()[i] = input_tensor->Memory()->ShmHandle();
              ++i;
            }
          }
        }
        catch (const PythonBackendException& exception) {
          pb_exception = exception;
        }

        bi::scoped_lock<bi::interprocess_mutex> lock{
            *(ipc_message->ResponseMutex())};
        ipc_message->ResponseCondition()->notify_all();
        ipc_message->ResponseCondition()->wait(lock);
      }

      if (pb_exception.what() != nullptr) {
        infer_response =
            request_executor->Infer(infer_request, &inference_response);

        if (infer_response) {
          infer_response->SaveToSharedMemory(Stub()->ShmPool());

          for (auto& output_tensor : infer_response->OutputTensors()) {
            // For GPU tensors we need to store the memory release id in memory
            // manager.
            if (!output_tensor->IsCPU()) {
#ifdef TRITON_ENABLE_GPU
              std::unique_ptr<MemoryRecord> gpu_memory_record =
                  std::make_unique<GPUMemoryRecord>(
                      output_tensor->Memory()->DataPtr());
              uint64_t memory_release_id =
                  Stub()->GetMemoryManager()->AddRecord(
                      std::move(gpu_memory_record));
              output_tensor->Memory()->SetMemoryReleaseId(memory_release_id);
#endif
            }
          }
          *response_handle = infer_response->ShmHandle();
        }

      } else {
        throw pb_exception;
      }
    }
  }
  catch (const PythonBackendException& pb_exception) {
    if (is_response_batch_set) {
      response_batch->has_error = true;
      LOG_IF_EXCEPTION(
          pb_error_message =
              PbString::Create(Stub()->ShmPool(), pb_exception.what()));

      if (pb_error_message != nullptr) {
        response_batch->is_error_set = true;
        response_batch->error = pb_error_message->ShmHandle();
      }
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, pb_exception.what());
    }
  }

  // At this point, the stub has notified the parent process that it has
  // finished loading the inference response from shared memory.
  {
    bi::scoped_lock<bi::interprocess_mutex> lock{
        *(ipc_message->ResponseMutex())};
    ipc_message->ResponseCondition()->notify_all();
    ipc_message->ResponseCondition()->wait(lock);
  }

  if (inference_response != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(inference_response),
        " failed to release BLS inference response.");
  }
}

void
ModelInstanceState::DecoupledMessageQueueMonitor()
{
  while (decoupled_thread_) {
    bi::managed_external_buffer::handle_t handle =
        Stub()->ParentMessageQueue()->Pop();
    if (handle == DUMMY_MESSAGE) {
      break;
    }
    std::unique_ptr<IPCMessage> message =
        IPCMessage::LoadFromSharedMemory(Stub()->ShmPool(), handle);

    // Need to notify the model instance thread that the execute response has
    // been received.
    if (message->Command() == PYTHONSTUB_ExecuteResponse) {
      std::lock_guard<std::mutex> guard{mu_};
      received_message_ = std::move(message);
      cv_.notify_one();
    } else if (message->Command() == PYTHONSTUB_ResponseSend) {
      std::shared_ptr<IPCMessage> response_send_message = std::move(message);
      std::packaged_task<void()> task([this, response_send_message] {
        ResponseSendDecoupled(response_send_message);
      });
      std::future<void> future =
          boost::asio::post(*thread_pool_, std::move(task));
      futures_.emplace_back(std::move(future));
    } else if (message->Command() == PYTHONSTUB_InferExecRequest) {
      std::shared_ptr<IPCMessage> bls_execute = std::move(message);
      std::packaged_task<void()> task(
          [this, bls_execute] { ExecuteBLSRequest(bls_execute); });
      std::future<void> future =
          boost::asio::post(*thread_pool_, std::move(task));
      futures_.emplace_back(std::move(future));
    }
  }
}

void
ModelInstanceState::ResponseSendDecoupled(
    std::shared_ptr<IPCMessage> response_send_message)
{
  AllocatedSharedMemory<ResponseSendMessage> send_message =
      Stub()->ShmPool()->Load<ResponseSendMessage>(
          response_send_message->Args());

  ResponseSendMessage* send_message_payload =
      reinterpret_cast<ResponseSendMessage*>(send_message.data_.get());
  std::unique_ptr<PbString> error_message;
  ScopedDefer _([send_message_payload] {
    {
      bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
      send_message_payload->is_stub_turn = true;
      send_message_payload->cv.notify_all();

      while (send_message_payload->is_stub_turn) {
        send_message_payload->cv.wait(guard);
      }
    }
  });

  TRITONBACKEND_ResponseFactory* response_factory =
      reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
          send_message_payload->response_factory_address);
  if (send_message_payload->flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    std::lock_guard<std::mutex> guard{closed_requests_mutex_};
    closed_requests_.push_back(send_message_payload->request_address);
  }

  if (send_message_payload->response != 0) {
    std::unique_ptr<InferResponse> infer_response =
        InferResponse::LoadFromSharedMemory(
            Stub()->ShmPool(), send_message_payload->response,
            false /* open cuda ipc handle */);

    bool requires_deferred_callback = false;
    std::vector<std::pair<std::unique_ptr<PbMemory>, void*>> gpu_output_buffers;
    std::shared_ptr<TRITONSERVER_Error*> error = infer_response->Send(
        response_factory, CudaStream(), requires_deferred_callback,
        send_message_payload->flags, Stub()->ShmPool(), gpu_output_buffers);
    SetErrorForResponseSendMessage(send_message_payload, error, error_message);

    if (requires_deferred_callback) {
      AllocatedSharedMemory<char> gpu_buffers_handle =
          Stub()->ShmPool()->Construct<char>(
              sizeof(uint64_t) +
              gpu_output_buffers.size() *
                  sizeof(bi::managed_external_buffer::handle_t));
      uint64_t* gpu_buffer_count =
          reinterpret_cast<uint64_t*>(gpu_buffers_handle.data_.get());
      *gpu_buffer_count = gpu_output_buffers.size();
      bi::managed_external_buffer::handle_t* gpu_buffers_handle_shm =
          reinterpret_cast<bi::managed_external_buffer::handle_t*>(
              gpu_buffers_handle.data_.get() + sizeof(uint64_t));
      send_message_payload->gpu_buffers_handle = gpu_buffers_handle.handle_;

      size_t index = 0;
      for (auto& output_buffer_pair : gpu_output_buffers) {
        std::unique_ptr<PbMemory>& pb_memory = output_buffer_pair.first;
        gpu_buffers_handle_shm[index] = pb_memory->ShmHandle();
        ++index;
      }

      // Additional round trip so that the stub can fill the GPU output buffers.
      {
        bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
        send_message_payload->is_stub_turn = true;
        send_message_payload->cv.notify_all();

        while (send_message_payload->is_stub_turn) {
          send_message_payload->cv.wait(guard);
        }
      }

      index = 0;
      bool cuda_copy = false;
      for (auto& output_buffer_pair : gpu_output_buffers) {
        auto& pb_memory = output_buffer_pair.first;

        if (pb_memory->MemoryType() == TRITONSERVER_MEMORY_CPU) {
          bool cuda_used;
          void* pointer = output_buffer_pair.second;

          CopyBuffer(
              "Failed to copy the output tensor to buffer.",
              TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
              pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
              CudaStream(), &cuda_used);
          cuda_copy |= cuda_used;
        }
        gpu_buffers_handle_shm[index] = pb_memory->ShmHandle();
        ++index;
#ifdef TRITON_ENABLE_GPU
        if (cuda_copy) {
          cudaStreamSynchronize(stream_);
        }
#endif  // TRITON_ENABLE_GPU
      }
    }
  } else {
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseFactorySendFlags(
        response_factory, send_message_payload->flags);
    SetErrorForResponseSendMessage(
        send_message_payload, WrapTritonErrorInSharedPtr(error), error_message);

    if (send_message_payload->flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      std::unique_ptr<
          TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
      response_factory(reinterpret_cast<TRITONBACKEND_ResponseFactory*>(
          send_message_payload->response_factory_address));
    }
  }
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequestsDecoupled(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<std::unique_ptr<InferRequest>>& pb_inference_requests,
    PbMetricReporter& reporter)
{
  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());
  closed_requests_ = {};
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  size_t total_batch_size = 0;
  RETURN_IF_ERROR(
      CheckIncomingRequests(requests, request_count, total_batch_size));

  // No request to process
  if (total_batch_size == 0) {
    return nullptr;  // success
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());

  AllocatedSharedMemory<char> request_batch;
  std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses;

  RETURN_IF_ERROR(SaveRequestsToSharedMemory(
      requests, request_count, pb_inference_requests, request_batch,
      responses));

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  reporter.SetComputeStartNs(compute_start_ns);

  std::unique_ptr<IPCMessage> ipc_message;
  RETURN_IF_EXCEPTION(
      ipc_message =
          IPCMessage::Create(Stub()->ShmPool(), false /*inline_response*/));
  ipc_message->Command() = PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest;
  ipc_message->Args() = request_batch.handle_;
  received_message_ = nullptr;
  ScopedDefer _([this] { Stub()->StubMessageQueue()->Push(DUMMY_MESSAGE); });

  {
    std::unique_lock<std::mutex> guard{mu_};
    Stub()->StubMessageQueue()->Push(ipc_message->ShmHandle());
    cv_.wait(guard, [this] { return received_message_ != nullptr; });
  }

  AllocatedSharedMemory<ResponseBatch> response_batch =
      Stub()->ShmPool()->Load<ResponseBatch>(received_message_->Args());

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  reporter.SetComputeEndNs(compute_end_ns);
  reporter.SetBatchStatistics(request_count);

  if (response_batch.data_->has_error) {
    if (response_batch.data_->is_error_set) {
      auto error = PbString::LoadFromSharedMemory(
          Stub()->ShmPool(), response_batch.data_->error);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, error->String().c_str());
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Failed to process the requests.");
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    bool& restart)
{
  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  std::string name = model_state->Name();

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // We take the responsibility of the responses.
  std::shared_ptr<std::vector<TRITONBACKEND_Response*>> responses(
      new std::vector<TRITONBACKEND_Response*>());
  responses->reserve(request_count);
  PbMetricReporter reporter(
      TritonModelInstance(), requests, request_count, responses);
  reporter.SetExecStartNs(exec_start_ns);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses->emplace_back(response);
    } else {
      responses->emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  size_t total_batch_size = 0;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      CheckIncomingRequests(requests, request_count, total_batch_size));

  // No request to process
  if (total_batch_size == 0) {
    return;
  }

  // Wait for all the pending BLS requests to be completed.
  ScopedDefer bls_defer([this] { WaitForBLSRequestsToFinish(); });
  std::vector<std::unique_ptr<InferRequest>> pb_inference_requests;
  AllocatedSharedMemory<char> request_batch;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      SaveRequestsToSharedMemory(
          requests, request_count, pb_inference_requests, request_batch,
          responses));

  std::shared_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(Stub()->ShmPool(), false /*inline_response*/);
  ipc_message->Command() = PYTHONSTUB_CommandType::PYTHONSTUB_ExecuteRequest;
  ipc_message->Args() = request_batch.handle_;

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  reporter.SetComputeStartNs(compute_start_ns);

  // This means that the stub process has exited and Python
  // backend failed to restart the stub process.
  if (Stub()->StubPid() == 0) {
    const char* error_message = "The stub process has exited unexpectedly.";
    RespondErrorToAllRequests(
        error_message, responses, requests, request_count);
    return;
  }

  bi::managed_external_buffer::handle_t response_message;
  {
    NVTX_RANGE(nvtx_, "StubProcessing " + Name());
    SendMessageAndReceiveResponse(
        ipc_message->ShmHandle(), response_message, restart, responses,
        requests, request_count);
  }

  ScopedDefer execute_finalize([this, &restart] {
    // Push a dummy message to the message queue so that
    // the stub process is notified that it can release
    // the object stored in shared memory.
    NVTX_RANGE(nvtx_, "RequestExecuteFinalize " + Name());
    if (!restart)
      Stub()->StubMessageQueue()->Push(DUMMY_MESSAGE);
  });
  if (restart) {
    return;
  }

  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      ipc_message = IPCMessage::LoadFromSharedMemory(
          Stub()->ShmPool(), response_message));

  // If the stub command is no longer PYTHONSTUB_InferExecRequest, it indicates
  // that inference request exeuction has finished and there are no more BLS
  // requests to execute. Otherwise, the Python backend will continuosly execute
  // BLS requests pushed to the message queue.
  while (ipc_message->Command() ==
         PYTHONSTUB_CommandType::PYTHONSTUB_InferExecRequest) {
    std::packaged_task<void()> task(
        [this, ipc_message] { ExecuteBLSRequest(ipc_message); });
    std::future<void> future =
        boost::asio::post(*thread_pool_, std::move(task));
    futures_.emplace_back(std::move(future));

    auto error = ReceiveMessageFromStub(response_message);
    if (error != nullptr) {
      restart = true;
      RespondErrorToAllRequests(
          TRITONSERVER_ErrorMessage(error), responses, requests, request_count);
      return;
    }

    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        responses, request_count,
        ipc_message = IPCMessage::LoadFromSharedMemory(
            Stub()->ShmPool(), response_message));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  reporter.SetComputeEndNs(compute_end_ns);

  // Parsing the request response
  AllocatedSharedMemory<char> response_batch;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      responses, request_count,
      response_batch = Stub()->ShmPool()->Load<char>(ipc_message->Args()));

  ResponseBatch* response_batch_shm_ptr =
      reinterpret_cast<ResponseBatch*>(response_batch.data_.get());

  // If inference fails, release all the requests and send an error response.
  // If inference fails at this stage, it usually indicates a bug in the model
  // code
  if (response_batch_shm_ptr->has_error) {
    if (response_batch_shm_ptr->is_error_set) {
      std::unique_ptr<PbString> error_message_shm;
      RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
          responses, request_count,
          error_message_shm = PbString::LoadFromSharedMemory(
              Stub()->ShmPool(), response_batch_shm_ptr->error));
      RespondErrorToAllRequests(
          error_message_shm->String().c_str(), responses, requests,
          request_count);
    } else {
      const char* error_message =
          "Failed to fetch the error in response batch.";
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    }
    return;
  }

  bi::managed_external_buffer::handle_t* response_shm_handle =
      reinterpret_cast<bi::managed_external_buffer::handle_t*>(
          response_batch.data_.get() + sizeof(ResponseBatch));

  // If the output provided by the model is in GPU, we will pass the list of
  // buffers provided by Triton to the stub process.
  bool has_gpu_output = false;
  std::vector<bool> requires_deferred_callback;

  std::vector<std::unique_ptr<InferResponse>> shm_responses;
  std::unordered_map<
      uint32_t, std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>>
      gpu_output_buffers;

  for (uint32_t r = 0; r < request_count; ++r) {
    NVTX_RANGE(nvtx_, "LoadingResponse " + Name());
    TRITONBACKEND_Response* response = (*responses)[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;
    requires_deferred_callback.push_back(false);

    shm_responses.emplace_back(nullptr);
    std::unique_ptr<InferResponse>& infer_response = shm_responses.back();
    try {
      infer_response = InferResponse::LoadFromSharedMemory(
          Stub()->ShmPool(), response_shm_handle[r],
          false /* open_cuda_handle */);
      if (infer_response->HasError()) {
        TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            infer_response->Error()->Message().c_str());

        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed sending response");
        TRITONSERVER_ErrorDelete(err);
        (*responses)[r] = nullptr;

        // If has_error is true, we do not look at the response tensors.
        continue;
      }
    }
    catch (const PythonBackendException& pb_exception) {
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              (*responses)[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");
      TRITONSERVER_ErrorDelete(err);
      (*responses)[r] = nullptr;
      continue;
    }

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    std::set<std::string> requested_output_names;
    for (size_t j = 0; j < requested_output_count; ++j) {
      const char* output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, j, &output_name));
      requested_output_names.insert(output_name);
    }

    bool require_deferred_callback = false;

    gpu_output_buffers[r] =
        std::vector<std::pair<std::unique_ptr<PbMemory>, void*>>{};
    std::shared_ptr<TRITONSERVER_Error*> error = infer_response->Send(
        nullptr, CudaStream(), require_deferred_callback,
        TRITONSERVER_RESPONSE_COMPLETE_FINAL, Stub()->ShmPool(),
        gpu_output_buffers[r], requested_output_names, response);
    GUARDED_RESPOND_IF_ERROR(responses, r, *error);

    requires_deferred_callback[r] = require_deferred_callback;

    // Error object will be deleted by the GUARDED_RESPOND macro
    *error = nullptr;
    error.reset();
    if (requires_deferred_callback[r]) {
      has_gpu_output = true;
    }
  }

  // Finalize the execute.
  execute_finalize.Complete();

  // If the output tensor is in GPU, there will be a second round trip
  // required for filling the GPU buffers provided by the main process.
  if (has_gpu_output) {
    size_t total_gpu_buffers_count = 0;
    for (auto& gpu_output_buffer : gpu_output_buffers) {
      total_gpu_buffers_count += gpu_output_buffer.second.size();
    }
    AllocatedSharedMemory<char> gpu_buffers_handle =
        Stub()->ShmPool()->Construct<char>(
            sizeof(uint64_t) +
            total_gpu_buffers_count *
                sizeof(bi::managed_external_buffer::handle_t));
    uint64_t* gpu_buffer_count =
        reinterpret_cast<uint64_t*>(gpu_buffers_handle.data_.get());
    *gpu_buffer_count = total_gpu_buffers_count;
    bi::managed_external_buffer::handle_t* gpu_buffers_handle_shm =
        reinterpret_cast<bi::managed_external_buffer::handle_t*>(
            gpu_buffers_handle.data_.get() + sizeof(uint64_t));

    size_t index = 0;
    for (auto& gpu_output_buffer : gpu_output_buffers) {
      for (auto& buffer_memory_pair : gpu_output_buffer.second) {
        gpu_buffers_handle_shm[index] = buffer_memory_pair.first->ShmHandle();
        ++index;
      }
    }

    ipc_message->Command() = PYTHONSTUB_CommandType::PYTHONSTUB_LoadGPUBuffers;
    ipc_message->Args() = gpu_buffers_handle.handle_;
    SendMessageAndReceiveResponse(
        ipc_message->ShmHandle(), response_message, restart, responses,
        requests, 0);

    bool cuda_copy = false;

    index = 0;
    for (auto& gpu_output_buffer : gpu_output_buffers) {
      for (auto& buffer_memory_pair : gpu_output_buffer.second) {
        auto& pb_memory = buffer_memory_pair.first;
        if (pb_memory->MemoryType() == TRITONSERVER_MEMORY_CPU) {
          bool cuda_used;
          uint32_t response_index = gpu_output_buffer.first;
          void* pointer = buffer_memory_pair.second;

          GUARDED_RESPOND_IF_ERROR(
              responses, response_index,
              CopyBuffer(
                  "Failed to copy the output tensor to buffer.",
                  TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
                  pb_memory->ByteSize(), pb_memory->DataPtr(), pointer,
                  CudaStream(), &cuda_used));
          cuda_copy |= cuda_used;
        }
        gpu_buffers_handle_shm[index] = pb_memory->ShmHandle();
        ++index;
      }
#ifdef TRITON_ENABLE_GPU
      if (cuda_copy) {
        cudaStreamSynchronize(stream_);
      }
#endif  // TRITON_ENABLE_GPU
    }
  }

  bls_defer.Complete();
  for (uint32_t r = 0; r < request_count; ++r) {
    if (requires_deferred_callback[r]) {
      shm_responses[r]->DeferredSendCallback();
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  reporter.SetExecEndNs(exec_end_ns);
  reporter.SetBatchStatistics(total_batch_size);

  return;
}

ModelInstanceState::~ModelInstanceState()
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  Stub()->UpdateHealth();
  if (Stub()->IsHealthy()) {
    if (model_state->IsDecoupled()) {
      futures_.clear();
      Stub()->ParentMessageQueue()->Push(DUMMY_MESSAGE);
      decoupled_monitor_.join();
    }
    // Wait for all the futures to be finished.
    thread_pool_->wait();
  }

  Stub()->TerminateStub();
  received_message_.reset();
  Stub().reset();
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

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->LaunchAutoCompleteStubProcess());
    (*state)->ModelConfig() = std::move((*state)->Stub()->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());

    (*state)->Stub()->UpdateHealth();
    (*state)->Stub()->TerminateStub();
    (*state)->Stub().reset();
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true /* allow_optional */)
{
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));

  const char* path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
  python_execution_env_ = "";
  force_cpu_only_input_tensors_ = true;
  decoupled_ = false;

  void* bstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  backend_state_ = reinterpret_cast<BackendState*>(bstate);
  triton::common::TritonJson::Value params;
  common::TritonJson::Value model_config;
  if (model_config_.Find("parameters", &params)) {
    // Skip the EXECUTION_ENV_PATH variable if it doesn't exist.
    TRITONSERVER_Error* error =
        GetParameterValue(params, "EXECUTION_ENV_PATH", &python_execution_env_);
    if (error == nullptr) {
      std::string relative_path_keyword = "$$TRITON_MODEL_DIRECTORY";
      size_t relative_path_loc =
          python_execution_env_.find(relative_path_keyword);
      if (relative_path_loc != std::string::npos) {
        python_execution_env_.replace(
            relative_path_loc, relative_path_loc + relative_path_keyword.size(),
            path);
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Using Python execution env ") + python_execution_env_)
              .c_str());
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }

    triton::common::TritonJson::Value model_transaction_policy;
    if (model_config_.Find(
            "model_transaction_policy", &model_transaction_policy)) {
      triton::common::TritonJson::Value decoupled;
      if (model_transaction_policy.Find("decoupled", &decoupled)) {
        auto error = decoupled.AsBool(&decoupled_);
        if (error != nullptr) {
          throw BackendModelException(error);
        }
      }
    }

    // Skip the FORCE_CPU_ONLY_INPUT_TENSORS variable if it doesn't exits.
    std::string force_cpu_only_input_tensor;
    error = nullptr;
    error = GetParameterValue(
        params, "FORCE_CPU_ONLY_INPUT_TENSORS", &force_cpu_only_input_tensor);
    if (error == nullptr) {
      if (force_cpu_only_input_tensor == "yes") {
        force_cpu_only_input_tensors_ = true;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Forcing CPU only input tensors.")).c_str());
      } else if (force_cpu_only_input_tensor == "no") {
        force_cpu_only_input_tensors_ = false;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Input tensors can be both in CPU and GPU. "
                         "FORCE_CPU_ONLY_INPUT_TENSORS is off."))
                .c_str());
      } else {
        throw BackendModelException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Incorrect value for FORCE_CPU_ONLY_INPUT_TENSORS: ") +
             force_cpu_only_input_tensor + "'")
                .c_str()));
      }
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }
  }

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + Name() + "'")
            .c_str()));
  }
}

TRITONSERVER_Error*
ModelState::LaunchAutoCompleteStubProcess()
{
  Stub() = std::make_unique<StubLauncher>("AUTOCOMPLETE_STUB");
  RETURN_IF_ERROR(Stub()->Initialize(this));
  try {
    RETURN_IF_ERROR(Stub()->Launch());
  }
  catch (const BackendModelException& ex) {
    Stub()->UpdateHealth();
    Stub()->TerminateStub();
    Stub().reset();
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;
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
  backend_state->shm_default_byte_size = 64 * 1024 * 1024;  // 64 MBs
  backend_state->shm_growth_byte_size = 64 * 1024 * 1024;   // 64 MBs
  backend_state->stub_timeout_seconds = 30;
  backend_state->shm_message_queue_size = 1000;
  backend_state->number_of_instance_inits = 0;
  backend_state->thread_pool_size = 32;
  backend_state->shared_memory_region_prefix =
      "triton_python_backend_shm_region_";

  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value shm_growth_size;
    std::string shm_growth_byte_size;
    if (cmdline.Find("shm-growth-byte-size", &shm_growth_size)) {
      RETURN_IF_ERROR(shm_growth_size.AsString(&shm_growth_byte_size));
      try {
        backend_state->shm_growth_byte_size = std::stol(shm_growth_byte_size);
        if (backend_state->shm_growth_byte_size <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-growth-byte-size") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_default_size;
    std::string shm_default_byte_size;
    if (cmdline.Find("shm-default-byte-size", &shm_default_size)) {
      RETURN_IF_ERROR(shm_default_size.AsString(&shm_default_byte_size));
      try {
        backend_state->shm_default_byte_size = std::stol(shm_default_byte_size);
        // Shared memory default byte size can't be less than 4 MBs.
        if (backend_state->shm_default_byte_size < 4 * 1024 * 1024) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-default-byte-size") +
               " can't be smaller than 4 MiBs")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value thread_pool_size;
    std::string thread_pool_count;
    if (cmdline.Find("thread-pool-size", &thread_pool_size)) {
      RETURN_IF_ERROR(thread_pool_size.AsString(&thread_pool_count));
      try {
        backend_state->thread_pool_size = std::stol(thread_pool_count);
        // Shared memory default byte size can't be less than 4 MBs.
        if (backend_state->thread_pool_size < 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("thread-pool-size") + " can't be less than 1.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_region_prefix;
    std::string shm_region_prefix_str;
    if (cmdline.Find("shm-region-prefix-name", &shm_region_prefix)) {
      RETURN_IF_ERROR(shm_region_prefix.AsString(&shm_region_prefix_str));
      // Shared memory default byte size can't be less than 4 MBs.
      if (shm_region_prefix_str.size() == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("shm-region-prefix-name") +
             " must at least contain one character.")
                .c_str());
      }
      backend_state->shared_memory_region_prefix = shm_region_prefix_str;
    }

    triton::common::TritonJson::Value shm_message_queue_size;
    std::string shm_message_queue_size_str;
    if (cmdline.Find("shm_message_queue_size", &shm_message_queue_size)) {
      RETURN_IF_ERROR(
          shm_message_queue_size.AsString(&shm_message_queue_size_str));
      try {
        backend_state->shm_message_queue_size =
            std::stol(shm_message_queue_size_str);
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value stub_timeout_seconds;
    std::string stub_timeout_string_seconds;
    if (cmdline.Find("stub-timeout-seconds", &stub_timeout_seconds)) {
      RETURN_IF_ERROR(
          stub_timeout_seconds.AsString(&stub_timeout_string_seconds));
      try {
        backend_state->stub_timeout_seconds =
            std::stol(stub_timeout_string_seconds);
        if (backend_state->stub_timeout_seconds <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("stub-timeout-seconds") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Shared memory configuration is shm-default-byte-size=") +
       std::to_string(backend_state->shm_default_byte_size) +
       ",shm-growth-byte-size=" +
       std::to_string(backend_state->shm_growth_byte_size) +
       ",stub-timeout-seconds=" +
       std::to_string(backend_state->stub_timeout_seconds))
          .c_str());

  // Use BackendArtifacts to determine the location of Python files
  const char* location;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  backend_state->python_lib = location;
  backend_state->env_manager = std::make_unique<EnvironmentManager>();

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

  RETURN_IF_ERROR(instance_state->LaunchStubProcess());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

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

  TRITONSERVER_Error* error = nullptr;

  // If restart is equal to true, it indicates that the stub process is
  // unhealthy and needs a restart.
  bool restart = false;
  ModelState* model_state =
      reinterpret_cast<ModelState*>(instance_state->Model());
  if (!model_state->IsDecoupled()) {
    instance_state->ProcessRequests(requests, request_count, restart);

    if (restart) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          "Stub process is unhealthy and it will be restarted.");
      instance_state->Stub()->KillStubProcess();
      LOG_IF_ERROR(
          instance_state->Stub()->Launch(),
          "Failed to restart the stub process.");
    }
  } else {
    std::vector<std::unique_ptr<InferRequest>> infer_requests;

    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    PbMetricReporter reporter(
        instance_state->TritonModelInstance(), requests, request_count,
        nullptr);
    reporter.SetExecStartNs(exec_start_ns);

    error = instance_state->ProcessRequestsDecoupled(
        requests, request_count, infer_requests, reporter);

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    reporter.SetExecEndNs(exec_end_ns);

    if (error != nullptr) {
      reporter.SetSuccessStatus(false);
      for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        if (!instance_state->ExistsInClosedRequests(
                reinterpret_cast<intptr_t>(request))) {
          TRITONBACKEND_Response* response = nullptr;
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseNew(&response, request),
              "Failed to create a new response.");

          if (response != nullptr) {
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(
                    response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
                "Failed to send the error response.");
          }
        }
      }

      // We should only delete the response factory for the requests that have
      // not been closed.
      for (auto& infer_request : infer_requests) {
        if (!instance_state->ExistsInClosedRequests(
                infer_request->RequestAddress())) {
          LOG_IF_ERROR(
              infer_request->DeleteResponseFactory(),
              "Failed to delete the response factory.");
        }
      }
    }
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       instance_state->Name() + " released " + std::to_string(request_count) +
       " requests")
          .c_str());

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
