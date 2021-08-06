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

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <memory>
#include "pb_utils.h"

#pragma once

namespace bi = boost::interprocess;

namespace triton { namespace backend { namespace python {

class Stub {
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  std::unique_ptr<bi::scoped_lock<bi::interprocess_mutex>> stub_lock_;
  std::string model_path_;
  std::string model_version_;
  std::string model_instance_name_;
  std::string triton_install_path_;
  IPCMessage* ipc_message_;
  IPCControl* ipc_control_;
  std::unique_ptr<SharedMemory> shm_pool_;
  py::object model_instance_;
  py::object deserialize_bytes_;
  py::object serialize_bytes_;
  std::vector<std::shared_ptr<PbTensor>> tensors_to_remove_;
  bool require_cleanup_;
  bool initialized_;
  static std::unique_ptr<Stub> stub_instance_;

#ifdef TRITON_ENABLE_GPU_TENSORS
  std::unordered_map<void*, cudaIpcMemHandle_t*> gpu_tensors_map_;
#endif // TRITON_ENABLE_GPU_TENSORS

 public:
  Stub(){};
  static std::unique_ptr<Stub>& GetOrCreateInstance();

  void Instantiate(
      int64_t shm_growth_size, int64_t shm_default_size,
      const std::string& shm_region_name, const std::string& model_path,
      const std::string& model_version, const std::string& triton_install_path,
      off_t ipc_control_offset, const std::string& model_instance_name);
  void NotifyParent();
  bool& Health();
  std::unique_ptr<SharedMemory>& GetSharedMemory();
  void SetErrorForResponse(Response* response, const char* err_message);
  void SetErrorForResponseBatch(
      ResponseBatch* response_batch, const char* err_message);
  void ProcessResponse(
      Response* response_shm, ResponseBatch* response_batch,
      InferResponse* response, py::object& serialize_bytes);
  std::unique_ptr<InferRequest> ProcessRequest(
      off_t request_offset, ResponseBatch* response_batch,
      py::object& deserialize_bytes);
  void SetResponseFromException(
      ResponseBatch* response_batch,
      const PythonBackendException& pb_exception);
  bool RunCommand();
  void Execute(ExecuteArgs* execute_args, ResponseBatch* response_batch);
  void Initialize(InitializeArgs* initialize_args);
  void UpdateHealth();
  void Cleanup();
  void Finalize();
  IPCMessage* GetIPCMessage();

  // Wait for notification from the server. Returns true if the parent process
  // has received a SIGTERM, and false otherwise.
  bool WaitForNotification();

  ~Stub();
};
}}}  // namespace triton::backend::python
