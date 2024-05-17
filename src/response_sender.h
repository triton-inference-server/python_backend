// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "infer_response.h"
#include "pb_cancel.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

class ResponseSender {
 public:
  ResponseSender(
      intptr_t request_address, intptr_t response_factory_address,
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      const std::shared_ptr<PbCancel>& pb_cancel);
  ~ResponseSender();
  void Send(std::shared_ptr<InferResponse> response, const uint32_t flags);
  bool IsCancelled();

 private:
  intptr_t request_address_;
  intptr_t response_factory_address_;
  std::unique_ptr<SharedMemoryManager>& shm_pool_;
  // The flag to indicate if the response sender is closed. It is set to true
  // once the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag is set, meaning that the
  // response_sender should not be used anymore. This flag is separate from the
  // `is_response_factory_cleaned_` flag because errors might occur after
  // complete_final flag is set but before the response_factory gets cleaned up.
  bool closed_;
  std::shared_ptr<PbCancel> pb_cancel_;
  // The flag to indicate if the response_factory is already cleaned up in the
  // python backend side. If not, the response_factory will be cleaned up in the
  // destructor.
  bool is_response_factory_cleaned_;
};
}}}  // namespace triton::backend::python
