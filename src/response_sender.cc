// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "response_sender.h"
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include "pb_stub.h"

namespace triton { namespace backend { namespace python {

ResponseSender::ResponseSender(
    long long request_address, std::unique_ptr<SharedMemoryManager>& shm_pool)
    : request_address_(request_address), shm_pool_(shm_pool)
{
}


void
ResponseSender::Send(std::shared_ptr<InferResponse>& infer_response)
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();

  AllocatedSharedMemory<ResponseSendMessage> response_send_message =
      shm_pool_->Construct<ResponseSendMessage>(
          1 /* count */, true /* aligned */);

  infer_response->SaveToSharedMemory(shm_pool_, false /* copy_gpu */);

  ResponseSendMessage* send_message_payload =
      reinterpret_cast<ResponseSendMessage*>(response_send_message.data_.get());
  new (&(send_message_payload->mu)) bi::interprocess_mutex;
  new (&(send_message_payload->cv)) bi::interprocess_condition;

  send_message_payload->is_stub_turn = false;
  send_message_payload->request_address = request_address_;
  send_message_payload->response = infer_response->ShmHandle();

  std::unique_ptr<IPCMessage> ipc_message =
      IPCMessage::Create(shm_pool_, false /* inline_response */);

  ipc_message->Command() = PYTHONSTUB_ResponseSend;
  ipc_message->Args() = response_send_message.handle_;
  stub->SendIPCMessage(ipc_message);

  {
    bi::scoped_lock<bi::interprocess_mutex> guard{send_message_payload->mu};
    while (!send_message_payload->is_stub_turn) {
      send_message_payload->cv.wait(guard);
    }
  }
}

void
ResponseSender::Close()
{
}

}}}  // namespace triton::backend::python
