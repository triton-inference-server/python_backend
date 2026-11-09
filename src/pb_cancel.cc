// Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_cancel.h"

#include <optional>

#include "pb_stub.h"
#include "pb_stub_log.h"

namespace triton { namespace backend { namespace python {

void
PbCancel::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  cancel_shm_ = shm_pool->Construct<IsCancelledMessage>();
  new (&(cancel_shm_.data_->mu)) bi::interprocess_mutex;
  new (&(cancel_shm_.data_->cv)) bi::interprocess_condition;
  cancel_shm_.data_->waiting_on_stub = false;
  cancel_shm_.data_->response_factory_address = response_factory_address_;
  cancel_shm_.data_->request_address = request_address_;
  cancel_shm_.data_->is_cancelled = is_cancelled_;
}

bi::managed_external_buffer::handle_t
PbCancel::ShmHandle()
{
  return cancel_shm_.handle_;
}

IsCancelledMessage*
PbCancel::ShmPayload()
{
  return cancel_shm_.data_.get();
}

bool
PbCancel::IsCancelled()
{
  // Release the GIL. Python objects are not accessed during the check.
  py::gil_scoped_release gil_release;

  std::unique_lock<std::mutex> lk(mu_);
  // The cancelled flag can only move from false to true, not the other way, so
  // it is checked on each query until cancelled and then implicitly cached.
  // Once released, the parent can no longer be asked, so the last observed
  // state is returned.
  if (is_cancelled_ || released_) {
    return is_cancelled_;
  }
  if (!updating_) {
    auto stub = Stub::GetOrCreateInstance();
    if (!stub->StubToParentServiceActive()) {
      LOG_ERROR << "Cannot communicate with parent service";
      return false;
    }
    stub->EnqueueIsCancelled(this);
    updating_ = true;
  }
  cv_.wait(lk, [this] { return !updating_; });
  return is_cancelled_;
}

void
PbCancel::ReportIsCancelled(bool is_cancelled)
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    is_cancelled_ = is_cancelled;
    updating_ = false;
  }
  cv_.notify_all();
}

void
PbCancel::Release()
{
  {
    // Without a query in flight there is nothing to wait for, so the GIL is
    // kept. A contended lock takes the slow path, so a thread holding the GIL
    // never blocks here.
    std::unique_lock<std::mutex> lk(mu_, std::try_to_lock);
    if (lk.owns_lock() && !updating_) {
      released_ = true;
      return;
    }
  }

  // Wait without the GIL. The stub thread that sends the query may need the
  // GIL to finish an earlier message first.
  std::optional<py::gil_scoped_release> gil_release;
  if (PyGILState_Check()) {
    gil_release.emplace();
  }

  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [this] { return !updating_; });
  released_ = true;
}

}}}  // namespace triton::backend::python
