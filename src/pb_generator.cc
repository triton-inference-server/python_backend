// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_generator.h"
#include <chrono>
#include "pb_stub.h"

#include <pybind11/embed.h>
namespace py = pybind11;

namespace triton { namespace backend { namespace python {

ResponseGenerator::ResponseGenerator(
    const std::shared_ptr<InferResponse>& response)
    : id_(response->Id()), is_finished_(false), is_cleared_(false)
{
  response_buffer_.push(response);
}

ResponseGenerator::~ResponseGenerator()
{
  {
    std::lock_guard<std::mutex> lock{mu_};
    response_buffer_.push(DUMMY_MESSAGE);
  }
  cv_.notify_all();
}

std::shared_ptr<InferResponse>
ResponseGenerator::Next()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  if (is_finished_) {
    if (!is_cleared_) {
      stub->EnqueueCleanupId(id_);
      is_cleared_ = true;
    }
    throw py::stop_iteration("Iteration is done for the responses.");
  }

  std::shared_ptr<InferResponse> response;
  {
    py::gil_scoped_release release;

    std::unique_lock<std::mutex> lock{mu_};
    {
      while (response_buffer_.empty()) {
        cv_.wait(lock);
      }
      response = response_buffer_.front();
      response_buffer_.pop();
      is_finished_ = response->IsLastResponse();
    }
  }

  // Handle the case where the last response is empty.
  if (is_finished_) {
    stub->EnqueueCleanupId(id_);
    is_cleared_ = true;
    if (response->OutputTensors().empty()) {
      throw py::stop_iteration("Iteration is done for the responses.");
    }
  }

  return response;
}

py::iterator
ResponseGenerator::Iter()
{
  bool done = false;
  while (!done) {
    try {
      responses_.push_back(Next());
    }
    catch (const py::stop_iteration& exception) {
      done = true;
    }
  }

  return py::make_iterator(responses_.begin(), responses_.end());
}

void
ResponseGenerator::EnqueueResponse(
    std::unique_ptr<InferResponse> infer_response)
{
  {
    std::lock_guard<std::mutex> lock{mu_};
    response_buffer_.push(std::move(infer_response));
  }
  cv_.notify_all();
}

void*
ResponseGenerator::Id()
{
  return id_;
}

}}}  // namespace triton::backend::python
