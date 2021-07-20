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

#include "infer_request.h"

#include "infer_response.h"

namespace triton { namespace backend { namespace python {

InferRequest::InferRequest(
    const std::string& request_id, uint64_t correlation_id,
    const std::vector<PbTensor>& inputs,
    const std::vector<std::string>& requested_output_names,
    const std::string& model_name, const int64_t model_version)
    : request_id_(request_id), correlation_id_(correlation_id), inputs_(inputs),
      requested_output_names_(requested_output_names), model_name_(model_name),
      model_version_(model_version)
{
}

const std::vector<PbTensor>&
InferRequest::Inputs()
{
  return inputs_;
}

const std::string&
InferRequest::RequestId()
{
  return request_id_;
}

uint64_t
InferRequest::CorrelationId()
{
  return correlation_id_;
}

const std::vector<std::string>&
InferRequest::RequestedOutputNames()
{
  return requested_output_names_;
}

// std::unique_ptr<InferResponse>
// InferRequest::Exec()
// {

// }

}}}  // namespace triton::backend::python
