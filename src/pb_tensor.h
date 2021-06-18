// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"
#include <dlpack/dlpack.h>

namespace py = pybind11;

namespace triton { namespace backend { namespace python {
class PbTensor {
  std::string name_;
  py::array numpy_array_;
  int dtype_;
  void* memory_ptr_;
  int64_t memory_type_id_;
  std::vector<int64_t> dims_;
  TRITONSERVER_MemoryType memory_type_;

 public:
  PbTensor(std::string name, py::object numpy_array);
  PbTensor(std::string name, py::object numpy_array, int dtype);
  PbTensor(std::string name, std::vector<int64_t> dims, int dtype,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      void* memory_ptr);
  py::capsule ToDLPack();
  const std::string& Name();
  py::array& AsNumpy();
  int TritonDtype();
};
}}}  // namespace triton::backend::python
