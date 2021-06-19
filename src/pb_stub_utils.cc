// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_stub_utils.h"

namespace triton { namespace backend { namespace python {

int
numpy_to_triton_type(py::object data_type)
{
  py::module np = py::module::import("numpy");
  if (data_type.equal(np.attr("bool_")))
    return 1;
  else if (data_type.equal(np.attr("uint8")))
    return 2;
  else if (data_type.equal(np.attr("uint16")))
    return 3;
  else if (data_type.equal(np.attr("uint32")))
    return 4;
  else if (data_type.equal(np.attr("uint64")))
    return 5;
  else if (data_type.equal(np.attr("int8")))
    return 6;
  else if (data_type.equal(np.attr("int16")))
    return 7;
  else if (data_type.equal(np.attr("int32")))
    return 8;
  else if (data_type.equal(np.attr("int64")))
    return 9;
  else if (data_type.equal(np.attr("float16")))
    return 10;
  else if (data_type.equal(np.attr("float32")))
    return 11;
  else if (data_type.equal(np.attr("float64")))
    return 12;
  else if (
      data_type.equal(np.attr("object_")) || data_type.equal(np.attr("bytes_")))
    return 13;
  return 0;
}

py::object
triton_to_numpy_type(int data_type)
{
  py::module np = py::module::import("numpy");
  if (data_type == 1)
    return np.attr("bool_");
  else if (data_type == 2)
    return np.attr("uint8");
  else if (data_type == 3)
    return np.attr("uint16");
  else if (data_type == 4)
    return np.attr("uint32");
  else if (data_type == 5)
    return np.attr("uint64");
  else if (data_type == 6)
    return np.attr("int8");
  else if (data_type == 7)
    return np.attr("int16");
  else if (data_type == 8)
    return np.attr("int32");
  else if (data_type == 9)
    return np.attr("int64");
  else if (data_type == 10)
    return np.attr("float16");
  else if (data_type == 11)
    return np.attr("float32");
  else if (data_type == 12)
    return np.attr("float64");
  else if (data_type == 13)
    return np.attr("object_");
  return py::none();
}

DLDataType
convert_triton_to_dlpack_type(int data_type)
{
  DLDataType dl_dtype;
  DLDataTypeCode dl_code;

  // Number of bits required for the data type.
  size_t dt_size = 0;
  // TODO: Fix for TYPE_BYTES

  dl_dtype.lanes = 1;
  if (data_type == 1) {
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 1;
  } else if (data_type == 2) {
    dl_code = DLDataTypeCode::kDLUInt;
    dt_size = 8;
  } else if (data_type == 3) {
    dl_code = DLDataTypeCode::kDLUInt;
    dt_size = 16;
  } else if (data_type == 4) {
    dl_code = DLDataTypeCode::kDLUInt;
    dt_size = 32;
  } else if (data_type == 5) {
    dl_code = DLDataTypeCode::kDLUInt;
    dt_size = 64;
  } else if (data_type == 6) {
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 8;
  } else if (data_type == 7) {
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 16;
  } else if (data_type == 8) {
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 32;
  } else if (data_type == 9) {
    dl_code = DLDataTypeCode::kDLInt;
    dt_size = 64;
  } else if (data_type == 10) {
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 16;
  } else if (data_type == 11) {
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 32;
  } else if (data_type == 12) {
    dl_code = DLDataTypeCode::kDLFloat;
    dt_size = 64;
  }
  // else if (data_type == 13)
  //   1 == 1;
  // return np.attr("object_"); TODO
  dl_dtype.code = dl_code;
  dl_dtype.bits = dt_size;

  return dl_dtype;
}

}}}  // namespace triton::backend::python
