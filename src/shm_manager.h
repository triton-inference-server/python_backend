// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "triton/core/tritonserver.h"
#include <memory>
#include <string>
#include <unistd.h>

const long PAGE_SIZE = sysconf(_SC_PAGE_SIZE);
namespace triton { namespace backend { namespace python {

class SharedMemory {
  int shm_fd_;
  std::string shm_key_;
  size_t *capacity_;
  off_t *offset_;

public:
  SharedMemory(const std::string& shm_key);
  TRITONSERVER_Error* MapOffset(char **shm_addr, size_t byte_size, off_t offset);
  TRITONSERVER_Error* Map(char **shm_addr, size_t byte_size, off_t &offset);
  void SetOffset(off_t offset);
  ~SharedMemory() noexcept(false);
};

// TRITONSERVER_Error* 
// UnmapSharedMemory(void* shm_addr, size_t byte_size);
// 
// TRITONSERVER_Error* 
// UnlinkSharedMemoryRegion(std::string shm_key);
// 
// TRITONSERVER_Error* 
// CloseSharedMemory(int shm_fd);
// 
// TRITONSERVER_Error* 
// MapSharedMemory(int shm_fd, size_t offset, size_t byte_size, void** shm_addr);

}}}  // namespace triton::backend::python
