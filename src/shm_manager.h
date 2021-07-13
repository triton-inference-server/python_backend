// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>


namespace triton { namespace backend { namespace python {

class SharedMemory {
  std::string shm_key_;
  size_t* capacity_;
  off_t* offset_;
  char* shm_addr_;

  // Current capcity, local to each process.
  size_t current_capacity_;

  // Amount of bytes to grow the shared memory when the pool is completely used.
  int64_t shm_growth_bytes_;

  // Get the amount of shared memory available.
  size_t GetAvailableSharedMemory();
  boost::interprocess::shared_memory_object shm_obj_;
  std::unique_ptr<boost::interprocess::mapped_region> shm_map_;
  std::vector<std::unique_ptr<boost::interprocess::mapped_region>>
      old_shm_maps_;

  void UpdateSharedMemory();

 public:
  SharedMemory(
      const std::string& shm_key, int64_t default_byte_size,
      int64_t shm_growth_bytes, bool truncate = false);
  void MapOffset(char** shm_addr, off_t offset);
  void Map(char** shm_addr, size_t byte_size, off_t& offset);
  void SetOffset(off_t offset);
  ~SharedMemory() noexcept(false);
};

}}}  // namespace triton::backend::python
