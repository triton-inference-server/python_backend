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

#include "shm_manager.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>
#include <string>
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

SharedMemory::SharedMemory(
    const std::string& shm_key, int64_t default_byte_size,
    int64_t shm_growth_bytes, bool truncate)
{
  if (truncate) {
    shm_obj_ = bi::shared_memory_object(
        bi::open_or_create, shm_key.c_str(), bi::read_write);
  } else {
    shm_obj_ = bi::shared_memory_object(
        bi::open_only, shm_key.c_str(), bi::read_write);
  }

  shm_growth_bytes_ = shm_growth_bytes;
  try {
    shm_obj_.truncate(default_byte_size);
  }
  catch (bi::interprocess_exception& ex) {
    std::string error_message =
        ("Unable to initialize shared memory key '" + shm_key +
         "' to requested size (" + std::to_string(default_byte_size) +
         " bytes). If you are running Triton inside docker, use '--shm-size' "
         "flag to control the shared memory region size. Each Python backend "
         "model instance requires at least 64MBs of shared memory. Flag "
         "'--shm-size=5G' should be sufficient for common usecases. Error: " +
         ex.what());

    // Remove the shared memory region if there was an error.
    bi::shared_memory_object::remove(shm_key_.c_str());
    throw PythonBackendException(std::move(error_message));
  }

  shm_map_ = std::make_unique<bi::mapped_region>(shm_obj_, bi::read_write);
  shm_addr_ = (char*)shm_map_->get_address();

  capacity_ = (size_t*)shm_addr_;
  *capacity_ = default_byte_size;
  current_capacity_ = *capacity_;

  // Set offset address
  offset_ = (off_t*)((char*)shm_addr_ + sizeof(size_t));

  if (truncate) {
    *offset_ = 0;
    *offset_ += sizeof(off_t);
    *offset_ += sizeof(size_t);
  }

  shm_key_ = shm_key;
}

SharedMemory::~SharedMemory() noexcept(false)
{
  bi::shared_memory_object::remove(shm_key_.c_str());
}

void
SharedMemory::Map(char** shm_addr, size_t byte_size, off_t& offset)
{
  size_t shm_bytes_added = 0;
  while (*offset_ + byte_size >= *capacity_) {
    // Increase the shared memory pool size by the amount of bytes available.
    *capacity_ += shm_growth_bytes_;
    shm_bytes_added += shm_growth_bytes_;
  }

  if (shm_bytes_added > 0) {
    try {
      shm_obj_.truncate(*capacity_);
    }
    catch (bi::interprocess_exception& ex) {
      *capacity_ -= shm_bytes_added;
      std::string error_message =
          ("Failed to increase the shared memory pool size for key '" +
           shm_key_ + "' to " + std::to_string(*capacity_) +
           " bytes. If you are running Triton inside docker, use '--shm-size' "
           "flag to control the shared memory region size. Error: " +
           ex.what());
      throw PythonBackendException(error_message);
    }
  }

  UpdateSharedMemory();

  *shm_addr = shm_addr_ + *offset_;
  offset = *offset_;

  *offset_ += byte_size;
}

void
SharedMemory::UpdateSharedMemory()
{
  if (current_capacity_ != *capacity_) {
    std::unique_ptr<bi::mapped_region> new_map;
    try {
      new_map = std::make_unique<bi::mapped_region>(shm_obj_, bi::read_write);
    }
    catch (bi::interprocess_exception& ex) {
      std::string error_message = std::string(
                                      "unable to process address space or "
                                      "shared-memory descriptor, err:") +
                                  ex.what();
      throw PythonBackendException(error_message);
    }

    old_shm_maps_.emplace_back(std::move(shm_map_));
    current_capacity_ = *capacity_;
    shm_map_ = std::move(new_map);
    shm_addr_ = (char*)shm_map_->get_address();
  }
}

void
SharedMemory::MapOffset(char** shm_addr, off_t offset)
{
  // Update shared memory pointer and capacity if necessary.
  UpdateSharedMemory();
  *shm_addr = shm_addr_ + offset;
}

void
SharedMemory::SetOffset(off_t offset)
{
  *offset_ = offset;
}

}}}  // namespace triton::backend::python
