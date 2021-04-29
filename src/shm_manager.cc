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

#include "shm_manager.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>
#include <string>
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

SharedMemory::SharedMemory(
    const std::string& shm_key, int64_t default_byte_size,
    int64_t shm_growth_bytes, bool truncate)
{
  if (truncate) {
    shm_fd_ = shm_open(
        shm_key.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  } else {
    shm_fd_ = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  }
  if (shm_fd_ == -1) {
    std::unique_ptr<PythonBackendError> err =
        std::make_unique<PythonBackendError>();
    err->error_message =
        ("unable to get shared memory descriptor for shared-memory key '" +
         shm_key + "'");
    throw PythonBackendException(std::move(err));
  }

  shm_growth_bytes_ = shm_growth_bytes;
  int res = posix_fallocate(shm_fd_, 0, default_byte_size);
  if (res != 0) {
    std::unique_ptr<PythonBackendError> err =
        std::make_unique<PythonBackendError>();
    err->error_message =
        ("unable to initialize shared-memory key '" + shm_key +
         "' to requested size: " + std::to_string(default_byte_size) +
         " bytes");
    throw PythonBackendException(std::move(err));
  }

  void* map_addr = mmap(
      NULL, default_byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);

  if (map_addr == MAP_FAILED) {
    std::unique_ptr<PythonBackendError> err =
        std::make_unique<PythonBackendError>();
    err->error_message =
        ("unable to process address space or shared-memory descriptor: " +
         std::to_string(shm_fd_));
    throw PythonBackendException(std::move(err));
  }
  shm_addr_ = (char*)map_addr;

  capacity_ = (size_t*)shm_addr_;
  *capacity_ = default_byte_size;
  current_capacity_ = *capacity_;

  // Set offset address
  offset_ = (off_t*)((char*)shm_addr_ + sizeof(size_t));

  *offset_ = 0;
  *offset_ += sizeof(off_t);
  *offset_ += sizeof(size_t);

  shm_key_ = shm_key;
}

SharedMemory::~SharedMemory() noexcept(false)
{
  // TODO: Deallocate the first address
  for (auto& pair : old_shm_addresses_) {
    int status = munmap(pair.second, pair.first);
    if (status == -1) {
      std::unique_ptr<PythonBackendError> err =
          std::make_unique<PythonBackendError>();
      err->error_message = "unable to munmap shared memory region";
      throw PythonBackendException(std::move(err));
    }
  }

  // Close fd
  if (close(shm_fd_) == -1) {
    std::unique_ptr<PythonBackendError> err =
        std::make_unique<PythonBackendError>();
    err->error_message =
        ("unable to close shared-memory descriptor: " +
         std::to_string(shm_fd_));
    throw PythonBackendException(std::move(err));
  }

  // Unlink shared memory
  int error = shm_unlink(shm_key_.c_str());
  if (error == -1) {
    std::unique_ptr<PythonBackendError> err =
        std::make_unique<PythonBackendError>();
    err->error_message =
        ("unable to unlink shared memory for key '" + shm_key_ + "'");
    throw PythonBackendException(std::move(err));
  }
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
    if (posix_fallocate(shm_fd_, 0, *capacity_) != 0) {
      // Revert the capacity to the previous value
      *capacity_ -= shm_bytes_added;
      std::unique_ptr<PythonBackendError> err =
          std::make_unique<PythonBackendError>();
      err->error_message =
          ("Failed to increase the shared memory pool size for key '" +
           shm_key_ + "' to " + std::to_string(*capacity_) + " bytes.");
      throw PythonBackendException(std::move(err));
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
    void* map_addr =
        mmap(NULL, *capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);

    if (map_addr == MAP_FAILED) {
      std::unique_ptr<PythonBackendError> err =
          std::make_unique<PythonBackendError>();
      err->error_message =
          ("unable to process address space or shared-memory descriptor: " +
           std::to_string(shm_fd_));
      throw PythonBackendException(std::move(err));
    }

    old_shm_addresses_.push_back({current_capacity_, shm_addr_});
    current_capacity_ = *capacity_;
    shm_addr_ = (char*)map_addr;
  }
}

void
SharedMemory::MapOffset(char** shm_addr, size_t byte_size, off_t offset)
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
