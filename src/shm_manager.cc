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
#include <unistd.h>
#include <string>
#include "triton/backend/backend_model.h"
#include "triton/core/tritonserver.h"


namespace triton { namespace backend { namespace python {

SharedMemory::SharedMemory(const std::string& shm_key)
{
  shm_fd_ =
      shm_open(shm_key.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (shm_fd_ == -1) {
    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("unable to get shared memory descriptor for shared-memory key '" +
         shm_key + "'")
            .c_str());
    throw BackendModelException(err);
  }

  size_t default_byte_size = 1024 * 1024 * 128;  // 128MB
  int res = ftruncate(shm_fd_, default_byte_size);
  if (res == -1) {
    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("unable to initialize shared-memory key '" + shm_key +
         "' to requested size: " + std::to_string(default_byte_size) + " bytes")
            .c_str());
    throw BackendModelException(err);
  }

  void* map_addr = mmap(
      NULL, default_byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);

  if (map_addr == MAP_FAILED) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("unable to process address space or shared-memory descriptor: " +
         std::to_string(shm_fd_))
            .c_str());
    throw BackendModelException(err);
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
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "unable to munmap shared memory region");
      throw BackendModelException(err);
    }
  }

  // Close fd
  if (close(shm_fd_) == -1) {
    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("unable to close shared-memory descriptor: " + std::to_string(shm_fd_))
            .c_str());
    throw BackendModelException(err);
  }

  // Unlink shared memory
  int error = shm_unlink(shm_key_.c_str());
  if (error == -1) {
    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("unable to unlink shared memory for key '" + shm_key_ + "'").c_str());
    throw BackendModelException(err);
  }
}

TRITONSERVER_Error*
SharedMemory::Map(char** shm_addr, size_t byte_size, off_t& offset)
{
  while (*offset_ + byte_size >= *capacity_) {
    // Increase the shared memory pool size by one page size.
    *capacity_ = *offset_ + byte_size + PAGE_SIZE;
    if (ftruncate(shm_fd_, *capacity_) == -1) {
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Failed to increase the shared memory pool size for key '" +
           shm_key_ + "' to " + std::to_string(*capacity_) + " bytes")
              .c_str());
      return err;
    }
  }

  // Update shared memory pointer and capacity if necessary.
  RETURN_IF_ERROR(UpdateSharedMemory());

  *shm_addr = shm_addr_ + *offset_;
  offset = *offset_;

  *offset_ += byte_size;

  return nullptr;  // success
}

TRITONSERVER_Error*
SharedMemory::UpdateSharedMemory()
{
  if (current_capacity_ != *capacity_) {
    void* map_addr =
        mmap(NULL, *capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);

    if (map_addr == MAP_FAILED) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unable to process address space or shared-memory descriptor: " +
           std::to_string(shm_fd_))
              .c_str());
      return err;
    }

    old_shm_addresses_.push_back({current_capacity_, shm_addr_});
    current_capacity_ = *capacity_;
    shm_addr_ = (char*)map_addr;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
SharedMemory::MapOffset(char** shm_addr, size_t byte_size, off_t offset)
{
  // Update shared memory pointer and capacity if necessary.
  RETURN_IF_ERROR(UpdateSharedMemory());
  *shm_addr = shm_addr_ + offset;

  return nullptr;  // success
}

void
SharedMemory::SetOffset(off_t offset)
{
  *offset_ = offset;
}

}}}  // namespace triton::backend::python
