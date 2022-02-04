// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>

#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

SharedMemoryManager::SharedMemoryManager(
    const std::string& shm_region_name, size_t shm_size, bool create)
{
  shm_region_name_ = shm_region_name;
  create_ = create;

  if (create) {
    shm_obj_ = std::make_unique<bi::shared_memory_object>(
        bi::open_or_create, shm_region_name.c_str(), bi::read_write);
    shm_obj_->truncate(shm_size);
  } else {
    shm_obj_ = std::make_unique<bi::shared_memory_object>(
        bi::open_only, shm_region_name.c_str(), bi::read_write);
  }

  current_capacity_ = shm_size;
  shm_map_ = std::make_shared<bi::mapped_region>(*shm_obj_, bi::read_write);
  old_shm_maps_.push_back(shm_map_);

  // Only create the managed external buffer for the stub process.
  if (create) {
    managed_buffer_ = std::make_unique<bi::managed_external_buffer>(
        bi::create_only, shm_map_->get_address(), shm_size);
  } else {
    int64_t shm_size = 0;
    shm_obj_->get_size(shm_size);
    managed_buffer_ = std::make_unique<bi::managed_external_buffer>(
        bi::open_only, shm_map_->get_address(), shm_size);
    current_capacity_ = shm_size;
  }

  // Construct a mutex in shared memory.
  shm_mutex_ =
      managed_buffer_->find_or_construct<bi::interprocess_mutex>("shm_mutex")();
  total_size_ = managed_buffer_->find_or_construct<uint64_t>("total size")();
  if (create) {
    *total_size_ = current_capacity_;
    new (shm_mutex_) bi::interprocess_mutex;
  }
}

void
SharedMemoryManager::GrowIfNeeded(size_t byte_size)
{
  if (*total_size_ != current_capacity_) {
    shm_map_ = std::make_shared<bi::mapped_region>(*shm_obj_, bi::read_write);
    old_shm_maps_.push_back(shm_map_);
    managed_buffer_ = std::make_unique<bi::managed_external_buffer>(
        bi::open_only, shm_map_->get_address(), *total_size_);
    current_capacity_ = *total_size_;
  }

  size_t free_memory = managed_buffer_->get_free_memory();

  // Multiply the requested bytes by 1.1
  size_t requested_bytes = byte_size * 1.1;
  if (requested_bytes > free_memory) {
    int64_t new_size = *total_size_ + (requested_bytes - free_memory);
    shm_obj_->truncate(new_size);

    shm_map_ = std::make_shared<bi::mapped_region>(*shm_obj_, bi::read_write);
    old_shm_maps_.push_back(shm_map_);
    managed_buffer_ = std::make_unique<bi::managed_external_buffer>(
        bi::open_only, shm_map_->get_address(), new_size);
    managed_buffer_->grow(new_size - current_capacity_);
    current_capacity_ = managed_buffer_->get_size();
    *total_size_ = new_size;
  }
}

size_t
SharedMemoryManager::FreeMemory()
{
  return managed_buffer_->get_free_memory();
}


SharedMemoryManager::~SharedMemoryManager() noexcept(false)
{
  if (create_) {
    bi::shared_memory_object::remove(shm_region_name_.c_str());
  }
}

}}}  // namespace triton::backend::python
