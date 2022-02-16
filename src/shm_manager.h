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

#include <sys/wait.h>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <iostream>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <vector>

#pragma once

namespace triton { namespace backend { namespace python {
namespace bi = boost::interprocess;

template <typename T>
struct AllocatedSharedMemory {
  AllocatedSharedMemory() = default;
  AllocatedSharedMemory(
      std::unique_ptr<T, std::function<void(T*)>>& data,
      bi::managed_external_buffer::handle_t handle)
      : data_(std::move(data)), handle_(handle)
  {
  }

  std::unique_ptr<T, std::function<void(T*)>> data_;
  bi::managed_external_buffer::handle_t handle_;
};

class SharedMemoryManager {
 public:
  SharedMemoryManager(
      const std::string& shm_region_name, size_t shm_size, bool create);

  template <typename T>
  AllocatedSharedMemory<T> Construct()
  {
    T* obj;
    bi::managed_external_buffer::handle_t handle;

    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(sizeof(T));
      obj = reinterpret_cast<T*>(managed_buffer_->allocate(sizeof(T)));
      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(obj));
    }

    return WrapObjectInUniquePtr(obj, handle);
  }

  template <typename T>
  AllocatedSharedMemory<T> ConstructAligned()
  {
    T* obj;
    bi::managed_external_buffer::handle_t handle;

    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(sizeof(T));
      const std::size_t alignment = 32;
      obj = reinterpret_cast<T*>(
          managed_buffer_->allocate_aligned(sizeof(T), alignment));
      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(obj));
    }

    return WrapObjectInUniquePtr(obj, handle);
  }

  template <typename T>
  AllocatedSharedMemory<T> ConstructMany(size_t number)
  {
    T* object;
    bi::managed_external_buffer::handle_t handle;
    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(sizeof(T) * number);

      object =
          reinterpret_cast<T*>(managed_buffer_->allocate(sizeof(T) * number));
      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(object));
    }

    return WrapObjectInUniquePtr(object, handle);
  }

  std::unique_ptr<bi::managed_external_buffer>& ManagedBuffer()
  {
    return managed_buffer_;
  }

  template <typename T>
  AllocatedSharedMemory<T> Load(bi::managed_external_buffer::handle_t handle)
  {
    T* object_ptr;
    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(0);
      object_ptr = reinterpret_cast<T*>(
          managed_buffer_->get_address_from_handle(handle));
    }

    return WrapObjectInUniquePtr(object_ptr, handle);
  }

  size_t FreeMemory();

  void Deallocate(bi::managed_external_buffer::handle_t handle)
  {
    bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
    GrowIfNeeded(0);
    void* ptr = managed_buffer_->get_address_from_handle(handle);
    managed_buffer_->deallocate(ptr);
  }

  ~SharedMemoryManager() noexcept(false);

 private:
  std::string shm_region_name_;
  std::unique_ptr<bi::managed_external_buffer> managed_buffer_;
  std::unique_ptr<bi::shared_memory_object> shm_obj_;
  std::shared_ptr<bi::mapped_region> shm_map_;
  std::vector<std::shared_ptr<bi::mapped_region>> old_shm_maps_;
  uint64_t current_capacity_;
  bi::interprocess_mutex* shm_mutex_;
  size_t shm_growth_bytes_;
  uint64_t* total_size_;
  bool create_;
  void GrowIfNeeded(size_t bytes);

  template <typename T>
  AllocatedSharedMemory<T> WrapObjectInUniquePtr(
      T* object, const bi::managed_external_buffer::handle_t& handle)
  {
    // Custom deleter to deallocate the object when it goes out of scope.
    std::function<void(T*)> deleter = [this, handle](T* memory) {
      if (memory != nullptr) {
        this->Deallocate(handle);
      }
    };

    auto data = std::unique_ptr<T, decltype(deleter)>(object, deleter);
    return AllocatedSharedMemory<T>(data, handle);
  }
};

}}}  // namespace triton::backend::python
