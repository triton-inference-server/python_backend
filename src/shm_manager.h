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

#include <sys/wait.h>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <memory>
#include <vector>

#pragma once

namespace bi = boost::interprocess;


class SharedMemoryManager {
 public:
  SharedMemoryManager(
      const std::string& shm_region_name, size_t shm_size, bool create);

  template <typename T, typename U, typename... Args>
  std::shared_ptr<U> Construct(Args&... args)
  {
    T* obj_ptr;
    bi::managed_external_buffer::handle_t handle;

    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(sizeof(T));
      obj_ptr = managed_buffer_->construct<T>(bi::anonymous_instance)();
      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(obj_ptr));
    }

    return U::Create(obj_ptr, handle, args...);
  }

  template <typename T, typename U, typename... Args>
  std::shared_ptr<U> ConstructMany(size_t number, Args&... args)
  {
    T* object;
    bi::managed_external_buffer::handle_t handle;
    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(sizeof(T) * number);

      object = managed_buffer_->construct<T>(bi::anonymous_instance)[number]();
      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(object));
    }

    return U::Create(object, handle, number, args...);
  }

  std::unique_ptr<bi::managed_external_buffer>& ManagedBuffer()
  {
    return managed_buffer_;
  }

  template <typename T, typename U, typename... Args>
  std::shared_ptr<U> Load(
      bi::managed_external_buffer::handle_t handle, Args&... args,
      bool release = false)
  {
    T* object_ptr;
    {
      bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
      GrowIfNeeded(0);
      object_ptr = reinterpret_cast<T*>(
          managed_buffer_->get_address_from_handle(handle));
    }
    return U::Load(object_ptr, handle, args...);
  }

  size_t FreeMemory();

  template <typename T>
  void DestroyPtr(bi::managed_external_buffer::handle_t handle)
  {
    bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};
    GrowIfNeeded(0);
    T* ptr =
        reinterpret_cast<T*>(managed_buffer_->get_address_from_handle(handle));
    managed_buffer_->destroy_ptr(ptr);
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
};

// All the objects that want to be stored in shared memory must
// extend this class.
template <typename T>
class ShmObject {
 public:
  void Release() { released_ = true; }
  ~ShmObject()
  {
    if (released_) {
      shm_manager_->DestroyPtr<T>(handle_);
    }
  }

 protected:
  bi::managed_external_buffer::handle_t handle_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  T* data_;
  bool released_;
};

template <typename T>
class Array : ShmObject<T> {
  T* data_;
  std::size_t array_size_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;

 public:
  static std::unique_ptr<Array<T>> Create(
      T* ptr, bi::managed_external_buffer::handle_t handle, std::size_t size,
      std::shared_ptr<SharedMemoryManager>& shm_manager)
  {
    auto array = std::make_unique<Array<T>>();
    array->data_ = ptr;
    array->array_size_ = size;
    array->handle_ = handle;
    array->shm_manager_ = shm_manager;

    return array;
  }

  static std::unique_ptr<Array<T>> Load(
      T* ptr, bi::managed_external_buffer::handle_t handle,
      std::shared_ptr<SharedMemoryManager>& shm_manager, std::size_t size)
  {
    auto array = std::make_unique<Array<T>>();
    array->data_ = ptr;
    array->array_size_ = size;
    array->handle_ = handle;
    array->shm_manager_ = shm_manager;

    return array;
  }

  const T& operator[](std::size_t idx) const { return data_[idx]; }
  T& operator[](std::size_t idx) { return data_[idx]; }
};
