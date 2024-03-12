// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/detail/atomic.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "pb_exception.h"

namespace triton { namespace backend { namespace python {
namespace bi = boost::interprocess;

class CUDAMemoryPoolManager {
 public:
  CUDAMemoryPoolManager() : triton_memory_manager_(nullptr) {}

  void SetCUDAPoolAddress(const int32_t device_id, void* cuda_pool_address);

  void* CUDAPoolAddress(const int32_t device_id);

  void SetTritonMemoryManager(void* triton_memory_manager);

  void* TritonMemoryManager();

  bool UseCudaSharedPool(const int32_t device_id);

  // Return cuda pool address map
  std::unordered_map<int32_t, void*>& CUDAPoolAddressMap();

 private:
  // The base address of the Triton CUDA memory pool
  std::unordered_map<int32_t, void*> cuda_pool_address_map_;
  // The mutex to protect the cuda_pool_address_map_
  std::mutex mu_;
  // TRITONBACKEND_MemoryManager
  void* triton_memory_manager_;
};

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

// The alignment here is used to extend the size of the shared memory allocation
// struct to 16 bytes. The reason for this change is that when an aligned shared
// memory location is requested using the `Construct` method, the memory
// alignment of the object will be incorrect since the shared memory ownership
// info is placed in the beginning and the actual object is placed after that
// (i.e. 4 plus the aligned address is not 16-bytes aligned). The aligned memory
// is required by semaphore otherwise it may lead to SIGBUS error on ARM.
struct alignas(16) AllocatedShmOwnership {
  uint32_t ref_count_;
};

class SharedMemoryManager {
 public:
  SharedMemoryManager(
      const std::string& shm_region_name, size_t shm_size,
      size_t shm_growth_bytes, bool create);

  SharedMemoryManager(const std::string& shm_region_name);

  template <typename T>
  AllocatedSharedMemory<T> Construct(
      uint64_t count = 1, bool aligned = false,
      const char* debug_file = __builtin_FILE(),
      int debug_line = __builtin_LINE(),
      const char* debug_fn = __builtin_FUNCTION())
  {
    T* obj = nullptr;
    AllocatedShmOwnership* shm_ownership_data = nullptr;
    bi::managed_external_buffer::handle_t handle = 0;

    {
      bi::scoped_lock<bi::interprocess_mutex> guard{*shm_mutex_};
      std::size_t requested_bytes =
          sizeof(T) * count + sizeof(AllocatedShmOwnership);
      GrowIfNeeded(0);

      void* allocated_data;
      try {
        allocated_data = Allocate(requested_bytes, aligned);
      }
      catch (bi::bad_alloc& ex) {
        // Try to grow the shared memory region if the allocate failed.
        GrowIfNeeded(requested_bytes);
        allocated_data = Allocate(requested_bytes, aligned);
      }

      shm_ownership_data =
          reinterpret_cast<AllocatedShmOwnership*>(allocated_data);
      obj = reinterpret_cast<T*>(
          (reinterpret_cast<char*>(shm_ownership_data)) +
          sizeof(AllocatedShmOwnership));
      shm_ownership_data->ref_count_ = 1;

      handle = managed_buffer_->get_handle_from_address(
          reinterpret_cast<void*>(shm_ownership_data));

      LogShmDebugInfo(
          handle, "ALLOC", shm_ownership_data->ref_count_, debug_file,
          debug_line, debug_fn);
    }

    return WrapObjectInUniquePtr(
        obj, shm_ownership_data, handle, debug_file, debug_line, debug_fn);
  }

  template <typename T>
  AllocatedSharedMemory<T> Load(
      bi::managed_external_buffer::handle_t handle, bool unsafe = false,
      const char* debug_file = __builtin_FILE(),
      int debug_line = __builtin_LINE(),
      const char* debug_fn = __builtin_FUNCTION())
  {
    T* object_ptr;
    AllocatedShmOwnership* shm_ownership_data;

    {
      bi::scoped_lock<bi::interprocess_mutex> guard{*shm_mutex_};
      GrowIfNeeded(0);
      shm_ownership_data = reinterpret_cast<AllocatedShmOwnership*>(
          managed_buffer_->get_address_from_handle(handle));
      object_ptr = reinterpret_cast<T*>(
          reinterpret_cast<char*>(shm_ownership_data) +
          sizeof(AllocatedShmOwnership));
      if (!unsafe) {
        shm_ownership_data->ref_count_ += 1;
      }

      LogShmDebugInfo(
          handle, "LOAD", shm_ownership_data->ref_count_, debug_file,
          debug_line, debug_fn);
    }

    return WrapObjectInUniquePtr(
        object_ptr, shm_ownership_data, handle, debug_file, debug_line,
        debug_fn);
  }

  size_t FreeMemory();

  void Deallocate(bi::managed_external_buffer::handle_t handle)
  {
    bi::scoped_lock<bi::interprocess_mutex> guard{*shm_mutex_};
    GrowIfNeeded(0);
    void* ptr = managed_buffer_->get_address_from_handle(handle);
    managed_buffer_->deallocate(ptr);
  }

  void DeallocateUnsafe(bi::managed_external_buffer::handle_t handle)
  {
    void* ptr = managed_buffer_->get_address_from_handle(handle);
    managed_buffer_->deallocate(ptr);
  }

  void GrowIfNeeded(uint64_t bytes);
  bi::interprocess_mutex* Mutex() { return shm_mutex_; }

  void SetDeleteRegion(bool delete_region);

  std::unique_ptr<CUDAMemoryPoolManager>& GetCUDAMemoryPoolManager()
  {
    return cuda_memory_pool_manager_;
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
  bool delete_region_;
  std::unique_ptr<CUDAMemoryPoolManager> cuda_memory_pool_manager_;

  std::ofstream shm_debug_info_;

  template <typename T>
  AllocatedSharedMemory<T> WrapObjectInUniquePtr(
      T* object, AllocatedShmOwnership* shm_ownership_data,
      const bi::managed_external_buffer::handle_t& handle,
      const char* debug_file, int debug_line, const char* debug_fn)
  {
    // Custom deleter to conditionally deallocate the object
    std::function<void(T*)> deleter = [this, handle, shm_ownership_data,
                                       debug_file, debug_line,
                                       debug_fn](T* memory) {
      bool destroy = false;
      bi::scoped_lock<bi::interprocess_mutex> guard{*shm_mutex_};
      // Before using any shared memory function you need to make sure that you
      // are using the correct mapping. For example, shared memory growth may
      // happen between the time an object was created and the time the object
      // gets destructed.
      GrowIfNeeded(0);
      shm_ownership_data->ref_count_ -= 1;
      if (shm_ownership_data->ref_count_ == 0) {
        destroy = true;
      }
      if (destroy) {
        DeallocateUnsafe(handle);

        LogShmDebugInfo(handle, "DEALLOC", 0, debug_file, debug_line, debug_fn);
      } else {
        LogShmDebugInfo(
            handle, "UNLOAD", shm_ownership_data->ref_count_, debug_file,
            debug_line, debug_fn);
      }
    };

    auto data = std::unique_ptr<T, decltype(deleter)>(object, deleter);
    return AllocatedSharedMemory<T>(data, handle);
  }

  void* Allocate(uint64_t requested_bytes, bool aligned)
  {
    void* ptr;
    if (aligned) {
      const std::size_t alignment = 32;
      ptr = managed_buffer_->allocate_aligned(requested_bytes, alignment);
    } else {
      ptr = managed_buffer_->allocate(requested_bytes);
    }

    return ptr;
  }

  void LogShmDebugInfo(
      bi::managed_external_buffer::handle_t handle, const std::string& action,
      uint32_t ref_count_after_action, const char* file, int line,
      const char* fn)
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    shm_debug_info_ << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min
                    << ':' << std::setw(2) << tm_time.tm_sec << '.'
                    << std::setw(6) << tv.tv_usec << ", ";
    shm_debug_info_ << handle << ", " << action << ", "
                    << ref_count_after_action << ", ";
    shm_debug_info_ << file << ":" << line << ", " << fn << "\n";
    shm_debug_info_.flush();
  }
};
}}}  // namespace triton::backend::python
