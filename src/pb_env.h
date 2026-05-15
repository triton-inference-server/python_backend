// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#ifdef WIN32
#include <windows.h>
#undef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
namespace triton { namespace backend { namespace python {

void ExtractTarFile(std::string& archive_path, std::string& dst_path);

bool FileExists(std::string& path);

//
// A class that manages Python environments
//
#ifndef _WIN32
class EnvironmentManager {
 public:
  class Environment {
   public:
    Environment(
        const std::string& source, const std::string& path,
        const time_t& last_modified_time);
    ~Environment();

    void Update(const time_t& last_modified_time);
    void AddOwner() { ++owners_counter_; }
    size_t RemoveOwner() { return --owners_counter_; }

    const std::string& Source() const { return source_; }
    const std::string& Path() const { return path_; }
    const time_t& LastModifiedTime() const { return last_modified_time_; }

   private:
    void Extract();
    void Delete();

    std::string source_;
    std::string path_;
    time_t last_modified_time_;

    size_t owners_counter_ = 0;
  };

  class EnvironmentProxy {
   public:
    EnvironmentProxy(const Environment* env) : env_(env) {}

    EnvironmentProxy(EnvironmentProxy&& other_proxy) : env_(other_proxy.env_)
    {
      other_proxy.env_ = nullptr;
    }

    ~EnvironmentProxy() = default;

    const std::string& Source() const & { return env_->Source(); }
    const std::string& Path() const & { return env_->Path(); }
    const time_t& LastModifiedTime() const & { return env_->LastModifiedTime(); }

   private:
    const Environment* env_;
  };

  class EnvironmentGuard {
   public:
    EnvironmentGuard(EnvironmentManager* manager, Environment* environment);

    EnvironmentGuard(const EnvironmentGuard&) = delete;
    EnvironmentGuard(EnvironmentGuard&&);

    EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;
    EnvironmentGuard& operator=(EnvironmentGuard&&);

    const EnvironmentProxy* operator->() const { return &environment_proxy_; }
    const EnvironmentProxy& operator*() const { return environment_proxy_; }

    ~EnvironmentGuard();

   private:
    EnvironmentManager* manager_;
    Environment* environment_;
    EnvironmentProxy environment_proxy_;
  };

  EnvironmentManager();
  friend class EnvironmentGuard;

  // Extracts the tar.gz file in the 'env_path' if it has not been
  // already extracted. Returns nullopt when env_path is an uncompressed
  // directory (caller uses that path directly).
  std::optional<EnvironmentGuard> ExtractIfNotExtracted(
      const std::string& env_path);

  ~EnvironmentManager();

 private:
  void DropEnvironment(Environment& environment);
  Environment& GetEnvironment(const std::string& env_path);

  size_t env_path_counter_ = 0;
  std::map<std::string, Environment> env_map_;
  char base_path_[PATH_MAX + 1];
  std::mutex mutex_;
};
#endif

}}}  // namespace triton::backend::python
