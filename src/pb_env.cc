// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "pb_env.h"

#include <archive.h>
#include <archive_entry.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "pb_utils.h"


namespace triton { namespace backend { namespace python {

EnvironmentManager::EnvironmentManager()
{
  char tmp_dir_template[PATH_MAX];
  strcpy(tmp_dir_template, "/tmp/python_env_XXXXXX");

  char* env_path = mkdtemp(tmp_dir_template);
  if (tmp_dir_template == nullptr) {
    throw PythonBackendException(
        "Failed to create temporary directory for Python environments.");
  }
  strcpy(base_path_, env_path);
}

std::string
EnvironmentManager::Extract(std::string env_path)
{
  std::lock_guard<std::mutex> lk(mutex_);
  boost::filesystem::path canonical_env_path =
      boost::filesystem::canonical(env_path);

  // Extract only if the env has not been extracted yet.
  if (env_map_.find(canonical_env_path.string()) == env_map_.end()) {
    boost::filesystem::path dst_env_path(
        std::string(base_path_) + "/" + std::to_string(env_map_.size()));

    bool directory_created = boost::filesystem::create_directory(dst_env_path);
    if (directory_created) {
      ExtractTarFile(canonical_env_path.string(), dst_env_path.string());
    } else {
      throw PythonBackendException(
          std::string("Failed to create environment directory for '") +
          dst_env_path.c_str() + "'.");
    }

    // Add the path to the list of environments
    env_map_.insert({canonical_env_path.string(), dst_env_path.string()});
    return dst_env_path.string();
  } else {
    return env_map_.find(canonical_env_path.string())->second;
  }
}

EnvironmentManager::~EnvironmentManager()
{
  boost::filesystem::remove_all(base_path_);
}

}}}  // namespace triton::backend::python
