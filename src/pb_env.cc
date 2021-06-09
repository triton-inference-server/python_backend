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
#include <fts.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "pb_utils.h"


namespace triton { namespace backend { namespace python {

void
RecursiveDirectoryDelete(const char* dir)
{
  FTS* ftsp = NULL;
  FTSENT* curr;

  char* files[] = {(char*)dir, NULL};

  ftsp = fts_open(files, FTS_NOCHDIR | FTS_PHYSICAL | FTS_XDEV, NULL);
  if (!ftsp) {
  }

  while ((curr = fts_read(ftsp))) {
    switch (curr->fts_info) {
      case FTS_NS:
      case FTS_DNR:
      case FTS_ERR:
        throw PythonBackendException(
            std::string("fts_read error: ") + curr->fts_accpath +
            " error: " + strerror(curr->fts_errno));
        break;

      case FTS_DC:
      case FTS_DOT:
      case FTS_NSOK:
        break;

      case FTS_D:
        // Do nothing. Directories are deleted in FTS_DP
        break;

      case FTS_DP:
      case FTS_F:
      case FTS_SL:
      case FTS_SLNONE:
      case FTS_DEFAULT:
        if (remove(curr->fts_accpath) < 0) {
          fts_close(ftsp);
          throw PythonBackendException(
              std::string("Failed to remove ") + curr->fts_path +
              " error: " + strerror(curr->fts_errno));
        }
        break;
    }
  }

  fts_close(ftsp);
}

EnvironmentManager::EnvironmentManager()
{
  char tmp_dir_template[PATH_MAX + 1];
  strcpy(tmp_dir_template, "/tmp/python_env_XXXXXX");

  char* env_path = mkdtemp(tmp_dir_template);
  if (env_path == nullptr) {
    throw PythonBackendException(
        "Failed to create temporary directory for Python environments.");
  }
  strcpy(base_path_, tmp_dir_template);
}

std::string
EnvironmentManager::ExtractIfNotExtracted(std::string env_path)
{
  // Lock the mutex. Only a single thread should modify the map.
  std::lock_guard<std::mutex> lk(mutex_);
  char canonical_env_path[PATH_MAX + 1];

  char* err = realpath(env_path.c_str(), canonical_env_path);
  if (err == nullptr) {
    throw PythonBackendException(
        std::string("Failed to get the canonical path for ") + env_path + ".");
  }

  // Extract only if the env has not been extracted yet.
  if (env_map_.find(canonical_env_path) == env_map_.end()) {
    std::string dst_env_path(
        std::string(base_path_) + "/" + std::to_string(env_map_.size()));

    std::string canonical_env_path_str(canonical_env_path);

    int status =
        mkdir(dst_env_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (status == 0) {
      ExtractTarFile(canonical_env_path_str, dst_env_path);
    } else {
      throw PythonBackendException(
          std::string("Failed to create environment directory for '") +
          dst_env_path.c_str() + "'.");
    }

    // Add the path to the list of environments
    env_map_.insert({canonical_env_path, dst_env_path});
    return dst_env_path;
  } else {
    return env_map_.find(canonical_env_path)->second;
  }
}

EnvironmentManager::~EnvironmentManager()
{
  RecursiveDirectoryDelete(base_path_);
}

}}}  // namespace triton::backend::python
