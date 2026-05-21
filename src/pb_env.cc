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

#include "pb_env.h"

#ifndef _WIN32
#include <archive.h>
#include <archive_entry.h>
#include <fts.h>
#endif
#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "pb_utils.h"


namespace triton { namespace backend { namespace python {

bool
FileExists(std::string& path)
{
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

void
LastModifiedTime(const std::string& path, time_t* last_modified_time)
{
  struct stat result;
  if (stat(path.c_str(), &result) == 0) {
    *last_modified_time = result.st_mtime;
  } else {
    throw PythonBackendException(std::string(
        "LastModifiedTime() failed as file \'" + path +
        std::string("\' does not exists.")));
  }
}

// FIXME: [DLIS-5969]: Develop platforom-agnostic functions
// to support custom python environments.
#ifndef _WIN32
void
CopySingleArchiveEntry(archive* input_archive, archive* output_archive)
{
  const void* buff;
  size_t size;
#if ARCHIVE_VERSION_NUMBER >= 3000000
  int64_t offset;
#else
  off_t offset;
#endif

  for (;;) {
    int return_status;
    return_status =
        archive_read_data_block(input_archive, &buff, &size, &offset);
    if (return_status == ARCHIVE_EOF)
      break;
    if (return_status != ARCHIVE_OK)
      throw PythonBackendException(
          "archive_read_data_block() failed with error code = " +
          std::to_string(return_status));

    return_status =
        archive_write_data_block(output_archive, buff, size, offset);
    if (return_status != ARCHIVE_OK) {
      throw PythonBackendException(
          "archive_write_data_block() failed with error code = " +
          std::to_string(return_status) + ", error message is " +
          archive_error_string(output_archive));
    }
  }
}

void
ExtractTarFile(std::string& archive_path, std::string& dst_path)
{
  char current_directory[PATH_MAX];
  if (getcwd(current_directory, PATH_MAX) == nullptr) {
    throw PythonBackendException(
        (std::string("Failed to get the current working directory. Error: ") +
         std::strerror(errno)));
  }
  if (chdir(dst_path.c_str()) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + dst_path +
         " Error: " + std::strerror(errno))
            .c_str());
  }

  struct archive_entry* entry;
  int flags = ARCHIVE_EXTRACT_TIME;

  struct archive* input_archive = archive_read_new();
  struct archive* output_archive = archive_write_disk_new();
  archive_write_disk_set_options(output_archive, flags);

  archive_read_support_filter_gzip(input_archive);
  archive_read_support_format_tar(input_archive);

  if (archive_path.size() == 0) {
    throw PythonBackendException("The archive path is empty.");
  }

  THROW_IF_ERROR(
      "archive_read_open_filename() failed.",
      archive_read_open_filename(
          input_archive, archive_path.c_str(), 10240 /* block_size */));

  while (true) {
    int read_status = archive_read_next_header(input_archive, &entry);
    if (read_status == ARCHIVE_EOF)
      break;
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(
          std::string("archive_read_next_header() failed with error code = ") +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(input_archive));
    }

    read_status = archive_write_header(output_archive, entry);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_header() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }

    CopySingleArchiveEntry(input_archive, output_archive);

    read_status = archive_write_finish_entry(output_archive);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_finish_entry() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }
  }

  archive_read_close(input_archive);
  archive_read_free(input_archive);

  archive_write_close(output_archive);
  archive_write_free(output_archive);

  // Revert the directory change.
  if (chdir(current_directory) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + current_directory)
            .c_str());
  }
}

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
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "EnvironmentManager constructor: initializing Python env manager");
  char tmp_dir_template[PATH_MAX + 1];
  strcpy(tmp_dir_template, "/tmp/python_env_XXXXXX");

  char* env_path = mkdtemp(tmp_dir_template);
  if (env_path == nullptr) {
    throw PythonBackendException(
        "Failed to create temporary directory for Python environments.");
  }
  strcpy(base_path_, tmp_dir_template);
}


std::optional<EnvironmentManager::EnvironmentGuard>
EnvironmentManager::ExtractIfNotExtracted(const std::string& env_path)
{
  std::string canonical_env_path = [&] {
    char canonical_env_path[PATH_MAX + 1];
    char* err = realpath(env_path.c_str(), canonical_env_path);
    if (err == nullptr) {
      throw PythonBackendException(
          "Failed to get the canonical path for " + env_path + ".");
    }
    return std::string(canonical_env_path);
  }();

  // If the path is not a conda-packed file, then bypass the extraction process
  struct stat info;
  if (stat(canonical_env_path.c_str(), &info) != 0) {
    throw PythonBackendException(
        "stat() of : " + canonical_env_path + " returned error.");
  } else if (S_ISDIR(info.st_mode)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        ("Returning canonical path since EXECUTION_ENV_PATH does "
         "not contain compressed path. Path: " +
         canonical_env_path)
            .c_str());
    return std::nullopt;
  }

  auto& env = GetEnvironment(canonical_env_path);
  return EnvironmentGuard(this, &env);
}

EnvironmentManager::Environment&
EnvironmentManager::GetEnvironment(const std::string& env_path)
{
  // Lock the mutex. Only a single thread should modify the map.
  std::lock_guard<std::mutex> lk(mutex_);

  time_t last_modified_time;
  LastModifiedTime(env_path, &last_modified_time);

  bool env_extracted = false;
  bool re_extraction = false;

  const std::string& env_key = env_path;
  auto env_itr = env_map_.find(env_key);
  Environment* env = nullptr;
  if (env_itr != env_map_.end()) {
    env = &env_itr->second;

    // Check if the environment has been modified and would
    // need to be extracted again

    if (env->LastModifiedTime() == last_modified_time) {
      env_extracted = true;
    } else {
      // Environment file has been updated. Need to clear
      // the previously extracted environment and extract
      // the environment to the same destination directory.
      re_extraction = true;
    }
  }

  // Extract only if the env has not been extracted yet.
  if (!env_extracted) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        ("Extracting Python execution env " + env_path).c_str());

    if (re_extraction) {
      // Just replace with new environment (by updated source)
      env->Update(last_modified_time);
    } else {
      std::string dst_env_path =
          std::string(base_path_) + "/" + std::to_string(env_path_counter_);
      ++env_path_counter_;

      // Add the environment to the list of environments
      env_itr = env_map_.try_emplace(env_key, env_path,
                                     dst_env_path, last_modified_time).first;
      env = &env_itr->second;
    }
  }

  env->AddOwner();
  return *env;
}

void
EnvironmentManager::DropEnvironment(EnvironmentProxy& env_proxy)
{
  std::lock_guard<std::mutex> lk(mutex_);

  const std::string& env_key = env.Source();
  auto env_itr = env_map_.find(env_key);
  auto& env = env_itr->second;

  size_t env_owners_counter = env.RemoveOwner();
  if (env_owners_counter == 0) {
    env_map_.erase(env.Source());
  }
}

EnvironmentManager::~EnvironmentManager()
{
  RecursiveDirectoryDelete(base_path_);
}

EnvironmentManager::Environment::Environment(
    const std::string& source, const std::string& path,
    const time_t& last_modified_time)
    : source_(source), path_(path), last_modified_time_(last_modified_time)
{
  Extract();
}

void
EnvironmentManager::Environment::Extract()
{
  int status = mkdir(path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if (status != 0) {
    throw PythonBackendException(
        "Failed to create environment directory for '" + path_ + "'.");
  }
  ExtractTarFile(source_, path_);
}

void
EnvironmentManager::Environment::Update(const time_t& last_modified_time)
{
  Delete();
  Extract();
  last_modified_time_ = last_modified_time;
}

void
EnvironmentManager::Environment::Delete()
{
  RecursiveDirectoryDelete(path_.c_str());
}

EnvironmentManager::Environment::~Environment()
{
  Delete();
}

EnvironmentManager::EnvironmentGuard::EnvironmentGuard(
    EnvironmentManager* manager, Environment* env)
    : manager_(manager), environment_proxy_(env)
{
}

EnvironmentManager::EnvironmentGuard::EnvironmentGuard(
    EnvironmentGuard&& other_guard)
    : manager_(other_guard.manager_),
      environment_proxy_(std::move(other_guard.environment_proxy_))
{
  other_guard.manager_ = nullptr;
}

EnvironmentManager::EnvironmentGuard&
EnvironmentManager::EnvironmentGuard::operator=(EnvironmentGuard&& other_guard)
{
  EnvironmentGuard new_guard(std::move(other_guard));
  std::swap(*this, new_guard);
  return *this;
}

EnvironmentManager::EnvironmentGuard::~EnvironmentGuard()
{
  if (environment_ != nullptr && manager_ != nullptr) {
    manager_->DropEnvironment(*environment_);
  }
}


#endif

}}}  // namespace triton::backend::python
