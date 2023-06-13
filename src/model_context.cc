// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "model_context.h"

#include <iostream>
#include "pb_exception.h"

namespace triton { namespace backend { namespace python {

void
ModelContext::Init(
    const std::string& model_path, const std::string& platform_model,
    const std::string& triton_install_path, const std::string& model_version)
{
  if (platform_model.empty()) {
    type_ = ModelType::DEFAULT;
    python_model_path_ = model_path;
  } else {
    type_ = ModelType::PLATFORM;
    fw_model_path_ = model_path;
    python_model_path_ =
        triton_install_path + "/platform_models/" + platform_model + "/model.py";
    // Check if model file exists in the path
    struct stat buffer;
    if (stat(python_model_path_.c_str(), &buffer) != 0) {
      throw PythonBackendException(
          ("[INTERNAL] " + python_model_path_ +
           " model file does not exist as platform model"));
    }
  }
  python_backend_folder_ = triton_install_path;
  model_version_ = model_version;
}

void
ModelContext::StubSetup(py::module* sys)
{
  std::string model_name =
      python_model_path_.substr(python_model_path_.find_last_of("/") + 1);

  // Model name without the .py extension
  auto dotpy_pos = model_name.find_last_of(".py");
  if (dotpy_pos == std::string::npos || dotpy_pos != model_name.size() - 1) {
    throw PythonBackendException(
        "Model name must end with '.py'. Model name is \"" + model_name +
        "\".");
  }
  // The position of last character of the string that is searched for is
  // returned by 'find_last_of'. Need to manually adjust the position.
  std::string model_name_trimmed = model_name.substr(0, dotpy_pos - 2);

  if (type_ == ModelType::DEFAULT) {
    std::string model_path_parent =
        python_model_path_.substr(0, python_model_path_.find_last_of("/"));
    std::string model_path_parent_parent =
        model_path_parent.substr(0, model_path_parent.find_last_of("/"));
    sys->attr("path").attr("append")(model_path_parent);
    sys->attr("path").attr("append")(model_path_parent_parent);
    sys->attr("path").attr("append")(python_backend_folder_);
    *sys = py::module_::import(
        (std::string(model_version_) + "." + model_name_trimmed).c_str());
  } else {
    // [FIXME] Improve the path generation logic to make it more flexible.
    std::string platform_model_dir(
        python_backend_folder_ + "/platform_models/tensorflow_savedmodel/");
    sys->attr("path").attr("append")(platform_model_dir);
    sys->attr("path").attr("append")(python_backend_folder_);
    *sys = py::module_::import(model_name_trimmed.c_str());
  }
}

}}}  // namespace triton::backend::python
