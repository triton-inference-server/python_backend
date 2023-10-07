# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonpythonbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_NVTX "Include nvtx markers collection in backend." OFF)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

# We need to use ExternalProject because we want to use Boost 1.76 which is not
# available in Ubuntu 20.04
include(ExternalProject)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11"
  # COMMIT ID for v2.10.0
  GIT_TAG "aa304c9c7d725ffb9d10af08a3b34cb372307020"
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pybind11)

#
# DLPack
#
FetchContent_Declare(
  dlpack
  GIT_REPOSITORY "https://github.com/dmlc/dlpack"
  GIT_TAG "v0.8"
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(dlpack)

#
# Boost
#
ExternalProject_Add(
  boostorg
  URL https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
  URL_HASH SHA256=273f1be93238a068aba4f9735a4a2b003019af067b9c183ed227780b8f36062c
  PREFIX "boost-src"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy_directory
                    <SOURCE_DIR>/boost/ ${CMAKE_BINARY_DIR}/boost
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
set(boostorg_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/boost/")

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)
elseif()
  message(WARNING "TRITON_ENABLE_GPU is OFF, GPU Tensor support will be disabled")
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CMAKE_BINARY_DIR})
configure_file(src/libtriton_python.ldscript libtriton_python.ldscript COPYONLY)

set(
  COMMON_SRCS
  src/infer_response.cc
  src/infer_response.h
  src/infer_request.cc
  src/infer_request.h
  src/message_queue.h
  src/ipc_message.cc
  src/ipc_message.h
  src/pb_string.cc
  src/pb_string.h
  src/pb_map.cc
  src/pb_map.h
  src/scoped_defer.cc
  src/scoped_defer.h
  src/pb_error.cc
  src/pb_error.h
  src/pb_log.cc
  src/pb_log.h
  src/pb_memory.cc
  src/pb_memory.h
  src/pb_tensor.cc
  src/pb_tensor.h
  src/pb_utils.cc
  src/pb_utils.h
  src/shm_manager.cc
  src/shm_manager.h
  src/pb_exception.h
  src/pb_preferred_memory.h
  src/metric.h
  src/metric.cc
  src/metric_family.h
  src/metric_family.cc
  src/gpu_buffers.cc
  src/gpu_buffers.h
  src/model_loader.h
  src/model_loader.cc
)

set(
  PYTHON_BACKEND_SRCS
  src/python_be.cc
  src/python_be.h
  src/pb_env.cc
  src/pb_env.h
  src/pb_metric_reporter.cc
  src/pb_metric_reporter.h
  src/memory_manager.cc
  src/memory_manager.h
  src/request_executor.cc
  src/request_executor.h
  src/stub_launcher.h
  src/stub_launcher.cc
  src/infer_payload.h
  src/infer_payload.cc
)

list(APPEND
  PYTHON_BACKEND_SRCS
  ${COMMON_SRCS}
)

add_library(
  triton-python-backend SHARED
  ${PYTHON_BACKEND_SRCS}
)

set(
  PYTHON_BACKEND_STUB_SRCS
  src/pb_stub_utils.h
  src/pb_stub_utils.cc
  src/response_sender.cc
  src/response_sender.h
  src/pb_stub.h
  src/pb_stub.cc
  src/pb_response_iterator.h
  src/pb_response_iterator.cc
  src/pb_cancel.cc
  src/pb_cancel.h
)

list(APPEND
  PYTHON_BACKEND_STUB_SRCS
  ${COMMON_SRCS}
)

add_executable(
  triton-python-backend-stub
  ${PYTHON_BACKEND_STUB_SRCS}
)

add_dependencies(triton-python-backend boostorg)
add_dependencies(triton-python-backend-stub boostorg)

set_property(TARGET triton-python-backend-stub PROPERTY OUTPUT_NAME triton_python_backend_stub)

add_library(
  TritonPythonBackend::triton-python-backend ALIAS triton-python-backend
)

target_compile_features(triton-python-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-python-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_compile_features(triton-python-backend-stub PRIVATE cxx_std_11)
target_compile_options(
  triton-python-backend-stub PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)
target_compile_definitions(triton-python-backend-stub PRIVATE TRITON_PB_STUB)

target_link_libraries(
  triton-python-backend
  PRIVATE
    dlpack
    Threads::Threads
    triton-backend-utils          # from repo-backend
    ${CMAKE_DL_LIBS}              # dlopen and dlclose
    -lrt                          # shared memory
    triton-core-serverstub        # from repo-core
    ZLIB::ZLIB
    -larchive
)

target_link_libraries(
  triton-python-backend-stub
  PRIVATE
   dlpack
   Threads::Threads
   triton-backend-utils           # from repo-backend
   ${CMAKE_DL_LIBS}               # dlopen and dlclose
   pybind11::embed
   -lrt                           # shared memory
   -larchive                      # libarchive
)

set_target_properties(
  triton-python-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_python
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_python.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_python.ldscript"
)

add_subdirectory(./src/shm_monitor)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonPythonBackend)

install(
  TARGETS
    triton-python-backend
    triton-python-backend-stub
  EXPORT
    triton-python-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
)

install(
  EXPORT
    triton-python-backend-targets
  FILE
    TritonPythonBackendTargets.cmake
  NAMESPACE
    TritonPythonBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

install(
  FILES
    src/resources/triton_python_backend_utils.py
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/backends/python
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonPythonBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-python-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendTargets.cmake
  NAMESPACE TritonPythonBackend::
)

export(PACKAGE TritonPythonBackend)
