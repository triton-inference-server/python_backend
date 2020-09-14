# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# GPU support is disabled by default because python backend doesn't
# because python backend does not need to access CUDA or GPUs
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY git@github.com:triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY git@github.com:triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-backend)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG "release-1.10.0"
  GIT_SHALLOW ON
)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG "v1.31.1"
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(googletest grpc)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

get_target_property(Protobuf_INCLUDE_DIRS ${_PROTOBUF_LIBPROTOBUF} INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(gRPC_INCLUDE_DIRS ${_GRPC_GRPCPP} INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${gRPC_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_filename_component(python_host_proto_abspath "src/python_host.proto" ABSOLUTE)
get_filename_component(python_host_proto_dir "${python_host_proto_abspath}" PATH)

set(GRPC_SRCS
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.pb.cc")

set(GRPC_HDRS
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.h"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.pb.h")

set(GRPC_PY
    "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2_grpc.py"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2.py")

add_custom_command(
  OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
    --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${python_host_proto_dir}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${python_host_proto_abspath}"
  DEPENDS "${python_host_proto_abspath}" ${_PROTOBUF_LIBPROTOBUF}
)

find_package(Python REQUIRED COMPONENTS Interpreter)   
add_custom_command(
  OUTPUT ${GRPC_PY}
  COMMAND ${Python_EXECUTABLE}
  ARGS
    -m grpc_tools.protoc
    -I "${python_host_proto_dir}"
    --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
    --python_out "${CMAKE_CURRENT_BINARY_DIR}"
    "${python_host_proto_abspath}"
  DEPENDS "${python_host_proto_abspath}" ${_PROTOBUF_LIBPROTOBUF}
)

add_custom_target(python-grpc-py-library ALL
                  DEPENDS ${GRPC_PY})

add_library(
  python-grpc-library EXCLUDE_FROM_ALL OBJECT
  ${GRPC_SRCS} ${GRPC_HDRS} 
)

configure_file(src/libtriton_python.ldscript libtriton_python.ldscript COPYONLY)

add_library(
  triton-python-backend SHARED
  src/python.cc

  $<TARGET_OBJECTS:python-grpc-library>
)


add_library(
  TritonPythonBackend::triton-python-backend ALIAS triton-python-backend
)

target_include_directories(
  triton-python-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-python-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-python-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror>
)

target_link_libraries(
  triton-python-backend
  PRIVATE
    triton-backend-utils  # from repo-backend
    ${_GRPC_GRPCPP}
)

set_target_properties(
  triton-python-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_python
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_python.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_python.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonPythonBackend)

install(
  TARGETS
    triton-python-backend
  EXPORT
    triton-python-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
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
    ${GRPC_PY}
    src/resources/startup.py
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

install(
  FILES
    src/resources/triton_python_backend_utils.py
  DESTINATION
    ${INSTALL_CONFIGDIR}
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
