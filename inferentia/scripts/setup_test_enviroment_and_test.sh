#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export TRITON_PATH="/home/ubuntu"
cd ${TRITON_PATH}

export PYTHON_BACKEND_BRANCH_NAME:=${PYTHON_BACKEND_BRANCH_NAME:="main"}
export TRITON_SERVER_BRANCH_NAME:=${TRITON_SERVER_BRANCH_NAME:="main"}
export TRITON_CLIENT_REPO_TAG:=${TRITON_CLIENT_REPO_TAG:="main"}
export TRITON_CONTAINER_VERSION="21.10"
export BASE_IMAGE=tritonserver
export SDK_IMAGE=tritonserver_sdk
export BUILD_IMAGE=tritonserver_build
export QA_IMAGE=tritonserver_qa


export TEST_JSON_REPO=${INFERENTIA_PATH}/server/qa/common/inferentia_perf_analyzer_input_data_json
export TEST_REPO=${INFERENTIA_PATH}/server/qa/L0_inferentia_perf_analyzer
export TEST_SCRIPT="test.sh"

# Clone necessary branches
git clone --single-branch --depth=1 -b ${TRITON_SERVER_BRANCH_NAME}
          https://github.com/triton-inference-server/python_backend.git
git clone --single-branch --depth=1 -b ${TRITON_SERVER_BRANCH_NAME}
          https://github.com/triton-inference-server/server.git
echo ${TRITON_VERSION} > server/TRITON_VERSION
cd ${INFERNTIA_PATH}/server
git clone --single-branch --depth=1 -b ${TRITON_CLIENT_REPO_TAG}
          https://github.com/triton-inference-server/client.git clientrepo

# First set up inferentia and run in detatched mode
cd ${TRITON_PATH}/python_backend
chmod 777 ${TRITON_PATH}/python_backend/inferentia/scripts/setup-pre-container.sh
sudo ${TRITON_PATH}/python_backend/inferentia/scripts/setup-pre-container.sh

# Build container with only python backend 
cd ${TRITON_PATH}
./build.py --build-dir=/tmp/tritonbuild \
           --cmake-dir=${TRITON_PATH}/server/build \
           --enable-logging --enable-stats --enable-tracing \
           --enable-metrics --enable-gpu-metrics --enable-gpu \
           --filesystem=gcs --filesystem=azure_storage --filesystem=s3 \
           --endpoint=http --endpoint=grpc \
           --image=base,nvcr.io/nvidia/tritonserver:${TRITON_CONTAINER_VERSION}-py3 \
           --repo-tag=common:main \ 
           --repo-tag=core:main \
           --repo-tag=backend:main \
           --repo-tag=thirdparty:main \
           --backend=identity:main \
           --backend=python:main \
           --repoagent=checksum:main
docker tag tritonserver_build "${BUILD_IMAGE}"
docker tag tritonserver "${BASE_IMAGE}"

# Build docker container for SDK
docker build -t ${SDK_IMAGE} \
             -f ${TRITON_PATH}/Dockerfile.sdk \
             --build-arg "TRITON_CLIENT_REPO_SUBDIR=clientrepo" .

# Build QA container
docker build -t ${QA_IMAGE}
                   -f ${TRITON_PATH}/Dockerfile.QA 
                   --build-arg "TRITON_PATH=${TRITON_PATH}"
                   --build-arg "SDK_IMAGE=${SDK_IMAGE}"
                   --build-arg "BASE_IMAGE=${BASE_IMAGE}" .

# log into the docker
docker run -v /home/ubuntu:$TRITON_PATH \
           -e TEST_REPO=${TEST_REPO} \
           -e TEST_JSON_REPO=${TEST_JSON_REPO} \
           --net host -ti ${QA_IMAGE}  \
           /bin/bash -c "bash -ex ${TEST_REPO}/${TEST_SCRIPT}"


