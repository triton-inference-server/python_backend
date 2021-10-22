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

USAGE="
usage: test_script.sh [options]

Sets up enviroment for neuron-pytorch
-h|--help                  Shows usage
-p|--python-backend-path   Python backend path, default is: /home/ubuntu/python_backend
-v|--python-version        Python version, default is 3.7
-i|--inferentia-path       Inferentia path, default is: /home/ubuntu
"

# Get all options:
OPTS=$(getopt -o hp:v:i: --long help,python-backend-path:,python-version:,inferentia-path: -- "$@")


export INFRENTIA_PATH="/home/ubuntu"
export PYTHON_BACKEND_PATH="/home/ubuntu/python_backend"
export PYTHON_VERSION=3.7

for OPTS; do
    case "$OPTS" in
        -h|--help)
        printf "%s\\n" "$USAGE"
        exit 2
        ;;
        -p|--python-backend-path)
        PYTHON_BACKEND_PATH=$2
        echo "Python backend path set to ${PYTHON_BACKEND_PATH}"
        shift 2
        ;;
        -v|--python-version)
        PYTHON_VERSION=$2
        shift 2
        echo "Python version set to ${PYTHON_VERSION}"
        ;;
        -i|--inferentia-path)
        INFRENTIA_PATH=$2
        echo "Inferentia path set to ${INFRENTIA_PATH}"
        shift 2
        ;;
    esac
done
# Get latest conda required https://repo.anaconda.com/miniconda/
cd ${INFRENTIA_PATH}
export CONDA_PATH=${INFRENTIA_PATH}/miniconda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
         -O ${PWD}/miniconda.sh --no-check-certificate && \
    /bin/bash ${PWD}/miniconda.sh -b -p ${CONDA_PATH} && \
    rm ${PWD}/miniconda.sh && \
    $(eval echo "${CONDA_PATH}/bin/conda clean -ya")
export PATH=${CONDA_PATH}/bin:${PATH}
conda info

# Install python_backend_stub installing dependencies
apt-get update && \
    apt-get install -y --no-install-recommends \
              zlib1g-dev \
              wget \
              python3-pip      \
              libarchive-dev   \
              rapidjson-dev

# CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
apt-get update && \
apt-get install -y --no-install-recommends \
cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1 && \
cmake --version

# Create Conda Enviroment
conda create -q -y -n test_conda_env python=${PYTHON_VERSION}
source ${CONDA_PATH}/bin/activate test_conda_env

# First compile correct python stub
cd ${PYTHON_BACKEND_PATH}
rm -rf build && mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make triton-python-backend-stub -j16

# Install Neuron Pytorch
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/install-pytorch.html
cd ${INFRENTIA_PATH}
. /etc/os-release
tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB |  apt-key add -
apt-get update && \ 
    apt-get install -y       \
        linux-headers-$(uname -r) \
        aws-neuron-dkms           \
        aws-neuron-runtime-base   \
        aws-neuron-runtime        \
        aws-neuron-tools 
export PATH=/opt/aws/neuron/bin:$PATH

# Install Neuron-pytorch (neuron-cc)
conda config --env --add channels https://conda.repos.neuron.amazonaws.com
conda install torch-neuron -y

# Upgrade torch-neuron and install transformers
# need to use pip to update: 
# https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade torch-neuron torchvision "transformers==4.6.0"

# Upgrade the python backend stub, rules and sockets
cp ${INFRENTIA_PATH}/python_backend/build/triton_python_backend_stub \
        /opt/tritonserver/backends/python/triton_python_backend_stub
cp /mylib/udev/rules.d/* /lib/udev/rules.d/
ln -s /myrun/neuron.sock /run/neuron.sock
export LD_LIBRARY_PATH=${CONDA_PATH}/envs/test_conda_env/lib:$LD_LIBRARY_PATH
