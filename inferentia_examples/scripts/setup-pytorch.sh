set -x 
export INFRENTIA_PATH=/home/ubuntu
export CONDA_PATH=${INFRENTIA_PATH}/miniconda
# Get latest conda required https://repo.anaconda.com/miniconda/
cd ${INFRENTIA_PATH}
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
         -O ${PWD}/miniconda.sh --no-check-certificate && \
    /bin/bash ${PWD}/miniconda.sh -b -p ${CONDA_PATH} && \
    rm ${PWD}/miniconda.sh && \
    $(eval echo "${CONDA_PATH}/bin/conda clean -ya")
export PATH=${CONDA_PATH}/bin:${PATH}
conda info
source ~/.bashrc

# install python_backend_stub installing dependencies
apt-get update && \
    apt-get install -y --no-install-recommends \
              zlib1g-dev \
              wget \
              python3.7        \
              python3-pip      \
              libarchive-dev   \
              rapidjson-dev

# CMake
export CMAKE_UBUNTU_VERSION=20.04
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    if [ "$CMAKE_UBUNTU_VERSION" = "20.04" ]; then \
      apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1; \
    elif [ "$CMAKE_UBUNTU_VERSION" = "18.04" ]; then \
      apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        cmake-data=3.18.4-0kitware1 cmake=3.18.4-0kitware1; \
    else \
      echo "ERROR: Only support CMAKE_UBUNTU_VERSION to be 18.04 or 20.04" && false; \
    fi && \
    cmake --version

conda create -q -y -n test_conda_env python=3.7
source ${CONDA_PATH}/bin/activate test_conda_env

# First compile correct python stub
export PYTHON_BACKEND_BRANCH_NAME="main"
git clone https://github.com/triton-inference-server/python_backend -b $PYTHON_BACKEND_BRANCH_NAME
cd python_backend
mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=OFF -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
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
# need to use pip to update: https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade torch-neuron torchvision "transformers==4.6.0"

cp /mylib/udev/rules.d/* /lib/udev/rules.d/
ln -s /myrun/neuron.sock /run/neuron.sock
# Upgrade the python backend stub
cp ${INFRENTIA_PATH}/python_backend/build/triton_python_backend_stub /opt/tritonserver/backends/python/triton_python_backend_stub
export LD_LIBRARY_PATH=${CONDA_PATH}/envs/test_conda_env/lib:$LD_LIBRARY_PATH
set +x 