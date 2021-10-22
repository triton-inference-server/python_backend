<!--
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Using Triton with Inferentia

Starting from 21.11 release, Triton supports
[AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) 
and the [Neuron Runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html).

## Inferentia setup

First step of running Triton with Inferentia is to create an AWS Inferentia
 instance with Deep Learning AMI (tested with Ubuntu 18.04).
`ssh -i <private-key-name>.pem ubuntu@<instance address>`
Note: It is recommended to set your storage space to greater than default value 
of 110 GiB. The current version of Triton has been tested
with storage of 500 GiB.

After logging into the inf1* instance, you will need to clone
[this current Github repo](https://github.com/triton-inference-server/python_backend). 
 Follow [steps on Github to set up ssh access](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) 
or simply clone with https.
Clone this repo with Github to home repo `/home/ubuntu`.

Then, start the Triton instance with:
``` 
docker run -v /home/ubuntu/python_backend:/home/ubuntu/python_backend -v /lib/udev:/mylib/udev -v /run:/myrun --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```

Where `/mylib/udev` and `/myrun` are used for Neuron parameter passing. 
For Triton container version xx.yy, please refer to 
[Triton Inference Server Container Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html).
 The current build script has been tested with container version `21.09`. 

After starting the Triton container, go into the `python_backend` folder and run the setup script.
```
source /home/ubuntu/python_backend/inferentia/scripts/setup-pytorch.sh
```
This script will:
1. Setup miniconda enviroment
2. Install necessary dependencies
3. Create a [Custom Python Execution Environment](https://github.com/triton-inference-server/python_backend#using-custom-python-execution-environments), 
   `python_backend_stub` to use for Inferentia
5. Install [neuron-cc](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-cc/index.html),
    the Neuron compiler and [neuron-rtd](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/overview.html) the Neuron Runtime

There are user configurable options available for the script as well. 
For example, to control the python version for the python environment to 3.6, 
you can run:
```
source /home/ubuntu/python_backend/inferentia/scripts/setup-pytorch.sh -v 3.6
```
Please use the `-h` or `--help` options to learn about more configurable options. 
