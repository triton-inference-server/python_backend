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

Ensure that the neuron runtime 1.0 demon (neuron-rtd) is not running and set up
and install neuron 2.X runtime builds with
```
 sudo ./python_backend/setup-pre-container.sh
```

Then, start the Triton instance with:
``` 
docker run --device /dev/neuron0 <more neuron devices> -v /home/ubuntu/python_backend:/home/ubuntu/python_backend -v /lib/udev:/mylib/udev --shm-size=1g -e "AWS_NEURON_VISIBLE_DEVICES=ALL" --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```
Note 1: The user would need to list any neuron device to run during container initialization.
For example, to use 4 neuron devices on an instance, the user would need to run with:
```
docker run --device /dev/neuron0 --device /dev/neuron1 --device /dev/neuron2 --device /dev/neuron3 ...`
```
Note 2: `/mylib/udev` is used for Neuron parameter passing. 

Note 3: For Triton container version xx.yy, please refer to 
[Triton Inference Server Container Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html).
 The current build script has been tested with container version `21.10`. 

After starting the Triton container, go into the `python_backend` folder and run the setup script.
```
source /home/ubuntu/python_backend/inferentia/scripts/setup .sh
```
This script will:
1. Setup miniconda enviroment
2. Install necessary dependencies
3. Create a [Custom Python Execution Environment](https://github.com/triton-inference-server/python_backend#using-custom-python-execution-environments), 
   `python_backend_stub` to use for Inferentia
4. Install [neuron-cc](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-cc/index.html),
    the Neuron compiler and [neuron-rtd](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/overview.html) the Neuron Runtime

There are user configurable options available for the script as well. 
For example, to control the python version for the python environment to 3.6, 
you can run:
```
source /home/ubuntu/python_backend/inferentia/scripts/setup.sh -v 3.6
```
Please use the `-h` or `--help` options to learn about more configurable options.

## Setting up the Inferentia model

Currently, we only support TorchScript models traced by [PyTorch-Neuron trace python API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.html) for execution on Inferentia.
Once the TorchScript model supporting Inferentia is obtained, use the [gen_triton_model.py](https://github.com/triton-inference-server/python_backend/blob/main/inferentia/scripts/gen_triton_model.py) script to generate triton python model directory.

An example invocation for the `gen_triton_model.py` can look like:

```
 $python3 inferentia/scripts/gen_triton_model.py --triton_input INPUT__0,INT64,4x384 INPUT__1,INT64,4x384 INPUT__2,INT64,4x384 --triton_output OUTPUT__0,INT64,4x384 OUTPUT__1,INT64,4x384 --compiled_model /home/ubuntu/bert_large_mlperf_neuron_hack_bs1_dynamic.pt --neuron_core_range 0:3 --triton_model_dir bert-large-mlperf-bs1x4
```

NOTE: Due to the absence of names for inputs and outputs in a
TorchScript model, the name of tensor of both the inputs and
outputs provided to the above script must follow a specific naming
convention i.e. `<name>__<index>`. Where `<name>` can be any
string and `<index>` refers to the position of the corresponding
input/output. This means if there are two inputs and two outputs
they must be named as: "INPUT__0", "INPUT__1" and "OUTPUT__0",
"OUTPUT__1" such that "INPUT__0" refers to first input and
INPUT__1 refers to the second input, etc.

Additionally, `--neuron-core-range` specifies the neuron cores to
be used while serving this models. Currently, only
`torch.neuron.DataParallel()` mode is supported. See
[Data Parallel Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/appnotes/perf/torch-neuron-dataparallel-app-note.html)
for more information.

The invocation should create a triton model directory with following
structutre:

```
bert-large-mlperf-bs1x4
 |
 |- 1
 |  |- model.py
 |
 |- config.pbtxt
```

Look at the usage message of the script to understand each option.

The script will generate a model directory with the user-provided
name. Move that model directory to Triton's model repository.
Ensure the compiled model path provided to the script points to
a valid torchscript file.

Now, the server can be launched with the model as below:

```
tritonserver --model-repository <path_to_model_repository>
```

Note: 
1. The `config.pbtxt` and `model.py` should be treated as
starting point. The users can customize these files as per
their need.
2. Triton Inferentia currently only works with **single** model. 