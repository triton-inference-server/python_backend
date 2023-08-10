<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# PyTorch Example

In this section, we demonstrate an end-to-end example for using the
[PyTorch Platform \[Experimental\]](../../README.md#pytorch-platform-experimental)
to serve a PyTorch model directly, **without** the need to implement the
`TritonPythonModel` class.

## Create a ResNet50 model repository

We will use the files that come with this example to create the model
repository.

First, download [client.py](client.py), [config.pbtxt](config.pbtxt),
[model.py](model.py),
[mug.jpg](https://raw.githubusercontent.com/triton-inference-server/server/main/qa/images/mug.jpg)
and [resnet50_labels.txt](resnet50_labels.txt) to your local machine.

Next, at the directory where the downloaded files are saved at, create a model
repository with the following commands:
```
$ mkdir -p models/resnet50_pytorch/1
$ mv model.py models/resnet50_pytorch/1
$ mv config.pbtxt models/resnet50_pytorch
```

## Pull the Triton Docker images

We need to install Docker and NVIDIA Container Toolkit before proceeding, refer
to the
[installation steps](https://github.com/triton-inference-server/server/tree/main/docs#installation).

To pull the latest containers, run the following commands:
```
$ docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3
$ docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk
```
See the installation steps above for the `<yy.mm>` version.

For example, if the version is `23.08`, then:
```
$ docker pull nvcr.io/nvidia/tritonserver:23.08-py3
$ docker pull nvcr.io/nvidia/tritonserver:23.08-py3-sdk
```

Be sure to replace the `<yy.mm>` with the version pulled for all the remaining
parts of this example.

## Start the Triton Server

At the directory where we created the PyTorch model (at where the "models"
folder is located), run the following command:
```
$ docker run -it --rm --gpus all --shm-size 1g -p 8000:8000 -v `pwd`:/pytorch_example nvcr.io/nvidia/tritonserver:<yy.mm>-py3 /bin/bash
```

Inside the container, we need to install PyTorch, Pillow and Requests to run this example.
We recommend using `pip` method for the installations, for example:
```
$ pip3 install torch Pillow requests
```

Finally, we need to start the Triton Server, run the following command:
```
$ tritonserver --model-repository=/pytorch_example/models
```

To leave the container for the next step, press: `CTRL + P + Q`.

## Test inference

At the directory where the client.py is located, run the following command:
```
$ docker run --rm --net=host -v `pwd`:/pytorch_example nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk python3 /pytorch_example/client.py
```

A successful inference will print the following at the end:
```
Result: COFFEE MUG
Expected result: COFFEE MUG
PASS: PyTorch platform handler
```
