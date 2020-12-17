<!--
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Python Backend

The Triton backend for Python. The goal of Python backend is to let you serve
models written in Python by Triton Inference Server without having to write
any C++ code.

## Quick Start

1. Run the Triton Inference Server container.
```
$ docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```
Replace <xx.yy> with the Triton version (e.g. 20.11).

2. Inside the container, clone the Python backend repository.

```
$ git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>
```

3. Install example model.
```
$ cd python_backend
$ mkdir -p models/add_sub/1/
$ cp examples/add_sub.py models/add_sub/1/model.py
$ cp examples/config.pbtxt models/add_sub/config.pbtxt
```

4. Start the Triton server.

```
$ tritonserver --model-repository `pwd`/models
```

5. In the host machine, start the client container.

```
 docker run -ti --net host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk /bin/bash
```

6. In the client container, clone the Python backend repository.

```
$ git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>
```

7. Run the example client.
```
$ python3 python_backend/examples/add_sub_client.py
```

## Building from Source

1. Requirements

* cmake >= 3.17
* numpy
* grpcio-tools
* grpcio-channelz
* rapidjson-dev

```
pip3 install grpcio-tools grpcio-channelz numpy
```

On Ubuntu or Debian you can use the command below to install `rapidjson`:
```
sudo apt-get install rapidjson-dev
```

2. Build Python backend

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

Set `DCMAKE_INSTALL_PREFIX` to the location where the Triton Server is installed. In the released containers,
this location is `/opt/tritonserver`.

3. Copy example model and configuration

```
$ mkdir -p models/add_sub/1/
$ cp examples/add_sub/model.py models/add_sub/1/model.py
$ cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
```

4. Start the Triton Server

```
$ /opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models
```

5. Use the client app to perform inference

```
$ python3 examples/add_sub/client.py
```

## Usage

In order to use the Python backend, you need to create a Python file that
has a structure similar to below:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate every request and create a
        # pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Perform inference on the request and append it to responses list...

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

```

Every Python backend can implement three main functions:

### `initialize`

`initialize` is called once the model is being loaded. Implementing
`initialize` is optional. `initialize` allows you to do any necessary
initializations before execution. In the `initialize` function, you are given
an `args` variable. `args` is a Python dictionary. Both keys and
values for this Python dictionary are strings. You can find the available
keys in the `args` dictionary along with their description in the table
below:

| key                      | description                                      |
| ------------------------ | ------------------------------------------------ |
| model_config             | A JSON string containing the model configuration |
| model_instance_kind      | A string containing model instance kind          |
| model_instance_device_id | A string containing model instance device ID     |
| model_repository         | Model repository path                            |
| model_version            | Model version                                    |
| model_name               | Model name                                       |

### `execute`

`execute` function is called whenever an inference request is made. Every Python
model must implement `execute` function. In the `execute` function you are given
a list of `InferenceRequest` objects. In this function, your `execute` function
must return a list of `InferenceResponse` objects that has the same length as 
`requests`.

In case one of the inputs has an error, you can use the `TritonError` object
to set the error message for that specific request. Below is an example of
setting errors for an `InferenceResponse` object:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            if an_error_occurred:
              # If there is an error, the output_tensors are ignored
              responses.append(pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError("An Error Occurred")))

        return responses
```

### `finalize`

Implementing `finalize` is optional. This function allows you to do any clean
ups necessary before the model is unloaded from Triton server.

You can look at the [add_sub example](examples/add_sub.py) which contains
a complete example of implementing all these functions for a Python model
that adds and subtracts the inputs given to it. After implementing all the
necessary functions, you should save this file as `model.py`.

## Model Config File

Every Python Triton model must provide a `config.pbtxt` file describing 
the model configuration. In order to use this backend you must set the `backend`
field of your model `config.pbtxt` file to `python`. You shouldn't set
`platform` field of the configuration.

Your models directory should look like below:
```
models
└── add_sub
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## Changing Python Runtime Path

Python backend by default uses `python3` available inside `PATH`. In order to change
the Python runtime used by Python backend, you can use the `--backend-config` flag:

```
/opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models --backend-config=python,python-runtime=<full path to custom Python location>
```

You will also need to install all the dependencies mentioned in the "Install
from Source" section in the new environment too. 

## Changing GRPC Timeout

Python backend uses gRPC to connect Triton to the Python model. You can change the 
timeout using the backend config:

```
/opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models --backend-config=python,grpc-timeout-milliseconds=3000
```

The default timeout value is 2000 milliseconds.

## Error Handling

If there is an error that affects the `initialize`, `execute`, or `finalize`
function of the Python model you can use `TritonInferenceException`. 
Example below shows how you can do error handling in `finalize`:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def finalize(self):
      if error_during_finalize:
        raise pb_utils.TritonModelException("An error occurred during finalize.")
```

# Examples

For using the Triton Python client in these examples you need to install
the [Triton Python Client Library](https://github.com/triton-inference-server/server/blob/master/docs/client_libraries.md#getting-the-client-libraries).
The Python client for each of the examples is in the `client.py` file.

## AddSub in Numpy

There is no dependencies required for the AddSub numpy example. Instructions
on how to use this model is explained in the quick start section. You can
find the files in [examples/add_sub](examples/add_sub).

## AddSubNet in PyTorch

In order to use this model, you need to install PyTorch. We recommend using
`pip` method mentioned in the [PyTorch
website](https://pytorch.org/get-started/locally/). Make sure that PyTorch is
available in the same Python environment as other dependencies. If you need
to create another Python environment, please refer to the "Changing Python
Runtime Path" section of this readme. You can find the files for this example
in [examples/pytorch](examples/pytorch).

# Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
