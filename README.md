<!--
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## User Documentation

* [Quick Start](#quick-start)
* [Usage](#usage)
* [Examples](#examples)
* [Using Custom Python Execution Environments](#using-custom-python-execution-environments)
* [Model Config File](#model-config-file)
* [Error Handling](#error-handling)
* [Managing Shared Memory](#managing-shared-memory)
* [Building From Source](#building-from-source)
* [Business Logic Scripting (beta)](#business-logic-scripting-beta)

## Quick Start

1. Run the Triton Inference Server container.
```
$ docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```
Replace \<xx.yy\> with the Triton version (e.g. 21.05).

2. Inside the container, clone the Python backend repository.

```
$ git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>
```

3. Install example model.
```
$ cd python_backend
$ mkdir -p models/add_sub/1/
$ cp examples/add_sub/model.py models/add_sub/1/model.py
$ cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
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
$ python3 python_backend/examples/add_sub/client.py
```

## Building from Source

1. Requirements

* cmake >= 3.17
* numpy
* rapidjson-dev
* libarchive-dev

```
pip3 install numpy
```

On Ubuntu or Debian you can use the command below to install `rapidjson` and `libarchive`:
```
sudo apt-get install rapidjson-dev libarchive-dev
```

2. Build Python backend. Replace \<GIT\_BRANCH\_NAME\> with the GitHub branch
   that you want to compile. For release branches it should be r\<xx.yy\> (e.g.
   r21.06).

```
$ mkdir build
$ cd build
$ cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME> -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. If the CMake variables below are not specified, "main" branch
of those repositories will be used. \<GIT\_BRANCH\_NAME\> should be the same 
as the Python backend repository branch that you are trying to compile.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=\<GIT\_BRANCH\_NAME\>
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=\<GIT\_BRANCH\_NAME\>
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=\<GIT\_BRANCH\_NAME\>

Set `-DCMAKE_INSTALL_PREFIX` to the location where the Triton Server is installed. In the released containers,
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

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
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

## Using Custom Python Execution Environments

Python backend shipped in the [NVIDIA GPU Cloud](https://ngc.nvidia.com/)
containers uses Python 3.8. If your Python model is compatible with Python 3.8
and requires only modules already included in the Triton container, then you can
skip this section. If you need to use a different version of Python or if you
have additional dependencies, you need to recompile the stub executable and
create an execution environment as described below and include that with your
model.

### 1. Building Custom Python Backend Stub

Python backend uses a *stub* process to connect your `model.py` file to the
Triton C++ core. This stub process has an embedded Python interpreter with
a fixed Python version. If you intend to use a Python interpreter with
different version from the default Python backend stub, you need to compile your own
Python backend stub by following the steps below:

1. Install the software packages below:
* [conda](https://docs.conda.io/en/latest/)
* [cmake](https://cmake.org)
* rapidjson and libarchive (instructions for installing these packages in Ubuntu or Debian are included in [Building from Source Section](#building-from-source))


2. Create and activate a [conda](https://docs.conda.io/en/latest/) environment with your desired Python version. In this example, we will be using Python 3.6:
```bash
conda create -n python-3-6 python=3.6
conda activate python-3-6

# NumPy is required for Python models
conda install numpy
```
3. Clone the Python backend repository and compile the Python backend stub
   (replace \<GIT\_BRANCH\_NAME\> with the branch name that you want to use,
   for release branches it should be r\<xx.yy\>):
```bash
$ git clone https://github.com/triton-inference-server/python_backend -b
<GIT_BRANCH_NAME>
$ cd python_backend
$ mkdir build && cd build
$ cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make triton-python-backend-stub
```

Now, you have access to a Python backend stub with Python 3.6. You can verify
that using `ldd`:

```
$ ldd triton_python_backend_stub
...
libpython3.6m.so.1.0 => /home/ubuntu/envs/miniconda3/envs/python-3-6/lib/libpython3.6m.so.1.0 (0x00007fbb69cf3000)
...
```

There are many other shared libraries printed in addition to the library posted
above. However, it is important to see `libpython3.6m.so.1.0` in the list of
linked shared libraries. If you use a different Python version, you should see
that version instead. You need to copy the `triton_python_backend_stub` to the
model directory of the models that want to use the custom Python backend
stub. For example, if you have `model_a` in your [model repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md), the folder
structure should look like below:

```
models
|-- model_a
    |-- 1
    |   |-- model.py
    |-- config.pbtxt
    `-- triton_python_backend_stub
```

Note the location of `triton_python_backend_stub` in the directory structure above.

### 2. Packaging the Conda Environment

It is also required to create a tar file that contains your conda environment.
Currently, Python backend only supports
[conda-pack](https://conda.github.io/conda-pack/) for this purpose.
[conda-pack](https://conda.github.io/conda-pack/) ensures that your conda
environment is portable. You can create a tar file for your conda environment
using `conda-pack` command:

```
$ conda-pack
Collecting packages...
Packing environment at '/home/iman/miniconda3/envs/python-3-6' to 'python-3-6.tar.gz'
[########################################] | 100% Completed |  4.5s
```

**Important Note:** Before installing the packages in your conda environment, make sure that you
have exported [`PYTHONNOUSERSITE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONNOUSERSITE) environment variable:

```
export PYTHONNOUSERSITE=True
```

If this variable is not exported and similar packages are installed outside your conda environment,
your tar file may not contain all the dependencies required for an isolated Python environment.

After creating the tar file from the conda environment, you need to tell Python
backend to use that environment for your model. You can do this by adding the
lines below to the `config.pbtxt` file:

```
name: "model_a"
backend: "python"

...

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/home/iman/miniconda3/envs/python-3-6/python3.6.tar.gz"}
}
```

### Important Notes

1. The version of the Python interpreter in the execution environment must match
the version of triton_python_backend_stub.

2. If you don't want to use a different Python interpreter, you can skip
[Building Custom Python Backend Stub Step](#1-building-custom-python-backend-stub).
In this case you only need to pack your environment using `conda-pack` and
provide the path to tar file in the model config. However, the previous note
still applies here and the version of the Python interpreter inside the conda
environment must match the Python version of stub used by Python backend. The
default version of the stub is Python 3.8.

3. You can share a single execution environment across multiple models. You need to
provide the path to the tar file in the `EXECUTION_ENV_PATH` in the
`config.pbtxt` of all the models that want to use the execution environment.

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

## Managing Shared Memory

Starting from 21.04 release, Python backend uses shared memory to connect
user's code to Triton. Note that this change is completely transparent and
does not require any change to the existing user's model code.

Python backend, by default, allocates 64 MBs for each model instance. Then,
it will grow the shared memory region by 64 MBs whenever an increase is
required. You can configure the default shared memory used by each model
instance using the `shm-default-byte-size` flag. The amount of shared memory
growth can be configured using the `shm-growth-byte-size`.

You can also configure the timeout used for connecting Triton main process
to the Python backend stubs using the `stub-timeout-seconds`. The default
value is 30 seconds.

The config values described above can be passed to Triton using `--backend-config`
flag:

```
/opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models --backend-config=python,<config-key>=<config-value>
```

Also, if you are running Triton inside a Docker container you need to
properly set the `--shm-size` flag depending on the size of your inputs and
outputs. The default value for docker run command is `64MB` which is very
small.

# Business Logic Scripting (beta)

Triton's
[ensemble](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#ensemble-models)
feature supports many use cases where multiple models are composed into a
pipeline (or more generally a DAG, directed acyclic graph). However, there are
many other use cases that are not supported because as part of the model
pipeline they require loops, conditionals (if-then-else), data-dependent
control-flow and other custom logic to be intermixed with model execution. We
call this combination of custom logic and model executions *Business Logic
Scripting (BLS)*. 

Starting from 21.08, you can implement BLS in your Python model. A new set of
utility functions allows you to execute inference requests on other models being
served by Triton as a part of executing your Python model. Example below shows
how to use this feature:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
  ...
    def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make sure
      # to replace `inputs` argument with a list of `pb_utils.Tensor` objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      # `pb_utils.InferenceRequest` supports request_id, correlation_id, and model
      # version in addition to the arguments described above. These arguments
      # are optional. An example containing all the arguments:
      # inference_request = pb_utils.InferenceRequest(model_name='model_name',
      #   requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
      #   inputs=[<list of pb_utils.Tensor objects>],
      #   request_id="1", correlation_id=4, model_version=1)

      # Execute the inference_request and wait for the response
      inference_response = inference_request.exec()

      # Check if the inference response has an error
      if inference_response.has_error():
          raise pb_utils.TritonModelException(inference_response.error().message())
      else:
          # Extract the output tensors from the inference response.
          output1 = pb_utils.get_output_tensor_by_name(inference_response, 'REQUESTED_OUTPUT_1')
          output2 = pb_utils.get_output_tensor_by_name(inference_response, 'REQUESTED_OUTPUT_2')

          # Decide the next steps for model execution based on the received output
          # tensors. It is possible to use the same output tensors to for the final
          # inference resposne too.
```

A complete example for BLS in Python backend is included in the
[Examples](#examples) section.

## Limitations

- The number of inference requests that can be executed as a part of your model
execution is limited to the amount of shared memory available to the Triton
server.  If you are using Docker to start the TritonServer, you can control the
shared memory usage using the
[`--shm-size`](https://docs.docker.com/engine/reference/run/) flag.
- You need to make sure that the inference requests performed as a part of your model
do not create a circular dependency. For example, if model A performs an inference request
on itself and there are no more model instances ready to execute the inference request, the
model will block on the inference execution forever.

# Examples

For using the Triton Python client in these examples you need to install
the [Triton Python Client Library](https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples).
The Python client for each of the examples is in the `client.py` file.

## AddSub in Numpy

There is no dependencies required for the AddSub numpy example. Instructions
on how to use this model is explained in the quick start section. You can
find the files in [examples/add_sub](examples/add_sub).

## AddSubNet in PyTorch

In order to use this model, you need to install PyTorch. We recommend using
`pip` method mentioned in the [PyTorch website](https://pytorch.org/get-started/locally/).
Make sure that PyTorch is available in the same Python environment as other
dependencies. Alternatively, you can create a [Python Execution Environment](#using-custom-python-execution-environments).
You can find the files for this example in [examples/pytorch](examples/pytorch).

## Business Logic Scripting

The BLS example needs the dependencies required for both of the above examples.
You can find the complete example instructions in [examples/bls](examples/bls/README.md).

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
