<!--
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

- [Python Backend](#python-backend)
  - [User Documentation](#user-documentation)
  - [Quick Start](#quick-start)
  - [Building from Source](#building-from-source)
  - [Usage](#usage)
    - [`auto_complete_config`](#auto_complete_config)
    - [`initialize`](#initialize)
    - [`execute`](#execute)
      - [Default Mode](#default-mode)
      - [Decoupled mode](#decoupled-mode)
        - [Use Cases](#use-cases)
        - [Known Issues](#known-issues)
    - [`finalize`](#finalize)
  - [Model Config File](#model-config-file)
  - [Using Custom Python Execution Environments](#using-custom-python-execution-environments)
    - [1. Building Custom Python Backend Stub](#1-building-custom-python-backend-stub)
    - [2. Packaging the Conda Environment](#2-packaging-the-conda-environment)
    - [Important Notes](#important-notes)
  - [Error Handling](#error-handling)
  - [Managing Shared Memory](#managing-shared-memory)
  - [Multiple Model Instance Support](#multiple-model-instance-support)
- [Business Logic Scripting](#business-logic-scripting)
  - [Using BLS with Stateful Models](#using-bls-with-stateful-models)
  - [Limitation](#limitation)
- [Interoperability and GPU Support](#interoperability-and-gpu-support)
  - [`pb_utils.Tensor.to_dlpack() -> PyCapsule`](#pb_utilstensorto_dlpack---pycapsule)
  - [`pb_utils.Tensor.from_dlpack() -> Tensor`](#pb_utilstensorfrom_dlpack---tensor)
  - [`pb_utils.Tensor.is_cpu() -> bool`](#pb_utilstensoris_cpu---bool)
  - [Input Tensor Device Placement](#input-tensor-device-placement)
- [Examples](#examples)
  - [AddSub in NumPy](#addsub-in-numpy)
  - [AddSubNet in PyTorch](#addsubnet-in-pytorch)
  - [Business Logic Scripting](#business-logic-scripting-1)
  - [Preprocessing](#preprocessing)
  - [Decoupled Models](#decoupled-models)
- [Running with Inferentia](#running-with-inferentia)
- [Reporting problems, asking questions](#reporting-problems-asking-questions)

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
* zlib1g-dev

```
pip3 install numpy
```

On Ubuntu or Debian you can use the command below to install `rapidjson`, `libarchive`, and `zlib`:
```
sudo apt-get install rapidjson-dev libarchive-dev zlib1g-dev
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

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME>`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME>`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME>`


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

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model assuming
        the server was not started with `--disable-auto-complete-config`. Implementing
        this function is optional. No implementation of `auto_complete_config` will
        do nothing. This function can be used to set `max_batch_size`, `input` and
        `output` properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with minimal
        model configuration in absence of a configuration file. This function returns
        the `pb_utils.ModelConfig` object with these properties. You can use the `as_dict`
        function to gain read-only access to the `pb_utils.ModelConfig` object.
        The `pb_utils.ModelConfig` object being returned from here will be used as
        the final configuration for the model.

        Note: The Python interpreter used to invoke this function will be destroyed
        upon returning from this function and as a result none of the objects created
        here will be available in the `initialize`, `execute`, or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build upon
          the configuration given by this object when setting the properties for 
          this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'INPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)
        
        # To enable a dynamic batcher with default settings, you can use 
        # auto_complete_model_config set_dynamic_batching() function. It is 
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

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

Every Python backend can implement four main functions:

### `auto_complete_config`

`auto_complete_config` is called only once when loading the model assuming
the server was not started with [`--disable-auto-complete-config`](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration).

Implementing this function is optional. No implementation of
`auto_complete_config` will do nothing. This function can be used to set 
[`max_batch_size`](
  https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#maximum-batch-size),
[dynamic_batching](
  https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher),
[`input`](
  https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#inputs-and-outputs) and
[`output`](
  https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#inputs-and-outputs)
properties of the model using `set_max_batch_size`, `set_dynamic_batching`, `add_input`, and
`add_output`. These properties will allow Triton to load the model with
[minimal model configuration](
  https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#minimal-model-configuration)
in absence of a configuration file. This function returns the
`pb_utils.ModelConfig` object with these properties. You can use the `as_dict`
function to gain read-only access to the `pb_utils.ModelConfig` object.
The `pb_utils.ModelConfig` object being returned from here will be used as the
final configuration for the model.

Note: The Python interpreter used to invoke this function will be destroyed
upon returning from this function and as a result none of the objects
created here will be available in the `initialize`, `execute`, or `finalize`
functions.

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
a list of `InferenceRequest` objects. There are two modes of implementing this
function. The mode you choose should depend on your use case. That is whether
or not you want to return decoupled responses from this model or not.

#### Default Mode

This is the most generic way you would like to implement your model and
requires the `execute` function to return exactly one response per request.
This entails that in this mode, your `execute` function must return a list of
`InferenceResponse` objects that has the same length as `requests`. The work
flow in this mode is:

* `execute` function receives a batch of pb_utils.InferenceRequest as a
  length N array.

* Perform inference on the pb_utils.InferenceRequest and append the
  corresponding pb_utils.InferenceResponse to a response list.

* Return back the response list.

  * The length of response list being returned must be N.

  * Each element in the list should be the response for the corresponding
    element in the request array.

  * Each element must contain a response (a response can be either output
    tensors or an error); an element cannot be None.


Triton checks to ensure that these requirements on response list are
satisfied and if not returns an error response for all inference requests.
Upon return from the execute function all tensor data associated with the
InferenceRequest objects passed to the function are deleted, and so
InferenceRequest objects should not be retained by the Python model.

In case one of the requests has an error, you can use the `TritonError` object
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


#### Decoupled mode

This mode allows user to send multiple responses for a request or
not send any responses for a request. A model may also send
responses out-of-order relative to the order that the request batches
are executed. Such models are called *decoupled* models. In
order to use this mode, the [transaction policy](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#model-transaction-policy)
in the model configuration must be set to decoupled.


In decoupled mode, model must use `InferenceResponseSender` object per
request to keep creating and sending any number of responses for the
request. The workflow in this mode may look like:

* `execute` function receives a batch of pb_utils.InferenceRequest as a
  length N array.

* Iterate through each pb_utils.InferenceRequest and perform for the following
  steps for each pb_utils.InferenceRequest object:

  1. Get `InferenceResponseSender` object for the InferenceRequest using
     InferenceRequest.get_response_sender().

  2. Create and populate pb_utils.InferenceResponse to be sent back.

  3. Use InferenceResponseSender.send() to send the above response. If
     this is the last request then pass pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
     as a flag with InferenceResponseSender.send(). Otherwise continue with
     Step 1 for sending next request.

* The return value for `execute` function in this mode should be None.

Similar to above, in case one of the requests has an error, you can use
the `TritonError` object to set the error message for that specific
request. After setting errors for an pb_utils.InferenceResponse
object, use InferenceResponseSender.send() to send response with the
error back to the user.

##### Use Cases

The decoupled mode is powerful and supports various other use cases:

* If the model should not send any response for the request,
  then call InferenceResponseSender.send() with no response
  but flag parameter set to pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL.

* The model can also send responses out-of-order in which it received
  requests.

* The request data and `InferenceResponseSender` object can be passed to
  a separate thread in the model. This means main caller thread can exit
  from `execute` function and the model can still continue generating
  responses as long as it holds `InferenceResponseSender` object.


The [decoupled examples](examples/decoupled/README.md) demonstrate
full power of what can be acheived from decoupled API. Read
[Decoupled Backends and Models](https://github.com/triton-inference-server/server/blob/main/docs/decoupled_models.md)
for more details on how to host a decoupled model.

##### Known Issues

* Currently, async BLS requests are not supported in decoupled mode.

### `finalize`

Implementing `finalize` is optional. This function allows you to do any clean
ups necessary before the model is unloaded from Triton server.

You can look at the [add_sub example](examples/add_sub/model.py) which contains
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

**Important Note: If your Python model and its dependencies use Python 3.8,
you can skip this section and start from section 2 since the Python backend stub
shipped in Triton containers uses Python 3.8 by default.**

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
$ cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME> -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
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
stub. For example, if you have `model_a` in your
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md),
the folder structure should look like below:

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

It is also possible to provide the execution environment path relative to the
model folder in model repository:

```
name: "model_a"
backend: "python"

...

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/python3.6.tar.gz"}
}
```

In this case, `python3.tar.gz` should be placed in the model folder and the
model repository should look like below:

```
models
|-- model_a
|   |-- 1
|   |   `-- model.py
|   |-- config.pbtxt
|   |-- python3.6.tar.gz
|   `-- triton_python_backend_stub
```

In the example above, `$$TRITON_MODEL_DIRECTORY` is resolved to `$pwd/models/model_a`.

This is useful if you want to use S3, GCS, or Azure and you do not have access
to the absolute path of the execution env that is stored in the cloud object
storage service.

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

4. If you need to compile the Python backend stub, it is recommended that you
compile it in the official Triton NGC containers. Otherwise, your compiled stub
may use dependencies that are not available in the Triton container that you are
using for deployment. For example, compiling the Python backend stub on an OS
other than Ubuntu 20.04 can lead to unexpected errors.

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

## Multiple Model Instance Support

Python interpreter uses a global lock known as
[GIL](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock).
Because of GIL, it is not possible have multiple threads running in the same
Python interpreter simultaneously as each thread requires to acquire the GIL
when accessing Python objects which will serialize all the operations. In order
to work around this issue, Python backend spawns a separate process for each
[model instance](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#multiple-model-instances).
This is in contrast with how other Triton backends such as
[ONNXRuntime](https://github.com/triton-inference-server/onnxruntime_backend),
[TensorFlow](https://github.com/triton-inference-server/tensorflow_backend), and
[PyTorch](https://github.com/triton-inference-server/pytorch_backend) handle
multiple instances. Increasing the instance count for these backends will create
additional threads instead of spawning separate processes.

# Business Logic Scripting

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
      #   request_id="1", correlation_id=4, model_version=1, flags=0)

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
          # inference response too.
```


In addition to the `inference_request.exec` function that allows you to
execute blocking inference requests, `inference_request.async_exec` allows
you to perform async inference requests. This can be useful when you do not
need the result of the inference immediately. Using `async_exec` function, it
is possible to have multiple inflight inference requests and wait for the
responses only when needed. Example below shows how to use `async_exec`:

```python
import triton_python_backend_utils as pb_utils
import asyncio


class TritonPythonModel:
  ...

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make sure
      # to replace `inputs` argument with a list of `pb_utils.Tensor` objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      infer_response_awaits = []
      for i in range(4):
        # async_exec function returns an
        # [Awaitable](https://docs.python.org/3/library/asyncio-task.html#awaitables)
        # object.
        infer_response_awaits.append(inference_request.async_exec())

      # Wait for all of the inference requests to complete.
      infer_responses = await asyncio.gather(*infer_response_awaits)

      for infer_response in infer_responses:
        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output1 = pb_utils.get_output_tensor_by_name(inference_response, 'REQUESTED_OUTPUT_1')
            output2 = pb_utils.get_output_tensor_by_name(inference_response, 'REQUESTED_OUTPUT_2')

            # Decide the next steps for model execution based on the received output
            # tensors.
```

A complete example for sync and async BLS in Python backend is included in the
[Examples](#examples) section.

Starting from the 22.04 release, the lifetime of the BLS output tensors have
been improved such that if a tensor is no longer needed in your Python model it
will be automatically deallocated. This can increase the number of BLS requests
that you can execute in your model without running into the out of GPU or shared
memory error.

Note: Async BLS is not supported on Python 3.6 or lower due to the `async` keyword
and `asyncio.run` being introduced in Python 3.7. 

## Using BLS with Stateful Models

[Stateful models](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#stateful-models)
require setting additional flags in the inference request to indicate the
start and of a sequence. The `flags` argument in the `pb_utils.InferenceRequest`
object can be used to indicate whether the request is the first or last request
in the sequence. An example indicating that the request is starting the
sequence:

```python
inference_request = pb_utils.InferenceRequest(model_name='model_name',
  requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
  inputs=[<list of pb_utils.Tensor objects>],
  request_id="1", correlation_id=4, flags=pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START)
```

For indicating the ending of the sequence you can use the 
`pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END` flag. If the request is both
starting and ending a sequence at the same time (i.e. the sequence has only a
single request), you can use the bitwise OR operator to enable both of the
flags:

```
flags = pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START | pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END
```

## Limitation

- You need to make sure that the inference requests performed as a part of your model
do not create a circular dependency. For example, if model A performs an inference request
on itself and there are no more model instances ready to execute the inference request, the
model will block on the inference execution forever.

- Currently, BLS can not run inference on a decoupled model.


# Interoperability and GPU Support

Starting from 21.09 release, Python backend supports
[DLPack](https://github.com/dmlc/dlpack) for zero-copy transfer of Python
backend tensors to other frameworks. The methods below are added to the
`pb_utils.Tensor` object to facilitate the same:

## `pb_utils.Tensor.to_dlpack() -> PyCapsule`

This method can be called on existing instantiated tensors to convert
a Tensor to DLPack. The code snippet below shows how this works with PyTorch:

```python
from torch.utils.dlpack import from_dlpack
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

  def execute(self, requests):
    ...
    input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")

    # We have converted a Python backend tensor to a PyTorch tensor without
    # making any copies.
    pytorch_tensor = from_dlpack(input0.to_dlpack())
```

## `pb_utils.Tensor.from_dlpack() -> Tensor`

This static method can be used for creating a `Tensor` object from the DLPack
encoding of the tensor. For example:

```python
from torch.utils.dlpack import to_dlpack
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

  def execute(self, requests):
    ...
    pytorch_tensor = torch.tensor([1, 2, 3], device='cuda')

    # Create a Python backend tensor from the DLPack encoding of a PyTorch
    # tensor.
    input0 = pb_utils.Tensor.from_dlpack("INPUT0", to_dlpack(pytorch_tensor))
```

This method only supports contiguous Tensors that are in C-order. If the tensor
is not C-order contiguous an exception will be raised.

## `pb_utils.Tensor.is_cpu() -> bool`

This function can be used to check whether a tensor is placed in CPU or not.

## Input Tensor Device Placement

By default, the Python backend moves all input tensors to CPU before providing
them to the Python model. Starting from 21.09, you can change this default
behavior. By setting `FORCE_CPU_ONLY_INPUT_TENSORS` to "no", Triton will not
move input tensors to CPU for the Python model. Instead, Triton will provide the
input tensors to the Python model in either CPU or GPU memory, depending on how
those tensors were last used. You cannot predict which memory will be used for
each input tensor so your Python model must be able to handle tensors in both
CPU and GPU memory. To enable this setting, you need to add this setting to the
`parameters` section of model configuration:

```
parameters: { key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {string_value:"no"}}
```

# Examples

For using the Triton Python client in these examples you need to install
the [Triton Python Client Library](https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples).
The Python client for each of the examples is in the `client.py` file.

## AddSub in NumPy

There is no dependencies required for the AddSub NumPy example. Instructions
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

## Preprocessing

The Preprocessing example shows how to use Python Backend to do model preprocessing.
You can find the complete example instructions in [examples/preprocessing](examples/preprocessing/README.md).

## Decoupled Models

The examples of decoupled models shows how to develop and serve
[decoupled models](../../README.md#decoupled-mode) in Triton using Python backend.
You can find the complete example instructions in [examples/decoupled](examples/decoupled/README.md).

# Running with Inferentia

Please see the [README.md](https://github.com/triton-inference-server/python_backend/tree/main/inferentia/README.md) located in the python_backend/inferentia sub folder.

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
