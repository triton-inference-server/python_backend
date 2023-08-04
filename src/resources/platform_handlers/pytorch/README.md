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

# Serving PyTorch models using Python Backend \[Experimental\]

**NOTE**: *This feature is subject to change and removal, and should not
be used in production.*

Starting from 23.08, we are adding an experimental support for loading and
serving PyTorch models directly via Python backend. The model can be provided
within the triton server model repository, and a
[pre-built Python model](model.py) will be used to load and serve the PyTorch
model.

## Model Layout

The model repository should look like:

```
model_repository/
`-- model_directory
    |-- 1
    |   |-- model.py
    |   `-- model.pt
    `-- config.pbtxt
```

The `model.py` contains the class definition of the PyTorch model. The class
should extend the
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
The `model.pt` may be optionally provided which contains the saved
[`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
of the model. For serving TorchScript models, a `model.pt` TorchScript can be
provided in place of the `model.py` file.

By default, Triton will use the
[PyTorch backend](https://github.com/triton-inference-server/pytorch_backend) to
load and serve PyTorch models. In order to serve from Python backend,
[model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
should explicitly provide the following settings:

```
backend: "python"
platform: "pytorch"
```

## PyTorch Installation

This feature will take advantage of the
[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)
optimization, make sure the
[PyTorch 2.0+ pip package](https://pypi.org/project/torch/2.0.1/) is available
in the same Python environment.

```
pip install torch==2.0.1
```
Alternatively, a
[Python Execution Environment](#using-custom-python-execution-environments)
with the PyTorch dependency may be used.

## Customization

The following PyTorch settings may be customized by setting parameters on the
`config.pbtxt`.

[`torch.set_num_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads)
- Key: NUM_THREADS
- Value: The number of threads used for intraop parallelism on CPU.

[`torch.set_num_interop_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch.set_num_interop_threads)
- Key: NUM_INTEROP_THREADS
- Value: The number of threads used for interop parallelism (e.g. in JIT
interpreter) on CPU.

[`torch.compile()` parameters](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)
- Key: TORCH_COMPILE_OPTIONAL_PARAMETERS
- Value: Any of following parameter(s) encoded as a JSON object.
  - fullgraph (*bool*): Whether it is ok to break model into several subgraphs.
  - dynamic (*bool*): Use dynamic shape tracing.
  - backend (*str*): The backend to be used.
  - mode (*str*): Can be either "default", "reduce-overhead" or "max-autotune".
  - options (*dict*): A dictionary of options to pass to the backend.
  - disable (*bool*): Turn `torch.compile()` into a no-op for testing.

For example:
```
parameters: {
    key: "NUM_THREADS"
    value: { string_value: "4" }
}
parameters: {
    key: "TORCH_COMPILE_OPTIONAL_PARAMETERS"
    value: { string_value: "{\"disable\": true}" }
}
``````

## Limitations

Following are few known limitations of this feature:
- Python functions optimizable by `torch.compile` may not be served directly in
the `model.py` file, they need to be enclosed by a class extending the
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
- Model weights cannot be shared across multiple instances on the same GPU
device.
- When using `KIND_MODEL` as model instance kind, the default device of the
first parameter on the model is used.
