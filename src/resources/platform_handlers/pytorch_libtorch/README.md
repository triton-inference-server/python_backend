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

*NOTE*: This feature is subject to change and removal, and should not
be used in production.

Starting from 23.08, we are adding an experimental support for loading and
serving PyTorch models directly without using TorchScript via Python backend.
The model can be provided within the triton server model repository without
crafting the `model.py`, and a pre-built Python model [`model.py`](model.py)
will be used to load and serve the provided PyTorch model.

The model repository structure should look like:

```
model_repository/
`-- pytorch_model
    |-- 1
    |   |-- model.py
    |   `-- model.py.pt
    `-- config.pbtxt
```

The `model.py` contains the class definition of the PyTorch model. The class
name needs to match the model name (i.e. `pytorch_model`), and the class should
extend the
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

The `model.py.pt` may be optionally provided which contains the saved
[`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
of the model.

By default, Triton will use the
[PyTorch backend](https://github.com/triton-inference-server/pytorch_backend) to
load and serve PyTorch models in TorchScript format. In order to use the Python
backend to serve PyTorch models directly without TorchScript,
[model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
should explicitly provide the following settings:

```
backend: "python"
platform: "pytorch_libtorch"
```

This feature will take advantage of the
[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)
optimization, make sure the
[PyTorch pip package](https://pypi.org/project/torch/2.0.1/) is available in the
same Python environment.

```
pip install torch==2.0.1
```
Alternatively, a
[Python Execution Environment](#using-custom-python-execution-environments)
with the PyTorch dependency may be used.

Following are few known limitations of this feature:
- GPU execution is not yet supported.
- List of requests received in model [`execute`](../../../../README.md#execute)
function are not yet ran in a single batch but one after the other.
