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

# TensorFlow SavedModel Platform Handler \[Experimental\]

This platform handler enables loading and serving of models
in [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model)
format via Python backend. The `model.savedmodel` can be provided within
the triton server model repository without `model.py` and backend will
automatically use a pre-baked platform handler (`model.py`) to load and
serve the TF SavedModel. The handler can [auto-complete](../../../../README.md#auto_complete_config)
the missing model configuration.

In order to use this handler, make sure that [TensorFlow pip package](https://pypi.org/project/tensorflow/2.13.0/)
is available in the same Python environment.

```
pip install tensorfow==2.13.0
```

Alternatively, you can create a
[Python Execution Environment](#using-custom-python-execution-environments)
with the TensorFlow dependency.

By default, Triton will use the [TensorFlow backend](https://github.com/triton-inference-server/tensorflow_backend)
to load and serve the saved model. In order to use the Python backend with
TensorFlow Platform Handler, [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
should explicitly provide the following settings:

```
backend: "python"
platform: "tensorflow_savedmodel"
```

Following are few known limitations of the TF platform handler:
- GPU execution is not supported.
- List of requests received in model [`execute`](../../../../README.md#execute) function are
not run in a single batch but one after the other.
