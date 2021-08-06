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

# BLS Example

In this example we demonstrate an end-to-end example for
[BLS](../../README.md#business-logic-scripting-beta) in Python backend. The
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md)
should contain [PyTorch](../pytorch), [AddSub](../add_sub), and [BLS](../bls) models.
The [PyTorch](../pytorch) and [AddSub](../add_sub) models
calculate the sum and difference of the `INPUT0` and `INPUT1` and put the
results in `OUTPUT0` and `OUTPUT1` respectively. The goal of the BLS model is
the same as [PyTorch](../pytorch) and [AddSub](../add_sub) models but the
difference is that the BLS model will not calculate the sum and difference by
itself. The BLS model will pass the input tensors to the [PyTorch](../pytorch)
or [AddSub](../add_sub) models and return the responses of that model as the
final response. The additional parameter `MODEL_NAME` determines which model
will be used for calculating the final outputs.

1. Create the model repository:

```console
$ mkdir -p models/add_sub/1
$ mkdir -p models/bls/1
$ mkdir -p models/pytorch/1

# Copy the Python models
$ cp examples/add_sub/model.py models/add_sub/1/
$ cp examples/add_sub/config.pbtxt models/add_sub/
$ cp examples/bls/model.py models/bls/1/
$ cp examples/bls/config.pbtxt models/bls/
$ cp examples/pytorch/model.py models/pytorch/1/
$ cp examples/pytorch/config.pbtxt models/pytorch/
```

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

3. Send inference requests to server:

```
python3 examples/bls/client.py
```

You should see an output similar to the output below:

```
=========='add_sub' model result==========
INPUT0 ([0.34984654 0.6808792  0.6509772  0.6211422 ]) + INPUT1 ([0.37917137 0.9080451  0.60789365 0.33425143]) = OUTPUT0 ([0.7290179 1.5889243 1.2588708 0.9553937])
INPUT0 ([0.34984654 0.6808792  0.6509772  0.6211422 ]) - INPUT1 ([0.37917137 0.9080451  0.60789365 0.33425143]) = OUTPUT0 ([-0.02932483 -0.22716594  0.04308355  0.28689077])


=========='pytorch' model result==========
INPUT0 ([0.34984654 0.6808792  0.6509772  0.6211422 ]) + INPUT1 ([0.37917137 0.9080451  0.60789365 0.33425143]) = OUTPUT0 ([0.7290179 1.5889243 1.2588708 0.9553937])
INPUT0 ([0.34984654 0.6808792  0.6509772  0.6211422 ]) - INPUT1 ([0.37917137 0.9080451  0.60789365 0.33425143]) = OUTPUT0 ([-0.02932483 -0.22716594  0.04308355  0.28689077])


=========='undefined' model result==========
Failed to process the request(s) for model instance 'bls_0', message: TritonModelException: Failed for execute the inference request. Model 'undefined_model' is not ready.

At:
  /tmp/python_backend/models/bls/1/model.py(110): execute
```

The [bls](./model.py) model file is heavily commented with explanations about
each of the function calls.

## Explanation of the Client Output

The [client.py](./client.py) sends three inference requests to the 'bls'
model with different values for the "MODEL_NAME" input. As explained earlier,
"MODEL_NAME" determines the model name that the "bls" model will use for
calculating the final outputs. In the first request, it will use the "add_sub"
model and in the seceond request it will use the "pytorch" model. The third
request uses an incorrect model name to demonstrate error handling during
the inference request execution.
