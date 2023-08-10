#!/usr/bin/env python3

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

import os
import sys

import numpy as np
from PIL import Image
from tritonclient import http as httpclient
from tritonclient.utils import *

script_directory = os.path.dirname(os.path.realpath(__file__))

server_url = "localhost:8000"
model_name = "resnet50_pytorch"
input_name = "INPUT"
output_name = "OUTPUT"
label_path = os.path.join(script_directory, "resnet50_labels.txt")
# https://raw.githubusercontent.com/triton-inference-server/server/main/qa/images/mug.jpg
image_path = os.path.join(script_directory, "mug.jpg")
expected_output_class = "COFFEE MUG"


def _load_input_image():
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB").resize((224, 224), Image.BILINEAR)
    input_image = np.array(raw_image).astype(np.float32)
    input_image = (input_image / 127.5) - 1
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.reshape(input_image, (1, 3, 224, 224))
    return input_image


def _check_output(output_tensors):
    with open(label_path) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}
    max_id = np.argmax(output_tensors, axis=1)[0]
    output_class = labels_dict[max_id]
    print("Result: " + output_class)
    print("Expected result: " + expected_output_class)
    if output_class != expected_output_class:
        return False
    return True


if __name__ == "__main__":
    input_image = _load_input_image()

    with httpclient.InferenceServerClient(server_url) as client:
        input_tensors = httpclient.InferInput(input_name, input_image.shape, "FP32")
        input_tensors.set_data_from_numpy(input_image)
        results = client.infer(model_name=model_name, inputs=[input_tensors])
        output_tensors = results.as_numpy(output_name)

    if not _check_output(output_tensors):
        print("PyTorch platform handler example error: Unexpected result")
        sys.exit(1)

    print("PASS: PyTorch platform handler")
    sys.exit(0)
