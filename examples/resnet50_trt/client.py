# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os, sys
import numpy as np
import json
import tritongrpcclient
import argparse
import time


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    This is a typical approach you'd like to use in DALI backend.
    DALI performs image decoding, therefore this way the processing
    can be fully offloaded to the GPU.
    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str, required=False,
                        default="ensemble_python_resnet50",
                        help="Model name")
    parser.add_argument("--image",
                        type=str,
                        required=True,
                        help="Path to the image")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v', "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument("--label_file",
                        type=str,
                        default="./model_repository/resnet50_trt/labels.txt",
                        help="Path to the file with text representation of available labels")
    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(args.label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    inputs = []
    outputs = []
    input_name = "INPUT"
    output_name = "OUTPUT"
    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)

    inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(image_data)
    start_time = time.time()
    # Test with outputs
    results = triton_client.infer(model_name=args.model_name,
                                  inputs=inputs,
                                  outputs=outputs)
    latency = time.time() - start_time

    output0_data = results.as_numpy(output_name)

    maxs = np.argmax(output0_data, axis=1)

    print("{}ms class: {}".format(latency, labels_dict[maxs[0]]))
