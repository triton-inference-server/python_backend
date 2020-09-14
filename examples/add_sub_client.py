from tritonclientutils import *
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient

import numpy as np

model_name = "python_float32_float32_float32"
shape = [1, 16]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input_data = np.ones(shape).astype(np.float32)
    inputs = [
        httpclient.InferInput("INPUT0", input_data.shape,
                              np_to_triton_dtype(input_data.dtype)),
        httpclient.InferInput("INPUT1", input_data.shape,
                              np_to_triton_dtype(input_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input_data)
    inputs[1].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("INPUT0 + INPUT1 = ", response.get_output("OUTPUT0"))
    print("INPUT0 - INPUT1 = ", response.get_output("OUTPUT1"))
