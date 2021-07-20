from tritonclient.utils import *
import tritonclient.http as httpclient

import numpy as np

model_name = "huggingface_t5_translation"
example_input = "translate English to German: The house is wonderful."

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.array([example_input], dtype=object)
    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape, "BYTES"),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0", binary_data=True),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("translate: INPUT0 ({}) to OUTPUT0 ({})".format(
        input0_data, response.as_numpy("OUTPUT0").astype(str)[0]))
