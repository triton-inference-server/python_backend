"""
Category model
"""
import time
from typing import cast

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import tests.stub.triton_python_backend_utils

    pb_utils: tests.stub.triton_python_backend_utils = cast(
        tests.stub.triton_python_backend_utils, None
    )


def breakpoint():
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        'host.docker.internal', port=5858, stdoutToServer=True, stderrToServer=True
    )


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils
        self.shm = triton_python_backend_utils.shared_memory
        self.candidates_cache = np.random.random((500000, 200)).astype(np.float32)

    def execute_request(self, request):
        n = int(pb_utils.get_input_tensor_by_name(request, "n").as_numpy()[0])
        candidates = np.random.randint(100000, size=n)
        candidate_tensor: pb_utils.Tensor = pb_utils.new_shm_tensor("candidatesss", self.shm, (n, 200), np.float32)
        np.take(self.candidates_cache, candidates, axis=0, out=candidate_tensor.as_numpy(), mode='clip')

        context_array = np.random.random((10, 200)).astype(np.float32)
        context_tensor = pb_utils.Tensor(
            "user_history",
            context_array,
        )

        inference_response = pb_utils.InferenceRequest(
            model_name="category_tensorflow_model",
            requested_output_names=["scores"],
            inputs=[candidate_tensor, context_tensor],
        ).exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            scores = pb_utils.get_output_tensor_by_name(inference_response, "scores")

        out_scores = pb_utils.Tensor("scores", scores.as_numpy()[:400])

        response = pb_utils.InferenceResponse(output_tensors=[out_scores])

        return response

    def execute(self, requests):
        return [self.execute_request(request) for request in requests]
