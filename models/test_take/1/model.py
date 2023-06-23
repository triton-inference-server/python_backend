"""
Category model
"""
import time
from typing import cast
import timeit
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
        shm = triton_python_backend_utils.shared_memory
        n = 100000
        candidate_tensor = pb_utils.new_shm_tensor("candidatesss", shm, (n, 200), np.float32) # Offset is 68
        buffer = candidate_tensor.as_numpy()

        pb_utils.Logger.log_error(f"buffer - {buffer}, {buffer.dtype}, {buffer.shape}, {buffer.flags}, {buffer.base}")
        candidates_cache = np.random.random((500000, 200)).astype(np.float32)
        candidates = np.random.randint(100000, size=n)
        np_out = np.empty((n, 200), dtype=np.float32)

        r1 = timeit.timeit("buffer[:] = np.take(candidates_cache, candidates, axis=0, mode='clip')", number=100, globals={"candidates_cache":candidates_cache, "candidates":candidates, "buffer": buffer, "np":np})*10
        r2 = timeit.timeit("np.take(candidates_cache, candidates, axis=0, mode='clip', out=buffer)", number=100, globals={"candidates_cache":candidates_cache, "candidates":candidates, "buffer": buffer, "np":np})*10
        r3 = timeit.timeit("r = np.take(candidates_cache, candidates, axis=0, mode='clip')", number=100, globals={"candidates_cache":candidates_cache, "candidates":candidates, "buffer": buffer, "np":np})*10
        r4 = timeit.timeit("np.take(candidates_cache, candidates, axis=0, mode='clip', out=np_out)", number=100, globals={"candidates_cache":candidates_cache, "candidates":candidates, "buffer": buffer, "np":np, "np_out":np_out})*10

        pb_utils.Logger.log_error(f"Buffer - assignment - {r1}")
        pb_utils.Logger.log_error(f"Buffer - output - {r2}")
        pb_utils.Logger.log_error(f"Baseline - assignment - {r3}")
        pb_utils.Logger.log_error(f"Baseline - np out - {r4}")
        pb_utils.Logger.log_error(f"numpy version {np.__version__}")


    def execute_request(self, request):
        pass

    def execute(self, requests):
        return [self.execute_request(request) for request in requests]
