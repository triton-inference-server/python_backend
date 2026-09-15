import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Returns an InferenceResponse with a specific TritonError code.

    Used to verify that BLS callers receive the original error code
    rather than a hardcoded INTERNAL.
    """

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            error_code = pb_utils.get_input_tensor_by_name(
                request, "ERROR_CODE"
            ).as_numpy()[0]

            code_map = {
                0: pb_utils.TritonError.UNKNOWN,
                1: pb_utils.TritonError.INTERNAL,
                2: pb_utils.TritonError.NOT_FOUND,
                3: pb_utils.TritonError.INVALID_ARG,
                4: pb_utils.TritonError.UNAVAILABLE,
                5: pb_utils.TritonError.UNSUPPORTED,
                6: pb_utils.TritonError.ALREADY_EXISTS,
                7: pb_utils.TritonError.CANCELLED,
            }

            triton_code = code_map.get(error_code, pb_utils.TritonError.INTERNAL)
            error = pb_utils.TritonError(
                f"Test error with code {error_code}", triton_code
            )
            responses.append(pb_utils.InferenceResponse(error=error))
        return responses
