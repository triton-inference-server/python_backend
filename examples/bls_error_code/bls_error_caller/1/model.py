import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Calls error_source via BLS and reports back the received error code.

    If error codes propagate correctly, RECEIVED_CODE should match the
    ERROR_CODE sent to error_source.
    """

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            error_code_tensor = pb_utils.get_input_tensor_by_name(
                request, "ERROR_CODE"
            )

            bls_request = pb_utils.InferenceRequest(
                model_name="error_source",
                requested_output_names=["OUTPUT"],
                inputs=[pb_utils.Tensor("ERROR_CODE", error_code_tensor.as_numpy())],
            )

            bls_response = bls_request.exec()

            if bls_response.has_error():
                error = bls_response.error()
                received_code = int(error.code())
                error_msg = str(error.message())
            else:
                received_code = -1
                error_msg = "no error"

            out_code = pb_utils.Tensor(
                "RECEIVED_CODE", np.array([received_code], dtype=np.int32)
            )
            out_msg = pb_utils.Tensor(
                "ERROR_MESSAGE", np.array([error_msg], dtype=object)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_code, out_msg])
            )
        return responses
