# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import threading
import numpy as np

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.

    This model demonstrates how to write a decoupled model where each
    request can generate 0 to many responses.

    This model has one input and one output. The model can support batching,
    with constraint that each request must be batch-1 request, but the shapes
    described here refer to the non-batch portion of the shape.
    
      - Input 'IN' must have shape [1] and datatype INT32.
      - Output 'OUT' must have shape [1] and datatype INT32.
         
    For a request, the backend will sent 'n' responses where 'n' is the
    element in IN. For each response, OUT will equal the element of IN.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to 
                serve this model""".format(args['model_name']))

        # Get IN configuration
        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        # Validate the shape and data type of IN
        in_shape = in_config['dims']
        if (len(in_shape) != 1) or (in_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'IN' to be
                [1], got {}""".format(args['model_name'], in_shape))
        if in_config['data_type'] != 'TYPE_INT32':
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'IN' to be
                'TYPE_INT32', got {}""".format(args['model_name'],
                                               in_config['data_type']))

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Validate the shape and data type of OUT
        out_shape = out_config['dims']
        if (len(out_shape) != 1) or (out_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'OUT' to be
                [1], got {}""".format(args['model_name'], out_shape))
        if out_config['data_type'] != 'TYPE_INT32':
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'OUT' to be
                'TYPE_INT32', got {}""".format(args['model_name'],
                                               out_config['data_type']))

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. The request.get_response_sender() must be used to
        get an InferenceResponseSender object associated with the request.
        Use the InferenceResponseSender.send() and InferenceResponseSender.end()
        calls to send responses and indicate no responses will be sent for
        the corresponding request respectively. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        None
        """

        for request in requests:
            self.process_request(request)

        return None

    def process_request(self, request):
        thread = threading.Thread(target=response_thread,
                                  args=(self, request.get_response_sender(),
                                        pb_utils.get_input_tensor_by_name(
                                            request, 'IN').as_numpy()))
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input):
        # The response_sender is used to send response(s) associated with the corresponding request.

        for idx in range(in_input[0]):
            out_output = pb_utils.Tensor("OUT",
                                         numpy.array([in_input[0]], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are done sending
        # responses for the corresponding request. We can't use the response sender after
        # closing it.
        response_sender.close()

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """

        print('Finalize invoked')

        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = (self.inflight_thread_count != 0)
            if inflight_threads:
                time.sleep(0.1)

        print('Finalize complete...')
