# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.

    This model has three inputs and two outputs. The model does not support batching.
    
      - Input 'IN' can have any vector shape (e.g. [4] or [12]), datatype must be INT32.
      - Input 'DELAY' must have the same shape as IN, datatype must be UINT32.
      - Input 'WAIT' must have shape [1] and datatype UINT32.
      - For each response, output 'OUT' must have shape [1] and datatype INT32.
      - For each response, output 'IDX' must have shape [1] and datatype UINT32.
         
    For a request, the model will send 'n' responses where 'n' is the number of elements in IN. 
    For the i'th response, OUT will equal the i'th element of IN and IDX will equal the zero-based
    count of this response for the request. For example, the first response for a request will
    have IDX = 0 and OUT = IN[0], the second will have IDX = 1 and OUT = IN[1], etc. The model
    will wait the i'th DELAY, in milliseconds, before sending the i'th response. If IN shape is [0]
    then no responses will be sent.
    
    After WAIT milliseconds the model will return from the execute function so that Triton can call
    execute again with another request. WAIT can be less than the sum of DELAY so that execute
    returns before all responses are sent. Thus, even if there is only one instance of the model,
    multiple requests can be processed at the same time, and the responses for multiple requests
    can be intermixed, depending on the values of DELAY and WAIT.
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

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(
            model_config, "OUT")

        # Get IDX configuration
        idx_config = pb_utils.get_output_config_by_name(
            model_config, "IDX")

        # Convert Triton types to numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(
            out_config['data_type'])
        self.idx_dtype = pb_utils.triton_string_to_numpy(
            idx_config['data_type'])

    

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        None
        """
        
        # This model does not support batching, so 'request_count' should always be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException("unsupported batch size " + len(requests))

        # Start a separate thread to send the responses for the request. In a full implementation we
        # would need to keep track of any running threads so that we could delay finalizing the
        # model until all response_thread threads have completed.
        thread = threading.Thread(target=response_thread,
                                  args=(self,
                                  requests[0].get_response_sender(),
                                  pb_utils.get_input_tensor_by_name(requests[0], 'IN').as_numpy(),
                                  pb_utils.get_input_tensor_by_name(requests[0], 'DELAY').as_numpy()))
        thread.daemon = True
        thread.start()
                 
        # Read WAIT input for wait time, then return so that Triton can call execute again with
        # another request.
        wait_input = pb_utils.get_input_tensor_by_name(requests[0], 'WAIT').as_numpy()
        time.sleep(wait_input[0] / 1000)

        return None

    def response_thread(self, response_sender, in_input, delay_input):
        # The response_sender is used to send response(s) associated with the corresponding request.
        # Iterate over input/delay pairs. Wait for DELAY milliseconds and then create and
        # send a response.

        idx_dtype = self.idx_dtype
        out_dtype = self.out_dtype

        for idx in range(in_input.size):
            in_value = in_input[idx]
            delay_value = delay_input[idx]
    
            time.sleep(delay_value / 1000)

            idx_output = pb_utils.Tensor("IDX", numpy.array([idx], idx_dtype))
            out_output = pb_utils.Tensor("OUT", numpy.array([in_value], out_dtype))
            response = pb_utils.InferenceResponse(output_tensors=[idx_output, out_output])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are done sending
        # responses for the corresponding request. We can't use the response sender after
        # closing it.
        response_sender.close() 


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')