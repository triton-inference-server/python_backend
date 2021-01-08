# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import triton_python_backend_utils as tpb_utils

import argparse
import concurrent.futures as futures
import importlib.util
import sys
import threading
import signal
import time
import struct
import traceback
from pathlib import Path

from grpc_channelz.v1 import channelz
from grpc_channelz.v1 import channelz_pb2
from grpc_channelz.v1 import channelz_pb2_grpc

import numpy as np

from python_host_pb2 import *
from python_host_pb2_grpc import PythonInterpreterServicer, add_PythonInterpreterServicer_to_server
import grpc

MAX_GRPC_MESSAGE_SIZE = 2147483647


def serialize_byte_tensor(input_tensor):
    """
        Serializes a bytes tensor into a flat numpy array of length prepend bytes.
        Can pass bytes tensor as numpy array of bytes with dtype of np.bytes_,
        numpy strings with dtype of np.str_ or python strings with dtype of np.object.
        Parameters
        ----------
        input_tensor : np.array
            The bytes tensor to serialize.
        Returns
        -------
        serialized_bytes_tensor : np.array
            The 1-D numpy array of type uint8 containing the serialized bytes in 'C' order.
        Raises
        ------
        InferenceServerException
            If unable to serialize the given tensor.
        """

    if input_tensor.size == 0:
        return np.empty([0])

    # If the input is a tensor of string/bytes objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C"
    # order.
    if (input_tensor.dtype == np.object) or (input_tensor.dtype.type
                                             == np.bytes_):
        flattened = bytes()
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order='C'):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if obj.dtype.type == np.bytes_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = bytes(obj)
            else:
                s = str(obj).encode('utf-8')
            flattened += struct.pack("<I", len(s))
            flattened += s
        flattened_array = np.asarray(flattened)
        if not flattened_array.flags['C_CONTIGUOUS']:
            flattened_array = np.ascontiguousarray(flattened_array)
        return flattened_array
    else:
        raise TritonModelException(
            "cannot serialize bytes tensor: invalid datatype")
    return None


def deserialize_bytes_tensor(encoded_tensor):
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects
    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in 'C' order.
    """
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=bytes))


def parse_startup_arguments():
    parser = argparse.ArgumentParser(description="Triton Python Host")
    parser.add_argument("--socket",
                        default=None,
                        required=True,
                        type=str,
                        help="Socket to comunicate with server")
    parser.add_argument("--model-path",
                        default=None,
                        required=True,
                        type=str,
                        help="Path to model code")
    parser.add_argument("--instance-name",
                        default=None,
                        required=True,
                        type=str,
                        help="Triton instance name")
    return parser.parse_args()


class PythonHost(PythonInterpreterServicer):
    """This class handles inference request for python script.
    """

    def __init__(self, module_path, *args, **kwargs):
        super(PythonInterpreterServicer, self).__init__(*args, **kwargs)

        module_path = Path(module_path).resolve()
        # Add model parent directories so that relative and absolute import work
        sys.path.append(str(module_path.parent))
        sys.path.append(str(module_path.parent.parent))

        # We need to import the parent directory of the module
        # so that the relative imports work too.
        spec = importlib.util.spec_from_file_location(
            f'{module_path.parent.name}.{module_path.name}', str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.module_path = module_path

        if hasattr(module, 'TritonPythonModel'):
            self.model_instance = module.TritonPythonModel()
        else:
            raise NotImplementedError(
                'TritonPythonModel class doesn\'t exist in ' + module_path)

    def Init(self, request, context):
        """Init is called on TRITONBACKEND_ModelInstanceInitialize. `request`
        object contains an args key which includes a `model_config` key
        containing the model configuration. This paramter is passed by
        default to every ModelInstance.
        """

        model_instance = self.model_instance

        if not hasattr(request, 'args'):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('request objects does\'nt have args attribute')
            return Empty()

        if hasattr(model_instance, 'initialize'):
            args = {x.key: x.value for x in request.args}
            try:
                self.model_instance.initialize(args)
            except Exception as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                tb = traceback.format_exc()
                context.set_details(tb)

        return Empty()

    def Fini(self, request, context):
        """Fini is called on TRITONBACKEND_ModelInstanceFinalize. Model
        can perform any necessary clean up in the `finalize` function.
        """

        if hasattr(self.model_instance, 'finalize'):
            try:
                self.model_instance.finalize()
            except Exception as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                tb = traceback.format_exc()
                context.set_details(tb)

        return Empty()

    def Execute(self, request, context):
        """Execute is called on TRITONBACKEND_ModelInstanceExecute. Inference
        happens in this function. This function mainly converts gRPC
        protobufs to the triton_python_backend_utils.InferenceRequest and
        triton_python_backend_utils.InferenceResponse.

        Parameters
        ----------
        request : python_host_pb2.ExecuteRequest
            Contains a `requests` attribute which is a list of python_host_pb2.InferenceRequest
        """

        requests = request.requests
        inference_requests = []
        for request in requests:
            # This object contains a list of tpb_utils.Tensor
            input_tensors = []
            for request_input in request.inputs:
                x = request_input
                numpy_type = tpb_utils.triton_to_numpy_type(x.dtype)

                # We need to deserialize TYPE_STRING
                if numpy_type == np.object_ or numpy_type == np.bytes_:
                    numpy_data = deserialize_bytes_tensor(x.raw_data)
                    tensor = tpb_utils.Tensor(x.name,
                                              numpy_data.reshape(x.dims))
                    input_tensors.append(tensor)
                else:
                    tensor = tpb_utils.Tensor(
                        x.name,
                        np.frombuffer(x.raw_data,
                                      dtype=numpy_type).reshape(x.dims))
                    input_tensors.append(tensor)

            request_id = request.id
            correlation_id = request.correlation_id
            requested_output_names = request.requested_output_names
            inference_request = tpb_utils.InferenceRequest(
                input_tensors, request_id, correlation_id,
                requested_output_names)
            inference_requests.append(inference_request)

        # Execute inference on the Python model instance. `responses` contains
        # a list of triton_python_backend_utils.InferenceResponse. Each backend
        # must implement an execute method.
        if not hasattr(self.model_instance, 'execute'):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f'Python model {self.module_path} does not implement `execute` method.'
            )
            return ExecuteResponse()

        try:
            responses = self.model_instance.execute(inference_requests)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            tb = traceback.format_exc()
            context.set_details(tb)

        # Make sure that number of InferenceResponse and InferenceRequest
        # objects match
        if len(inference_requests) != len(responses):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                'Number of inference responses and requests don\'t match ( requests='
                + len(inference_requests) + ' != responses=' + len(responses) +
                ')')
            return ExecuteResponse()

        exec_responses = []
        for response in responses:
            # If there is an error do not look into output_tensors
            if response.has_error():
                error = Error(message=response.error().message())
                inference_response = InferenceResponse(outputs=[],
                                                       error=error,
                                                       failed=True)
                exec_responses.append(inference_response)
                continue

            output_tensors = response.output_tensors()
            response_tensors = []

            for output_tensor in output_tensors:
                output_np_array = output_tensor.as_numpy()
                output_shape = output_np_array.shape

                # We need to serialize TYPE_STRING
                if output_np_array.dtype == np.object or output_np_array.dtype.type is np.bytes_:
                    output_np_array = serialize_byte_tensor(output_np_array)

                tensor = Tensor(name=output_tensor.name(),
                                dtype=tpb_utils.numpy_to_triton_type(
                                    output_np_array.dtype.type),
                                dims=output_shape,
                                raw_data=output_np_array.tobytes())

                response_tensors.append(tensor)
            exec_responses.append(InferenceResponse(outputs=response_tensors))
        execute_response = ExecuteResponse(responses=exec_responses)

        return execute_response


def watch_connections(address, event):
    # Sleep for 4 seconds to ensure that the Python gRPC server has started
    time.sleep(4)

    number_of_failures = 0
    timeout_sec = 1

    while True:
        # If there are three failures, shutdown the gRPC server.
        if number_of_failures == 3:
            event.set()
            break

        channel = grpc.insecure_channel(address)
        try:
            channelz_stub = channelz_pb2_grpc.ChannelzStub(channel)
            # Wait for the channel to be ready.
            grpc.channel_ready_future(channel).result(timeout=timeout_sec)
        except grpc.FutureTimeoutError:
            number_of_failures += 1
            continue
        servers = channelz_stub.GetServers(
            channelz_pb2.GetServersRequest(start_server_id=0))
        sockets = channelz_stub.GetServerSockets(
            channelz_pb2.GetServerSocketsRequest(
                server_id=servers.server[0].ref.server_id, start_socket_id=0))

        # There should be always more than one socket connected to the server
        if len(sockets.socket_ref) == 1:
            number_of_failures += 1
        else:
            number_of_failures = 0

        # Sleep for 2 seconds before polling next time
        time.sleep(2)


if __name__ == "__main__":
    signal_received = False
    FLAGS = parse_startup_arguments()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
                         options=[
                             ('grpc.max_send_message_length',
                              MAX_GRPC_MESSAGE_SIZE),
                             ('grpc.max_receive_message_length',
                              MAX_GRPC_MESSAGE_SIZE),
                         ])
    channelz.add_channelz_servicer(server)
    # Create an Event to keep the GRPC server running
    event = threading.Event()
    python_host = PythonHost(module_path=FLAGS.model_path)
    add_PythonInterpreterServicer_to_server(python_host, server)

    def interrupt_handler(signum, frame):
        pass

    def sigterm_handler(signum, frame):
        global signal_received
        if not signal_received:
            signal_received = True
        else:
            return

        event.set()

    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    server.add_insecure_port(FLAGS.socket)
    server.start()

    # A Background thread to monitor the status of the gRPC server
    background_thread = threading.Thread(target=watch_connections,
                                         args=(FLAGS.socket, event))

    # Set background_thread as a daemon thread so that it doesn't interrupt the
    # program termination
    background_thread.daemon = True
    background_thread.start()
    event.wait()
    server.stop(grace=5)
    sys.exit()
