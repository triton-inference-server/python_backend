# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
from tensorflow.python.tools import saved_model_utils
import tensorflow as tf
from tensorflow.core.framework import types_pb2

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

#def _validate_model_config(config):

TF_STRING_TO_TRITON = {
    'DT_BOOL': 'TYPE_BOOL',
    'DT_UINT8': 'TYPE_UINT8',
    'DT_UINT16': 'TYPE_UINT16',
    'DT_UINT32': 'TYPE_UINT32',
    'DT_UINT64': 'TYPE_UINT64',
    'DT_INT8': 'TYPE_INT8',
    'DT_INT16': 'TYPE_INT16',
    'DT_INT32': 'TYPE_INT32',
    'DT_INT64': 'TYPE_INT64',
    'DT_HALF': 'TYPE_FP16',
    'DT_FLOAT': 'TYPE_FP32',
    'DT_DOUBLE': 'TYPE_FP64',
    'DT_STRING': 'TYPE_STRING',
}


def _parse_signature_def(config):
    if (config['parameters']):
        if ('TF_SIGNATURE_DEF' in config['parameters'].keys()):
            return config['parameters']['TF_SIGNATURE_DEF']['string_value']
    return None


def _parse_graph_tag(config):
    if (config['parameters']):
        if ('TF_GRAPH_TAG' in config['parameters'].keys()):
            return config['parameters']['TF_GRAPH_TAG']['string_value']
    return None


def _parse_num_intra_threads(config):
    if (config['parameters']):
        if ('TF_NUM_INTRA_THREADS' in config['parameters'].keys()):
            return int(
                config['parameters']['TF_NUM_INTRA_THREADS']['string_value'])
    return None


def _parse_num_inter_threads(config):
    if (config['parameters']):
        if ('TF_NUM_INTER_THREADS' in config['parameters'].keys()):
            return int(
                config['parameters']['TF_NUM_INTER_THREADS']['string_value'])
    return None


def _get_truth_value(string_value):
    val = string_value.casefold()
    if (val == 'yes' or val == '1' or val == 'on' or val == 'true'):
        return True
    else:
        return False


def _parse_use_per_session_thread(config):
    if (config['parameters']):
        if ('USE_PER_SESSION_THREAD' in config['parameters'].keys()):
            val = config['parameters']['USE_PER_SESSION_THREAD']['string_value']
            return _get_truth_value(val)
    return False


def _get_signature_def(savedmodel_path, config):
    tag_sets = saved_model_utils.get_saved_model_tag_sets(savedmodel_path)
    graph_tag = _parse_graph_tag(config)
    if (graph_tag is None):
        if 'serve' in tag_sets[0]:
            graph_tag = 'serve'
        else:
            graph_tag = tag_sets[0][0]

        meta_graph_def = saved_model_utils.get_meta_graph_def(
            savedmodel_path, graph_tag)

        signature_def_map = meta_graph_def.signature_def
        signature_def_k = _parse_signature_def(config)
        if (signature_def_k is None):
            if 'serving_default' in signature_def_map.keys():
                signature_def_k = 'serving_default'
            else:
                signature_def_k = signature_def_map.keys()[0]

        if (signature_def_k not in signature_def_map.keys()):
            raise pb_utils.TritonModelException(
                f" The model does not include the signature_def '" +
                signature_def_k + "'")

    return signature_def_map[signature_def_k]


def _has_batch_dim(tensor_info):
    if (tensor_info.tensor_shape.unknown_rank):
        return True
    elif (tensor_info.tensor_shape.dim[0].size == -1):
        return True
    else:
        return False


def _get_batching_hint_from_signature(signature_def):
    batching_hint = True
    for input_info in signature_def.inputs.values():
        if not _has_batch_dim(input_info):
            return False

    for output_info in signature_def.outputs.values():
        if not _has_batch_dim(output_info):
            return False

    return True


def _convert_proto_to_dict_tensor(name, tensor_proto, batching_enabled):
    tensor_dict = {}
    tensor_dict['name'] = name
    dtype_dict = {value: key for (key, value) in types_pb2.DataType.items()}
    tensor_dict['data_type'] = TF_STRING_TO_TRITON[dtype_dict[
        tensor_proto.dtype]]
    if (tensor_proto.tensor_shape.unknown_rank):
        # FIXME: Fix the handling of unknown rank
        dims = [-1]
    else:
        dims = [dim.size for dim in tensor_proto.tensor_shape.dim]
    if batching_enabled:
        tensor_dict['dims'] = dims[1:]
    else:
        tensor_dict['dims'] = dims

    return tensor_dict


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def auto_complete_config(auto_complete_model_config, fw_model_path):
        if (fw_model_path is None):
            raise pb_utils.TritonModelException(
                f"[INTERNAL]: The path to the framework model should be"
                " provided")
        config = auto_complete_model_config.as_dict()

        if (config['platform'] != 'tensorflow_savedmodel'):
            raise pb_utils.TritonModelException(
                f"[INTERNAL]: The platform field for using this plugin should be set to"
                " \'tensorflow_savedmodel\' in model config, got '" +
                config['platform'] + "'")
        if (config['batch_input']):
            raise pb_utils.TritonModelException(
                f" The plugin does not support model with batch_input")
        if (config['batch_output']):
            raise pb_utils.TritonModelException(
                f" The plugin does not support model with batch_output")

        batching_enabled = False
        if (config['max_batch_size'] != 0):
            batching_enabled = True

        if (config['output']):
            auto_complete_output = True

        signature_def = _get_signature_def(fw_model_path, config)

        input_tensor_info = signature_def.inputs
        output_tensor_info = signature_def.outputs

        batching_hint = False
        if not batching_enabled:
            batching_hint = _get_batching_hint_from_signature(signature_def)

        # FIXME: Currently the presence of dynamic batch dimension is
        # being treated as sufficient proof for enabling batching.
        # Need to visit the tensors that are already provided in config
        # to confirm the hint
        batching_enabled = batching_hint

        config_input_names = []
        config_output_names = []
        for input in config['input']:
            config_input_names.append(input['name'])
        for output in config['output']:
            config_output_names.append(output['name'])

        # TODO: Add auto-completion of partial tensor specification.
        for input_name in input_tensor_info.keys():
            if input_name not in config_input_names:
                auto_complete_model_config.add_input(
                    _convert_proto_to_dict_tensor(input_name,
                                                  input_tensor_info[input_name],
                                                  batching_enabled))

        for output_name in output_tensor_info.keys():
            if output_name not in config_output_names:
                auto_complete_model_config.add_output(
                    _convert_proto_to_dict_tensor(
                        output_name, output_tensor_info[output_name],
                        batching_enabled))

        if batching_enabled:
            if (config['max_batch_size'] == 0):
                auto_complete_model_config.set_max_batch_size(4)
            auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config


    def initialize(self, args, fw_model_path):
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
        fw_model_path : str
          The complete path to the framework model
        """

        #print("Initializing Framework Model Path => " + fw_model_path)
        self.model_name = args['model_name']
        print("Initializing plugin model for " + self.model_name)

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        self.tf_session = tf.compat.v1.Session(graph=tf.Graph())
        tf.compat.v1.saved_model.loader.load(self.tf_session, ["serve"],
                                             fw_model_path)
        # Get the output node from the graph.
        graph = tf.compat.v1.get_default_graph()

        print(graph)

        # [TODO] Get the output nodes from the graph

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
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # [TODO] Collect requests in single batch, run execution and split the
        # result buffer into separate responses

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Removing plugin model for ' + self.model_name)
