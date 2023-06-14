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
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.client import session
import os

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

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

_DEFAULT_ARTIFACT_NAME = "model.savedmodel"


def _get_savedmodel_path(args, config):
    artifact_name = config['default_model_filename']
    if not artifact_name:
        artifact_name = _DEFAULT_ARTIFACT_NAME

    savedmodel_path = os.path.join(args['model_repository'],
                                   args['model_version'], artifact_name)
    if not os.path.exists(savedmodel_path):
        raise pb_utils.TritonModelException(f" No model file found in " +
                                            savedmodel_path)

    return savedmodel_path


def _parse_signature_def(config):
    if config['parameters']:
        if 'TF_SIGNATURE_DEF' in config['parameters'].keys():
            return config['parameters']['TF_SIGNATURE_DEF']['string_value']
    return None


def _parse_graph_tag(config):
    if config['parameters']:
        if 'TF_GRAPH_TAG' in config['parameters'].keys():
            return config['parameters']['TF_GRAPH_TAG']['string_value']
    return None


def _parse_num_intra_threads(config):
    if config['parameters']:
        if 'TF_NUM_INTRA_THREADS' in config['parameters'].keys():
            return int(
                config['parameters']['TF_NUM_INTRA_THREADS']['string_value'])
    return None


def _parse_num_inter_threads(config):
    if config['parameters']:
        if 'TF_NUM_INTER_THREADS' in config['parameters'].keys():
            return int(
                config['parameters']['TF_NUM_INTER_THREADS']['string_value'])
    return None


def _get_truth_value(string_value):
    val = string_value.casefold()
    if val == 'yes' or val == '1' or val == 'on' or val == 'true':
        return True
    else:
        return False


def _parse_use_per_session_thread(config):
    if config['parameters']:
        if 'USE_PER_SESSION_THREAD' in config['parameters'].keys():
            val = config['parameters']['USE_PER_SESSION_THREAD']['string_value']
            return _get_truth_value(val)
    return False


def _get_signature_def(savedmodel_path, config):
    tag_sets = saved_model_utils.get_saved_model_tag_sets(savedmodel_path)
    graph_tag = _parse_graph_tag(config)
    if graph_tag is None:
        if 'serve' in tag_sets[0]:
            graph_tag = 'serve'
        else:
            graph_tag = tag_sets[0][0]

    meta_graph_def = saved_model_utils.get_meta_graph_def(
        savedmodel_path, graph_tag)
    signature_def_map = meta_graph_def.signature_def
    signature_def_k = _parse_signature_def(config)
    if signature_def_k is None:
        serving_default = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        if serving_default in signature_def_map.keys():
            signature_def_k = serving_default
        else:
            signature_def_k = signature_def_map.keys()[0]

    if signature_def_k not in signature_def_map.keys():
        raise pb_utils.TritonModelException(
            f" The model does not include the signature_def '" +
            signature_def_k + "'")

    return graph_tag, signature_def_map[signature_def_k]


def _has_batch_dim(tensor_info):
    if tensor_info.tensor_shape.unknown_rank:
        return True
    elif tensor_info.tensor_shape.dim[0].size == -1:
        return True
    else:
        return False


def _get_batching_hint_from_signature(signature_def):
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
    if tensor_proto.tensor_shape.unknown_rank:
        # FIXME: Fix the handling of unknown rank
        dims = [-1]
    else:
        dims = [dim.size for dim in tensor_proto.tensor_shape.dim]
    if batching_enabled:
        tensor_dict['dims'] = dims[1:]
    else:
        tensor_dict['dims'] = dims

    return tensor_dict


def _validate_datatype(tf_dtype, triton_datatype, tensor_name):
    dtype_dict = {value: key for (key, value) in types_pb2.DataType.items()}
    if triton_datatype != TF_STRING_TO_TRITON[dtype_dict[tf_dtype]]:
        raise pb_utils.TritonModelException(
            f" Mismatch between datatype for tensor '" + tensor_name +
            "', expected '" + TF_STRING_TO_TRITON[dtype_dict[tf_dtype]] +
            "', got '" + triton_datatype)


def _validate_dims(tf_shape, triton_dims, batching_enabled, tensor_name):
    if tf_shape.unknown_rank:
        return

    index = 0
    offset = 1 if batching_enabled else 0
    if len(tf_shape.dim) != (offset + len(triton_dims)):
        raise pb_utils.TritonModelException(
            f" Mismatch in the number of dimension with the model for tensor '"
            + tensor_name + "', expected " + str(len(tf_shape.dim) - offset) +
            ", got " + str(len(triton_dims)))

    for dim in tf_shape.dim:
        if index == 0 and batching_enabled:
            if dim.size != -1:
                raise pb_utils.TritonModelException(
                    f" The first dimension of a batching model should be dynamic, "
                    "however, got shape of first dimension in model for tensor '"
                    + tensor_name + "' as " + str(dim.size))
        else:
            if dim.size != triton_dims[index - offset]:
                raise pb_utils.TritonModelException(
                    f" Mismatch in " + str(index - offset) +
                    "th dimension for tensor '" + tensor_name + "', expected " +
                    str(dim.size) + ", got " + str(triton_dims[index - offset]))
        index = index + 1


def _validate_model_config(model_config, signature_def):
    signature_supports_batching = _get_batching_hint_from_signature(
        signature_def)
    if (not signature_supports_batching) and (model_config['max_batch_size'] !=
                                              0):
        raise pb_utils.TritonModelException(
            f" The model signature does not support batching, yet model config"
            " has max_batch_size set to '" +
            str(model_config['max_batch_size']) + "'")

    batching_enabled = model_config['max_batch_size'] != 0

    if model_config['platform'] != 'tensorflow_savedmodel':
        raise pb_utils.TritonModelException(
            f"[INTERNAL]: The platform field for using this model should be set to"
            " \'tensorflow_savedmodel\' in model config, got '" +
            model_config['platform'] + "'")
    if model_config['batch_input']:
        raise pb_utils.TritonModelException(
            f"The platform model '" + model_config['platform'] +
            "' does not support model with batch_input")
    if model_config['batch_output']:
        raise pb_utils.TritonModelException(
            f"The platform model '" + model_config['platform'] +
            "' does not support model with batch_output")

    # Validate input tensors
    input_tensor_info = signature_def.inputs
    config_input_names = [input['name'] for input in model_config['input']]
    for input_name in input_tensor_info.keys():
        if input_name not in config_input_names:
            raise pb_utils.TritonModelException(
                f" Missing input tensor configuration for tensor '" +
                input_name + "'")
    for input in model_config['input']:
        config_input_name = input['name']
        if config_input_name not in input_tensor_info.keys():
            supported_names = ""
            for valid_name in input_tensor_info.keys():
                supported_names = supported_names + ";" + valid_name
            raise pb_utils.TritonModelException(
                f" No input tensor with name '" + config_input_name +
                "', only supported input names are " + supported_names)
        _validate_datatype(input_tensor_info[config_input_name].dtype,
                           input['data_type'], config_input_name)
        _validate_dims(input_tensor_info[config_input_name].tensor_shape,
                       input['dims'], batching_enabled, config_input_name)

    # Validate output tensors
    output_tensor_info = signature_def.outputs
    for output in model_config['output']:
        config_output_name = output['name']
        if config_output_name not in output_tensor_info.keys():
            supported_names = ""
            for valid_name in output_tensor_info.keys():
                supported_names = supported_names + ";" + valid_name
            raise pb_utils.TritonModelException(
                f" No output tensor with name '" + config_output_name +
                "', only supported output names are " + supported_names)

        _validate_datatype(output_tensor_info[config_output_name].dtype,
                           output['data_type'], config_output_name)
        _validate_dims(output_tensor_info[config_output_name].tensor_shape,
                       output['dims'], batching_enabled, config_output_name)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config, args):

        config = auto_complete_model_config.as_dict()

        if config['platform'] != 'tensorflow_savedmodel':
            raise pb_utils.TritonModelException(
                f"[INTERNAL]: The platform field for using this model should be set to"
                " \'tensorflow_savedmodel\' in model config, got '" +
                config['platform'] + "'")
        if config['batch_input']:
            raise pb_utils.TritonModelException(
                f"The platform model '" + config['platform'] +
                "' does not support model with batch_input")
        if config['batch_output']:
            raise pb_utils.TritonModelException(
                f"The platform model '" + config['platform'] +
                "' does not support model with batch_output")

        savedmodel_path = _get_savedmodel_path(args, config)

        if savedmodel_path is None:
            raise pb_utils.TritonModelException(
                f"[INTERNAL]: The path to the framework model should be"
                " provided")

        batching_enabled = False
        if config['max_batch_size'] != 0:
            batching_enabled = True

        _, signature_def = _get_signature_def(savedmodel_path, config)

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

        config_input_names = [input['name'] for input in config['input']]
        config_output_names = [output['name'] for output in config['output']]

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
            if config['max_batch_size'] == 0:
                auto_complete_model_config.set_max_batch_size(4)
            auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

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

        savedmodel_path = _get_savedmodel_path(args, model_config)

        self.model_name = args['model_name']
        self.logger = pb_utils.Logger
        self.logger.log_info("Initializing platform model for " +
                             self.model_name)

        if args['model_instance_kind'] != 'KIND_CPU':
            self.logger.log_warn(
                "GPU instances are not supported by this backend. Falling back to KIND_CPU for "
                + self.model_name)

        tag_set, signature_def = _get_signature_def(savedmodel_path,
                                                    model_config)
        _validate_model_config(model_config, signature_def)

        self.signature_def = signature_def
        self.input_tensor_info = self.signature_def.inputs
        output_tensor_info = self.signature_def.outputs

        # Get the input output names from model config
        self.input_names = [input['name'] for input in model_config['input']]
        self.output_names = [
            output['name'] for output in model_config['output']
        ]

        # Get the output tensor names
        self.output_tensor_names = [
            output_tensor_info[output_name].name
            for output_name in self.output_names
        ]

        # load the session model
        # FIXME Add more configuration options for the model.
        sess_config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=_parse_num_inter_threads(model_config),
            intra_op_parallelism_threads=_parse_num_intra_threads(model_config),
            use_per_session_threads=_parse_use_per_session_thread(model_config))
        self.tf_session = session.Session(graph=tf.Graph(), config=sess_config)
        loader.load(self.tf_session, [tag_set], savedmodel_path)

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

        # FIXME: Instead of iterating through each request, run
        # the inference as a single batch.
        for request in requests:
            # Prepare the input feed for the model.
            input_feed_dict = {}
            for input_name in self.input_names:
                input_feed_dict[self.input_tensor_info[input_name].
                                name] = pb_utils.get_input_tensor_by_name(
                                    request, input_name).as_numpy()

            outputs = self.tf_session.run(self.output_tensor_names,
                                          feed_dict=input_feed_dict)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensors = []
            for i, output in enumerate(outputs):
                output_tensors.append(
                    pb_utils.Tensor(self.output_names[i], output))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        if self.tf_session is not None:
            self.tf_session.close
        self.logger.log_info('Removing platform model for ' + self.model_name)
