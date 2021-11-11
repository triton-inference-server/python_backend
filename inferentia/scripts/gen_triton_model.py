# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os

def parse_io_tensors(tensors):
    tensors_dict = {}
    for t in [t for tensor in tensors for t in tensor]:
        name, datatype, shape_str = t.split(",")
        shape = [int(i) for i in shape_str.split("x")]
        tensors_dict[name] = [datatype, shape]

    return tensors_dict

def get_parameter_spec(key1, value):
    param_spec = "parameters: {{key: \"{}\", value: {{string_value: \"{}\"}}}} \n".format(
        key1, value)

    return param_spec

def create_modelconfig(model_name, max_batch_size, inputs, outputs,
                       compiled_model_path, nc_start_idx, nc_end_idx,
                       threads_per_core, instance_count):
    config = "name: \"{}\"\n".format(model_name)
    config += "backend: \"python\"\n"
    config += "max_batch_size: {}\n".format(max_batch_size)
    for input_name in inputs.keys():
        data_type, shape = inputs[input_name]
        config +='''
input [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(input_name, "TYPE_" + data_type, shape)
    for output_name in outputs.keys():
        data_type, shape = outputs[output_name]
        config +='''
output [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(output_name, "TYPE_" + data_type, shape)
    config += '''
instance_group [
    {{
        kind: KIND_MODEL
        count: {}
    }}
]\n'''.format(instance_count)
    config += get_parameter_spec("COMPILED_MODEL", compiled_model_path)
    config += get_parameter_spec("NEURON_CORE_START_INDEX", nc_start_idx)
    config += get_parameter_spec("NEURON_CORE_END_INDEX", nc_end_idx)
    config += get_parameter_spec("NUM_THREADS_PER_CORE", threads_per_core)
    return config

def get_model_license():
    lic = '''# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    '''
    return lic

def get_initialize_impl():
    init_impl = '''
    def _validate_and_get_index(self, name):
        parts = name.split('__')
        if len(parts) != 2:
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index>, got {}"
                .format(name))

        if not parts[1].isnumeric():
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index> where <index> should be numeric, got {}"
                .format(name))

        return int(parts[1])

    def _validate_input_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.input_dict:
                raise pb_utils.TritonModelException(
                    "input corresponding to index {} not found".format(i))

    def _validate_output_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.output_dict:
                raise pb_utils.TritonModelException(
                    "output corresponding to index {} not found".format(i))

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

        if (len(model_config['instance_group']) != 1):
            raise pb_utils.TritonModelException(
                "this model supports only a single instance group, got {}".
                format(len(model_config['instance_group'])))

        instance_group_config = model_config['instance_group'][0]
        instance_count = instance_group_config['count']

        instance_idx = 0
        if instance_count > 1:
            instance_name_parts = args['model_instance_name'].split("_")
            if not instance_name_parts[-1].isnumeric():
                raise pb_utils.TritonModelException(
                    "internal error: the model instance name should end with \'_<instance_idx>\', got {}"
                    .format(args['model_instance_name']))
            instance_idx = int(instance_name_parts[-1])

        self.input_dict = {}
        expected_input_count = 0
        for config_input in model_config['input']:
            index = self._validate_and_get_index(config_input['name'])
            self.input_dict[index] = [
                config_input['name'], config_input['data_type'],
                config_input['dims']
            ]
            expected_input_count += 1
        self._validate_input_dict(expected_input_count)

        self.output_dict = {}
        for config_output in model_config['output']:
            index = self._validate_and_get_index(config_output['name'])
            self.output_dict[index] = [
                config_output['name'], config_output['data_type'],
                config_output['dims']
            ]

        params = model_config['parameters']
        compiled_model = params['COMPILED_MODEL']['string_value']
        nc_start_idx = int(params['NEURON_CORE_START_INDEX']['string_value'])
        nc_end_idx = int(params['NEURON_CORE_END_INDEX']['string_value'])
        if nc_end_idx < nc_start_idx:
            raise pb_utils.TritonModelException(
                "the neuron core end index should be greater than or equal to the start index")

        threads_per_core = int(params['NUM_THREADS_PER_CORE']['string_value'])
        if threads_per_core < 1:
            raise pb_utils.TritonModelException(
                "the number of threads per core should be greater than or equal to 1")
        num_threads = (nc_end_idx - nc_start_idx + 1) * threads_per_core

        total_core_count = nc_end_idx - nc_start_idx + 1
        if (instance_count > total_core_count):
            raise pb_utils.TritonModelException(
                    "can not distribute {} triton model instances to {} neuron cores"
                    .format(instance_count, total_core_count))
        cores_per_instance = total_core_count // instance_count
        adjusted_nc_start_idx = (instance_idx *
                                 cores_per_instance) + nc_start_idx
        cores_range = '{}-{}'.format(
            adjusted_nc_start_idx,
            (adjusted_nc_start_idx + cores_per_instance - 1))
        os.environ["NEURON_RT_VISIBLE_CORES"] = cores_range

        consumed_cores_list = [i for i in range(cores_per_instance)]

        self.model_neuron = torch.neuron.DataParallel(
            torch.jit.load(compiled_model), device_ids=consumed_cores_list)
        self.model_neuron.num_workers = num_threads

'''
    return init_impl

def get_execute_impl():
    exec_impl = '''
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

        for request in requests:
            inputs = []
            for i in range(len(self.input_dict)):
                name, dt, shape = self.input_dict[i]
                tensor = pb_utils.get_input_tensor_by_name(request,
                                                           name).as_numpy()
                inputs.append(torch.as_tensor(tensor))

            results = self.model_neuron(*inputs)

            output_tensors = []
            for i in self.output_dict.keys():
                name, dt, shape = self.output_dict[i]
                output_tensor = pb_utils.Tensor(
                    name, results[i].numpy().astype(
                        pb_utils.triton_string_to_numpy(dt)))

                output_tensors.append(output_tensor)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)

        return responses
'''
    return exec_impl

def get_finalize_impl():
    finalize_impl = '''
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

'''
    return finalize_impl

def get_triton_python_model_impl():
    triton_pmi = '''
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    '''

    triton_pmi += get_initialize_impl()
    triton_pmi += get_execute_impl()
    triton_pmi += get_finalize_impl()

    return triton_pmi

def create_model_file():
    triton_model = get_model_license()
    triton_model += '''
from concurrent import futures
import json
import numpy as np
import os
import sys
import torch
import torch.neuron
import triton_python_backend_utils as pb_utils
    '''

    triton_model += get_triton_python_model_impl()

    return triton_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version',
                        type=int,
                        default=1,
                        help='The version of the model')
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=0,
        help='The maximum batch size for the model being generated')
    parser.add_argument(
        '--triton_input',
        type=str,
        required=True,
        action='append',
        nargs="*",
        help='''The name, datatype and shape of the model input in
        format <input_name>,<triton_datatype>,<shape>. This
        option can be provided multiple times for multiple
        inputs. For example, to provide a FP16 input with
        shape [1,384] specify the following: INPUT0,FP16,1x384.''')
    parser.add_argument(
        '--triton_output',
        type=str,
        required=True,
        action='append',
        nargs="*",
        help='''The name, datatype and shape of the model output in
        format <output_name>,<triton_datatype>,<shape>. This
        option can be provided multiple times for multiple
        outputs. For example, to provide a FP16 output with
        shape [1,384] specify the following: OUTPUT0,FP16,1x384.''')
    parser.add_argument('--compiled_model',
                        type=str,
                        required=True,
                        help='Fullpath to the compiled model')
    parser.add_argument('--neuron_core_range',
                        type=str,
                        required=True,
                        help='''The range of neuron core indices
                        where the model needs to be loaded. The
                        range should be specified in format
                        <start_idx>:<end_idx>. For example to
                        load model on neuron cores (0-7), specify
                        the following: 0:7. NOTE: when using
                        multiple triton model instances the neuron
                        cores will get equally distributed. Assuming
                        the instance count is 4, Instance0 will get
                        loaded on cores 0:1, Instance1 will get loaded
                        on cores 2:3, Instance2 will get loaded on
                        cores 4:5 and Instance 3 will get loaded on
                        cores 6:7''')
    parser.add_argument('--threads_per_core',
                        type=int,
                        default=1,
                        help='The number of threads per neuron core.')
    parser.add_argument('--triton_model_instance_count',
                        type=int,
                        default=1,
                        help='The number of triton model instances.')
    parser.add_argument('--triton_model_dir',
                        type=str,
                        required=True,
                        help='''Path to the triton model
                        directory where script will generate
                        config.pbtxt and model.py''')
    FLAGS, unparsed = parser.parse_known_args()

    inputs = parse_io_tensors(FLAGS.triton_input)
    outputs = parse_io_tensors(FLAGS.triton_output)

    nc_start_idx, nc_end_idx = [int(i) for i in FLAGS.neuron_core_range.split(":")]

    model_version_dir = FLAGS.triton_model_dir + "/" + str(FLAGS.model_version)
    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    model_name = os.path.basename(FLAGS.triton_model_dir)
    mc = create_modelconfig(model_name, FLAGS.max_batch_size, inputs, outputs,
                            FLAGS.compiled_model, nc_start_idx, nc_end_idx,
                            FLAGS.threads_per_core, FLAGS.triton_model_instance_count)
    with open(FLAGS.triton_model_dir + "/config.pbtxt", "w") as config_file:
        config_file.write(mc)

    mf = create_model_file()
    with open(FLAGS.triton_model_dir + "/1/model.py", "w") as model_file:
        model_file.write(mf)
