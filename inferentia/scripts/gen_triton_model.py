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
                       compiled_model_path, avbl_neuron_cores_count,
                       threads_per_core, batch_per_thread):
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
    config += "instance_group [ { kind: KIND_MODEL }]\n"
    config += get_parameter_spec("COMPILED_MODEL", compiled_model_path)
    config += get_parameter_spec("AVAIL_NEURONCORES", avbl_neuron_cores_count)
    config += get_parameter_spec("NUM_THREADS_PER_PREDICTOR", threads_per_core)
    config += get_parameter_spec("BATCH_PER_THREAD", batch_per_thread)
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

def get_neuron_simple_data_parallel_impl():
    neuron_sdpi = '''\n
class NeuronSimpleDataParallel():

    def __init__(self, model_file, num_neuron_cores, num_threads, batch_size):
        # Construct a list of models
        self.num_neuron_cores = num_neuron_cores
        self.batch_size = batch_size
        self.num_threads = num_threads

        class SimpleWrapper():

            def __init__(self, model):
                self.model = model

            def eval(self):
                self.model.eval()

            def train(self):
                self.model.train()

            def __call__(self, *inputs):
                results = self.model(*inputs)
                # Make the output iterable - if it is not already a tuple or list
                if not isinstance(results, tuple) or isinstance(results, list):
                    results = [results]

                return results

        self.models = [
            SimpleWrapper(torch.jit.load(model_file))
            for i in range(self.num_threads)
        ]
        nc_env = ','.join(['1'] * num_neuron_cores)
        os.environ['NEURONCORE_GROUP_SIZES'] = nc_env

        self.executor = futures.ThreadPoolExecutor(max_workers=self.num_threads)

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train()

    def __call__(self, *args):

        args_per_core = [None for i in range(self.num_threads)]
        # Split args
        for a in args:

            # Based on batch size for arg
            step_size = self.batch_size
            for i in range(self.num_threads):
                # Append a slice of a view
                start = i * step_size
                end = (i + 1) * step_size

                # Slice
                args_per_core[i] = []
                for input in a:
                    args_per_core[i].append(input[start:end])
        # Call each core with their split and wait to complete
        running = {
            self.executor.submit(self.models[idx], *args_per_core[idx]): idx
            for idx in range(self.num_threads)
        }

        results = [None] * self.num_threads

        for future in futures.as_completed(running):
            idx = running[future]
            results[idx] = future.result()

        return results
    
    '''
    return neuron_sdpi

def get_initialize_impl():
    init_impl = '''
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

        self.input_dict = {}
        for config_input in model_config['input']:
            self.input_dict[config_input['name']] = [
                config_input['data_type'], config_input['dims']
            ]

        self.output_dict = {}
        for config_output in model_config['output']:
            self.output_dict[config_output['name']] = [
                config_output['data_type'], config_output['dims']
            ]

        params = model_config['parameters']
        compiled_model = params['COMPILED_MODEL']['string_value']
        avbl_neuron_cores_count = int(
            params['AVAIL_NEURONCORES']['string_value'])
        threads_per_core = int(
            params['NUM_THREADS_PER_PREDICTOR']['string_value'])
        batch_per_thread = int(params['BATCH_PER_THREAD']['string_value'])
        self.num_threads = avbl_neuron_cores_count * threads_per_core
        self.model_neuron = NeuronSimpleDataParallel(compiled_model,
                                                     avbl_neuron_cores_count,
                                                     self.num_threads,
                                                     batch_per_thread)

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
            num_threads = self.num_threads
            inputs = []
            for name in self.input_dict.keys():
                tensor = pb_utils.get_input_tensor_by_name(request,
                                                           name).as_numpy()
                inputs.append(torch.LongTensor(tensor))
            results = self.model_neuron(inputs)

            output_tensors = []
            for name in self.output_dict.keys():
                result_shards = []
                for i in range(num_threads):
                    result_shards.append(results[i][len(output_tensors)])
                merged_result = np.concatenate(result_shards, axis=0)
                dt, shape = self.output_dict[name]
                output_tensor = pb_utils.Tensor(name,
                                           merged_result.astype(pb_utils.triton_string_to_numpy(dt)))

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

    triton_model += get_neuron_simple_data_parallel_impl()
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
    parser.add_argument('--avbl_neuron_cores_count',
                        type=int,
                        default=4,
                        help='The number of available neuron cores')
    parser.add_argument('--threads_per_core',
                        type=int,
                        default=1,
                        help='The number of threads per neuron core')
    parser.add_argument('--batch_per_thread',
                        type=int,
                        default=1,
                        help='The batch size per threads')
    parser.add_argument('--triton_model_dir',
                        type=str,
                        required=True,
                        help='''Path to the triton model
                        directory where script will generate
                        config.pbtxt and model.py''')
    FLAGS, unparsed = parser.parse_known_args()

    inputs = parse_io_tensors(FLAGS.triton_input)
    outputs = parse_io_tensors(FLAGS.triton_output)

    model_version_dir = FLAGS.triton_model_dir + "/" + str(FLAGS.model_version)
    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    model_name = os.path.basename(FLAGS.triton_model_dir)
    mc = create_modelconfig(model_name, FLAGS.max_batch_size, inputs, outputs,
                            FLAGS.compiled_model, FLAGS.avbl_neuron_cores_count,
                            FLAGS.threads_per_core, FLAGS.batch_per_thread)
    with open(FLAGS.triton_model_dir + "/config.pbtxt", "w") as config_file:
        config_file.write(mc)

    mf = create_model_file()
    with open(FLAGS.triton_model_dir + "/1/model.py", "w") as model_file:
        model_file.write(mf)
