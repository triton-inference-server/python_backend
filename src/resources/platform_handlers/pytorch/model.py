#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import json
import os

try:
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError(
        "Missing/Incomplete PyTorch package installation... (Did you install PyTorch?)"
    ) from error

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


def _get_model_path(config):
    filenames = ["model.py", "model.pt"]
    if config["default_model_filename"]:
        filenames.insert(0, config["default_model_filename"])
    for filename in filenames:
        model_path = os.path.join(pb_utils.get_model_dir(), filename)
        if os.path.exists(model_path):
            return model_path
    raise pb_utils.TritonModelException(
        "No model found in " + pb_utils.get_model_dir() + "/" + str(filenames)
    )


def _get_model_data_path(model_path):
    data_path_extensions = [".pt"]
    model_path_no_extension = model_path[: -(len(model_path.split(".")[-1]) + 1)]
    for extension in data_path_extensions:
        data_path = model_path_no_extension + extension
        if os.path.exists(data_path):
            return data_path
    # data file not provided
    return ""


def _is_py_class_model(model_path):
    return model_path[-3:] == ".py"


def _import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_model_class_from_module(module):
    names = dir(module)
    for name in names:
        attr = getattr(module, name)
        try:
            if issubclass(attr, torch.nn.Module):
                return attr
        except TypeError:
            # attr may not be a class
            pass
    raise pb_utils.TritonModelException("Cannot find a subclass of torch.nn.Module")


def _parse_io_config(io_config):
    io = []
    for conf in io_config:
        io.append({"name": conf["name"]})
    return io


def _get_device_name(kind, device_id):
    if kind == "GPU":
        return "cuda:" + device_id
    if kind == "CPU":
        return "cpu"
    # unspecified device
    return ""


def _get_device(kind, device_id, model):
    device_name = _get_device_name(kind, device_id)
    if device_name == "":
        for param in model.parameters():
            return param.device
        raise pb_utils.TritonModelException("Cannot determine model device")
    return torch.device(device_name)


def _set_torch_parallelism(config):
    log_msg = ""
    parallelism_settings = ["NUM_THREADS", "NUM_INTEROP_THREADS"]
    for setting in parallelism_settings:
        val = "1"
        if setting in config["parameters"]:
            val = config["parameters"][setting]["string_value"]
        getattr(torch, "set_" + setting.lower())(int(val))
        log_msg += setting + " = " + val + "; "
    return log_msg


def _get_torch_compile_params(config):
    params = {}
    if "TORCH_COMPILE_OPTIONAL_PARAMETERS" in config["parameters"]:
        val = config["parameters"]["TORCH_COMPILE_OPTIONAL_PARAMETERS"]["string_value"]
        params = json.loads(val)
        if "model" in params:
            raise pb_utils.TritonModelException(
                "'model' is not an optional parameter for 'torch.compile'"
            )
    return params


def _gather_torch_tensors(scatter_tensors):
    gather_tensors = []
    sections = []
    for i in range(len(scatter_tensors)):
        tensors = scatter_tensors[i]
        for j in range(len(tensors)):
            tensor = tensors[j]
            if j < len(gather_tensors):
                # add to existing tensor
                gather_tensors[j] = torch.cat((gather_tensors[j], tensor), 0)
            else:
                # start a new tensor
                gather_tensors.append(tensor)
        # record section
        section_length = tensors[0].size()[0]
        sections.append(section_length)
    return gather_tensors, sections


def _scatter_torch_tensors(gather_tensors, sections):
    scatter_tensors = []
    for j in range(len(gather_tensors)):
        scatter_tensor = torch.split(gather_tensors[j], sections)
        for i in range(len(scatter_tensor)):
            tensor = scatter_tensor[i]
            if i < len(scatter_tensors):
                # add to existing response
                scatter_tensors[i].append(tensor)
            else:
                # start a new response
                scatter_tensors.append([tensor])
    return scatter_tensors


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

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
        self._model_name = args["model_name"]
        for_model = "for '" + self._model_name + "'"
        self._logger = pb_utils.Logger
        self._logger.log_info("Initializing model instance " + for_model)

        self._model_config = json.loads(args["model_config"])
        self._kind = args["model_instance_kind"]
        self._device_id = args["model_instance_device_id"]
        self._support_batching = self._model_config["max_batch_size"] > 0
        self._inputs = _parse_io_config(self._model_config["input"])
        self._outputs = _parse_io_config(self._model_config["output"])

        setting_msg = _set_torch_parallelism(self._model_config)
        self._logger.log_verbose(
            "Torch parallelism settings " + for_model + ": " + setting_msg
        )

        self._infer_mode = torch.inference_mode(mode=True)
        self._infer_mode.__enter__()

        params = _get_torch_compile_params(self._model_config)
        self._logger.log_verbose(
            "'torch.compile' optional parameter(s) " + for_model + ": " + str(params)
        )
        if self._support_batching:
            self._gather = torch.compile(_gather_torch_tensors, **params)
            self._scatter = torch.compile(_scatter_torch_tensors, **params)

        model_path = _get_model_path(self._model_config)
        if not _is_py_class_model(model_path):
            self._logger.log_info("Loading '" + self._model_name + "' as TorchScript")
            self._model = torch.jit.load(model_path)
            self._device = _get_device(self._kind, self._device_id, self._model)
            self._model.to(self._device)
            self._model.eval()
            return

        self._model_module = _import_module_from_path(self._model_name, model_path)
        self._model_class = _get_model_class_from_module(self._model_module)
        self._raw_model = self._model_class()
        self._device = _get_device(self._kind, self._device_id, self._raw_model)
        data_path = _get_model_data_path(model_path)
        if data_path != "":
            self._raw_model.load_state_dict(
                torch.load(data_path, map_location=self._device)
            )
        else:
            self._logger.log_info("Model parameter file not found " + for_model)
        self._raw_model.to(self._device)
        self._raw_model.eval()
        self._model = torch.compile(self._raw_model, **params)

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

        requests_tensors = []
        for request in requests:
            tensors = []
            for io in self._inputs:
                tensor = pb_utils.get_input_tensor_by_name(
                    request, io["name"]
                ).to_dlpack()
                tensor = torch.from_dlpack(tensor).to(self._device)
                tensors.append(tensor)
            requests_tensors.append(tensors)

        sections = None
        if self._support_batching:
            requests_tensors, sections = self._gather(requests_tensors)
            requests_tensors = [requests_tensors]

        responses_tensors = []
        for input_tensors in requests_tensors:
            output_tensors = self._model(*input_tensors)
            if not isinstance(output_tensors, tuple) and not isinstance(
                output_tensors, list
            ):
                output_tensors = [output_tensors]
            responses_tensors.append(output_tensors)

        if self._support_batching:
            responses_tensors = self._scatter(responses_tensors[0], sections)

        for response_tensors in responses_tensors:
            output_tensors = []
            for i in range(len(self._outputs)):
                io = self._outputs[i]
                tensor = response_tensors[i].detach()
                tensor = pb_utils.Tensor.from_dlpack(io["name"], tensor)
                output_tensors.append(tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self._logger.log_info("Removing model instance for '" + self._model_name + "'")
        self._infer_mode.__exit__(exc_type=None, exc_value=None, traceback=None)
