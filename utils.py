#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from logging import getLogger
from typing import Optional, Union, List
import functools

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS
from qlinear import ExLlamaV2DeviceTensors, QuantLinear

logger = getLogger(__name__)


def get_layers(module: nn.Module,
               layers=[Conv1D, nn.Conv2d, nn.Linear],
               prefix: Optional[str] = None,
               skip: Optional[List] = None,
               name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
        module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
        `Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    if skip is None:
        skip = []
    for layer in layers:
        if isinstance(module, layer):
            if prefix is not None:
                if name.startswith(prefix) and all(pattern not in name for pattern in skip):
                    return {name: module}
            else:
                if all(pattern not in name for pattern in skip):
                    return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            get_layers(child,
                       layers=layers,
                       prefix=prefix,
                       skip=skip,
                       name=name + "." + name1 if name != "" else name1))
    return res


def get_block_name_with_pattern(model: nn.Module):
    """
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(pattern_candidate in name for name in modules_names):
            return pattern_candidate
    raise ValueError(
        "Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`"
    )


def get_preceding_modules(model: nn.Module,
                          module_name: str,
                          reverse: bool = False):
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model: nn.Module,
                               module_name: str,
                               name: str = ""):
        nonlocal stop_adding
        modules = model.named_children()
        if reverse:
            modules = reversed(list(modules))
        for name_bis, child in modules:
            new_name = name + "." + name_bis if name != "" else name_bis
            if new_name == module_name:
                stop_adding = True
                break
            _get_preceding_modules(child, module_name, name=new_name)
        if not stop_adding:
            previous_module_name.append(name)
        return previous_module_name

    return _get_preceding_modules(model, module_name)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_seqlen(model: nn.Module):
    if hasattr(model, "config"):
        model_config = model.config.to_dict()
        if any(k in model_config for k in SEQLEN_KEYS_TRANFORMERS):
            for key in SEQLEN_KEYS_TRANFORMERS:
                if key in model_config:
                    return model_config[key]
    logger.info(
        "We couldn't get the model sequence length. Setting it to 2048. You can overwrite this value by passing `model_seqlen` in` GPTQQuantizer`"
    )
    return 2048


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def post_init(model):
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if isinstance(submodule, QuantLinear):
            model_uses_exllamav2 = True
            device = submodule.q_weight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed,
                                      fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(
                device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if isinstance(submodule, QuantLinear):
                device = submodule.q_weight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model
