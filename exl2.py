# Copyright 2023 HuggingFace Inc. team and GPTQ and exllama authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from logging import getLogger
from typing import Any, List, Optional, Union

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from accelerate import (
    Accelerator,
    cpu_offload_with_hook,
)
from accelerate.hooks import remove_hook_from_module

from data import get_dataset, prepare_dataset
from utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    recurse_getattr,
)
from gptq import GPTQ
from optimizer import optimize
from qlinear import QuantLinear
from qparam import QParams, qparams_options, qparams_headoptions

logger = getLogger(__name__)


def do_measure(quantizer):
    original_weight = quantizer.layer.weight.data.clone()
    origin_layer_outputs = []
    for inp in quantizer.inps:
        layer_output = quantizer.layer(inp)
        origin_layer_outputs.append(
            layer_output.view(-1, layer_output.shape[-1]).float())
    result = {"numel": quantizer.rows * quantizer.columns, "options": []}
    logger.info("Measuring ...")
    for qp in qparams_options:
        quantizer.configure(qp.group_size, qp.bits, qp.bits_prop,
                            qp.scale_bits)
        quantizer.quantize()
        desc = qp.desc
        bpw = qp.bpw((quantizer.rows, quantizer.columns))
        dsum = 0.0
        dcount = 0.0
        for j, inp in enumerate(quantizer.inps):
            layer_output = quantizer.layer(inp)
            layer_output = layer_output.view(-1, layer_output.shape[-1])
            rfn = torch.linalg.norm(
                layer_output.float() - origin_layer_outputs[j],
                'fro') / torch.linalg.norm(layer_output.float(), 'fro')
            dsum += rfn * inp.shape[0]
            dcount += inp.shape[0]
        option = {
            "desc": desc,
            "bpw": bpw,
            "total_bits": quantizer.rows * quantizer.columns * bpw,
            "err": dsum / dcount,
            "qparams": qp.get_dict()
        }
        logger.info(
            f" -- {desc:30} {bpw:2.2f} bpw    rfn_error: {option['err']:2.5f}")
        result["options"].append(option)
        quantizer.layer.weight.data = original_weight.clone()
    return result


class Exl2Quantizer(object):
    r"""
    A simple API for EXL2 Quantization
    """

    def __init__(
        self,
        bits: float = 4,
        head_bits: int = 8,
        dataset: Optional[Union[List[str], str]] = None,
        cache_examples_on_gpu: bool = False,
        model_seqlen: int = None,
        damp_percent: float = 0.01,
        true_sequential: bool = True,
        block_name_to_quantize: Optional[str] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.bits = bits
        self.head_bits = head_bits
        self.dataset = dataset
        self.cache_examples_on_gpu = cache_examples_on_gpu
        self.damp_percent = damp_percent
        self.model_seqlen = model_seqlen
        self.true_sequential = true_sequential
        self.block_name_to_quantize = block_name_to_quantize
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.quant_method = 'EXL2'
        self.module_name_preceding_first_block = None
        self.lm_head_name = None
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def convert_model(self, model: nn.Module):
        """
        Convert the model to a Quip model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = list(
            get_layers(model, prefix=block_name).keys())
        module_names_after_last_block = get_preceding_modules(
            model, self.block_name_to_quantize, reverse=True)
        layers_to_be_replaced += [module_names_after_last_block[0]]
        self._replace_by_quant_layers(model, layers_to_be_replaced)
        return model

    def get_no_split_module_classes(self, model):
        """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """

        block_class_name = recurse_getattr(
            model, self.block_name_to_quantize)[0].__class__.__name__
        no_split_module_classes = [block_class_name]
        return no_split_module_classes

    def _replace_by_quant_layers(self,
                                 module: nn.Module,
                                 names: List[str],
                                 name: str = ""):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                bias = hasattr(layer, "bias") and layer.bias is not None
                new_layer = QuantLinear(in_features, out_features, bias)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(
                child, names, name + "." + name1 if name != "" else name1)

    @torch.inference_mode()
    def quantize_model(self, model: nn.Module, tokenizer: Any):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (`Any`):
                The tokenizer to use in order to prepare the dataset. You can pass either:
                    - A custom tokenizer object.
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        Returns:
            `nn.Module`: The quantized model
        """
        model.eval()
        # For Transformer model
        has_config = False
        has_device_map = False
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        if hasattr(model, "hf_device_map"):
            devices = list(model.hf_device_map.values())
            if "disk" in devices:
                raise ValueError(
                    "disk offload is not supported with QUiP quantization")
            if "cpu" in devices and len(model.hf_device_map) > 1:
                logger.info(
                    "Cpu offload is not recommended. There might be some issues with the memory"
                )
                hook = None
                for name, device in model.hf_device_map.items():
                    if device == "cpu":
                        module = recurse_getattr(model, name)
                        remove_hook_from_module(module, recurse=True)
                        module, hook = cpu_offload_with_hook(
                            module, prev_module_hook=hook)
            # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
            has_device_map = True

        if self.model_seqlen is None:
            self.model_seqlen = get_seqlen(model)

        if isinstance(tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                    with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.
                    For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                )
        # generate measurements
        measure = self._quantize_model_helper(model,
                                             tokenizer,
                                             measure=True,
                                             has_device_map=has_device_map)
        # find best measure
        optimize(measure, self.bits)
        # real quant
        qweights = self._quantize_model_helper(model,
                                              tokenizer,
                                              measure=False,
                                              qparams=measure)
        # pack model
        self.pack_model(model=model, quantizers=qweights)

        model.is_quantized = True
        if has_config:
            model.config.use_cache = use_cache

        torch.cuda.empty_cache()
        return model

    def _quantize_model_helper(self,
                               model: nn.Module,
                               tokenizer: Any,
                               measure: bool = True,
                               qparams: dict = None,
                               has_device_map: bool = False):
        device = get_device(model)

        # Step 1: Prepare the data
        if self.dataset is None:
            raise ValueError(
                "You need to pass `dataset` in order to quantize your model")
        elif isinstance(self.dataset, str):
            dataset = get_dataset(self.dataset,
                                  tokenizer,
                                  nsamples=128 if not measure else 16,
                                  seqlen=self.model_seqlen,
                                  split="train")
        elif isinstance(self.dataset, list):
            dataset = [
                tokenizer(data, return_tensors="pt") for data in self.dataset
            ]
        else:
            raise ValueError(
                "You need to pass a list of string or a string for `dataset`")

        dataset = prepare_dataset(dataset,
                                  pad_token_id=self.pad_token_id,
                                  batch_size=self.batch_size)

        # Step 2: get the input of the 1st block
        # To do that, we need to put the modules preceding the first block on the same device as the first bloc.
        # Then we run the model and it will stop at the first bloc as we added a prehook that raise an Exception after storing the inputs.

        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []

        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)

        self.module_name_preceding_first_block = get_preceding_modules(
            model, self.block_name_to_quantize)

        blocks = recurse_getattr(model, self.block_name_to_quantize)

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(
                        f"Module {module_name} was not found in model")
                module = module.to(0)
            blocks[0] = blocks[0].to(0)

        def store_input_hook(_, input, *args):
            kwargs = args[0]
            input = input[0]
            if input is None:
                if "hidden_states" in kwargs:
                    input = kwargs["hidden_states"]
                else:
                    raise ValueError("No input value found in the foward pass")
            layer_inputs.append(input)
            other_kwargs = {}
            for k, v in kwargs.items(
            ):  # make sure other arguments also be captured
                if k not in ["hidden_states"]:
                    other_kwargs[k] = v
            layer_input_kwargs.append(other_kwargs)
            raise ValueError

        handle = blocks[0].register_forward_pre_hook(store_input_hook,
                                                     with_kwargs=True)
        for data in dataset:
            for k, v in data.items():
                # put the data on gpu, we won't put them back to cpu
                data[k] = v.to(0)
            try:
                model(**data)
            except ValueError:
                pass

        handle.remove()
        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(
                        f"Module {module_name} was not found in model")
                module.to(device)

        torch.cuda.empty_cache()

        # Step 3: Quantization
        result = {}
        for i, block in enumerate(
                tqdm(
                    blocks,
                    desc=f"Quantizing {self.block_name_to_quantize} blocks ")):
            logger.info(
                f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}"
            )
            # move block to cuda if needed
            if not has_device_map or get_device(block) == torch.device("cpu"):
                block = block.to(0)
            layers = get_layers(block)
            if self.true_sequential:
                # lazy sequential but works well
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f"Module to quantize {layers_name_list}")
            for subset_name_list in tqdm(
                    layers_name_list,
                    leave=False,
                    desc="Quantizing layers inside the block"):
                subset_layers = {
                    name: layers[name]
                    for name in subset_name_list
                }
                quant_method = {}
                handles = []
                # add hook for each layer in subset_layers
                for name in subset_layers:
                    quant_method[name] = GPTQ(subset_layers[name])

                    def add_batch(name):

                        def tmp(_, input, output):
                            quant_method[name].add_batch(input[0].data)

                        return tmp

                    # because it adding a hook will replace the old one.
                    handles.append(subset_layers[name].register_forward_hook(
                        add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook
                for j in range(len(dataset)):
                    if not self.cache_examples_on_gpu:
                        layer_inputs[j] = layer_inputs[j].to(get_device(block))
                    # the args are already on the gpu
                    # don't need to store the output
                    block(layer_inputs[j], **layer_input_kwargs[j])
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(
                        f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    quant_method[name].prepare(percdamp=self.damp_percent, actorder=True)
                    if measure:
                        logger.info(
                            f"Measuring {name} in block {i + 1}/{len(blocks)}..."
                        )
                        measure_data = do_measure(quant_method[name])
                        result[
                            f"{self.block_name_to_quantize}.{i}.{name}"] = measure_data
                    else:
                        qp = QParams.from_dict(qparams[
                            f"{self.block_name_to_quantize}.{i}.{name}"]
                                               ["best_option"]["qparams"])
                        logger.info(
                            f"Quantizing {name} in block {i + 1}/{len(blocks)} -> {qp.get_desc()}, {qp.bpw((quant_method[name].rows, quant_method[name].columns)):.2f} bpw"
                        )
                        quant_method[name].configure(qp.group_size, qp.bits,
                                                     qp.bits_prop,
                                                     qp.scale_bits)
                        quant_method[name].quantize(keep_qweight=True)
                        quant_data = quant_method[name].pack("", qp)
                        result[
                            f"{self.block_name_to_quantize}.{i}.{name}"] = quant_data
                    del quant_method[name]
                    torch.cuda.empty_cache()
                del subset_layers

            for j in range(len(dataset)):
                layer_output = block(layer_inputs[j],
                                     **layer_input_kwargs[j])[0]
                if not self.cache_examples_on_gpu:
                    layer_output = layer_output.to('cpu')
                layer_outputs.append(layer_output)
            # put back to device
            if not has_device_map:
                blocks[i] = block.to(device)
            del layers
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        if measure:
            return result
        module_names_after_last_block = get_preceding_modules(
            model, self.block_name_to_quantize, reverse=True)
        module = nn.Sequential(*[
            recurse_getattr(model, name)
            for name in reversed(module_names_after_last_block)
        ])
        if not has_device_map:
            module = module.to(0)
        self.lm_head_name = module_names_after_last_block[0]
        lm_head = recurse_getattr(model, self.lm_head_name)
        quant_method[self.lm_head_name] = GPTQ(lm_head)
        handle = lm_head.register_forward_hook(add_batch(self.lm_head_name))
        for j in range(len(dataset)):
            if not self.cache_examples_on_gpu:
                layer_inputs[j] = layer_inputs[j].to(get_device(module))
            # the args are already on the gpu
            # don't need to store the output
            module(layer_inputs[j])
        handle.remove()
        logger.info(f"Quantizing {self.lm_head_name}")
        quant_method[self.lm_head_name].prepare(percdamp=self.damp_percent, actorder=True)
        qp = qparams_headoptions[self.head_bits]
        quant_method[self.lm_head_name].configure(qp.group_size, qp.bits,
                                                  qp.bits_prop, qp.scale_bits)
        quant_method[self.lm_head_name].quantize(keep_qweight=True)
        result[self.lm_head_name] = quant_method[self.lm_head_name].pack(
            "", qp)
        del quant_method[self.lm_head_name]
        if not has_device_map:
            module = module.to(device)
        torch.cuda.empty_cache()
        return result

    def pack_model(self, model: nn.Module, quantizers: dict):
        """
        Pack the model by replacing the layers by quantized layers

        Args:
            model (`nn.Module`):
                The model to pack
            quantizers (`Dict[str,Tuple]`):
                A mapping of the layer name and the data needed to pack the layer
        """
        logger.info("Packing model...")
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}
        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            qlayers[name].load_state_dict(quantizers[name], strict=False)
        logger.info("Model packed.")

    def save(self,
             model: nn.Module,
             save_dir: str,
             max_shard_size: str = "10GB",
             safe_serialization: bool = False):
        """
        Save model state dict and configs

        Args:
            model (`nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_dir (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        os.makedirs(save_dir, exist_ok=True)
        # save model and config
        accelerator = Accelerator()
        accelerator.save_model(model,
                               save_dir,
                               max_shard_size=max_shard_size,
                               safe_serialization=safe_serialization)
        model.config.save_pretrained(save_dir)
