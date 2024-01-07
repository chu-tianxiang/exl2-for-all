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
import glob
from typing import Optional, List, Dict, Union

import torch
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights

from exl2 import Exl2Quantizer
from utils import post_init


class Exl2ForCausalLM(nn.Module):
    """
    A simple wrapper class for the Exl2 Model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def to(self, device: str):
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @classmethod
    def _download_weight(
        cls,
        model_name_or_path: str,
        revision: Optional[str] = None,
        use_safetensors: bool = True
    ) -> List[str]:
        """
        Downloads the model weights.

        Args:
            model_name_or_path (str): The name or path of the model.
            revision (Optional[str]): The revision of the model. Defaults to None.
            use_safetensors (bool): Whether to use safetensors. Defaults to True.

        Returns:
            List[str]: A list of paths to the downloaded model weight files.
        """
        is_local = os.path.isdir(model_name_or_path)
        if use_safetensors:
            allow_patterns = ["*.safetensors"]
        else:
            allow_patterns = ["*.bin", "*.pt"]
        if not is_local:
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          revision=revision)
        else:
            hf_folder = model_name_or_path
        hf_weights_files = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if not use_safetensors:
            hf_weights_files = [
                x for x in hf_weights_files
                if not x.endswith("training_args.bin")
            ]

        if len(hf_weights_files) == 0 and use_safetensors:
            return cls._download_weight(model_name_or_path,
                                        use_safetensors=False,
                                        revision=revision)
        return hf_weights_files

    @classmethod
    def from_quantized(
        cls,
        model_path: str,
        revision: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.float16,
        trust_remote_code: bool = True,
        safetensors: bool = True,
        no_split_module_classes: Optional[List[str]] = None,
        modules_to_not_convert: Optional[List] = None,
        device_map: Optional[Dict[str, int]] = None,
    ) -> "Exl2ForCausalLM":
        """
        Initializes an Exl2ForCausalLM instance from a quantized model.

        Args:
            model_path (str): The path to the quantized model.
            revision (Optional[str]): The revision of the model. Defaults to None.
            torch_dtype (Any): The torch data type to use. Defaults to torch.float16.
            trust_remote_code (bool): Whether to trust remote code. Defaults to True.
            safetensors (bool): Whether to use safetensors. Defaults to False.
            no_split_module_classes (Optional[List[str]]): A list of module classes that should not be split.
                Defaults to None.
            device_map (Optional[Dict[str, str]]): A dictionary mapping module names to device index.
                Defaults to None.

        Returns:
            Exl2ForCausalLM: An instance of the Exl2ForCausalLM class.
        """
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, revision=revision)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype)
        quantizer = Exl2Quantizer(modules_to_not_convert=modules_to_not_convert)
        model = quantizer.convert_model(model)
        # move model to cpu
        model = model._apply(lambda t: torch.zeros_like(t, device="cpu")
                             if t.device == torch.device("meta") else t)

        weight_files = cls._download_weight(model_path,
                                            revision=revision,
                                            use_safetensors=safetensors)
        for checkpoint_file in weight_files:
            if checkpoint_file.endswith(".safetensors"):
                state_dict = load_file(checkpoint_file)
            else:
                state_dict = torch.load(checkpoint_file)
            model.load_state_dict(state_dict, strict=False)

        if device_map is None:
            if no_split_module_classes is None:
                no_split_module_classes = quantizer.get_no_split_module_classes(
                    model)
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=no_split_module_classes,
                dtype=torch_dtype,
            )
        model = dispatch_model(model, device_map)
        model = post_init(model)
        model.eval()
        return cls(model=model)
