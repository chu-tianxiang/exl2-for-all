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
import torch
import torch.nn as nn
from typing import Tuple, Union

from exllamav2.ext import exllamav2_ext as ext_c


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class AdaptiveQuantizer(nn.Module):
    '''
    Adaptive Quantizer for GPTQ.
    Use double quantization for scales to further save space

    Args:
        shape (Union[Tuple, int]): The placeholder shape for parameters
        norm (float): The norm used for quantization. Defaults to 3.5.
        max_p (float): The maximum parameter used for quantization. Defaults to 1.0.
        min_p (float): The minimum parameter used for quantization. Defaults to 0.75.
        p_grid (int): The grid size used for quantization. Defaults to 48.
    '''
    def __init__(self,
                 shape: Union[Tuple, int] = 1,
                 norm: float = 3.5,
                 max_p: float = 1.0,
                 min_p: float = 0.75,
                 p_grid: int = 48) -> None:
        super().__init__()
        self.norm: float = norm
        self.max_p: float = max_p
        self.min_p: float = min_p
        self.p_grid: int = p_grid
        self.scale_range = 1.0
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.register_buffer('qscale', torch.zeros(shape))
        self.register_buffer('qscale_max', torch.zeros(shape))

    def configure(self, bits: int, scale_bits: int = 4,
                  perchannel: bool = True, sym: bool = True) -> None:
        """
        Configures the AdaptiveQuantizer instance with the desired quantization parameters.

        Args:
            bits (int): The number of bits to be used for quantization.
            scale_bits (int): The number of bits to be used for scale quantization.
                Defaults to 4.
            perchannel (bool): Whether to use per-channel quantization. Defaults to True.
            sym (bool): Whether to use symmetric quantization. Defaults to True.
        """
        assert perchannel is True
        assert sym is True
        self.maxq = torch.tensor(2 ** bits - 1)
        self.scale_maxq = 2 ** scale_bits - 1
        self.perchannel = perchannel
        self.sym = sym

    def find_params(self, x: torch.Tensor) -> None:
        """
        Finds the optimal quantization parameters for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor to be quantized.
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)
        self.zero = (self.maxq + 1) / 2

        x = x.flatten(1)
        xmax, _ = torch.max(torch.abs(x), dim=0)
        xmax += 1e-12
        base_scale = xmax / (self.maxq / 2)

        # double quant
        qscale_max_t = torch.max(base_scale) * self.scale_range
        scale_tp = base_scale / qscale_max_t
        scale_tp = torch.sqrt(scale_tp)
        scale_tp *= (self.scale_maxq + 1)
        qscale_t = torch.clamp(torch.round(scale_tp), 1, self.scale_maxq + 1)
        qscale_tw = qscale_t / (self.scale_maxq + 1)
        qscale_tw = qscale_tw**2
        qscale_tw *= qscale_max_t

        # dynamic search best scale-max based on the quantized scales
        q = torch.zeros((self.p_grid + 1, 128),
                        dtype=torch.float,
                        device=x.device)
        ext_c.quantize_err(x, q, qscale_tw, self.zero, self.maxq, self.norm,
                           self.min_p, self.max_p, self.p_grid)
        q = torch.sum(q, dim=1)
        best_pi = torch.argmin(q)
        best_pif = best_pi / self.p_grid
        best_p = self.max_p * best_pif + self.min_p * (1 - best_pif)

        # save double quants.
        self.qscale = qscale_t.to(torch.short)
        self.scale = qscale_tw * best_p
        self.qscale_max = qscale_max_t * best_p

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Quantizes the input tensor using the previously determined quantization parameters.

        Args:
            x (torch.Tensor): The input tensor to be quantized.

        Returns:
            q (torch.Tensor): The quantized tensor.
        '''
        return quantize(x, self.scale, self.zero, self.maxq)
