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
import math
from logging import getLogger
from typing import Optional, Union, List, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
import transformers

from qparam import QParams
from adaptive_quant import AdaptiveQuantizer
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

logger = getLogger(__name__)


class GPTQ:
    """
    Class for GPTQ quantization.

    Args:
        layer: The linear layer to be quantized.
    """
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.dev = layer.weight.device

        weight = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            weight = weight.t()

        self.rows = weight.shape[1]
        self.columns = weight.shape[0]
        self.hessian = torch.zeros((self.rows, self.rows), device=self.dev)
        self.nsamples = 0
        self.inps = []
        self.quantizer = AdaptiveQuantizer()

    def configure(self,
                  group_size: int = 128,
                  bits: Optional[Union[int, List[int]]] = None,
                  bits_prop: Optional[List[float]] = None,
                  scale_bits: int = 4) -> None:
        """
        Configures the quantization settings.

        Args:
            group_size: The size of each quantization group.
            bits: The number of bits used for quantization. Can be a single integer or a list of integers for different groups.
            bits_prop: The proportion of each group size to be quantized with a specific number of bits. Must be provided if bits is a list.
            scale_bits: The number of bits used for quantizing the scaling factor.
        """
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.total_groups = (self.rows + self.group_size -
                             1) // self.group_size

        if isinstance(bits, list):
            self.bits = bits
            g128 = (self.rows + 128 - 1) // 128
            self.bits_groups = [
                max(round(g128 * p), 1) * 128 // self.group_size
                for p in bits_prop
            ]
            e = sum(self.bits_groups) - self.total_groups
            self.bits_groups[-1] -= e
        else:
            self.bits = [bits]
            self.bits_groups = [self.total_groups]

    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Adds a batch of input and update the Hessian matrix.

        Args:
            inp: The input batch to be added.
        """
        tmp = inp.shape[0]
        self.inps.append(inp)
        if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size,
                               dilation=self.layer.dilation,
                               padding=self.layer.padding,
                               stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.hessian *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hessian += inp.matmul(inp.t())

    def prepare(self, percdamp: float = 0.01, actorder: bool = False) -> None:
        """
        Prepares the Hessian matrix inverse for further computations.

        Args:
            percdamp: The percentage of damping to be applied.
            actorder: Whether to reorder the Hessian matrix based on activation order.
        """
        with torch.inference_mode():
            hessian = self.hessian
            del self.hessian

            # Activation order
            if actorder:
                self.perm = torch.argsort(torch.diag(hessian), descending=True)
                hessian = hessian[self.perm][:, self.perm]
            else:
                self.perm = torch.arange(self.rows, device=self.dev)

            # Damping
            damp = percdamp * torch.mean(torch.diag(hessian))
            diag = torch.arange(self.rows, device=self.dev)

            # Inverse of H
            attempts = 0
            while True:
                try:
                    hessian[diag, diag] += damp
                    hessian_inv = torch.linalg.cholesky(hessian)
                    hessian_inv = torch.cholesky_inverse(hessian_inv)

                    # The Cholesky inverse will sometimes fail to compute due to accumulated rounding errors when H
                    # is very large (e.g. 70B MLP down proj) and a lot of calibration data is used (e.g. 100 rows of
                    # 4096 tokens). This won't always throw an exception and sometimes just results in a NaN tensor.
                    if torch.any(torch.isnan(hessian_inv)): raise RuntimeError
                    hessian_inv = torch.linalg.cholesky(hessian_inv,
                                                        upper=True)
                    hessian_inv = hessian_inv.contiguous()
                    break
                except RuntimeError:
                    # If inverting failed, assume there were non-positive eigenvalues, so apply more damping to shift
                    # the eigenvalues in a positive direction.
                    logger.warning(" !! Warning: Applied additional damping")
                    attempts += 1
                    if attempts == 10:
                        raise ValueError("Hessian is not invertible")

            self.hessian_inv = hessian_inv

    def quantize(self, keep_qweight: bool = False) -> None:
        """
        Quantizes the weights of the layer based on the configured quantization settings.

        Args:
            keep_qweight: Whether to save the quantized weights in self.qweight.
        """
        with torch.inference_mode():
            weights = self.layer.weight.data.clone()
            if isinstance(self.layer, nn.Conv2d):
                weights = weights.flatten(1)
            if not isinstance(self.layer, transformers.Conv1D):
                weights = weights.t()
            weights = weights.float()
            weights = weights[self.perm, :]

            quants = torch.zeros_like(weights)
            if keep_qweight:
                self.qweight = torch.zeros_like(weights, dtype=torch.short)

            # Quantize groups

            scale = []
            qscale = []
            qscale_max = []
            qgroups = []

            bits_idx = -1
            bits_idx_r = 1

            error = weights.clone()

            for group in range(self.total_groups):
                a = group * self.group_size
                b = min(a + self.group_size, self.rows)
                bits_idx_r -= 1
                if bits_idx_r == 0:
                    bits_idx += 1
                    bits_idx_r = self.bits_groups[bits_idx]
                    bits = self.bits[bits_idx]
                    self.quantizer = AdaptiveQuantizer()
                    self.quantizer.configure(bits=bits,
                                             scale_bits=self.scale_bits)

                qgroups.append(bits)
                qgroups.append(0)

                self.quantizer.find_params(weights[a:b, :])
                scale.append(self.quantizer.scale)
                qscale.append(self.quantizer.qscale)
                qscale_max.append(self.quantizer.qscale_max)

                ext_c.quantize_range(
                    quants, self.quantizer.scale,
                    self.qweight if keep_qweight else none_tensor,
                    self.quantizer.zero, self.quantizer.maxq, self.hessian_inv,
                    weights, error, a, b)

            # Create g_idx to store inverse activation order
            rows = [i // self.group_size for i in range(self.rows)]
            self.g_idx = torch.tensor(rows, dtype=torch.int32, device=self.dev)
            self.invperm = torch.argsort(self.perm)
            self.g_idx = self.g_idx[self.invperm]

            # Store scales
            self.scale = torch.stack(scale, dim=0)
            self.qscale = torch.stack(qscale, dim=0)
            self.qscale_max = torch.tensor(qscale_max,
                                           dtype=torch.float16,
                                           device=self.dev)
            self.qgroups = torch.tensor(qgroups,
                                        dtype=torch.short,
                                        device=self.dev)

            weight = quants[self.invperm, :]
            if not isinstance(self.layer, transformers.Conv1D):
                weight = weight.t()

            self.layer.weight.data = weight.reshape(
                self.layer.weight.shape).type_as(self.layer.weight.data)

    def quant_error(self) -> Tuple[float, float, float]:
        """
        Computes the quantization error between original weights compared to the quantized weights.

        Returns:
            The quantization errors at thresholds of 0.01, 0.05, and 0.10.
        """
        with torch.inference_mode():
            q = self.quant[self.invperm, :]
            diff = torch.abs(q - self.layer.weight.data.T)
            mat_error_1 = (diff > 0.01).sum().item() / diff.numel()
            mat_error_5 = (diff > 0.05).sum().item() / diff.numel()
            mat_error_10 = (diff > 0.10).sum().item() / diff.numel()
            return mat_error_1, mat_error_5, mat_error_10

    def free(self) -> None:
        """
        Frees up memory.
        """
        self.hessian = None
        self.hessian_inv = None
        self.qweight = None
        self.g_idx = None
        self.invperm = None
        self.scale = None
        self.qscale = None
        self.qscale_max = None
        torch.cuda.empty_cache()

    def pack(self, key: str, qparams: QParams) -> Dict[str, torch.Tensor]:
        """
        Packs the quantization related parameters into a state dict.

        Args:
            key: The key to be added as a prefix to the parameter names.
            qparams: The quantization parameters.

        Returns:
            A dictionary containing the packed parameters.
        """
        assert qparams.scale_bits in [4]
        # assert self.columns % 32 == 0
        output = {}
        if key != "":
            key += "."
        output[key + "q_invperm"] = self.invperm.to(torch.int).cpu()
        output[key + "q_scale_max"] = self.qscale_max.cpu()
        output[key + "q_groups"] = self.qgroups.cpu()
        columns = self.columns
        rem_rows = self.rows
        padding = -columns % 32

        if padding != 0:
            logger.warning(f" !! Note: Padding quantized tensor {key}")
        qst = F.pad(self.qscale, (0, padding)).contiguous()
        qwt = F.pad(self.qweight, (0, padding)).contiguous()

        qst_packed = torch.zeros(
            (qst.shape[0], qst.shape[1] * qparams.scale_bits // 32),
            dtype=torch.int32,
            device=self.dev)
        if qparams.scale_bits == 4:
            ext_c.pack_rows_4(qst, qst_packed)
        output[key + "q_scale"] = qst_packed.cpu()

        qwt_packed = []
        i = 0
        row = 0
        out_row = 0
        while i < self.qscale.shape[0]:
            bits = self.qgroups[i * 2].item()
            self.qgroups[i * 2 + 1] = out_row
            i += 1
            rows = min(self.group_size, rem_rows)
            wpqr = 32 / bits
            qrows = rows / wpqr
            assert i == self.qgroups.shape[-1] or qrows == int(qrows)
            qrows = math.ceil(qrows)
            g_qwt = qwt[row:row + rows, :].contiguous()
            g_qwt_packed = torch.zeros((qrows, columns + padding),
                                       dtype=torch.int32,
                                       device=self.dev)

            if padding > 0:
                g_qwt[:, -padding:] = 2**(bits - 1)
            ext_c.pack_columns(g_qwt, g_qwt_packed, bits)
            qwt_packed.append(g_qwt_packed)
            row += rows
            out_row += qrows
            rem_rows -= rows

        qwt_packed = torch.cat(qwt_packed, dim=0)
        output[key + "q_weight"] = qwt_packed.cpu()
        if self.layer.bias is not None:
            output[key + "bias"] = self.layer.bias.clone().half().cpu()

        return output
