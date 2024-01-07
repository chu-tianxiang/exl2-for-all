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
from typing import Tuple, Dict, List, Any

class QParams:
    def __init__(self, group_size: dict, bits: List[int], bits_prop: List[float], scale_bits: int):
        """
        Represents the quantization parameters for a specific layer.

        Args:
            group_size: The size of the quantization group.
            bits: A list of the number of bits to use.
            bits_prop: A list of the proportion for each bits.
            scale_bits: The number of bits to use for scaling the quantized values.
        """
        self.bits = bits
        self.bits_prop = bits_prop
        self.scale_bits = scale_bits
        if isinstance(group_size, dict):
            self.group_size = {int(b): g for b, g in group_size.items()}
        elif isinstance(group_size, list):
            assert len(group_size) == len(bits)
            self.group_size = {b: g for g, b in zip(group_size, bits)}
        else:
            self.group_size = {b: group_size for b in bits }
        self.desc = self.get_desc()

    def get_dict(self) -> Dict[str, Any]:
        """
        Returns the quantization parameters as a dictionary.

        Returns:
            A dictionary containing the quantization parameters.
        """
        return {
            "group_size": self.group_size,
            "bits": self.bits,
            "bits_prop": self.bits_prop,
            "scale_bits": self.scale_bits
        }

    @staticmethod
    def from_dict(qp_dict: Dict[str, Any]) -> "QParams":
        """
        Creates a QParams object from a dictionary.

        Args:
            qp_dict: A dictionary containing the quantization parameters.

        Returns:
            A QParams object.
        """
        return QParams(qp_dict["group_size"], qp_dict["bits"],
                       qp_dict["bits_prop"], qp_dict["scale_bits"])

    def total_bits(self, shape: Tuple[int, int]) -> int:
        """
        Calculates the total number of bits based on the quantization parameters.

        Args:
            shape: The shape of the weight matrix.

        Returns:
            The total number of bits.
        """
        rows = shape[0]
        columns = shape[1]

        groups = 0
        remaining_rows = rows
        bits_groups = []
        for b, p in zip(self.bits, self.bits_prop):
            gsz = self.group_size[b]
            g = math.ceil(min(rows * p, remaining_rows) / gsz)
            groups += g
            remaining_rows -= g * gsz
            bits_groups.append(g)

        assert remaining_rows <= 0

        total_bits = 0
        tr = rows

        for g, b in zip(bits_groups, self.bits):
            r = self.group_size[b] * g
            c = columns
            if r > tr: r = tr
            tr -= r
            total_bits += r * c * b  # q_weight

        total_bits += groups * 16  # q_scale_max
        total_bits += groups * (16 + 16)  # q_groups
        total_bits += groups * columns * self.scale_bits  # q_scale
        total_bits += rows * 32  # q_invperm
        return total_bits

    def bpw(self, shape: Tuple[int, int]) -> float:
        """
        Calculates the number of bits per weight based on the quantization parameters.

        Args:
            shape: The shape of the weight matrix.

        Returns:
            The number of bits per weight.
        """
        rows = shape[0]
        columns = shape[1]
        numel = rows * columns
        return self.total_bits(shape) / numel

    def get_desc(self) -> str:
        """
        Generates a description of the quantization parameters.

        Returns:
            A string describing the quantization parameters.
        """
        s = ""
        for b, p in zip(self.bits, self.bits_prop):
            if s != "": s += "/"
            g = self.group_size[b]
            s += f"{p}:{b}b_{g}g"
        s += f" s{self.scale_bits}"
        return s


# kernels require groupsize divisible by 32
qparams_options = [
    QParams(32, [3, 2], [0.05, 0.95], 4),
    QParams(32, [3, 2], [0.25, 0.75], 4),
    QParams(32, [4, 2], [0.25, 0.75], 4),
    QParams(32, [4, 3, 2], [0.1, 0.4, 0.5], 4),
    QParams(32, [4, 3], [0.1, 0.9], 4),
    QParams(32, [6, 3], [0.2, 0.8], 4),
    QParams(128, [3], [1.0], 4),
    QParams(32, [3], [1.0], 4),
    QParams(32, [4, 3], [0.05, 0.95], 4),
    QParams(32, [4, 3], [0.4, 0.6], 4),
    QParams(64, [4, 3], [0.6, 0.4], 4),
    QParams(128, [4], [1.0], 4),
    QParams(32, [4], [1.0], 4),
    QParams(32, [5, 4], [0.1, 0.9], 4),
    QParams(32, [6, 4], [0.1, 0.9], 4),
    QParams(128, [5], [1.0], 4),
    QParams(32, [6, 5], [0.1, 0.9], 4),
    QParams(32, [8, 6, 5], [0.05, 0.05, 0.9], 4),
    QParams(32, [6, 5], [0.4, 0.6], 4),
    QParams(32, [8, 6, 5], [0.1, 0.3, 0.6], 4),
    QParams(128, [6], [1.0], 4),
    QParams(32, [6], [1.0], 4),
    QParams(128, [8, 6], [0.1, 0.9], 4),
    QParams(32, [8], [1.0], 4),
]

qparams_headoptions = {
    2: QParams(32, [4, 2], [0.3, 0.7], 4),
    3: QParams(32, [4, 3], [0.15, 0.85], 4),
    4: QParams(32, [6, 4], [0.15, 0.85], 4),
    5: QParams(32, [6, 5], [0.15, 0.85], 4),
    6: QParams(32, [8, 6], [0.15, 0.85], 4),
    8: QParams(32, [8], [1.0], 4),
    16: None
}
