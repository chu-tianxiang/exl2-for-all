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

from exllamav2.ext import exllamav2_ext as ext_c

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"

def make_group_map(q_groups, num_qrows):
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]
    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)

def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    w["q_scale_max"] /= 256
    w["q_perm"] = w["q_perm"].short()
    w["q_invperm"] = w["q_invperm"].short()
    if "q_group_map" not in w:
        w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])
    return ext_c.make_q_matrix(w["q_weight"], w["q_perm"], w["q_invperm"],
                               w["q_scale"], w["q_scale_max"], w["q_groups"], w["q_group_map"],
                               none_tensor, none_tensor, none_tensor, temp_dq)


class QuantLinear(nn.Module):

    def __init__(self, infeatures, outfeatures, bias, **kwargs):
        super().__init__()

        self.q_handle = None
        self.q_tensors = None
        self.padding = -outfeatures % 32

        self.infeatures = infeatures
        self.outfeatures = outfeatures + self.padding

        self.register_buffer('q_weight',
                             torch.zeros((1, outfeatures), dtype=torch.int32))
        self.register_buffer('q_groups', torch.zeros(1, dtype=torch.short))
        self.register_buffer(
            'q_scale',
            torch.zeros((1, outfeatures // 32 * 4), dtype=torch.int32))
        self.register_buffer('q_scale_max', torch.zeros(1,
                                                        dtype=torch.float16))
        self.register_buffer(
            'q_invperm',
            torch.tensor([i for i in range(infeatures)], dtype=torch.int32))

        if bias:
            self.register_buffer(
                'bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self, temp_dq):
        assert self.q_weight.device.type == "cuda"
        assert self.q_weight.device.index is not None
        self.q_tensors = {
            "q_weight": self.q_weight,
            "q_invperm": self.q_invperm,
            "q_scale": self.q_scale,
            "q_scale_max": self.q_scale_max,
            "q_groups": self.q_groups,
            "q_perm": torch.argsort(self.q_invperm).to(torch.int),
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ["q_weight", "q_groups", "q_scale", "q_scale_max"]:
            if prefix + key in state_dict:
                setattr(self, key, state_dict[prefix + key].clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, force_cuda=False):
        output_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.view(-1, x.shape[-1])
        old_dtype = x.dtype
        output = torch.empty((x.shape[0], self.outfeatures),
                             dtype=torch.half,
                             device=x.device)
        ext_c.gemm_half_q_half(x.to(torch.float16), self.q_handle, output, force_cuda)
        output = output.view(output_shape)

        if self.bias is not None:
            output.add_(self.bias)
        return output.to(old_dtype)

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(
            max_input_len, max_batch_size)


class ExLlamaV2DeviceTensors:

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes
        self.scratch = None

    def prepare(self):
        self.scratch = torch.empty((self.scratch_bytes // 2, ),
                                   dtype=torch.half,
                                   device=_torch_device(self.device_idx))

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()
        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
