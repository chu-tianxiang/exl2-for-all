"""
Monkey patch the llama implementation in the huggingface/transformers to use flash-attn
"""
import torch
import transformers
from exllamav2.ext import exllamav2_ext as ext_c
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from flash_attn.ops.triton.rotary import apply_rotary

from model import Exl2ForCausalLM


# Copied from https://github.com/turboderp/exllamav2/blob/master/exllamav2/rmsnorm.py
def rms_forward(self, hidden_states):
    output_shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    norm = torch.empty_like(hidden_states)
    ext_c.rms_norm(hidden_states, self.weight, norm, self.variance_epsilon)
    hidden_states = norm.view(output_shape)
    return hidden_states

# Flash-attn use head_dim//2 for cos and sin, while transformers use head_dim
# so we need to remove the extra concatenation.
def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)

# Flash-attn attention forward
# todo: the padding tokens are not handled right now, which may lead to worse result in batch generation.
def attention_forward(self, hidden_states, attention_mask, position_ids, past_key_value, *args, **kwargs):
    batch, seqlen = hidden_states.shape[:2]
    query_states = self.q_proj(hidden_states).view(batch, seqlen, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(batch, seqlen, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(batch, seqlen, self.num_key_value_heads, self.head_dim)
    cos, sin  = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached
    if past_key_value is None:
        query_states = apply_rotary(query_states, cos, sin)
        key_states = apply_rotary(key_states, cos, sin)
        past_key_value = [
            torch.empty((batch, self.max_position_embeddings, 2, self.num_key_value_heads, self.head_dim),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype),
            seqlen,
        ]
        past_key_value[0][:, :seqlen, 0, ...] = key_states
        past_key_value[0][:, :seqlen, 1, ...] = value_states
        context = flash_attn_func(query_states, key_states, value_states, causal=True)
    else:
        context = flash_attn_with_kvcache(
            query_states,
            past_key_value[0][:, :, 0],
            past_key_value[0][:, :, 1],
            k=key_states,
            v=value_states,
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=past_key_value[1],
            causal=True,
            rotary_interleaved=False,
        )
        past_key_value[1] += seqlen
    context = context.view(batch, seqlen, -1)
    attn_output = self.o_proj(context)
    return attn_output, None, past_key_value

def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, **kwargs
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }
    return model_inputs

def replace_llama():
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = rms_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = attention_forward
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding._set_cos_sin_cache = _set_cos_sin_cache
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation

if __name__ == "__main__":
    replace_llama()
    quant_model = Exl2ForCausalLM.from_quantized("turboderp/Llama2-7B-exl2", revision="2.5bpw")
    tokenizer = transformers.AutoTokenizer.from_pretrained("turboderp/Llama2-7B-exl2", revision="2.5bpw")
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
    output_ids = quant_model.generate(input_ids, do_sample=True)
    print(tokenizer.decode(output_ids[0]))