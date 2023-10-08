# EXL2 for all

EXL2 is a mixed-bits quantization method proposed in [exllama v2](https://github.com/turboderp/exllamav2). This repo is created from exllamav2 with support for more model architectures.
Unlike repos like [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) which seek for best performance, this repo only contains minimal code for quantization and inference.

# Installation

exllama v2 kernels have to installed first. See `requirements.txt` for dependencies.

# Examples

* Quantization

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exl2 import Exl2Quantizer

model_name = "meta-llama/Llama-2-7b-hf"
quant_dir = "llama-exl2-4bits"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = Exl2Quantizer(bits=4.0, dataset="c4")
quant_model = quant.quantize_model(model, tokenizer)
quant.save(quant_model, quant_dir)
tokenizer.save_pretrained(quant_dir)
```

* Inference

```python
import torch
from transformers import AutoTokenizer
from model import Exl2ForCausalLM

quant_model = Exl2ForCausalLM.from_quantized("turboderp/Llama2-7B-exl2", revision="2.5bpw")
tokenizer = AutoTokenizer.from_pretrained("turboderp/Llama2-7B-exl2", revision="2.5bpw")
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
output_ids = quant_model.generate(input_ids, do_sample=True)
print(tokenizer.decode(output_ids[0]))
```