# Quickly Gemm Performance Tuning for Popular Models

This is an instruction for quickly important kernels performance tuning including Gemm/FlashAttn/MoE for some popular models. 

### 1. Install aiter:
`cd $aiter_path` 
`python3 setup.py develop`

### 2. Create model configuration files
Taking GEMM kernel in Llama-70B model for example, `aiter/configs/model_configs/llama70B_untuned_gemm.csv` contains the usually used GEMM kernels in Llama-70B including those in Attention and MLP moduldes. The GEMM shape in this file also consider model deployed with different parallelism recipes inlcuding TP1/TP4/TP8. 

### 3. Tune the best configuration
#### 3.1 General GEMM kernel performance tuning
  If you want to tune the Llama-70B GEMM kernels, then copy `aiter/configs/model_configs/llama70B_untuned_gemm_bf16.csv` to `aiter/configs/untuned_gemm.csv`.
  Then refer to the steps in [GEMM performance tune](https://github.com/ROCm/aiter/tree/main/gradlib) for kernel performance tuning. 


#### 3.2 FP8 GEMM kernel performance tuning

- ##### MXFP8 GEMM with Per Token or Per Tensor quantization. Weight in plain format:
  If you want to tune the Llama-70B GEMM kernels with FP8 datatype and the weight is in plain format, then copy `aiter/configs/model_configs/llama70B_untuned_gemm.csv` to `aiter/configs/a8w8_untuned_gemm.csv`.Then refer to the steps in [CK gemm a8w8 tune](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8) for kernel performance tuning. 


- ##### MXFP8 GEMM with Per Token or Per Tensor quantization. Weight in preshuffling format:
   If you want to tune the Llama-70B GEMM kernels with FP8 datatype and the weight is in preshuffling format, then copy `aiter/configs/model_configs/llama70B_untuned_gemm.csv` to `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`.
   Then refer to the steps in [CK gemm a8w8 tune preshuffle](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_bpreshuffle) for kernel performance tuning. 


- ##### MXFP8 GEMM with Per Block quantization.
  If you want to tune the Llama-70B GEMM kernels with FP8 datatype and use the Per Block quantization, then copy `aiter/configs/model_configs/llama70B_untuned_gemm.csv` to `aiter/configs/a8w8_blockscale_untuned_gemm.csv`.
  Then refer to the steps in [CK gemm a8w8 blockscale tune](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale) for kernel performance tuning. 


#### 3.3 MXFP4 GEMM kernel performance tuning
- ###### MXFP4 GEMM with Per Block quantization (weight preshuffle).
  If you want to tune the Llama-70B GEMM kernels with MXFP4 datatype, then copy `aiter/configs/model_configs/llama70B_untuned_gemm.csv` to `aiter/configs/a4w4_blockscale_untuned_gemm.csv`
  Then refer to the steps in [CK gemm a4w4 blockscale tune](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a4w4_blockscale) for kernel performance tuning. 


## Model List
The model list we have summarized including:
- [Llama3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Llama-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

## MXFP4 A4W4 model configurations

`llama2_7b_a4w4_blockscale_tuned_gemm.csv`, `llama31_8b_...`, `qwen3_8b_...` and
`qwen3_small_...` were moved unchanged from Lumen
`7b5931d21a56869f3e13b55992d7a4e1f8484267` (MIT). They contain historical
gfx950 / 256-CU measurements, not measurements made during the configuration
refactor; `qwen3_small` covers the small Qwen3 tiers and keeps its explicit model identity. Model configuration selection lives in `aiter/configs/mxfp4_profiles.json` and
is consumed by `aiter.utility.mxfp4_tuning`:

| Profile | Model configuration | Rows |
|---|---|---:|
| llama2_7b | `llama2_7b_a4w4_blockscale_tuned_gemm.csv` | 9 |
| llama31_8b | `llama31_8b_a4w4_blockscale_tuned_gemm.csv` | 7 |
| qwen3_8b | `qwen3_8b_a4w4_blockscale_tuned_gemm.csv` | 8 |
| qwen3_small | `qwen3_small_a4w4_blockscale_tuned_gemm.csv` | 13 |
| qwen3_fsdp | qwen3_8b + qwen3_small, in that order | 21 |

The default source list is `aiter/configs/mxfp4_default_sources.json`: the stock
table plus the seven existing DeepSeek V3, Gemma4 31B, Kimi K3, narrow-N,
Qwen3 VL, Qwen3.8 27B and Qwen3.8 2.4T tables. It preserves their tuned coverage
without scanning the directory. New model tables must be selected by profile
or added to this default list after checking key conflicts.

A profile places its named model tables before these defaults. The first row
for `(gfx, cu_num, M, N, K)` wins. An explicit `AITER_CONFIG_GEMM_A4W4` replaces
both the profile and the defaults. Source CSVs are never rewritten.

Without a profile, existing kernel choices stay the same. With a profile,
choices stay the same outside its lookup coverage: the exact M and the two
padded-M candidates used by `get_GEMM_config`. An unlisted M can use a profile
row through padding, so absence from the profile CSV alone does not imply an
unchanged choice. See the [validation report](../../../docs/mxfp4_tuning_validation/README.md).

MIT License

Copyright (c) 2026 ZhangDanyang-AMD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
