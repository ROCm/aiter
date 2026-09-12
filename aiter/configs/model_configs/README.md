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

