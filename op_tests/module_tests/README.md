# Operators benchmark for real cases in LLM workloads


Here is the introduction of key operators benchmarking for real scenairos in LLM workloads. The operators now includes:
* Gemm


## Gemm kernel benchmark at workload level

#### Enviroment Setup
```
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
export PYTHON=/path_to_aiter/
cd aiter/op_tests/module_tests
```

#### Benchmark all Gemm cases
The test_model_gemm.py now supports Gemm benchmark for cases in `Qwen3-32B`, `Llama3-70B` and `Llama3-405B`.
Users can use the following command to get the benchmark results for all workloads. For each workload, a csv file will be generated. Taking `Llama3-70B` for example, a csv file named as `Llama3-405B.csv` will be generated in currect folder. The result contains `attn qkv fused gemm`, `attn output gemm`, `mlp up-gate fused gemm`, `mlp up-gate non-fused gemm`, `mlp down gemm` considering different Tensor Parallel sizes in the list of [1, 4, 8]. By defalut, the script will benchmark all the FP8, BF16 and FP4 (if supported by the platform) if user does not specify the datatype.
```
python3 -u test_model_gemm.py
```
#### Benchmark Gemm in specified configuration
Users can also specify the model and the corresponding configuration they need. The following example indicates Gemm kernel will be benchmark using the shapes in `Llama3-70B` with tensor parallel size equaling to 8, input and weight in FP8 quantization datatype and output in BF16 datatype.
```
python3 -u test_model_gemm.py -m "Llama3-70B" -tp 8 -d bf16 -q fp8
```


