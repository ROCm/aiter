# Operators benchmark for real cases in LLM workloads


Here is the introduction of key operators benchmarking for real scenairos in LLM workloads. The operators now includes:
* SDPA


## SDPA kernel benchmark at workload level

#### Enviroment Setup
```
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
export PYTHON=/path_to_aiter/
cd aiter/op_tests/module_tests
```

#### Benchmark all SDPA cases in all supported models
The test_model_gemm.py now supports Gemm benchmark for cases in `Llama3.1-70B`, and `Qwen3-235B`.
Users can use the following command to get the benchmark results for all workloads. For each workload, a csv file will be generated.
```
python3 -u test_model_mha.py -o ./sdpa_bench_results
```
