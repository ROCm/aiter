```
                      _ _ _ _     
   __ _ _ __ __ _  __| | (_) |__  
  / _` | '__/ _` |/ _` | | | '_ \ 
 | (_| | | | (_| | (_| | | | |_) |
  \__, |_|  \__,_|\__,_|_|_|_.__/ 
  |___/ 
```
## What is gradlib
It is a library of tools derived from vLLM for optimization and tuning, mainly used for performance tuning of matrix multiplication (GEMM).

By gradlib, we can confirm the parameter of GEMMs with best performance in the specific hardware currently in use. As a result, we can **improve the inference speed of the model**.

## How to use gradlib

1. to get GEMM shapes to be tuned, replace F.linear by tgemm.mm under aiter/tuned_gemm.py,
   run

   `
    VLLM_TUNE_GEMM=1 python {workload_tests}
shapes will be captured in aiter/configs/untuned_gemm.csv
   `
2. to tune GEMMs in aiter/configs/untuned_gemm.csv,
   run
   
   ` 
    python3 gradlib/gradlib/gemm_tuner.py --tuned_file aiter/configs/tuned_gemm.csv  --input_file aiter/configs/untuned_gemm.csv
   `
4. then run your test as normal~
