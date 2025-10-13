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
    AITER_TUNE_GEMM=1 python {workload_tests}
   `

    then shapes will be captured in aiter/configs/untuned_gemm.csv
2. to tune GEMMs in aiter/configs/untuned_gemm.csv,
    You can find the results of this tuning in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`.
    |**cu_num**|**M**|**N**|**K**|**bias**|   **dtype**  | **outdtype** |**scaleAB**|**libtype**|**solidx**|**splitK**|**soltimes**|**kernelName**|**tflops**|**bw**|
    |----------|-----|-----|-----|--------|--------------|--------------|-----------|-----------|----------|----------|------------|--------------|----------|------|
    |80        |128  |1536 |7168 |  False |torch.bfloat16|torch.float32 | False     | hipblast  |667788    |0         | 10.6       | xxxxxxx      |  xx      | xx   |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `dtype` means the input data type
    `libtype` means the kernel library type: hipblaslt or rocblas or asm
    `splitK` only be valid in libtype==asm
    `tflops`  TFLOPS 
    `bw`  means bandwidth of the implement, GB/s
   
   run
   
   ` 
    python3 gradlib/gradlib/gemm_tuner.py --tuned_file aiter/configs/tuned_gemm.csv  --input_file aiter/configs/untuned_gemm.csv
   `
3. then run your test as normal~
