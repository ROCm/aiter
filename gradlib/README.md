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

    then shapes will be captured in aiter/configs/bf16_untuned_gemm.csv
2. to tune GEMMs in aiter/configs/bf16_untuned_gemm.csv,
    You can find the results of this tuning in `aiter/configs/bf16_tuned_gemm.csv`.
    |**cu_num**|**M**|**N**|**K**|**bias**|   **dtype**  | **outdtype** |**scaleAB**|**bpreshuffle**|**libtype**|**solidx**|**splitK**|**soltimes**|**kernelName**|**tflops**|**bw**|
    |----------|-----|-----|-----|--------|--------------|--------------|-----------|---------------|-----------|----------|----------|------------|--------------|----------|------|
    |80        |128  |1536 |7168 |  False |torch.bfloat16|torch.float32 | False     | False         | hipblast  |667788    |0         | 10.6       | xxxxxxx      |  xx      | xx   |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `bpreshuffle` means whether weight will be shuffled
    `dtype` means the input data type
    `libtype` means the kernel library type: hipblaslt or rocblas or asm
    `splitK` only be valid in libtype==asm
    `tflops`  TFLOPS 
    `bw`  means bandwidth of the implement, GB/s
   
   run
   
   ` 
    python3 gradlib/gradlib/gemm_tuner.py --tuned_file aiter/configs/bf16_tuned_gemm.csv  --input_file aiter/configs/bf16_untuned_gemm.csv
   `
   more features:
      #### `-o2, --profile_file`
      - **Type**: String
      - **Default**: `""` (empty string)
      - **Required**: No
      - **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.
      
      **Example**:
      ```bash
      --profile_file /path/to/all_results.csv
      ```  
      #### `--mp`
      - **Type**: Integer
      - **Default**: `torch.cuda.device_count()` (number of available GPUs)
      - **Description**: Number of parallel processes to use for tuning across multiple GPUs. Each process runs on a separate GPU.
      
      **Examples**:
      ```bash
      --mp 1           # Single GPU tuning
      ```
      ### Tuning Configuration
      
      #### `--errRatio`
      - **Type**: Float
      - **Default**: `0.05` (5%)
      - **Description**: Tolerable error ratio threshold. Only kernels with error ratios below this threshold will be considered valid candidates.
      
      **Example**:
      ```bash
      --errRatio 0.01  # Stricter tolerance (1% error)
      --errRatio 0.10  # More lenient tolerance (10% error)
      ```
      
      #### `--sort`
      - **Type**: Flag (boolean)
      - **Default**: `False`
      - **Description**: Sort the output file according to the key columns (e.g., `cu_num`, `N`, `M`, `K` for GEMM). Useful for maintaining consistent ordering in result files.
      
      **Example**:
      ```bash
      --sort          # Sort results by keys
      ```
      
      #### `--all`
      - **Type**: Flag (boolean)
      - **Default**: `False`
      - **Description**: Retune all shapes based on file relationship:
        - If `tune_file` == `untune_file`: Retune all shapes in the tune file
        - If `tune_file` != `untune_file`: Retune shapes that exist in untuned file
      
      **Example**:
      ```bash
      --all           # Retune all shapes
      ```
      
      ### Debugging and Verbose Output
      
      #### `-v, --verbose`
      - **Type**: Flag (boolean)
      - **Default**: `False`
      - **Description**: Enable verbose output with detailed logging information, including skipped shapes, tuning progress, and detailed error messages.
      
      **Example**:
      ```bash
      --verbose       # Enable verbose mode
      -v              # Short form
      ```
3. then run your test as normal~
