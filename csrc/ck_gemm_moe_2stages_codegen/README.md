# CK GEMM MoE 2-Stages Codegen or asm 1-stage Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add MoE shapes in `aiter/configs/untuned_fmoe.csv`
    |**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|
    |---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|
    |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|


3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build moe 2-stages kernels via jit:
`python3 csrc/ck_gemm_moe_2stages_codegen/tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv`
You can find the results of this tuning in `aiter/configs/tuned_fmoe.csv`, like this:
    |**cu_num**|**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|**block_m**|**ksplit**|**us1**|**kernelName1**|**err1**|**us2**|**kernelName2**|**err2**|**us**|**run_1stage**|**tflops**|**bw**|
    |----------|---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|-----------|----------|-------|---------------|--------|-------|---------------|--------|------|--------------|----------|------|
    |80        |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|64|0|45.23|kernel_stage1|0.5%|38.67|kernel_stage2|0.3%|83.90|0|125.4|89.5|

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `run_1stage` indicates whether to run fused 1-stage kernel (1) or 2-stages kernels (0).

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py` and run it, please wait a few minutes as it will build moe tuned kernels in `aiter/configs/tuned_fmoe.csv` via jit:
`python3 op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py`
If you have built moe kernels before tuning new MoE shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_moe.py`. It will rebuild kernels from `AITER_CONFIG_FMOE`, the default one will be results merged from `aiter/configs/tuned_fmoe.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_tuned_fmoe_xx.csv`, the merged result is store in `/tmp/aiter_configs/tuned_fmoe.csv`.

## More Options
- Use `--last` flag to only tune the last kernel in the CSV file:
  `python3 csrc/ck_gemm_moe_2stages_codegen/tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv --last`
  
- Use `--profile_file` to save all profiling results for analysis:
  `python3 csrc/ck_gemm_moe_2stages_codegen/tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv --profile_file aiter/configs/profile_fmoe_all.csv`

- Adjust error ratio threshold with `--errRatio` (default: 0.5):
  `python3 csrc/ck_gemm_moe_2stages_codegen/tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv --errRatio 0.01`

- Set number of parallel processes with `--mp` (default: number of GPU):
  `python3 csrc/ck_gemm_moe_2stages_codegen/tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv --mp 8`

## Notes
- This tuner supports both 1-stage fused MoE kernels and 2-stages MoE kernels (stage1 and stage2)
- The tuner will automatically select the best kernel configuration based on performance
- Only G1U1 (gate-up fused) MoE configurations are currently supported for tuning
- Supported quantization types include: per_Token, per_1x128 (blockscale), per_1x32 (MXFP4, gfx950 only)
- If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build moe kernels in tuned csv by default. If you want to use the new result of moe tuning, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.

