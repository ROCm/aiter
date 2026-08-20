#!/bin/bash

# clean JIT cache
rm -rf aiter/jit/build aiter/jit/*.so

# Define the two combinations as arrays
data_inits=(constant uniform)
scale_inits=(constant auto)

for knl_name in f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps_low_eff; do
  ut_cmd="python op_tests/test_f4gemm.py --apre 1 --intype mxfp4 --outtype bf16 -mnk 16384,16384,16384 --knl-name ${knl_name}"
  echo "==================== benchmark ${knl_name} ===================="
  echo "ut_cmd=${ut_cmd}"
  echo "================================================================"
  eval ${ut_cmd}
done

for knl_name in f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps_low_eff; do
  for i in 0 1; do
    data_init=${data_inits[$i]}
    scale_init=${scale_inits[$i]}

    # Build the base command without --data-init and --scale-init
    base_cmd="python op_tests/test_f4gemm.py --apre 1 --intype mxfp4 --outtype bf16 -mnk 16384,16384,16384 --knl-name ${knl_name}"

    prof_cmd="rocprofv3 --att \
      --att-simd-select 0 \
      --att-target-cu 1 \
      --att-shader-engine-mask 0x1 \
      --kernel-include-regex \"${knl_name}\" \
      --kernel-iteration-range \"[50]\" \
      --truncate-kernels \
      -d ${knl_name}_${data_init}_${scale_init}_prof \
      -o att \
      -- ${base_cmd} --data-init ${data_init} --scale-init ${scale_init}"
    
    echo "==================== profile ${knl_name} with data ${data_init} init & scale ${scale_init} init ===================="
    echo "prof_cmd=${prof_cmd}"
    eval ${prof_cmd}
    echo "================================================================"
  done
done
