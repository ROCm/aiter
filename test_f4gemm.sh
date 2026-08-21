#!/bin/bash

# All artifacts (e2e log + the 4 ttrace prof dirs) land under one directory so the
# whole run can be copied out with a single `docker cp`. The name is timestamped;
# pass an explicit name as $1 (e.g. from the host) if you need to know it upfront.
out_dir="${1:-f4gemm_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${out_dir}"

# Tee everything (both benchmark and profile phases) into the same dir as the ttraces.
exec > >(tee "${out_dir}/f4gemm.log") 2>&1

echo "==================== output dir: ${out_dir} ===================="

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

    prof_dir="${out_dir}/${knl_name}_${data_init}_${scale_init}_prof"
    # --kernel-trace populates the KERNEL_DISPATCH domain (real dispatch time);
    # --att alone leaves it empty. --under-rocprof disables the UT's internal
    # torch.profiler so it doesn't collide with rocprof (see att_freq.py).
    prof_cmd="rocprofv3 --att \
      --att-simd-select 0 \
      --att-target-cu 1 \
      --att-shader-engine-mask 0x1 \
      --kernel-trace \
      --kernel-include-regex \"${knl_name}\" \
      --kernel-iteration-range \"[50]\" \
      --truncate-kernels \
      -d ${prof_dir} \
      -o att \
      -- ${base_cmd} --data-init ${data_init} --scale-init ${scale_init} --under-rocprof"

    echo "==================== profile ${knl_name} with data ${data_init} init & scale ${scale_init} init ===================="
    echo "prof_cmd=${prof_cmd}"
    eval ${prof_cmd}
    # Read real dispatch time / gfx cycles / avg clock for the ATT-traced iter.
    python3 op_tests/att_freq.py "${prof_dir}"
    echo "================================================================"
  done
done

echo "==================== done. all artifacts in: ${out_dir} ===================="
