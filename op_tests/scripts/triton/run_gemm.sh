AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

# rm -rf $HOME/.triton/cache
N=$1
K=$2
for M in 1 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    echo $M
    # rocprofv2 --kernel-trace -o res python3 ${AITER_HOME}/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py --shape $M $N $K --metric time
    # rocprofv2 --kernel-trace -o res python3 ${AITER_HOME}/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py --shape $M $N $K --metric time
    # rocprofv2 --kernel-trace -o res python3 ${AITER_HOME}/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py --shape $M $N $K --metric time
    rocprofv2 --kernel-trace -o res python3 ${AITER_HOME}/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py --shape $M $N $K --metric time
    python3 rprof.py results_res.csv -k gemm
done
# grep -rnw "vgpr_count" $HOME/.triton/cache
# grep -rnw "vgpr_spill_count" $HOME/.triton/cache
