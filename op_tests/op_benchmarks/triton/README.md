# Triton/Gluon benchmarks

## Rules
1. Use RotatingBufferBench from `op_tests/op_benchmarks/triton/utils/rotating_buffer_bench.py` for creating rotating buffer pool and benchmarking
2. Do not use custom benchmarking functions or any triton lang do_bench functions. 
3. RotatingBufferBench pool creation API requires input generator function. Try to share
that with the unit test. If it is shared then import the function from unit test file
4. For ops that are multi-GPU, use bench_synced
5. No testing for accuracy or assert_close calls in the benchmarks. Do that in the unit
test.
6. A GEMM benchmark that the tuning scripts (`aiter/ops/triton/utils/_triton/tuning/`) use
has a `make_bench_case(bencher, ...)` that returns `(launch, rot_bufs, flops, mem)`, where
`launch(config, **bufs)` takes a config (from `config_launch` in `utils/benchmark_utils.py`). Its
own `bench_gemm_fn` times through `make_bench_case` too, so the tuner and the CLI time the same
code. See `bench_gemm_a16w16.py`.
7. Benchmark only Triton ops. You may see some current benchmarks that do that, but they will be moved out in future.