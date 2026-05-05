# zero-init splitK demo results

Outputs from `op_tests/run_zero_init_demo.sh` and `op_tests/bench_zero_init_splitk_demo.py`.

Each timestamped sub-directory has three CSVs (`none.csv`, `splitk.csv`, `splitk_fused.csv`) with median µs for the producer-quant + bpreshuffle CKTile GEMM combo across the Qwen3-Next-80B-A3B per_1x128 decode shape set.

Run with `TRACE=1` to additionally capture one `torch.profiler` chrome trace per shape per config under `traces/`.  In `splitk` the trace shows a per-iteration `FillFunctor<BFloat16>` kernel between the producer-quant and the GEMM; in `splitk_fused` that fill kernel is absent (its work is absorbed into the producer's grid-strided uint4 zero-fill loop).
