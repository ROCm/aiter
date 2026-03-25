# Blockscale Tuning Notes

## Key Learnings

### GPU Process Cleanup
When killing tuning processes (pkill screen.py/ut_*.py), GPU contexts become stale
and hold memory. `rocm-smi --showpidgpus` shows them even after processes die.
`kill -9` does NOT release them. Only `rocm-smi --gpureset` clears them, but that
kills active work too. Always check for stale contexts before starting new tuning.

### Baseline Collection
- Must use `rocprof --stats` (not rocprofv3) — parse `results.stats.csv` column 4 (AverageNs)
- Run sequentially on single GPU to avoid file conflicts (rocprof writes to cwd)
- Preshuffle needs `-preshuffle` flag on bench script (only on feature branch)
- Switch to main branch + Triton 3.4 for baselines

### Tuning Search Space
- BK=128 only (kernel constraint: GROUP_K == BLOCK_SIZE_K)
- For K=1536: must use `--num-ksplit-range 1 2 3 4 6 8 12` (default range causes failures)
- For K=512: must use `--num-ksplit-range 1 4`
- `matrix_instr_nonkdim=32` + `num_warps=2` found major improvements for several shapes
- `GROUP_SIZE_M=32` helped M=8192 shapes significantly
- Always include `--num-stages-range 2 3`
- M=8192 shapes with large N*K are very slow to tune (hours per shape)

### Validation
- Sequential on single GPU with `rocprof --stats` for clean measurements
- Parallel validation gave noisy results due to stale GPU contexts
- Regression tolerance: `new > old * 1.005 + 500ns`

## Final Results

### Non-preshuffle (gemm_a8w8_blockscale)
- 1.513x geomean, 143/144 improved, 1 regression
- Sole regression: M=64 N=7168 K=16384 (+9.4%) — exhaustive search found no fix
- 18/18 (N,K) pairs pass geomean >= 1.0

### Preshuffle (gemm_a8w8_blockscale_preshuffle)
- 4.090x geomean, 104/104 improved, 0 regressions
- 13/13 (N,K) pairs pass geomean >= 1.0

## Branch Layout
- `alizaidy/gfx950-kernel-fixes-cherry-picked`: All tuned configs + bench script changes
- `alizaidy/blockscale-tuning-state`: Scripts, baselines, validation JSONs
