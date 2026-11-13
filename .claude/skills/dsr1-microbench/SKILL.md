---
name: dsr1-microbench
description: Run the DeepSeek-R1 MLPerf operator microbench on gfx1250 (MoE FC1/FC2 and MLA decode/prefill at the MLPerf per-GPU shapes, via op_tests/bench_gfx1250_combo.py --dsr1) and turn the result into one self-contained report.html with the operators, shapes, kernels, environment, run command and per-case performance. Use when asked to benchmark, re-measure or report DSR1 MoE/MLA kernels, or to compare them across aiter commits or PRs.
argument-hint: <out_dir> [gpu]
---

# DSR1 MLPerf microbench

One command runs the suite and writes the report:

```bash
cd /app/aiter            # the aiter checkout under test
bash .claude/skills/dsr1-microbench/run.sh /data/$USER/dsr1_runs/<tag> 3
```

`<out_dir>` ends up with `combo.log` (the suite's JSON tables), `provenance.txt`
(host, aiter commit, ROCm / torch / triton / FlyDSL versions, combo sha256, exact
command), `status.txt` (start, end, exit code) and `report.html`. Rebuild the
report from an existing directory with `python3 .claude/skills/dsr1-microbench/report.py <out_dir>`.

Options: `OPS="moe mla_decode"` picks operators, `AITER_BENCH_TOKENS=1,1536`
overrides the bs / T sweep, `IMAGE=<tag@digest>` records the image (it is not
visible from inside the container).

## Before running

1. The GPU must be idle (`amd-smi monitor -v -u`): other processes skew every number.
2. Record what is under test: the aiter commit and any PR applied on top. A run
   against a different aiter is a different result; keep each in its own `<out_dir>`.
3. After switching aiter commits, clear stale JIT builds (`aiter/jit/*.so`,
   `aiter/jit/build`, `~/.aiter/build`) and install the FlyDSL version
   `requirements.txt` pins.

## Reading the result

- `status.txt` exit code 0 and the header's failed-case count 0 mean every case
  ran; MLA decode also checks each case against the UT's fp32 reference (`err`).
- MoE GB/s is the grouped-GEMM UT's own effective-bandwidth model. Report it as
  is; do not patch it here.
- MoE runs the plain grouped GEMM (`fused_moe` -> `grouped_moe_gfx1250`). The S1F
  stage-1 fusion (dispatch + GEMM1) exists only inside MegaMoE, so it is not
  measured. Only gemm1 and gemm2 are timed; the activation quant kernel is not.
- MoE tile configs come from `aiter/configs/tuned_grouped_fmoe.csv`; rows for
  65 experts / top-9 / 7168 / 2048 change the MoE numbers, so note which config
  the run used.
- MLA decode cases above 32M KV tokens (batch x ctx) are skipped by design: they
  do not fit the UT on one GPU.
- Heads are per GPU. DP attention keeps all 128 query heads on every GPU; decode
  reads one latent KV head all of them share, prefill is causal (`is_causal=True`,
  as in ATOM's non-chunked prefill) MHA with 128 KV heads.
- Prefill times the asm kernel and `flash_attn_varlen_func` (the call ATOM
  makes). The UT's FlyDSL candidate is switched off: DSR1 never runs it.
- The MoE sweep starts at bs 4.
