---
name: run-bench
description: Run a model's gfx1250 operator microbench suite (op_tests/bench_gfx1250_combo.py --dsr1, --dsv41, ...) and turn the result into one self-contained report.html with the run command, environment and per-case performance (DSR1 also gets operators, shapes and kernels). Use when asked to benchmark, re-measure or report a model's kernels at its serving shapes, or to compare them across aiter commits or PRs.
argument-hint: <suite> <out_dir> [gpus]
---

# Run bench

One command runs a suite and writes the report:

```bash
cd /app/aiter            # the aiter checkout under test
bash .claude/skills/run-bench/run.sh dsr1  /data/$USER/bench_runs/<tag> 3
bash .claude/skills/run-bench/run.sh dsv41 /data/$USER/bench_runs/<tag> 0,1,2,3
```

`<suite>` is a combo suite flag without the dashes. `[gpus]` becomes
`HIP_VISIBLE_DEVICES`: single-GPU ops run on the first one, `mega_moe` needs four.
On a shared node with `gpu-lock`, put it in front: `gpu-lock bash .../run.sh ...`.

`<out_dir>` ends up with `combo.log` (the suite's JSON tables), `provenance.txt`
(host, suite, aiter commit, ROCm / torch / triton / FlyDSL versions, combo sha256,
exact command), `status.txt` (start, end, exit code) and `report.html`. Rebuild the
report from an existing directory with `python3 .claude/skills/run-bench/report.py <out_dir>`.

Options: `OPS="moe mla_v4_decode"` picks operators, `AITER_BENCH_TOKENS=1,1536`
overrides every sweep, `IMAGE=<tag@digest>` records the image (it is not visible
from inside the container).

To add a model suite, see the `add-bench-suite` skill.

## Before running

1. The GPUs must be idle (`amd-smi monitor -v -u`): other processes skew every number.
2. Record what is under test: the aiter commit and any PR applied on top. A run
   against a different aiter is a different result; keep each in its own `<out_dir>`.
3. After switching aiter commits, clear stale JIT builds (`aiter/jit/*.so`,
   `aiter/jit/build`, `~/.aiter`, `~/.flydsl`, `~/.triton`) and install the FlyDSL
   version `requirements.txt` pins.

## Reading the result

- `status.txt` exit code 0 and the header's failed-case count 0 mean every case
  ran.
- MoE GB/s is the grouped-GEMM UT's own effective-bandwidth model. Report it as
  is; do not patch it here.
- MoE tile configs come from `aiter/configs/tuned_grouped_fmoe.csv` and the
  model configs beside it; note which config the run used.

### dsr1

- MoE runs the plain grouped GEMM (`fused_moe` -> `grouped_moe_gfx1250`). The S1F
  stage-1 fusion (dispatch + GEMM1) exists only inside MegaMoE, so it is not
  measured. Only gemm1 and gemm2 are timed; the activation quant kernel is not.
- MLA decode checks each case against the UT's fp32 reference (`err`). Cases
  above 32M KV tokens (batch x ctx) are skipped by design: they do not fit the UT
  on one GPU.
- Heads are per GPU. DP attention keeps all 128 query heads on every GPU; decode
  reads one latent KV head all of them share, prefill is causal (`is_causal=True`,
  as in ATOM's non-chunked prefill) MHA with 128 KV heads.
- Prefill times the asm kernel and `flash_attn_varlen_func` (the call ATOM
  makes). The UT's FlyDSL candidate is switched off: DSR1 never runs it.

### dsv41

- Shapes are one rank of the V4.1-Flash DEP4 recipe (EP4 + DP attention, 64 heads
  a rank); token counts are per rank.
- `mega_moe` rows come in pairs: `default` is bench_mega_moe.py's own setup,
  `recipe` adds the recipe's MegaMoE environment (fp4 dispatch wire, S1F, compact
  route, IOD-local weights). It needs four GPUs.
- MLA decode / prefill run bf16 (opus, triton) and v41_fp4 (asm) at 64 heads.
