# CK Unified Attention — status & how to run

Tooling + status for the CK `unified_attention` (UA) kernel work, focused on the
**fp8 d128 prefill** path on gfx950 (MI350/MI355).

| Repo | Branch | HEAD |
|---|---|---|
| `ROCm/aiter` | `jukorhon/unified-attention-ck-fav4` | (UA harness + analysis tooling) |
| `ROCm/composable_kernel` (submodule) | `jukorhon/fa4-k-preread` | `9aa380e6c` wide 32x32x64 fp8 MMA |

The aiter branch pins the CK submodule to the matching commit; a recursive
checkout is enough. Experiment branch `jukorhon/fa4-kv128-vgpr` (CK) carries the
default-off kv128 VGPR toggles.

---

## Status (2026-06-12)

**Perf, canonical fp8 prefill** (`b1 hq=hk=5 sq=sk=75600 d128 non-causal`,
FLOP ≈ 1.463e13). Kernel-only `@perftest` medians, idle GPU:

| kernel | TFLOPs | vs ASM-exact |
|---|---|---|
| CK-UA, narrow 32x32x16 MMA (pre-wide-MMA) | ~1171 | 0.56× |
| CK-UA, **wide 32x32x64 MMA** (current default) | ~1716 | 0.83× |
| ASM fp8 `fwd_hd128_fp8.co`, exact | ~2073 | 1.00× |
| ASM fp8, full (exp2 + rowmax-freeze) | ~2384 | 1.15× |

(CK 1716 / ASM-exact 2073 / ASM-full 2384 are the same idle-GPU `@perftest`
head-to-head after the kernel-only timing fix. An earlier standalone CUDA-event
pass put the wide-MMA CK kernel at ~1634; treat ~1.6–1.7k TF as the run-to-run
band.)

The remaining gap is **mostly structural** (~1.2×): symmetric 8-wave compute +
load-role specialization, finer load/MFMA interleave. The two numeric tricks
(Schraudolph `exp2` ~+8%, max-freeze that also skips the per-tile rowmax ~+5%)
add ~14% on top.

> ✅ **Fixed: prefill_fp8 accuracy regression.** The wide-MMA commit `9aa380e6c`
> shipped a "cvt-only, layouts coincide" P relayout for the 32x32x64 MMA. That
> assumption was wrong: QK-C holds one kv across many query rows while PV-A needs
> one query across many kv (a transpose), so the relayout MUST do the cross-lane
> `permlane32_swap` — the same one strategy A already does for 32x32x16 (both
> tiles share an identical 32x32 C-output distribution; only `kABKPerLane` 8→32
> changes). The bug was masked by near-uniform softmax. Fix: route K=64 through
> strategy A's permute and delete the cvt-only branch. prefill_fp8 + the full
> matrix now PASS at the loose fp8 tol; standalone perf holds ~1.66k TFLOPs.

**kv128 (KV tile) — why it doesn't fit:** see `kv128_vgpr_findings.md`. Short
version: at 256-VGPR/wave, kv64 uses 214 VGPR / 0 spill; naive kv128 hits the cap
with 173 spills. No register trick closes it (the score double-buffer is load-
bearing for the deferred-PV pipeline; biggest lever recovers 173→126). The only
fit is to keep the kv64 compute width and widen *only* the K/V load+barrier to
128 (sub-tile kBlockN), which captures DRAM/barrier amortization but not softmax
amortization. Not yet implemented; blocked behind the accuracy bug anyway.

---

## How to run the CK kernel

All commands from the aiter repo root. Pick an idle GPU with `rocm-smi --showuse`
(its index ≠ `HIP_VISIBLE_DEVICES`). `python3` here is the env the CK JIT modules
were built against.

**Correctness vs torch reference** (one shape):
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 2 -sq 8192 -sk 8192 --num-heads 12,2 --head-size 128 \
    --block-size 64 --dtype fp8 --num-blocks auto --no-triton --seed 42
# prints "CK vs ref: PASS/FAIL" + max abs delta
```

**Clean rebuild + full correctness matrix** (regression fixtures + shape matrix;
hard-deletes the .so and the build dir, forces `AITER_REBUILD=1`, verifies the
.so was regenerated):
```bash
HIP_VISIBLE_DEVICES=2 ua-test-scripts/rebuild_and_test.sh
SKIP_BUILD=1 ua-test-scripts/rebuild_and_test.sh   # test-only, reuse current .so
```

Key flags: `--dtype fp8|bf16`, `--num-heads HQ,HK`, `--block-size` (KV page;
fp8 needs ≥32), `--mask-type 0` (non-causal) `2` (causal), `--contiguous` (flip
the CK leg to the non-paged/THD kernel), `--no-triton`, `--no-reference`.

---

## How to compare perf

**vs Triton UA** (default comparison leg) — just omit `--no-triton`:
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 16,2 --head-size 128 \
    --block-size 64 --dtype fp8 --num-blocks auto --mask-type 0 --seed 42
```

**vs the ASM fp8 kernel** (`--contiguous --asmfp8`; dense, per-tensor fp8,
single-shape / non-SWA). Q/K/V are quantised once outside the timed region so
both legs measure kernel-only time:
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 64 --dtype fp8 --contiguous --asmfp8 --mask-type 0 \
    --no-triton --seed 42
# prints CK TFLOPs, ASMfp8 TFLOPs, and the UA-vs-ASM ratio
```
(Building/swapping the ASM `.co` itself is documented in the kernel repo:
`diffusion-models-inference-private/asm/fmha_sage_fwd/RUNNING_FP8.md`.)

**vs SageAttention-v1 (Triton):** add `--sagev1` (requires `--contiguous`).

**Standalone, JIT-free perf** (`standalone/perf.sh`) — compiles only the one
instance, stamped per build, so there's never a "stale JIT module?" question. One
build sweeps all sequence lengths (only dtype/d/mask changes trigger a rebuild):
```bash
GPU=2 DTYPE=fp8 ua-test-scripts/standalone/perf.sh 75600 5 5 128 0
SWEEP="1024 2048 4096 8192 16384" GPU=2 ua-test-scripts/standalone/perf.sh
# accuracy of the standalone setup vs a host float reference:
GPU=2 ua-test-scripts/standalone/check.sh 512 16 2 128 0
```

---

## How to get the trace overlay

Two paths produce the two-warp-group **MATRIX‖SOFTMAX overlap** timeline
(`overlap_simd0.png` + `report.md`). Good pipelining = one wave in MATRIX while
the other is in SOFTMAX; both in BARRIER/wait at once = a bubble.

**A) Standalone (fast, torch-free — seconds to trace):**
```bash
# build+trace+report; sq hq hk d mask iters
GPU=2 DTYPE=fp8 ua-test-scripts/standalone/run.sh 75600 5 5 128 0 3
# -> ua-test-scripts/rocprof_analysis/runs/att_std_d128_fp8_noncausal_sq75600/att_analysis/
```

**B) Through the PyTorch harness** (`att_analysis/run.sh`; needs the line-tables
build for ISA→C++ source mapping). `LINETABLES=1` injects `-gline-tables-only`
without changing `-O3` codegen; `SLIM=1` (default) builds only the traced
instance (after a SLIM build, other shapes fail with "no matching kernel" until
you rebuild with `SLIM=0 LINETABLES=1`):
```bash
LINETABLES=1 ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000      # rebuild+trace+report
ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000 0 3                # reuse build; simd 0, 3 iters
```

**Raw ATT for ROCprof Compute Viewer (RCV) GUI** (`rocprof_att_prefill.sh`):
```bash
ua-test-scripts/rocprof_att_prefill.sh fp8 1 75600 75600
# tarball of ui_output_* printed at the end; scp + open in rocprof-compute-viewer
```
See `att_analysis/README.md` for the overlay internals and how to read it.

---

## VGPR / spill probe

Fast per-instance VGPR/spill readout (~20s, no full JIT build); `XCFLAGS` injects
`-D` toggles:
```bash
LABEL=kv64 ua-test-scripts/measure_vgpr.sh
XCFLAGS="-DUA_PREFILL_D128_BLOCKSIZE=128 -DUA_FA4_SHARED_SPCOMPUTE=1" \
    LABEL=kv128 ua-test-scripts/measure_vgpr.sh
```

---

## Script inventory

| path | purpose |
|---|---|
| `rebuild_and_test.sh` | clean rebuild + correctness matrix (stale-proof) |
| `standalone/` | JIT-free driver: `build.sh` `perf.sh` `check.sh` `run.sh` (+`ua_trace_main.cpp`) |
| `att_analysis/` | headless overlap-timeline analyzer (`run.sh` + python modules; see its README) |
| `rocprof_att_prefill.sh` | collect raw ATT for the RCV GUI |
| `measure_vgpr.sh` | per-instance VGPR/spill probe |
| `kv128_vgpr_findings.md` | full kv128 VGPR analysis + the prefill_fp8 bisection |
| `isa_source_map.sh` | disassemble one instance with interleaved C++ source |
| `rocprof_*.py`, `rocprof_prefill_d128.sh`, `regression_decode.sh`, `sweep_amir_shapes.py`, `analyze_sweep.py`, `addr_breakdown.py` | older rocprof/sweep analysis helpers |

Generated artifacts (`rocprof_analysis/`, `*.log`, `*.csv`, `vgpr_probe/`,
`standalone/build/`, `isa_analysis*/`) are gitignored and safe to delete; the run
scripts regenerate them.
