# CK Unified Attention — status & how to run

Tooling + status for the CK `unified_attention` (UA) kernel work on gfx950
(MI350/MI355): **fp8 / bf16 / fp16**, **prefill + decode**, contiguous + paged
KV.

| Repo | Branch |
|---|---|
| `ROCm/aiter` | `jukorhon/unified-attention-ck-fav4` (UA harness + analysis tooling) |
| `ROCm/composable_kernel` (submodule) | `jukorhon/fa4-k-preread` (the kernel) |

The aiter branch pins the CK submodule to the matching commit; a recursive
checkout is enough.

---

## Status (2026-06-16)

**Correctness:** the full matrix is green — `--full` is **263/263 PASS, 0 fail**
(32 expected skips: fp8 with `block_size < 32`), including every split-KV /
causal / non-dividing-GQA and prefill regression fixture.

**Perf vs Triton UA**, FP8 production shapes (GQA-6 `Hq,Hk=12,2`, `d=128`,
`block_size=64`), `@perftest` medians on an idle MI355 — from
`sweep_amir_shapes.py`:

| regime | shape band | CK vs Triton |
|---|---|---|
| **Prefill** (`Sq=Sk`) | all of `b∈{4..32}`, `Sq∈{1k,5k,10k}` | **1.08×–1.44× (CK wins every cell)**, median ~1.3× |
| **Decode** (`Sq=1`) | short ctx `Sk≤1k`, or any `Sk` at tiny batch | **1.1×–1.5× (CK wins)** |
| **Decode** (`Sq=1`) | long ctx `Sk≥10k`, mid/large batch | **0.83×–0.97× (Triton wins)** |

Decode at long context / larger batch is **HBM-bandwidth bound** and is the open
work item — see `decode_pipeline_research_plan.md`. (The lone sweep failure,
`b512×Sk196608`, is a *host-side* OOM in the test harness allocating the 96 GiB
fp8-quant reference, not a kernel error.)

**Canonical prefill TFLOPs** (`b1 hq=hk=5 sq=sk=75600 d128 non-causal`, idle GPU):
CK-UA fp8 wide 32×32×64 MMA ≈ **1.7k TFLOPs** (~0.83× of the ASM-exact fp8
kernel; the remaining ~1.2× gap is structural — symmetric 8-wave compute + finer
load/MFMA interleave).

> ✅ **Recent: bf16/fp16 paged `block_size=128` now at contiguous parity.**
> `prefill_d128` gained constexpr `ps128` instances for bf16/fp16 (was falling
> into the `PageSize=0` runtime-divide catch-all → ~54% slower paged). On top,
> the single-page SRD rebase was **decoupled from the scalar-promote gate** so
> single-issue (`NRepeat==1`) tiles also fold the page base into the SRD and hoist
> the per-lane scatter offsets — the paged `addr` phase disappears from the ATT
> trace. bf16 paged causal canonical: **9.97 ms, 1.19× over Triton** (was 10.5 ms
> / 1.13×; the old paged-vs-contiguous "50% cliff" is now ~13%). fp8 paged
> unchanged (1.64×).

---

## How to run

All commands from the aiter repo root. Pick an idle GPU with `rocm-smi --showuse`
(its index = the `HIP_VISIBLE_DEVICES` value). `python3` here is the env the CK
JIT modules were built against.

### Correctness

**One shape, vs torch reference:**
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 2 -sq 8192 -sk 8192 --num-heads 12,2 --head-size 128 \
    --block-size 64 --dtype fp8 --num-blocks auto --no-triton --seed 42
# prints "CK vs ref: PASS/FAIL" + max abs delta
```

**Full correctness matrix** (~290 configs incl. all regression fixtures, ~3 min):
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --full
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --quick   # smoke subset
```

**Clean rebuild + matrix** (stale-proof: hard-deletes the `.so` + build dir,
forces `AITER_REBUILD=1`, verifies the `.so` was regenerated):
```bash
HIP_VISIBLE_DEVICES=2 ua-test-scripts/rebuild_and_test.sh
SKIP_BUILD=1 ua-test-scripts/rebuild_and_test.sh   # test-only, reuse current .so
```

Key flags: `--dtype fp8|bf16|fp16`, `--num-heads HQ,HK`, `--block-size` (KV page;
fp8 needs ≥32), `--mask-type 0` (non-causal) `2` (causal), `--contiguous` (flip
the CK leg to the non-paged/THD kernel), `--no-triton`, `--no-reference`.

### Perf comparison

**vs Triton UA, one shape** — just omit `--no-triton`; prints the `UA vs Triton`
ratio + TFLOPs/BW for both legs:
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 128 --dtype bf16 --num-blocks auto --mask-type 2 --triton --seed 42
```

**vs Triton UA, the production (Amir) shape sweep** — the boss-report deliverable.
Sweeps the decode + prefill bands from the vLLM trace and writes a CSV:
```bash
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2          # full, ~7 min -> sweep_amir_shapes.csv
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2 --quick  # ~10 cells
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2 --phase decode --dtype bf16
```

**vs the ASM fp8 kernel** (`--contiguous --asmfp8`; dense per-tensor fp8,
single-shape). Q/K/V are quantised once outside the timed region:
```bash
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 64 --dtype fp8 --contiguous --asmfp8 --mask-type 0 --no-triton --seed 42
```

**vs SageAttention-v1 (Triton):** add `--sagev1` (requires `--contiguous`).

**Standalone, JIT-free perf** (`standalone/perf.sh`) — compiles only the one
instance, stamped per build, so there's never a "stale JIT module?" question:
```bash
GPU=2 DTYPE=fp8 ua-test-scripts/standalone/perf.sh 75600 5 5 128 0
SWEEP="1024 2048 4096 8192 16384" GPU=2 ua-test-scripts/standalone/perf.sh
GPU=2 ua-test-scripts/standalone/check.sh 512 16 2 128 0   # accuracy vs host float ref
```

### Trace overlay (two-warp-group MATRIX‖SOFTMAX timeline)

Produces `overlap_simd0.png` + `report.md`. Good pipelining = one wave in MATRIX
while the other is in SOFTMAX; both in BARRIER/wait at once = a bubble.

```bash
# A) Standalone (fast, torch-free):  sq hq hk d mask iters
GPU=2 DTYPE=fp8 ua-test-scripts/standalone/run.sh 75600 5 5 128 0 3
# paged path: PAGED=1 + PAGE_BLK matching a built instance (must divide into the 32-token tile):
PAGED=1 PAGE_BLK=128 DTYPE=bf16 GPU=2 \
    TARGET_INSTANCE=unified_attention_d128_bf16_mask_ps128 \
    ua-test-scripts/standalone/run.sh 75600 5 5 128 2 3
# -> ua-test-scripts/rocprof_analysis/runs/<tag>/att_analysis/
```
```bash
# B) Through the PyTorch harness (ISA->C++ source mapping via line tables):
LINETABLES=1 ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000   # rebuild+trace+report
ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000 0 3            # reuse build
```
The standalone tracer forces `num_splits=1`, so the per-split page-table window
must fit the 4096-entry LDS cache: `PAGE_BLK=16` at `Sk=75600` needs 4725 entries
and will fault — trace ps32/64/128 instead. See `att_analysis/README.md`.

### VGPR / spill probe

```bash
LABEL=kv64 ua-test-scripts/measure_vgpr.sh
XCFLAGS="-DUA_PREFILL_D128_BLOCKSIZE=128" LABEL=kv128 ua-test-scripts/measure_vgpr.sh
```

---

## Script inventory

| path | purpose |
|---|---|
| `rebuild_and_test.sh` | clean rebuild + correctness matrix (stale-proof) |
| `sweep_amir_shapes.py` | production-shape perf sweep vs Triton → CSV (boss report) |
| `analyze_sweep.py` | diff two sweep CSVs (`--pre A.csv --post B.csv`) |
| `standalone/` | JIT-free driver: `build.sh` `perf.sh` `check.sh` `run.sh` (+`ua_trace_main.cpp`) |
| `att_analysis/` | headless overlap-timeline analyzer (`run.sh` + python modules; see its README) |
| `rocprof_att_prefill.sh` | collect raw ATT for the ROCprof Compute Viewer GUI |
| `measure_vgpr.sh` | per-instance VGPR/spill probe |
| `decode_pipeline_research_plan.md` | the decode (bandwidth-bound) optimization plan |
| `kv128_vgpr_findings.md` | full kv128 VGPR analysis + the prefill_fp8 bisection |
| `isa_source_map.sh` | disassemble one instance with interleaved C++ source |

Generated artifacts (`rocprof_analysis/`, `*.log`, `*.csv`, `vgpr_probe/`,
`standalone/build*/`, `isa_analysis*/`) are gitignored and safe to delete; the run
scripts regenerate them.
