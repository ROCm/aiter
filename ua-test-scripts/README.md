# CK Unified Attention — status & how to run

Tooling + status for the CK `unified_attention` (UA) kernel work on gfx950
(MI350/MI355): **fp8 / bf16 / fp16**, **prefill + decode**, contiguous + paged KV.

| Repo | Branch |
|---|---|
| `ROCm/aiter` | `jukorhon/unified-attention-ck-fav4` (UA harness + analysis tooling) |
| `ROCm/composable_kernel` (submodule) | `jukorhon/fa4-k-preread` (the kernel) |

The aiter branch pins the CK submodule to the matching commit; a recursive
checkout (`git submodule update --init --recursive`) is enough. The kernel source
and its architecture notes live in
`3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/` (see the
README there).

---

## Layout

```
ua-test-scripts/
  README.md                       <- you are here: correctness + perf
  rebuild_and_test.sh             <- clean rebuild + correctness matrix (stale-proof)
  sweep_amir_shapes.py            <- production-shape perf sweep vs Triton -> CSV
  analyze_sweep.py                <- diff two sweep CSVs (--pre A.csv --post B.csv)
  regression_decode.sh            <- decode regression guard
  decode_pipeline_research_plan.md / decode_pipeline_findings.md  <- decode (BW-bound) optimization notes
  kv128_vgpr_findings.md          <- kv128 VGPR analysis + the prefill_fp8 bisection
  analysis/                       <- deep-dive profiling/trace/ISA tooling (see analysis/README.md)
```

---

## ⚠️ Fresh build first — never trust a stale `.so`

The JIT module `aiter/jit/module_unified_attention.so` is **not** rebuilt
automatically when you edit kernel source, so any perf/correctness run can
silently use an old kernel. Before trusting a number, force a clean rebuild:

```bash
rm -rf aiter/jit/build/module_unified_attention aiter/jit/module_unified_attention.so
AITER_REBUILD=1 HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py ...
```

`rebuild_and_test.sh` does this for you (hard-deletes the `.so` + build dir, sets
`AITER_REBUILD=1`, and verifies the `.so` was regenerated). When switching git
commits, **always** wipe the `.so` + build dir first. For ambiguity-free perf, the
standalone driver in `analysis/` stamps every build — see `analysis/README.md`.

---

## Status (2026-06-17)

**Correctness:** the full matrix is green — `--full` is **263/263 PASS** (32
expected skips: fp8 with `block_size < 32`), including every split-KV / causal /
non-dividing-GQA and prefill regression fixture.

**Perf vs Triton UA** (FP8 GQA-6 `Hq,Hk=12,2`, `d=128`, `block_size=64`,
`@perftest` medians on an idle MI355, from `sweep_amir_shapes.py`):

| regime | shape band | CK vs Triton |
|---|---|---|
| **Prefill** (`Sq=Sk`) | all of `b∈{4..32}`, `Sq∈{1k,5k,10k}` | **1.08×–1.44× (CK wins every cell)**, median ~1.3× |
| **Decode** (`Sq=1`) | short ctx `Sk≤1k`, or any `Sk` at tiny batch | **1.1×–1.5× (CK wins)** |
| **Decode** (`Sq=1`) | long ctx `Sk≥10k`, mid/large batch | **~0.92× (Triton wins)** |

Decode at long context is **HBM-bandwidth bound** (CK ≈ 4.5 TB/s vs Triton ≈ 4.9
TB/s of an ~8 TB/s peak) — see `decode_pipeline_research_plan.md` /
`decode_pipeline_findings.md`. This is *not* a regression: a commit bisect showed
CK decode has been flat (~88 µs on the `b4 Sk196608` shape) across the whole
branch; the head-to-head only flipped because upstream Triton got faster. Merging
the latest `origin/main` did not change it further (the new Triton Gluon UA kernel
is gfx1250-only). Closing the gap requires genuine CK load/access-efficiency work,
not deeper prefetch (the multi-stage ring experiment was perf-neutral).

---

## How to run

All commands from the aiter repo root; `python3` = the env the CK JIT modules were
built against. Pick an idle GPU with `rocm-smi --showuse`.

### Correctness

```bash
# One shape, vs torch reference:
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 2 -sq 8192 -sk 8192 --num-heads 12,2 --head-size 128 \
    --block-size 64 --dtype fp8 --num-blocks auto --no-triton --seed 42
# -> "CK vs ref: PASS/FAIL" + max abs delta

# Full matrix (~290 configs incl. all regression fixtures, ~3 min):
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --full
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --quick   # smoke subset

# Clean rebuild + matrix (stale-proof):
HIP_VISIBLE_DEVICES=2 ua-test-scripts/rebuild_and_test.sh
SKIP_BUILD=1 ua-test-scripts/rebuild_and_test.sh   # test-only, reuse current .so
```

Key flags: `--dtype fp8|bf16|fp16`, `--num-heads HQ,HK`, `--block-size` (KV page;
fp8 needs ≥32), `--mask-type 0` (non-causal) / `2` (causal), `--contiguous` (flip
the CK leg to the non-paged/THD kernel), `--no-triton`, `--no-reference`.

### Perf comparison

```bash
# vs Triton UA, one shape — omit --no-triton; prints the "UA vs Triton" ratio:
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 128 --dtype bf16 --num-blocks auto --mask-type 2 --triton --seed 42

# vs Triton UA, the production (Amir) shape sweep -> CSV:
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2            # full, ~7 min
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2 --quick    # ~10 cells
python3 ua-test-scripts/sweep_amir_shapes.py --gpu 2 --phase decode --dtype bf16

# Diff two sweep CSVs (e.g. before/after a change):
python3 ua-test-scripts/analyze_sweep.py --pre old.csv --post new.csv

# vs the ASM fp8 kernel (dense per-tensor fp8, single shape):
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 64 --dtype fp8 --contiguous --asmfp8 --mask-type 0 --no-triton --seed 42
```

`--sagev1` (requires `--contiguous`) adds a SageAttention-v1 (Triton) baseline.

### Deeper analysis (overlap timelines, VGPR, ISA, rocprof)

See **`analysis/README.md`**. Quick pointers:
```bash
GPU=2 DTYPE=fp8 ua-test-scripts/analysis/standalone/perf.sh 75600 5 5 128 0   # JIT-free perf
GPU=2 DTYPE=fp8 ua-test-scripts/analysis/standalone/run.sh  75600 5 5 128 0 3 # overlap timeline
LABEL=kv64      ua-test-scripts/analysis/measure_vgpr.sh                       # VGPR/spill probe
```
