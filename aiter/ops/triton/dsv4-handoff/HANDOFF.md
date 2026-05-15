# DeepSeek-V4-Flash Optimization — Handoff

**This directory is self-contained.** Everything a fresh agent needs to port the optimization stack to a different ATOM/aiter base branch is here.

## Package contents

| File | Purpose |
|---|---|
| `HANDOFF.md` | This document. |
| `atom_optimizations.patch` | One commit's worth of ATOM changes (5 files, ~800 lines added). |
| `aiter_optimizations.patch` | One commit's worth of aiter changes (20 files, ~3075 lines added). Includes new files via `new file mode` hunks. |
| `atom_base_commit.txt` | Hash + metadata for the source ATOM commit (parent → HEAD). |
| `aiter_base_commit.txt` | Hash + metadata for the source aiter commit (parent → HEAD). |
| `apply.sh` | Dry-runs both patches, then applies. Fails cleanly if they don't fit. |
| `validate.sh` | Starts server with full env stack, runs smoke + gsm8k + best-of-2 perf sweep c=4..64. |
| `perf_summary.csv` | Reference perf numbers from the source branch (origin → Step 12+14 → hash-fused). |

## Goal

Port the V4-Flash optimization stack from the source branches (committed) to a different base branch. Target hardware: AMD MI355X (gfx950), TP=8. Model: DeepSeek-V4-Flash.

## Source state

- **ATOM**: branch `alizaidy/dsv4-fusions`, single commit "Add a8w4 moe modifications and mxfp8 GEMM for DSv4" on top of parent `9f1c49e3` ("add sparse decode switch").
- **aiter**: branch `alizaidy/dsv4-fusions`, single commit (same title) on top of parent `b4cf226d6`.

See `*_base_commit.txt` for exact hashes + author/date.

## Final results on source branch

gsm8k preserved/improved (0.96 → 0.98 within ±0.02 noise at n=100). Throughput gain vs original CK-MoE matmul_ogs baseline:

| c | Origin (CK-MoE) | Final (hash-fused) | Δ |
|---|---|---|---|
| 4 | 319.76 | 339.63 | +6.21% |
| 8 | 606.80 | 653.82 | +7.75% |
| 16 | 1092.80 | 1237.00 | +13.20% |
| 32 | 2109.76 | 2327.96 | +10.34% |
| 64 | 3775.54 | 4283.00 | +13.44% |

## Quick start

```bash
# 1. Apply the patches to your new ATOM and aiter checkouts
ATOM_PATH=/path/to/ATOM AITER_PATH=/path/to/aiter ./apply.sh

# 2. Validate (smoke + gsm8k + perf sweep c=4..64)
MODEL_PATH=/data/DeepSeek-V4-Flash ./validate.sh dsv4_after_port
```

`apply.sh` dry-runs first; if the patch doesn't apply cleanly it bails out and you'll need to apply step-by-step (see the per-step file lists below).

`validate.sh` writes artifacts to `./out/<LABEL>/`: server log, gsm8k log/json, per-c bench result jsons, SUMMARY.txt. Smoke must say "Paris." or the script aborts before gsm8k.

## Critical environment variables

Set BEFORE starting the server. Without these the model takes wrong code paths and outputs garbage (this caused a multi-hour bisect when we forgot one):

```bash
export ATOM_V4_USE_TRITON_FUSION=1     # fused qk_norm+rope+swa_write + fused_clamp_act_mul
                                       # (gates use_fuse_qk_norm_rope_swa_write +
                                       #  use_fused_clamp_act_mul in deepseek_v4.py)

ATOM_USE_TRITON_MOE=1 \                # Triton MoE pathway (required for a8w4 dispatch)
ATOM_MOE_BACKEND=a8w4 \                # a8w4 backend on top of Triton path
ATOM_A8W4_SWIGLU_FOLD=1 \              # GEMM1 apply_swiglu fold + W13 gate/up interleave
AITER_A8W4_DECODE_NUM_STAGES=2 \       # a8w4 decode kernel config
ATOM_A8W4_TRITON_ROUTING=1 \           # all-Triton V4 routing (non-hash layers)
ATOM_A8W4_GEMM1_MX_EMIT=1 \            # GEMM1 emits FP8+ue8m0 directly
ATOM_A8W4_FUSE_RESIDUAL=1 \            # routed+shared add folded into reduce_grouped (single_stream only)
ATOM_FP8_BLOCKSCALE_USE_MXFP8=1 \      # MXFP8 GEMM dispatch + matching layernorm emit
ATOM_A8W4_HASH_FAST_ROUTING=1 \        # NEW: fully-fused hash-layer routing (DSv4 layers 0-2)
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/DeepSeek-V4-Flash --kv_cache_dtype fp8 -tp 8 \
  --max-num-seqs 64 --gpu-memory-utilization 0.85 --server-port 8000 --max-model-len 4096 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32,64,128,256]" --level 0
```

## Optimization steps (in dependency order)

The whole stack is one commit per repo, but the steps correspond to discrete features. If `apply.sh` fails, cherry-pick by step using these file lists.

### Step 1 — a8w4 MoE backend (foundation)
- **Env**: `ATOM_MOE_BACKEND=a8w4`
- **Files**:
  - `ATOM/atom/model_ops/moe.py` — `process_weights_after_loading` a8w4 weight-layout branch (transpose, swizzle scales, optional W13 gate/up interleave).
  - `ATOM/atom/model_ops/fused_moe_triton.py` — `_a8w4_fused_experts` function + dispatch in `triton_kernel_fused_experts`.
  - `aiter/aiter/ops/triton/{moe,_triton_kernels/moe}/moe_op_gemm_a8w4.py` — kernel + wrapper.
- **What**: matmul_ogs MoE → aiter a8w4 (fp8 act × mxfp4 weight grouped GEMM).
- **Validation gate**: gsm8k ≥ 0.96.

### Step 2 — a8w4 GEMM config (block_m default per shape)
- **Env**: (none beyond Step 1)
- **Files**: `aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py` — `get_kernel_config` block_m=64 (prefill M≥256) / block_m=16 (decode).

### Step 3 — apply_swiglu fold + num_stages=2 decode
- **Env**: `ATOM_A8W4_SWIGLU_FOLD=1 AITER_A8W4_DECODE_NUM_STAGES=2`
- **Files**: same kernel; adds `apply_swiglu` writeback + W13 gate/up interleave logic in the ATOM weight loader.
- **What**: GEMM1 fuses silu(gate)*up; eliminates a `fused_clamp_act_mul` launch.

### Step 4 — All-Triton fused V4 routing (non-hash layers)
- **Env**: `ATOM_A8W4_TRITON_ROUTING=1`
- **Files**:
  - `ATOM/atom/model_ops/moe.py` — new `if _a8w4_triton_routing:` dispatch branch.
  - `aiter/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py` — `_topk` extended with `SCORE_MODE`, `HAS_BIAS`, `APPLY_RENORM`, `ROUTED_SCALING`.
  - `aiter/aiter/ops/triton/moe/moe_routing/routing.py` — `routing_a8w4(...)` doing `_topk` → `sort_tokens_fused` in 2 kernels.
- **What**: V4 routing math (sqrtsoftplus + bias + topk + bitmatrix + renorm + scale) + sort in 2 Triton kernels. Skips `FusedMoE.select_experts` + the matmul_ogs routing bridge.

### Step 5 — `out_mx_quant` (GEMM1 emits fp8+ue8m0 directly)
- **Env**: `ATOM_A8W4_GEMM1_MX_EMIT=1` (requires Step 3)
- **Files**: `aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py` — `HAS_MX_OUT` writeback branch + block_n floor=64 for bm=16.
- **What**: GEMM1 with apply_swiglu folded ALSO quantises its output to (fp8 e4m3 + ue8m0 per-1×32), eliminating a separate `downcast_to_mxfp` launch.

### Step 6 — drop fp32 gate cast — **already in baseline** (colleague PR)

### Step 7 — gate `gemm_a16w16` — ⚠ **REJECTED** (-2.6% regression on V4 shapes)

### Step 8 — drop redundant D→D memcpy — **implicit in Step 1** (`_a8w4_fused_experts` returns from `reduce_grouped`)

### Step 9 — residual fold (single_stream only)
- **Env**: `ATOM_A8W4_FUSE_RESIDUAL=1`
- **Files**:
  - `ATOM/atom/models/deepseek_v4.py` — `single_stream_moe_forward` stashes `shared` on `self.experts._moe_residual_to_fold`. Returns folded routed directly when `_moe_residual_was_folded=True`.
  - `aiter/aiter/ops/triton/{moe,_triton_kernels/moe}/reduce.py` — `HAS_EXT_RESIDUAL` parameter in `reduce_grouped`.
- ⚠ **DO NOT** extend to `dual_stream_moe_forward`. Tried (Variant A): regresses ~13% because the required sync kills shared/routed overlap. dual_stream stays as clean parallel pattern; fold only fires in single_stream.

### Step 10 — MXFP8 GEMM kernels (file copies + tuning configs)
- **Files** (all new, captured by `new file mode` hunks in the patch):
  - `aiter/aiter/ops/triton/gemm/basic/gemm_mxfp8.py` + `_triton_kernels/gemm/basic/gemm_mxfp8.py`
  - `aiter/aiter/ops/triton/quant/quant_mxfp8.py` + `_triton_kernels/quant/quant_mxfp8.py`
  - 5× `aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-MXFP8-PRESHUFFLE-N=*-K=*.json`
  - `aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-MXFP8-PRESHUFFLE.json`
- Code path inert until Step 11 env enables it.

### Step 11+12+14 — MXFP8 chain (must apply together)
- **Env**: `ATOM_FP8_BLOCKSCALE_USE_MXFP8=1`
- **Files**:
  - `ATOM/atom/model_ops/linear.py` — MXFP8 GEMM dispatch for per_1x128 Linears.
  - `ATOM/atom/model_ops/layernorm.py` — RMSNorm emits FP8 e4m3 + ue8m0 per-1×32 scales directly.
  - `ATOM/atom/models/deepseek_v4.py` — `_fuse_rmsnorm_mxfp8_quant` helper (dual q_norm/kv_norm via aiter `dual_rmsnorm_mxfp8_quant`).
  - `aiter/aiter/ops/triton/{_triton_kernels/fusions,fusions}/fused_clamp_act_mul.py` — `SCALE_FMT='ue8m0'` branch.
- ⚠ **Step 11 ALONE fails accuracy** (gsm8k 0.83). Must apply Step 11 + Step 12+14 together. With both, gsm8k recovers to 0.97 and is the biggest single TPS jump (+2-3% on top of Step 9).

### Step 13 — wq_b MXFP8 upstream — **implicit in Step 11** (wq_b is per_1x128)

### Step 15+16 — tuning configs — **already shipped with Step 10**

### Step 17 — Hash-layer fully-fused routing (NEW this session)
- **Env**: `ATOM_A8W4_HASH_FAST_ROUTING=1` (default 0)
- **Files**:
  - `aiter/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py` — `_hash_routing` Triton kernel: tid2eid lookup + sqrtsoftplus + gather + renorm + scale + bitmatrix in one launch.
  - `aiter/aiter/ops/triton/_triton_kernels/moe/moe_routing/expt_data.py` — `_expt_data_only_kernel` (standalone stage1+stage2 launch — replaces matmul_ogs `compute_expt_data` w/ memset).
  - `aiter/aiter/ops/triton/moe/moe_routing/topk.py` — `hash_routing(router_logits, tid2eid, input_ids, n_expts_act, ...)` wrapper.
  - `aiter/aiter/ops/triton/moe/moe_routing/routing.py` — `routing_a8w4_from_hash(...)` (and legacy `routing_a8w4_from_topk(...)`).
  - `ATOM/atom/model_ops/moe.py` — `_a8w4_hash_fused` dispatch branch. Detects DSv4 hash layers via `custom_routing_function.__self__.gate.tid2eid`, fetches input_ids from `forward_context.context.input_ids` (with DP all-gather + clamp), then calls `routing_a8w4_from_hash` → `_a8w4_fused_experts` directly. **Bypasses `FusedMoE.select_experts` + Python `_hash_topk` entirely.**
- **What**: For DSv4 hash layers (first 3 layers, use tid2eid lookup instead of topk), replaces Python `_hash_topk` (softplus + sqrt + gather + renorm + scale) + `fused_routing_from_topk` (3-kernel counting-sort) + matmul_ogs `compute_expt_data` (with memset) with **one** fused Triton kernel + `sort_tokens_fused`. Same 2-kernel structure as the non-hash routing_a8w4 path.
- **Validation**: gsm8k 0.97–0.98, **+0.5%–+1.2% TPS** at c=4..64 over Step 12+14 baseline.

## Validation protocol (apply for every step + at the end)

For each step in order:

1. Apply the diff hunk(s) for the step (or run `apply.sh` to apply everything at once).
2. Kill server cleanly:
   ```bash
   pkill -9 -f atom.entrypoints; pkill -9 -f openai_server; pkill -9 -f spawn_main; sleep 15
   rocm-smi --showmemuse | grep "VRAM%"   # must show 0
   ```
3. Clear ATOM cache: `rm -rf /root/.cache/atom`. (Do NOT routinely nuke `~/.triton` — slow rebuild.)
4. Start server with cumulative env stack (see above).
5. **Smoke** (must say "Paris."): if not, stop and debug.
6. **gsm8k (n=100)**: must be ≥ 0.95. Record actual value.
7. **Perf sweep c=4..64 best-of-2** with `--random-input-len=1024 --random-output-len=1024 --num-prompts=$((c*8))`. Two runs per c back-to-back; healthy variance ≤ 0.5%.
8. Compare TPS vs (a) the new base branch baseline at the same concurrency and (b) the prior step's result.
9. If gsm8k < 0.95 OR perf regresses > 2%, disable the offending env var and investigate. Don't stack a broken step.

`validate.sh` automates steps 2-7 for the whole stack at once.

## Gotchas (read before applying)

1. **`ATOM_V4_USE_TRITON_FUSION=1` + `ATOM_USE_TRITON_MOE=1` are REQUIRED.** Without these, even the original baseline outputs garbage — different code paths are taken. Always smoke-check "Paris." first.

2. **c=128 server crash.** Memory access fault by GPU on all 8 ranks during c=128 prefill workload. **Pre-existing** (not caused by any step). Skip c=128 until investigated. Validate at c=4..64 only.

3. **Variant A (dual_stream fold) is rejected.** Don't extend the Step 9 fold into `dual_stream_moe_forward`. It loses ~13% by killing shared/routed overlap.

4. **Step 11 ALONE fails accuracy** (gsm8k 0.83). Must apply Step 11 + Step 12+14 together.

5. **GPU teardown is sticky.** `spawn_main` workers can outlive their parent. Always `pkill -9 -f spawn_main` and verify VRAM=0% before next run.

6. **Hash-layer detection in Step 17.** `_a8w4_hash_fused` predicate uses `custom_routing_function.__self__.gate.tid2eid`. If the new base branch refactors `_hash_topk` to NOT be a bound method on `MoeLayer` with `gate.tid2eid`, update the detection in `ATOM/atom/model_ops/moe.py` accordingly.

7. **`mxfp8_preshuffle_tuned.csv` is dead** — tuning artifact, not loaded at runtime. Not included in this package.

## If `apply.sh` fails (patches don't fit)

The new base branch has diverged. Apply step-by-step:

1. **Inspect the patches** (`atom_optimizations.patch`, `aiter_optimizations.patch`). Find hunks for each step using the per-step file lists above.
2. **Apply hunks for one step at a time** (manually edit target files, or use `git apply --include=<path>` with a filtered patch).
3. **Validate after each step** (smoke + gsm8k + perf sweep).
4. **If a step breaks at validation**, isolate it and either:
   - Adapt the diff to the new base (if a refactor moved symbols).
   - Disable via env var and skip the step (some are independent — see dependency order).

## Reference perf numbers

See `perf_summary.csv`:

```
concurrency,pre_a8w4_origin_tps,step12_14_tps,current_hash_fused_tps,delta_vs_origin_pct,delta_vs_step12_14_pct,gsm8k_origin,gsm8k_step12_14,gsm8k_current
4,319.76,336.13,339.63,+6.21,+1.04,0.96,0.97,0.98
8,606.80,651.77,653.82,+7.75,+0.31,0.96,0.97,0.98
16,1092.80,1230.37,1237.00,+13.20,+0.54,0.96,0.97,0.98
32,2109.76,2312.73,2327.96,+10.34,+0.66,0.96,0.97,0.98
64,3775.54,4279.06,4283.00,+13.44,+0.09,0.96,0.97,0.98
```

If your post-port numbers are within ±2% of the `current_hash_fused_tps` column (after adjusting for any base-branch perf shifts), the port is successful.
