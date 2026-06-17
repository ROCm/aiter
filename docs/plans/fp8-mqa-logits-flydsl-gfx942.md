# Plan: FlyDSL dense FP8 MQA logits kernel for gfx942

- **Ticket:** SILOTIGER-614 — FP8 MQA logits (DeepSeek lightning indexer), gfx942 vs gfx950
- **Scope (this phase):** Dense `_fp8_mqa_logits` equivalent in FlyDSL for gfx942. Paged variant deferred.
- **Status:** Draft — iterating.

## 1. Background

The DeepSeek sparse-attention lightning indexer needs a cheap score matrix to choose
which KV positions each query attends to. The FP8 MQA logits kernel computes those
scores: per query row, an FP8 Q·K dot over a KV window, dequantized by a per-K scale,
ReLU'd, multiplied by per-head weights, and summed over heads. Output is fp32 logits
consumed by downstream top-k.

### Math (dense)

For query row `m`:

```
logits[m, n] = sum_h ReLU(<Q[m,h,:], K[n,:]> * kv_scale[n]) * weights[m,h]
```

Masked outside the per-row window `[cu_starts[m], cu_ends[m])`.

Shapes:
- `Q`:         `[seq_len, NUM_HEADS, HEAD_SIZE]`, fp8
- `KV`:        `[seq_len_kv, HEAD_SIZE]`, fp8
- `kv_scales`: `[seq_len_kv]`, f32
- `weights`:   `[seq_len, NUM_HEADS]`, f32
- `cu_starts`/`cu_ends`: `[seq_len]`, int32
- `logits`:    `[seq_len, seq_len_kv]`, fp32 (init to -inf for causal masking)

DeepSeek dims: `NUM_HEADS=64`, `HEAD_SIZE (D) ∈ {64, 128}`.

## 2. Current state (reference implementations)

- **Triton dense kernel** (gfx942 primary): `aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py` — `_fp8_mqa_logits_kernel`.
- **Triton launcher** (mirror behavior): `aiter/ops/triton/attention/fp8_mqa_logits.py` — `fp8_mqa_logits`.
- **Gluon dense** (gfx950-only, CDNA4 intrinsics): `aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py`.
- **Torch reference + tests:** `op_tests/triton_tests/attention/test_fp8_mqa_logits.py` — `ref_fp8_mqa_logits`, `calc_diff` (< 1e-3).

There is **no** FlyDSL MQA-logits kernel today. FlyDSL, CK, and standalone HIP have none.

> **Note:** The ticket references cloning from `pa_decode_fp8.py`. That file does **not**
> exist in this repo. The substitute templates are:
> - `aiter/ops/flydsl/kernels/fused_compress_attn.py` (CSA decode: fp8, online reductions, host launcher) — closest analog.
> - `aiter/ops/flydsl/kernels/preshuffle_gemm.py` (fp8 MFMA `rocdl.mfma_f32_16x16x32_fp8_fp8`).
> - `aiter/ops/flydsl/kernels/reduce.py` (wave64 shuffle + LDS cross-wave reduction).

## 3. Deliverables / file changes

1. **New kernel module** — `aiter/ops/flydsl/kernels/fp8_mqa_logits.py`
   - `compile_fp8_mqa_logits(*, num_heads, head_size, block_kv=128)` — `@functools.lru_cache`'d
     on the constexpr config, returns a `@flyc.jit` launcher. Keep a `paged=False` arg in the
     signature as the Phase-2 hook, but only implement the dense path.
   - `flydsl_fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits=True)` —
     public host wrapper. Mirrors the Triton `fp8_mqa_logits`: same args, allocates the
     `-inf`-filled (or `empty`) `logits[seq_len, seq_len_kv]` using the 256-aligned padding
     trick, computes strides, dispatches the compiled launcher, returns `logits`.
2. **Registration** — `aiter/ops/flydsl/__init__.py`: export `flydsl_fp8_mqa_logits` under the
   `is_flydsl_available()` guard, so FlyDSL runs alongside Triton on gfx942 for A/B comparison.
3. **Test** — `aiter/ops/flydsl/test_flydsl_fp8_mqa_logits.py`: reuse `ref_fp8_mqa_logits` +
   `calc_diff` from the Triton test; skip cleanly if flydsl/GPU absent; assert `diff < 1e-3`
   across DeepSeek dims (H=64, D ∈ {64,128}) and the existing `(s_q, s_k)` shape matrix,
   including empty-window, tail, and both `clean_logits` modes.

## 4. Kernel structure

`@flyc.kernel(known_block_size=[256, 1, 1])`, grid `(seq_len,)`, one CTA per query row, 4 waves of 64.

```
row = num_programs.x - block_idx.x - 1          # reverse for tail-effect balancing
start = max(cu_starts[row], 0); end = min(cu_ends[row], seq_len_kv)
load Q[H, D] (fp8) and weights[H] (f32) once
unmasked_end = (end - start) // BLOCK_KV * BLOCK_KV
for n0 in scf.for(start, start + unmasked_end, BLOCK_KV):     # full tiles, fast path
    s[H, BLOCK_KV] = mfma_fp8(Q, K[n0:])                      # 4x K32 steps over D=128
    s *= kv_scale[n0:] (broadcast over H)
    s = max(s, 0)
    s *= w[H] (broadcast over N)
    logit[BLOCK_KV] = head_reduce_sum(s)                      # reduce over H
    store logits[row, n0 : n0 + BLOCK_KV]
# tail tile: masked load (other=0) + masked store honoring [start, end) and clean_logits
```

## 5. gfx942 mapping decisions

- **MMA:** `rocdl.mfma_f32_16x16x32_fp8_fp8`, accumulator `vec<4 x f32>`. Tile `M=H` (H/16
  row-tiles), `N=BLOCK_KV=128` (8 col-tiles of 16), `K=D` (D/32 steps -> 4 for D=128).
  Distribute the M/N tile grid across the 4 waves.
- **fp8 dtype:** `fx.Float8E4M3FNUZ` on gfx942, selected via `get_hip_arch()`.
- **Head reduction (key risk):** `s` rows (H) are laid across the MFMA output lane/wave
  layout, so summing over H needs a two-level reduce: in-wave butterfly
  `gpu.ShuffleOp(mode="xor")`, then a single LDS cross-wave pass + `gpu.barrier()`.
  Follows the reduction patterns in `fused_compress_attn.py` / `reduce.py`. The exact
  fragment->lane mapping will be pinned early to minimize cross-wave traffic.
- **LDS:** only cross-wave reduce scratch (+ optional Q/K staging) — well under 64 KB.
- **Masking:** interior tiles unmasked; tail tile uses masked `buffer_load(other=0)` and a
  predicated store; replicate `clean_logits` (host `-inf` prefill) semantics exactly.
- **Launcher hints:** `waves_per_eu=2`, fast math; `BLOCK_KV=128` to match gfx942 Triton.

## 6. Build-up / verification order (incremental, on gfx942)

1. Scaffolding + launcher + `-inf` host buffer; kernel writes a known constant -> validate
   plumbing / strides / masking shape.
2. Add fp8 MFMA dot only (no epilogue) -> compare raw `Q·K` against torch.
3. Add scale · ReLU · weights, no reduction -> spot-check a single head.
4. Add head-reduce -> full `calc_diff < 1e-3` against `ref_fp8_mqa_logits`.
5. Sweep the full shape matrix (empty-window, tail, both `clean_logits`, D ∈ {64,128}).
6. Quick A/B perf vs Triton on DeepSeek dims.

## 7. Risks / open items

- **Head-reduction layout** is the main complexity; lock the MFMA fragment->lane mapping
  early (steps 2 & 4).
- Substituting `fused_compress_attn.py` + `preshuffle_gemm.py` for the nonexistent
  `pa_decode_fp8.py`.
- Paged variant (`_deepgemm_fp8_paged_mqa_logits`, SplitKV, KV-gather) is out of scope here.

## 8. Expected performance

gfx942 has native fp8 MFMA at the rate this workload needs; the kernel is likely
bandwidth/issue-bound on the fp8 KV stream plus head reduction, not MMA-bound. A
well-scheduled FlyDSL version should match or modestly beat gfx942 Triton by controlling
the head-reduce and KV-load width. Levers: `BLOCK_KV=128`, `waves_per_eu=2`, widest feasible
fp8 buffer load, register + single-LDS-pass head reduce.

## 9. Open questions (to resolve while iterating)

- [ ] Confirm 256-thread / 4-wave block vs alternative (e.g. 64-thread single-wave to avoid
      cross-wave reduce entirely at small H).
- [ ] Decide whether Q/K stage through LDS or stay in registers.
- [ ] Confirm whether `flydsl_fp8_mqa_logits` should also be wired into the top-level
      `fp8_mqa_logits` dispatcher for automatic gfx942 selection, or kept as an explicit
      opt-in for A/B only.
