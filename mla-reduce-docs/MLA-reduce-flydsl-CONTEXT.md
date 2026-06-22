# Background & Context — FlyDSL MLA decode reduce/combine for gfx942 (GLM-5.1)

This is the orientation doc for executing the plan in
`AITER-Implement-FlyDSL-sparse-MLA-decode-reduce-for-gfx942-targetting-glm5.1-model.md`.
It captures *what the kernel does*, *what already exists in this repo*, *the exact
files you'll touch*, and *the traps* — so you can start coding without re-deriving.

Repo root: `/home/anguyenh/aiter`. Target HW: MI300X / **gfx942 (CDNA3)**.

---

## 1. What this kernel is (the algorithm)

Flash-decoding / split-KV MLA cuts the KV sequence into many tiles ("splits") so a
short decode batch can fill all CUs. **Stage-1** produces, per split, a partial
attention output `O_i` (fp32) and a log-sum-exp `LSE_i`. **Stage-2** — *this kernel* —
merges the splits with an LSE-weighted online-softmax combine:

```
final_O   = Σ_i exp(LSE_i − LSE_max) · O_i  /  Σ_i exp(LSE_i − LSE_max)
final_LSE = LSE_max + log( Σ_i exp(LSE_i − LSE_max) )
```

It is **pure reduction**: no MFMA, no matrix core, memory-bandwidth bound. That single
fact drives everything — see §6.

## 2. Why this task exists

- MLA decode stage-1 has FlyDSL producers (ASM/HipKittens/Opus/FlyDSL/Triton/Gluon all
  feed the same reduce). But **FlyDSL has no native reduce kernel** — its MLA decode
  tests currently call the HIP `aiter.mla_reduce_v1`.
- Goal: a **native FlyDSL MLA reduce** so the MLA stack is fully FlyDSL-native on gfx942,
  no separate HIP dependency. Targeting the **GLM-5.1** model's decode path.
- Risk is **low**, effort **~1–2 days**: the reference HIP kernel is already arch-neutral
  and runs on gfx942 today; this is a port, not a new design.

## 3. The reference HIP kernel (study this first)

`csrc/kernels/mla/reduce.cu` — `kn_mla_reduce_v1` (grid-launch) and
`kn_mla_reduce_v1_ps` (persistent). Read it as the spec. Key structure:

- **Traits** (`MlaReduceKernelV1Traits`, line 13): `kNumWarps=2`, `kNumThreads=128`,
  `kOccupancy=8`, `kVecWidth = kSizeDV / kNumThreads`. For `Dv=512`, `kVecWidth=4` floats
  per thread (the plan's `VEC = Dv//128`). `kMassiveThreshold=4`.
- **Two code paths**, branched on `num_splits = reduce_indptr[tile+1] − reduce_indptr[tile]`:
  - **Simple** (`mla_reduce_v1_impl_simple`, line 538), used when `2 ≤ num_splits < 4`:
    sequential online-softmax merge directly in registers (`reg_out`, running
    `max_lse`/`sum_e_lse`, `old_scale`/`new_scale`), final divide by `sum_e_lse`. This is
    the plan's "simple path".
  - **Massive** (`mla_reduce_v1_impl_massive`, line 416), used when `num_splits ≥ 4`:
    warp 0 computes per-split LSE → global max → global LSE → writes a normalized
    `lse_scale[s]` to LDS (`reduce_lse_massive`), then `reduce_output_massive` does a
    weighted accumulate `reg_out += lse_scale[s]·O_s` with a 2-wide software-pipelined
    loop (prefetch next tile's partial-map + data while computing current). This is the
    plan's "massive path".
  - Sub-tiers inside massive by split count: `kUpTo64Splits` / `kUpTo256Splits` /
    `kUpToLdsLimit` (`MlaReduceProblemSize`). The first 256 LSEs live in VGPR, overflow
    spills to LDS (`LocalLse`, line 128). For a first FlyDSL port you can likely target
    just the ≤256 (or even ≤64) tiers and fall back to HIP otherwise.
- **LDS layout** (line 425): `reduce_partial_map[max_splits]` (int32) +
  `lse_scale[max_splits]` (float) + overflow `local_lse[max(0,max_splits−256)]`. Tiny —
  a few KB. `max_splits` is set to `multiProcessorCount` (num CUs) at host (line 1070).
- **Addressing assumptions** (comments at lines 460–462, 579–581):
  - partial_output is `[bs, h, d]`, contiguous → stride d=1, h=`kSizeDV`, b/s=`kSizeDV·H`.
  - partial_lse `[bs, h]` → stride h=1, b/s=`H`.
  - final_output uses runtime `params.stride_s_o`, `stride_h_o` (NOT assumed contiguous).
  - final_lse `[bs, h]` → stride h=1, b/s=`H`.
- **gmem descriptors** built from kernel-arg pointers via `opus::make_gmem<T>` →
  buffer_load/store with SGPR base (no waterfall). FlyDSL equivalent is the bounded
  buffer-descriptor idiom (see §5 / KB).
- **NaN handling**: `sum==0 || sum!=sum → LSE=INF`; partial NaN outputs zeroed
  (the torch ref does this explicitly, lines 694–710).
- **Host dispatch** (`mla_reduce_v1`, line 1024 → `dispatch_mla_reduce_v1`, line 919):
  computes `num_work_group_per_bh`, chooses grid-launch vs persistent based on whether
  total work fits `num_cu·kOccupancy·2`, validates LDS ≤ shared-mem-per-CU. Router
  (`MLA_REDUCE_ROUTER`, line 870) enumerates supported `(num_heads, head_dim)` combos —
  note **(128,512), (16,512), (128,128)** the plan verifies are all present.

## 4. The reduce I/O contract (match this exactly)

From `mla_reduce_v1` signature (line 1024) and `torch_mla_reduce_v1` (test, line 620):

| arg | shape / dtype | notes |
|---|---|---|
| `partial_output` | fp32 `[max_partial_row, H, Dv]` | contiguous |
| `partial_lse` | fp32 `[max_partial_row, H]` | contiguous |
| `reduce_indptr` | int32 `[#work+1]` | CSR-style per-tile split ranges |
| `reduce_partial_map` | int32 `[reduce_indptr[-1]]` | split → partial-row index (gather) |
| `reduce_final_map` | int32 `[#work,2]` (optional) | `[q_start,q_end]`; if absent, derive from `reduce_partial_map[1]−[0]` |
| `max_seqlen_q` | int | fallback for q-range when no final_map |
| `final_output` | bf16/fp16 `[bs,H,Dv]` | runtime strides |
| `final_lse` | fp32 `[bs,H]` (optional) | only if `output_lse` |

`softmax_scale` in the plan's pseudocode is **not** in the HIP signature — the
combine itself needs no scale (scale was applied in stage-1). Treat it as optional/unused
unless a producer requires it; confirm against the FlyDSL decode caller before adding it.

## 5. What FlyDSL scaffolding to reuse (don't reinvent)

- **Reduction skeleton**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py :: compile_moe_reduction`
  (line 3590). It already does `@flyc.kernel` arg marshalling, a buffer-resource builder
  (`_ptr_buffer_resource`, line 3658), vectorized gather/store with `VEC_WIDTH`/`copy_vec_width`,
  and an optional `valid_mask` gather. This is the closest structural twin — copy its shape.
- **Wave/block reduce helpers**: `aiter/ops/flydsl/kernels/reduce.py` —
  `make_block_reduce` (max, line 38), `make_block_reduce_add` (line 170),
  `make_block_reduce_add2` (line 310). These give you xor-shuffle warp reduce + LDS
  cross-wave combine for the LSE max/sum. (Plan also cites `topk_gating_softmax_kernel.py`;
  the helpers in `reduce.py` are the de-duplicated versions — prefer these.)
- **Math intrinsics**: `rocdl.exp2(T.f32, x)` and `rocdl.rcp(T.f32, x)` — already used in
  `compile_moe_reduction`/softmax/rmsnorm. Use `exp2` for the LSE weights (`use_exp2`
  knob); the online merge uses base-2 (`exp2(lse − max)`), matching the simple path's
  `expf` semantics after a `log2e` fold, or keep `expf` for exact parity — verify
  numerics against torch ref.
- **Bounded buffer descriptors & varlen idioms**: KB topic `flydsl-jdbba/00-layout-api-idioms`
  documents verified `flydsl.expr` idioms — device scalar read (for `reduce_indptr`
  values), runtime base-offset views (for the per-split gather via `reduce_partial_map`),
  bounded buffer descriptors (HW OOB-drop for partial tiles), and the bf16 epilogue cast.
  **Read this before writing the gather/store** — it is the same memory-bound, varlen,
  gather-heavy regime.

## 6. Performance expectation & the one thing to check first

This is a **memory-bound elementwise reduction**. Expectation: **match HIP `mla_reduce_v1`
within noise** on gfx942. The FlyDSL port's primary value is maintenance/toolchain
(FlyDSL-native MLA stack), **not** speed.

**Before optimizing anything**: benchmark the current HIP `mla_reduce_v1` on gfx942 with
`rocprofv3` and check if it's already HBM-bandwidth bound. If it saturates HBM, there is
no perf to win — the port is a toolchain win, full stop. (This mirrors the jdbba KB
lesson: at the byte floor, compute-scheduling levers don't move the needle; only
HBM-traffic levers do, and a pure reduce already reads each partial exactly once.)

Optimization knobs the plan exposes (sweep only if *not* HBM-bound):
`prefetch_depth ∈ {2,4,6,8}`, `waves_per_eu ∈ {4,6,8}` (jointly), `use_exp2`,
`use_packed_cvt` (packed fp32→bf16/fp16 store), `use_packed_f32_fma` (V_PK_FMA_F32 on CDNA3).

## 7. Concrete change list (from the plan)

1. **New file** `aiter/ops/flydsl/kernels/mla_reduce.py` with
   `compile_mla_reduce(*, H, Dv, out_dtype, persistent, prefetch_depth=2, waves_per_eu=8,
   use_exp2=True, use_packed_cvt=False, use_packed_f32_fma=False)`.
   - Constexpr `VEC = Dv // 128`, `known_block_size=[128,1,1]`.
   - Implement simple path (`num_splits < 4`) and massive path (`≥4`) per §3.
2. **Reuse** `compile_moe_reduction` scaffolding (arg marshalling, buffer atoms, masked
   gather) + `reduce.py` wave-reduce helpers + `rocdl.exp2`/`rcp`.
3. **Dispatcher wiring** for `flydsl_mla_reduce`, **falling back to HIP `mla_reduce_v1`**
   (`aiter/ops/attention.py:1127`) when FlyDSL is unavailable or shape unsupported.
4. **Verify** against `torch_mla_reduce_v1` (test_mla_persistent.py:620) for
   `(H,Dv) ∈ {(128,512),(16,512),(128,128)}`, `num_kv_splits ∈ 1..16`, and boundary
   `n_splits ∈ {1,2,3,4,8,16}` (covers both path branches and the 4-split threshold).

## 8. Traps & gotchas (FlyDSL-specific)

- **`range_constexpr` vs runtime `range`**: `num_splits` is *runtime* (from `reduce_indptr`),
  so the merge loop must be a runtime loop with loop-carried state (running max/sum/accum).
  The plan's pseudocode uses `range_constexpr` — that only works for a compile-time-fixed
  split count; for the general kernel use a `scf.for` with carried `(o_frag, max_lse,
  sum_e)`. Loop-carried state packing is a documented FlyDSL pitfall — see the
  `debug-flydsl-kernel` skill.
- **Online-softmax loop-carried tuple**: simple path carries `(reg_out_vec, max_lse,
  sum_e_lse)`; ensure all three thread through the range loop's `init=`/yield correctly.
- **Numerics / NaN**: replicate `sum==0||NaN → LSE=INF` and zero-out NaN partials, else
  parity vs torch ref fails on edge tiles.
- **Buffer OOB on the gather**: `reduce_partial_map` indexes into `partial_output` rows;
  use bounded buffer descriptors so partial/last tiles HW-drop instead of reading garbage.
- **final_output strides are runtime** (`stride_s_o`/`stride_h_o`) — don't assume
  contiguous on the store side; partial_* ARE contiguous on the load side.
- **Cache invalidation when iterating**: FlyDSL compile cache can serve stale kernels —
  the `debug-flydsl-kernel` skill covers clearing it. Use `/capture-kernel-trace` +
  `/debug-flydsl-kernel` when output is wrong.
- **Two launch modes**: HIP has grid-launch and persistent (`_ps`). The plan's
  `persistent` flag mirrors this. Start with grid-launch (simpler), add persistent only
  if needed for the small-batch occupancy case.

## 9. Suggested execution order

1. Read `reduce.cu` simple+massive paths and `compile_moe_reduction` side by side.
2. Skim KB `flydsl-jdbba/00-layout-api-idioms` for the gather/store/buffer-desc idioms.
3. Benchmark HIP `mla_reduce_v1` on gfx942 (rocprofv3) → settle whether perf is even on
   the table.
4. Write `compile_mla_reduce` simple path first; verify vs `torch_mla_reduce_v1` at
   `n_splits ∈ {2,3}`, `(H,Dv)=(128,512)`.
5. Add massive path (≤64 then ≤256 tier); verify `n_splits ∈ {4,8,16}` and other shapes.
6. Wire dispatcher + HIP fallback; run the full verify matrix.
7. Only if not HBM-bound: sweep the knobs in §6.

---

### Key file map
- Reference HIP kernel: `csrc/kernels/mla/reduce.cu`
- HIP host entry / fallback target: `aiter/ops/attention.py:1127` (`mla_reduce_v1`)
- Python MLA callers: `aiter/mla.py` (decode `:492`, prefill `:647`)
- Torch reference for verify: `op_tests/test_mla_persistent.py:620` (`torch_mla_reduce_v1`)
- FlyDSL reduction twin: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:3590`
- FlyDSL reduce helpers: `aiter/ops/flydsl/kernels/reduce.py`
- New file to create: `aiter/ops/flydsl/kernels/mla_reduce.py`
- KB: `~/claude-knowledge-base/flydsl-jdbba/` (varlen idioms + memory-bound methodology)
</content>
</invoke>
