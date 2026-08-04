# SILOTIGER-667 — Implementation Plan (Living Document)

**Ticket:** [SILOTIGER-667] MoE decode warp-decode kernels (small-M): FP8 + MXFP4 gate_up/down
**Goal of this doc:** Track the FlyDSL reimplementation of the CK-Tile warp-decode MoE
kernels. This is a *living document* — update the status boxes and notes as work
progresses. Ticket description is in `SILOTIGER-667.md`.

---

## 1. Interpretation & scope (agreed)

- The CK branch is the **reference**; the deliverable is a **FlyDSL reimplementation**
  in `aiter`. Supported by: ticket Components = `FlyDSL, Kernels`; FlyDSL Sprints 2/3/4;
  the reference design doc explicitly written "to reimplement the kernels from scratch
  in another framework (e.g. FlyDSL, ...)"; and the existing FlyDSL MoE ecosystem under
  `aiter/ops/flydsl/`.
- **First target (this plan):** *both* kernels (`gate_up` + `down_reduce`) on the
  **FP8 fast path**, as the correctness + perf baseline.
- **Target HW:** gfx950 (CDNA4, wave64). Hardware available for run/bench.
- MXFP4 (incl. the H2 2-outputs/wave `down` win) and fast FP4 `gate_up` are **follow-on**
  work, tracked but out of scope for the first baseline.

## 2. Locked decisions

- **Kernel location:** `aiter/ops/flydsl/kernels/warp_decode_moe.py` (+ a Python
  wrapper/entry point in `aiter/ops/flydsl/`), matching the existing MoE FlyDSL layout.
- **`v_dot2_f32_bf16` primitive:** implement as a **local helper inside the kernel
  module** via `llvm.inline_asm` — do **not** add a dependency by editing the installed
  FlyDSL package. (Pattern reference only: `flydsl/expr/rocdl/inline_asm.py`.)
- **`kVector` for the FP8 baseline:** default `kVector=16` (one 128-bit FP8 transaction)
  when `HIDDEN % 1024 == 0` (gate_up) / `INTER % 1024 == 0` (down); fall back to `kVector=8`
  otherwise. Matches the reference "best-known config" (§9.2 of the reference doc).
- **dot2 inner-loop form for the FP8 baseline:** use the **serialized `s_nop 2`** dot2
  (`dot2_bf16_packed_raw`) for correctness-first Phases 2/3. The s_nop-free + `dot2_drain4`
  ILP scheme (multiple independent accumulators) is introduced **only** with the MXFP4 work.

## 3. Feasibility (verified)

The fast path rests on three primitives, all reachable in FlyDSL on gfx950 wave64:

| Primitive | Availability | Plan |
|---|---|---|
| Packed FP8→BF16 convert (`v_cvt_scalef32_pk_bf16_fp8`) | ROCDL op `cvt_scalef32_pk_bf16_fp8(src,scale,lo_hi_sel)` present (2-wide, **exact match** to reference builtin) | Use generated op directly |
| Packed FP4→BF16 convert (`v_cvt_scalef32_pk_bf16_fp4`) | ROCDL op `cvt_scalef32_pk_bf16_fp4(src,scale,sel_index)` present | Use directly (MXFP4 phase) |
| `v_dot2_f32_bf16` (BF16·BF16→FP32 dot) | **Not** a ROCDL op | **Local inline-asm helper** (only primitive needing it) |
| 64-lane butterfly reduce | `shuffle_xor` (shifts 32→1) | Standard wave64 pattern |

## 4. Reference map (source of truth)

Repo: `/workspaces/rocm-libraries/projects/composablekernel`, commit `62e30c9098`.

| File | Contents |
|---|---|
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp` | `WarpDecodeGateUpKernel` (+ LdsX variant) |
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_down_reduce_kernel.hpp` | `WarpDecodeDownReduceKernel` (+ LdsInter variant) |
| `include/ck_tile/ops/warp_decode/kernel/warp_decode_numeric.hpp` | dot / convert / reduce primitives |
| `include/ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp` | problems + scale-layout tags |
| `include/ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp` | tile distributions |
| `include/ck_tile/ops/warp_decode/WARP_DECODE_MOE_KERNELS.md` | full reimplementation guide |
| `test/ck_tile/warp_decode/test_warp_decode.cpp` | correctness tests (CPU reference oracle) |
| `test/ck_tile/warp_decode/bench_warp_decode.cpp` | standalone C++ benchmark + variant typedefs |

Reference shapes: DeepSeek-V3 (H=7168, I=2048, TOPK=8, E=256), MiniMax (3072/1536/8/256),
Qwen3Next TP1/2/4 (2048 / 512·256·128 / 10 / 512).

---

## 5. Phased plan & status

Status legend: [ ] todo · [~] in progress · [x] done

### Phase 0 — Study the reference in full  [x]
- [x] Read `gate_up`, `down_reduce`, `warp_decode_numeric` kernels end-to-end.
- [x] Extract exact lane→data mapping (lane `l` owns `[l*kVector, (l+1)*kVector)`).
- [x] Understand Block2D scale broadcast through LDS.
- [x] Understand `s_nop` drain scheduling / independent-accumulator pattern.
- [x] Read the CPU reference in `test/ck_tile/warp_decode/` → this is the correctness oracle.
- **Output:** concrete lane-mapping design notes recorded in §7 below.

### Phase 1 — Primitives  [ ]
- [ ] Local `v_dot2_f32_bf16` inline-asm helper in the kernel module; unit-test vs torch.
- [ ] Validate `cvt_scalef32_pk8_fp8_bf16` convert (scaled) vs torch.
- [ ] Validate 64-lane butterfly reduce vs torch.

### Phase 2 — `gate_up` FP8  [ ]
- [ ] Grid `B*TOPK*INTER` waves; HIDDEN tiled in `64*kVector`.
- [ ] Per-K-block scale application (`x_scale * w_scale`).
- [ ] `silu(gate·x) * (up·x)`; lane-0 writes BF16 `inter[B,TOPK,INTER]`.
- [ ] Correctness vs torch/CPU reference.
- [ ] Perf pass on gfx950.

### Phase 3 — `down_reduce` FP8  [ ]
- [ ] Grid `B*ceil(HIDDEN/HPerWarp)`; sum over TOPK then INTER.
- [ ] Fold `router_wt * scale` into accumulator; butterfly reduce; lane-0 write `y[B,HIDDEN]`.
- [ ] Start `kHPerWarp=1`; then add H2 (2 outputs/wave) variant.
- [ ] Correctness + perf vs reference.

### Phase 4 — Scale layouts + integration  [ ]
- [ ] Support PerTensor / PerToken / Block2D scale layouts.
- [ ] Python entry point in `aiter/ops/flydsl/`.
- [ ] op_test per `aiter-op-test` skill (candidate loop + torch reference + markdown perf table + `__main__` guard).
- [ ] Benchmark on gfx950 vs reference numbers in the ticket.

### Follow-on (out of first baseline scope)
- [ ] MXFP4 `down` fast path + H2 layout (beats best FP8 `down` at B≥2).
- [ ] FP4 `gate_up` (ticket's #1 remaining item; gate on accuracy).
- [ ] B=1 FP4 `down` software-pipelined prefetch.
- [ ] Cross-block split-K on `down` + zero-init fusion.
- [ ] Re-test XCD swizzle on small-grid Qwen.

---

## 6. Tuning knobs (from ticket, for later sweeps)

`kVector` 8/16/32 · `kHPerWarp` (down) 1/2 (**2 best at B≥2**) · `kUseDot2` vs
`kUsePackedFp32` · `kNPerWarp` (gate_up) 1/2 · `kWarpsPerBlock` (LDS staging) ·
`kLanesPerOutput` (short-INTER subgroup).

## 7. Design notes (from Phase 0 deep read)

Reference files read: `warp_decode_numeric.hpp`, `warp_decode_gate_up_kernel.hpp`,
`warp_decode_down_reduce_kernel.hpp`, `warp_decode_problem.hpp`,
`host/reference/reference_warp_decode.hpp`, `test/ck_tile/warp_decode/test_warp_decode.cpp`.

### 7.1 Reference math (the correctness oracle)

Per token `b`, for each of its `TOPK` experts `k` with `e = router_ids[b,k]`:

```
# gate_up  (FP32 accumulate, BF16 store)
gate_acc = Σ_i  (x[b,i]·xs) · (w_gate[e,j,i]·gs)          # i over HIDDEN
up_acc   = Σ_i  (x[b,i]·xs) · (w_up  [e,j,i]·us)
inter[b,k,j] = silu(gate_acc) · up_acc                    # silu(z)=z/(1+e^-z)

# down_reduce  (FP32 accumulate, BF16 store)
y[b,out_j] = Σ_k router_wt[b,k] · Σ_i inter[b,k,i] · (w_down[e,out_j,i]·ds)
```

Scale lookup (`lookup_scale`): PerTensor→`p[0]`; PerToken→`p[row_idx]`;
Block2D<BN,BK>→`p[(row/BN)*(max_cols/BK) + col/BK]`. Router weights are normalized
to sum 1 per token. FP8 activation needs `xs`; BF16 activation is unscaled (`xs=1`).

### 7.2 Tensors, strides, kargs (row-major)

- `x[B,HIDDEN]` stride `stride_x≥HIDDEN`.
- `w_gate,w_up`: `[E,INTER,HIDDEN]` flat `[E*INTER,HIDDEN]`, row `w_row=e*INTER+neuron_j`, stride `≥HIDDEN`.
- `inter[B,TOPK,INTER]` flat `[B*TOPK,INTER]`, row `b*TOPK+k`, stride `≥INTER`.
- `w_down`: `[E,HIDDEN,INTER]` flat `[E*HIDDEN,INTER]`, row `e*HIDDEN+out_j`, FP8 stride `≥INTER`.
- `y[B,HIDDEN]` stride `≥HIDDEN`.
- gate_up kargs: `p_x,p_x_scale,p_w_gate,p_w_gate_scale,p_w_up,p_w_up_scale,p_router_ids,p_intermediate` + dims `b,hidden,inter,top_k,e` + strides.
- down kargs: `p_intermediate,p_w_down,p_w_down_scale,p_router_ids,p_router_wts,p_y` + dims + strides.

### 7.3 gate_up FP8 fast path — mapping (default: 1 warp/block, kNPerWarp=1, dot2, kVector=16)

```
GridSize = B * TOPK * INTER ;  BlockSize = 64
neuron_j = block_id % INTER ; d = block_id / INTER
expert_k = d % TOPK ; token_b = d / TOPK
e = router_ids[token_b*TOPK + expert_k] ; w_row = e*INTER + neuron_j
kTileN = 64*kVector ; num_iter = HIDDEN / kTileN
lane l owns K-range [l*kVector, (l+1)*kVector) each iteration; k_base = i*kTileN + l*kVector
```

Inner loop per iter (BF16 activation default): x already 2×BF16 per uint32
(`kVector/2` words). FP8 weight word holds 4 FP8; for `ipair in [0,kVector/2)`:
`w_word=ipair/2`, `w_sel=ipair%2`; convert `g_pair=cvt_scalef32_pk_bf16_fp8(w_gate_word, 1.0, w_sel)`
(same for up); `gate_dot = dot2(gate_dot, x_pair, g_pair)`, `up_dot = dot2(...)`.
After the pair loop: `gate_acc += gate_dot*(xs*gs)`, `up_acc += up_dot*(xs*us)`.
(FP8 activation variant: also convert `x_pair` via the same fp8→bf16 op.)
Reduce both accs (64-lane butterfly); lane 0 writes `silu(gate_acc)*up_acc` as BF16
to `inter[(token_b*TOPK+expert_k)*stride + neuron_j]`.

### 7.4 down_reduce FP8 fast path — mapping

**Baseline 1-output/wave** (`kHPerWarp=1`): `GridSize=B*HIDDEN`,
`out_j=block_id%HIDDEN`, `token_b=block_id/HIDDEN`. Loop `k` over TOPK
(`e=router_ids`, `w=router_wts`, `w_row=e*HIDDEN+out_j`); inner loop over
`INTER/kTileN`. Shared BF16 `inter` tile + FP8 weight tile → per-pair
`cvt_scalef32_pk_bf16_fp8` + dot2; `acc += dot*(w*ds)`. Reduce; lane 0 writes `y`.

**H2 = 2 outputs/wave** (`kHPerWarp=2`, the current FP8 best): `GridSize=B*ceil(HIDDEN/2)`,
`out_j0=(block_id%ceil(HIDDEN/2))*2`, `out_j1=out_j0+1`, `token_b=block_id/ceil(HIDDEN/2)`.
Load `inter` **once**, two weight rows (`w_row0,w_row1`), two dot accumulators
`dot0/dot1`, `acc0/acc1 += dot*(w*ds{0,1})`. Reduce both; lane 0 writes `y[out_j0]`,`y[out_j1]`.
The win is MLP (two weight loads in flight + activation reuse). Ship 1-output first, then H2.

### 7.5 Primitives — what to build locally vs. reuse

| primitive | plan |
|---|---|
| `v_dot2_f32_bf16` | **local inline-asm helper** (kernel module). Serialized form: `"v_dot2_f32_bf16 $0,$1,$2,$0\ns_nop 2"`, constraints `"=v,v,v,0"`, no side effects. (s_nop-free + `dot2_drain4` variant deferred to the MXFP4 phase.) |
| fp8→bf16 convert | **ROCDL op** `fx...cvt_scalef32_pk_bf16_fp8(src_i32, scale_f32, lo_hi_sel)` — matches the reference builtin exactly; **no inline asm needed**. |
| fp4→bf16 convert | ROCDL op `cvt_scalef32_pk_bf16_fp4(src, scale, sel_index)` (MXFP4 phase). |
| pack bf16 pair | the FlyDSL tile already exposes each 2×BF16 as one `uint32`; read it directly as the `a`/`b` dot2 operand. |
| 64-lane reduce | `val.shuffle_xor(sh,64)` for `sh in [1,2,4,8,16,32]`, summing. |
| silu | `z * sigmoid(z)`; compute in FP32 on lane 0. |

### 7.6 Correctness harness (mirror the CK test)

Input fills: `x∈[-1,1]` (BF16) or `[-0.5,0.5]` (FP8 act); FP8 weights `∈[-0.25,0.25]`;
`inter∈[-1,1]`; scales `∈[0.5,2]`; router ids uniform `[0,E-1]`, weights normalized to
sum 1. Tolerances (atol) from the CK test: `~0.3` for small block2d shapes; `~5.0` for
full DeepSeek/MiniMax (BF16 accumulation over long K legitimately drifts). Build a torch
reference replicating §7.1 for the op_test.

### 7.7 Divisibility / support constraints

- gate_up: `HIDDEN % (64*kVector) == 0`; dot2 requires **even kVector**; FP8 aligned → `kVector=16` (else 8).
- down full-wave: `INTER % (64*kVector) == 0`; H2 requires `HIDDEN % 2 == 0`, one warp/block.
- Block2D scales: `HIDDEN%BK==0` and `(E*INTER)%BN==0` (gate_up); `INTER%BK==0` and `(E*HIDDEN)%BN==0` (down).

## 8. Open questions / risks

- [resolved] FP8/FP4→BF16 converts: exact 2-wide ROCDL ops exist
  (`cvt_scalef32_pk_bf16_fp8/fp4`), matching the reference builtins — no inline asm.
- [resolved] Only `v_dot2_f32_bf16` needs a local inline-asm helper.
- [open] Exact FlyDSL `llvm.inline_asm` result type + tied-operand form for `v_dot2_f32_bf16`
  (validate the `"=v,v,v,0"` + `s_nop 2` string emits correct ISA; check `FLYDSL_DUMP_IR`).
- [open] How the FlyDSL tile/copy path exposes each lane's `kVector` FP8/BF16 chunk as
  consecutive `uint32` words (reference relies on `get_as<uint32_t>(word)`); decide between
  copy-atom tiles vs. `make_buffer_tensor` raw loads.
- [open] Block2D scale indexing in FlyDSL (start with direct HBM per-K-block read;
  the LDS-broadcast optimization from §5.4 is gate_up-only and can come later).
- [deferred] MXFP4 s_nop-free dot2 + `dot2_drain4` scheduling (MXFP4 phase).

## 9. Changelog

- _init_ — plan created; scope, decisions, feasibility, phases recorded.
- _phase 0_ — deep read of reference kernels + numeric primitives + CPU oracle done;
  §7 design notes filled (math, mappings, primitives, harness, constraints); feasibility
  refined (2-wide converts are ROCDL ops; only dot2 needs a local helper).
- _phase 0 close-out_ — locked two baseline choices in §2: `kVector=16` (→8 fallback)
  and serialized `s_nop 2` dot2 for the FP8 baseline (ILP/drain deferred to MXFP4).
  Paused before Phase 1.
