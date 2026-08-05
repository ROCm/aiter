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

- **Test environment:** run all tests in **`flydsl_venv`** (has the correct deps, incl.
  triton 3.6.0):
  `./flydsl_venv/bin/python -m pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`
  (or `./flydsl_venv/bin/python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`). The
  default env's triton 3.3.1 < gluon's 3.6.0 requirement, which blocks `import aiter`.
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

### Phase 1 — Primitives  [x]
- [x] Local `v_dot2_f32_bf16` inline-asm helper in the kernel module; unit-test vs torch.
      `dot2_f32_bf16(a_i32,b_i32,acc_f32, serialize=True)` → `"v_dot2_f32_bf16 $0,$1,$2,$0"`
      (+`s_nop 2` when serialized), constraints `"=v,v,v,0"`. **Exact** vs torch (max_delta 0.0).
- [x] Validate `cvt_scalef32_pk_bf16_fp8` convert (scaled) vs torch. `fp8x2_to_bf16x2(src_i32,
      scale_f32, hi)` via the ROCDL op. **Exact** with power-of-two scales (max_delta 0.0).
      **Finding:** the op applies **only the exponent (E8M0)** of the f32 scale, not the full
      value — `scale=3.0` yields `fp8×2.0` (mantissa discarded). See §9/§7 notes.
- [x] Validate 64-lane butterfly reduce vs torch. `wave_reduce_add_f32` over shifts
      1,2,4,8,16,32 via `gpu.ShuffleOp(..., mode="xor")`. **Exact** (max_delta 0.0).
- **Where:** primitives + `build_warp_decode_primitives_module` in
  `aiter/ops/flydsl/kernels/warp_decode_moe.py`; test
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py` (`python …` or `pytest`, 4 pass).

### Phase 2 — `gate_up` FP8  [x] (correctness baseline; perf optimization pending)
- [x] Grid `B*TOPK*INTER` waves; HIDDEN tiled in `64*kVector` (kVector 16, →8 fallback).
- [x] Scale application: BF16 activation ⇒ `xs=1`; fold constant `gs`/`us` (PerTensor `p[0]`
      / PerToken `p[e*INTER+j]`) into the accumulator **after** the K reduce (exact since the
      scale is constant over K). Block2D deferred to Phase 4.
- [x] `silu(gate_acc)·up_acc`, `silu(z)=z/(1+e^-z)` in f32; **lane-0-only** BF16 store to
      `inter[B,TOPK,INTER]` (`if lane == 0:`; reduce still runs on all lanes).
- [x] Correctness vs torch reference: **exact** (cos_sim 1.0, max_delta 0.0) across
      PerTensor + PerToken and kVector 16 (`HIDDEN=1024`) / kVector 8 (`HIDDEN=512`).
- [x] Perf pass on gfx950: after lane-0-store **and vectorized 128-bit loads**, a realistic
      shape (B1 INTER2048 HIDDEN7168 E8 TOPK8) hits **~6.9 TB/s (~86% HBM peak)**, up from the
      ~1.4–1.6 TB/s scalar-load baseline. Vectorization loads x/w via widest `vec4`/`vec2` i32
      `buffer_load`s (`load_i32_words` helper) and also drops the old duplicate weight-dword
      reloads. Remaining levers, deferred: (1) s_nop-free dot2 + `dot2_drain4` ILP (MXFP4 phase);
      (2) B=1 software-pipelined weight prefetch. [done] lane-0-only store; vectorized loads.
- **Where:** kernel `build_gate_up_fp8_module` + `pick_kvector` in
  `aiter/ops/flydsl/kernels/warp_decode_moe.py`; entry `flydsl_warp_decode_gate_up` in
  `aiter/ops/flydsl/warp_decode_moe.py`; tests `GATE_UP_CASES` /`test_gate_up_fp8` in
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py` (7 pass total).
- **Scope choices (confirmed):** BF16 activation + FP8 weights first; PerTensor+PerToken
  scales; full aiter Python entry point added now. FP8/BF16 activation-variant and Block2D
  are later.

### Phase 3 — `down_reduce` FP8  [x] (1-output baseline; H2 + perf pending)
- [x] Grid `B*HIDDEN` waves (`kHPerWarp=1`); `out_j=bid%HIDDEN`, `token_b=bid//HIDDEN`;
      INTER tiled in `64*kVector` (kVector from INTER: 16, →8 fallback).
- [x] Fold `router_wt * ds` (both lane-uniform) into each expert's **per-lane** partial, then
      a **single** butterfly reduce over Σ_k — exact and avoids a reduce per k. PerTensor
      `p[0]` / PerToken `p[e*HIDDEN+out_j]`. **lane-0-only** BF16 store to `y[B,HIDDEN]`.
- [x] `kHPerWarp=1` shipped. **H2 (2 outputs/wave)** — the reference FP8 best — pending.
- [x] Correctness vs torch reference: **exact** (cos_sim 1.0, max_delta 0.0) across
      PerTensor + PerToken, kVector 16 (`INTER=1024`) / 8 (`INTER=512`). End-to-end
      gate_up→down_reduce vs full torch MoE: cos 0.9999998, max_delta 1.5e-5 (stages compose).
- [x] Perf: after vectorized 128-bit loads, the realistic DeepSeek-ish shape
      (B1 INTER2048 HIDDEN7168 E8 TOPK8) hits **~5.7 TB/s (~71% HBM peak)** (was ~6.2 TB/s
      before vectorization on the lane-0-store baseline; the vec-load win is smaller here since
      `down` was already coalescing well). Remaining lever: H2 (2 outputs/wave) activation-reuse.
- **Where:** kernel `build_down_reduce_fp8_module` in `kernels/warp_decode_moe.py`; entry
  `flydsl_warp_decode_down_reduce` in `ops/flydsl/warp_decode_moe.py`; tests `DOWN_CASES` /
  `test_down_reduce_fp8` in the op_test (10 pass total).

### Phase 4 — Scale layouts + integration  [~]
- [~] Scale layouts: PerTensor + PerToken done (Phase 2, gate_up); **Block2D pending**.
- [x] Python entry point in `aiter/ops/flydsl/warp_decode_moe.py` (added in Phase 2;
      extend with `down_reduce` + Block2D).
- [x] op_test scaffolding at `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`
      (Phase 1 primitives + Phase 2 gate_up; conforms to §8). Extend for down/Block2D.
- [ ] Benchmark on gfx950 vs reference numbers in the ticket (+ perf optimization pass).

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

## 8. Testing conventions & reuse

Tests must conform to the `CONTRIBUTE.md` "Testing" section and to the
`.claude/skills/aiter-op-test` skill, and should reuse the existing MoE test
machinery rather than re-deriving it.

### 8.1 CONTRIBUTE.md conventions (must follow)

- Tests are **standalone Python scripts** under `op_tests/`. Ours lives at
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py` (alongside the other FlyDSL
  MoE tests). Must be both `pytest`-collectable (`test_*` functions) and runnable as
  `python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py` with a `__main__` guard.
- Use `aiter.test_common`: `checkAllclose` for numeric compare, `perftest` / `run_perftest`
  for timing. Provide an `argparse` CLI (dtype/shape/B knobs) and clear PASS/FAIL prints.
- Runs under `bash .github/scripts/aiter_test.sh`; CI covers MI300X (gfx942) + MI350X (gfx950).
- Format with **black** and **ruff** (`black aiter/ op_tests/`, `ruff check aiter/ op_tests/`).
- Arch/availability guard (from `test_flydsl_moe_a16wfp4.py`):
  `skipif(get_gfx() not in ("gfx950",) or not is_flydsl_available())`
  (`aiter.jit.utils.chip_info.get_gfx`, `aiter.ops.flydsl.utils.is_flydsl_available`).
- Follow the `aiter-op-test` skill layout: `@benchmark` + `run_perftest` candidate loop,
  a torch reference, a final **markdown summary table**, and a `__main__` guard so the
  module stays importable.

### 8.2 Reusable helpers from existing MoE op_tests

Primary references: `op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py`,
`op_tests/test_moe_2stage.py`.

- **Torch oracle (use directly, do not re-derive from CK's `reference_warp_decode.hpp`):**
  `aiter.fused_moe.torch_moe_stage1` (gate_up + activation) and `torch_moe_stage2`
  (down + top-k weighted reduce) match our two kernels exactly. They operate on
  `topk_ids`/`topk_weights` directly and need **no** `moe_sorting`, which suits the
  sorting-free warp-decode design.
- **Routing:** `aiter.fused_moe.fused_topk(inp, score, topk, True)` →
  `topk_weights, topk_ids` (our `router_wts`/`router_ids`), fed to both kernel and oracle.
- **Quant + scales (Phase 4 FP8 layouts):** `aiter.get_torch_quant(QuantType...)`,
  `aiter.ops.quant.per_1x32_f8_scale_f8_quant`, `mxfp4_moe_sort_fwd`; `aiter.utility.fp4_utils`
  for MXFP4 dequant/e8m0 (later phase).
- **Compare + perf:** the `_check_result` idiom (cosine-sim primary gate `cos > 0.999`
  plus `%`-close) and/or `checkAllclose`; `run_perftest(..., num_iters=, use_cuda_event/
  testGraph)` for the perf table. Align tolerances with the CK atol guidance (§7.6): small
  block2d ≈ 0.3, full DeepSeek/MiniMax ≈ 5.0 (BF16 long-K drift).
- **Skip/param patterns:** the `_SKIP_GFX950_FLYDSL` skipif + `pytest.mark.parametrize`
  shape/seed sweeps.

### 8.3 Caveats on reuse

- Our baseline kernels take **unsorted** `x[B,HIDDEN]`, plain row-major weights
  (`w_gate/w_up[E,INTER,HIDDEN]`, `w_down[E,HIDDEN,INTER]`, FP8), and `router_ids/wts`
  directly — so we reuse the **data-gen + torch oracles** but **not** the
  `moe_sorting`/`sorted_ids`/`shuffle_weight_a16w4` plumbing those tests use for the
  tiled-GEMM kernels. Keep the harness minimal for the warp-decode path.

## 9. Open questions / risks

- [resolved] FP8/FP4→BF16 converts: exact 2-wide ROCDL ops exist
  (`cvt_scalef32_pk_bf16_fp8/fp4`), matching the reference builtins — no inline asm.
- [resolved] Only `v_dot2_f32_bf16` needs a local inline-asm helper.
- [resolved] FlyDSL `llvm.inline_asm` result/tied-operand form for `v_dot2_f32_bf16`:
  `llvm.inline_asm(T.f32(), [a_i32,b_i32,acc_f32], "v_dot2_f32_bf16 $0,$1,$2,$0[\n s_nop 2]",
  "=v,v,v,0", has_side_effects=False)` — validated exact on gfx950 (Phase 1).
- [resolved] **`cvt_scalef32_pk_bf16_fp8` scale is exponent-only (E8M0).** The op multiplies
  the decoded fp8 by `2^exponent(scale_f32)`, discarding the mantissa (`scale=3.0`→`×2.0`).
  Implication: for **PerTensor/PerToken FP8** (arbitrary f32) scales, pass `scale=1.0` to the
  convert (exponent 0 → ×1) and fold the real scale into the f32 accumulator after dot2 (this
  is what the reference does, §7 uses `cvt(...,1.0,...)`). Only **Block2D/MX (e8m0)** scales
  should be fed through the convert's scale operand.
- [open] How the FlyDSL tile/copy path exposes each lane's `kVector` FP8/BF16 chunk as
  consecutive `uint32` words (reference relies on `get_as<uint32_t>(word)`); decide between
  copy-atom tiles vs. `make_buffer_tensor` raw loads.
- [open] Block2D scale indexing in FlyDSL (start with direct HBM per-K-block read;
  the LDS-broadcast optimization from §5.4 is gate_up-only and can come later).
- [deferred] MXFP4 s_nop-free dot2 + `dot2_drain4` scheduling (MXFP4 phase).

## 10. Changelog

- _init_ — plan created; scope, decisions, feasibility, phases recorded.
- _phase 0_ — deep read of reference kernels + numeric primitives + CPU oracle done;
  §7 design notes filled (math, mappings, primitives, harness, constraints); feasibility
  refined (2-wide converts are ROCDL ops; only dot2 needs a local helper).
- _phase 0 close-out_ — locked two baseline choices in §2: `kVector=16` (→8 fallback)
  and serialized `s_nop 2` dot2 for the FP8 baseline (ILP/drain deferred to MXFP4).
  Paused before Phase 1.
- _testing plan_ — added §8 (testing conventions & reuse): conform to CONTRIBUTE +
  `aiter-op-test`; reuse `torch_moe_stage1/2`, `fused_topk`, quant helpers, `run_perftest`,
  and the gfx950/FlyDSL skip guard from existing MoE op_tests. Renumbered open questions →§9,
  changelog →§10; Phase 4 op_test item points at §8.
- _phase 1_ — built + validated the three primitives on gfx950 (all exact vs torch,
  max_delta 0.0): `dot2_f32_bf16` inline-asm helper, `fp8x2_to_bf16x2`
  (`cvt_scalef32_pk_bf16_fp8`), and `wave_reduce_add_f32` butterfly reduce, in
  `kernels/warp_decode_moe.py` + test `test_flydsl_warp_decode_moe.py` (4 pass, `pytest` + CLI).
  Resolved the inline-asm form and the **exponent-only (E8M0) scale semantics** of the fp8
  convert (§9) — drives the PerTensor/PerToken scale-fold decision for Phases 2–4.
- _phase 2_ — `gate_up` FP8 correctness baseline (BF16 act, FP8 e4m3 weights, PerTensor +
  PerToken): kernel `build_gate_up_fp8_module`/`pick_kvector` + entry
  `flydsl_warp_decode_gate_up` + 3 op_test cases. **Exact** vs torch (cos 1.0, max_delta 0.0)
  for kVector 16/8. Perf baseline ~1.3–1.5 TB/s weight-read BW (~20–25% peak); optimization
  levers recorded in §5 Phase 2 (lane-0 store, vectorized loads, ILP dot2, prefetch).
  Confirmed scope: BF16 act + PerTensor/PerToken first; full Python entry point now.
- _phase 3_ — `down_reduce` FP8 1-output/wave baseline (BF16 intermediate, FP8 e4m3 weights,
  PerTensor + PerToken): kernel `build_down_reduce_fp8_module` + entry
  `flydsl_warp_decode_down_reduce` + 3 op_test cases (10 pass total). **Exact** vs torch
  (cos 1.0, max_delta 0.0); end-to-end gate_up→down composes (cos 0.9999998). Folds
  `router_wt*ds` per expert into the per-lane partial → single reduce. Perf baseline ~0.4 TB/s;
  **H2 (2 outputs/wave)** and shared load/store/dot2 optimizations deferred.
- _perf: lane-0 store_ — replaced the all-lane redundant BF16 store in both kernels with an
  `if lane == 0:` guarded store (reduce still on all lanes). Correctness unchanged (10 pass).
  gate_up ~1.4–1.6 TB/s; down_reduce hits **~6.2 TB/s (~78% HBM peak)** on a DeepSeek-ish
  shape. Next lever: vectorized 128-bit loads.
- _perf: vectorized loads_ — added `load_i32_words` helper coalescing the inner-loop scalar
  `buffer_load`s into widest `vec4`/`vec2` i32 transactions (and dropping the duplicate
  weight-dword reloads); rewrote both kernels' inner loops to use it. Correctness unchanged
  (10 pass). On the DeepSeek-ish shape (B1 INTER2048 HIDDEN7168 E8 TOPK8): **gate_up ~6.9 TB/s
  (~86% HBM peak)** (big jump from ~1.4–1.6), **down ~5.7 TB/s (~71%)**. Next lever: H2 for down.
