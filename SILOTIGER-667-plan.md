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

## 3. Feasibility (verified)

The fast path rests on three primitives, all reachable in FlyDSL on gfx950 wave64:

| Primitive | Availability | Plan |
|---|---|---|
| Packed FP8→BF16 convert (`v_cvt_scalef32_pk_bf16_fp8`) | ROCDL op `rocdl.cvt.scalef32.pk8.fp8.bf16` present | Use generated op directly |
| Packed FP4→BF16 convert (`..._fp4`) | ROCDL op `rocdl.cvt.scalef32.pk8.fp4.bf16` present | Use directly (MXFP4 phase) |
| `v_dot2_f32_bf16` (BF16·BF16→FP32 dot) | **Not** a ROCDL op | **Local inline-asm helper** |
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

### Phase 0 — Study the reference in full  [ ]
- [ ] Read `gate_up`, `down_reduce`, `warp_decode_numeric` kernels end-to-end.
- [ ] Extract exact lane→data mapping (lane `l` owns `[l*kVector, (l+1)*kVector)`).
- [ ] Understand Block2D scale broadcast through LDS.
- [ ] Understand `s_nop` drain scheduling / independent-accumulator pattern.
- [ ] Read the CPU reference in `test/ck_tile/warp_decode/` → this is the correctness oracle.
- **Output:** concrete lane-mapping design notes appended to §7 below.

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

## 7. Design notes (filled in during Phase 0)

_TBD — lane mapping, scale broadcast, accumulator/drain scheduling._

## 8. Open questions / risks

- Exact inline-asm operand/constraint string for `v_dot2_f32_bf16` (VOP2/VOP3 form, s_nop drain).
- Whether the `pk8` (8-wide) converts match the reference's `kVector` choices without extra packing.
- Block2D scale LDS-broadcast mapping in FlyDSL layout algebra.

## 9. Changelog

- _init_ — plan created; scope, decisions, feasibility, phases recorded.
