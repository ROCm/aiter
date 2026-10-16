# MegaMoE EP perf-optimization brief (shared by optimizer + judge subagents)

## Goal
Optimize the one-launch MegaMoE **EP** kernel so its performance **reaches or beats the
two-stage MegaMoEV2** on **8-GPU, a4w4, v4_pro**, WITHOUT regressing correctness.

## Where (edit the host tree; it is bind-mounted into the container)
- Repo: `/shared/amdgpu/home/yanbo_shao_qle/ywx/megamoe/aiter` (host) == `/home/ywx/megamoe/aiter` (in `xyw_docker`). Branch `megamoe_ep`.
- EP module: `aiter/ops/flydsl/kernels/mega_moe_ep/`:
  - `gemm.py` — THE GEMM (currently naive; the optimization target).
  - `schedule.py` — dyn_plan / gather_recv_chunk / compute_units / unit_tile (planner, in LDS).
  - `communication.py` — dispatch_send / combine_send / combine_final / barriers.
  - `mega_moe_ep_kernel.py` — kernel entry + roles() bands.
  - `mega_moe_ep_config.py` — geometry + LDS layout (L_A/L_AS/planner tables).
  - `mega_moe_ep.py` — host engine (forward, _launch_args, arena, scratch).
  - `common.py` — re-exported device primitives.
- Reference = V2 (the optimization SOURCE; use these, NOT the TP kernel):
  - `aiter/ops/flydsl/kernels/mega_moe/gemm_util.py` — atoms: `ATileLoader` (A-ring LDS
    staging, supports `indexed_input`+`row_map_rsrc` = source-key gather), `AS2RLoader`,
    `MfmaScaleGU` (`.call` / `.call_pipe`), `BWeightLoader`, `BScaleLoader`, `AScaleLoader`
    (also `indexed_input`), `SiluQuantEpilogue` (cshuffle→act→quant→global).
  - `aiter/ops/flydsl/kernels/mega_moe/gemm1.py` — `build_fused_gemm1(...)` + `do_tile(...)`
    (the software-pipelined GEMM1: A double-buffer + `call_pipe` + weight prefetch).
  - `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`, `mega_moe_v2.py`, `dispatch.py`.

## Current state
- EP is **numerically correct** — relL2 single-rank + real 8-GPU both pass. NEVER regress this.
- EP GEMM is **naive**: `gemm.py` reads A direct-from-global per MFMA fragment (ported from
  the TP raw `mfma`), no LDS A staging, no software pipeline, re-reads A per N-tile. L_A/L_AS
  LDS is reserved but unused. This is latency-bound.
- Perf (8-GPU, a4w4, v4_pro, eager e2e, worst-rank, 20 iters), bs = 8/64/256/512/1024:
  - EP: 2.70 / 4.32 / 5.82 / 5.69 / 11.36 ms
  - V2: 0.41 / 0.56 / 0.57 / 0.64 / 0.86 ms   (target)
  - TP: 0.19 / 0.37 / 0.47 / 0.79 / 1.31 ms   (context only; different parallelization)
  - EP is 5-15x slower. GEMM FLOPs are ~0.04 ms at bs=8 -> the gap is orchestration/memory
    latency (no overlap/pipeline), NOT compute.

## Plan (adopt V2's gemm_util pipelined GEMM; TP's GEMM opts do NOT apply to EP)
1. Replace EP's naive GEMM with V2's **pipelined** path (`build_fused_gemm1` / the atoms),
   using **`indexed_input=True` + `row_map_rsrc`** for the source-key gather
   (`row_map[r] = L_RIX[r] // topk` = payload row). This brings A-ring LDS double-buffering,
   `MfmaScaleGU.call_pipe`, weight prefetch, and the cshuffle epilogue.
2. **Integration bridge**: the atoms consume GLOBAL resources (`row_map_rsrc`, `expert_rsrc`,
   `trb_rsrc`/`tib_rsrc`), but EP's planner/gather results live in LDS (`L_RIX`, unit fields in
   `L_CTL`). So the planner (or `compute_units`) must write per-unit `row_map`, `expert`,
   `tile_row_base` into a **per-CTA GLOBAL scratch**, then call the V2 tile runner per unit.
3. **LDS views** the atoms need (`a_buf`/`a_scale_lds`/`c_tile`): either add typed `pool` +
   `A_scale` members to the kernel's Shared struct (like `compile_gemm1.SharedStorage`), or
   derive from the flat buf with
   `fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Shared, alignment=4), uni(i32(L)+i32(off)))`
   — IMPORTANT: `T.i32` from `flydsl.expr.typing`, NOT `fx.Int32` (that was the original bug).
   Config `L_A` (= `max(2*a_lds_size, RG*cs_tile_n*4)`) and `L_AS` (= `RG*(H//32)`) are already
   sized to match compile_gemm1's pool / A_scale.
4. **GEMM2**: take V2 stage2's down-proj GEMM; its scatter is P2P-entangled, so keep the GEMM
   and feed EP's `combine_send` (gout -> source `part`).
Do it incrementally and keep correctness green at every step (fall back / revert anything that
regresses relL2 or hangs).

## Build / run (container `xyw_docker`, podman). ALWAYS check GPU first (shared box).
Common env prefix:
```
podman exec -i -e PYTHONPATH=/home/ywx/megamoe/.pydeps_flydsl0341:/home/ywx/megamoe/aiter \
  -e AITER_USE_SYSTEM_TRITON=1 -e PYTHONDONTWRITEBYTECODE=1 -e GPU_ARCHS=gfx950 \
  -e PYTORCH_ROCM_ARCH=gfx950 -e AITER_JIT_DIR=/home/ywx/megamoe/aiter/aiter/jit \
  <EXTRA_ENV> xyw_docker bash -lc 'cd /home/ywx/megamoe/aiter && <CMD>'
```
- GPU check: `rocm-smi --showuse | grep "GPU use"` (inside xyw_docker). If other users' jobs
  occupy GPUs, WAIT (never kill others'); kill only your own stale torchrun/python. Perf is
  noisy under contention — measure only when the 8 GPUs are ~idle.
- Lint: on host, `python3 -m py_compile aiter/ops/flydsl/kernels/mega_moe_ep/*.py` and
  `ruff check aiter/ops/flydsl/kernels/mega_moe_ep/*.py`.
- Compile trace: EXTRA_ENV none, CMD `python3 _ep_compile_probe.py` (surfaces device trace errors).
- CORRECTNESS 8-GPU: EXTRA_ENV `-e EP_QUANT=a8w4` (and a4w4), CMD
  `torchrun --standalone --nproc_per_node=8 _ep_8gpu.py`. Every bs line must be `PASS err=False`,
  relL2 a8w4 < 0.10 / a4w4 < 0.25. Single-rank sweep: `python3 _ep_bringup_probe.py` (EP_QUANT env).
- PERF 8-GPU: EXTRA_ENV `-e ENGINE=EP` (also V2 for the baseline), CMD
  `torchrun --standalone --nproc_per_node=8 _ep_bench.py` -> `[EP] bs=N e2e=X.XXXms`.

## HARD correctness gates (never trade for perf)
- 8-GPU relL2 a8w4 < 0.10 and a4w4 < 0.25, all 8 ranks err=0, bs 8/64/512.
- Single-rank relL2 sweep bs 1..512, both quants, all PASS.

## Perf target
EP eager-e2e (8-GPU, a4w4, worst-rank) <= V2 at each bs. Report EP vs V2 (and the ratio) per bs
(8/64/256/512/1024). "Reached" = EP <= ~1.0x V2 across the sweep.
