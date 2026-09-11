# cco-LSA intranode MoE dispatch / combine (ops v2, FlyDSL)

Intranode (single-node, EP8) MoE dispatch + combine built on **mori-cco LSA**
(intra-node P2P over the flat symmetric VA) and **FlyDSL** device kernels. A
mori-parity reimplementation that swaps the mori-shmem provider for cco-LSA:
peer addresses are computed in-kernel via `cco.Window(arena).lsa_ptr(pe, off)`,
no host P2P tables. Reference = ROCm/FlyDSL PR #522
(`dispatch_combine_intranode_{kernel,op}.py`).

Supported token dtypes: **bf16**, **f32**, **fp8** (gather-only; OCP e4m3 on
gfx950, e4m3**fnuz** max 240 on gfx942) and **fp4** (e2m1, gather-only,
**gfx950-only** — the `cvt_scalef32_*_fp4` intrinsics don't exist on gfx942).
Combine: gather (UseP2PRead) **and** scatter (`_nop2p`); weighted combine
(`out_weights`); StdMoE (ConvertDispatchOutput / ConvertCombineInput, standalone
+ wired into the op); fp8 combine-wire **quant** (`fp8_direct_cast` **and**
`fp8_blockwise`, scatter-only — distinct from the plain fp8 token dtype, which
keeps a bf16 external payload); per-token scales forwarding;
`max_total_recv_tokens` cap; mori-parity host op-layer + per-device,
dtype-aware tuning table. Not done: `skip_stage1` (FlyDSL-only).

> **Test-only, not a mori API (yet).** These modules import each other by
> top-level name (`from intranode_kernels import ...`, `import flydsl_prims as P`)
> and have no `__init__.py`; they only resolve after the tests do
> `sys.path.insert(0, <this dir>)`. There is no `mori.ops.dispatch_combine_v2`
> package export, so `import mori.ops.dispatch_combine_v2.dispatch_combine_op`
> will fail. Use it via the test/bench harnesses below. Wiring it in as a real
> package (relative imports + `__init__.py` + `ops/__init__.py` export) is a
> follow-up.

## Layout

| file | role |
|---|---|
| `flydsl_prims.py` | device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | all FlyDSL intranode kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `dispatch_combine_op.py` | `SymmArena` + `EpDispatchCombineOp` / `EpDispatchCombineConfig` (+`.tuned()`) / `EpDispatchRoutingHandle` — mori-parity host op-layer (scales, scatter/quant combine, StdMoE, recv cap, LEC, reset, replay) |
| `tuning_configs.py` | per-(world,hidden,topk) block/warp lookup |

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay) |
| `bench_dispatch_combine.py` | eager + CUDA-graph perf bench + e2e correctness. Envs: `DTYPE=bf16\|f32\|fp8\|fp4`, `COMBINE=gather\|scatter`, `QUANT=none\|fp8_direct_cast\|fp8_blockwise`, `STDMOE=1`, `SCALE_DIM`, `SWEEP`, `DISP_BLOCK`/`COMB_BLOCK`, `WARP_NUM`/`COMB_WARP`, `MODE`, `TUNED` |
| `run_bench.sh` | bench launcher (runs `bench_dispatch_combine.py` in the container) |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. Intranode only (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
torchrun --standalone --nproc_per_node=8 bench_dispatch_combine.py    # perf + e2e correctness
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DTYPE`, `COMBINE`, `QUANT`,
`DISP_BLOCK`/`COMB_BLOCK`, `WARP_NUM`/`COMB_WARP`, `MODE=eager|graph|both`, `TUNED`.

## Design notes

- **dispatch**: per (token, k) dedup same-dest-PE via ballot; lane0 remote
  `atomic_add` allocates a recv slot; publish origin id + idx/wts + 16B dual-issue
  token copy to the peer; grid barrier; per-peer count signal; collect `total_recv`.
- **combine** (gather, = mori `UseP2PRead`): cross-device entry barrier, then each
  local token gathers its k expert outputs **remotely** from `peer.out_tok[dest_tok_id]`
  and reduces in f32. Register-light i32 reads (2 bf16 / `v2f32` accumulate) + 2-way
  unroll keep VGPRs low so 16 warps/block run at high occupancy to hide xGMI read
  latency; remote reads are latency-bound so combine needs ~128 blocks, while
  dispatch's posted writes saturate at ~64 blocks (half the CUs).
- Self-written volatile/atomic spin-waits (`flydsl_prims.spin_until_*`) — mori-shmem's
  `wait_until_*` assert on a cco-only stack. Counters self-reset in-kernel → CUDAGraph-safe.

## Perf (EP8, hidden=7168, top-k=8, 256 experts; dispatch 64blk / combine 128blk × 16warp, CUDA-graph, bf16)

Per-rank bandwidth = `recv_tok * per_token_bytes / time` (the bench sizes the
payload per dtype, `hidden*2` for bf16). Indicative bf16 numbers on **MI308X
(gfx942)** xGMI:

| tok/rank | dispatch | combine |
|---:|---:|---:|
| 512  | 268 GB/s | 213 GB/s |
| 2048 | 306 GB/s | 294 GB/s |
| 8192 | 314 GB/s | 323 GB/s |

Cross-impl (v2 vs mori v1) latency tables for fp8/fp4 are in PR ROCm/mori#448.
