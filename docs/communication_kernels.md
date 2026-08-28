# Communication kernels in aiter: inventory, algorithms, and gfx950 testing

Status: initial draft. Covers what exists today, how the kernels work, what CI
actually runs, and how to exercise them across the GPUs of a gfx950 node.

The immediate target is **DeepSeek-V4 at TP4 and TP8** — §8 narrows everything
here to that model and evaluates the dispatch boundaries at its actual
dimensions. §1–§7 are the general picture it rests on.

Everything below was read out of the tree at the time of writing; file/line
references are the source of truth if this drifts.

---

## 1. The four stacks

aiter has four largely independent communication implementations. They are not
alternatives to each other in a clean way — they overlap in what they compute
but differ in transport, dtype support, and who dispatches them.

| Stack | Transport | JIT module | Entry point |
|---|---|---|---|
| Custom all-reduce (HIP/C++) | hipIPC peer pointers (VMM on gfx1250) | `module_custom_all_reduce` | [aiter/ops/custom_all_reduce.py](../aiter/ops/custom_all_reduce.py) |
| Quick all-reduce (quantized) | hipIPC peer buffers | `module_quick_all_reduce` | [aiter/ops/quick_all_reduce.py](../aiter/ops/quick_all_reduce.py) |
| Triton / Iris | Iris symmetric heap, GPU-initiated | n/a (Triton JIT) | [aiter/ops/triton/comms/](../aiter/ops/triton/comms/) |
| Mori / FlyDSL EP all2all | mori shmem heap | n/a | [aiter/dist/device_communicators/all2all.py](../aiter/dist/device_communicators/all2all.py) |

RCCL sits underneath all of them as the fallback, reached through
[communicator_pynccl.py](../aiter/dist/device_communicators/communicator_pynccl.py).

---

## 2. Background: what the algorithms actually do

### 2.1 Why not just call RCCL

RCCL is ring- or tree-based and is tuned for bandwidth at large message sizes.
LLM tensor-parallel inference spends most of its collective time on *small*
messages — a decode step all-reduces a `(batch, hidden)` activation, often only
tens of kilobytes. At that size the collective is latency-bound, and a ring
costs `2(N-1)` sequential hops. On a fully XGMI-connected node every rank can
read every other rank's memory directly, so you can do the whole thing in one or
two hops. That is what the custom kernels exploit.

The second motive is fusion. An all-reduce in a transformer is almost always
followed by an RMSNorm and often a quantization. Each of those, done separately,
re-reads the activation from HBM. Fusing them into the all-reduce epilogue means
the data is normalized and quantized while it is still in registers.

### 2.2 The signal protocol

All custom kernels synchronize through a `Signal` struct that lives in
IPC-shared memory, one per rank, with a per-block flag array. See
[custom_all_reduce.cuh:156-260](../csrc/include/custom_all_reduce.cuh#L156-L260).

`start_sync` has each block's first `ngpus` threads write a monotonically
increasing flag into *every* peer's signal slot, then spin until its own slot has
been written by all peers. That is one p2p write of latency, not a tree. The
counter is monotonic (`flag = _flag[blockIdx.x] + 1`) rather than a reset-to-zero
boolean, which is what makes it safe to reuse across CUDA-graph replays.

`end_sync` is the same shape but uses release/acquire ordering so that data
written before the barrier is visible after it. `final_sync=true` downgrades to
relaxed, since the last barrier in a kernel has nothing to publish.

Note the comment at [custom_all_reduce.cuh:44](../csrc/include/custom_all_reduce.cuh#L44):
peer atomics are not supported over PCIe, which is why this uses plain writes plus
spinning rather than atomics. It also means the whole thing is only valid
intra-node over XGMI.

### 2.3 One-shot ("1stage") all-reduce

Every rank reads the full input from all `N` peers and reduces locally. Traffic
is `N-1` full-size reads per rank; latency is a single round trip. Optimal below
a few hundred KB.

Two implementations exist:

- `cross_device_reduce_1stage_naive`
  ([:278](../csrc/include/custom_all_reduce.cuh#L278)) — each thread reads a
  16-byte pack from each peer and accumulates in fp32. Straight-line, no LDS.
- `cross_device_reduce_1stage`
  ([:377](../csrc/include/custom_all_reduce.cuh#L377)) — the tuned version.
  The 512-thread block is split into `N` warp-groups of `512/N` lanes; warp-group
  `g` is responsible for pulling rank `g`'s data into LDS, and warp-group 0 does
  the reduction out of LDS. It double-buffers: while warp 0 reduces buffer `b`,
  all warps prefetch the next tile into buffer `b^1`. This turns `N` dependent
  peer loads into `N` concurrent ones and hides the XGMI latency behind the
  arithmetic.

Both keep the accumulation order identical on every rank (`ptrs[0]` first, then
1..N-1, never reordered) so that all ranks produce **bitwise identical** results.
This matters: TP ranks that disagree in the last bit will diverge over layers.

### 2.4 Two-shot ("2stage") all-reduce = reduce-scatter + all-gather

Each rank owns a `size/N` slice. Stage 1: rank `r` reduces *only its slice*,
reading that slice from all `N` peers, and writes the result into its own
IPC-visible scratch buffer. Barrier. Stage 2: every rank gathers the `N` reduced
slices back. Traffic per rank is `2 * (N-1)/N * size` versus `(N-1) * size` for
one-shot — so two-shot wins as soon as the message is big enough for bandwidth
to dominate latency.

Implementations: `cross_device_reduce_2stage_naive`
([:316](../csrc/include/custom_all_reduce.cuh#L316)) and the LDS-staged
`cross_device_reduce_2stage` ([:484](../csrc/include/custom_all_reduce.cuh#L484)).

There is a subtle correctness requirement, spelled out in the comment at
[:557-560](../csrc/include/custom_all_reduce.cuh#L557-L560): thread `i` must
gather the same index in stage 2 that it reduced in stage 1, because cross-device
visibility after the barrier is only guaranteed between threads with matching
tid. Changing the indexing of one stage without the other is a silent-corruption
bug, not a compile error — and an *intermittent* one, since it depends on how the
barrier and the memory system happen to interleave. That failure mode is why
[test_fused_ar_rms_memory_order.py](../op_tests/multigpu_tests/test_fused_ar_rms_memory_order.py)
replays the same seeded shapes for `--iters` iterations and reports the first
iteration that goes bad, rather than checking once.

### 2.5 Write-mode (push instead of pull)

`cross_device_reduce_2stage_write_mode`
([:569](../csrc/include/custom_all_reduce.cuh#L569)) inverts the data movement:
instead of each rank *pulling* its slice from peers, every rank *pushes* its
contribution into the peers' scratch buffers, reduces what landed in its own
buffer, then pushes the reduced slice out to the peers' registered output
buffers using non-temporal stores.

The trade is push-latency vs pull-latency on XGMI; the writes are fire-and-forget
where reads stall. It is currently enabled only for
`world_size == 8 && bytes > 512*4096*2 && arch == gfx942`
([:3798-3803](../csrc/include/custom_all_reduce.cuh#L3798-L3803)) — **on gfx950
this path is dead code**. Whether it should be enabled on gfx950 is an open
question worth measuring (see §9).

### 2.6 Host-side dispatch

From [custom_all_reduce.cuh:3760-3805](../csrc/include/custom_all_reduce.cuh#L3760-L3805):

```
world_size == 2                                -> 1stage (always)
full_nvlink && world<=4 && bytes < 160 KiB     -> 1stage
full_nvlink && world<=8 && bytes <  80 KiB     -> 1stage
otherwise                                      -> 2stage
```

and then `DISPATCH_REDUCE` picks the vectorized kernel only when
`bytes % (ngpus * 16) == 0 && world_size != 6`, else the `_naive` variant. So
**TP=6 always runs the naive path** — that is by construction, not an oversight,
but it means TP=6 has a completely different performance profile and is currently
untested anywhere.

Above the kernel, the Python layer applies a size window before custom AR is used
at all: `_DEFAULT_CAR_MAX_SIZE = 8192*8192` bytes (64 MiB), overridable via
`AITER_CUSTOM_AR_MAX_SIZE` / `AITER_CUSTOM_AR_MIN_SIZE`
([custom_all_reduce.py:117-135](../aiter/dist/device_communicators/custom_all_reduce.py#L117-L135)).
Outside that window it is RCCL.

### 2.7 all_gather and reduce_scatter

These are the two halves of two-shot AR exposed as standalone ops. The
interesting part is layout. A reduce-scatter has to split along *some* dimension,
and the vectorization strategy depends on where that dimension sits. The kernel
header ([:780-790](../csrc/include/custom_all_reduce.cuh#L780-L790)) collapses
any tensor into one of three canonical shapes:

| Case | Input | Output | Kernel |
|---|---|---|---|
| scatter on first dim | `(N*m, n, ...)` | `(m, n, ...)` | `reduce_scatter_split_first_dim` |
| scatter on last dim | `(m, N*n)` | `(m, n)` | `..._split_lastdim` (+ `_naive` fallback) |
| scatter on middle dim | `(m, N*n, k)` | `(m, n, k)` | `..._split_middim` (+ `_naive`) |

First-dim scatter is trivially contiguous. Last-dim and mid-dim need
`n % (ngpus * pack_size) == 0` to pack 16 B per thread; when that fails they drop
to a naive fallback. The fallback path has an *exact* reference (no reduction
reordering), so any error there is a genuine bug — which is why
[test_reduce_scatter.py](../op_tests/multigpu_tests/test_reduce_scatter.py)
has a dedicated `--suite fallback` that asserts on non-zero error rather than
tolerance-checking.

All-gather mirrors this: `allgather_vec`, `allgather_lastdim`, `allgather_naive`.

### 2.8 Fused epilogues

Seven fused variants, all in [custom_all_reduce.cu](../csrc/kernels/custom_all_reduce.cu):

| Op | What it fuses |
|---|---|
| `fused_allreduce_rmsnorm` | AR → residual add → RMSNorm |
| `fused_allreduce_rmsnorm_pad` | same, with output padding |
| `fused_allreduce_rmsnorm_quant` | + per-token FP8 quant |
| `fused_allreduce_rmsnorm_quant_per_group` | + per-group FP8 quant |
| `fused_allreduce_rmsnorm_mxfp4_quant` | + MXFP4 (E2M1 + E8M0 block scales) |
| `fused_qknorm_allreduce` | QK-norm → AR |
| `fused_qknorm_allreduce_rope` | QK-norm → AR → RoPE |

RMSNorm is row-local, which is what makes the fusion legal: after the reduce, a
rank holds complete rows and can normalize without further communication.
Internally these share a `reduce_scatter_cross_device_store` first step
([custom_all_reduce.cuh:1258](../csrc/include/custom_all_reduce.cuh#L1258)) —
the two-shot stage 1 with the result parked in peer-visible scratch, so the
epilogue can run before the gather. The
MXFP4 variant has its own 1-stage/2-stage split with a 512 KiB LDS budget for the
reduce-scatter stage, plus a decode-shape gate
(`token_num <= 4 || (K <= 4096 && token_num <= 32) || ...`) at
[communicator_cuda.py:636-642](../aiter/dist/device_communicators/communicator_cuda.py#L636-L642),
and falls back to unfused AR+RMSNorm + `dynamic_mxfp4_quant` for shapes neither
kernel supports. Three dispatch paths, which is why the mxfp4 test sweeps
`--stage` explicitly.

### 2.9 Quick all-reduce: quantized two-shot

[quick_all_reduce.cuh](../csrc/include/quick_all_reduce.cuh). Same two-shot
skeleton as §2.4, but the data crossing XGMI is **quantized**, trading precision
for wire bandwidth. `AllReduceTwoshot::run`
([:744](../csrc/include/quick_all_reduce.cuh#L744)):

1. Load the input tile into registers via `buffer_load_dwordx4`.
2. **Phase 1A** — for each peer `r`, `codec.send()` the segment `r` is
   responsible for into `r`'s comm buffer, then set a sync flag.
3. **Phase 1B** — spin on the flags, `codec.recv()` all `N` contributions to
   *this* rank's segment, accumulate.
4. **Phase 2** — `codec.send()` the reduced segment to every peer, flag,
   then `codec.recv()` all `N` reduced segments back and write the output.

The codec is a template parameter, so the quantization is inlined into the
send/recv and never materializes in memory:

| Codec | Wire format | Notes |
|---|---|---|
| `CodecFP` | fp16/bf16 unquantized | baseline, 2 B/elem |
| `CodecFP8` | block-scaled FP8 | 1 B/elem |
| `CodecQ6` | block-scaled INT6 | 0.75 B/elem |
| `CodecQ4` | block-scaled INT4 | 0.5 B/elem |
| `CodecQ3` | block-scaled INT3 | 0.375 B/elem, **TP2 only** |

All the integer codecs are uniform symmetric quantization (round-to-int, clip)
over blocks of `4 * kThreadGroupSize` elements, with a per-block scale sent
alongside. INT3's signed range is `[-4, +3]`.

`cast_bf2half` is a separate template flag: bf16 input can be converted to fp16
for the wire (more mantissa, less exponent range) under
`AITER_QUICK_REDUCE_CAST_BF16_TO_FP16`.

Two things gate this in practice, both easy to trip over:

- **QR is entirely disabled unless `AITER_QUICK_REDUCE_QUANTIZATION` is set** to a
  regime name (`FP`/`FP8`/`INT6`/`INT4`/`INT3`). See
  [quick_all_reduce.py:30-38](../aiter/dist/device_communicators/quick_all_reduce.py#L30-L38).
- It only engages **above** a size threshold, from a hand-tuned table indexed by
  `(dtype, world_size)` and codec
  ([:79-88](../aiter/dist/device_communicators/quick_all_reduce.py#L79-L88)). Some
  entries are `2048 MB`, i.e. effectively "never" — e.g. bf16 at TP=8 for every
  quantized codec. Under bf16/TP8 only `CodecFP` is reachable at 16 MB+.

Arch gate is `gfx94*` or `gfx95*`, so gfx950 is supported.

There is also a fused `qr_all_reduce_rmsnorm`. Its dispatch is narrower than the
plain epilogue: the host only launches it when each hidden row fits evenly inside
one 32 KiB QR tile ([:947-950](../csrc/include/quick_all_reduce.cuh#L947-L950)).

### 2.10 Triton / Iris

[aiter/ops/triton/comms/](../aiter/ops/triton/comms/) implements reduce-scatter,
all-gather, and a fused `reduce_scatter_rmsnorm_quant_all_gather` on top of
[Iris](https://github.com/ROCm/iris). The model is different from everything
above: Iris exposes a *symmetric heap* — an allocation at the same offset on
every rank — and the Triton kernel issues loads/stores against remote heap
pointers directly. Communication is expressed inside the Triton kernel rather
than being a separate launch, which is what makes the RS→RMSNorm→quant→AG
pipeline fusable end to end.

Requires the `iris` package. See [docs/triton_comms.md](triton_comms.md) for the
heap-sizing helper (`calculate_heap_size`).

### 2.11 Mori EP all2all

For expert parallelism the collective is not a reduction but a *routing*: each
token goes to its top-k experts, which live on other ranks. `MoriAll2AllManager`
([all2all.py:28](../aiter/dist/device_communicators/all2all.py#L28)) wraps mori's
`EpDispatchCombineOp` — dispatch scatters tokens to expert owners, combine
gathers the expert outputs back and applies routing weights. Kernel type is
`IntraNode` or `InterNodeV1` depending on topology.

`FlyDSLAll2AllManager` ([:116](../aiter/dist/device_communicators/all2all.py#L116))
keeps mori's shmem heap for P2P buffer allocation but replaces the comm
primitives with FlyDSL-generated ones.

---

## 3. Python API surface

[aiter/dist/](../aiter/dist/) is a vLLM-shaped distributed layer:

- [parallel_state.py](../aiter/dist/parallel_state.py) — builds TP / DP / PP / EP /
  PCP groups. Layout order is `ExternalDP x DP x PP x PCP x TP`, with
  `ep_size = dp * pcp * tp`. PCP (prefill context parallel) is an independent
  dimension that grows world size; it is *not* decode context parallel, which
  reuses TP GPUs.
- [communication_op.py](../aiter/dist/communication_op.py) — ~35 entry points:
  `tensor_model_parallel_*`, `expert_parallel_*`, `data_parallel_*`,
  `pipeline_model_parallel_*`, plus `custom_all_reduce` / `custom_all_gather` /
  `custom_reduce_scatter` that operate on an explicitly passed group.
- [device_communicators/](../aiter/dist/device_communicators/) — the dispatch
  layer that chooses custom AR vs quick AR vs RCCL per call.

`aiter/ops/communication.py` is just a test/bench convenience wrapper
(`init_dist_env` / `destroy_dist_env`).

---

## 4. Environment knobs

Every one of these selects a different code path and none of them are swept in CI.

| Var | Effect |
|---|---|
| `AITER_AR_1STAGE` | `1` force 1-stage, `0` force 2-stage, unset auto |
| `AITER_AR_1STAGE_MAX_KB` | override the 1-stage size cutoff |
| `AITER_AR_QUANT_MAX_BYTES` | size cap for the quantized fused AR path |
| `AITER_AR_QUANT_NO_PREFILL_MAX_BYTES` | same, decode-only variant |
| `AITER_CUSTOM_AR_MAX_SIZE` | upper bound of the custom-AR window (default 64 MiB); `0` disables custom AR entirely |
| `AITER_CUSTOM_AR_MIN_SIZE` | lower bound of the window (default 0) |
| `AITER_CUSTOM_AR_FORCE_IPC` / `_FORCE_VMM` | force transport (gfx1250) |
| `AITER_CUSTOM_AR_DISABLE_GFX1250` | force the old-arch path on gfx1250 |
| `AITER_QUICK_REDUCE_QUANTIZATION` | **required** to enable QR at all; `FP`/`FP8`/`INT6`/`INT4`/`INT3` |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | QR buffer size |
| `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` | send bf16 as fp16 on the wire |
| `AITER_ROCM_VERSION` | override ROCm detection (transport choice) |

---

## 5. Tests and benchmarks

22 files in [op_tests/multigpu_tests/](../op_tests/multigpu_tests/), 2 in
`triton_test/`, 4 in `gfx1250_poc/`.

Structural notes, because they affect how you run them:

- They are **argparse sweep scripts run by plain `python3`**, not pytest modules.
- Ranks are spawned with `multiprocessing.Pool` (spawn start method), and rank
  `i` binds to `cuda:i`. Only the mega-MoE tests use `torchrun`.
- Correctness and perf are fused: each rank runs under `@perftest` /
  `run_perftest`, results go through `checkAllclose`, and a markdown latency
  table is printed at the end.
- Four files (`test_fused_ar_rms.py`, `test_fused_ar_mhc_post_only.py`,
  `test_fused_qknorm_ar.py`, `test_quick_all_reduce_rmsnorm.py`) *additionally*
  carry pytest classes with `device_count()` skip guards, so they are dual-mode.

| Test | Covers | TP control |
|---|---|---|
| [test_custom_allreduce.py](../op_tests/multigpu_tests/test_custom_allreduce.py) | custom AR, ±CUDA graph | hardcoded 8 |
| [test_custom_allreduce_fp8.py](../op_tests/multigpu_tests/test_custom_allreduce_fp8.py) | AR on fp8 inputs | hardcoded 8 |
| [test_quick_all_reduce.py](../op_tests/multigpu_tests/test_quick_all_reduce.py) | QR codecs, variable input, INT3@TP2 | hardcoded 8 / 4 / 2 |
| [test_quick_all_reduce_rmsnorm.py](../op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py) | QR+RMSNorm epilogue | pytest, world=2 |
| [test_allgather.py](../op_tests/multigpu_tests/test_allgather.py) | AG custom vs RCCL, dim 0 and -1 | `-t 2\|4\|8` |
| [test_reduce_scatter.py](../op_tests/multigpu_tests/test_reduce_scatter.py) | RS custom + forced-fallback suite | `-t`, `--suite custom\|fallback\|all` |
| [test_fused_ar_rms.py](../op_tests/multigpu_tests/test_fused_ar_rms.py) | AR+RMSNorm(+quant), largest case matrix | `-t/-p`, `--test fused quant` |
| [test_fused_ar_rms_per_group_quant.py](../op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py) | per-group FP8 + validator negative tests | `-t`, `--sweep-group-size` |
| [test_fused_ar_rms_mxfp4_quant.py](../op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py) | MXFP4; CI smoke (~2 min) vs `--full` (~50 min) | `-t`, `--stage`, `--emit-bf16` |
| [test_fused_ar_rms_memory_order.py](../op_tests/multigpu_tests/test_fused_ar_rms_memory_order.py) | fused AR+RMSNorm memory-ordering regression; replays seeded shapes N times looking for an intermittent bad iteration | `--tp`, `--iters`, `--benchmark` |
| [test_fused_qknorm_ar.py](../op_tests/multigpu_tests/test_fused_qknorm_ar.py) | QK-norm+AR(+RoPE) | `--tp-sizes 2,4,8` |
| [test_fused_ar_mhc_post_only.py](../op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py) | split vs fused AR+mhc_post | `-t`, `--breakdown` |
| [test_custom_group.py](../op_tests/multigpu_tests/test_custom_group.py) | AR/AG/RS on custom TP/DP/2D/multi groups | hardcoded world 8 |
| [test_parallel_groups.py](../op_tests/multigpu_tests/test_parallel_groups.py) | TP×DP×PP×PCP group construction | `--tp --dp --pp --pcp -w` |
| [test_communication.py](../op_tests/multigpu_tests/test_communication.py) | AR + AR-RMSNorm-quant, graph capture | hardcoded 8 |
| [test_car_rccl_latency.py](../op_tests/multigpu_tests/test_car_rccl_latency.py) | **bench**: custom AR vs RCCL latency | — |
| [test_collective_profile.py](../op_tests/multigpu_tests/test_collective_profile.py) | `record_param_comms` instrumentation, chrome trace | `device_count()` |
| [test_mori_all2all.py](../op_tests/multigpu_tests/test_mori_all2all.py), [test_dispatch_combine.py](../op_tests/multigpu_tests/test_dispatch_combine.py) | Mori EP all2all + fused_moe | needs `mori` |
| [test_mega_moe_v2.py](../op_tests/multigpu_tests/test_mega_moe_v2.py) / [bench_mega_moe_v2.py](../op_tests/multigpu_tests/bench_mega_moe_v2.py) | multi-layer EP MoE accuracy + **perf guards** | torchrun 8 |
| [test_mega_moe_gfx1250.py](../op_tests/multigpu_tests/test_mega_moe_gfx1250.py) | gfx1250 fused-scatter MoE | torchrun 8, arch-gated |
| [triton_test/](../op_tests/multigpu_tests/triton_test/) ×2 | Iris RS/AG, fused RS+RMSNorm+quant+AG | `-n num_gpus`, `--heap_size` |
| [gfx1250_poc/](../op_tests/multigpu_tests/gfx1250_poc/) ×4 | MI450 AR/AG, XGMI P2P BW, sync latency | MI450 only |

---

## 6. What CI runs

One job: **Multi-GPU Tests (8 GPU)**,
[aiter-test.yaml:862](../.github/workflows/aiter-test.yaml#L862). The matrix has
exactly one runner, `linux-aiter-do-mi350x-8` — so *all* communication coverage
in this repo is a single 8-GPU gfx950 node.

It sets `MULTIGPU=TRUE`, and [aiter_test.sh](../.github/scripts/aiter_test.sh)
then runs `python3 <file>` over every `.py` under `multigpu_tests/` recursively,
unsharded, 60 min per file, 120 min job cap. Three files get special torchrun
invocations (`bench_mega_moe_v2.py`, `test_mega_moe_gfx1250.py`,
`test_mega_moe_v2.py`).

Six files are hard-skipped
([aiter_test.sh:37-44](../.github/scripts/aiter_test.sh#L37-L44)):
`test_dispatch_combine`, `test_communication`, `test_mori_all2all`,
`test_fused_ar_rms` (skipped 2026-03-15 in #2280), and both `triton_test/` files.

### Gaps

1. **Only TP=8 is exercised for the core AR path.** `test_custom_allreduce*`,
   `test_quick_all_reduce`, `test_custom_group`, `test_communication` hardcode
   world 8 in `__main__`. The 1-stage/2-stage dispatch boundary (§2.6) is a
   function of both world size and byte count, and CI only ever samples one
   corner of it. The sole exception is
   `test_fused_ar_rms_mxfp4_quant.py`, which sweeps `tp_size ∈ {2,4,8}` by
   default — see §8.3.
2. **TP=6 is never run.** `CustomAllreduce._SUPPORTED_WORLD_SIZES` advertises
   `{2,4,6,8}`, and TP=6 is explicitly routed to the `_naive` kernels — a
   distinct code path with zero coverage. The fused mxfp4 path excludes it
   outright (`self.world_size != 6`).
3. **Quick all-reduce is dark.** Nothing in the workflow sets
   `AITER_QUICK_REDUCE_QUANTIZATION`, so the FP8/INT6/INT4/INT3 codecs are only
   covered to the extent `test_quick_all_reduce.py` forces them internally.
4. **`gfx1250_poc/*` has no arch guard and runs on the gfx950 runner.**
   `test_gfx1250_allreduce.py` only warns, and only behind an opt-in
   `--check-arch` flag; the other three call `*_gfx1250` ops unconditionally.
   They will JIT-build an `ENABLE_CK=0` module and consume runner time on
   hardware they do not target. Contrast `test_mega_moe_gfx1250.py`, which the CI
   script arch-gates explicitly.
5. **No perf regression gate for comms.** `tuned_op_bench` consumes only the
   1-GPU mi35x logs. The comm latency tables are printed and discarded.
   `bench_mega_moe_v2.py --perf-guard` is the only perf assertion in the whole
   multi-GPU job.
6. **Write-mode is gfx942-only** (§2.5) and therefore never exercised on the
   gfx950 CI runner at all.

---

## 7. Running these on a gfx950 node

The reference box is 8× gfx950, all-pairs XGMI (`rocm-smi --showtopotype` shows
XGMI for every pair) — the same topology class as the CI runner.

Because ranks bind to `cuda:<rank>`, `HIP_VISIBLE_DEVICES` is the device-selection
lever: it decides *which* physical gfx950s participate and in what order.

> If your shell has `HIP_VISIBLE_DEVICES` pinned to a single GPU (common when
> benchmarking single-GPU kernels), every multi-GPU test will see one device and
> either hang or silently degenerate. Clear it or override it per command.

### Verified example

```bash
HIP_VISIBLE_DEVICES=0,1 python3 op_tests/multigpu_tests/test_allgather.py \
    -t 2 -s 128,8192 -d bf16
```

Result on the reference box (`err=0`, both dims):

| tp | shape | dtype | use_custom | dim | min_us | max_us |
|---|---|---|---|---|---|---|
| 2 | (128, 8192) | bf16 | False (RCCL) | 0 | 49.74 | 49.75 |
| 2 | (128, 8192) | bf16 | True | 0 | 45.94 | 45.96 |
| 2 | (128, 8192) | bf16 | False (RCCL) | -1 | 57.68 | 57.70 |
| 2 | (128, 8192) | bf16 | True | -1 | 46.04 | 46.06 |

The first invocation JIT-builds `module_custom_all_reduce` (~2 min); it is cached
in `aiter/jit/build/` afterwards.

### The four axes worth sweeping

**Axis 1 — world size.** Drive every `-t`-capable test across 2/4/8, and add 6
for custom AR to cover the naive path:

```bash
for tp in 2 4 8; do
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
    python3 op_tests/multigpu_tests/test_allgather.py -t $tp -d bf16
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
    python3 op_tests/multigpu_tests/test_reduce_scatter.py -t $tp --suite all
done
```

The hardcoded-8 tests need a `-t` flag threaded into `__main__` before they can
join this loop — a small, self-contained change.

**Axis 2 — device subset.** On uniform XGMI the subsets are not
bandwidth-differentiated, but they *are* NUMA- and PCIe-differentiated, and they
catch rank↔device assumptions. Run the same TP=2 case as `0,1` / `0,4` / `3,7`,
and TP=4 as `0,1,2,3` vs `0,2,4,6`, then diff the latency tables. Any spread is
real topology signal; any correctness delta is a bug.

**Axis 3 — kernel variant.** Do not trust the default dispatch to reach every
kernel; force it:

```bash
# both sides of the 1-stage / 2-stage split, same shape
for s in 0 1; do
  AITER_AR_1STAGE=$s HIP_VISIBLE_DEVICES=0,1,2,3 \
    python3 op_tests/multigpu_tests/test_fused_ar_rms.py -t 4 --test fused
done

# light up the quick-reduce codecs (dark by default)
for q in FP FP8 INT6 INT4; do
  AITER_QUICK_REDUCE_QUANTIZATION=$q HIP_VISIBLE_DEVICES=0,1,2,3 \
    python3 op_tests/multigpu_tests/test_quick_all_reduce.py -d fp16
done
# INT3 is TP2-only
AITER_QUICK_REDUCE_QUANTIZATION=INT3 HIP_VISIBLE_DEVICES=0,1 \
  python3 op_tests/multigpu_tests/test_quick_all_reduce.py -d fp16
```

Watch the QR min-size table (§2.9) when interpreting results: for bf16 at TP=8
every quantized codec has a 2048 MB threshold, so QR will simply not engage and
you will be measuring the fallback. Test QR at fp16 and/or smaller world sizes.

**Axis 4 — graph vs eager.** `-g true` / `-g false` where exposed. CUDA-graph
buffer registration is a distinct code path, and the monotonic-counter barrier
design (§2.2) exists specifically to survive replay.

### Measurement hygiene

- Idle sclk on these parts sits around 94 MHz and ramps *during* the first
  measurement. Discard the first run; cold vs warm has been observed to differ by
  4.3× on identical commands. This matters more here than usual because you will
  be diffing latency tables across device subsets.
- `sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'` — aiter warns about this
  on every rank, and for XGMI traffic it is not just noise.
- Run one process per configuration rather than sweeping inside one process;
  large sweeps in a single process have been observed to OOM on tail shapes in
  other aiter tests.
- The spawn-Pool tests leave stragglers (`CudaIPCTypes` producer-terminated
  warnings appear even on a clean pass). After a hang, use
  [.github/scripts/clean_up_rocm.sh](../.github/scripts/clean_up_rocm.sh), which
  kills all ROCm PIDs.

### Not runnable without extra packages

`mori` and `iris` are not installed on the reference box, which blocks
`test_mori_all2all.py`, `test_dispatch_combine.py`, both `triton_test/` files,
and the mega-MoE tests — i.e. roughly a third of the comm surface, and exactly
the subset CI also skips.

---

## 8. DeepSeek-V4 at TP4 and TP8

This is the initial focus, so this section narrows everything above to the
kernels that DSv4 actually touches, with the dispatch boundaries evaluated at
its real dimensions.

### 8.1 Model geometry, as it appears in this tree

From [test_mega_moe_v2.py:25-31](../op_tests/multigpu_tests/test_mega_moe_v2.py#L25-L31)
and [test_mega_moe_gfx1250.py:27](../op_tests/multigpu_tests/test_mega_moe_gfx1250.py#L27):

| Field | Value |
|---|---|
| `model_dim` (hidden) | **7168** |
| `inter_dim` | 3072 |
| experts | 384 |
| top-k | 6 |
| MoE layers | 61 |
| routing score fn | `sqrtsoftplus` (V4-Pro default, [topk.py:54](../aiter/ops/topk.py#L54)) |
| MoE quant | `a8w4_mxfp4` |
| swiglu limit | 10.0 |

Attention is MLA with the V4.0 layout `nope=448 / rope=64 / qk_packed=512`
([mla.py:1009-1012](../aiter/mla.py#L1009-L1012)) plus the V4-Pro Indexer.

CI already pins DeepSeek-V4-Pro to **TP8** on `linux-aiter-do-mi350x-8`
([atom-test.yaml:120](../.github/workflows/atom-test.yaml#L120)) — under the
`ci:atom_full` label, not on every PR. There is no TP4 pin for DSv4 anywhere in
the tree; TP4 pins exist only for Qwen3.5-397B and Kimi-K2.7.

**hidden = 7168 is the number that drives everything below.** Every dispatch
threshold in §2.6 and §2.8 is a byte count, and at bf16 a DSv4 activation row is
`7168 * 2 = 14336 B` exactly.

### 8.2 Which kernels are on the critical path

Per transformer layer, at tensor parallelism, there are two collectives: one
after the attention out-projection and one after the MLP/MoE down-projection.
Both are all-reduces of `(tokens, 7168)`. With 61 layers that is ~122
all-reduces per forward pass, which is why the small-message latency work in §2.3
matters more here than peak bandwidth.

**Relevant:**

| Kernel | Why | Test |
|---|---|---|
| Custom AR 1-stage / 2-stage | the two per-layer TP all-reduces | [test_custom_allreduce.py](../op_tests/multigpu_tests/test_custom_allreduce.py) |
| `fused_allreduce_rmsnorm` | AR + residual + norm is the actual layer boundary | [test_fused_ar_rms.py](../op_tests/multigpu_tests/test_fused_ar_rms.py) |
| `fused_allreduce_rmsnorm_mxfp4_quant` | DSv4 MoE is `a8w4_mxfp4`; the norm output feeds a quantized GEMM | [test_fused_ar_rms_mxfp4_quant.py](../op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py) |
| Mori / FlyDSL EP dispatch+combine | 384 experts at top-6 → expert parallel all2all | [test_mori_all2all.py](../op_tests/multigpu_tests/test_mori_all2all.py), [test_mega_moe_v2.py](../op_tests/multigpu_tests/test_mega_moe_v2.py) |
| `reduce_scatter` / `all_gather` | sequence-parallel layouts, and the halves of two-shot | [test_reduce_scatter.py](../op_tests/multigpu_tests/test_reduce_scatter.py), [test_allgather.py](../op_tests/multigpu_tests/test_allgather.py) |

**Probably not relevant, despite the name:**

- `fused_qknorm_allreduce` / `_rope`. The shape matrix in
  [test_fused_qknorm_ar.py:275-282](../op_tests/multigpu_tests/test_fused_qknorm_ar.py#L275-L282)
  is annotated *"MiniMax-M2 per-rank QKV geometry"*, and its per-rank widths
  (`3072/512/512` at TP2 down to `768/128/128` at TP8) are not MLA shapes. DSv4's
  QK-norm path is [fused_qk_norm_rope_cache_quant.py](../aiter/ops/fused_qk_norm_rope_cache_quant.py)
  — explicitly *"DeepSeek-V4 fused Q/K RMSNorm + RoPE + group-quant"* — which is a
  **compute** kernel with no collective in it. Don't assume the qknorm+AR fusion
  applies here without checking against the serving stack.
- Quick all-reduce. See §8.4 — largely unreachable at DSv4's sizes.

**Confirmed relevant, and currently unfused:** `fused_allreduce_mhc_post`. mHC is
manifold-constrained Hyper Connection, ported from `tilelang/examples/deepseek_mhc`
([test_mhc.py:44](../op_tests/test_mhc.py#L44)). The trace in §8.6 shows DSv4
running `mhc_post` / `mhc_pre_*` kernels at hidden 7168 with hc_mult=4, so mHC is
in the model — but as a *separate* launch from the all-reduce, meaning the fusion
exists and is not being taken.

### 8.3 Where the dispatch boundaries fall at hidden = 7168

This is the part that differs between TP4 and TP8, and it is the main reason to
test both rather than assuming TP8 coverage generalizes.

**Plain custom all-reduce** (thresholds from §2.6, `bytes = M * 14336`):

| TP | 1-stage while | → in tokens | 2-stage from |
|---|---|---|---|
| 4 | `bytes < 160 KiB` | **M ≤ 11** | M ≥ 12 |
| 8 | `bytes < 80 KiB` | **M ≤ 5** | M ≥ 6 |

Both crossovers sit *inside* the decode batch range. So at DSv4 both kernels are
on the critical path at both TPs, and they cross over at different batch sizes —
TP8 switches to two-shot more than twice as early as TP4. A batch sweep over
`M ∈ {1,2,4,5,6,8,11,12,16,32}` at both TPs walks straight through both
boundaries; anything coarser will miss them.

**Fused AR+RMSNorm+MXFP4** ([communicator_cuda.py:636-665](../aiter/dist/device_communicators/communicator_cuda.py#L636-L665)),
which is the DSv4-relevant fused path:

| Tokens | Path | Why |
|---|---|---|
| M ≤ 4 | 1-stage direct | only the `token_num <= 4` clause of `use_direct_mxfp4` can match |
| 5 ≤ M ≤ 36 | 2-stage | `total_bytes <= 512 KiB` → `M ≤ 36` |
| M ≥ 37 | fallback | unfused AR+RMSNorm then `dynamic_mxfp4_quant` |

Two things to notice. First, **K=7168 falls into a gap in the 1-stage gate**: the
clauses are `K <= 4096 && M <= 32`, `K <= 6144 && M <= 16`, and `K == 8192 && M <= 8`.
7168 matches none of them, so only the blanket `M <= 4` applies. DSv4 gets a
narrower 1-stage window than either a 6144- or an 8192-hidden model. Whether that
is deliberate or an artifact of the clause list being written for other models is
worth asking.

Second, the `prefer_2stage` override (`world_size == 8 && token_num >= 16 && K <= 6144`)
also excludes 7168 — so unlike most models, DSv4 at TP8 does *not* get the
2-stage preference, and TP4/TP8 share identical fused-mxfp4 boundaries even
though their plain-AR boundaries differ.

Both TPs satisfy the divisibility requirements: `block_size = 7168/8 = 896`, and
`896 % 8 == 0`, `896 % 4 == 0`. The vectorized (non-naive) reduce path also holds,
since `M * 14336 % (ngpus * 16) == 0` for both 4 and 8.

The good news: [test_fused_ar_rms_mxfp4_quant.py](../op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py)
already uses DSv4's hidden size in its CI shapes — `(1, 4096)`, `(16, 7168)`,
`(128, 7168)` are chosen precisely as one-per-dispatch-path at 7168 — and it is
the **one** multi-GPU test that already sweeps `tp_size ∈ {2, 4, 8}` by default
rather than hardcoding 8. It is the closest thing to a DSv4 TP4/TP8 gate that
exists today.

### 8.4 Quick all-reduce is mostly unreachable for DSv4

Cross-referencing the min-size table (§2.9) against DSv4 message sizes:

- DSv4 serves in **bf16** activations. At `(bf16, 8)` every quantized codec has a
  2048 MB threshold — a "never" sentinel — and even `CodecFP` needs 16 MB.
- 16 MB of bf16 at hidden 7168 is `M ≈ 1170` tokens. That is a prefill-sized
  all-reduce, well above the 64 MiB custom-AR cap only at `M ≈ 4681`.
- At `(bf16, 4)` it is 8 MB for `CodecFP` and 16–64 MB for the quantized codecs.

So for TP8 decode, QR never engages; for TP4 it engages only on large prefill
chunks. Unless someone intends to enable `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16`
(which moves you onto the much lower fp16 thresholds — 16 MB → 4 MB at TP8), QR
is not a DSv4 lever and should be deprioritized relative to the fused AR paths.

### 8.5 DSv4-specific gaps

Ordered after the §8.6 trace, so items 1 and 5 have changed status.

1. **No tuned EP4 dispatch/combine geometry — but not on the current path.**
   Per §8.6, DSv4 at TP4 runs tensor-parallel MoE with all 384 experts replicated
   per rank, and no all2all kernel appears in the trace. This item is therefore
   *latent*: it bites only if DSv4 moves to expert parallelism. Recorded here
   because it is silent when it does bite. The FlyDSL intranode all2all
   resolves its launch geometry from `flydsl_{arch}_{model}_{kernel}_ep{n}.json`
   ([flydsl_dispatch_combine_intranode_op.py:138-165](../aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py#L138-L165)),
   and the tuning directory contains exactly one file:
   `flydsl_gfx950_mi355x_IntraNode_ep8.json`. At EP4 the glob returns no
   candidates, `build_geometry_tuning_table_for_config` returns `None`, and both
   phases silently fall back to `block_num=128` with the default warp counts.
   **TP4 MoE runs untuned, with no warning.** Note the arch/model match is only a
   soft sort key, so an MI350X still picks up the mi355x file at EP8 — the EP4
   miss is a hard miss, not a soft one.
2. **DSv4 at TP4 is not gated anywhere.** ATOM pins V4-Pro to TP8 only, and that
   is behind `ci:atom_full` rather than the per-PR set. If TP4 is a target
   configuration it currently has zero end-to-end coverage.
3. **The production decode shapes are untested.** `test_custom_allreduce.py`
   hardcodes TP8 and sweeps `(2, 7168)` and `(128, 8192)`. The trace's four real
   decode shapes are `(1|2|4|8, 7168)` — only `M=2` overlaps, and `M=8`, the
   single hottest 1-stage shape (300 calls), is covered nowhere. `M=8` is also
   precisely the shape that changes kernel between TP4 and TP8.
4. **`test_fused_ar_rms.py` covers 7168 but is CI-skipped.** Its shape list
   includes `(17, 7168)` — a genuine DSv4 2-stage shape — but the file has been
   in the skip list since 2026-03-15 (#2280). For DSv4 work this is the most
   consequential of the six skips.
5. **mHC is in DSv4 and the AR fusion is not taken (§8.6).** Two sub-gaps:
   `test_fused_ar_mhc_post_only.py`'s `DEFAULT_SHAPES` are all hidden **4096**,
   not the production 7168 — the `mhc_pre` config tables do branch on
   `hidden_size >= 7168` ([mhc.py:192](../aiter/ops/mhc.py#L192)), so the kernel
   handles it, but the AR-fusion test never exercises that branch. And the
   fused path is not enabled in the serving stack at all, so its benefit on
   DSv4 is currently unmeasured.
6. **The custom-AR cap is 12% away from binding.** Prefill chunks are at 88% of
   the 64 MiB limit (§8.6 item 4). No test anywhere checks what happens at the
   boundary, and crossing it degrades silently to RCCL rather than failing.
7. **`should_custom_ar` assumes full connectivity unconditionally.** `fully_connected`
   is hardcoded `True` in Python ([custom_all_reduce.py:858](../aiter/dist/device_communicators/custom_all_reduce.py#L858));
   the topology probe is commented out. Fine on an all-XGMI node — which both TP4
   and TP8 subsets of a gfx950 node are — but it means a TP4 group placed across a
   partially-connected topology would take the `full_nvlink_` branch anyway.

### 8.6 Evidence from a real DSv4 TP4 trace

`op_tests/dump_data/aiter_kernels_union.json` is a kernel-attribution dump from a
DSv4 run under vLLM (`vllm 0.26.1rc1.dev1261`, `aiter v0.1.19`, torch 2.12,
ROCm 7.2.3, gfx950, 128 CU). It confirms most of the analysis above and settles
two open questions outright.

**Caveat first:** every one of the 41 kernels in the file has `is_aiter: true`, so
this is an aiter-attributed dump, not a full kernel trace. RCCL kernels are
excluded by construction — their absence here is not evidence they were unused.
Percentages below are shares of the 1.830 s of aiter kernel time in the dump, not
of total GPU time.

**Only two communication kernels appear, and they dominate the profile:**

| Kernel | Calls | Total | Share | Avg |
|---|---|---|---|---|
| `aiter::cross_device_reduce_1stage<bf16, 4, false>` | 813 | 507.4 ms | **27.7%** | 624 µs |
| `aiter::cross_device_reduce_2stage<bf16, 4, false>` | 258 | 421.8 ms | **23.0%** | 1635 µs |

Together **50.7%** of aiter kernel time, and the top two entries by a wide margin
(next is a CK GEMM at 14.1%). Both launch from `vllm::all_reduce`.

**What the template arguments confirm:** `<std::bfloat16_t, 4, false>` — bf16,
`ngpus=4`, `is_broadcast_reg_outptr=false`. This is a **TP4** run using the
vectorized (non-naive) kernels.

**The observed shapes match the predicted dispatch boundaries exactly:**

| Kernel | Observed token counts | Bytes | vs §8.3 prediction |
|---|---|---|---|
| 1-stage | 1, 2, 4, 8 | 14 – 112 KiB | all `< 160 KiB` → 1-stage ✓ (window is M ≤ 11) |
| 2-stage | 1024, 1031, 4096, 4100 | 14.0 – 56.1 MiB | all `≥ 160 KiB` → 2-stage ✓ |

Every activation is `(M, 7168)` bf16, confirming the hidden size. 105 of the 813
1-stage calls carry no attributed signature (`<unattributed>` launcher).

Four things follow that were not obvious before:

1. **M=8 is a real production shape that flips path between TP4 and TP8.** At
   112 KiB it is comfortably 1-stage at TP4 (`< 160 KiB`) but would be 2-stage at
   TP8 (`≥ 80 KiB`). It is also the *most frequent* 1-stage shape in the trace
   (300 of 708 attributed calls). Moving this workload to TP8 silently moves its
   hottest collective onto a different kernel — this is the single most important
   thing to measure before committing to a TP.
2. **The workload never lands near the crossover.** Decode is M ≤ 8, prefill is
   M ≥ 1024; nothing in between. The TP4 boundary at M=12 and the TP8 boundary at
   M=6 are both outside the sampled range — except that M=8 sits *just past* the
   TP8 boundary. So the crossover sweep from §8.7 matters mainly as a TP8
   readiness check, not as a TP4 tuning exercise.
3. **The 1-stage time is almost entirely barrier wait, not wire time.**
   Weighted-average 1-stage message is 68 KiB; per-rank traffic at `(N-1)·size`
   is 0.20 MiB against a 624 µs average, i.e. an implied **0.3 GB/s** — three
   orders of magnitude below XGMI. That 27.7% is exposed rank skew absorbed at
   `start_sync`, not communication cost. Optimizing the one-shot kernel will not
   recover it; reducing skew or shortening the serialized region will. By
   contrast 2-stage averages 33.6 MiB per call at 50.3 MiB per-rank traffic and
   1635 µs → **~32 GB/s**, which is plausibly bandwidth-bound real work. Treat
   the two halves of that 50.7% completely differently.
4. **Prefill sits at 88% of the custom-AR size cap.** The 4096- and 4100-token
   chunks are 56.0/56.1 MiB against the 64 MiB `_DEFAULT_CAR_MAX_SIZE`; the cap
   is hit at M=4681. A chunked-prefill chunk size above ~4681 tokens drops
   silently to RCCL. Anyone raising the chunk size should raise
   `AITER_CUSTOM_AR_MAX_SIZE` with it, or know they've left the custom path.

**mHC is confirmed in DSv4** — resolving the §8.2 unknown. `aiter::mhc_post_kernel`,
`mhc_pre_gemm_sqrsum_kernel`, `mhc_pre_big_fuse_rmsnorm_kernel` and
`mhc_fused_post_pre_gemm_sqrsum_kernel` are all present (~5.3% of kernel time
combined), with `mhc_post` input shape `[8192, 4, 7168]` — hc_mult=4 at hidden
7168. Two consequences:

- The AR+mhc_post fusion is **available but not being used**: `aiter::mhc_post`
  launches separately from `vllm::all_reduce`. `fused_allreduce_mhc_post`
  ([test_fused_ar_mhc_post_only.py](../op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py))
  would collapse a 48.0 ms mhc_post into the 929 ms of all-reduce. Given how much
  of the 1-stage time is barrier wait, fusing removes a whole
  launch-plus-barrier round per layer — likely worth more than the 2.6% the
  mhc_post kernel itself costs.
- §8.5 item 5 is now a live gap rather than a conditional one: production hidden
  is **7168**, and that test's `DEFAULT_SHAPES` are all 4096.

**DSv4 at TP4 uses tensor-parallel MoE, not expert parallel.** The `fused_moe_`
weights are `[384, 1536, 3584]` and `[384, 7168, 384]` (fp4-packed): all 384
experts replicated on every rank with the intermediate dim sharded 4 ways
(`2 * 3072 / 4 = 1536`, `3072 / 4 = 768`). No mori/all2all dispatch or combine
kernel appears anywhere in the dump, even though the container image has mori
v1.1.0 built in. So the entire §2.11 EP stack — and the EP4 tuning-config gap in
§8.5 item 1 — is **not on this configuration's path at all**. It would become
relevant only if DSv4 moves to expert parallelism.

**Also absent, consistent with §8.2 and §8.4:** no `all_gather`, no
`reduce_scatter`, no quick-all-reduce (`qr_*`), no `fused_allreduce_rmsnorm*` of
any kind, and no `fused_qknorm_allreduce`. The RMSNorm is riding in
`mhc_pre_big_fuse_rmsnorm` instead, which is why the AR+RMSNorm fusions do not
appear — worth noting, because it means the fused AR+RMSNorm+MXFP4 path that §8.3
analyzes in most detail is **not currently exercised by DSv4**, despite the
mxfp4 test being the best-matched test in the tree.

### 8.7 Suggested first pass

Reordered after §8.6: the trace says the traffic is plain `cross_device_reduce_*`
at four decode shapes and four prefill shapes, so cover *those* first. The
fused-AR tests, though better-matched on paper, exercise a path DSv4 is not
currently taking.

[bench_comm_allreduce.py](../op_tests/multigpu_tests/bench_comm_allreduce.py)
was written for exactly this: it drives `ca_comm.all_reduce` directly at
DSv4 shapes, reports which of the two kernels each row hit, and gives
algbw/busbw plus an RCCL baseline. Steps 1–3 below are one invocation of it.
Note TP2 can only reach the 1-stage kernel (the C++ dispatch hardcodes it at
`world_size == 2`), so `-t 4` is required to measure both.

```bash
# both kernels, both sides of the M=11/12 crossover, vs RCCL
HIP_VISIBLE_DEVICES=4,5,6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
    -t 2 4 -s 1,7168 8,7168 11,7168 12,7168 1024,7168 4096,7168
```

Measured on 4× gfx950 (bf16, barrier before each timed region):

| tp | tokens | kernel | KiB | custom µs | busbw GB/s | rccl µs | speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 8 | 1stage | 112 | 11.70 | 9.8 | 41.25 | 3.53 |
| 2 | 4096 | 1stage | 57344 | 1055 | 55.7 | 1117 | 1.06 |
| 4 | 8 | 1stage | 112 | 12.33 | 14.0 | 50.42 | 4.09 |
| 4 | 11 | 1stage | 154 | 13.28 | 17.8 | 54.07 | 4.07 |
| 4 | 12 | 2stage | 168 | **10.30** | 25.1 | 48.32 | 4.69 |
| 4 | 4096 | 2stage | 57344 | 548.8 | 160.5 | 593.7 | 1.08 |

Two results worth acting on:

- **The TP4 1-stage window looks too wide on gfx950.** M=12 (168 KiB, 2-stage)
  completes in 10.30 µs while M=11 (154 KiB, 1-stage) takes 13.28 µs — a *larger*
  message 22% faster, purely because the dispatch switched kernels. The 160 KiB
  threshold at [custom_all_reduce.cuh:3779](../csrc/include/custom_all_reduce.cuh#L3779)
  is holding shapes on one-shot past the point where two-shot wins. Lowering it
  would speed up the M∈[6,11] band, which includes real decode batches.
- **TP2 has no escape at prefill sizes.** Because `world_size == 2` hardcodes
  one-shot, a 56 MiB all-reduce at TP2 runs one-shot at 55.7 GB/s busbw versus
  160.5 GB/s for TP4's two-shot. One-shot reads the entire buffer from the peer;
  at 56 MiB that is the wrong algorithm and there is currently no way to opt out.

Both are dispatch-heuristic findings, not kernel bugs — and neither is visible
without a benchmark that reports which kernel ran.

### 8.8 QRInt4 (ROCm/aiter#4970) as an all-reduce candidate

[#4970](https://github.com/ROCm/aiter/pull/4970) adds `QRInt4`, a FlyDSL INT4
two-shot all-reduce (reduce-scatter + all-gather, 1152 B rank tile = 1024 B INT4
payload + 128 B group-16 E4M3 scales). Constraints from
`aiter/ops/flydsl/kernels/qr_int4.py`: bf16 in/out, contiguous, byte size a
multiple of 16, TP ∈ {2,4,8}, gfx942/gfx950, node-local, and it needs the **gloo**
group — it rejects an NCCL group because IPC handle exchange goes through
`broadcast_object_list`. It is a Python launcher only, not wired into the HIP
QuickReduce dispatch tables.

**It is usable as an all-reduce**: the API is `QRInt4(group=..., device=..., rank=...,
world_size=...)` then `.compile(inp, out)` once and `.allreduce(inp, out)`, which is
a drop-in shape match for `ca_comm.all_reduce`. `bench_comm_allreduce.py` picks it
up automatically when present (optional import; the bench degrades to
custom-vs-RCCL when #4970 is not merged).

**But it is lossy**, and that is the whole story. Measured on 4× gfx950 at DSv4
hidden 7168, `qr_int4 vs custom` (>1 means QRInt4 faster), against SQNR:

| tp | tokens | KiB | custom µs | qr_int4 µs | qr_int4 vs custom | custom SQNR | qr_int4 SQNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 14 | 10.59 | 12.23 | 0.87× | 54.9 dB | 19.3 dB |
| 2 | 8 | 112 | 7.44 | 12.48 | 0.60× | 54.9 dB | 19.2 dB |
| 2 | 4096 | 57344 | 1055 | **339.3** | **3.11×** | 54.9 dB | 19.1 dB |
| 4 | 8 | 112 | 12.32 | 12.27 | 1.00× | 55.4 dB | 19.2 dB |
| 4 | 12 | 168 | 10.29 | 12.32 | 0.84× | 55.5 dB | 19.2 dB |
| 4 | 1024 | 14336 | 147.2 | **57.6** | **2.55×** | 55.4 dB | 19.2 dB |
| 4 | 4096 | 57344 | 548.8 | **196.8** | **2.79×** | 55.4 dB | 19.2 dB |

Three conclusions:

1. **Nothing at decode.** For M ≤ 12 (≤168 KiB) QRInt4 is 0.60–1.19× — no
   consistent win, frequently slower. That is expected from §8.6 item 3: decode
   all-reduce is latency- and skew-bound, not bandwidth-bound, so a 4×-smaller
   wire buys nothing while the fixed two-shot barrier cost stays. Since DSv4
   decode is exactly the M ∈ {1,2,4,8} band, **QRInt4 does not help the 27.7%
   one-stage share at all.**
2. **2.5–3.1× at prefill**, which is genuinely bandwidth-bound. At TP4 M=4096 it
   reaches 447 GB/s of payload-equivalent bandwidth versus 160 GB/s for the exact
   two-shot.
3. **It gives TP2 the two-shot it otherwise cannot have.** Per §8.7, TP2 is
   hardcoded to one-shot and crawls at 55.7 GB/s on a 56 MiB prefill. QRInt4 at
   TP2 M=4096 does 173 GB/s — 3.11×. If TP2 prefill matters, this is the only
   path to it short of changing the C++ dispatch.

The cost is 19.2 dB SQNR versus the 55 dB bf16-rounding floor of the exact
kernels — i.e. ~11% relative error on the reduced activation, every layer, 61
times. Whether that is acceptable is a model-accuracy question, not a kernel
question, and it should be settled with an end-to-end eval rather than a
microbenchmark. Note the existing HIP QuickReduce takes the conservative view
already: its `_QR_MIN_SIZE` table (§2.9) disables every quantized codec for bf16
at TP8 and requires 16 MB+ at TP4, i.e. quantized AR is treated as a
large-message-only tool. These measurements agree with that framing.

```bash
# 1. The exact production shapes, at both TPs. M=8 is the one that changes
#    kernel between TP4 (1-stage) and TP8 (2-stage) -- that is the headline.
for tp in 4 8; do
  for m in 1 2 4 8; do
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
      python3 op_tests/multigpu_tests/test_allgather.py -t $tp -s $m,7168 -d bf16
  done
done

# 2. Prefill shapes, including the 64 MiB custom-AR cap at M=4681.
for tp in 4 8; do
  for m in 1024 4096 4681 4682; do
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
      python3 op_tests/multigpu_tests/test_allgather.py -t $tp -s $m,7168 -d bf16
  done
done

# 3. Force both kernels at M=8 to separate the dispatch choice from the
#    kernels. If forced-2stage at TP4 matches TP8's auto pick, the TP8
#    regression (if any) is the kernel; if not, it is the barrier/skew.
for s in 0 1; do
  for tp in 4 8; do
    AITER_AR_1STAGE=$s HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
      python3 op_tests/multigpu_tests/test_fused_ar_rms.py -t $tp -s 8,7168 --test fused
  done
done

# 4. The fused paths DSv4 does not use today, as forward-looking coverage.
python3 op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py            # TP 2/4/8
for tp in 4 8; do
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((tp-1))) \
    python3 op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py -t $tp -s 8,7168 --breakdown
done
```

Steps 1–2 use `test_allgather.py` because it takes an arbitrary `-s M,K` and
reports custom-vs-RCCL side by side; the equivalent for plain AR needs the `-t`
flag added to `test_custom_allreduce.py` first (§7, axis 1). Step 4's second
command needs `-s` to accept the DSv4 hidden size — its defaults are 4096
(§8.5 item 5).

All four are reachable today. Nothing here needs `mori`, since DSv4 at TP4 does
not use the EP stack (§8.6).

Beyond testing, the trace points at two things worth *measuring* rather than
covering: enabling `fused_allreduce_mhc_post` (a launch-plus-barrier saved per
layer, on top of the 48 ms of `mhc_post`), and attacking the 1-stage barrier wait
directly, since 27.7% of kernel time is currently rank skew rather than
communication.

---

## 9. Open questions

- Should write-mode (§2.5) be enabled on gfx950? It is gated to gfx942 and
  `world_size == 8`; nobody appears to have measured it on gfx950. The push/pull
  trade-off is arch-dependent and this is cheap to test.
- Is TP=6 actually used by any deployment? If yes it needs coverage; if no,
  consider dropping it from `_SUPPORTED_WORLD_SIZES` rather than carrying an
  untested naive-only path.
- Should the QR min-size table be re-tuned for gfx950? The entries read as
  MI300-era measurements, and several are effectively "disabled" sentinels.
- What is the fix for the six CI-skipped tests — install `mori`/`iris` on the
  runner, or arch/dependency-gate them so they self-skip cleanly instead of
  living in a hardcoded skip list?
- Does "test across gfx950 devices" mean the 8 GPUs of one node (assumed
  throughout §7) or genuinely different gfx950 SKUs (MI350X vs MI355X)? The
  latter is a CI runner-matrix change, not a local harness change.

DeepSeek-V4 specific (revised after the §8.6 trace):

- ~~Does V4-Pro enable mHC?~~ **Resolved: yes.** `mhc_post` / `mhc_pre_*` appear
  in the trace at hidden 7168, hc_mult 4. The live question is now why the
  AR+mhc_post fusion is not enabled, and what it is worth.
- **Why is 27.7% of kernel time spent spinning at the one-shot barrier?** 68 KiB
  average messages taking 624 µs is 0.3 GB/s implied — this is rank skew, not
  communication. Where does the skew come from, and is it structural (MoE
  routing imbalance across ranks) or fixable scheduling?
- **Is TP4 or TP8 the target?** ATOM pins TP8; the trace is TP4. `M=8` — the
  hottest decode shape — dispatches to different kernels under the two, so this
  is not a neutral choice.
- **Is the 1-stage mxfp4 gate's clause list deliberate?** `K <= 6144` and
  `K == 8192` are enumerated but 7168 is not, so DSv4 gets a narrower 1-stage
  window than models on either side of it (§8.3). Currently moot — DSv4 does not
  use that path — but it will matter if the fused mxfp4 AR is ever enabled.
- **What chunked-prefill chunk size is planned?** Above ~4681 tokens the
  all-reduce silently leaves custom AR for RCCL (§8.6 item 4). Today's 4096 is at
  88% of the cap.
- **Should a missing `ep{n}.json` warn?** Latent for now (§8.5 item 1), but today
  it produces default launch geometry with no log at default verbosity.
