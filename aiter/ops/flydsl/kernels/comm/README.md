# FlyDSL all-reduce kernels

This directory contains the FlyDSL all-reduce kernels for gfx942 and gfx950.
The kernels support bf16 payloads and world sizes (TP) of 2, 4 and 8 in one
node.

## Files

| File | Contents |
|---|---|
| `one_shot_allreduce.py` | The exact one-shot kernel and its tuning ladder. |
| `quick_allreduce_int4.py` | The quantized two-shot **mesh** kernel and its tuning ladder. |
| `quick_allreduce_int4_ring.py` | The quantized two-shot **ring** kernel and its tuning ladder. |
| `quick_allreduce_codec.py` | The INT4, INT6 and FP16 wire codecs. The mesh and the ring use them. |
| `allreduce_shared.py` | Tile geometry, the inbox cache policy, and the peer store and load primitives. All three kernels use them. |

The host classes are in `aiter/ops/flydsl/`:

| File | Contents |
|---|---|
| `one_shot_allreduce.py` | `OneShotAllReduce`, the host class for the one-shot kernel. |
| `quick_allreduce_int4.py` | `QuickAllReduceInt4`, the host class for the mesh and the ring. |
| `allreduce_policy.py` | The payload-size boundaries between the three kernels. |
| `allreduce_shared.py` | IPC inbox setup, link detection and other host helpers. |

## Kernels

`N` is the world size. `S` is the payload size in bytes.

| Kernel | Hops | Wire bytes for each rank | Accuracy | Use |
|---|---|---|---|---|
| One-shot | 1 | `(N-1)*S` | Exact: fp32 accumulate, one bf16 rounding | Small payloads (decode) |
| Mesh | 2 | `2(N-1)/N*S` | INT4 on the wire | Medium and large payloads |
| Ring | `2(N-1)` | `2(N-1)/N*S` | INT4 or INT6 on the wire | Large payloads, mostly on PCIe |

### One-shot

Each rank pushes its full payload into the inbox of each peer. Each rank then
reduces the `N` copies. The kernel has no flags. An inbox slot holds a NaN
sentinel until the data of a peer arrives. The reader polls its slot until no
16-bit word is the sentinel.

### Mesh

The mesh is a two-shot all-reduce: a reduce-scatter lap, then an all-gather
lap. In each lap, each rank pushes to all `N-1` peers. The wire format is INT4
with one E4M3 scale for each group of 16 values. FP16 is also available for
tests.

### Ring

The ring is also a two-shot all-reduce. The ranks make a cycle. In each step,
a rank pushes one contiguous run of data into the inbox of its successor only.
This is better on PCIe, where each GPU has one shared x16 uplink.

### Mesh and ring compared

| Item | Mesh | Ring |
|---|---|---|
| Quantizations in the reduce-scatter lap | 1 | `N-1`, one at each hop |
| Default reduce-scatter codec | INT4 | INT4, INT6 at TP8 |
| Writers for each inbox slot | `N-1` peers | 1, the ring predecessor |
| LDS staging (TP8, block 256) | Packets for all destinations, 9216 B with INT4 | Packet for one destination, 1664 B with INT6 |

The ring quantizes the running partial sum again at each hop. The maximum
absolute value of the partial sum increases with each hop. For this reason,
the ring needs INT6 on the reduce-scatter lap at TP8. The all-gather lap sends
the received bytes without a change, so it adds only one quantization.

## Inbox memory

Each kernel writes into an IPC inbox on each peer. The `inbox_memory` argument
selects the memory type of the inbox:

* `uncached`: A store cannot stay in a cache. The peers can see the data when
  the store retires.
* `finegrained`: The memory is cacheable. Each publish writes the payload back
  from L2 before the flag store. On PCIe, this mode is faster than `uncached`,
  because uncached peer writes become slow when the fanout is wide.
* `auto` (default): `uncached` on xGMI hosts, `finegrained` on PCIe hosts. At
  TP2, `auto` selects `uncached` on PCIe also, because there is only one
  remote peer.

The cache policy for each mode is in `_INBOX_POLICY` in `allreduce_shared.py`.

## Dispatch

The FlyDSL kernels are not on by default. Set `AITER_FLY_AR=1` to use them.
Then `CustomAllreduce` uses the one-shot for small plain bf16 all-reduce
operations. `QuickAllReduce` uses the mesh and the ring when the INT4
quantization regime is set.

The communicator tries `QuickAllReduce` first, then `CustomAllreduce`. For
this reason, a payload goes to the mesh or the ring if it is larger than the
one-shot boundary. Other payloads go to `CustomAllreduce`, which can then use the
one-shot.

The payload-size boundaries for each link type and world size are in
`FAMILY_POLICY` in `aiter/ops/flydsl/allreduce_policy.py`. On an arch that has
its own measurements, a row of `ARCH_FAMILY_POLICY` replaces the
`FAMILY_POLICY` row. gfx950 with xGMI has its own rows: the ring is never used
there, and at TP4 and TP8 the payloads between the one-shot and the mesh go to
the 2-stage `cross_device_reduce` kernel, which is faster than both there.

## Environment variables

### Enable the FlyDSL kernels

| Variable | Values | Effect |
|---|---|---|
| `AITER_FLY_AR` | `1`, or not set | `1` enables the FlyDSL kernels. All other values, and no value, disable them. |

The code reads `AITER_FLY_AR` when it creates `CustomAllreduce` and
`QuickAllReduce`. A change after this time has no effect.

`AITER_FLY_AR=1` does not enable a kernel on an unsupported system. The
kernels need gfx942 or gfx950, a world size of 2, 4 or 8, and the `flydsl`
package.

### Change the payload-size boundaries

These variables replace values from `FAMILY_POLICY`. Each value is a number of
bytes.

If a variable is not set, is empty, or is `-1`, the code uses the value from
the table. If the value is not an integer, the code logs a warning and uses
the value from the table.

| Variable | Effect |
|---|---|
| `AITER_FLY_AR_ONESHOT_MIN_BYTES` | The smallest payload for the one-shot (inclusive). `CustomAllreduce` does not send smaller payloads to the one-shot. The default is 0. |
| `AITER_FLY_AR_ONESHOT_MAX_BYTES` | The one-shot boundary. This variable sets two values. In `CustomAllreduce`, it is the largest payload for the one-shot (inclusive). In `QuickAllReduce`, it is the floor of the mesh (exclusive). |
| `AITER_FLY_AR_MESH_MAX_BYTES` | The largest payload for the mesh (inclusive). Larger payloads go to the ring. If this value is less than the one-shot boundary, the code uses the one-shot boundary. |

Without `AITER_FLY_AR_ONESHOT_MAX_BYTES`, the two one-shot values come from
different columns of `FAMILY_POLICY`:

* `oneshot_max_exact` is the one-shot ceiling in `CustomAllreduce`. Above it,
  `cross_device_reduce` or RCCL is faster.
* `oneshot_max` is the mesh floor in `QuickAllReduce`. Above it, the mesh is
  faster than the fastest exact kernel.

### Related aiter variables

These variables are not new, but they change where the FlyDSL kernels run.

| Variable | Effect on the FlyDSL kernels |
|---|---|
| `AITER_QUICK_REDUCE_QUANTIZATION` | Set to `INT4` to use the mesh and the ring. With other values, `QuickAllReduce` does not use them. |
| `AITER_CUSTOM_AR_MIN_SIZE` | `CustomAllreduce` uses the one-shot only for payloads larger than this value. |
| `AITER_CUSTOM_AR_MAX_SIZE` | `CustomAllreduce` uses the one-shot only for payloads up to this value. With `0`, the one-shot is not used. |

If the `AITER_CUSTOM_AR_*` window and the one-shot window do not overlap,
`CustomAllreduce` does not build the one-shot.

### Development variables

| Variable | Effect |
|---|---|
| `FLYDSL_EXTRA_SOURCE_DIRS` | Adds directories to the FlyDSL JIT cache key. Refer to [The JIT cache does not see changes in shared modules](#the-jit-cache-does-not-see-changes-in-shared-modules). |

## Tests and benchmarks

Two multi-GPU tests check and time the kernels. The `--scenario` option selects
what a test does:

- `verify` is the default. It checks correctness and times the FlyDSL kernels.
- `bench` compares the FlyDSL kernels with the stock kernels on the same
  payloads.

| Test | FlyDSL kernels | Stock kernels in `bench` |
|---|---|---|
| `op_tests/multigpu_tests/test_flydsl_one_shot_allreduce.py` | One-shot | `cdr` and `cdr_naive`, the `CustomAllreduce` kernels with `use_new` on and off, and `rccl` |
| `op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py` | Mesh and ring | `hip`, the HIP `QuickAllReduce` kernel at the INT4 regime |

```bash
python3 op_tests/multigpu_tests/test_flydsl_one_shot_allreduce.py --scenario bench --tp 4
python3 op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py --scenario bench \
    --tp 4 --report /tmp/int4.md
```

The one-shot bench has two comparisons:

- `flydsl vs prod` compares with the kernel that production runs when the
  FlyDSL path is off.
- `flydsl vs best` compares with the fastest stock kernel.

The `variant` columns show which kernel each candidate ran. For `cdr` this is
`ll`, `ll128_<block>`, `1stage` or `2stage`. The INT4 bench compares each
schedule with `hip`. A ratio above 1.0 means that the FlyDSL kernel is faster.

All candidates are timed with the same method, on the same input, in the same
process. Each candidate is a CUDA graph of 10 back-to-back calls. The test
times the replays in rounds and rotates the order of the candidates in each
round. `op_tests/multigpu_tests/flydsl_allreduce_bench.py` describes the method
and how it prevents bias in the comparison.

## Notes

### The JIT cache does not see changes in shared modules

The FlyDSL JIT cache key includes the source of the kernel function. It does
not include helper functions from other modules, for example
`allreduce_shared.py` or `quick_allreduce_codec.py`. If you change a helper,
FlyDSL can run an old binary from the cache.

To prevent this, set this variable at the repository root before you run a
test:

```bash
export FLYDSL_EXTRA_SOURCE_DIRS=$PWD/aiter/ops/flydsl/kernels
```

FlyDSL then includes all the `.py` files in this directory tree in the cache
key. The tree includes `kernels_common.py`, which the codec also uses.

### Each rank compiles its own binary

The rank is a compile-time value in all three kernels, so each process
compiles a different binary. The JIT symbol name contains an `_r<n>_` field
for the rank. With a compile-time rank, a kernel can keep the part of the data
for its own rank in registers. That part does not go through the inbox.

### Engine construction is a collective operation

A host class builds one engine for each different config in its ladder. Each
engine exchanges IPC handles with all ranks. All ranks must build the same
engines in the same order. If they do not, the processes deadlock.

### The mesh and the ring are persistent kernels

All workgroups of a persistent kernel must be on the GPU at the same time.
If the grid is too large, the kernel deadlocks. `clamp_grid_cap` limits the
grid with measured occupancy values. The host then uses the minimum grid of
all ranks. A grid that is too small is always safe, because each block then
does more tiles.

### Change `acquire` and `recv` together

A workgroup-scope acquire does not invalidate a cache. It is safe only if the
payload loads go past L1 and L2 (`sc0 sc1`).
