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

After the all-gather, every rank holds the same bits. Each rank decodes every
chunk from the packets that the lap sends, and the owner of a chunk decodes
its own chunk from the packet that it sent.

### Accuracy limits

The kernels convert the bf16 payload to fp16 and do all arithmetic in packed
fp16, for every wire format.

* A magnitude above 65504 saturates to 65504. The `fp16` wire format is not 
  lossless for bf16 values outside the fp16 range.
* The group scale is an E4M3 byte with a largest magnitude of 480. A group
  with an extremum above 480 clips at about 480, and NaN and Inf become finite
  values. The ring requantizes the partial sum at each reduce-scatter hop, so a
  sum of several values near 300 can also clip.

## Inbox memory

Each kernel writes into an IPC inbox on each peer. The `inbox_memory` argument
selects the memory type of the inbox:

* `uncached`: A store cannot stay in a cache. The peers can see the data when
  the store retires.
* `finegrained`: The memory is cacheable. Each publish writes the payload back
  from L2 before the flag store. On PCIe, this mode is faster than `uncached`,
  because uncached peer writes become slow when the fanout is wide.
* `auto` (default): `uncached` on xGMI hosts, `finegrained` on PCIe hosts. The
  interconnect comes from the KFD topology or from `AITER_FLY_AR_LINK` if the KFD 
  topology cannot be read. At TP2, `auto` selects `uncached` on PCIe also, because 
  there is only one remote peer.

The cache policy for each mode is in `_INBOX_POLICY` in `allreduce_shared.py`.

## Dispatch

The FlyDSL kernels are not on by default. Set `AITER_FLY_AR=1` to use them.
Then `CustomAllreduce` uses the one-shot for small plain bf16 all-reduce
operations. `QuickAllReduce` uses the mesh and the ring when the INT4
quantization regime is set.

The communicator tries `QuickAllReduce` first, then `CustomAllreduce`. For
this reason, a payload goes to the mesh or the ring if it is in the mesh or
ring window. Other payloads go to `CustomAllreduce`, which uses the one-shot in
the one-shot window and `cross_device_reduce` or RCCL outside it.

`aiter/ops/flydsl/allreduce_policy.py` contains every tuned value, in two
levels:

1. Which algorithm runs a payload. `FAMILY_POLICIES` has one `FamilyPolicy`
   for each arch, link type and world size. A policy has one payload-size
   window for each algorithm (`AllReduceAlgorithm`): `oneshot`, `mesh` and
   `ring`. A window includes its minimum and excludes its maximum, and a
   maximum of `None` means no upper limit. A missing or empty window disables
   the algorithm. The windows do not overlap, and a payload that is in no
   window goes to `cross_device_reduce` or RCCL. `FamilyPolicy.select()` gives
   the algorithm for a payload size.

   These windows apply with the INT4 quantization regime. Without it, only
   the one-shot runs, in the `exact_oneshot` window if the policy has one.
   This window can be larger, because without the mesh the one-shot is the
   fastest kernel up to a larger payload.
2. Which configuration of that algorithm runs. Each algorithm has a tuning
   ladder for each arch, link type and world size: `ONESHOT_LADDER`,
   `MESH_ST_LADDER` and `RING_ST_LADDER`.


## Environment variables

### Enable the FlyDSL kernels

| Variable | Values | Effect |
|---|---|---|
| `AITER_FLY_AR` | `1`, or not set | `1` enables the FlyDSL kernels. All other values, and no value, disable them. |

The code reads `AITER_FLY_AR` when it creates `CustomAllreduce` and
`QuickAllReduce`. A change after this time has no effect.

`AITER_FLY_AR=1` does not enable a kernel on an unsupported system. The
kernels need gfx942 or gfx950, a world size of 2, 4 or 8, and the `flydsl`
package. These limits, and the other shared constants of the kernels, are in
the "Limits" section of `aiter/ops/flydsl/kernels/comm/allreduce_shared.py`.

### Set the interconnect type

| Variable | Values | Effect |
|---|---|---|
| `AITER_FLY_AR_LINK` | `xgmi`, `pcie`, or not set | The interconnect between the GPUs of the node. |

The interconnect selects the policy row, the tuning ladders and the inbox
memory. An MI350X (xGMI) and an MI350P (PCIe) both report `gfx950`, so the
arch does not tell them apart.

* If the variable is set, the code uses it as a backup for reading the KFD topology.
* If it is not set, the code reads the link types from
  `/sys/class/kfd/kfd/topology/nodes`. One xGMI link anywhere on the host makes
  the host xGMI.

### Change the payload-size windows

These variables replace the bounds of the windows in `FAMILY_POLICIES`. Each
value is a number of bytes. A minimum is included in the window, and a maximum
is not.

| Variable | Effect |
|---|---|
| `AITER_FLY_AR_ONESHOT_MIN_BYTES` | The minimum of the one-shot window. |
| `AITER_FLY_AR_ONESHOT_MAX_BYTES` | The maximum of the one-shot window. |
| `AITER_FLY_AR_MESH_MIN_BYTES` | The minimum of the mesh window. |
| `AITER_FLY_AR_MESH_MAX_BYTES` | The maximum of the mesh window. |
| `AITER_FLY_AR_RING_MIN_BYTES` | The minimum of the ring window. |
| `AITER_FLY_AR_RING_MAX_BYTES` | The maximum of the ring window. |
| `AITER_FLY_AR_EXACT_ONESHOT_MIN_BYTES` | The minimum of the one-shot window without quantization. |
| `AITER_FLY_AR_EXACT_ONESHOT_MAX_BYTES` | The maximum of the one-shot window without quantization. |

* If a variable is not set, is empty, or is `-1`, the code uses the value from
  the table.
* `none` or `inf` sets no upper limit. Only the `MAX` variables accept it.
* A maximum of `0` disables the algorithm. A minimum that is equal to the
  maximum also disables it.
* If an algorithm has no window in the table, a variable creates one. The
  other bound is then `0` for the minimum, or no upper limit for the maximum.
* The `ONESHOT` variables change only the one-shot window with quantization.
  The `EXACT_ONESHOT` variables change the window of the exact one-shot
  (`exact_oneshot`), which `CustomAllreduce` serves. Where a policy has no
  `exact_oneshot` window, the exact one-shot uses the `oneshot` window, and
  the `ONESHOT` variables change it. To disable the one-shot in all cases,
  set both maximums to `0`.
* If a value is not valid, the code logs a warning and uses the value from the
  table.
* If the `oneshot`, `mesh` and `ring` windows overlap after the change, the
  code logs a warning and does not use the FlyDSL kernels.
* The exact one-shot window can start and end anywhere, with one limit. With
  quantization on, the mesh and the ring take their payloads first. A payload
  of the exact window that is outside the `oneshot` window must be in the mesh
  or the ring window. If it is not, the code logs a warning and does not use the
  FlyDSL kernels. The mesh and the ring windows do not have to move with it.

The windows are independent. To move the boundary between the one-shot and
the mesh, set both `AITER_FLY_AR_ONESHOT_MAX_BYTES` and
`AITER_FLY_AR_MESH_MIN_BYTES`. To change only the payloads that the exact
one-shot serves, set `AITER_FLY_AR_EXACT_ONESHOT_MAX_BYTES`.

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

A test fails if the arch is not supported, or if a world size that you name
with `--tp` needs more GPUs than the node has. The default sweep skips the world
sizes that do not fit, and fails if none of them does; name a size that fits with
`--tp` instead.

The `verify` scenario drives the engines directly with `AITER_FLY_AR=0`, but it
also includes a production-dispatch check. That check drives payloads across the policy
window boundaries through `CudaCommunicator.all_reduce`. It verifies that
`_init_fly_backend` / `_init_fly_oneshot` built the slot, that dispatch routes
the in-window payloads to FlyDSL, and that the result is correct.

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
