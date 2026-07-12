# IFOE cross-node custom all-reduce (gfx1250)

A proof-of-concept + benchmark that runs aiter's custom all-reduce **across nodes**
over the UALink-over-Ethernet (IFOE) fabric, by sharing peer buffers with HIP
**fabric handles** instead of IPC handles.

## Why this works with no kernel change

On gfx1250 the fabric (IFOE) makes a remote GPU's memory addressable exactly like
a local peer's. Two primitives the all-reduce relies on both work cross-node:

- **peer data read/write** — validated in `ubench/07_ualoe` (~500 GB/s per GPU,
  ~2 TB/s aggregate across 4 GPUs).
- **peer atomic signals** — the `start_sync`/`end_sync` barrier (system-scope
  atomic store into a peer's flag + device-scope spin-load on own flag) completes
  cross-node.

So `cross_device_reduce_2stage` / `start_sync` / `end_sync` are used **verbatim**;
only the buffer-sharing mechanism changes:

| | intra-node (IPC) | cross-node (this) |
|---|---|---|
| export | `hipIpcGetMemHandle` | `hipMemExportToShareableHandle` (fabric) |
| import | `hipIpcOpenMemHandle` | `hipMemImportFromShareableHandle` (fabric) |
| buffer alloc | any `hipMalloc` | VMM (`hipMemCreate` + map, fabric-exportable) |
| handle exchange | 1 node | any node (host learns the peer IP) |

`ualoe_allreduce.cpp` is standalone (one process per rank; rank 0 is a TCP
coordinator that all-gathers + broadcasts the 64-byte fabric handles). It keeps
the aiter kernel; the harness replaces the torch-distributed IPC path.

## Build & run

```bash
# single-node TP4 (builds + runs 4 ranks, asserts correctness, prints busbw)
python3 test_ifoe_allreduce.py --mb 256

# manual, e.g. TP4:
./ualoe_allreduce --rank 0 --world 4 --gpu 0 --coord 127.0.0.1 --port 55570 --mb 256
# ... ranks 1..3 similarly (--gpu 1..3)

# cross-node TP8 (rank 0 on node A is the coordinator):
#   node A (ip A):   ranks 0..3, --gpu 0..3  --coord <A> --world 8
#   node B:          ranks 4..7, --gpu 0..3  --coord <A> --world 8
for r in 0 1 2 3; do ./ualoe_allreduce --rank $r --world 8 --gpu $r      --coord <A> --port 55571 --mb 256 & done   # node A
for r in 4 5 6 7; do ./ualoe_allreduce --rank $r --world 8 --gpu $((r-4)) --coord <A> --port 55571 --mb 256 & done   # node B
```

The default kernel is `allreduce2_opt` — an MLP-unrolled reduce-scatter (U=8
float4 in flight per thread) + per-peer contiguous allgather, which saturates the
fabric far better than the naive upstream loop.

Flags:
- `--naive` uses the verbatim upstream 2-stage kernel; `--tdm` uses a TDM
  (`tensor_load_to_lds`) reduce-scatter path; `--prof` prints per-phase cycle
  breakdown.
- `--unroll N` sets the MLP unroll factor (default 8); `--blocks N` / `--threads N`
  override launch geometry (defaults: 512 threads, blocks auto-capped at 208).

## Results (gfx1250, ROCm 7.15, fp32, busbw = 2(N-1)/N · S / t)

| size | TP4 (1 node) | TP8 (2 nodes) |
|---:|---:|---:|
| 16 MB | 83 | 78 |
| 64 MB | 200 | 197 |
| 256 MB | 303 | 278 |
| 1 GB | **360** | **341** |

- **Cross-node TP8 ≈ single-node TP4** (341 vs 360 GB/s busbw at 1 GB): crossing
  the node boundary costs essentially nothing — intra-node also rides IFOE, and
  a world=2 test measures the same ceiling on-node and across nodes.

### This is the fabric's *bidirectional* ceiling, not a kernel limit

Profiling (`--prof`, TP8 1 GB) shows the sync barriers cost only **~7%**
(start_sync ~0%, end_sync ~6%); reduce-scatter and allgather (the fabric traffic)
are **~94%**. So the kernel is memory-bound, and the relevant hardware limit is
**not** the ~493 GB/s *unidirectional* read peak from `ubench/07_ualoe`.

An all-reduce is inherently **bidirectional**: every GPU reads its peers (inbound)
while its own buffers are read by peers (outbound) — 2(N-1)/N bytes each way. The
measured per-GPU *bidirectional* wall (a `world=2` all-reduce, which is a pure
simultaneous read+write test) is:

| bidir test (1 GB) | busbw/direction |
|---|---:|
| world=2 cross-node | ~350 |
| world=2 intra-node | ~359 |

So busbw for any lossless fp32 all-reduce on this fabric tops out around
**~350–360 GB/s**, and the tuned kernel is already there. The 493 figure only
applies to one-directional traffic; the shared per-GPU uplink drops to ~355/dir
under simultaneous read+write. Reaching higher requires reducing bytes on the
wire (bf16/fp8 transfer) or in-network reduction, not kernel tuning.

### Tuning notes

- **MLP unroll** took TP8 1 GB from 299 → 341 (+14%): issuing many outstanding
  float4 loads per thread (the ubench-02 HBM lesson) is what saturates the fabric.
  U=8 is best for TP8 (7 peers → VGPR pressure caps higher unroll); U=16 helps
  world=2 (1 peer).
- **Block count**: `kMaxBlocks` was raised from 80 (under-occupies) to 304; ~208
  blocks is the sweet spot (too many inflates the per-block barrier cost).
- **TDM** (`--tdm`) does **not** help: the collective is fabric-bandwidth-bound,
  not read-latency-bound, so LDS staging only adds overhead. (TDM does help pure
  copy — see ubench 07.)
