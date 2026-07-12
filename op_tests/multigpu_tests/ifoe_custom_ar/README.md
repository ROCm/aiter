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

Flags: `--tdm` selects a TDM (`tensor_load_to_lds`) reduce-scatter path.

## Results (gfx1250, ROCm 7.15, fp32, busbw = 2(N-1)/N · S / t)

| size | TP4 (1 node) | TP8 (2 nodes) |
|---:|---:|---:|
| 16 MB | 111 | 114 |
| 64 MB | — | 156 |
| 256 MB | 177 | 187 |
| 1 GB | **184** | **192** |

- **Cross-node TP8 ≈ single-node TP4** (192 vs 184 GB/s busbw at 1 GB): crossing
  the node boundary costs essentially nothing, because intra-node also rides IFOE.
- Reaches ~39% of the raw per-GPU fabric bandwidth from `ubench/07_ualoe`
  (~493 GB/s); the gap is the algorithm (two sync barriers + reduction), not the
  fabric.
- **TDM** (`--tdm`) does **not** help here (187 vs 192 at 1 GB): the all-reduce is
  sync/reduction-bound, not read-bandwidth-bound, so TDM's deeper-outstanding
  reads only add LDS-staging overhead. (TDM does help pure copy — see ubench 07.)
