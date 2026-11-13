# K3 A8W4 four-node snapshot and launch configuration

This snapshot continues AITER PR [#7297](https://github.com/ROCm/aiter/pull/7297),
the historical A8W4 improvements and the four-node bring-up changes. It adds
the relevant nine-file change from PR
[#7542](https://github.com/ROCm/aiter/pull/7542), head
`5173e7ac5efee2ebf293753b30cefd136b7b6510`, and prepares the separate-buffer
Stage2 SDMA path for physical four-node EP32.

The branch is `dev/megamoe-k3-a8w4-ep32-sdma-20261113`, created from
`57f7262c69ad1a32666675fd4e56bd8977dc4f82` with the pending four-node changes
preserved. The M32 version with the dynamic Stage2 scratch fix completed the
physical EP32 K3 numerical sweep. This snapshot restores historical Stage1
M16 and retains that scratch fix. **The restored M16 version passed physical
EP16 validation; it has not yet been rerun on EP32.** The earlier EP32
TPR4096 timing anomaly remains unresolved.

## Changes included

- C1 separates local payload buffers from bounded NIC windows. Each rail
  window has its own handle and registered size; local windows use
  `register_window(..., rdma_register=False)`. Kernels read a device table
  containing each region's pointer, handle, offset and parity stride.
- C2/C3 extend dispatch, source planes, remote partials, readiness and return
  credits to four logical nodes. Peer arrays enumerate nodes in ascending
  order excluding self; source accumulator planes retain absolute node IDs.
  Receive-capacity guards and the coordinated overflow path are retained.
- C5 runs the full four-logical-node protocol on two physical eight-GPU
  machines. CROSSNODE/WORLD connections address the required global GPU
  rank directly; same-host logical peers use LSA copies. LSA peer bases
  restrict GEMM/scatter operations to the appropriate four-GPU group.
- A8W4 retains the historical record quantization/packing work and the
  historical Stage1 M16 sub-steps for BM128, reusing the B fragment while
  limiting the live FP8 A fragment. The M32 addition from the EP16 comparison
  was reverted because it did not establish an overall gain over the historical
  optimized version. Historical epilogue/quant lane rematerialization and
  Stage2 FP8 BM128 M16 streaming are retained, together with the EP32 64-bit
  input-address path. The restored JIT helper and affected Stage1 kernel names
  distinguish this schedule from previously compiled M32 handles. This rollback
  has not yet been revalidated on GPUs in the EP32 SDMA configuration.
  Existing weight-layout, graph/replay, empty-input and tensor-lifetime handling
  remains part of this snapshot.
- PR #7542 adds K3 A8W4 small-op configurations for 56, 28 and 14 local
  experts, masked EP MXMOE support, and tile-local GEMM1 buffer stores for
  large addresses. Its tuning drivers and EP tests are included. The
  unrelated Triton changes from a base/head comparison were excluded.
- Separate-buffer SDMA now requests one queue from the **local** DevComm.
  A trailing device-table metadata row carries that DevComm pointer without
  changing any region's four-word ABI. Stage2 uses the local arrival window
  for engine-side waits, and sizes inverse maps and chunk counters for the
  actual source-node count.
- Stage2 allocates disjoint scratch spans for RAIL requests, the SDMA epoch,
  completion counts and chunk counters from the topology and chunk geometry.
  This retains the correctness fix used by the completed EP32 run: publishing
  an i64 RAIL request must not overwrite SDMA's running signal epoch. Scratch
  offsets participate in the compiled-handle cache key, with a host regression
  covering two through eight source nodes and multiple QP/chunk counts.
- SDMA is selected only for eight-GPU logical nodes and the remote-first
  Stage2 path. Virtual four-GPU nodes stay on xGMI. Token-ready remains
  limited to the legacy two-node layout; it is disabled for separate buffers
  and four-node operation.

## 16 rank 4x4 versus 32 rank 4x8

Here, **16 rank 4x4 means four logical nodes on two physical eight-GPU
machines**, not four physical four-GPU machines. `torchrun --nnodes` counts
physical launch hosts. `--logical-gpus-per-node` controls the fused
operator's logical grouping.

| Item | 16 rank, logical 4x4 | 32 rank, physical 4x8 |
|---|---|---|
| Physical hosts | 05 and 06 | 05, 06, 07 and 08 |
| torchrun hosts / workers per host | 2 / 8 | 4 / 8 |
| `world_size` | 16 | 32 |
| Fused `gpus_per_node` | 4 | 8 |
| Logical source nodes `world_size / gpus_per_node` | 4 | 4 |
| Logical rank groups | 0–3, 4–7, 8–11, 12–15 | 0–7, 8–15, 16–23, 24–31 |
| Physical CCO LSA width | 8 | 8 |
| `lsa_peer_base` | 0 or 4 on each physical host | 0 |
| `gda_peer_stride` / `gda_peer_lane` | 4 / `rank % 4` | 1 / 0 |
| Rail DevComm connection | `GDA_CONNECTION_CROSSNODE` | `GDA_CONNECTION_RAIL` |
| GDA device team / peer argument | `Team.WORLD` / `peer_node * 4 + rank % 4` | `Team.GDA` / peer node ID |
| Logical peers on the same physical host | LSA payload/value copies | No distinct same-host logical node |
| NIC selection | `ionic_0` | `ionic_0` through `ionic_7` |
| K3 experts per rank, with 896 total experts | 56 | 28 |
| Fused intra-node Stage2 transport | xGMI | xGMI or SDMA |

Both use `_initialize_separate_cco_runtime()` and `CcoBufferSet`. Each rank
initializes or borrows **two distinct full-world communicators**, one for
local windows and one for rail windows. These are not four separately
initialized four-/eight-rank communicators. Rank 0 broadcasts a unique ID
for each owned communicator. All ranks allocate and register windows in the
same plan order, upload the device binding table, synchronize and perform
the rail communicator barrier before use. The numerical runner pools these
communicators and their allocations across cases.

The runtime detects the physical LSA width from the local DevComm. If that
width differs from the logical width, it sets the LSA base and the
CROSSNODE/WORLD mapping; otherwise it keeps the direct RAIL/GDA mapping.
For example, rank 1 sending to logical node 1 in virtual 4x4 addresses global
rank 5 through LSA, while sending to logical node 2 addresses global rank 9
through CROSSNODE. In physical 4x8, rank 1 addresses peer nodes 1 and 2
through the aligned RAIL connections to ranks 9 and 17.

Both use four source planes and three remote-peer slices, but buffer sizes
and kernel specializations differ. Source capacity is `world_size * TPR`;
the expert shard is 56 versus 28 experts for K3. The device table contains
per-window offsets and two parity strides, so offsets from one topology
cannot be reused in the other. The GEMM2/combine specialization includes
topology, buffer bindings, capacities and transport configuration.

Concurrent single-kernel Stage1/Stage2 operation reserves separate
eight-context ranges: Stage1 base 0, Stage2 base 8, 16 GDA contexts total.
The two-kernel path uses eight contexts and Stage2 base 0. This is a launch
mode difference shared by both topologies, rather than a fixed EP16/EP32
difference. `MORI_NUM_QP_PER_PE=2` is a separate MORI EP configuration;
it is not the fused kernel's GDA context count.

The MORI small-op/reference configuration still describes the **physical
eight-GPU hosts**: `gpu_per_node=8`, world size 16 or 32, with 56 or 28 K3
experts per rank. Only the fused operator is logically regrouped to 4x4
in the C5 run. Do not set the physical MORI `GPU_PER_NODE` environment to 4
to emulate C5.

## Runtime dependency

Use AITER's pinned `flydsl==0.5.0`. MORI upstream main was checked at
`b80cbe4ac876e63c2c4260d1bd88ab12edce68e7`, which includes PR
[#944](https://github.com/ROCm/mori/pull/944), merge
`cd9c4f11a4c822668a6f3e8806a090be5c476461`.

The installed task runtime additionally contains the local-window API and
DevComm CQ/QP ownership/lifecycle fixes. The exact source delta is supplied
in [the MORI dependency patch](patches/mori_local_windows_lifecycle_b80cbe4.patch).
Apply it to that upstream commit and rebuild MORI's CCO extension and
shared libraries together with `BUILD_CCO_SDMA=ON` and `MORI_GPU_ARCHS=gfx950`.
This AITER branch does not itself modify or install the MORI repository.

On 07/08, the test container `ziming_megamoe_20261112` now uses the new MORI
through both its editable package installation and `/workspace/env.sh`.
Fresh imports loaded the expected candidate library. The 05/06 test
launches use the same candidate through explicit Python/library paths.
All four candidates have matching library hashes:

| Library | SHA256 |
|---|---|
| `libmori_cco.so` | `0d504ab61646d6ddebc106476fcfdd2f25cbf86095dab1151a7d252614f39aa8` |
| `libmori_application.so` | `5754812f4bd76ae2eb6a4183bd3c854befc6d134aa1311af40a0e4c3b0bcd268` |

Prepend the candidate to both `PYTHONPATH` and `LD_LIBRARY_PATH`. Changing
only `PYTHONPATH` can import the new Python module while the loader resolves
an old `libmori_cco.so` from another directory. Do not force even SDMA engine
IDs; unset `MORI_SDMA_PEER_RANK_ENGINE`.

## Launch examples for 05/06/07/08

These commands are reference configurations. Run inside the test container on each participating
host, with the branch checked out at `AITER_ROOT` and the matching MORI
runtime at `MORI_ROOT`. Keep logs/results/compiler traces on remote storage.

Common environment:

```bash
export AITER_ROOT=/workspace/ep32_4node_20261113/aiter-k3-pr7542-sdma
export MORI_ROOT=/workspace/ep32_4node_20261113/mori-c5-lifecycle
export PATH=/workspace/venv/bin:$PATH
export PYTHONPATH="$AITER_ROOT:$MORI_ROOT/python"
export LD_LIBRARY_PATH="$MORI_ROOT/python/mori:${LD_LIBRARY_PATH:-}"
export GLOO_SOCKET_IFNAME=eth1 MORI_SOCKET_IFNAME=eth1 NCCL_SOCKET_IFNAME=eth1
export MORI_DEVICE_NIC=ionic MORI_IB_GID_INDEX=1 MORI_NUM_QP_PER_PE=2
export MORI_ENABLE_RAIL_ONLY=1 MORI_DISABLE_IONIC_CCQE=1
export MORI_CCO_FABRIC_DISABLE=1 MORI_EP_DISABLE_RDMA_ATOMICS=1
export MORI_IB_ENABLE_RELAXED_ORDERING=0 MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export GPU_PER_NODE=8 OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_DEBUG=WARN
export AMD_SERIALIZE_KERNEL=0 FLYDSL_RUNTIME_ENABLE_CACHE=1
export AITER_MOE_EXPERT_BALANCE=true MORI_SHMEM_HEAP_SIZE=4G
unset MORI_SDMA_PEER_RANK_ENGINE MORI_CCO_BC MORI_VPOD_ID
unset AITER_CONFIG_MEGAMOE_INTERNODE
cd "$AITER_ROOT"
```

The 4G SHMEM heap is the task's representative-case setting, separate from
the fused CCO local/rail allocations; reassess it for larger/ragged cases.
Use a distinct unused port and tag for each run. For a future SDMA run,
setting `MORI_ENABLE_SDMA=1` alone does not prove SDMA was selected:
the fused operator must report `_stage2_sdma=True`, a nonzero local SDMA
queue count, and the SDMA kernel specialization. The default rule selects
SDMA at TPR >= 2048; lower buckets can use xGMI. An `intra_comm=xgmi` tuned
row takes precedence and can be used for an explicit xGMI control.

16 rank, virtual 4x4: start physical rank 0 on 05 and rank 1 on 06.
There are **two** torchrun launch hosts, each with eight workers.

```bash
export MORI_RDMA_DEVICES=ionic_0 MORI_ENABLE_SDMA=0
# Set NODE_RANK=0 on 05, NODE_RANK=1 on 06.
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr=10.15.20.6 --master_port=30951 \
  --max_restarts=0 op_tests/multigpu_tests/test_megamoe_internode.py \
  --network kimi_k3 --logical-gpus-per-node 4 --cco-buffer-layout separate \
  --fused-quant a8w4 --smallop-quant a8w4 --comm-quant fp8 \
  --situ-beta 4 --situ-linear-beta 25 --tpr-list 2048 --fixtures random,eplb \
  --paths fused,smallop --timing total --warmup 10 --iters 40 \
  --out-dir /workspace/ep32_4node_20261113/results --tag c5_k3_documented_example
```

32 rank, physical 4x8: start physical ranks 0, 1, 2 and 3 on 05, 06, 07 and
08 respectively. There are **four** torchrun launch hosts, each with eight
workers. This is the physical EP32 SDMA configuration.

```bash
export MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export MORI_ENABLE_SDMA=1
# Set NODE_RANK=0/1/2/3 on 05/06/07/08 respectively.
python -m torch.distributed.run --nnodes=4 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr=10.15.20.6 --master_port=30952 \
  --max_restarts=0 op_tests/multigpu_tests/test_megamoe_internode.py \
  --network kimi_k3 --logical-gpus-per-node 8 --cco-buffer-layout separate \
  --fused-quant a8w4 --smallop-quant a8w4 --comm-quant fp8 \
  --situ-beta 4 --situ-linear-beta 25 --tpr-list 2048 --fixtures random,eplb \
  --paths fused,smallop --timing total --warmup 10 --iters 40 \
  --out-dir /workspace/ep32_4node_20261113/results --tag ep32_k3_sdma_documented_example
```

Both examples use the in-repository numerical driver and K3 A8W4 tune table.
The task's additional runner preserves independent native small-op MLP
checks and audits the actual local SDMA queues. That task wrapper and the
remote MORI installation scripts are outside this AITER commit. Confirm the
actual #7542 kernel selection in logs; table inclusion is not a performance
result. The physical data-network mapping is:

| Host | SSH address | torchrun rank in EP32 |
|---|---|---:|
| 05 | 168.144.57.63 | 0 |
| 06 | 168.144.62.62 | 1 |
| 07 | 201.79.28.54 | 2 |
| 08 | 201.79.23.229 | 3 |

The rendezvous address is `10.15.20.6`, reachable through `eth1`.
The older 46/47/50 probe launchers use another network and container setup;
their hard-coded interfaces/addresses are not the 05/06/07/08 launch config.

## Verification status

The pre-existing xGMI snapshots completed 40 true EP32 A8W4 cases with
640 fused checks, and 40 virtual 4x4 cases with 640 fused checks plus
280 independent native small-op checks. These runs predate #7542 and the
new EP32 SDMA wiring. They establish the underlying bring-up status, not
validation of this new commit.

For physical EP16 on 05/06, the 16 cases that actually selected Stage2 SDMA
were TPR 2048/4096/8192/16384, random/EPLB, K3 and DS4.1. Current numerical
checks passed. Against the frozen stacked EP16 implementation, K3 p50
changed by -1.72% to +0.37%, and DS4.1 by -2.71% to +1.61%. Each sample
is a complete-forward GPU-event maximum over ranks; comparisons used
frozen/current/current/frozen execution. Both sides used SDMA, so these are
recovery checks rather than SDMA-on/off gains. No odd engine ID was recorded.

The MORI lifecycle candidate on 05/06 completed 32 create/destroy rounds
and 1536 exact communication checks while preserving a resident DevComm;
the before-fix control failed CQ/QP creation in round 4. This probe used
SDMA off and does not validate the new EP32 SDMA operator.

The M32 version plus the dynamic scratch fix completed the physical 4x8
EP32 run `ep32_k3_sdma_bounded_full20_r4`: 20 K3 cases at TPR32–16384,
random/EPLB, with 320 fused and 140 independent Small-op checks passing.
Three-seed repeats at TPR4096/16384 passed another 192 fused and 84 Small-op
checks. The TPR4096 timing anomaly repeated and remains unresolved. These
runs used the #7542 source above; only six main cases hit tuned Small-op
rows, while the other fourteen used its defaults.

After restoring Stage1 M16, nine single-GPU GEMM1 cases (BM32/64/128 with
three seeds) matched the preserved historical M16 payload and scales byte
for byte. Physical 2x8 EP16 run `ep16_k3_m16_pr7542_r1` passed all 20 K3
cases, with 320 fused and 140 independent Small-op checks. Three additional
TPR16384/EPLB seeds passed another 48 fused and 21 Small-op checks. This
validates M16 on EP16, not the changed schedule on EP32. Against the EP32
run's source fingerprints, the runtime code changes are confined to the
GEMM1 schedule and its Stage1 cache identity; Stage2 and communication match.

The restored snapshot passed local formatting, repository Ruff, Python
parsing, whitespace checks, the CK dependency check and 117 CPU
plugin-interface pre-checks. Both GPU configurations used FlyDSL 0.5.0 and
the matching MORI libraries above. The benchmark wrappers and full results
remain outside this commit. EP32 validation of the restored M16 schedule
is still pending.
