# MegaMoE 4-node protocol bring-up on two MI355 hosts

This file retains chronological bring-up results from earlier snapshots.
For the current full-forward topology, MORI dependency and launch settings,
see [the 2026-11-13 K3 four-node snapshot](megamoe_k3_a8w4_ep32_sdma_20261113.md).

Date: 2026-11-12. This records the C0/C5 and C1 prerequisite work in
`/home/zihuang/trace/doc/requirements.md`. The tests below exercise the CCO
transport overlay. They do not yet run the fused Stage1 or Stage2 kernels with
four logical nodes.

## Topology and runtime

- Physical hosts: `mi355-gpu-47` (`ziming_huang_work`, data IP `10.2.80.18`)
  and `mi355-gpu-51` (`hzm_work`, data IP `10.2.80.22`). Host-major world ranks
  are 0–7 and 8–15. Each host has 8 MI355 GPUs.
- Logical topology: four nodes with four GPUs each. Logical nodes 0/1 live on
  47; nodes 2/3 live on 51. Physical CCO LSA teams remain 8 GPUs wide and GDA
  rail peers remain the aligned rank on the other host.
- Both containers use PyTorch 2.9.1 with ROCm 7.2.0. The installed MORI shared
  libraries and Python CCO extension have matching SHA-256 hashes on 47 and 51.
  47 lacked an importable FlyDSL package; the FlyDSL 0.3.1 runtime from 51 was
  unpacked in the isolated task directory on 47. No system package was changed.
- Both launchers set the same Gloo/MORI/NCCL data interface, the eight Ionic
  rail selection, and `MORI_ENABLE_RAIL_ONLY=1`. The first attempt used a
  mismatched rail-only setting and failed at CCO initialization; this is fixed.

`VirtualNodeTopology` plans 48 directed logical sends: 16 same-host LSA,
16 direct cross-host rail, and 16 cross-host rail followed by one LSA relay.
The installed MORI FlyDSL binding takes **world-rank peer IDs** for rail posts.
It does not expose the newer `Team.GDA` API, so the standalone probe uses the
aligned physical world rank. Ready and credit words are generation tagged.
Both hosts also have an isolated `venv_mori895` whose MORI binding exposes
`Team`. At the initial transport bring-up their FlyDSL packages lacked
`fx.generic_load` and `fx.generic_store` used by the current AITER branch.
The later pinned-runtime bring-up below resolves the FlyDSL API prerequisite;
the MORI PR895 interface must still be combined with local-only registration
before the current fused kernels can run with the new buffers.

## Measured transport checks

All rows ran on the same 47+51 pair. Both node exit codes were zero for the
passing rows. Logs and `.rc` files remain under
`/home/hzm/virtual4_transport_20261112/` on each host; no trace or tensor was
copied to the local machine.

| Run tag | GDR | Registered windows per rank | Probe result |
|---|---:|---:|---|
| `v4_mr_4m_b` | 0 | 4+4+4 MiB | 48/48 rail writes to window tails |
| `v4_mr_4m_gdr1` | 1 | 4+4+4 MiB | 48/48 rail writes to window tails |
| `v4_transport_4k_b` | 0 | one small window | 48/48 directed payloads, ready/credit passed |
| `v4_transport_256k_3r` | 0 | two small windows | 144/144 directed payloads; 256 KiB, 3 generations, 2 GDA contexts, rank 11 delayed 250 ms per generation |
| `v4_transport_256k_gdr1` | 1 | two small windows | 96/96 directed payloads; 256 KiB, 2 generations, 2 GDA contexts, rank 11 delayed 250 ms per generation |
| `v4_mr_960_960` | 1 | 960+960 MiB | 32/32 high-address rail writes |
| `v4_mr_640x3` | 1 | 640+640+640 MiB | 48/48 high-address rail writes |
| `v4_c1_control_640x3` | 1 | 640+640+640 MiB | 48/48 high-address rail writes (repeat control) |
| `v4_c1_k3_832x2` | 1 | 832+832 MiB | 32/32 high-address rail writes |
| `v4_c1_ds_704x3` | 1 | 704+704+704 MiB | 48/48 high-address rail writes |

The `960+960+64`, `960+960+128`, `960+960+256`, and `960+960+704` MiB GDR
configurations all failed while allocating their third window on all 16 ranks:
`ccoMemAlloc: hipMemSetAccess failed after retries: 1 (invalid argument)`.
They did not reach registration or GPU kernels. Because 640+640+640 MiB works,
this is an allocation/mapping boundary affected by window sizes, not a blanket
three-window prohibition. The exact limit is not established.

Two capacity-shaped runs also failed in `ccoMemAlloc` at the second window's
`hipMemSetAccess`, before registration: K3 `576+832` MiB and DSv4.1
`672+704+512` MiB. Equal-size padded controls (`832+832` MiB and
`704+704+704` MiB) passed on both hosts. These observations do not establish
that equal-size windows are required; the cause of the allocation difference
is still unknown. The passing controls validate transport writes only, not
fused-kernel buffer partitioning.

## Continued session: allocation size versus registered size

The resumed session kept the same 47+51 pair and GDR=1. Reversing K3's
allocation order to `832+576 MiB` still failed on both hosts (exit code 1),
in the second allocation's `hipMemSetAccess`. Ordering alone is not a
working workaround for the observed unequal-size case.

The MR probe now accepts separate physical allocation and MR sizes. It also
writes the tail of each padded allocation through all seven same-host LSA
peers, beyond the NIC-registered prefix. New runs are in the isolated remote
directory `/home/hzm/virtual4_alloc_20261112/`; no installed package or
unrelated source was replaced.

| Final run tag | Allocation sizes per rank | Registered MR sizes per rank | Rail tail writes | LSA writes beyond MR |
|---|---|---|---:|---:|
| `k3_padded_alloc_lsa` | 832+832 MiB | 576+832 MiB | 32/32 | 112/112 |
| `ds_padded_alloc_lsa` | 704+704+704 MiB | 704+704+512 MiB | 48/48 | 112/112 |

Both hosts returned zero for both final runs. The preceding rail-only runs
`k3_actual_mr_padded_alloc` and `ds_actual_mr_padded_alloc` also passed, with
32/32 and 48/48 writes respectively. Padding thus need not enlarge the NIC
MR: the tested combinations use 256 MiB (K3) and 192 MiB (DSv4.1) extra
allocation per rank while retaining capacity-sized registered prefixes.
The LSA checks confirm that the allocation tail remains mapped to same-host
peers even when it lies outside the MR. These checks cover individual i64
tail writes, not full payload transfer, GEMM numerics, graph replay, large
local-only allocations, or arbitrary allocation-size combinations.

Reproduce inside each container, starting node 0 and then node 1 promptly:

```bash
cd /home/hzm/virtual4_alloc_20261112
# Use NODE_RANK=0 on 47, NODE_RANK=1 on 51; choose free matching ports.
bash launch_megamoe_virtual4_mr.sh NODE_RANK 29675 k3_padded_alloc_lsa 576,832 1 832,832
bash launch_megamoe_virtual4_mr.sh NODE_RANK 29676 ds_padded_alloc_lsa 704,704,512 1 704,704,704
```

The final optional launcher argument is the allocation-size list. Omitting
it retains the original behavior of allocating exactly the registered sizes.
Invalid list lengths or undersized allocations are rejected before process
group initialization. Local Python parsing, shell syntax, whitespace, and
Black 26.5.1 checks passed for the changed probe/launcher. This is not a claim
that the whole pending PR has passed CI static checks.

## MORI EP32 registration audit

MORI advertises MI355X EP32-V1-LL, but its MI355X EP default is `shmem`:
`python/mori/ops/dispatch_combine.py` selects CCO by default only for gfx125x.
The two backends register memory differently:

| Path | Node-local access | NIC MR registration |
|---|---|---|
| MORI `shmem`, default `static_heap` | One symmetric GPU heap; allocations are subregions and share its P2P mapping | Register the entire heap once during shmem init, if RDMA peers exist, through `RegisterRdmaMemoryRegionAuto` (`ibv_reg_mr` first, dmabuf fallback by default); subregion allocations do not create MRs |
| MORI `shmem`, explicit `vmm_heap` | VMM heap, allocated and mapped per chunk | Each allocated chunk has its own dmabuf MR, lkey, and per-peer rkey; `MORI_SHMEM_VMM_CHUNK_SIZE` controls chunk size (default at least 64 MiB) |
| MORI `shmem`, `ShmemSymmetricRegister(..., false)` | HIP IPC / P2P registration | Skip NIC MR for buffers used only within a node |
| MORI EP, explicit `MORI_EP_COMM=cco` | Subregions of one CCO LSA arena | One CCO window/MR for that arena, subject to the CCO registration path |
| Current fused AITER CCO | Entire CCO allocation mapped for same-node LSA; only its rail prefix exposed as a window | One dmabuf MR with IOVA zero for the rail prefix; guard at 1 GiB |

The local-only `ShmemSymmetricRegister(..., false)` API exists in MORI, but
the EP handle's ordinary `MallocSymm` buffers use `ShmemExtMallocWithFlags`
and the selected heap mode. MORI EP32 does not, by default, split its EP
buffers into local-only and rail MRs.

Source anchors: `mori/src/shmem/init.cpp` (`InitializeStaticHeap`),
`mori/src/application/memory/symmetric_memory.cpp` (`RegisterSymmMemObj`,
`RegisterRdmaChunks`), `mori/src/application/transport/rdma/rdma.cpp`
(`RegisterRdmaMemoryRegionAuto`), and `mori/src/cco/cco_init.cpp`
(`ccoWindowRegister`). The README EP32 result does not identify its shmem heap
mode or MR sizes. The 1 GiB guard in this fused CCO operator is a conservative
limit for the path observed here, **not a proven universal Ionic NIC MR limit**.

A direct 47+51 check confirms the distinction: one rank per host, MI355 GPU 0,
`MORI_EP_COMM` unset, `MORI_SHMEM_MODE=static_heap`,
`MORI_SHMEM_HEAP_SIZE=2G`, Ionic `rocep121s0`, and
`MORI_ENABLE_DMABUF_REG` unset. Both ranks completed shmem initialization and
finalization with exit code 0. Each trace reported
`RegisterRdmaMemoryRegionAuto[ibv_reg_mr]` with
`size=2147483648` (2 GiB) for the heap. This proves 2 GiB MR registration
works via MORI shmem's ordinary-address path on these hosts. It does not test
high-offset RDMA payload writes or the CCO dmabuf IOVA-zero path. The probe
used `torchrun --nnodes=2 --nproc_per_node=1` and the same Ionic device
selection and data interface as the CCO transport probes.

CCO cannot simply switch this operator to MORI shmem's `ibv_reg_mr` call:
`ccoWindowDevice` and `ccoGda::put/get` encode the remote address as a window
offset from IOVA zero. Ordinary-address registration would also require
exchanging each peer's registered base IOVA and changing the device-side GDA
address calculation. A smaller C1 change is to keep bounded CCO rail windows
and give local-only LSA buffers a path that skips NIC registration. CCO's
base `ccoWindowRegister` maps the full allocation for same-node peers but
has no local-only switch; it registers a NIC MR whenever an RDMA context exists.

## Local-only CCO API and large-buffer validation

Local MORI commit `2d8345c4` on branch `codex/cco-local-only-20261112`
adds `ccoWindowRegisterLocal` and
`Communicator.register_window(ptr, size, rdma_register=False)`. This creates
the same LSA mappings and GPU window metadata while skipping NIC registration;
the returned window has zero lkey and peer rkeys. Teardown records whether
an MR was registered and skips NIC deregistration for local-only windows.
The existing C++ registration symbol and default Python calls remain supported.
The commit is local only and does not include the user's dirty MORI EP test.

The candidate library and Cython extension were compiled in an isolated
51 worktree at `/home/hzm/cco_local_only_20261112/` from MORI base
`84a33cc0`; only the small runtime bundle was transferred directly from 51
to 47. Installed packages were not changed, and no binary or tensor was
transferred to the local workstation. Both hosts used identical artifacts:

- `libmori_cco.so`: `6cc6926fa5cc158d3f169be82839e4e92e8ce8e1a2edb4c598a80ae65f572695`
- Cython extension: `e0816a506661ee3646a5a305b3d5b6c5dcb93026104adf125d27d46f68158c56`

| Run tag | VA spaces | Local-only buffer per GPU | Dedicated rail MRs per GPU | Result |
|---|---|---:|---|---|
| `local4_rail4_gdr0` | One communicator | 4 MiB | 4+4 MiB | 384 checks, both host exit codes 0 |
| `local128_separate` | Separate local/rail communicators | 128 MiB | 4+4 MiB | 384 checks, both host exit codes 0 |
| `local32g_k3_rail832` | Separate local/rail communicators | 32 GiB | 832+832 MiB | 384 checks, both host exit codes 0 |
| `local32g_ds_rail704x3` | Separate local/rail communicators | 32 GiB | 704+704+704 MiB | 448 checks, both host exit codes 0 |
| `local32g_k3_gdr1` | Separate local/rail communicators | 32 GiB, GDR=1 | 832+832 MiB | 384 checks, both host exit codes 0 |

The first four rows use `CCO_GDR_CAPABLE=0` for local allocations; the final
row retains GDR=1 for both local and rail allocations. Local-only registration
still returns zero NIC keys with GDR=1, so this tested combination does not
require temporarily changing the allocation flag. Rail allocations use
`CCO_GDR_CAPABLE=1`. The two-communicator rows request no
GDA connections for the local communicator, and rail-only connections for
the rail communicator. Tests check all window keys, 8 KiB at the local
allocation tail through all seven same-host peers in two generations,
collective local-window deregistration/re-registration, and rail MR tail
writes before and after local-window teardown. They exercise allocation
and communication, not the full contents of the 32 GiB buffer or fused GEMMs.

The one-communicator `local128_rail4_gdr0` case still fails in a rail
allocation's `hipMemSetAccess`, before local-window registration. A private
experiment reserving a 2 MiB unmapped VA guard after each allocation also
fails (`local128_guard2`); that change was rejected and is absent from the
local MORI commit. Separate VA spaces are a measured working workaround,
not a diagnosis of the driver's exact failure mechanism.

Reproduce the large tests in both task directories with node rank 0 on 47,
node rank 1 on 51, and unused matching ports:

```bash
bash launch_local_windows.sh NODE_RANK 29785 local32g_k3_rail832 32768 832 0 --separate-communicators
bash launch_local_windows.sh NODE_RANK 29786 local32g_ds_rail704x3 32768 704 0 --separate-communicators --rail-windows 3
bash launch_local_windows.sh NODE_RANK 29787 local32g_k3_gdr1 32768 832 1 --separate-communicators
```

For this MORI change, Python parsing, Black 25.1.0, Ruff 0.12.0 with the
repository's `tools/codestyle/pyproject.toml`, clang-format 20.1.8, and
whitespace checks passed. Native CCO and Cython 3.1.3 builds passed on 51.
This does not establish CI success for the pending AITER PR, EP16 fused
regression, or completed C1/C2/C3: the AITER operator still uses the original
combined arena and must be wired to the separate allocations and handles.

## C1 region partition and shared resource manager

`cco_buffers.py` now provides `CcoBufferPlan` and `CcoBufferSet`. The planner
assigns each Stage1/Stage2 region a window name and window-relative offset,
preserving its shape, dtype, alignment and parity stride. It excludes the
legacy Stage2 embed slot from physical accounting. Local payload/state is
packed into one local-only window; each stage's rail regions are packed into
whole-region MRs within the conservative 1 GiB CCO guard. A single region
that exceeds the guard is rejected with a chunking diagnostic. Further
splitting of individual regions for larger EP shapes remains unfinished.

All rail allocations are padded to the largest registered prefix in the
plan; the NIC registers only the occupied prefix of each window. The resource
manager borrows distinct local and rail communicators, requires the new
MORI local registration API, and creates eight GDA contexts for the rail
communicator. It owns and releases only its device communicators, windows and
allocations; caller-owned communicators remain usable. Cleanup synchronizes
the owning GPU, attempts every release in dependency order, and reports the
first error. Injected host checks verified cleanup failure handling,
idempotence, closed-pointer rejection, MR limits and invalid parity handling.

The actual proposed A8W4 layouts were allocated and tested on all 16 GPUs,
with four logical nodes, TPR16384, tile group 4 and the illustrative receive
bound `max_recv_routes_per_rank = 1.5 * TPR * topk`:

| Model | Local-only allocation | Rail physical allocations | NIC MR prefixes | Total allocation per GPU | Final checks |
|---|---:|---|---|---:|---:|
| K3 | 18880 MiB | 832+832 MiB | 576+832 MiB | 20544 MiB | 31840 |
| DSv4.1 Flash | 9920 MiB | 704+704+704 MiB | 704+704+512 MiB | 12032 MiB | 31872 |

Both hosts returned zero for the final combined run
`c1_actual_models_16384_final`. The probe checks every region at its head/tail
in both parities (persistent state uses its single plane), through all seven
same-host peers over two generations. Head/tail samples are disjoint and at
most 8 KiB each. It also writes the tail of every rail region/parity and NIC
registered prefix, then transfers 8 KiB from high `node_accumulator` offsets
to high `remote_partial_rx` offsets through every GDA context and parity.
For DS these source and destination regions use different MRs. Both models
run in one process group, including collective resource/communicator teardown
between their allocations. No full validation tensor was saved or copied.

An earlier probe revision incorrectly overlapped two 8 KiB samples within
K3's 13188-byte `tile_expert_group` plane; its failure was a test overwrite,
fixed by bounding sample width before the passing runs. Another revision's
postponed annotations hid function-local `fx` from FlyDSL's signature
resolution; it failed during compilation and was corrected. These attempts
are retained as small remote logs, not transport-failure evidence.

Reproduce inside both isolated directories
`/home/hzm/cco_c1_buffers_20261112/` (rank 0 on 47, rank 1 on 51):

```bash
bash launch_megamoe_cco_buffers.sh NODE_RANK 29804 c1_actual_models_16384_final
python3 probe_megamoe_cco_buffers.py --kernel-dir kernel --plan-only
```

The initial large run used the local-only MORI overlay and the existing
FlyDSL 0.3.1 transport runtime; the pinned-runtime section below records the
successful repeat with 0.4.1. The test does not compile AITER's fused
kernels, run the four-logical-node routing protocol, or validate the receive
bound against a routing fixture. `MegaMoEInternode` still uses the legacy
combined arena: kernel arguments and LSA/GDA address resolution must consume
the new region locations before the public operator can switch allocations.

Black 26.5.1, repository-pinned Ruff 0.16.0, Python parsing and whitespace
checks passed for all 18 changed Python files relative to local main merge
base `0dd271600990389dc9eeea60307a758bebf3c040`, including pending/untracked
files. The three probe launchers passed shell syntax checks. Existing probe
imports/default stream declarations were corrected for Ruff; Stage1's
bounded LDS load retains its nested runtime guard with a documented SIM102
suppression. The repository dependency pre-check was attempted locally and
failed to clone CK because Git could not verify GitHub's server certificate,
including with the system CA bundle explicitly selected. No PR/push occurred;
this is not a claim that every CI check passed. Obsolete runtime tarballs,
the rejected guard libraries, temporary Cython objects and the verified
task-owned HTTP server were cleaned up; required runtime libraries and small
reproduction logs remain remote.

## Pinned FlyDSL runtime bring-up

The repository pins FlyDSL 0.4.1. That exact cp312 wheel was installed on both
hosts under `/home/hzm/cco_c1_buffers_20261112/flydsl_041`; no installed system
package was changed. Both imports expose `fx.generic_load` and
`fx.generic_store`. The 77.3 MB wheel transferred directly from 51 to 47 over
the internal data network, with SHA-256
`ead3549cd8d151832a33ddb0519f6194ec1050ee5c25ca9cf1a342c85f183983`
on both hosts. The transfer server and both redundant wheel copies were
removed after installation; necessary runtime libraries remain remote.

The task's `flydsl_runtime` symlink now selects that pinned version. The same
CCO buffer probe passed for both A8W4 models at TPR128, four logical nodes,
two generations and eight GDA contexts (`c1_041_models_128`, port 29805):
K3 26912 checks and DSv4.1 Flash 27360 checks, with both host exit codes zero.
The TPR16384 layouts were then repeated with 0.4.1
(`c1_041_models_16384`, port 29806): K3 31840 checks and DSv4.1 Flash 31872
checks, with both host exit codes zero. Allocations and NIC prefixes matched
the large-layout table above, and all eight GDA contexts and both parities
passed again.

```bash
bash launch_megamoe_cco_buffers.sh NODE_RANK 29806 c1_041_models_16384
```

The local-only MORI overlay still uses its older world-rank GDA device API
and does not expose `Team`. Consequently the 0.4.1 smoke test establishes
compiler/transport compatibility for this probe, not compatibility with
the fused operator's PR895 team and caller-ordered doorbell API. The PR895
runtime integration below now validates matching local-only, team and
caller-ordered APIs. Fused-kernel region wiring remains the next step.

## Layout evidence and remaining work

For A8W4, four logical nodes, TPR16384 and illustrative
`max_recv_routes_per_rank = 1.5 * TPR * topk`, the current layout reports:

| Model | Stage1 total, including embedded Stage2 | Stage2 arena | Stage1 rail prefix | Stage2 rail prefix | Stage2 inbox |
|---|---:|---:|---:|---:|---:|
| K3 (H3584, K16) | 20015 MiB | 15163 MiB | 512 MiB | 809 MiB | 14336 MiB |
| DSv4.1 (H5120, K6) | 11697 MiB | 8843 MiB | 672 MiB | 1156 MiB | 7680 MiB |

The bound is only a sizing example; no routing fixture has yet validated
alpha=1.5. The present operator embeds the entire Stage2 arena in one CCO
allocation. That is unsuitable for these shapes even though the rail-addressed
prefix is much smaller. C1 must separate large local/LSA payloads from the
registered rail windows, split the DS rail prefix across MRs within the
validated CCO registration size, and respect the multi-MR allocation behavior
observed above. With 64 MiB registration alignment, the Stage1 rail controls
raise K3's 512 MiB payload to a 576 MiB window and DSv4.1's 672 MiB payload
to 704 MiB. Stage2 needs about 832 MiB for K3; DSv4.1 needs separate
704 MiB and 512 MiB windows for its largest two rail regions.

EP16 layout metadata was compared against the branch's unchanged `HEAD`:
K3 and DSv4.1, TPR 32/128/1024/4096/16384, tile group 1/2/4. All 30 Stage1
and 10 Stage2 variants had identical region names, shapes, offsets, sizes and
total bytes. A separate tune-table check confirmed quant/topology key isolation
for A4W4 16×8, A8W4 16×8, A8W4 16×4 and A8W4 32×8. These are host checks;
EP16 GPU correctness and performance still need a run on the changed kernels.

Next implementation steps from the guide:

1. Complete C1 allocation separation and capacity/fail-fast semantics. Check
   registered window creation and high-address writes with the actual proposed
   rail partition before connecting GEMM buffers.
2. Complete C0/C2/C3 in the fused operator: multi-peer Stage1 state and posts,
   N-plane Stage2 arrival/return/reduce, runtime rank/node values, and the
   virtual overlay for cross-host relays. The current public operator still
   rejects `gpus_per_node=4` because these kernels are incomplete.
3. Run virtual 4×4 A8W4 numerical and replay fixtures, then EP16 regression.
   Performance and EP32 claims require four real nodes.

To rerun a probe, use `bash launch_megamoe_virtual4_transport.sh` in the task
directory on both hosts, passing node ranks 0/1, the same unused port and tag,
and identical probe arguments. Set `CCO_GDR_CAPABLE=1` on both hosts for the GDR
path. Keep the 47 and 51 data-network interface and rail environment from the
launcher; inspect both `.log` and `.rc` files.

## PR895 local-only integration and virtual rail stress continuation

The isolated PR895 runtime at `/home/hzm/cco_pr895_c1_20261112` on 47 and
51 now combines local-only registration with the team and caller-ordered
device API from MORI base `8073754b`. The native library and Cython extension
match across hosts; device bitcode is selected per host's actual Ionic CQ
mode. FlyDSL remains the repository-pinned 0.4.1 in the isolated overlay.

The initial PR895 rail attempt used 51's CCQE bitcode on 47, whose firmware
`1.117.5-a-56` selects ordinary CQE. It timed out before transport compatibility
was established. The corrected 47 bitcode uses CQE; 51 firmware
`1.117.5-a-147` uses CCQE. Both probes reject an explicit bitcode/CQ mode
mismatch before allocating GPU buffers. No firmware or installed package
was modified.

| Artifact | SHA-256 |
|---|---|
| Native library, both hosts | `e2ee96fa84e8e7a1a87be65c3b666674d1c32497edf2348a7c2e54ad9dfbc8cf` |
| Cython extension, both hosts | `5cce3cf999d55533fac97f2bf640727a46adc4ef57fce5dd5fcf90efa6a6d661` |
| 47 CQE device bitcode | `8616e292f097ccef616698c3a176736437cc036f605177f617d43fff885ab1bc` |
| 51 CCQE device bitcode | `e196e3e01e643b27492606e3f6dce85d7bfef0ca6f0e87b8d60a1af5d5529188` |

The deferred-registration single-node probe previously passed 80 checks:
FULL creates MRs for ordinary windows while local-only windows retain zero
NIC keys, including windows registered after FULL communicator creation.
The corrected two-host Team.GDA/caller-ordered model-buffer runs also pass:

| Tag | TPR | K3 checks | DSv4.1 Flash checks | Host exits |
|---|---:|---:|---:|---|
| `pr895_team_pbf_128_modes` | 128 | 26912 | 27360 | 0 / 0 |
| `pr895_team_pbf_16384_modes_r2` | 16384 | 31840 | 31872 | 0 / 0 |

The large allocations and MR prefixes retain the preceding C1 table:
20544 MiB/GPU for K3 and 12032 MiB/GPU for DSv4.1. Both parities, both
generations, all eight contexts, region heads/tails and cross-MR high-offset
transfers pass. The alpha=1.5 receive bound remains a sizing example.

The virtual transport probe now rotates its actual posting/ready/credit
context each generation through MORI's per-call `ctx` API. Previously
`--gda-contexts` created several contexts while every operation used context
zero. New options `--rail-team` and `--caller-ordered` exercise team-indexed
peers and deferred payload/ready posts followed by a caller-ordered flush.
Payload and ready WQEs stay in order on the same QP. Earlier single-context
runtimes retain their default context-zero call signature; unsupported
multi-context/team/caller-ordered requests fail during argument checks.

| Tag | Delay per generation | Rounds | Actual contexts | Directed payload checks | Host exits |
|---|---|---:|---:|---:|---|
| `virtual4_pr895_8ctx_delay_r2` | rank11, 250 ms | 16 | 8 | 768 / 768 | 0 / 0 |
| `virtual4_pr895_8ctx_final_r3` | rank3, 250 ms | 16 | 8 | 768 / 768 | 0 / 0 |
| `virtual4_pr895_world_control_r4` | none | 3 | 1 | 144 / 144 | 0 / 0 |

All rows use two MRs, GDR=1 and 256 KiB payloads; ready/credit and all three
route classes (LSA, direct rail, rail then LSA relay) pass. R3 and R4 use final
probe SHA `e6b433dd8609763286f864ba77a0b06b50b4dcf28e5fe472644be36870a7696e`.
R2 used the earlier direct-ctx source
`ebc56e14a5579354e31a6f03cbd3d049fdbcc1c011314807a6cb1f855e82eb9b`;
the final source independently repeats the multi-context test.

A private supervisor now exchanges each host's eight-GPU idle status and
releases both hosts after a joint ten-second idle interval. Its CPU test
injects an occupied peer and checks that the idle interval resets and both
releases synchronize. R3 release timestamps differ by 0.435 ms; R4 by
0.435 ms. Each owned test process group has a five-minute cap and a
twenty-second all-GPUs-busy/no-log-progress watchdog. An earlier R1 waiting
attempt was stopped before transport kernels because an external task
occupied 51. It is excluded from passing results.

Reproduce with node0 on 47 (`ziming_huang_work`) and node1 on 51
(`hzm_work`), in the isolated task directory. Set each host's explicit
`MORI_CCO_BC`, `CCO_TEST_BITCODE_CCQE=0/1`, `BUILD_CCO_SDMA=ON`,
`CCO_TEST_LIBRARY_DIR=$PWD/runtime/mori` and `CCO_GDR_CAPABLE=1`:

```bash
python3 supervise_rail_test.py --launcher launch_megamoe_virtual4_transport.sh \
  NODE_RANK 29816 NEW_TAG -- --payload-bytes 262144 --multi-mr \
  --gda-contexts 8 --rounds 16 --delay-rank 3 --delay-ms 250 \
  --rail-team --caller-ordered
```

Final PR-plus-pending scope: all 18 changed Python files pass Black26.10.1
and parsing; full-repository Ruff0.16.0 passes; all three changed shell
scripts and diff whitespace checks pass. Dependency pre-check was not
repeated for this private test continuation. Both hosts have no owned test
workers or transfer servers left; all GPUs returned to idle and approximately
298 MB/card. Logs, bitcode and runtime assets remain remote; local records
contain only small summaries and reproduction information.

This completes the PR895 runtime/transport prerequisite. The fused operator
still needs C1 region-handle/address wiring, multi-peer Stage1 and N-plane
Stage2 before four-logical-node GEMM numerics and EP16 regression. No
four-physical-node, EP32 performance or full-fused accuracy claim follows
from these transport checks. Compact evidence:
`/home/zihuang/work/mi355_cco_local_20261112/RAIL_TEST_FINAL_SUMMARY.json`.

## RDMA atomic capability audit

Read-only verbs queries on 47 and 51 report `atomic_cap=2` (`ATOMIC_GLOB`)
on all eight Ionic rails, with `max_qp_rd_atom=16` and
`max_qp_init_rd_atom=16`. The ctypes query on 51 agrees with `ibv_devinfo -v`;
47 has no installed `ibv_devinfo`, so it uses the same libibverbs query.
These device attributes establish advertised capability, not a tested
NIC/GPU-memory atomic coherence or performance result.

The PR895 CCO FlyDSL `Gda.signal(INC/ADD)` and data PUT with an optional
signal post uint64 RDMA fetch-add WQEs. Their returned old value is stored
in an internal endpoint buffer and is not returned by this wrapper. Native
C++ also has `ccoGda_WindowSignalAdd` for a registered window-relative
counter; the current Python/FlyDSL wrapper exposes communicator signal
slots only. The Ionic provider also implements compare-swap WQEs, but this
CCO FlyDSL API does not expose general CAS or arbitrary-address fetch-add.

Current MegaMoE `RailGda` uses RDMA WRITE and inline eight-byte WRITE for
ready/credit. Both the fused operator and virtual transport probe request
zero GDA signal slots. Their GPU system-scope atomic operations on LSA peer
pointers are separate from NIC RDMA atomics. None of the passing transport
rows above exercises RDMA atomic. Multi-writer arrival signalling could use
CCO atomic signal slots after allocating them and validating ordering,
generation reuse and concurrent updates. Compact-slot allocation that needs
the old fetch-add value additionally needs a result-returning binding.

## 05/06 continuation: operator buffers and logical-peer windows

The continuation uses 05 (`168.144.57.63`, data `10.15.20.6`) and
06 (`168.144.62.62`, data `10.15.20.7`), each with eight GPUs. The Docker
container is `ziming_megamoe_20261112`; the isolated container task directory
is `/workspace/virtual4_05_06_20261112`, mounted from
`/opt/mi355-4node-bringup-20261112/virtual4_05_06_20261112` on each host.
FlyDSL is the repository-pinned 0.4.1. The task uses the local-only CCO
runtime with PR895 `Team.GDA` and caller-ordered doorbells. Node-specific
device bitcode matches the disabled Ionic CCQE configuration. Large logs,
build files and compiler caches remain on the remote hosts.

The operator now consumes the separate C1 local/LSA and rail region handles
through `buffer_addressing.py`. Concurrent single-kernel Stage1 and Stage2
use separate eight-context ranges: 16 contexts total, Stage2 base 8.
Two-kernel operation retains eight contexts. Stage1 explicitly joins the
local mask writers and remote mask parser before publishing its Stage2
handoff; destination tile EOS alone did not establish mask completion.

DSv4.1 Flash TPR1024 reproduced the mask publication gap with swapped-route
relL2 0.1886 against the existing 0.05 threshold. Two-kernel and legacy-arena
controls passed. The repaired candidate passes random/EPLB at seeds
123/124/125. Original numerical tolerances, routing fixtures, all 16
collective rank-max checks per case, and graph/replay checks were preserved.
The numerical operator runs in the following table use **physical 2x8 EP16**.

| Run tag | Port | Validation | Result |
|---|---:|---|---|
| `c1_operator_041_k3_maskready_clean_r17` | 29945 | K3 A8W4, TPR32/128/1024, random/EPLB | 6 cases, 96 checks pass |
| `c1_operator_041_ds_compact9216_r19` | 29947 | DSv4.1 Flash A8W4, TPR1024, capacity9216, random/EPLB | 2 cases, 32 checks pass |
| `c1_operator_041_k3_a4w4_streamcompat_r24` | 29952 | K3 A4W4, TPR128/1024, random/EPLB | 4 cases, 64 checks pass |
| `c1_operator_041_ds_a4w4_streamcompat_r25` | 29953 | DSv4.1 Flash A4W4, TPR128/1024, random/EPLB | 4 cases, 64 checks pass |

All table rows have both worker and supervisor RC0 on both hosts. The
compact DS case uses 9216 input slots instead of 16384 and validates these
fixtures and seeds; it does not prove a receive bound for arbitrary routing.
The A4W4 reference frontend needed `tensor_shim.current_stream_arg`, absent
from the older task snapshot. The exact helper from the review checkout was
added only to the isolated reference overlay, with provenance in
`reference_stream_compat.json`. The candidate GEMM bodies were unchanged.

`c1_operator_041_ds_overflow_cap128_r26` (port29954) injects an undersized
capacity128 into DS TPR128 EPLB, which has 768 routes/rank. Both workers exit
within approximately 28 seconds with HSA exception 0x1016, SIGABRT and RC1;
both supervisors complete and all 16 GPUs return idle without KFD owners.
This negative test uses `--no-check` and produces no numerical pass. Its
observed diagnostic is a GPU hardware exception; there is no captured
capacity-counter message or friendly Python capacity exception.

Earlier attempts remain in the evidence: r16 has both worker RC0 but a
missing 05 supervisor receipt after sysfs EOPNOTSUPP, so monitoring is
incomplete; r17 is its clean repeat. r18 used the invalid `dsv4_flash` key
and exited before GPU computation (`dsv41_flash` is correct). r20/r21 failed
in the A4W4 reference import. r22 failed a C1 check because initial relay
scratch was not parity-indexed; r23 and r27 use the corrected layout.

### Actual logical-peer window validation

`peer_layout.py` plans peer slots in ascending logical-node order, excluding
self. Stage1 uses `remote_dispatch_rx`, `remote_chunk_ready` and
`remote_chunk_credit`; Stage2 uses `remote_partial_rx`, `return_group_ready`
and `return_consumed`. Allocation-shaped parity strides are retained, while
source accumulator planes use absolute node indices. CPU checks cover 140
disjoint peer slices across 2/4/8 logical nodes and reject self peers.

`megamoe_peer_window_probe.py` extends the existing C1 buffer probe with
all three logical peers in a virtual 4x4 topology. Full peer payload slices
travel over LSA, direct rail and rail-plus-LSA relay. Separate payload/credit
relay banks and parity slots avoid inbox clobbering. Its extra `peer_bridge`
window is bounded test scratch. CPU plan-only checks pass on both hosts for
TPR128 and reject TPR16384 before GPU allocation when relay scratch exceeds
the existing padded rail window.

| Run tag | Port | Generations | Peer payload/credit checks |
|---|---:|---:|---|
| `virtual4_operator_peer_windows_smoke_r23` | 29951 | 2 | K3 384, DSv4.1 Flash 384, all pass |
| `virtual4_operator_peer_windows_stress_r27` | 29955 | 16 | K3 3072, DSv4.1 Flash 3072, all pass |

Both runs delay rank3 by 250ms per stage. r27 rotates through eight GDA
contexts, retains the C1 region/address checks (K3 78528, DS 80000), and
completes with both worker and supervisor RC0 on both hosts. Each supervisor
records 51 occupancy samples, no foreign KFD owners and no telemetry
errors. All GPUs return to baseline 299544576 bytes/card and idle.

Stage2 transfers in this probe are allocation-shaped raw bytes. They do not
validate the encoded-row ABI, GEMM, four-plane reduction or combine. Host
barriers order the probe; full-forward graph replay is not exercised.
**Full virtual 4x4 fused forward remains unimplemented.** The operator
continues to reject `gpus_per_node=4`.

Remaining consumers are concrete: Stage1 must post/parse all peers and
expand routes across source planes, generalize its eight-writer EOS and
two-plane mask joins, and feed GMM1. Stage2 still uses `remote_node = 1 -
node` and two-plane iteration; it must reduce four source planes, send
peer-specific partials/credits and perform the four-way final combine.
These consumers must be wired into the existing numerical driver before
reporting C2/C3 acceptance or EP32 performance.

Reproduce from the host on both machines, using NODE=0 on 05 and NODE=1 on
06. Select a new shared port and tag; the supervisor enforces a joint
ten-second idle gate and does not overwrite existing receipts:

```bash
VIRTUAL4_FLYDSL_PIN=0.4.1 python3 \
  /opt/mi355-4node-bringup-20261112/virtual4_05_06_20261112/supervise_probe.py \
  NODE PORT UNIQUE_TAG probe_megamoe_cco_buffers.py \
  --kernel-dir /workspace/virtual4_05_06_20261112/kernel \
  --tokens 128 --models kimi_k3 dsv41_flash --generations 16 \
  --rail-team --caller-ordered --logical-peer-windows \
  --delay-rank 3 --delay-ms 250
```

The supervisor now catches per-field sysfs OSError, records unknown values
and telemetry errors, blocks idle release on unknown data, and continues
KFD ownership scans. Fault simulation covers normal idle, unknown telemetry
and foreign ownership. Both summary collectors retain worker-only receipts
as monitoring-incomplete and preserve telemetry failures. Small checkpoints,
hashes and reproduction information are in
`/home/zihuang/work/mi355_virtual4_05_06_20261112/report.md` and the adjacent
JSON receipts. Black26.10.1 over all 21 PR-plus-pending Python files,
Ruff0.16.0 over the repository, AST, whitespace and three shell syntax checks
pass. The unchanged dependency pre-check also passes using the exact
working-index gitlink and `.gitmodules` with the original CI script in an
isolated metadata checkout. This continuation has not been pushed.

## A8W4 优先：A1/A2 初步对照（05/06，2026-11-12）

按 `/home/zihuang/trace/doc/requirements.md` 的 A1 → A2/A4 → A3/A5 顺序推进。以下是物理 2×8 EP16；虚拟 4×EP4 仍只有协议/窗口压力验证，尚无完整 fused-forward 数值结果。

A1 使用独立 CSV，G2/G4 仅改变 Stage1 tile_group，Stage2 保持相同 BM32/BM64、BN128 和 xGMI。K3/DS4.1 的 TPR256/1024、random/EPLB 共 16 个配置、256 项原检查通过，两端 worker/supervisor RC0、无 foreign KFD、无遥测错误。G2 消除了 VGPR spill；Stage1 scratch 为 64 B/workitem。G4 的 K3 scratch 为 368/464 B/workitem，DS4.1 为 528。该实测支持 GMM1 溢出假设；A1 的 TPR128/512/4096/16384 与 G1 资源对照尚未完成。

A2 保留的候选是 `_gemm1_body_sc4`：仅 fp8 A、BM128 按四个 M32 scale group 读取 A；全 BM128 共用 B/B scale，最后一个 M group 消费 J 后才 refill，因此 B 每 K 步的加载次数不增加。累加器及 scale 使用原全局 sub 索引，A 片段使用组内索引。A4 和其余 BM 保持原流水顺序。

两个 M64 段的 sc3、M32 加提前 scale 写入的 sc5 均有保留实验副本。M64 内联 Stage1 仍有较多 spill；提前 scale 写入对 K3 资源无收益、时间差在约 ±2% 内，最终选择改动更小的 M32 sc4。独立 GMM1 对照覆盖两个模型、A8/A4、BM32/64/128、seed123/124/125，共 36 个 case，h1 payload/scale 均与原 baseline 逐字节相同。A4 的六组 BM/模型组合生成指令逐条相同，资源也相同；这不代替多卡端到端 ≤1% 回退验收。

M32 候选 A8W4 的 8 个多卡 case、128 项原检查均通过。K3 Stage1 scratch=112、VGPR spill=9；DS4.1 scratch=144、VGPR spill=17；两者 VGPR=256。独立 GMM1 BM128 的 scratch=0，但完整 Stage1 尚有 spill。链接的 `__assert_fail` 有 64 B 私有栈，G2 基线也包含该栈。**A2 的 ≤40 B/workitem 验收尚未达到，目标未放宽。**

| 模型 | TPR | 路由 | 原 G4 Stage1 µs | M32 Stage1 µs | 下降 | 原 max-rank p50 µs | M32 max-rank p50 µs |
|---|---:|---|---:|---:|---:|---:|---:|
| DS4.1 | 256 | eplb | 248.3 | 179.2 | 27.8% | 505.2 | 426.8 |
| DS4.1 | 256 | random | 293.6 | 201.3 | 31.4% | 601.6 | 490.0 |
| DS4.1 | 1024 | eplb | 433.0 | 303.0 | 30.0% | 832.3 | 698.5 |
| DS4.1 | 1024 | random | 572.9 | 455.1 | 20.6% | 1071.4 | 955.7 |
| K3 | 256 | eplb | 413.1 | 284.8 | 31.0% | 777.9 | 641.3 |
| K3 | 256 | random | 453.8 | 300.5 | 33.8% | 894.6 | 735.8 |
| K3 | 1024 | eplb | 1346.5 | 849.3 | 36.9% | 2008.3 | 1487.4 |
| K3 | 1024 | random | 1138.8 | 728.1 | 36.1% | 1868.0 | 1441.0 |

Stage1 口径是各 rank 从最后 5 个完整 profiler replay 中按最小 kernel sum 选择的流内核时间，再求均值；max-rank p50/p90 来自独立 full-forward GPU-event 样本。不是 host elapsed。以上是相同 seed123 的初步配对实测，尚非全量调优结果。K3 沿用本轮对照 driver 的 SiTU beta=1、linear_beta=1；实际模型 4/25 合约另需回归。

基线 tag：`a1_k3_g4_t256_1024_r28`、`a1_ds_g4_t256_1024_r30`；M32 tag：`a2_k3_sc4_g4_t256_1024_r36`、`a2_ds_sc4_g4_t256_1024_r39`。小型资源、原检查和 max-rank p50/p90 摘要在本地任务目录的 `a2_a8w4_m32_comparison.json` 与两端 `*.a1_checkpoint.json`；完整数据留 05/06，已分析的 MLIR/LLVM 中间产物及时清理。候选 gemm.py SHA256：`282138145f06b51345cf2a6caeb4d1d952dc16b8fb468c7f10d75c7521ac15bf`。

下一步：补 A1 全 TPR/G1；进一步降低内联 Stage1 寄存器峰值；核查 Stage2 BM128/SDMA；做真实 SiTU 合约、A4 性能 guard 与 A8 全量多卡回归，再进入 A3 和 C2/C3。TPR4096/16384 已有 CPU 容量计划，random 用各模型 1.5·T·K receive cap，EPLB 用默认理论 cap；二者不能混用同一固定容量参数。大 TPR 的 MORI heap 与 supervisor deadline 仍需先核实。

A4 候选多卡回归收尾：r40（K3）、r41（DS4.1）在 TPR256/1024、random/EPLB、相同 G4/xGMI 配置下共 8 个 case、128 项原检查通过；两端 worker/supervisor RC0，无 foreign KFD/遥测错误。最终 M32 源码 A8/A4 合计 256 项原检查通过。A8/A4 的 Stage1 时间比为 1.206–1.435；DS TPR256 EPLB 等初步观测仍超过 1.4× 目标，不能称为全范围性能验收通过。A4 端到端相对旧源码 ≤1% 回退 guard 尚未单独配对测量。

最终两端 88 份任务源码、41 份 overlay 源码哈希一致，16 卡 idle、每卡 VRAM=299544576 B、无 KFD 进程。Black26.10.1 覆盖整个 PR 相对 base 加 pending 的 21 个 Python 文件，Ruff0.16.0 覆盖整个仓库；AST、3 个 shell launcher 和 diff whitespace 通过。dependency precheck 沿用本地原 CI 脚本通过结果，已重新核对 gitlink/.gitmodules/script 均未变化。保留关键 ISA、指标和复现信息，清理已完成实验的 MLIR/LLVM 及多余 ISA 副本；本轮没有提交或推送。
