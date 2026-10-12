# TP2 relay for the mesh quick all-reduce

Design notes and measurements behind the relay in `quick_allreduce_relay.py`,
`quick_allreduce.py` and `kernels/quick_allreduce_mesh.py`. The code comments
describe what the code does; this file records why the defaults and cache
policies are what they are, and what has been measured. Every measurement here
is on gfx950 (MI355X, 8 GPUs on xGMI).

## What it does

A TP2 pair on an 8-GPU xGMI node uses one of each GPU's seven links, and a large
all-reduce already runs close to that link's bandwidth. The relay sends part of
each direction's traffic through a third GPU, over the two links that GPU has to
the pair.

- `FlyQuickAllReduce(..., relay_devices=(r01, r10))` names one relay GPU per
  direction. Only a TP2 mesh on xGMI accepts it. Production dispatch never
  passes it.
- Block `b` is a relay block when `b % den < num`. It writes its packed packet
  into a bounce buffer on the relay GPU instead of the peer's inbox. The flag
  still goes to the peer's inbox, and flags, colours and the reduction order
  are those of the direct path.
- The peer's matching block waits on that flag and reads the packet from the
  bounce in place of its inbox.
- Nothing runs on the relay GPU. Each receiving rank allocates its incoming
  bounce there from the uncached IPC heap (`InGroupRelayProvider`). A
  `relay_provider` can supply the bounce buffers instead.
- Payloads below `relay_min_bytes` use the direct engine even when the relay is
  on.

## Cache policy

- **Relay stores** are system scope (`sc0 sc1 nt`). The `vmcnt(0)` before the
  flag then means the packet has landed on the relay, which the flag, written to
  a different GPU, does not otherwise order.
- **Relay loads** are plain loads. Before them, an agent-scope acquire
  (`buffer_inv sc1`) invalidates L1 once per tile (per group of tiles under
  super-tiling), after the reduce-scatter flag wait. A block's first tile skips
  it, because L1 is invalidated at the start of every dispatch.
- **Direct loads** stay `sc0 sc1` behind a workgroup-scope acquire, which emits
  no invalidate.

### Why the invalidate is needed

A probe had a reader wave load 64 dwords of an uncached (UC) buffer, a writer
on another GPU overwrite them with `sc0 sc1` stores and raise a flag, and the
reader reload them:

- L1 keeps UC lines. With nothing in between, plain and `sc0` reloads returned
  the old value in 262,144 of 262,144 lane reads, for `global_load` and for
  `buffer_load` with the kernel's descriptor, wherever the buffer lived.
- `nt`, `sc1` and `sc0 sc1` reloads were never stale on UC memory.
- Only plain traffic between the two reads evicts the line: up to 24 KiB left
  every lane stale, 28-36 KiB about half, and 40 KiB or more none.

A block reads the same bounce slots on every tile, so plain loads without the
invalidate are correct only while the block moves enough other traffic through
L1 between two reads of a slot. Blocks 256 and 512 happen to; blocks 64 and 128
do not. In a kernel revision where every block, relay and direct, read with
plain loads and no invalidate, 31.1M of 142M output lanes went wrong at blocks
64 and 128, while blocks 256 and 512 matched. With the agent-scope invalidate
on relay blocks, outputs matched the `sc0 sc1` kernel bit for bit: 0 of 142M
lanes at blocks 64/128, and 0 of 2.27e9 at blocks 256/512 over INT4, INT6 and
FP16 with super-tiles 1 and 8. Each comparison ran the two kernels in separate
processes on the same seeded inputs, four distinct calls back to back.

The multi-GPU suite has a regression row for this: block 64, every block
relaying, so each block re-reads its bounce slots with too little traffic in
between to evict them from L1. With the relay acquire narrowed to workgroup
scope, exactly that row fails, eager and under graph replay.

### Why plain loads on relay blocks

Two pairs, pair A relaying 3/8 of its blocks through pair B's GPUs while B runs
a GEMM `[8192,5120]x[5120,17408]` or a 1 GiB copy; CUDA-graph replay, INT4,
80 MiB. Changing only the receive-load bits of one kernel:

| receive loads | A's relay µs | B's GEMM slowdown | B's copy slowdown |
| :-- | --: | --: | --: |
| `sc0 sc1` | 334-339 | +23 / +27% | +16 / +18% |
| `sc1`, `sc1 nt` or `nt` | 337-338 | +21 to +22% | +12 to +17% |
| plain or `sc0` (no invalidate, unsafe) | 306-309 | +3% | +5% |

Loads that bypass L1 cost the relay GPU's own work 20-27%; loads that may
allocate in L1 cost it about 3%. Why bypassing reads of UC memory cost the
relay GPU that much is not established.

An invalidate on every block of both engines kept B's GEMM within 1% but slowed
the direct all-reduce by 6-15%. So only relay blocks take plain loads and the
invalidate, and direct blocks keep `sc0 sc1`, which needs none.

The current kernel, against the one before the receive split (which also moved
the tile schedule into a per-block function), in the same two-pair setup:

| | before | current |
| :-- | --: | --: |
| A's relay all-reduce, 80 MiB | 330.6-339.8 µs | 311.1-313.5 µs |
| B's GEMM slowdown, 10 MiB | +15.4 to +17.8% | +4.2 to +4.6% |
| B's GEMM slowdown, 80 MiB | +24.1 to +26.3% | +1.1 to +1.5% |
| B's 1 GiB copy slowdown, 80 MiB | +13.9 to +16.7% | +4.2 to +4.6% |

## Defaults

`MESH_RELAY_DEFAULTS[("xgmi", 2, "gfx950")] = (480 KiB, (3, 8))`. Other arches
have no defaults, so a caller must pass `relay_fraction` and `relay_min_bytes`.

### Block placement

The GPU has 8 XCDs and blocks reach them round-robin, so a `den` that divides 8
keeps the relay blocks on whole XCDs. Over 10-120 MiB that beat spreading the
same share evenly over every XCD, for both 3/8 and 1/2: at 80 MiB, 1.41-1.43x
against 1.37x over direct for 3/8, and 1.26x against 1.19x for 1/2. At 1/4,
spreading was 0.3-1.3% faster at 20-80 MiB. These were measured on an earlier
kernel than the tables below, so only the comparison carries over. `den` above
8 has not been tried.

### Speedup and cutoff

One pair, CUDA-graph replay, INT4, hidden 5120, relay fraction 3/8, the four
GPUs in use otherwise idle. Each time is the median over three repeats of the
slower rank's mean per replay; ranges are over two runs in separate processes.

| MiB | tokens | direct µs | relay µs | speedup |
| --: | --: | --: | --: | --: |
| 5 | 512 | 38.7-38.8 | 29.1-29.3 | 1.32-1.33x |
| 10 | 1024 | 69.7-70.0 | 51.0-51.9 | 1.34-1.37x |
| 20 | 2048 | 124.3-124.9 | 86.7-86.8 | 1.43-1.44x |
| 40 | 4096 | 235.3-235.9 | 158.6-158.7 | 1.48-1.49x |
| 80 | 8192 | 462.2-462.7 | 304.5-306.2 | 1.51-1.52x |
| 120 | 12288 | 688.9-689.9 | 450.2-452.1 | 1.53x |

Below 5 MiB the relay is 1.23-1.24x at 2.5 MiB and 1.07-1.08x at 480 KiB,
breaks even near 0.23 MiB, and loses about 3% at 0.16 MiB and below (one run
below 0.47 MiB). The 480 KiB cutoff sits inside the winning range.

### Fraction

In that one-pair sweep, 1/2 is 1.4-6% faster than 3/8 from 20 MiB up (1.60x at
80 MiB) and 1-7% slower at 2.5-10 MiB. From 0.47 MiB up, 1/4 and 5/8 never beat
3/8; below that, 1/4 is slightly ahead of it.

Under two-pair contention (pair A relaying through pair B's GPUs, three runs per
fraction, interleaved), 1/2 and 3/8 are within run-to-run spread at 2.5 and
10 MiB with the relay GPUs idle, and at 80 MiB A's relay all-reduce takes
288.8-289.9 µs at 1/2 against 310.4-313.4 µs at 3/8. The cost to B's work is:

| B's work | MiB | at 3/8 | at 1/2 |
| :-- | --: | --: | --: |
| GEMM `[8192,5120]x[5120,17408]` | 2.5 | +0.3 to +0.4% | +1.5 to +1.9% |
| | 10 | +2.8 to +5.9% | +10.6 to +11.3% |
| | 80 | +1.0 to +1.7% | +9.0 to +10.3% |
| 1 GiB copy | 2.5 | +1.3 to +1.9% | +3.6 to +4.0% |
| | 10 | +4.1 to +6.2% | +12.7 to +15.5% |
| | 80 | +4.3 to +4.6% | +5.6 to +6.1% |

3/8 is the default: 1/2 buys about 7% of relay time at 80 MiB, for 2-10 times
the GEMM slowdown on the relay GPUs at every size, and 2-4 times the copy
slowdown at 2.5 and 10 MiB (1.3 times at 80 MiB).

## Other measurements (3/8)

Two-pair contention harness, current kernel. This harness times the 80 MiB
relay at 311-313 µs, against 305-306 µs in the one-pair sweep above.

- **Direct all-reduce on the relay GPUs.** B's own direct all-reduce while A
  relays through it is unchanged at 2.5 and 80 MiB and about 1% slower at
  10 MiB.
- **Layer stand-in** (gate-up GEMM, down GEMM, 80 MiB all-reduce, o-proj GEMM,
  80 MiB all-reduce; 8192 tokens). Direct: 2780-2786 µs. Relaying through idle
  GPUs: 2473-2489 µs (0.89x the time). Both pairs relaying through each other:
  2482-2493 µs for A and 2466-2475 µs for B.
- **Memory.** At INT4 a rank allocates 13.6 MiB for relaying: two bounces on a
  relay GPU and the two relay engines' inboxes on its own GPU, half each. The
  first uncached allocation on a GPU also holds about 150 MiB of driver
  overhead that `hipFree` never returns.

## Not measured

- gfx942 has not been run. It takes the same code path and emits the same
  acquire, but whether its L1 keeps UC lines has not been probed, and the relay
  has no defaults there.
- The context cost of the relay on the relay GPUs.
- All four TP2 pairs of a node relaying at once.
- More than one relay per direction.

## Testing

`op_tests/multigpu_tests/test_flydsl_quick_allreduce.py --tp 2` runs the relay
rows on an xGMI host with four or more GPUs, through devices 2 and 3 when the
arch has relay defaults; on an xGMI arch without defaults, pass `--relay 2,3`.
The rows check bit identity against the direct engine, eager and under graph
replay, on the block-256 and block-512 rungs, the cutoff walk around 480 KiB,
and the block-64 reuse row. The wires default to INT4 and FP16; add
`--relay-codec int4 int5 int6 fp16` for all four.
