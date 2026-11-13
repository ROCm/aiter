# Collective bandwidth roofline: design (v2, implemented)

Grades a FlyDSL collective, and RCCL doing the same operation, against a
TransferBench roofline. The roofline is the time the stock TransferBench needs to
move the same bytes, in the op's own pattern, into the op's own memory type,
with the op's own launch geometry.

Changes from v1 (iteration 3):
- **The ops describe themselves.** A small `traffic()` method on
  `AttentionA2AIntraNodeOp` and on `FlyQuickAllReduce`. The description types
  live in aiter, so the ops do not import from `op_tests`.
- **The all-reduce op is in scope:** `FlyQuickAllReduce`, both mesh and ring.
- **Timing is injected.** The scripts pass their own timer into the tool. The
  tool times RCCL with *that same timer*, so eager vs graph matches by
  construction.
- **RCCL rows get `op roof / rccl us`.**
- **Attention A2A is graded per Q/K/V trio only.**

Unchanged: TransferBench is an optional runtime binary, parsed in its own
module; there is no optimization over patterns, memory or grid; exact sizes only.

## 1. Architecture

```
 aiter/ops/flydsl/comm_traffic.py        Traffic, Phase, Pattern, PeerMem, RcclCall (no torch)
 AttentionA2AIntraNodeOp.traffic()       -> Traffic   (shape fixed per op)
 FlyQuickAllReduce.traffic(nbytes)       -> Traffic   (rung/codec/grid depend on size)

 op_tests/op_benchmarks/collective_bw/     (flat sibling modules, no package)
   measure.py   Timer protocol, FnTimer, Record, measure(), gather(), spawn()
   rccl.py      RcclBaseline: RcclCalls -> one thunk (torch.distributed or PyNccl)
   patterns.py  Traffic -> TransferBench config lines + env; busbw factor (pure, no GPU)
   tb_text.py   find + run stock TransferBench, parse stdout (only TB-aware module)
   roofline.py  Roofline.measure(traffics) -> {traffic: Roof}
   report.py    grade(), markdown / CSV, shared CLI options, finish(), --tb-dry-run
   test_collective_bw.py       GPU-free tests (patterns, byte accounting, parser, grading)
   testdata/transferbench_v1.71.02.txt   captured output the parser is checked against
   bench_bw_attention_a2a.py   thin: reuses bench_attention_a2a.py's setup + timer
   bench_bw_quick_allreduce.py thin: reuses test_flydsl_quick_allreduce.py's setup + timer

 flow:  spawn ranks --> per rank: build op, timer = script's timer
                        measure(op thunk + RCCL thunk) --> gather (slowest rank)
        join (GPUs idle) --> roofline: one TB run per distinct Traffic --> report
```

## 2. The op's self-description (in aiter)

```python
class PeerMem(str, Enum):          # TransferBench memory letters
    COARSE = "G"; FINE = "F"; UNCACHED = "U"

class Pattern(str, Enum):          # who moves bytes to whom; not collective semantics
    PUSH_ALL  = "push_all"         # rank i writes a chunk into every rank j's buffer
    PULL_ALL  = "pull_all"         # rank i reads a chunk from every rank j's buffer
    PUSH_NEXT = "push_next"        # rank i writes a chunk into rank (i+1) % N  (ring hop)
    PULL_PREV = "pull_prev"

@dataclass(frozen=True)
class Phase:                       # one lap of peer traffic; phases run back to back
    pattern: Pattern
    chunk_bytes: int               # bytes per (src, dst) pair, in the wire format
    include_self: bool             # also copy the own chunk (A2A does, two-shot laps do not)
    repeat: int                    # identical steps in a row (ring lap: N-1)
    peer_mem: PeerMem
    workgroups: int                # grid of the kernel running this phase
    waves_per_workgroup: int       # threads / 64

@dataclass(frozen=True)
class RcclCall:                    # the equivalent RCCL work, for the baseline
    collective: str                # "all_reduce" | "all_to_all"
    nbytes: int                    # per rank
    dtype: str                     # "bf16"

@dataclass(frozen=True)
class Traffic:
    world_size: int
    collective: str                # busbw factor: A2A (N-1)/N, AR 2(N-1)/N
    logical_bytes: int             # per-rank payload in the caller's dtype
    phases: tuple[Phase, ...]
    rccl: tuple[RcclCall, ...]
```

Only phases that move payload between GPUs are listed. The reuse barrier, the
amax pre-exchange and the local dequant are left out: they carry no payload
across the fabric, and the roof amortizes launches anyway.

### 2.1 `AttentionA2AIntraNodeOp.traffic()`: the trio

| field | value | source in the op |
|---|---|---|
| phases | Q, K, V: `PUSH_ALL`, `include_self=True`, `repeat=1` | the kernel stores into `p2p_outputs_sets` (peer pointers) |
| chunk_bytes | (role payload + role scale bytes) / N | `outputs_sets[0][role]`, scale storage |
| peer_mem | `inbox_memory` (`uncached` -> U, `finegrained` -> F) | `resolve_inbox_flags` |
| workgroups / waves | `block_num` / per-role warp count (8, or 4 for packed V and amax roles) | `__init__`, already computed as `role_warp_counts` |
| rccl | 3 x `all_to_all(local tensor bytes, bf16)` | `self.shape` |
| logical_bytes | 3 x local tensor bytes | |

### 2.2 `FlyQuickAllReduce.traffic(nbytes)`: two laps in one kernel

The codec costs `(bits + 0.5) / 16` of bf16: E5M2 scale per 16 elements, so INT4 is
~0.28x and INT6 ~0.41x. The rung, `grid_x` and codecs come from the same
`_pick_cfg` / `_grid_x` that `allreduce()` and `variant()` already use.

| algorithm | phases |
|---|---|
| mesh | RS: `PUSH_ALL`, chunk = wire(rs) / N, no self; then AG: `PUSH_ALL`, chunk = wire(ag) / N, no self |
| ring | RS: `PUSH_NEXT`, chunk = wire(rs) / N, repeat N-1; then AG: same with wire(ag) |

Both phases use `workgroups = grid_x`, `waves = rung.block / 64` and
`peer_mem = inbox_memory`. `rccl` = 1 x `all_reduce(nbytes, bf16)`.

A later `OneShotAllReduce.traffic(nbytes)` would be a single `PUSH_ALL` phase
with chunk = nbytes (each rank pushes its whole buffer to every peer). No new
pattern is needed.

## 3. Injected timing

```python
class Timer(Protocol):
    method: str                                    # "eager" | "graph"
    def __call__(self, thunks: dict[str, Callable[[], None]]) -> dict[str, float]:
        """Microseconds per call on *this* rank, for every thunk. Every rank
        passes the same names in the same order (thunks are collectives)."""

def measure(case: str, op_name: str, op_thunk, traffic: Traffic, timer: Timer,
            *, rccl_group, device, rotations: int = 1) -> list[Record]:
    """Adds the RCCL thunk built from traffic.rccl and times both with ONE call
    to timer, so they share rounds, rotation and method. Returns this rank's records."""

def gather(records, cpu_group) -> list[Record] | None:   # rank 0: slowest rank per (case, impl)
```

The two scripts' timers, wrapped in the thin scripts (no edits to the originals):

| script | method | adapter |
|---|---|---|
| `test_flydsl_quick_allreduce.py` | graph | `fab.time_graphs({n: fab.capture(t) for n, t in thunks.items()}, gloo, reps=, rounds=)`. The same calls its `_bench_rank` makes, and the same dict-in / dict-out shape. |
| `bench_attention_a2a.py` | eager | `_time_roles(run_trio=lambda i: thunk(), run_role=<record-only no-op>, warmup, iterations, group)["trio"]`, per thunk |

Input rotation (the attention script cycles 4 input sets) stays inside each
thunk, as a counter in the closure. The RCCL thunk takes `rotations=` to match.

RCCL thunks use plain `torch.distributed` on the script's device group:
`all_reduce` in place on a scratch buffer, and `all_to_all_single` per call
(three in a row for the trio).

## 4. Patterns: `Traffic` -> TransferBench

Each phase is one TransferBench test. The phase's waves are split evenly over the
transfers each GPU executes: `w = workgroups * waves_per_workgroup / T`, where T is
the number of transfers per GPU (N or N-1 for `*_ALL`, 1 for ring hops).

```
PUSH_ALL  rank i, each j (j != i unless include_self):  (G{i} G{i} {mem}{j} w chunk)
PULL_ALL  rank i, each j:                               ({mem}{j} G{i} G{i} w chunk)
PUSH_NEXT rank i:                                       (G{i} G{i} {mem}{i+1} w chunk)  x repeat
```

The environment is set per invocation: `GFX_SE_TYPE=1` (sub-executor = one wave),
`GFX_BLOCK_SIZE=64*waves_per_workgroup`, `USE_SINGLE_STREAM=1`,
`NUM_SUBITERATIONS=k` (amortizes the launch), `USE_HIP_EVENTS=1` and
`ALWAYS_VALIDATE=-1`. Phases with different block sizes need separate invocations.

`roof_us(traffic)` is the sum over phases of `repeat * slowest executor time per sub-iteration`.

**Worked example.** Attention A2A, `wan2.2` bf16, TP8, xGMI. The Q phase is
`chunk = 12,124,160 / 8 = 1,515,520 B`, with 128 x 8 waves over 8 destinations,
so 128 waves each. GPU 0's part:

```
(G0 G0 U0 128 1515520) (G0 G0 U1 128 1515520) ... (G0 G0 U7 128 1515520)
```

Quick AR mesh, INT4, TP8: two tests, RS and then AG, each `(G{i} G{i} U{j} w c)`
for j != i. They are summed, which makes it a soft ceiling because the kernel can
overlap its laps (KB 60/02 section 10.3).

## 5. Metrics and report

From the slowest rank's `t`:

| column | formula |
|---|---|
| algbw | `logical_bytes / t` |
| busbw | `algbw * factor` (rccl-tests convention) |
| wire GB/s | fabric bytes / t (op rows only) |
| roof us | `roof_us(op traffic)`, blank if TransferBench is absent |
| op eff | `roof us / op us` |
| rccl vs op roof | `roof us / rccl us`. Same bytes and fabric, but RCCL uses its own pattern and memory |

The table has one row per case:

`case | timing | roof us | op us | op eff | op busbw | op wire GB/s | rccl us | rccl busbw | rccl vs op roof`

It is preceded by a provenance header (device, link, NUMA map, `HIP_VISIBLE_DEVICES`,
TB version and path, timing method, command).

## 6. Status

1. **Done, no GPU needed.**
   - `comm_traffic.py` and the two `traffic()` methods (no behavior change).
   - patterns, tb_text, roofline, report (`--tb-dry-run` prints the configs).
   - `test_collective_bw.py`, 8 tests:
     - fabric bytes equal `busbw factor * logical * wire ratio` for A2A, mesh and ring;
     - wave split and exact config text;
     - the parser on captured v1.71.02 output (MI350P, PCIe);
     - the grading formulas.
2. **Pending GPU.**
   - Run both thin scripts at TP4 (GPUs 4-7).
   - Check that `torch.distributed` all-reduce captures under `fab.capture`. If
     not, make `--rccl-backend pynccl` the default for the AR script.
   - Check `traffic()` on live ops against the existing scripts' wire-byte figures.
3. **Later.** `OneShotAllReduce.traffic()`, `FlyQuickAllToAll`.

## 7. Decisions

- Ops carry a small `traffic()` description. The types live in aiter.
- TransferBench is optional at run time and parsed in its own module.
- Timing is injected by the scripts, and RCCL is timed with the same timer.
  The AR script falls back to PyNccl (`--rccl-backend pynccl`) if
  `torch.distributed` does not capture.
- The roof replays the op's pattern, memory type and grid, at exact sizes.
  Only phases that move payload between GPUs count: no barrier, amax exchange or
  dequant.
- The two laps of a two-shot all-reduce are timed as separate TransferBench
  tests and summed, so the roof is a soft ceiling where the kernel overlaps them.
- RCCL rows show `op roof / rccl us`.
- Attention A2A is graded per trio.
- Existing scripts stay unchanged. The thin scripts reuse their helpers and live
  in `op_tests/op_benchmarks/collective_bw/`.
