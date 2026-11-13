# Collective bandwidth roofline: design (v3, implemented)

Grades a FlyDSL collective, and RCCL doing the same operation, against
TransferBench rooflines. The op's roof is the fastest time the stock
TransferBench needs to move the op's bytes in the op's own pattern. RCCL gets
its own roof for its own (unquantized) bytes. Separately, every
implementation's per-rank wire rate is set against the fabric's per-rank peak
(section 5.1), which answers: given the link, the collective and the number
of ranks, what bandwidth per rank is possible, and how close does each
implementation get?

Changes from v2 (iteration 4), after v2 reported all-reduce eff up to 1.26
and no RCCL efficiency:
- **RCCL has its own roof.** v2 graded RCCL against the op's roof, which moves
  the op's INT4 wire bytes (0.28x of bf16), so RCCL looked 3-7x worse than it
  is. `rccl_traffic()` derives RCCL's bf16 data movement from `Traffic.rccl`
  alone; the report shows `rccl roof us` and `rccl eff`.
- **The roof is the best of a few candidates** (section 4), not one replay:
  the op's own grid and fixed copy widths (`WAVE_SWEEP`), lap after lap and,
  for ops that overlap their laps (`Traffic.pipelined`), fused into one test.
  The winner is shown in `roof cfg` / `rccl roof cfg`.
- **One pass per launch, timed inside the kernel** (section 4.2). Repeated
  sub-iterations let cacheable peer memory (`G`, `F`) absorb the writes in
  L2; one pass ends with a fence that writes them back, so the roof can use
  the op's own memory type and the RCCL roof can try every type.

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
module; the op's pattern is never changed; exact sizes only.

## 1. Architecture

```
 aiter/ops/flydsl/comm_traffic.py        Traffic, Phase, Pattern, PeerMem, RcclCall (no torch)
 AttentionA2AIntraNodeOp.traffic()       -> Traffic   (shape fixed per op)
 FlyQuickAllReduce.traffic(nbytes)       -> Traffic   (rung/codec/grid depend on size)

 op_tests/op_benchmarks/collective_bw/     (flat sibling modules, no package)
   measure.py   Timer protocol, FnTimer, Record, measure(), gather(), spawn()
   rccl.py      RcclBaseline: RcclCalls -> one thunk (torch.distributed or PyNccl)
   roof_settings.py  RoofSettings: every tunable TransferBench setting, one dataclass
   patterns.py  Traffic -> roof candidates (config lines + env); busbw factor (pure, no GPU)
   tb_text.py   find + run stock TransferBench, parse stdout (only TB-aware module)
   roofline.py  Roofline(binary, settings).measure(traffics) -> {traffic: Roof};
                measure_peaks(N) -> per-rank fabric peaks
   tb_dump.py   write every candidate as a runnable .cfg + README.md index
   report.py    grade(), markdown / CSV, shared CLI options, finish(), --tb-dry-run
   test_collective_bw.py       GPU-free tests (patterns, byte accounting, parser, grading)
   testdata/transferbench_v1.71.02.txt   captured output the parser is checked against
   bench_bw_attention_a2a.py   thin: reuses bench_attention_a2a.py's setup + timer
   bench_bw_quick_allreduce.py thin: reuses test_flydsl_quick_allreduce.py's setup + timer

 flow:  spawn ranks --> per rank: build op, timer = script's timer
                        measure(op thunk + RCCL thunk) --> gather (slowest rank)
        join (GPUs idle) --> roofline: each distinct TB test once --> configs + report
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
across the fabric, and the roof's in-kernel timing leaves launches out anyway.

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

A roof is the fastest of several *candidates*; a candidate is a list of
TransferBench tests whose times add up. In the op's own replay (candidate
`serial own`), each phase is one test, and the phase's waves are split evenly
over the transfers each GPU executes: `w = workgroups * waves_per_workgroup / T`,
where T is the number of transfers per GPU (N or N-1 for `*_ALL`, 1 for ring hops).

```
PUSH_ALL  rank i, each j (j != i unless include_self):  (G{i} G{i} {mem}{j} w chunk)
PULL_ALL  rank i, each j:                               ({mem}{j} G{i} G{i} w chunk)
PUSH_NEXT rank i:                                       (G{i} G{i} {mem}{i+1} w chunk)  x repeat
```

Every test carries its full environment (`patterns.env()`):
- fixed by the method (`patterns.FIXED_ENV`): `GFX_SE_TYPE=1` (sub-executor =
  one wave; the wave split depends on it) and `NUM_SUBITERATIONS=1` (section 4.2);
- from the op: `GFX_BLOCK_SIZE=64*waves_per_workgroup`;
- from `RoofSettings` (`roof_settings.py`): `NUM_ITERATIONS`, `NUM_WARMUPS` and
  further env (`USE_HIP_EVENTS=1`, `ALWAYS_VALIDATE=-1`). The settings also
  hold the width sweep, the RCCL block and memory types, and the timeout.
  `--tb-iters` / `--tb-warmup` override them.

`tb_text` adds only what its parser needs (`OUTPUT_TO_CSV=0`, `HIDE_ENV=1`).
Tests with different environments need separate invocations.

A test's time is its slowest transfer, timed inside the kernel (the
`Transfer` rows: first wave's start to last wave's stop, averaged over the
iterations). A candidate's time is the sum over its tests of `repeat` x that;
`roof_us` is the smallest candidate. Each distinct test runs once, however
many candidates share it.

### 4.1 Candidates

| knob | op roof | RCCL roof |
|---|---|---|
| bytes, pattern | the op's `Traffic` (wire format) | `rccl_traffic()`: bf16 all-pairs push. AR = RS + AG of `nbytes/N` per peer (`2(N-1)/N` x nbytes sent, the busbw convention); A2A = `nbytes/N` to every rank, self included, one phase per call |
| waves per transfer | `own` (as above) and `WAVE_SWEEP` = 16..256 | `WAVE_SWEEP` |
| block | the op's | 256 threads |
| schedule | `serial`; plus `fused` if `pipelined` | same (a single AR call is pipelined) |
| peer memory | the op's | `U`, `F`, `G` |

`fused` streams every phase's bytes in one test: transfers on the same link
and direction merge and carry the sum (mesh: RS + AG per peer; ring:
`(RS + AG) x (N-1)` on the next hop). It is the roof of a kernel that
overlaps its laps, as the quick all-reduce does tile by tile.

Measured on MI350X (8 GPUs, xGMI), INT4 quick AR, 16384x7168, roof in us
(event-timed, uncached, before section 4.2's change; the ranking holds):

| TP | v2 roof (serial own) | serial, best width | fused, best width | op |
|---|---:|---:|---:|---:|
| 2 | 1284 | 1191 (w32) | 1188 (w32) | 1276 |
| 4 | 852 | 650 (w16) | 666 (w256) | 674 |
| 8 | 422 | 339 (w64) | 341 (w64) | 408 |

The op's grid is the main error: an op wave also packs and reduces, so the
op runs far more waves than a copy kernel needs, and a copy kernel that wide
oversubscribes the links (TP4: 1365 waves per transfer is 30% slower than
16). Fusing matters for small messages (TP8, 64 tokens: 2.9 -> 1.7 us) and
not for large ones, where one lap already saturates the links.

### 4.2 One pass per launch, timed in the kernel

TransferBench's copy kernel loops over its sub-iterations with no fence in
between; each wave runs `__threadfence_system()` once, after the last pass,
then takes its stop timestamp. With `G` or `F` peer memory, once the buffer
fits in the L2 the writing waves span, every pass after the first rewrites
dirty lines in L2 and nothing crosses the link. The time then falls as
`1 / NUM_SUBITERATIONS`. Uncached (`U`) writes bypass L2 and always cross:

| TP2, 1024x7168 bf16 lap (7.3 MB), 64 waves, us | sub 10, event | sub 1, event | sub 1, in-kernel |
|---|---:|---:|---:|
| `G` | 28.4 | 150.5 | 149.3 |
| `F` | 28.7 | 150.2 | 149.2 |
| `U` | 132.0 | 135.6 | 134.3 |

One xGMI direction carries ~55 GB/s, so 7.3 MB cannot take under ~130 us.
TransferBench has no option to flush between sub-iterations, so every test
runs one pass per launch: the fence at its end writes the dirty lines back
before the stop timestamp, and the next launch starts clean. The HIP-event
executor time would then include the launch, so the roof reads the
in-kernel `Transfer` rows instead (`tb_text.TestResult.kernel_ms`).

Consequences:
- The roof writes peers' buffers in **the op's own memory type**, and the
  type's real cost shows. Into `G`/`F`, a 16384x7168 INT4 TP4 lap takes 398
  us vs 330 us into `U`, and an A2A TP8 role 51 vs 37 us.
- For large messages nothing else changes: `U` one-pass in-kernel matches
  `U` with many sub-iterations within 1-2%.
- For small messages the roof keeps the per-pass fixed cost: the wave ramp
  and the final fence, which an op also pays once per call, but not the host
  launch. A TP8 64-token bf16 lap takes 8.7 us in one pass vs 2.6 us when
  streamed. The roof is "one copy-kernel call", not steady-state streaming.
- A test's time is the slowest *transfer*, not the executor's span over all
  of its transfers, which TransferBench does not print in-kernel. The
  transfers of one GPU run in one launch and start together, so the
  difference is the start skew between them, and the roof errs fast.
- Pulls are left out of the RCCL roof: remote reads can be served from
  cache across launches (a 0.9 MB TP2 pull "ran" at ~620 GB/s with
  sub-iterations).

### 4.3 What the roof is not

It is the best of a stock copy kernel, not the link's peak. RCCL beats it
for large bf16 all-reduces (`rccl eff` 1.07-1.22 at 1024 and 16384 tokens).
No TransferBench executor gets past ~58 GB/s per xGMI direction on one link:
not GFX or SDMA, push or pull, any memory type, width, unroll, temporal hint
or block size, nor zero-filled data. RCCL sustains 64.8 GB/s over the same
link (TP2, 16384x7168), and PyNccl times it identically. Read `rccl eff > 1`
as "RCCL's copy loop beats TransferBench's", not as a bug. The candidates use
TransferBench's default kernel (`GFX_SE_TYPE=1`, default unroll); the best
knob settings found by hand are ~8% faster for large bf16, which would narrow
but not close the gap.

**Worked example.** Attention A2A, `wan2.2` bf16, TP8, xGMI. The Q phase is
`chunk = 12,124,160 / 8 = 1,515,520 B`, with 128 x 8 waves over 8 destinations,
so 128 waves each. GPU 0's part:

```
(G0 G0 U0 128 1515520) (G0 G0 U1 128 1515520) ... (G0 G0 U7 128 1515520)
```

Quick AR mesh, INT4, TP8: `serial` is two tests, RS and then AG, each
`(G{i} G{i} U{j} w c)` for j != i, summed; `fused` is one test of
`(G{i} G{i} U{j} w 2c)`. The kernel overlaps its laps (KB 60/02 section 10.3).

## 5. Metrics and report

From the slowest rank's `t`:

| column | formula |
|---|---|
| algbw | `logical_bytes / t` |
| busbw | `algbw * factor` (rccl-tests convention) |
| wire GB/s | bytes one rank sends to other GPUs / t: the op's wire format, RCCL's bf16 |
| wire/peak | `wire GB/s` / the all-peers fabric peak (section 5.1) |
| roof us | `roof_us(op traffic)`, blank if TransferBench is absent |
| op eff | `roof us / op us` |
| rccl roof us | `roof_us(rccl_traffic(op traffic))`: RCCL's bf16 bytes |
| rccl eff | `rccl roof us / rccl us` |
| roof cfg, rccl roof cfg | the winning candidate, e.g. `fused w64` |

The table has one row per (case, op):

`case | op | timing | roof us | op us | op eff | op busbw | op wire GB/s | op wire/peak | rccl roof us | rccl us | rccl eff | rccl busbw | rccl wire GB/s | rccl wire/peak | op spread us | variant | roof cfg | rccl roof cfg`

`eff` and `wire/peak` answer different questions. `eff` is implementation
efficiency: the op against a copy kernel making the *same moves* (its
pattern, bytes, memory). `wire/peak` is fabric use: how busy the op keeps
the rank's links, against one peak shared by every implementation. A ring op
can have a high `eff` while using one link of N - 1.

**TransferBench configs.** Every run, measured or not, writes every
candidate of every roof as a `.cfg` that runs as is with the stock binary
(`tb_dump.py`), into `--tb-dir` (default: `<report>.tb/` next to the report):

```
<report>.tb/README.md                       every candidate, its time, the roof marked
<report>.tb/<case>/<op>/<candidate>.cfg     e.g. 16384x7168/fly_mesh/fused_w64.cfg
<report>.tb/<case>/rccl/<candidate>.cfg     e.g. 16384x7168/rccl/serial_w64_U.cfg
```

Each file starts with `##` lines (TransferBench echoes them) naming the
candidate and its measured time, then the command with its environment, then
one config line per test with its `repeat`. A candidate whose tests need
different block sizes is split into `<candidate>.<i>.cfg`. A rerun clears a
directory it wrote (marker `.collective_bw_tb`), so no stale files remain.
`--tb-dry-run` writes the files without running TransferBench.

It is preceded by a provenance header (device, link, NUMA map, `HIP_VISIBLE_DEVICES`,
TB version and path, timing method, the peaks of section 5.1, command).

### 5.1 Per-rank bandwidth vs peak

The question: given the interconnect, the collective and N, what is the
bandwidth per rank at best, and how close does an implementation get?

- **Metric: wire bytes per rank per second.** The bytes one rank sends to
  other GPUs per call (`fabric_bytes_per_rank`) over the slowest rank's time.
  For an unquantized collective on a bandwidth-optimal schedule this is the
  rccl-tests busbw. Collectives are symmetric, so every rank's rate is the
  same; the slowest rank's time sets it.
- **Peak per rank, P(link, N).** Depends on the hardware and N only, never on
  an op. Measured on every run (`Roofline.measure_peaks`, configs in
  `<report>.tb/peak/`): every rank pushes `RoofSettings.peak_bytes_per_peer`
  (64 MiB, enough to saturate) at once to
  - `all_peers`: every other rank. The headline P: on an xGMI mesh this
    uses all N - 1 links of a rank.
  - `one_peer`: the next rank only. What a single ring can use.

  Both sweep widths and memory types (`patterns.sweep_candidates`).
- **Two more references.** `best known`: the highest per-rank wire rate in
  the run, P included; TransferBench is not a ceiling (section 4.3), RCCL
  beats it. `nominal`: an optional datasheet rate per link (`--link-gbps`,
  `RoofSettings.link_gbps`) times the links a rank uses (N - 1 on xGMI, 1
  on PCIe).
- **Max means the large-message plateau.** The summary takes each
  implementation's largest message and shows its `wire MB per rank`, so a
  shape that is too small to saturate is visible.
- **Quantized ops** get both: `wire/peak` (fabric use) and `busbw/peak`
  (logical rate, can exceed 1).

The report ends with one row per implementation:

`link | collective | ranks | impl | case | wire MB per rank | wire GB/s per rank | wire/peak | wire/best known | wire/nominal | busbw GB/s | busbw/peak`

Measured on MI350X, xGMI, 16384x7168 all-reduce (peaks repeat within 0.3%
across runs):

| N | P all peers (per link) | P one peer | FlyDSL INT4 wire, /P, /best | RCCL wire, /P | FlyDSL busbw/P |
|---|---:|---:|---:|---:|---:|
| 2 | 55.6 (55.6) | 55.6 | 51.7, 0.93, 0.80 | 64.8, 1.17 | 3.3 |
| 4 | 152.3 (50.8) | 57.5 | 147.0, 0.97, 0.79 | 185.7, 1.22 | 3.4 |
| 8 | 335.1 (47.9) | 56.5 | 282.0, 0.84, 0.67 | 420.5, 1.26 | 3.0 |

A link carries ~18% more alone than when all links of a rank are busy
(TP8: 56.5 vs 47.9 GB/s).

## 6. Status

1. **Done.**
   - `comm_traffic.py` and the two `traffic()` methods (no behavior change);
     the quick all-reduce declares `pipelined=True`.
   - roof_settings, patterns, tb_text, roofline, tb_dump, report; the configs
     of every roof are written on every run (`--tb-dir`).
   - `test_collective_bw.py`, 19 GPU-free tests:
     - fabric bytes equal `busbw factor * logical * wire ratio` for A2A, mesh and ring;
     - wave split and exact config text;
     - fused tests carry every lap's bytes; candidate sets for pipelined and
       non-pipelined traffic;
     - the RCCL traffic is bf16, op-independent and in the busbw convention;
     - the roofline picks the fastest candidate and runs each test once;
     - settings reach every test, and the CLI overrides them;
     - the config dump: one runnable file per candidate, the index, cleanup;
     - the peak probes, `measure_peaks`, and the per-rank summary;
     - the parser on captured v1.71.02 output (MI350P, PCIe);
     - the grading formulas.
   - Both thin scripts run at TP2/4/8 on MI350X. `torch.distributed`
     all-reduce captures under `fab.capture`: PyNccl gives the same time.
2. **Pending.**
   - Check `traffic()` on live ops against the existing scripts' wire-byte figures.
   - Optionally sweep TransferBench's kernel knobs (`GFX_SE_TYPE=0`, unroll)
     in the candidates: ~8% faster roofs for large bf16 (section 4.3).
3. **Later.** `OneShotAllReduce.traffic()`, `FlyQuickAllToAll`.

## 7. Decisions

- Ops carry a small `traffic()` description. The types live in aiter.
- TransferBench is optional at run time and parsed in its own module.
- Timing is injected by the scripts, and RCCL is timed with the same timer.
  The AR script falls back to PyNccl (`--rccl-backend pynccl`) if
  `torch.distributed` does not capture.
- The roof replays the op's pattern at exact sizes, and takes the fastest of
  the op's own grid and a few fixed copy widths, serial or (pipelined ops)
  fused. Only phases that move payload between GPUs count: no barrier, amax
  exchange or dequant.
- Roofs run one pass per launch and are timed in the kernel, so cacheable
  memory cannot absorb the writes in L2. The op's roof uses its own memory
  type; RCCL's tries all three.
- RCCL is graded against its own bf16 roof (`rccl eff`), which depends only
  on its calls, not on the op.
- Fabric use is graded against one per-rank peak per (link, N), measured by
  TransferBench on every run, next to the best rate seen and an optional
  datasheet rate; it is separate from `eff`.
- The tunable TransferBench settings live in one dataclass (`RoofSettings`);
  the config lines are generated from the op's `traffic()`, not stored,
  because they follow the op's runtime choices (size tier, codec, grid, inbox
  memory). Every run writes them out for review and reproduction.
- Attention A2A is graded per trio.
- Existing scripts stay unchanged. The thin scripts reuse their helpers and live
  in `op_tests/op_benchmarks/collective_bw/`.
