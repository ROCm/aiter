# GDR MTP tiling: a ladder on the grid instead of a doubling loop

## What this changes

There are no gfx942 MTP rows in `gdr_decode_tuned.csv`, so what runs on that
part today is `_mtp_tiling` alone: a loop that doubles `NUM_BLOCKS_PER_V_DIM`
while `batch * num_v_heads * num_blocks` is under four blocks per CU. Measured
against the upstream each contract is defined by, that loop is **slower than
upstream on 18 of 23 shapes** and 2.4% slower in geomean -- the FlyDSL kernel is
faster than upstream on every one of those shapes under some tiling, and the
rule was picking the wrong one.

`_mtp_rung` replaces the loop. Nothing else moves: the table still wins wherever
it has a row, and the 3707 gfx950 rows are untouched.

## The rule

Tilings are `(blocks, warps, K group, waves per EU)`:

```python
_MTP_SPLIT_4, _MTP_SPLIT_2, _MTP_WHOLE = (4, 4, 8, 0), (2, 4, 8, 0), (1, 4, 8, 0)
_MTP_WIDE_K, _MTP_THIN = (2, 4, 16, 0), (4, 1, 8, 3)

# grid coverage in blocks per CU, up to which splitting still fills the grid
_MTP_FILL = ((0.75, _MTP_SPLIT_4), (1, _MTP_SPLIT_2), (2, _MTP_WHOLE))

for coverage, tiling in _MTP_FILL:
    if grid <= coverage * num_sms:
        return tiling
if grid <= 8 * num_sms:        # covered; the block's own shape decides
    return _MTP_SPLIT_2 if variant == MTP_MODE_SNAPSHOT else _MTP_THIN
return _MTP_WIDE_K if seq_length <= 2 else _MTP_THIN
```

Verify runs at the batch with draft tokens outstanding, which is small by
construction, so the grid is short and the split is what fills it. The loop it
replaces had the right idea and the wrong target: at four blocks per CU it is
still splitting where the grid already covers the part, and carrying the split
through the grids just past coverage is the expensive mistake: over the 14 cells
at one to two blocks per CU, holding the four-way split there instead of
dropping to one costs 20% in geomean and up to 38%. That is why the ladder comes
back **down** to one split at one to two blocks per CU, which is the shape of it
that is easy to get wrong.

Past coverage the tilings land within a few percent of each other and what is
left is how a block spends itself, not how many there are. Two branches carry
that, and both were kept only because they survived cross-validation:

- **The contract, between 2 and 8 blocks per CU.** The tree reads a parent for
  every token and vLLM's chain rolls the state back by the accepted count; both
  do more per token than SGLang's chain, and both want a single warp over a
  quarter of the value dimension. Over the 35 cells in that band, `(2,4,8)` is
  1.11x of oracle in geomean for the tree contract and `(4,1,8)` is 1.01x;
  for SGLang's chain the two swap, at 1.01x and 1.09x.
- **The draft length, past 8 blocks per CU.** A draft longer than a pair wants
  the same narrow shape; a pair wants `(2,4,16)`.

`WAVES_PER_EU=3` rides with the single-warp shape and only with it. That block
is 64 threads and thin enough to want the hint: across the two bands where the
rule picks the shape, dropping the hint costs about 4% in geomean and up to 33%
on the worst cell. It is not a default the rule sets anywhere else, and it does
not travel. On the four-warp shape at low grid it is worth nothing either way
(geomean 1.000 over 24 cells), and on the single-warp shape at low grid -- where
the rule does not pick it -- setting it is about 5% slower than leaving it off.

The grid is otherwise the whole of the key. Launches reaching the same grid by
trading batch against heads measure the same -- checked at grid 128 and 256
across four decompositions each -- so the rule reads `batch * num_v_heads` and
never either factor.

## What it measures on gfx942

23 shapes, both SGLang contracts, bf16 state, batch 32 and 128, draft 2 and 4,
all three arms interleaved in one process and rotated so none of them always
runs first:

| | geomean | worst cell | cells slower than upstream |
|---|---|---|---|
| new rule vs the loop that ships | **0.8589x** | 0.954x | — |
| new rule vs upstream | 0.8794x | 1.000x | **1 / 23** |
| the shipping loop vs upstream | 1.0238x | 1.087x | 18 / 23 |
| a table of per-shape winners vs upstream | 0.8806x | 0.968x | 0 / 23 |

The rule is faster than the loop on every one of the 23. The single cell that
does not clear upstream, `sglang_chain b32 sq4 hv32`, is at 1.0002x, which is a
tie inside the run-to-run spread rather than a regression.

That table is 23 rows tuned on this part, which is the best a table could do
here. Against it the rule comes in at **0.9986x**, worst cell 1.033x. It is not
close to the table; it is slightly ahead of it, which is why no gfx942 rows are
added.

Scored offline against the best tiling per cell, the rule is 1.0119x of oracle
over all 86 cells swept. The loop it replaces reaches `nb=8` on nine of them,
which the sweep excludes for the reason below; over the 77 that can be scored
both ways the rule is 1.0120x and the loop 1.0871x.

## How the ladder was chosen

Hand-fitting overfits a surface this small, so the bands and branches come from
a search over rule *structures* -- grid band boundaries in units of CUs, crossed
with whether each band branches on the contract, the draft length or the state
dtype -- ranked by 5-fold cross-validation over 8 shuffles, so no structure is
scored on the cells that chose its tilings. Fitting inflates the result by about
0.6%: the best structure is 1.0151x of oracle in sample and 1.0210x
cross-validated.

What the search rejected is as much of the result as what it kept. The **state
dtype** never earns a branch -- no structure in the top fifteen splits on it,
and fp32 cells follow the same ladder as bf16 -- and neither does every band,
since only the two branches above survive being scored out of sample.

The shipped ladder is not the first structure by cross-validation. It is 0.3%
behind one banding at (1, 2, 3, 4) CUs, which is inside what this sweep
resolves, and it is the one carried forward because it is the one measured
end-to-end against upstream and the table above.

The rule is fixed, so cells measured after it was settled are a real holdout: it
scores 1.0090x of oracle on the 37 cells the structure was chosen from and
1.0140x on the 49 measured afterwards.

## Open: does the same hold on gfx950?

Unknown. gfx950 keeps its rows, so nothing there moves either way, but the rule
runs wherever a row is missing and the constants in it are gfx942 measurements.
`get_num_sms()` reads the CU count off the device so the band edges scale on
their own; what does not scale is the ladder's shape, the two branches, and
`WAVES_PER_EU=3`.

```
python op_tests/bench_flydsl_gdr_mtp_tiling.py --table <csv-with-gfx950-rows> --repeats 3
```

Both arms run against the same tensors, alternating which goes first. Read the
result as:

1. `rule / upstream` with any cell over 1.0 is the failure that matters. The
   rows exist to keep the kernel ahead of upstream; a rule that is 2% off the
   table but still ahead of upstream everywhere has not cost anything.
2. If the rule loses, the printed configs say where. A first element that
   disagrees is the ladder or a band edge; a `waves_per_eu` that disagrees is
   the occupancy hint, which is the entry most likely to be arch-shaped.

## Measuring this at all

Two things will produce confident nonsense on this kernel:

- **`NUM_BLOCKS_PER_V_DIM=8` compiles bimodally.** Six processes on one GPU and
  one config gave 13.4us once and 17.8-18.0us five times, about 30% apart, and
  which mode a process lands in does not appear to depend on anything the caller
  controls. Every sweep behind this document is `nb <= 4`, where repeat runs
  agree to about 1%.
- **The GPUs on one node are not interchangeable.** GPU 0 measured about 28%
  slower than GPUs 4 and 6 on identical work, and config rankings flip between
  them. Everything here is one GPU, sequential.

`test_gdr_mtp_perf` is built to produce a comparison row, so per call it rebuilds
the problem, runs a torch reference, checks every candidate against it, and times
every candidate. A tiling sweep wants one candidate and one problem. Keeping its
timing -- profiler device time, `num_rotate_args=1`, an iteration count adapted
to a sample budget -- and dropping the rest is worth about 40x per config, which
is what made a search over structures affordable. Do not replace the timing
itself with a wall-clock loop: at batch 32 with four heads the kernel is 20us and
the python around it is 40, so the GPU starves and every tiling measures the
same. Pre-allocate the intermediate buffer too, or the runner allocates and zeroes
up to a gigabyte inside the timed window and flattens the ranking the same way.
