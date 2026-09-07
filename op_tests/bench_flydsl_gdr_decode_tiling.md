# GDR decode tiling: a host rule instead of a table keyed on batch

## What this changes

The gated delta rule decode path used to take its tiling from a flat default
of `(NUM_BLOCKS_PER_V_DIM, NUM_WARPS, WARP_THREADS_K) = (1, 4, 8)`, overridden
by `gdr_decode_tuned.csv`, which carried a row per batch size: 1792 gfx942
decode rows, 14 shapes times every batch from 1 to 128. `_decode_tiling` now
derives the same three numbers on the host, and those rows are dropped.

The 3584 gfx950 decode rows -- the same 14 shapes, batch 1 to 256 -- are
untouched, so nothing about that part moves yet. That is the open question
below.

## The rule

Decode launches one block per (batch, v-head) pair. Below some batch that grid
does not cover the machine, and splitting the value dimension is the only way
to make more blocks; each split costs a reduction across the blocks sharing a
head, so it stops paying once the grid is covered. The whole decision is
therefore a function of `batch_size * num_v_heads` against a target grid, and
the warp shape follows from the value tile the split leaves:

```python
target = get_num_sms() * (2 if f32_state else 1)
for num_blocks in ((8, 2, 1) if f32_state else (8, 4, 2, 1)):
    if head_v_dim % num_blocks or batch_size * num_v_heads * num_blocks > target:
        continue
    ...
```

Three things in it were measured rather than reasoned, and are the parts most
likely to be gfx942-shaped:

- **`_DECODE_GRID_PER_CU`, 1 block/CU for a bf16 state and 2 for fp32.** The
  fp32 state moves twice the bytes per block, so it stays latency-bound one
  doubling longer.
- **The split ladder.** With an fp32 state the 32-wide value tile is beaten at
  every grid size by either the 16-wide tile above it or the 64-wide one below,
  so the fp32 ladder steps from 8 straight to 2 and never lands on 4. This is
  the one non-obvious entry; a plain doubling loop picks `nb=4` at
  `batch*num_v_heads == 32` and gives up 0.5-0.9% on six cells.
- **`_DECODE_WARP_SHAPE`.** A 64-wide-or-more tile wants four warps over a
  narrow K group; from 32 down there is not enough value width for four warps
  and the coverage has to come from a wider K group instead.

## What it measures on gfx942

`bench_flydsl_gdr_decode_tiling.py --repeats 5`, 112 cells, rule against the
table it replaces:

| | geomean | worst | best | below 0.995 |
|---|---|---|---|---|
| all 112 cells | 1.0022x | 0.9795x | 1.0278x | 5 |
| the 57 where the two arms picked different configs | 1.0052x | 0.9948x | 1.0278x | 1 |
| the 55 where they picked the same config | 0.9991x | 0.9795x | 1.0029x | 4 |

The last row is the point: on 55 cells both arms compile the same kernel, and
they still spread 0.9795 to 1.0029. That is the floor. Four of the five cells
under 0.995 are in that row, and the fifth is the 0.9948 worst case of the row
above -- so nothing in the sweep is below the floor for a reason the rule is
responsible for. Scored offline against the best config
per cell rather than against the table, the rule reaches 0.9975x of oracle and
the table 0.9936x -- the table is the one further from optimal, because it was
tuned per batch on a surface that is not per batch.

## Open: does the same hold on gfx950?

Unknown, and the reason the gfx950 rows are still there. What has to be
re-measured is only the three items above; `get_num_sms()` already reads the CU
count off the device, so the target scales on its own and is not what is in
question. The ladder and the warp shape table are.

```
python op_tests/bench_flydsl_gdr_decode_tiling.py --repeats 5 --check --out /tmp/ab.json
```

It reads the tuned table for the running part, so on a gfx950 box the two arms
become "the 3584 gfx950 decode rows" against "the rule", over the shapes those
rows cover, with no edit to the CSV. Read the result as:

1. Compare the diverging cells only. The same-config cells are the noise floor
   and belong in the denominator of nothing.
2. If the rule loses on diverging cells, the printed table-vs-rule configs say
   which of the three items is wrong. A disagreement in the first element is
   the target or the ladder; a disagreement in the other two is the warp shape.
3. Both are keyed structures already, so a gfx950 answer that differs is a key
   on the arch, not a fork in the logic.

**Do not** conclude anything from `test_flydsl_linear_attention.py` run twice
in two processes. Two runs of that sweep disagree by about 10% at batch 1,
which is an order of magnitude more than anything being decided here. The
harness above interleaves the arms in one process against the same tensors to
get under that, and it is still only good to about half a percent.

## The MTP path, which moves the same way

Decode splits to cover a grid, and so does verify -- the MTP tiling is the same
kind of decision and gets the same treatment, in
[`bench_flydsl_gdr_mtp_tiling.md`](bench_flydsl_gdr_mtp_tiling.md). The two
share `_tile_warps`, which is why it is a helper rather than inline in either.

They differ in where the answer stops being the split. Decode's is a split all
the way out. Verify's stops at coverage and then turns on how a block spends its
lanes, which is a branch on the contract and the draft length that decode has no
equivalent of.
