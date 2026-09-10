# All-reduce dispatch sweep shapes

Shape files for `op_tests/multigpu_tests/bench_comm_allreduce.py --shape-csv`.
Columns are `M,K,label`; the benchmark reads `M` and `K` and ignores the rest.

They live here rather than under `op_tests/dump_data/` with the reports they
produce because `dump_data/` is gitignored. A sweep *result* is an artifact; a
sweep *specification* is the thing that makes the result reproducible, and it
has to survive a clone.

They exist because the default `L_SHAPE` list cannot answer the question the
FlyDSL dispatch heuristics need answered. It jumps 168 KiB → 1.75 MiB → 14 MiB,
and both family crossovers live in those gaps: at TP4 the one-shot→mesh
crossover is bracketed only to 70–84 KiB, and the mesh→ring crossover only to
the octave and a half between 1.75 and 14 MiB, where the two kernels differ by
30%.

| file | shapes | payload range | what it is for |
|---|---|---|---|
| `ar_sweep_a_small.csv` | 32 | 14 KiB – 3.5 MiB | the one-shot→mesh crossover, and the low end of mesh→ring |
| `ar_sweep_a_large.csv`  | 21 | 3.5 MiB – 112 MiB | the mesh→ring crossover and the ring's own super-tile ladder |
| `ar_sweep_b_ksens.csv`  | 24 | 32 KiB – 16 MiB | whether the thresholds are keyed on bytes or on shape |

## Phase A — the crossover ladder

`K = 7168` (DeepSeek-V4 hidden). `M = 1..16`, then geometric to 8192 at four
steps per octave. Below M=16 the byte ladder is quantised to whole tokens
anyway — one token is 14 KiB — so consecutive integers *are* the finest ladder
available there.

Four steps per octave brackets any crossover to within 19% in payload size,
which is inside the ~10% latency tolerance the heuristics are allowed. The two
files overlap at M=256 (3.5 MiB) so the halves can be spliced without a gap;
they are split because a single run would hold every candidate's buffers for
both the 14 KiB and the 112 MiB shape.

## Phase B — K-sensitivity

The same eight byte targets (32/64/128/256 KiB, 1/2/8/16 MiB) at K ∈ {4096,
7168, 8192}. If a threshold fitted at K=7168 does not hold at the other two,
then payload bytes is the wrong key for the table and that has to surface before
anything is fitted.

`M` is rounded per K, so byte counts match across hidden sizes to within one
token rather than exactly — 7168 does not divide a power of two. The `label`
column records both the target and what the row actually is.

## Running them

FlyDSL candidates only; the one-shot's 192 KiB policy ceiling is lifted so its
rows survive past every plausible crossover — at TP2 its wire volume `(N-1)·S`
*equals* the mesh's `2(N-1)/N·S`, so there is no structural reason for it to
lose and no measurement yet showing it does.

```bash
FLY="fly_int4 fly_int4_st1 fly_int4_g128
     fly_int4_ring fly_int4_ring_st8 fly_int4_ring_st16 fly_int4_ring_st32
     fly_int4_ring_st8_int6 fly_int4_ring_st32_int6
     fly_1stage fly_1stage_a2 fly_1stage_a4 fly_1stage_fa
     fly_1stage_a2_fa fly_1stage_a4_fa fly_1stage_g128 fly_1stage_a4_g128"

for tp in 2 4 8; do
  AITER_BENCH_FLY1S_MAX_KB=8192 \
  python3 op_tests/multigpu_tests/bench_comm_allreduce.py -tp $tp -c $FLY \
    --shape-csv op_tests/multigpu_tests/shapes/ar_sweep_a_small.csv \
    -o   op_tests/dump_data/ar_sweep_a_small_tp$tp.md \
    --output-csv op_tests/dump_data/ar_sweep_a_small_tp$tp.csv
done
```

`--output-csv` is the one the fitting reads: it carries `_nbytes` and every
candidate's `median us` (noise-robust, for threshold fitting) next to `us` (max
across ranks, the metric the model actually waits on).
