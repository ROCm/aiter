# FA4 ATT overlap analysis (headless)

Headless equivalent of [ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer):
parse rocprofv3 Advanced-Thread-Trace (`ui_output_*`) data, map per-wave
instruction latencies back to C++ source and FA4 pipeline **phases**
(MATRIX / SOFTMAX / LDS / LOAD / ADDR / BARRIER), and plot two co-resident
warp-group waves on a shared cycle axis to expose pipeline overlap & imbalance.

## One command

```bash
# rebuild with line tables (after code changes) + trace + report:
LINETABLES=1 ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000

# subsequent traces reuse the line-tables build:
ua-test-scripts/att_analysis/run.sh fp8 16 10000 10000 0 3   # simd 0, 3 iters
```

Outputs land in `rocprof_analysis/runs/<TAG>/att_analysis/`:
`overlap_simd0.png` (timeline) and `report.md` (cycle rollups + metrics).

## Why the line-tables build

The deployed JIT `.so` is built without debug info, so the trace's `Source`
column is empty (ISA only). `LINETABLES=1` injects `-gline-tables-only`
(via the `AITER_EXTRA_HIP_FLAGS` hook in `aiter/jit/core.py`) which embeds DWARF
line tables **without changing `-O3` codegen**, so the trace stays
representative *and* every instruction carries `file:line` -> phase.

Without it, phase tagging falls back to a coarse ISA-mnemonic heuristic.

## Pieces

| module | role |
|---|---|
| `model.py`     | load `filenames/code/se*_sm*_sl*_wv*` ; find co-resident slot0/slot1 waves |
| `phases.py`    | classify each instruction (source file + auto-parsed pipeline lambda ranges + mnemonic) |
| `window.py`    | pick a few steady-state loop iterations via barrier cadence |
| `timeline.py`  | render two waves x (phase row, state row) to PNG |
| `aggregate.py` | per-wave / per-phase cycle rollups + MATRIX//SOFTMAX overlap metric |
| `report.py`    | timeline + Markdown report bundle |
| `run.sh`       | rebuild(optional) -> collect -> report |

## Direct use on an existing run

```bash
cd ua-test-scripts
python3 -m att_analysis.timeline rocprof_analysis/runs/<TAG> --simd 0 --iters 3
python3 -m att_analysis.report   rocprof_analysis/runs/<TAG> --simd 0 --iters 3
```

## Reading the timeline

Two waves co-resident on one SIMD (one per warp group on the FA4 ping-pong),
each drawn as two rows:
* **phase**  — what work the issued instruction is (matrix=blue, softmax=orange,
  lds=cyan, load=brown, addr=purple, barrier=red, memwait=gray).
* **state**  — what the hardware did (exec=green, stall=orange, wait=gray, idle).

Good pipelining = one wave in MATRIX while the other is in SOFTMAX. Both in
BARRIER/wait at the same cycle = the warp groups are synchronizing instead of
overlapping (a bubble). The report's *MATRIX//SOFTMAX overlap %* quantifies this.
