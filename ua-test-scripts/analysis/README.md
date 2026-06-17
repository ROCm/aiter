# UA kernel — deep-dive analysis tooling

Profiling / tracing / ISA-inspection tooling for the CK `unified_attention` (UA)
kernel. **None of this is needed for correctness or the perf-vs-Triton numbers**
— those live one level up (`../README.md`). Reach for this folder only when you
need to understand *why* the kernel performs the way it does (pipeline overlap,
VGPR pressure, memory stalls, ISA).

All commands are run from the **aiter repo root** unless noted. Pick an idle GPU
with `rocm-smi --showuse` and pass its index as `GPU=` / `HIP_VISIBLE_DEVICES=`.

---

## ⚠️ Always trace/measure a FRESH build

Every perf or trace number is worthless if it came from a stale binary. There are
two independent build products and each has a freshness rule:

1. **The JIT module** `aiter/jit/module_unified_attention.so` (what the PyTorch
   harness and `sweep_amir_shapes.py` use). It is **NOT** automatically rebuilt
   when you edit kernel source. To force a clean rebuild:
   ```bash
   rm -rf aiter/jit/build/module_unified_attention aiter/jit/module_unified_attention.so
   AITER_REBUILD=1 HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py ...
   ```
   (`../rebuild_and_test.sh` does exactly this and verifies the `.so` reappeared.)

2. **The standalone driver** `analysis/standalone/build/ua_trace`. This one is
   **self-guarding**: `standalone/build.sh` rebuilds automatically iff `REBUILD=1`,
   the build stamp (`arch,dtype,d,mask,XCFLAGS`) changed, or **any** ck_tile /
   UA-example / driver source is newer than the exe. So `perf.sh` / `check.sh` /
   `run.sh` can be called unconditionally and never measure a stale kernel. Force
   a rebuild with `REBUILD=1`.

When in doubt, prefer the standalone path — it removes all "is the JIT stale?"
ambiguity by stamping every build.

---

## Standalone driver (`standalone/`) — JIT-free perf, accuracy & trace

Compiles **only** the one instance under test (every other UA instance becomes a
`-DUA_STUB_INSTANCE` host stub), so `rocprofv3 --att` disassembles ~1 kernel
instead of PyTorch's 26 MB HIP lib → trace collection drops from ~2.5 min to
seconds. Instance is chosen at compile time by `(dtype, d, mask)`; the shape
(`sq/hq/hk`) is a runtime arg, so one build sweeps all sequence lengths.

```bash
# Perf (one shape, or a sweep). DTYPE=fp8|bf16, MASK 0=non-causal 2=causal.
GPU=2 DTYPE=fp8 analysis/standalone/perf.sh 75600 5 5 128 0
SWEEP="1024 2048 4096 8192 16384" GPU=2 analysis/standalone/perf.sh

# Accuracy vs a host float reference built from the same fp8 bytes (keep sq small):
GPU=2 analysis/standalone/check.sh 512 16 2 128 0

# Force a rebuild (e.g. after switching machines):
REBUILD=1 GPU=2 DTYPE=fp8 analysis/standalone/perf.sh 75600 5 5 128 0
```

## ATT overlap analysis (`att_analysis/`) — headless attviewer

Headless equivalent of [ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer):
parse rocprofv3 Advanced-Thread-Trace data, map per-wave instruction latencies
back to C++ source and pipeline **phases** (MATRIX / SOFTMAX / LDS / LOAD / ADDR /
BARRIER), and plot the two co-resident warp-group waves on a shared cycle axis to
expose pipeline overlap & imbalance. Outputs `overlap_simd0.png` + `report.md`.

```bash
# A) Fast, torch-free (traces the standalone exe — seconds):  sq hq hk d mask iters
GPU=2 DTYPE=fp8 analysis/standalone/run.sh 75600 5 5 128 0 3

# B) Through the PyTorch harness (ISA->C++ source mapping via DWARF line tables).
#    LINETABLES=1 rebuilds the JIT .so with -gline-tables-only (does NOT change -O3
#    codegen) so the trace's Source column is populated. SLIM=1 (default) builds
#    only the traced instance — other shapes/dtypes then fail with "no matching
#    kernel" until you restore the full module (SLIM=0 LINETABLES=1, or AITER_REBUILD=1).
LINETABLES=1 analysis/att_analysis/run.sh fp8 16 10000 10000   # rebuild + trace + report
analysis/att_analysis/run.sh fp8 16 10000 10000 0 3            # reuse build (simd 0, 3 iters)
```
Reading the timeline: good pipelining = one wave in MATRIX while the other is in
SOFTMAX; both in BARRIER/wait at the same cycle = a synchronization bubble. The
report's *MATRIX//SOFTMAX overlap %* quantifies it.

`att_analysis/` python modules (run from this `analysis/` dir as `python3 -m att_analysis.<m>`):
`model` (load waves) · `phases` (classify each instr) · `window` (pick steady-state
iters) · `timeline` (render PNG) · `aggregate` (cycle rollups + overlap metric) ·
`report` (bundle) · `run.sh` (rebuild→collect→report).

> The standalone tracer forces `num_splits=1`, so the per-split page-table window
> must fit the 4096-entry LDS cache: `PAGE_BLK=16` at `Sk=75600` needs 4725 entries
> and faults — trace ps32/64/128 instead.

## ISA + VGPR inspection

```bash
# Per-instance VGPR / spill / LDS / scratch footer (fast: device -S only):
LABEL=kv64 analysis/measure_vgpr.sh
XCFLAGS="-DUA_PREFILL_D128_BLOCKSIZE=128" LABEL=kv128 analysis/measure_vgpr.sh

# Disassemble one instance with interleaved C++ source (needs a JIT build present):
analysis/isa_source_map.sh   unified_attention_d128_fp8_mask
analysis/isa_source_map_g.sh unified_attention_d128_fp8_mask   # -gline-tables variant
```

## rocprofv3 phase profiling (RCV trace + counter analyzers)

```bash
# Collect a raw ATT for the ROCprof Compute Viewer GUI (tar + scp + open locally):
analysis/rocprof_att_prefill.sh fp8 16 10000 10000
# Four-phase (trace / compute PMC / stall PMC / pc-sampling) collection + analyze:
analysis/rocprof_prefill_d128.sh fp8 16 10000 10000
python3 analysis/rocprof_analyze.py <run_dir>          # roll up the four phases
python3 analysis/rocprof_phase_split.py <run_dir>      # MATRIX/SOFTMAX/... pc-samp split
python3 analysis/rocprof_warpgroup_balance.py <run_dir>
python3 analysis/rocprof_barrier_latency.py <run_dir>
```
`ua_pmc_counters.txt` is the PMC counter list used for these collections.

---

## Generated artifacts (all gitignored, safe to delete)

`rocprof_analysis/`, `standalone/build*/`, `vgpr_probe/`, `isa_analysis*/`,
`*.csv`/`*.json`/`*.log`, `__pycache__/`. The run scripts regenerate them.
