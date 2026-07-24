---
name: kernel-trace-analysis
description: >
 Analyze an ATT trace captured from an AITER kernel and turn it into an
 actionable perf report: top stall reasons (VMEM, LDS, SALU, VALU,
 barrier, EXP), occupancy, VGPR/SGPR usage, LDS usage, latency hiding
 quality, and — when ISA-to-source mapping is available — the hot source
 lines. Output is formatted for a PR comment / perf report and ends with
 a short list of concrete optimization suggestions that map to the other
 AITER skills (`aiter-triton-kernel`, `aiter-ck-tune`, LDS prefetch, etc.).
 Use this skill after `capture-kernel-trace`, or when a user provides an
 `.att` / `.rgp` file and asks "what's slow", "why is this kernel
 stalling", "analyze the trace", or "find the bottleneck".
allowed-tools: Bash Read Grep Glob
---

# Analyze an AITER Kernel ATT Trace

Input: a `rocprofv3` ATT output directory (typically from the
`capture-kernel-trace` skill) containing `.att` files, a `.rgp`, and a
`codeobj/` folder.

Output: a structured report with:

1. Header (kernel name, arch, grid, block, dispatches analyzed).
2. Occupancy + resource usage (VGPR, SGPR, LDS, waves/CU).
3. Stall breakdown by reason, sorted.
4. Top hot ISA lines mapped back to source.
5. Concrete next actions.

## Step 1: Inventory the trace

```bash
ATT_DIR=/tmp/aiter_att           # adjust
ls "$ATT_DIR"
ls "$ATT_DIR"/att 2>/dev/null | head
ls "$ATT_DIR"/codeobj 2>/dev/null | head
```

Sanity checks:

- There must be a `codeobj/` with HSACO + disassembly, otherwise ISA
  can't be mapped to source and you'll only get coarse stats.
- `att/<dispatch_id>/` should contain per-CU `.out` files.
- `*.rgp` should be present for RGP-based inspection.

If any is missing, go back to the `capture-kernel-trace` skill and
re-capture — don't try to guess around incomplete data.

## Step 2: Pull kernel metadata

From the dispatched kernel's HSACO metadata:

```bash
# Find the HSACO and dump metadata
HSACO=$(find "$ATT_DIR/codeobj" -name "*.hsaco" | head -1)
llvm-readelf -n "$HSACO" | head -80
# Look for:
#   .amdgpu_metadata: .kernarg_segment_size, .vgpr_count, .sgpr_count,
#                     .group_segment_fixed_size (LDS), .wavefront_size
```

Fill in:

| Metric | Source | Typical healthy range |
|--------|--------|-----------------------|
| VGPR count | `.vgpr_count` | ≤ 128 for 8 waves/SIMD on gfx942 |
| SGPR count | `.sgpr_count` | ≤ 102 |
| LDS (bytes) | `.group_segment_fixed_size` | ≤ 65 536 (64 KB); ≤ 163 840 (160 KB) on gfx950 |
| Wave size | `.wavefront_size` | 64 (standard) or 32 (if `--wavefrontsize32`) |
| Grid / block | from the ATT dispatch record | problem-specific |

Occupancy estimate (gfx942, wave64):
```
waves_per_simd = min( 8,
                      floor(256 / ceil(vgpr/4) ),      # VGPR: 256 regs per SIMD / lane-blocks
                      floor(800 / sgpr),               # SGPR: 800 per SIMD
                      floor(LDS_per_CU / lds_per_wg) * wg_per_cu_from_blocksize )
```

If waves/SIMD < 4, flag "low occupancy → latency hiding starved".

## Step 3: Summarize stall reasons

`rocprofv3` emits per-instruction stall tags. Roll them up to a
kernel-level breakdown. Run the bundled aggregator if available in the
user's ROCm install:

```bash
rocprof-sdk-att-summary "$ATT_DIR"/att/*/   # name varies between ROCm versions
```

If no aggregator is installed, parse manually:

```bash
# quick-and-dirty: count stall tags
cat "$ATT_DIR"/att/*/*.out | grep -oE "stall_(vmem|lds|salu|valu|barrier|exp|branch|export|lgkmcnt|vmcnt)" \
  | sort | uniq -c | sort -rn
```

Map raw tags to human categories:

| Tag(s) | Category | What it usually means |
|--------|----------|-----------------------|
| `stall_vmem`, `vmcnt` | Global memory latency | Loads/stores from HBM aren't overlapped with compute. |
| `stall_lgkmcnt` | LDS / scalar mem | Waiting on LDS loads/stores or constant memory. |
| `stall_lds` | LDS bank conflicts | Bank conflicts or serialized LDS reads. |
| `stall_barrier` | `s_barrier` | Block-level barrier — often too-fine-grained sync or wave imbalance. |
| `stall_valu` | VALU busy | Compute-bound (this is often what you want). |
| `stall_salu` | SALU busy | Scalar path bottleneck (address calc, branch decisions). |
| `stall_exp` / `stall_export` | Output / export | Writing results back; rare in pure compute kernels. |
| `stall_branch` | Divergence | Branchy code path — consider predication. |

Produce a bar-chart-in-text:

```
VMEM (HBM latency)     ████████████████████  52.3%
BARRIER                ██████                15.1%
LGKMCNT (LDS)          █████                 12.8%
VALU (actual compute)  ███                    9.4%
SALU                   ██                     5.1%
other                  ██                     5.3%
```

## Step 4: Locate hot ISA lines and map to source

```bash
# disassemble and annotate with hit-counts
llvm-objdump -d --mcpu=$ARCH "$HSACO" > /tmp/kernel.isa
```

RGP / the `rocprofiler-sdk-att-parser` will emit per-PC counts.
Intersect:

```bash
# pseudo — adapt to whatever tool your ROCm gives you
rocprof-sdk-att-parser --annotate "$ATT_DIR" --output /tmp/hot.csv
sort -t, -k2 -nr /tmp/hot.csv | head -20
```

For each top-10 PC, resolve to source with `llvm-dwarfdump` on the
HSACO (requires the kernel built with `-g`):

```bash
for pc in $(sort -t, -k2 -nr /tmp/hot.csv | head -10 | cut -d, -f1); do
  echo "== PC $pc =="
  llvm-dwarfdump --lookup=$pc "$HSACO" | head -5
done
```

If `-g` wasn't set, you'll only get the ISA opcode and no file:line.
Tell the user to rebuild the relevant module with debug info:

- HIP/CK: add `-g` to `flags_extra_hip` of the module in
  `aiter/jit/optCompilerConfig.json`, set `AITER_REBUILD=1`, rerun.
- Triton: set `TRITON_DEBUG=1` and re-run; Triton emits source line info
  by default.

## Step 5: Write the report

Template (copy, fill in):

```
### Kernel: <mangled or make_kernel_repr string>
- Arch: gfx942
- Grid: (X, Y, Z)   Block: (bx, by, bz)   Wavefront: 64
- Resources: VGPR=128  SGPR=98  LDS=49152 B
- Estimated occupancy: 4 waves/SIMD (8 max)

### Stall breakdown (cycle-weighted)
| Category | Share |
|----------|-------|
| VMEM (HBM)  | 52.3% |
| BARRIER     | 15.1% |
| LGKMCNT     | 12.8% |
| VALU        |  9.4% |
| SALU        |  5.1% |
| other       |  5.3% |

### Top hot lines
| PC  | ISA                   | Source (file:line) | Hits |
|-----|-----------------------|--------------------|------|
| 0x540 | `global_load_dwordx4` | gemm_a16w16.py:132 | 18.1% |
| 0x620 | `ds_read_b128`        | gemm_a16w16.py:156 |  9.7% |
| ...

### Interpretation
- Kernel is memory-latency bound: VMEM + LGKMCNT ≈ 65%.
- Occupancy (4 waves/SIMD) is the main reason latency isn't hidden.
- BARRIER share of 15% suggests producer/consumer imbalance between
  the load and compute stages.

### Recommended next actions
1. Reduce VGPR pressure to 96 → 6 waves/SIMD. Candidates:
   - Split accumulator into two passes.
   - Drop the unused `idx_hi` cache.
2. Add LDS double-buffering (see `prefetch-data-load` skill, if present)
   so `ds_read` and `global_load` pipeline overlap.
3. Replace the per-iteration `s_barrier` with a warp-level barrier.
4. Consider raising `BLOCK_K` from 32 to 64 (tune via `aiter-ck-tune`
   / `aiter-triton-kernel` configs and re-measure).
```

## Step 6: Tie back to other skills

Match each recommendation to the right AITER skill:

| Finding | Next skill |
|---------|-----------|
| VMEM-bound, low occupancy | Reduce VGPR; if Triton, re-tune configs via `aiter-triton-kernel`. |
| BARRIER-heavy | `aiter-triton-kernel` or `aiter-ck-tune` — revisit pipeline stages. |
| LDS bank conflicts | Look for `ds_read_b*`/`ds_write_b*` patterns; pad LDS arrays (swizzle). |
| Compute-bound but low TFLOPS | MFMA instruction mix is wrong; compare vs `rocblas`. |
| Wrong kernel picked | `aiter-ck-tune` / `aiter-moe-tuning`: re-run the tuner. |
| Unexpectedly slow on new shape | `bisect-perf-regression`. |

## Common pitfalls

| Symptom | Fix |
|---------|-----|
| "No stall data" | Kernel ran too briefly (< 1 wave-cycle). Increase tensor size or run more iterations and widen `--kernel-iteration-range`. |
| Percentages don't add to 100 | ATT samples are stochastic; a 2–3% gap is fine. A larger gap → re-capture with longer runtime. |
| Source lines point to headers / std libs | Inlining hid the real source. Rebuild with `-O2 -g -fno-inline-functions` for HIP/CK (performance will drop, but you'll get true lines). Only for debugging. |
| "VALU 90%" but TFLOPS low | You're likely emitting scalar/packed ops instead of MFMA. Check the ISA for `v_mfma_*` instructions; if absent, fix kernel/tile types to enable MFMA. |
| Same hot line dominates across arches | That's a real bottleneck — prioritize it. |
| Barrier % > 20% | Almost always a pipeline-stage imbalance. Re-examine the producer/consumer split. |

## References
- `capture-kernel-trace` skill (upstream step)
- `bisect-perf-regression` skill (if the regression is commit-related)
- AMD RGP documentation (`.rgp` viewer)
- `llvm-objdump`, `llvm-dwarfdump`, `rocprof-sdk-att-parser`
