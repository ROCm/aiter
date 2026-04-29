---
name: capture-kernel-trace
description: >
 Capture a `rocprofv3` ATT (Advanced Thread Trace) of an AITER kernel —
 HIP, CK, Triton, or FlyDSL — so it can be loaded in the Radeon GPU
 Profiler (RGP) or post-processed offline. Covers the kernel-name filter
 (`--att-kernel` / `--kernel-iteration-range`), environment setup for ROCm
 7.x ATT, how to name the kernel from an AITER op (`make_kernel_repr`,
 `@compile_ops` mangling), and how to shrink the trace to a single
 dispatch.
 Use this skill when the user asks to "capture an ATT trace", "profile
 kernel with rocprofv3", "record an RGP capture", or "grab a trace of the
 GEMM / attention / MoE kernel".
allowed-tools: Bash Read Grep Glob
---

# Capture an ATT Kernel Trace of an AITER Op

Goal: produce a single, minimal `rocprofv3` ATT capture that focuses on
exactly the dispatch(es) the user cares about, then hand the resulting
`.rgp` / `.att` files to the `kernel-trace-analysis` skill.

## Step 1: Verify the environment

```bash
rocprofv3 --version            # need ROCm >= 6.2 for ATT; 7.x preferred
rocminfo | grep -E "Name:.*gfx"     # confirm the target arch
```

Targeting a specific GPU on a multi-GPU box:

```bash
export HIP_VISIBLE_DEVICES=0
```

ATT data is *large* (hundreds of MB per capture). Make sure the output
directory has >= 5 GB free.

## Step 2: Identify the kernel name

AITER kernels show up under different names depending on backend:

| Backend | Name pattern | Where to find it |
|---------|--------------|------------------|
| HIP C++ | C++-mangled, e.g. `_Z18rmsnorm_kernel_...` | `csrc/.../*.cu` function name |
| CK-tile | Long templated: `ck_tile::kernel<...>` | generated instance file in `build/` |
| Triton | `make_kernel_repr(...)` string or JIT-hashed | `aiter/ops/triton/.../*.py` |
| ASM | Symbol from `.s` file | `hsaco/` binaries |

Fastest way to get the exact name: run the op once under a plain
`rocprofv3` kernel-stats capture first, then grep for a distinctive
substring.

```bash
rocprofv3 --stats --kernel-trace -d /tmp/aiter_stats -- \
    python -c "import torch, aiter; \
               a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); \
               b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); \
               aiter.gemm_a16w16(a, b)"
ls /tmp/aiter_stats
cat /tmp/aiter_stats/*.kernel_stats.csv | head
```

Pick the substring that uniquely matches your kernel (e.g. `gemm_a16w16`,
`rmsnorm`, `fused_moe_2stages`, `flashatt_fwd`). For Triton, the
repr string from `make_kernel_repr` (defined in
`aiter/ops/triton/utils/*`) is what you'll see.

## Step 3: Minimize the Python driver

Keep the capture to a **single kernel launch** if possible. Warm-up
shows up in the trace and inflates it. A minimal driver:

```python
# /tmp/aiter_trace_driver.py
import os, torch, aiter
torch.manual_seed(0)

# warm-up OUTSIDE the recorded range — see Step 4's --kernel-iteration-range
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    aiter.gemm_a16w16(a, b)

torch.cuda.synchronize()
# This is the dispatch we want traced:
out = aiter.gemm_a16w16(a, b)
torch.cuda.synchronize()
```

For ops that live inside a larger graph, isolate with:

```python
with torch.profiler.record_function("aiter_target"):
    out = aiter.target_op(...)
```

`rocprofv3 --hip-activity` can filter by that marker.

## Step 4: Run the ATT capture

```bash
mkdir -p /tmp/aiter_att
rocprofv3 \
    --att \
    --att-activity \
    --att-perfcounter-ctrl 0x3 \
    --att-target-cu 0 \
    --kernel-include-regex "gemm_a16w16" \
    --kernel-iteration-range "[1]" \
    -d /tmp/aiter_att \
    -- python /tmp/aiter_trace_driver.py
```

Flag cheat-sheet (verify with `rocprofv3 --help` on your ROCm version —
the flag names changed between ROCm 6.2 / 6.3 / 7.0):

| Flag | Purpose |
|------|---------|
| `--att` | Enable ATT. |
| `--kernel-include-regex "<substr>"` | Only trace kernels whose name matches. |
| `--kernel-iteration-range "[N]"` / `"[N:M]"` | Only dispatch N (or N..M) of matched kernels. Use `[1]` to skip warm-up. |
| `--att-target-cu 0` | Which compute unit to record (ATT is per-CU). `0` is fine for most kernels. |
| `--att-se-mask 0x1` | Shader engine bitmask (default covers all). |
| `-d <dir>` | Output directory. |
| `--output-format rgp` | Emit RGP-loadable `.rgp` in addition to raw `.att`. |

Minimum useful capture for a single kernel (ROCm 7.x style):

```bash
rocprofv3 --att \
    --kernel-include-regex "<substr>" \
    --kernel-iteration-range "[1]" \
    --output-format rgp \
    -d /tmp/aiter_att \
    -- python /tmp/aiter_trace_driver.py
```

If the regex matches nothing, the run will complete normally but the
output directory will have no `.att` / `.rgp`. Fix the regex (use the
kernel-stats CSV from Step 2 to copy an exact substring).

## Step 5: Verify the capture

```bash
ls -lh /tmp/aiter_att
# expect: *.att (raw), *.rgp (RGP capture), and a codeobj-tracing folder
```

Good signs:

- `.rgp` file of a few tens of MB — that's a healthy single-kernel capture.
- `att/<dispatch_id>/` directories with `.out` files per CU.
- `codeobj/` with the HSACO and ISA dump.

Bad signs:

- Directory is empty → regex didn't match. Re-check kernel name.
- `.rgp` is multiple GB → too many dispatches captured. Narrow
  `--kernel-iteration-range` to a single index.
- Missing `codeobj/` → ATT can't map ISA back to source. Make sure the
  driver uses a debug or line-mapped build: set
  `HIPCC_VERBOSE=1` and include `-g` via `AITER_EXTRA_HIP_FLAGS="-g"`
  when rebuilding (for HIP/CK kernels); Triton emits line info by default.

## Step 6: Hand off

Report back to the user:

- Kernel name captured (exact string)
- Path to the `.rgp` and output directory
- Number of dispatches captured (`ls /tmp/aiter_att/att/ | wc -l`)
- GPU arch (`gfx942`, `gfx950`, …)

Then either open the `.rgp` in RGP, or run the
`kernel-trace-analysis` skill on `/tmp/aiter_att/` for stall / ISA-level
analysis.

## Common pitfalls

| Symptom | Cause / fix |
|---------|-------------|
| Empty output dir | `--kernel-include-regex` didn't match. Grab the exact name from the kernel-stats CSV first. |
| Capture is several GB | Too many iterations. Add `--kernel-iteration-range "[1]"` and reduce warm-up loops. |
| `rocprofv3: unrecognized option '--att'` | ROCm too old. Upgrade to ≥ 6.2; ATT was experimental before. |
| RGP shows no ISA mapping | Code object wasn't emitted with line info. Rebuild with `-g` (HIP/CK) or `TRITON_DEBUG=1` (Triton). |
| `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Too many counters requested. Drop `--att-perfcounter-ctrl` or pick a smaller mask. |
| Triton kernel not filterable by name | Use `TRITON_KERNEL_DUMP=1` to find the auto-generated repr, or wrap the call in `torch.profiler.record_function(...)` and filter on that marker. |
| Captured wrong dispatch (a warm-up) | Use `--kernel-iteration-range "[<idx>]"` with `<idx>` > warm-up count; or remove warm-up from the driver. |

## References
- `rocprofv3 --help` (authoritative for your installed ROCm)
- AMD ATT user guide: `https://rocm.docs.amd.com/projects/rocprofiler-sdk/`
- `aiter/ops/triton/utils/_triton/kernel_repr.py` for Triton `make_kernel_repr`
- Use the `kernel-trace-analysis` skill next to read the capture.
