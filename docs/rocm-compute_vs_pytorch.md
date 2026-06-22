# Profiling a PyTorch + FlyDSL app with rocprof-compute / rocprofv3

How to make ROCm Compute Profiler (`rocprof-compute`, ex-Omniperf) and the
underlying `rocprofv3` work on a Python process that imports a **pip-wheel
PyTorch**, on a machine where ROCm is installed system-wide. Two independent
problems; both must be solved. Canonical implementation:
`aiter/aiter/ops/flydsl/kernels/profile_roofline.sh`.

## TL;DR

1. **Driver deps:** run `rocprof-compute` from its **own venv** (it needs
   pandas/dash/matplotlib; pin `pandas==2.2.3`). Launch the profiled app with the
   *app's* python. Never install profiler deps into the app venv.
2. **Binary clash:** the PyTorch wheel bundles its own
   `librocprofiler-register.so`, `librocprofiler-sdk.so`, `libroctracer64.so` and
   loads them via `RPATH=$ORIGIN`. With a profiler tool active this creates a
   *second* rocprofiler stack → fatal double-registration. Fix: make torch use
   the **single system stack** by removing those 3 files from `torch/lib` for the
   duration of the run + `LD_LIBRARY_PATH=/opt/rocm/lib`, then restore. Works only
   because the bundled libs are the **same ROCm version** as the system.

---

## Symptom

Any counter/trace collection on a process that does `import torch` aborts during
torch import:

```
api registration failed with error code 16: Configuration request occurred
outside of valid rocprofiler configuration period
... rocprofiler_configure ... dlopen ... PyImport_ImportModuleLevelObject
*** SIGABRT received ...
[ERROR] Profiling execution failed.
```

Reproduce minimally (no profiler-specific code needed):

```bash
rocprofv3 --pmc SQ_WAVES -d /tmp/x -- <app_venv>/bin/python \
  -c "import torch; a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16); (a@a).sum().item()"
```

## Environment where this was solved

- MI300X (gfx942), ROCm **7.2.0** system install at `/opt/rocm`.
- App venv: torch **2.12.1+rocm7.2** (hip 7.2.53211) — i.e. wheel ROCm == system ROCm.
- `rocprof-compute` **3.4.0** (`rocprofiler-compute` apt pkg), `rocprofv3` from `rocprofiler-sdk`.

## Root cause

`rocprofiler-sdk` works via `librocprofiler-register`: when a ROCm runtime
(HSA/HIP) initializes, it asks the *register* library whether a tool (a library
exporting `rocprofiler_configure`) is present, and the tool is configured **once,
inside a fixed window**. The profiler tool (`rocprofv3`) opens that window at
process start.

A pip PyTorch wheel ships its **own** copies of the ROCm runtime + profiler libs
in `site-packages/torch/lib` and links them with **`DT_RPATH=$ORIGIN`**:

```bash
readelf -d <app_venv>/.../torch/lib/libamdhip64.so | grep -E 'RPATH|RUNPATH|SONAME'
# -> RPATH  $ORIGIN     SONAME libamdhip64.so.7
```

`DT_RPATH` is searched **before** `LD_LIBRARY_PATH` and `LD_PRELOAD`, so torch
always loads *its* bundled HSA/HIP/rocprofiler libs. `import torch` initializes
HSA **late** (after the tool's config window closed) and through a **second**
`librocprofiler-register`/`librocprofiler-sdk` instance → the late registration is
rejected → error 16 → abort.

### Why the obvious fixes don't work
- `LD_PRELOAD=/opt/rocm/lib/librocprofiler-*.so` — torch *still* loads its own
  copies too; you end up with **both** mapped (verify below). RPATH wins for
  torch's own `NEEDED` deps.
- `LD_LIBRARY_PATH=/opt/rocm/lib` alone — ignored for torch's libs (RPATH first).

Verify the double-load (both paths appear → clash):
```bash
LD_PRELOAD=/opt/rocm/lib/librocprofiler-register.so.0 <app_venv>/bin/python -c "
import torch; torch.randn(8,device='cuda')
print(*[l.split()[-1] for l in open('/proc/self/maps')
        if 'rocprofiler-register' in l or 'rocprofiler-sdk' in l], sep='\n')"
# shows BOTH /opt/rocm/lib/...  AND  .../torch/lib/...
```

## The solution

### Part 1 — isolate the profiler driver
`rocprof-compute` is a Python app. Put its deps in a separate venv; keep the app
venv pristine. Pin pandas (3.x breaks rocprof-compute 3.4.0's CSV merge):

```bash
python3 -m venv rocprof_venv
rocprof_venv/bin/pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
rocprof_venv/bin/pip install "pandas==2.2.3"
```

Invoke the driver with `rocprof_venv`, but give it the **app's** python as the
workload command (the launcher is `#!/usr/bin/env python3`, so the active venv
selects the driver interpreter):

```bash
rocprof_venv/bin/python /opt/rocm/libexec/rocprofiler-compute/rocprof-compute \
  profile -n NAME -p OUTDIR --roof-only -- <app_venv>/bin/python app.py ...
```

### Part 2 — collapse to a single ROCm stack
Because RPATH beats `LD_*`, the only reliable way to stop torch loading its own
profiler libs is for those files to be **absent from `torch/lib`** at load time.
Then torch's `NEEDED librocprofiler-register.so.0` etc. fall through to
`LD_LIBRARY_PATH=/opt/rocm/lib` → the system copy → one consistent stack the
profiler tool agrees with. Safe **only because versions match**.

Do this reversibly around the run (the wrapper traps EXIT/INT/TERM to restore):

```bash
TORCH_LIB=$(<app_venv>/bin/python -c \
  'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
LIBS=(librocprofiler-register.so librocprofiler-sdk.so libroctracer64.so)

# disable (restore any leftovers first for idempotency)
for n in "${LIBS[@]}"; do [ -e "$TORCH_LIB/$n" ] && mv "$TORCH_LIB/$n" "$TORCH_LIB/$n.disabled"; done

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
#   ... run rocprof-compute / rocprofv3 here ...

# restore (always, even on failure)
for n in "${LIBS[@]}"; do [ -e "$TORCH_LIB/$n.disabled" ] && mv -f "$TORCH_LIB/$n.disabled" "$TORCH_LIB/$n"; done
```

## Reproduce on another machine (checklist)

1. **Confirm version match:** `python -c "import torch;print(torch.version.hip)"`
   vs `cat /opt/rocm/.info/version`. The wheel's ROCm major.minor.patch should
   equal the system's (here 7.2.70200). If they differ, this swap is unsafe —
   either match them or profile a non-torch process.
2. **Confirm the bundled offenders exist:**
   `ls $TORCH_LIB | grep -E 'rocprofiler-(register|sdk)|roctracer'`.
3. **Confirm RPATH:** `readelf -d $TORCH_LIB/libamdhip64.so | grep RPATH`.
4. Set up `rocprof_venv` (Part 1).
5. Wrap the run with the disable/restore + `LD_LIBRARY_PATH` (Part 2).
6. Or just use `profile_roofline.sh`, which does all of the above.

## Verification (fix works)

```bash
# torch still runs against system libs:
LD_LIBRARY_PATH=/opt/rocm/lib <app_venv>/bin/python -c \
  "import torch;a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16);print(float((a@a).sum()))"
# rocprofv3 no longer aborts:
LD_LIBRARY_PATH=/opt/rocm/lib rocprofv3 --pmc SQ_WAVES -d /tmp/ok -- \
  <app_venv>/bin/python -c "import torch;a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16);(a@a).sum().item()"
# -> no "error code 16", produces counter CSVs.
```

## Caveats

- **Version match is mandatory.** Swapping to the system libs only works because
  they are bit-for-bit the same ROCm release as the wheel's. On a mismatched box,
  upgrade/downgrade one side first.
- **Reversible but global.** The disable window mutates `torch/lib`; don't run two
  profiling sessions (or other torch jobs) against the same venv concurrently. The
  wrapper restores on exit and recovers leftovers on the next start.
- **`pandas==2.2.3`** is specific to rocprof-compute 3.4.0's CSV post-processing
  (the `Agent_Id` merge fails on pandas 3.x).
- Future torch wheels may stop bundling these libs (kineto is migrating to
  rocprofiler-sdk); re-check step 2 before assuming the swap is needed.
