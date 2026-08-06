# Opus GEMM (C++ side)

The user-facing documentation for the opus a16w16 GEMM lives at
[**aiter/ops/opus/README.md**](../../aiter/ops/opus/README.md). It
covers Quick Start, dispatch architecture, tuning workflow, env vars,
testing, internals, and troubleshooting.

This directory holds the C++ / JIT build inputs only.

## Layout

| File | Role |
|---|---|
| `opus_gemm.cu` | Top-level entry points (`opus_gemm()` / `opus_gemm_a16w16_tune()`) and arch routers that switch on `opus_get_gfx_arch()` |
| `opus_gemm_common.py` | Kernel instance metadata — all kids (a16w16 split-barrier, flatmm, flatmm_splitk) live here |
| `gen_instances.py` | JIT codegen driver; `--tune_file` bakes the tuned CSV into `opus_gemm_lookup.h` |
| `opus_gemm_tune.py` | Offline tuner CLI (see `aiter/ops/opus/README.md` §3 for usage) |
| `include/opus_gemm.h`, `include/opus_gemm_arch.cuh` | Cross-arch declarations + `OpusGfxArch` enum + `opus_get_arch_info()` probe |
| `include/opus_gemm_common.cuh`, `include/opus_gemm_utils.cuh` | Cross-arch traits umbrella + opus.hpp shim |
| `include/gfx950/*.cuh` | gfx950-specific pipelines (a16w16 split-barrier / flatmm / flatmm_splitk, a8w8 noscale / scale), traits, splitk reduce, heuristic dispatch (`opus_a16w16_heuristic_dispatch_gfx950`), and the dispatch glue (`opus_gemm_arch_gfx950.cuh`). |

The dispatch flow (gfx950 today):

```
opus_gemm() / opus_gemm_a16w16_tune()      [opus_gemm.cu]
        │
        ├─ opus_get_gfx_arch()  ──────────► OpusGfxArch::{Gfx950, ...}   [opus_gemm_arch.cuh]
        │
        └─ switch (arch) {
             case Gfx950: opus_dispatch_a16w16_gfx950<T>(...)            [gfx950/opus_gemm_arch_gfx950.cuh]
                 │
                 ├─ tuned (M,N,K) lookup map  (baked from CSV)
                 └─ opus_a16w16_heuristic_dispatch_gfx950<T>(...)        [gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh]
           }
```

Per-arch headers carry an `_<arch>` suffix on every file, every kargs
struct, every traits class, and the heuristic dispatch function so two
arches' headers can be visible in the same TU without ODR collisions.
The launcher symbol names (e.g. `opus_gemm_512x256x256x64_2x4_16x16x32_0x0x0`)
are not suffixed because each arch picks a different valid tile set, so
collisions cannot happen unless a future arch picks an identical tile;
if that happens, prefix `gen_instances.py`'s emitted launcher names with
the arch tag at that point.

## Adding a new arch

The codebase is staged so a new arch (e.g. `gfx942`) can be brought up
without touching gfx950 code. The Python and JIT layers are
arch-aware; only the kernel pipelines / traits are gfx950-specific
today.

### 1. Arch enum and runtime probe

Edit [`include/opus_gemm_arch.cuh`](include/opus_gemm_arch.cuh):

```cpp
enum class OpusGfxArch
{
    Unknown = 0,
    Gfx950,
    Gfx942,           // (1) add the enum value
};

// inside opus_get_arch_info():
if (name.rfind("gfx950", 0) == 0)  a = OpusGfxArch::Gfx950;
else if (name.rfind("gfx942", 0) == 0)  a = OpusGfxArch::Gfx942;   // (2) prefix-match
```

### 2. Per-arch headers

Create `include/gfx942/` and mirror the gfx950 layout:

```
include/gfx942/
├── opus_gemm_arch_gfx942.cuh                        # dispatch glue (lookup + heuristic wrapper)
├── opus_gemm_heuristic_dispatch_gfx942.cuh          # M-bucket → launcher symbol heuristic
├── opus_gemm_traits_a16w16_gfx942.cuh               # traits + 5 kargs structs
├── opus_gemm_traits_a8w8_noscale_gfx942.cuh
├── opus_gemm_traits_a8w8_scale_gfx942.cuh
├── opus_gemm_pipeline_a16w16_gfx942.cuh             # __global__ kernel bodies
├── opus_gemm_pipeline_a16w16_flatmm_gfx942.cuh
├── opus_gemm_pipeline_a16w16_flatmm_splitk_gfx942.cuh
├── opus_gemm_pipeline_a8w8_noscale_gfx942.cuh
├── opus_gemm_pipeline_a8w8_scale_gfx942.cuh
└── splitk_reduce_gfx942.cuh
```

Naming rules (mirror gfx950, replace the suffix):

- File names: `opus_gemm_*_gfx942.cuh` (one suffix per file).
- Traits / kargs structs: `opus_gemm_a16w16_traits_gfx942`,
  `opus_gemm_noscale_kargs_gfx942`, `opus_gemm_flatmm_kargs_gfx942`,
  `opus_gemm_flatmm_splitk_kargs_gfx942`,
  `opus_gemm_a16w16_flatmm_traits_gfx942`,
  `opus_flatmm_splitk_traits_gfx942`,
  `opus_gemm_a8w8_noscale_traits_gfx942`,
  `opus_gemm_a8w8_scale_traits_gfx942`,
  `opus_gemm_scale_kargs_gfx942`.
- Shared-ABI kargs guard macro: `OPUS_GEMM_NOSCALE_KARGS_GFX942_DEFINED`.
- Heuristic dispatch function: `opus_a16w16_heuristic_dispatch_gfx942<T>`.
- Per-arch dispatch glue:
  - `opus_dispatch_a16w16_gfx942<T>(int M, int N, int K, int batch)`
  - `opus_a16w16_tune_dispatch_gfx942<T>(int id)`

If gfx942 reuses the same launcher tile sizes, also rename the emitted
launcher symbols (see step 5) to avoid ODR collisions in the manifest.

### 3. Cross-arch umbrella

Edit [`include/opus_gemm_common.cuh`](include/opus_gemm_common.cuh) to
include the new arch's traits headers alongside gfx950's:

```cpp
#include "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"
#include "gfx942/opus_gemm_traits_a8w8_scale_gfx942.cuh"     // new
#include "gfx942/opus_gemm_traits_a8w8_noscale_gfx942.cuh"   // new
#include "gfx942/opus_gemm_traits_a16w16_gfx942.cuh"         // new
```

The `_gfx942` suffix on every struct keeps the two arches' definitions
from clashing in the same TU.

### 4. Arch routers in `opus_gemm.cu`

Edit [`opus_gemm.cu`](opus_gemm.cu) — add the include and one `case`
per router:

```cpp
#include "gfx950/opus_gemm_arch_gfx950.cuh"
#include "gfx942/opus_gemm_arch_gfx942.cuh"   // new

template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch)
{
  switch (opus_get_gfx_arch()) {
    case OpusGfxArch::Gfx950:
      return opus_dispatch_a16w16_gfx950<CDataType>(M, N, K, batch);
    case OpusGfxArch::Gfx942:                                          // new
      return opus_dispatch_a16w16_gfx942<CDataType>(M, N, K, batch);   // new
    default: { /* TORCH_CHECK with arch_info */ }
  }
}
```

Same edit for `opus_a16w16_tune_dispatch<T>` (id-based router) and the
`a8w8` block (it currently `TORCH_CHECK`s on `arch == Gfx950`; widen
the check or move a8w8 to its own arch router).

### 5. Codegen tables

`gen_instances.py` keeps four arch-tagged tables that drive launcher
emission. Today they hard-code gfx950; the cleanest extension is to
make them dispatch on a per-`OpusGemmInstance` `arch` field:

| Table | What it controls | gfx950 entry today |
|---|---|---|
| `PIPELINE_HEADER_MAP` | `#include "{pipeline_header}"` in each launcher TU | `gfx950/opus_gemm_pipeline_*_gfx950.cuh` |
| `TRAITS_NAME_MAP` | `using Traits = {traits_name}<...>` | `opus_gemm_*_traits_gfx950` |
| `KARGS_NAME_MAP` | `{kargs_name} kargs{};` | `opus_gemm_*_kargs_gfx950` |
| `KERNEL_FUNC_MAP` | `__global__` template name (unchanged across archs) | `gemm_*_kernel` |

Two implementation options:

- **Per-arch dicts** (smallest surface change): add
  `PIPELINE_HEADER_MAP_GFX942`, etc., and pick the right one inside the
  `opus_gemm_codegen` methods based on the instance's arch.
- **Tagged keys**: keep one dict but key by `(arch, kid_tag)`. Cleaner
  long-term; needs more adapter code.

Also extend `OpusGemmInstance` (in `opus_gemm_common.py`) with an
`arch: str` field, and make the launcher symbol name include the arch
suffix when a launcher with the same tile already exists for another
arch — otherwise the manifest will see two prototypes with the same
function name.

### 6. Python import-time guard

Edit [`aiter/ops/opus/__init__.py`](../../aiter/ops/opus/__init__.py)
to widen the supported arch set:

```python
_SUPPORTED = {"gfx950", "gfx942"}   # new
```

The probe helper at
[`aiter/ops/opus/_arch.py`](../../aiter/ops/opus/_arch.py) is
non-raising: ``_detect_arch(supported)`` returns ``(ok, detected)`` so
the package can install stubs and emit a ``RuntimeWarning`` instead of
breaking ``from aiter.ops.opus import *`` (which sits inside the
swallow-ImportError ``try`` block in ``aiter/__init__.py``). Calling a
stub raises ``RuntimeError`` with the detected arch and the supported
set. ``_check_arch`` (raising variant) is still available for callers
that prefer hard failure.

### 7. Tuning data

If gfx942 needs its own tuned CSV (different tile choices), either:

- Co-locate per-arch CSV files (e.g.
  `aiter/ops/opus/configs/opus_gemm_a16w16_tuned_gfx942.csv`) and have
  `gen_instances.py --tune_file` consume the right one for the active
  arch (or both, baked into separate macros consumed by the per-arch
  glue header).
- Keep one CSV with an `arch` column and filter at codegen time.

### 8. Multi-arch wheel build

The Layer-3 device-pass guard
(`#if defined(__gfx950__)` wrapping the kernel body) is per-arch and
already in place for gfx950. Add the matching guard at the top of each
new gfx942 kernel body so multi-arch wheels (e.g.
`GPU_ARCHS=gfx950;gfx942`) compile cleanly:

```cpp
__global__ void gemm_a16w16_kernel_gfx942(...) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    /* real body */
#else
    /* empty stub: unreachable at runtime; here so other arches' device pass
       does not try to instantiate gfx942-only intrinsics */
#endif
#endif
}
```

### 9. Validation

Run the standard regression on a gfx950 box:

```bash
GPU_ARCHS=gfx950 python op_tests/test_opus_a16w16_gemm.py -m 128 -n 256 -k 1024 -b 1
# CSV sweep (optional, if you have a shapes file):
GPU_ARCHS=gfx950 python op_tests/test_opus_a16w16_gemm.py --csv /path/to/shapes.csv
```

Then, on the new arch hardware, repeat with `GPU_ARCHS=gfx942` and
provide a tuned CSV (if any) to populate the lookup map.

For a multi-arch wheel sanity check, do a full rebuild:

```bash
rm -f aiter/jit/module_deepgemm_opus.so
AITER_REBUILD=1 GPU_ARCHS="gfx942;gfx950" python -c \
    "from aiter.ops.opus import gemm_a16w16_opus; print('ok')"
```

The build must finish without errors; both `--offload-arch=gfx950` and
`--offload-arch=gfx942` must appear in the hipcc invocation; and the
runtime call on whichever device the host machine has must produce
correct results (no clean way to cross-test on a single-arch host).

## Splitk workspace and TBO (two-batch overlap)

The splitk kid family (kid 200..299, a16w16_flatmm_splitk) writes per-split
partials into an fp32 scratch buffer that the reduce kernel folds into the
final output. That buffer is owned host-side as a stable
`opus_splitk_ws_handle*` slot: the kernel reads `slot->ptr` at launch, so
captured HIP graphs bake in the slot address, not the raw buffer. The
launcher grows `slot->ptr` lazily (4 MiB-aligned), draining outstanding work
with `hipDeviceSynchronize` before freeing the old buffer.

### Ownership: per-stream, not per-thread

The slot is registered in a **process-global, mutex-protected map keyed by
`hipStream_t`** (see `opus_splitk_ws_get` in `opus_gemm.cu`). vLLM/sglang-
style TBO drives two CUDA streams from two CPU threads; each captured graph
must bake in its own buffer pointer so concurrent replays write disjoint
scratch. A per-thread cache would either share one buffer between the two
streams (corrupting concurrent replays) or fail the in-capture grow guard on
the second thread's first call.

### Framework usage

```python
import aiter

compute = torch.cuda.Stream()
comm    = torch.cuda.Stream()

with torch.cuda.stream(compute):
    aiter.opus_gemm_workspace_init()         # register handle for this stream
    _ = gemm_a16w16_opus(A_max, B_max)       # warm: grow buffer to max size
    # ... capture compute graph ...

with torch.cuda.stream(comm):
    aiter.opus_gemm_workspace_init()
    _ = gemm_a16w16_opus(A_max, B_max)
    # ... capture comm graph ...

# Replay both graphs concurrently from two threads -- each sees its own
# workspace; no aborts, no cross-stream interference.
```

Rules:

* Call `opus_gemm_workspace_init()` once per TBO stream, **eagerly** (outside
  HIP graph capture). Calling it during capture raises; capture cannot run
  `hipHostMalloc`.
* Before capture, run the largest expected splitk shape on that stream once,
  so the buffer grows to its final size. Grow inside capture is illegal
  (`hipMalloc` / `hipFree` are stream-capture-illegal) and aborts with a
  message pointing back here.
* Single-stream, single-thread (non-TBO) callers do **not** need to call
  `opus_gemm_workspace_init` -- the registry lazy-creates a handle on the
  first eager call on each new stream. Init only becomes mandatory when the
  first call on a stream is inside HIP graph capture.

### When the buffer is freed

Never automatically. Each registered stream holds one `opus_splitk_ws_handle`
plus its current `hipMalloc` buffer for the process lifetime. Streams the
torch CUDAStream pool reuses are correctly reused via the same map entry;
streams the pool never reclaims leak one handle + buffer (same shape as the
prior `thread_local` cache's thread-exit leak). A future
`opus_gemm_workspace_release` can be added if frameworks need explicit
teardown.
