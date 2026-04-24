# Opus a16w16 GEMM

BF16 × BF16 → BF16/FP32 matmul backed by the opus kernel family (AMD
gfx950 / MI300X class). Provides a shape-driven Python API, a runtime
dispatcher with CSV-baked lookup + heuristic fallback, and a tuning
pipeline that populates the lookup.

Underlying JIT module: `module_deepgemm_opus`
(see `aiter/jit/optCompilerConfig.json`).

---

## 1. Quick Start

```python
import torch
from aiter.ops.opus import gemm_a16w16_opus

A = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
B = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)  # [N, K]

Y = gemm_a16w16_opus(A, B)               # bf16 output, tune table + C++ heuristic
Y = gemm_a16w16_opus(A, B, dtype=torch.float32)   # fp32 output
Y = gemm_a16w16_opus(A, B, out=preallocated_Y)    # reuse buffer
```

First call triggers a JIT build of `module_deepgemm_opus` (~30-45s on
the dev container). Subsequent Python processes reuse the compiled
`.so`.

**Inputs**: `A` is `[M, K]` or `[batch, M, K]` bf16; `B` is `[N, K]`
bf16 (plain layout, not pre-shuffled). **Output**: `[M, N]` or
`[batch, M, N]`.

Not currently supported (returns `NotImplementedError`): `bias`,
pre-shuffled B, non-bf16 A/B. Scale / FP8 paths are handled by other
opus submodules (a8w8 / a8w8_blockscale, landing in follow-up PRs);
they share the same `module_deepgemm_opus` JIT build but expose their
own Python entry under `aiter.ops.opus.*`.

---

## 2. How Dispatch Works

When the user calls `gemm_a16w16_opus(A, B)` without an explicit kernel
id, the wrapper routes the request through **two independent lookup
mechanisms plus a heuristic**:

```
gemm_a16w16_opus(A, B)
  ├─ explicit kernelId=N?  ───yes──►  opus_gemm_a16w16_tune(N, ...)
  │
  ├─ Python-side CSV lookup  ───hit──►  opus_gemm_a16w16_tune(solidx, splitK)
  │       (aiter/ops/opus/common.py, lru_cache'd from
  │        aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv)
  │
  └─ miss ──► (optional autolog) ──► opus_gemm() [C++]
                                         ├─ C++ compile-time lookup (same CSV
                                         │   baked into opus_gemm_lookup.h
                                         │   at JIT time)
                                         └─ miss ──► opus_a16w16_heuristic_dispatch
                                                     (hand-written if-else by M)
```

Two CSV lookups coexist on purpose:

- The **Python layer** reads the CSV live every new process; edits to
  the CSV take effect immediately.
- The **C++ layer** (generated into `opus_gemm_lookup.h` by
  `gen_instances.py --tune_file`) serves the `opus_gemm()` C++ entry
  and has zero Python overhead per call, but **requires
  `AITER_REBUILD=1` to pick up CSV edits**.

Explicit `kernelId=` bypass exists for tuning, debugging, and future
integrations (e.g. `aiter.tuned_gemm.solMap["opus"]`).

---

## 3. Tuning Your Shapes

The shipped `aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv` covers
a handful of models. For anything else the heuristic kicks in, which is
correct but not necessarily fastest. To get peak perf on your shapes:

### 3.1 One-shot: run the tuner directly

```bash
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -i aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv \
    -o aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv
```

The input CSV needs only `M, N, K` columns (anything else is ignored).
`-o` defaults to `$AITER_OPUS_A16W16_TUNED_CSV`, which points at the
shipped opus-private tuned CSV (see [§6 Environment](#6-environment)).

Verify winners work by running the lookup test (it reads the tuned CSV
and launches every row through `opus_gemm_a16w16_tune`):

```bash
python3 op_tests/test_opus_a16w16_lookup.py
# expected: 212/212 passed
```

### 3.2 Autolog: collect shapes first, tune later

Point your workload at `gemm_a16w16_opus` with autolog enabled, then
feed the untuned CSV back to the tuner:

```bash
# 1. Run your model / benchmark with autolog on.
AITER_OPUS_LOG_UNTUNED=1 python3 your_script.py

# 2. Tune the shapes the Python layer observed.
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -i aiter/ops/opus/configs/opus_a16w16_untuned_gemm.csv \
    -o aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv

# 3. (Optional) Rebuild the JIT C++ lookup so the zero-Python
#    opus_gemm() entry also benefits.
AITER_REBUILD=1 python3 -c "from aiter.ops.opus import gemm_a16w16_opus"
```

Autolog only records shapes that **missed** the tuned CSV. Hits are not
rewritten, and duplicates are deduped on each write.

---

## 4. API Reference

### `gemm_a16w16_opus(A, B, bias=None, dtype=bf16, *, kernelId=None, splitK=None, out=None)`

Primary user entry. Implemented in
[aiter/ops/opus/gemm_op_a16w16.py](gemm_op_a16w16.py).

| Param | Type | Default | Notes |
|---|---|---|---|
| `A` | Tensor | required | `[M, K]` or `[batch, M, K]` bf16 |
| `B` | Tensor | required | `[N, K]` bf16 (plain layout) |
| `bias` | Tensor? | `None` | Must be `None`; opus kernels compile with `HAS_BIAS=false` |
| `dtype` | torch.dtype | `bf16` | Output dtype; fp32 is supported but splitk kids (bf16-only Y) are skipped on the fp32 path |
| `kernelId` | int? | `None` | Override: skip CSV/heuristic and launch this specific instance |
| `splitK` | int? | `None` | Only honored with explicit `kernelId`; literal KBatch for splitk kids |
| `out` | Tensor? | `None` | Reuse a preallocated output buffer |

### `opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)`

Low-level id-based dispatcher. Used by the tuner and the high-level
wrapper. Accepts 3D inputs only (`[batch, M, K]`, `[batch, N, K]`,
`[batch, M, N]`). Prefer `gemm_a16w16_opus` unless you need explicit
control.

### Legacy shim

`aiter.ops.deepgemm.opus_gemm_a16w16_tune` still works and forwards to
`aiter.ops.opus.*` with a `DeprecationWarning`; scheduled for removal
one aiter minor release later.

`aiter.ops.deepgemm.deepgemm_opus` (the old aggregate entry that
exposed FP8 grouped + a16w16 no-scale through a single function) has
been **removed** along with any internal opus binding in that module.
Migration:

- BF16 no-scale GEMM: use `gemm_a16w16_opus` from this module.
- FP8 grouped GEMM: future `aiter.ops.opus.a8w8*` modules (separate
  PR). Until they land, bind `opus_gemm` yourself via `compile_ops`
  against `module_deepgemm_opus` / `fc_name="opus_gemm"`.

`aiter.ops.deepgemm.deepgemm()` is now a thin forwarder to
`deepgemm_ck`; the `AITER_DEEPGEMM_BACKEND=opus` dispatch env is no
longer recognized.

---

## 5. Testing

All tests run inside the project's container
(`docker exec -w /wksp/aiter demon_test bash -lc ...`).

| Test | Purpose | Pass criterion |
|---|---|---|
| `op_tests/test_opus_a16w16_tune.py` | Per-kid correctness smoke (iterate every compiled kid, launch once at a kid-appropriate min-K) | 17/17 PASS |
| `op_tests/test_opus_a16w16_lookup.py` | Iterate every row in `opus_gemm_a16w16_tuned.csv`, launch with `(solidx, splitK)` from CSV, compare against `torch.bmm` | 212/212 PASS (3 skipped if flatmm kids are not compiled in the current JIT) |
| `op_tests/test_opus_a16w16_gemm.py` | End-to-end test of `gemm_a16w16_opus` (shape-driven API); supports single-shape smoke and CSV sweep (e.g. gptoss untuned) | `allclose` passes on all shapes |
| `op_tests/check_opus_splitk_graph.py` | HIP graph capture + replay smoke on two splitk shapes | Both shapes complete |

Filter flags on `test_opus_a16w16_lookup.py`:

```bash
python3 op_tests/test_opus_a16w16_lookup.py --max-rows 30       # smoke subset
python3 op_tests/test_opus_a16w16_lookup.py --kid-min 200       # only splitk kids
python3 op_tests/test_opus_a16w16_lookup.py --tuned-csv /tmp/t.csv  # custom CSV
python3 op_tests/test_opus_a16w16_lookup.py -v                  # per-row output
```

---

## 6. Environment

All opus-specific env vars live here; **nothing opus-specific leaks
into `aiter/jit/core.py`'s `AITER_CONFIGS`** (that module is reserved
for aiter-global configuration shared across backends).

| Env var | Default | Effect |
|---|---|---|
| `AITER_OPUS_A16W16_TUNED_CSV` | `aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv` | Path to the opus-private tuned CSV. Read by `common.py` (Python lookup) and by `gen_instances.py --tune_file` (C++ lookup bake-in). |
| `AITER_OPUS_A16W16_UNTUNED_CSV` | `aiter/ops/opus/configs/opus_a16w16_untuned_gemm.csv` | Autolog target. Written with dedup when `AITER_OPUS_LOG_UNTUNED=1`. |
| `AITER_OPUS_LOG_UNTUNED` | `0` | `1` turns on runtime shape collection at every CSV miss. |
| `AITER_REBUILD` | `0` | `1` forces JIT rebuild of `module_deepgemm_opus`. Needed after CSV edits if you want the C++ lookup to pick them up. |
| `FLATMM_HIP_CLANG_PATH` | unset | Optional hipcc override (see `optCompilerConfig.json`). |

---

## 7. Under the Hood

### 7.1 Two-level dispatch (mirrors `csrc/ck_gemm_a8w8/gemm_a8w8.cu`)

`opus_dispatch_a16w16<CDataType>` in
[csrc/opus_gemm/opus_gemm.cu](../../../csrc/opus_gemm/opus_gemm.cu):

```cpp
template <>
OpusA16W16NoscaleKernel opus_dispatch_a16w16<bf16_t>(int M, int N, int K, int batch)
{
  static const auto lookup = [] {
    return OpusA16W16RuntimeMap{GENERATE_OPUS_LOOKUP_TABLE_BF16(bf16_t)};
  }();
  auto it = lookup.find({M, N, K});
  if (it != lookup.end()) return it->second;
  return opus_a16w16_heuristic_dispatch<bf16_t>(M, N, K, batch);
}
```

The `<fp32_t>` specialization is analogous, using
`GENERATE_OPUS_LOOKUP_TABLE_FP32`.

### 7.2 Kernel inventory (see [opus_gemm_common.py](../../../csrc/opus_gemm/opus_gemm_common.py))

Two a16w16-class pipelines are compiled today:

- **Split-barrier a16w16** (kid 4..9): traditional 2-stage double-
  buffered pipeline. Both `<bf16_t>` and `<fp32_t>` instantiations
  emitted. Requires even `ceil_div(K, B_K)`.
- **Warp-specialized flatmm_splitk** (kid 200..210): 4-wave warp-spec
  kernel with runtime splitK (literal KBatch), fp32 workspace, reduce
  kernel casts to bf16 Y. Only `<fp32_t>` instantiations emitted;
  launcher forces bf16 Y. Handles arbitrary N/K via `mask_va_tail` +
  reduce-kernel tail path.

Representative instances (full table lives in
[opus_gemm_common.py](../../../csrc/opus_gemm/opus_gemm_common.py)):

| kid | Pipeline | Tile (B_M, B_N, B_K) | WG/CU | Notes |
|-----|----------|-----|-------|-------|
|   9 | a16w16 split-barrier | (256, 256, 64) | 2 | Traditional sweet spot for large aligned M/N |
| 200 | flatmm_splitk | (64, 64, 64) | 2 | splitk default for M ≤ 128 |
| 208 | flatmm_splitk | (64, 64, 128) | 1 | Deep K / very skinny M |

**16 additional a16w16_flatmm kid slots (100..115) are reserved but
currently empty** (`a16w16_flatmm_kernels_list = {}`). Filling them
is orthogonal to this module and does not require changes here.

### 7.3 Heuristic fallback (bf16 Y path)

In
[csrc/opus_gemm/opus_gemm.cu](../../../csrc/opus_gemm/opus_gemm.cu)
`opus_a16w16_heuristic_dispatch<bf16_t>`:

| M range | Kernel | Rationale |
|---|---|---|
| `M ≤ 4` | kid 208 splitk `(64, 64, 128)` WG=1 | Very skinny M; deep K keeps splitk workspace small |
| `M ≤ 64` | kid 206 splitk `(64, 32, 128)` WG=2 | cc-recommended mid-M tile |
| `M ≤ 128` | kid 200 splitk `(64, 64, 64)` WG=2 | splitk sweet spot |
| `M > 128`, N%16 + K%64 + loops even | kid 9 a16w16 `(256, 256, 64)` | Traditional split-barrier wins large aligned |
| `M > 128`, misaligned | kid 200 splitk | splitk tolerates arbitrary N/K |

`<fp32_t>` path always returns kid 9 (splitk forbids fp32 Y at the
launcher level).

### 7.4 The splitk `<fp32_t>` trick in the BF16 lookup map

splitk kid instantiations exist only as `<fp32_t>` (their traits
`static_assert(D_C == float)` — the main kernel writes an fp32
workspace; the reduce kernel then casts to bf16 Y). So the BF16 lookup
map contains mixed template arguments:

```cpp
// aiter/jit/build/module_deepgemm_opus/blob/opus_gemm_lookup.h (generated)
#define GENERATE_OPUS_LOOKUP_TABLE_BF16(CTYPE)                         \
   {                                                                   \
       {{1, 100, 5120},                                                \
        opus_gemm_flatmm_splitk_256x32x32x64_..._wgpcu2<fp32_t>},      \
       {{256, 51200, 5120},                                            \
        opus_gemm_512x256x256x64_2x4_16x16x32_0x0x0<CTYPE>},           \
       ...
   }
```

`gen_instances.py:gen_lookup_dict` hardcodes `<fp32_t>` for splitk kids
regardless of which per-CTYPE map they land in. FP32 map drops splitk
entries entirely (launcher `TORCH_CHECK`s `Y.dtype() == BFloat16`).

### 7.5 JIT build pipeline

1. `aiter.ops.opus.gemm_op_a16w16` triggers `compile_ops("module_deepgemm_opus")`.
2. [aiter/jit/optCompilerConfig.json](../../jit/optCompilerConfig.json)
   invokes
   `csrc/opus_gemm/gen_instances.py --working_path {blob_dir} --tune_file aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv`
3. `gen_instances.py` writes:
   - `impl/*.cuh` — per-kid kernel launcher templates
   - `instances/*.cpp` — explicit template instantiations
   - `opus_gemm_manifest.h` — forward declarations
   - `opus_gemm_a16w16_tune_lookup.h` — int-id → kernel maps
   - `opus_gemm_lookup.h` — **(M, N, K) → kernel** maps baked from the
     tuned CSV (two macros: `_BF16`, `_FP32`)
4. `opus_gemm.cu` is compiled and linked against the generated
   instances into `module_deepgemm_opus.so`.

---

## 8. Troubleshooting

### CSV edits don't seem to take effect

Python-side lookup in `common.py` is read lazily per process
(`functools.lru_cache(maxsize=1)`). Restart the process — that is
enough for Python-layer routing.

The **C++ compile-time lookup** (`opus_gemm_lookup.h`) only picks up
CSV changes on JIT rebuild:

```bash
AITER_REBUILD=1 python3 -c "from aiter.ops.opus import gemm_a16w16_opus"
```

The whole rebuild takes ~30-45s on dev hardware.

### `NotImplementedError: gemm_a16w16_opus does not currently support bias`

Opus kernels are compiled with `HAS_BIAS=false` (splitk plan §8).
Options:

- Drop the bias from the call (`bias=None`) and add it yourself
  afterwards.
- Use a different backend (`aiter.gemm_a16w16` / `F.linear`).
- Extend `opus_flatmm_splitk_traits` + reduce kernel to honor the
  reserved `ptr_bias` field. Non-trivial; out of scope for this
  module.

### `Kernel id N not found in a16w16 ... tune lookup table`

The CSV references a kid that the current JIT build didn't compile
(usually a flatmm kid 100..115 from an older tuning run, since
`a16w16_flatmm_kernels_list` is currently empty). Either re-tune the
affected shapes, or just let `test_opus_a16w16_lookup.py` report them
as `SKIP` (it handles this explicitly).

### Old split-barrier K-parity bug

The traditional a16w16 pipeline (kid 4..9) has a known silent-
corruption bug when `ceil_div(K, B_K)` is **odd** on M > 1: the last
prefetch reads one tile past the valid K range and corrupts the
accumulator.

- The Python CSV lookup and C++ heuristic both prefer splitk kids
  (200..210) for `M ≤ 128`, sidestepping the pipeline entirely.
- The tuner's accuracy gate discards any split-barrier kid that fails
  on a given (M, N, K).
- Only way to hit it today: pass `kernelId=4..9` explicitly with an
  adversarial K.

### HIP graph compatibility

splitk kernels allocate a fresh fp32 workspace via `torch::empty` per
call (same pattern as triton `gemm_a16w16` uses for `y_pp`). This works
under `torch.cuda.graph` capture + replay — validated by
`op_tests/check_opus_splitk_graph.py` (first consumer of
`testGraph=True` in aiter). See splitk plan §5.

---

## 9. File Map

| Path | Role |
|---|---|
| [aiter/ops/opus/gemm_op_a16w16.py](gemm_op_a16w16.py) | `gemm_a16w16_opus` wrapper + low-level `opus_gemm_a16w16_tune` pybind + private `_opus_gemm_bf16_dispatch` fallback binding |
| [aiter/ops/opus/common.py](common.py) | Python tuned-CSV lookup + autolog |
| [aiter/ops/opus/__init__.py](__init__.py) | Public symbol aggregator |
| [aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv](configs/opus_gemm_a16w16_tuned.csv) | Shipped tuned winners (17-column schema) |
| [aiter/ops/deepgemm.py](../deepgemm.py) | CK backend (`deepgemm_ck` + `deepgemm()` forwarder). Also hosts the `opus_gemm_a16w16_tune` deprecation shim. |
| [csrc/opus_gemm/opus_gemm_common.py](../../../csrc/opus_gemm/opus_gemm_common.py) | Kernel instance metadata (all kids live here) |
| [csrc/opus_gemm/opus_gemm_tune.py](../../../csrc/opus_gemm/opus_gemm_tune.py) | Offline tuner |
| [csrc/opus_gemm/gen_instances.py](../../../csrc/opus_gemm/gen_instances.py) | JIT codegen; `--tune_file` drives `opus_gemm_lookup.h` |
| [csrc/opus_gemm/opus_gemm.cu](../../../csrc/opus_gemm/opus_gemm.cu) | Runtime dispatch (two-level) + pybind entries |
| [csrc/opus_gemm/include/pipeline/](../../../csrc/opus_gemm/include/pipeline/) | Kernel source (a16w16, flatmm, flatmm_splitk) |
| [op_tests/test_opus_a16w16_tune.py](../../../op_tests/test_opus_a16w16_tune.py) | Per-kid correctness |
| [op_tests/test_opus_a16w16_lookup.py](../../../op_tests/test_opus_a16w16_lookup.py) | Tuned-CSV lookup correctness |
| [op_tests/test_opus_a16w16_gemm.py](../../../op_tests/test_opus_a16w16_gemm.py) | End-to-end `gemm_a16w16_opus` (single-shape + CSV sweep) |
| [op_tests/check_opus_splitk_graph.py](../../../op_tests/check_opus_splitk_graph.py) | HIP graph smoke |

---

## 10. Related Plans

- [splitk_flatmm_aiter](/.cursor/plans/splitk_flatmm_aiter_446c6aa0.plan.md) — splitk kernel integration (kid 200..210, 17-column CSV schema, validation matrix).
- [opus_a16w16_refactor](/.cursor/plans/opus_a16w16_refactor_71298e24.plan.md) — this module's refactor (PR1: layout + shim; PR2: two-level dispatch + Python wrapper).

Future work (separate plans / PRs):

- Plug `opus` into `aiter.tuned_gemm.solMap` so `aiter.gemm_a16w16` can
  pick `libtype=opus` from a merged global tuned CSV.
- Fill the a8w8 / a8w8_blockscale Python interfaces under
  `aiter/ops/opus/`, mirroring this module's shape.
- Populate `a16w16_flatmm_kernels_list` (kid 100..115) and extend
  heuristic's `64 < M ≤ 128` branch to prefer flatmm when aligned.
