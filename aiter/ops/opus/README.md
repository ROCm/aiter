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

**Inputs**: `A` is `[M, K]` or `[batch, M, K]` bf16. `B` is bf16
(plain layout, not pre-shuffled) in one of two shapes:

- `[N, K]` — only when `batch == 1`.
- `[batch, N, K]` — must be contiguous (strides `(N*K, K, 1)`); broadcast
  views like `B.unsqueeze(0).expand(batch, -1, -1)` are rejected because
  the opus launcher hardcodes `stride_b_batch == N*K`. Use
  `B.expand(batch, -1, -1).contiguous()` (or pass a real per-batch
  weight) when you need to broadcast.

**Output**: `[M, N]` or `[batch, M, N]`. bf16 and fp32 both supported on
all bias-aware kid families (split-barrier 4..9 and a16w16_flatmm_splitk
200..299) — the splitk reduce kernel templated on `D_OUT` selects the
right path at launch time.

**Optional bias** (per-row, broadcast across N): pass via `bias=` with one
of two shapes:

- `[M]`           — broadcast across batch; requires `batch == 1`.
- `[batch, M]`    — per-batch row vector.

bias dtype must equal `dtype` (match-output convention). Bias is folded
into the fp32 accumulator before cast → output. Both the CSV-tuned and
heuristic-fallback paths carry bias through; see [§2 Dispatch](#2-how-dispatch-works)
for the routing rules.

### Constraints (hard rejects)

The launchers reject these up front to avoid silent miscompares:

| Constraint | Why |
|---|---|
| `K` must be **even** | The splitk pipeline accumulates a ~3-7% error on odd K (latent K-tail bug); split-barrier independently requires `ceil_div(K, B_K)` even. |
| `A` and `B` dtype = bf16 | a16w16-family kernels lock the input dtype at launch time. |
| `B` is `[N, K]` only when `batch == 1`; otherwise `[batch, N, K]` contiguous | `stride_b_batch == N*K` is hardcoded; broadcast views silently corrupt. |
| pre-shuffled B | not supported; pass plain layout. |
| `bias.dtype == dtype` | match-output; otherwise a host-side `TORCH_CHECK` fires. |
| `bias` shape ∈ {`[M]` (only `batch==1`), `[batch, M]`} | reduce / split-barrier kernels expect this exact layout. |

Scale / FP8 paths are handled by other opus submodules (a8w8 /
a8w8_blockscale, landing in follow-up PRs); they share the same
`module_deepgemm_opus` JIT build but expose their own Python entry
under `aiter.ops.opus.*`. Those paths currently reject non-empty
`bias` at the dispatcher.

---

## 2. How Dispatch Works

When the user calls `gemm_a16w16_opus(A, B)` without an explicit kernel
id, the wrapper routes the request through **two independent lookup
mechanisms plus a heuristic**:

```
gemm_a16w16_opus(A, B, bias=...)
  ├─ explicit kernelId=N?  ───yes──►  opus_gemm_a16w16_tune(N, ..., bias)
  │       (C++ dispatcher TORCH_CHECKs that kid is bias-aware
  │        when bias.has_value())
  │
  ├─ Python-side CSV lookup  ───hit──►  opus_gemm_a16w16_tune(solidx, splitK, bias)
  │       (key = (cu_num, M, N, K, bias, dtype, outdtype, scaleAB,
  │        bpreshuffle); lru_cache'd from
  │        aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv)
  │
  └─ miss ──► (optional autolog) ──► opus_gemm(..., bias) [C++]
                                         ├─ C++ compile-time lookup (same CSV
                                         │   baked into opus_gemm_lookup.h
                                         │   at JIT time; key is (M,N,K) only,
                                         │   bias is forwarded to the matched
                                         │   launcher)
                                         └─ miss ──► opus_a16w16_heuristic_dispatch
                                                     (hand-written if-else by M;
                                                      always returns a bias-aware
                                                      kid family, so bias is safe
                                                      to forward unconditionally)
```

Two CSV lookups coexist on purpose:

- The **Python layer** reads the CSV live every new process; edits to
  the CSV take effect immediately. Its key includes `bias`, so `bias=False`
  and `bias=True` rows do not collide.
- The **C++ layer** (generated into `opus_gemm_lookup.h` by
  `gen_instances.py --tune_file`) serves the `opus_gemm()` C++ entry
  and has zero Python overhead per call, but **requires
  `AITER_REBUILD=1` to pick up CSV edits**. The C++ map keys on
  `(M, N, K)` only — it is bias-agnostic and relies on the lookup
  containing only bias-aware kid families (`a16w16_flatmm_kernels_list`
  is empty by default).

Explicit `kernelId=` bypass exists for tuning, debugging, and future
integrations (e.g. `aiter.tuned_gemm.solMap["opus"]`). The C++
dispatcher gates `bias` to bias-aware kid ranges (split-barrier 4..9
or a16w16_flatmm_splitk 200..299) when `bias.has_value()`; passing
bias to a non-bias-aware kid is a hard error.

---

## 3. Tuning Your Shapes

The shipped `aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv` covers
a handful of models. For anything else the heuristic kicks in, which is
correct but not necessarily fastest. To get peak perf on your shapes:

### 3.1 One-shot: run the tuner directly

```bash
# bias=False shapes (legacy bf16_untuned_gemm.csv layout)
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -i aiter/configs/bf16_untuned_gemm.csv \
    -o aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv

# bias=True shapes (e.g. gpt-oss; the input CSV's `bias` column is
# already True for every row)
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -i aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv \
    -o aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv

# single-shape with bias from CLI (no input CSV)
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -m 128 -n 2880 -k 4096 --bias --dtype bf16 --outdtype bf16
```

The input CSV must have at least `M, N, K` columns; `bias`, `dtype`,
`outdtype` columns are honored when present and override the CLI
defaults. Missing columns fall back to `--bias` / `--dtype` /
`--outdtype` at the CLI. `-o` defaults to `$AITER_OPUS_A16W16_TUNED_CSV`,
which points at the shipped opus-private tuned CSV (see
[§6 Environment](#6-environment)).

The tuner's dedup / `self.keys` is a 7-tuple
`(cu_num, M, N, K, bias, dtype, outdtype)`, so the same `(M, N, K)`
shape can be tuned independently with `bias=True` and `bias=False` (and
similarly across `outdtype`). bias=True candidates are restricted to
the bias-aware kid families (split-barrier 4..9 and splitk 200..299);
flatmm kid 100..115 are filtered out before mp_tuner sees them.

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
| `A` | Tensor | required | `[M, K]` or `[batch, M, K]` bf16. **K must be even.** |
| `B` | Tensor | required | `[N, K]` (batch=1 only) or contiguous `[batch, N, K]` bf16; broadcast views are rejected (see §1). |
| `bias` | Tensor? | `None` | Optional per-row bias. Shape `[M]` (broadcast across batch; requires `batch == 1`) or `[batch, M]`. dtype must equal `dtype` (match-output). Bias is folded into fp32 acc before cast. Honored on split-barrier (kid 4..9) and splitk (kid 200..299) families; the C++ dispatcher rejects bias on other kids. CSV-miss requests fall through to the heuristic dispatcher, which always returns a bias-aware kid. |
| `dtype` | torch.dtype | `bf16` | Output dtype; both `bf16` and `fp32` are supported on every kid family (the splitk reduce kernel templated on `D_OUT` casts at launch time). |
| `kernelId` | int? | `None` | Override: skip CSV/heuristic and launch this specific instance. With `bias is not None`, must be a kid in `[4, 10) ∪ [200, 300)`. |
| `splitK` | int? | `None` | Only honored with explicit `kernelId`; literal KBatch for splitk kids. |
| `out` | Tensor? | `None` | Reuse a preallocated output buffer. |

### `opus_gemm_a16w16_tune(XQ, WQ, Y, bias=None, kernelId=0, splitK=0)`

Low-level id-based dispatcher. Used by the tuner and the high-level
wrapper. Accepts 3D inputs only (`[batch, M, K]`, `[batch, N, K]`,
`[batch, M, N]`) and requires contiguous strides on all three tensors
(`(M*K, K, 1)`, `(N*K, K, 1)`, `(M*N, N, 1)` respectively); a Python
guard raises `NotImplementedError` for broadcast / transpose / slice
views before launching the kernel. `bias` is optional and follows the
same shape / dtype rules documented for `gemm_a16w16_opus`.

For backwards compatibility, the legacy 5-arg call form
`opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)` (positional `int`
in slot 4) still works: when the 4th positional argument is an `int`,
it is silently reinterpreted as `kernelId` and the rest of the args
shift accordingly. Mixed-style calls
(`..., bias=t, kernelId=k`) keep their kwargs semantics. Prefer
`gemm_a16w16_opus` unless you need explicit control.

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
  emitted. Requires even `ceil_div(K, B_K)` (and the cross-family
  `K % 2 == 0` rule). Supports per-row bias via two specializations
  (`HAS_BIAS = true / false`); the launcher dispatches at runtime on
  `bias.has_value()`.
- **Warp-specialized flatmm_splitk** (kid 200..210): 4-wave warp-spec
  kernel with runtime splitK (literal KBatch), fp32 workspace, reduce
  kernel casts to bf16/fp32 Y. Only `<fp32_t>` main-kernel
  instantiations emitted; the reduce kernel is templated on `D_OUT`
  and dispatches `__bf16` / `float` at launch time, so both bf16 and
  fp32 Y are valid. Handles arbitrary even-K / any N via `mask_va_tail`
  + reduce-kernel tail path. Bias is folded inside the reduce kernel
  via SGPR scalar load (`s_load_dword`), per-row, in fp32 acc before
  cast — reduce kernel emits 4 specializations
  (`{__bf16, float} × {HAS_BIAS true, false}`) so non-bias callers
  pay no bias-add overhead.

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
| `M > 128`, misaligned | kid 200 splitk | splitk tolerates arbitrary N (per-element tail store) |

The same heuristic table is used for both `<bf16_t>` and `<fp32_t>`
specializations; the splitk kid 200..210 main-kernel instantiation is
locked to `<fp32_t>` (its traits `static_assert(D_C == float)`), but
the reduce kernel honors the actual `Y.dtype()` so both bf16 and fp32
output land on the same heuristic kid. Every kid the heuristic returns
supports bias (`HAS_BIAS=true`); CSV-miss requests with bias are
forwarded to the same path unchanged.

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

### `RuntimeError: K=... must be even`

The a16w16-family launchers reject odd `K` because the splitk pipeline
silently accumulates a ~3-7% maxdelta on odd K (latent K-tail bug;
e.g. `K=257` / `513`). Even K is unaffected. Pad / round your `K` to
an even number (typically 4-aligned for VEC_A=8 layout) or wait for
the K-tail handling fix.

### `RuntimeError: bias is currently only supported on a16w16 split-barrier kids [4, 10) or a16w16_flatmm_splitk kids [200, 300)`

Triggered when an explicit `kernelId` outside the bias-aware ranges is
passed together with a non-empty `bias`. Pick a kid in `[4, 10) ∪
[200, 300)`, or drop the explicit override and let the dispatcher pick
a bias-aware kid.

### `Kernel id N not found in a16w16 ... tune lookup table`

The CSV references a kid that the current JIT build didn't compile
(usually a flatmm kid 100..115 from an older tuning run, since
`a16w16_flatmm_kernels_list` is currently empty). Either re-tune the
affected shapes, or just let `test_opus_a16w16_lookup.py` report them
as `SKIP` (it handles this explicitly).

### Why the cross-family `K % 2 == 0` rule exists

Two independent K-tail problems on the a16w16 family motivate the
launcher-side `TORCH_CHECK(K % 2 == 0, ...)`:

- Split-barrier (kid 4..9): the prefetched double-buffer reads one
  tile past the valid K range and corrupts the accumulator on
  `ceil_div(K, B_K)` odd. The launcher additionally enforces
  `loops_ % 2 == 0`, which already covers most cases (B_K is 32 or 64,
  so K must be a multiple of B_K; K must therefore be 64 / 128 aligned
  in practice).
- Splitk (kid 200..299): on odd K (e.g. 257 / 513) the
  `mask_va_tail` + reduce-tail interplay yields a 3-7% maxdelta vs.
  reference, while even K stays at the bf16 noise floor. The exact
  root cause is still under investigation.

The launchers reject odd K uniformly to give callers a clear error
instead of silent miscompares; relax once the underlying handling is
fixed.

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
  `aiter/ops/opus/`, mirroring this module's shape; extend bias support
  through them.
- Fix the splitk K-tail accumulation on odd K and lift the `K % 2 == 0`
  launcher assert.
- Optionally repopulate `a16w16_flatmm_kernels_list` (kid 100..115)
  with a bias-aware warp-spec epilogue (currently empty; the splitk
  pipeline with `splitK=0` covers the same shapes bit-identically).
