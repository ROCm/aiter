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

First call triggers a JIT build of `module_deepgemm_opus` (~11s on
the dev container; see [§7.6](#76-compile-time-techniques)).
Subsequent Python processes reuse the compiled `.so`.

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
| GPU arch must be **gfx950 (MI350)** today | opus uses gfx950-only intrinsics (MFMA-32x32x16, ds_read_b64_tr) and the 160 KiB LDS budget. Three-layer enforcement: Python import-time `_check_arch` rejects the package on non-gfx950 devices; the C++ host dispatcher routes per `gcnArchName` and currently only implements the gfx950 branch (others fail with a clear "pipeline TBD" message); each `__global__` kernel body wraps real code in `#if defined(__gfx950__)` so multi-arch wheels (e.g. `GPU_ARCHS='gfx942;gfx950'`) still compile, but the gfx942 device pass produces an empty kernel stub that is unreachable at runtime. To add support for a new arch, extend `OpusGfxArch` in `csrc/opus_gemm/opus_gemm.cu` and add a per-arch dispatch function. |

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

### 7.6 Compile-time techniques

JIT build of `module_deepgemm_opus` is on the user-visible critical
path (first call into any opus entry point on a fresh checkout pays
for it). Five landed rounds of optimization, in order:

1. **Host/device pass split** -- the codegen-emitted `.cuh` files
   guard their `<torch/all.h>` + launcher body behind
   `#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)`,
   so the device pass parses ~10K lines instead of ~70K.
2. **Fusion** -- 38 per-kid host TUs collapse into one
   `all_instances_host.cu`. The heavy `<torch/extension.h>` parse
   only happens once per module rebuild instead of 38 times.
3. **Torch removal in launchers** (mirrors PR #2932 for quant) --
   the dispatcher entry points and the codegen-emitted launcher
   bodies use `aiter_tensor_t` (POD, defined in
   `csrc/include/aiter_tensor.h`) instead of `torch::Tensor`.
   `<torch/all.h>` is replaced with a ~200-line header. The
   splitk launcher's fp32 workspace is allocated stream-ordered
   via `hipMallocAsync` / `hipFreeAsync` instead of `torch::empty`.
4. **Dispatcher TU torch removal** -- one stale
   `#include "py_itfs_common.h"` in `opus_gemm_arch_gfx950.cuh`
   was still pulling `<torch/all.h>` + the full `<ATen/...>` stack
   into the dispatcher TU even after step 3. Replaced with the
   torch-free `opus_gemm_utils.cuh` (same `bf16_t` / `fp32_t`
   aliases). Drops dispatcher TU preprocessed input from 401K
   lines to 154K.
5. **Lookup map: `unordered_map` + `std::function` -> sorted flat
   array + function pointer + `std::lower_bound`** -- the runtime
   `(M,N,K) -> kernel` and `kid -> kernel` tables used to be
   `std::unordered_map<..., std::function<...>>`, which alone
   added ~1s of frontend / template instantiation per dispatcher
   TU because of the `std::function` + hashtable templates. The
   replacement is a `static constexpr` array of POD entries
   `{shape, kernel_function_pointer}` plus
   `std::lower_bound`. No template instantiation overhead, no
   heap allocation on first call, faster runtime lookup.
6. **`splitk_reduce_kernel` carved out of every splitk kid's
   device.cu into one dedicated `splitk_reduce.device.cu`** --
   the 4 reduce specialisations (D_OUT bf16/fp32 x HAS_BIAS
   true/false) used to be appended to every splitk kid's
   `template __global__` instantiation list, so all 23 splitk
   TUs each compiled the 4 reduce kernels redundantly. Linker
   deduped the resulting weak symbols, but each TU still paid
   the full RA + ISA emit cost on its own compile (~0.3-0.5s
   per TU x 23 = ~9s of duplicated CPU work). Now they live in
   a single 0.3s TU.
7. **`#pragma unroll` on `tiled_mma_adaptor` MMA-tile loops in
   `opus.hpp`** -- the runtime
   `for (I = 0; I < EXPAND_K * EXPAND_M * EXPAND_N; I++)`
   outer loop and the inner `for (j = 0; j < a_len; j++)`
   extract / insert loops in the `vector_t` overload of
   `operator()` and `step_k` were relying on clang's default
   unroll heuristic to fold trip counts that are all
   constexpr. For small tiles (e.g. non-splitk's
   2x2x2 = 8 outer iters) clang did unroll them; but for the
   large splitk tiles
   (`flatmm_splitk_64x96x64_wgpcu1`'s 4x6x2 = 48 outer iters
   x mma_a_len=4 inner, 192 reads in total) the loop was
   left as a runtime loop over the `vtype_a` / `vtype_b` /
   `vtype_c` register arrays. GFX9 has no "load VGPR by
   runtime index" instruction, so LLVM expanded each
   `s_a[j] = a[i_a + j]` into an N-way
   `s_cmp_eq + s_cselect_b64 + v_cndmask_b32` select tree.
   Result on the worst kid: 8931 SGPR spills, 12100
   `s_cselect_b64`, 17956 `v_writelane_b32`, 388 KB ISA, and
   5.2s of LLVM Greedy RA Evict time = 7.7s slowest TU wall
   (the build's critical path). Forcing the unroll lets every
   index resolve at compile time, eliminates the select trees,
   and collapses register pressure. Numbers after the fix on
   the same kid:

   | Metric | Before | After | Change |
   |---|---:|---:|---:|
   | Slowest TU wall | 7.7s | **1.3s** | **-83%** |
   | ISA size | 388 KB | 27 KB | -93% |
   | `s_cselect_b64` | 12 098 | 2 | -99.98% |
   | `v_writelane_b32` | 17 956 | 0 | -100% |
   | `sgpr_spill_count` | 8 931 | 0 | -100% |
   | `sgpr_count` | 106 | 46 | -57% |
   | `vgpr_count` | 423 | 310 | -27% |
   | `agpr_count` | 167 | 54 | -68% |
   | `private_segment` | 24 704 B | 0 | -100% |
   | ASM total lines | 69 426 | 1 958 | -97% |

   This change touches `opus.hpp` (a shared header used by
   opus_attn / opus_fmm / opus_gemm). Smaller tile configs
   that already unrolled spontaneously are unaffected: they
   unroll the same way they did before, just under the
   explicit pragma instead of the heuristic.

**Headline** (128-core demon_test, ROCm 7.2.2, `MAX_JOBS=102`,
3-trial average over `AITER_REBUILD=1` with cleared build dir):

| Build | wall time |
|---|---:|
| Pre-split baseline (38 .cpp's, each parses torch + has both passes) | **48.4s** (49.2 / 47.2 / 48.9) |
| Host/device split (38 .cpp's, kernel decl in `#ifdef`) | **32.5s** (31.8 / 34.2 / 31.6) |
| Fused host TU + 38 device TUs | **22.3s** (22.5 / 22.0 / 22.4) |
| + torch removal in launchers | **19.4s** (19.5 / 19.5 / 19.2) |
| + dispatcher torch removal + flat-array lookup | **14.0s** (14.1 / 14.0 / 14.0) |
| + dedicated splitk_reduce TU | **14.4s** (14.0 / 14.7 / 14.4) |
| **+ MMA-tile unroll on opus.hpp (current)** | **11.1s** (11.1 / 11.3 / 11.0) |
| **Saving vs. baseline** | **−37.3s (−77%)** |

Rounds 6 + 7 together flipped the build's critical path. Round
6 (dedicated reduce TU) doesn't move end-to-end wall on its own
because round 7's eventual fix is in the slowest kid's main
kernel, but it cuts ~9s of duplicated reduce codegen across all
splitk TUs (a real win on hardware with smaller MAX_JOBS).
Round 7 cracks the slowest-TU bottleneck the previous five
rounds had left untouched: every splitk TU drops to
~1.2-1.5s wall (vs the worst's 7.7s before), and the new
critical path is the pybind TU (4.7s, mostly pybind11 + libtorch
parse) and the fused host TU (2.4s). Perf on the 24-shape
dsv3+gptoss bf16 benchmark is unchanged across rounds 6+7
(geomean +0.37% per shape, well within measurement noise; total
+0.27% sum of best-kernel us).

Functional regression: `op_tests/test_opus_a16w16_tune.py` 30/30 and
`op_tests/test_opus_a16w16_lookup.py` 338/338 still pass.

**Per-TU breakdown** (single-TU `-ftime-report` wall):

| TU class | host pass | device pass | TU wall |
|---|---:|---:|---:|
| Pre-split `instance.cpp` | 13.6s | 13.3s | ~26s |
| Host/device-split `instance.cpp` | 11.7s | 1.2s (instantiations only) | ~15s |
| Fused `all_instances_host.cu` (with torch) | ~11.7s (one-time, all 38 launchers) | ~0.4s (device pass empty) | ~12s |
| Fused `all_instances_host.cu` (current, torch-free) | **2.05s** (FE 1.33s 65% / OPT 0.32s 16% / MCG 0.34s 17%) | 0.42s (empty) | **~2.5s** |
| Per-kid `*.device.cu` (typical a16w16) | ~0.10s (RTC) | ~1.5s (single Traits codegen) | ~1.8s |
| Per-kid `*.device.cu` (worst splitk: 64x96x64-wgpcu1, pre-round-7) | ~0.11s | **8.29s (MCG 6.93s 84%, RA 5.74s 77%)** | **~8.5s** |
| Per-kid `*.device.cu` (worst splitk: 96x64x128-wgpcu1, post-round-7) | ~0.10s | ~1.4s (clean codegen, 0 spill) | **~1.5s** |
| Pre-split dispatcher (`opus_gemm.cu`, with torch) | 12.5s | 15.0s | ~27s |
| Split dispatcher (with torch) | 12.5s | 0.4s | ~13s |
| Dispatcher after torch + lookup overhaul (current) | **1.43s** (FE 1.21s 85%) | 0.42s | **~2.0s** |
| Pre-split pybind | ~13s | ~13s | ~26s |
| Split pybind (current) | ~5s | 0.42s | ~5.5s |

The end-to-end wall is now bounded by **the slowest single
pybind TU (~4.7s, mostly pybind11 + libtorch parse on host
pass)** plus ninja schedule + link + Python startup overhead
(~6s). Critical path breakdown post-round-7:

```
opus_gemm_pybind.cu host pass    ~4.7s   ← pybind11 + libtorch parse
ninja + link + python startup    ~6.4s
total wall                       ~11s
```

The slowest device TU now finishes in 1.5s (vs 7.7s before
round 7); every device.cu's GPU codegen is bounded comfortably
under the pybind TU. Round 7's `#pragma unroll` on
`tiled_mma_adaptor` was the breakthrough -- see §7.6.1 below
for the original forensic breakdown of the spill blow-up that
the unroll fixed.

#### 7.6.1 Device-pass forensics on the slowest splitk kid (historical)

> Status: **fixed in round 7**. This section documents the
> diagnostic path for posterity. The numbers below are
> pre-fix and no longer reproducible -- they referred to the
> kernel before the `#pragma unroll` was added to the MMA-tile
> loops in `opus.hpp::tiled_mma_adaptor::operator()` /
> `step_k`. After the fix, this same kid compiles in 1.3s
> with 0 SGPR spills.

`flatmm_splitk_64x96x64_wgpcu1<fp32_t>` *was* the build's
critical-path TU. `hipcc -ftime-report` on it shows:

```
Pass 1 (DEVICE pass): 8.29s total
├── Front end:                0.66s ( 8%)   parse pipeline header + traits
├── Optimizer:                0.61s ( 7%)   middle-end opt passes
├── LLVM IR generation:       0.09s ( 1%)
└── Machine code generation:  6.94s (84%)   ← AMDGPU backend
    ├── Greedy Register Allocator: 5.74s (77% of total)
    │   └── Evict sub-pass:        5.22s (99% of RA)
    ├── Machine Instruction Sched: 0.56s ( 7%)
    └── ~150 other passes:         ~0.45s

Pass 2 (HOST pass): 0.11s   ← RTC short-circuits libtorch + libstdc++
```

GPU codegen metadata (extracted from the resulting fat binary):

| Metric | Value |
|---|---:|
| Main kernel ISA size | **388 KB** (vs ~8 KB for non-splitk kids) |
| `vgpr_count` (logical) | 423 |
| `agpr_count` (acc registers) | 167 |
| `sgpr_count` | 106 |
| `sgpr_spill_count` | **8931** |
| `vgpr_spill_count` | 0 |
| `private_segment_fixed_size` (scratch / wave) | 24 704 bytes |
| `group_segment_fixed_size` (LDS / WG) | 144 KB |
| `prefetch_k_iter` | 7 |
| `WG_PER_CU` | 1 (wgpcu1) |

**Why `Evict` runs for 5.2s** -- ISA size, register pressure
and SGPR spill all stem from one structural choice. `wgpcu1`
sets WG_PER_CU=1, which gives the kernel the entire CU's
registers + LDS. `prefetch_k_iter = LDS_total / per_iter_LDS = 7`
on this shape, so the K-loop carries **7 prefetch buffers'
worth of register tile state simultaneously**. With
`comrep=(2,6)` (2 M x 6 N MFMA tiles per consumer wave) that's
~336 independent vector-register live ranges open at once
across 256 physical VGPRs + 167 AGPRs. LLVM's Greedy RA enters
its `Evict` policy: every conflict triggers an enumeration of
candidate live ranges to spill, with recursive spill-cost
calculation. On this kernel the candidate count and recursion
depth combine into ~5s of pure RA work.

**Why `sgpr_spill_count = 8931`** -- the K-loop indexes the
prefetch buffers via `slot = issue_k % pfk` (a runtime int).
GFX9 has no "load VGPR by runtime index" instruction
(`v_movrels_b32` exists but is restricted), so for each of the
~192 register tile elements that need slot-indexed access, LLVM
expands the read into a 7-way `s_cmp_eq + s_cselect_b64 +
v_cndmask_b32` select tree:

```asm
s_cmp_eq_u32 s0, 1
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v130, v6, v183, vcc
s_cmp_eq_u32 s0, 2
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v130, v130, v7, vcc
... ×7 (one per prefetch slot) ×192 tile elements
```

Every `s_cselect_b64` produces a `vcc` (a 64-bit SGPR pair)
live range. With ~12 000 `s_cselect_b64` and ~12 000
`s_cmpk_eq_i32 / s_cmp_eq_u32` instructions in the kernel, the
SGPR pressure exceeds the gfx950 physical SGPR cap (~100
addressable). LLVM's AMDGPU backend handles the overflow by
**spilling SGPRs to VGPR lanes** (`v_writelane_b32 v252-v255,
sN, lane_idx`) instead of going to scratch memory. We see
8931 such spill points statically; the assembly contains 17956
total `v_writelane` / `v_readlane` instructions because each
spilled SGPR is reloaded once on average. Four whole VGPRs
(v252, v253, v254, v255) are reserved as 64-lane SGPR scratch.

Stack frame metadata confirms it:

```
Function: gemm_a16w16_flatmm_splitk_kernel<...wgpcu1...>
  private_segment_fixed_size = 24704 bytes
  ~96 x 256-byte slots (each = one 64-lane wave-level register
  tile spilled to scratch as a fallback when even the 4 VGPR
  scratch lanes can't hold a particular live range)
```

**Bottom line on this TU**: the 8.5s wall is the price of
choosing `prefetch_k_iter = 7` for runtime perf reasons. The
Greedy RA Evict cost and the SGPR spill blow-up are downstream
consequences of that one structural choice. Reducing wall here
requires either:

1. structurally lowering `pfk` (perf trade-off -- shallower
   K-pipeline = less L1/LDS reuse), or
2. teaching the kernel author to write the slot-rotation
   without runtime indexing (hand-unroll the slot dispatch),
   which removes the `s_cselect` chains entirely and probably
   cuts SGPR spill 10x.

Neither is a build-system change. The B-track flag sweep
(see §7.7) confirms no `-mllvm` knob recovers any meaningful
wall on this kernel.

**What the codegen actually emits** (post-fusion):

1. **`csrc/include/opus/hip_minimal.hpp`** — extended into a dual-pass
   header. Adds device-pass replicas of `threadIdx` / `blockIdx` /
   `blockDim` / `gridDim` (backed by `__builtin_amdgcn_*`, matching
   the `extern const __device__ __attribute__((weak))` ABI HIP uses)
   plus `__forceinline__` / `__noinline__` macro fallbacks. Lets the
   device pass skip `<hip/hip_runtime.h>` (~100K preprocessed lines)
   while keeping HIP source compatibility.

2. **`csrc/opus_gemm/include/opus_gemm_utils.cuh`** — three include
   modes:
   * `__HIP_DEVICE_COMPILE__` (any device pass): `<opus/hip_minimal.hpp>`.
   * `__HIPCC_RTC__` (RTC mode, set per-source on `*.device.cu`):
     `<opus/hip_minimal.hpp>` on both passes. The device TU's host
     pass is empty content-wise, so the bare minimal header is
     enough; `<hip/hip_runtime.h>` would be wasted parse and would
     also pull in `<hip/amd_detail/hip_fp16.h>` which depends on the
     wrapper that `__HIPCC_RTC__` short-circuits.
   * Otherwise: the full `<hip/hip_runtime.h>` + `<hip/hip_bf16.h>` +
     `<hip/hip_fp8.h>`. Used by `all_instances_host.cu`,
     `opus_gemm.cu`, `opus_gemm_pybind.cu`.

3. **`csrc/opus_gemm/gen_instances.py`** — restructured around three
   file shapes:
   * `impl/{name}.cuh` (one per kid): Traits aliases + launcher body.
     Three guard combinations: skip torch headers when the host pass
     is irrelevant (`__HIP_DEVICE_COMPILE__` or `__HIPCC_RTC__` set);
     pick `traits header + forward kernel decl` over `pipeline body`
     when `OPUS_FUSED_HOST_TU` is set (avoids the ODR clash on
     same-named layout helpers between pipeline headers).
   * `instances/all_instances_host.cu` (one for the WHOLE module):
     defines `OPUS_FUSED_HOST_TU`, includes `aiter_tensor.h` +
     `aiter_stream.h` + `<optional>` once, includes every kid's
     `.cuh`, and emits all `template void xxx<dtype>(...)`
     instantiations. The launcher's `<<<...>>>` calls produce
     undefined `__device_stub__` references. Wrapped in
     `#ifndef __HIP_DEVICE_COMPILE__` so the device pass sees an
     empty TU.
   * `instances/{name}_C{dtype}.device.cu` (one per kid, dtype):
     includes the kid's `.cuh` with neither `OPUS_FUSED_HOST_TU` nor
     `__HIP_DEVICE_COMPILE__` (so the full pipeline header IS
     visible), but with `-D__HIPCC_RTC__` from
     `flags_extra_hip_per_source` so the host pass takes the lean
     branch. Emits `template __global__ void kernel<...>(...)`,
     producing the host stub + device GPU IR that the linker pairs
     with the fused host TU's undefined references.

4. **Torch removal across the dispatcher graph** -- mirrors PR #2932
   (`csrc/kernels/quant_kernels.cu`):
   * **`csrc/opus_gemm/include/opus_gemm.h`** -- entry-point
     signatures take `aiter_tensor_t&` (POD,
     `csrc/include/aiter_tensor.h`) instead of `torch::Tensor&`,
     return `void`. The header costs ~200 preprocessed lines instead
     of ~50K.
   * **`csrc/opus_gemm/include/opus_gemm_arch.cuh`** and
     **`opus_gemm_arch_gfx950.cuh`** -- `TORCH_CHECK` →
     `AITER_CHECK`, `<c10/util/Exception.h>` → `aiter_hip_common.h`.
   * **`csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh`**
     -- `OpusA16W16NoscaleKernel` is now
     `std::function<void(aiter_tensor_t&, ...)>` so every dispatch
     map entry is torch-free.
   * **`csrc/opus_gemm/opus_gemm.cu`** -- both entry points
     (`opus_gemm`, `opus_gemm_a16w16_tune`) take `aiter_tensor_t`,
     use `AiterDtype` enum (`AITER_DTYPE_bf16` / `_fp32` / `_fp8`)
     instead of `at::ScalarType::*` / `torch_fp8`, return `void`.
   * **`csrc/opus_gemm/gen_instances.py`** -- the codegen-emitted
     launcher signatures use `aiter_tensor_t&`, the bias validator
     calls `AITER_CHECK` + `bt.is_contiguous() / dtype() / dim() /
     size()` (POD accessors that `aiter_tensor_t` provides
     PyTorch-compatible by design). The splitk launcher allocates
     its fp32 workspace with `hipMallocAsync(stream)` + matching
     `hipFreeAsync(stream)` after the reduce kernel, replacing
     `torch::empty(... TensorOptions().dtype(kFloat32).device(...))`
     while preserving the same stream-ordered lifetime invariant
     PyTorch's caching allocator gave us.
   * **`aiter/ops/opus/gemm_op_a16w16.py`** --
     `@compile_ops("module_deepgemm_opus", develop=True)` on both
     `_opus_gemm_a16w16_tune_raw` and `_opus_gemm_bf16_dispatch`.
     `develop=True` makes the JIT wrapper (a) inject the current
     torch CUDA stream into the C++ `aiter::getCurrentHIPStream`
     thread-local via `module._set_current_hip_stream` before the
     call and (b) auto-convert any `torch.Tensor` arg to
     `aiter_tensor_t` via `torch_to_aiter_pybind`. Because the C++
     side now returns `void`, `opus_gemm_a16w16_tune` keeps its
     `return Y` contract by returning the in-place tensor directly.

5. **`csrc/opus_gemm/opus_gemm.cu` and `csrc/pybind/opus_gemm_pybind.cu`**
   — entire-file `#ifndef __HIP_DEVICE_COMPILE__` skip. Pure host
   code with no `__global__` / `<<<>>>`; their device pass is dead
   weight (12.5–15s → 0.4s).
   `opus_gemm_pybind.cu` additionally registers
   `AITER_SET_STREAM_PYBIND` so Python can call
   `module._set_current_hip_stream(...)`.

6. **Per-file flag plumbing in `aiter/jit/`** — `core.py` reads the
   new `flags_extra_hip_per_source` dict from
   `optCompilerConfig.json` and forwards it through `_jit_compile`
   into `_write_ninja_file`, which emits a per-build
   `cuda_post_cflags = $cuda_post_cflags <extra>` override on
   matching ninja rules. Used by opus to apply `-D__HIPCC_RTC__` to
   `*.device.cu` only — the dispatcher / pybind TUs would break with
   it because they transitively pull in ck_tile / pybind11, both of
   which depend on the wrapper that RTC short-circuits.

**Why fusion was the right move on this hardware**: in the
host/device-split layout, the critical path was ~15s (each
`instance.cpp` had to parse `<torch/extension.h>` AND run device
codegen, in series inside one hipcc invocation). The fused host TU
detaches those: launcher instantiations all live in one .cu that
parses headers ONCE but skips device codegen entirely; per-kid
device codegen lives in 38 tiny self-contained .cu's that finish in
~2s each in parallel. After torch removal the fused TU's parse
drops from ~12s to ~7s, and that's now the end-to-end critical
path.

**What did NOT help** on this hardware:

- **MAX_JOBS tweaks**: aiter already auto-sets it to
  `min(80% × cpu_count, free_mem / 0.5 GB)` = 102 on the test host;
  CPU saturation isn't the bottleneck — the host TU's serial parse
  of `<torch/extension.h>` is.

### 7.7 Future compile-time work

The current 11.1s floor is now set by **the pybind TU at ~4.7s**
(mostly pybind11 + libtorch parse, neither of which the
dispatcher / launcher refactors can touch since the pybind
layer is the boundary with Python). The previous champion --
the slowest device TU's GPU codegen on
`flatmm_splitk_64x96x64_wgpcu1` -- went from 7.7s down to 1.3s
in round 7 once we found the SGPR spill root cause and added
`#pragma unroll` (see §7.6 round 7 + §7.6.1). Remaining attack
surface, in rough order of expected payoff vs. invasiveness:

1. **(MEASURED, NEGATIVE)** AMDGPU-side `-mllvm` flag sweep --
   the build's device passes inherit five aiter-global -mllvm
   flags (`--amdgpu-kernarg-preload-count=16`,
   `--lsr-drop-solution=1`, `-amdgpu-early-inline-all=true`,
   `-amdgpu-function-calls=false`, `-enable-post-misched=0`)
   plus opus-private `--amdgpu-mfma-vgpr-form`. Earlier
   speculation was that `-amdgpu-early-inline-all=true` +
   `-amdgpu-function-calls=false` (which together force every
   call to inline into one mega-function for the RA) might be
   responsible for the long Greedy RA Evict pass. We measured
   five build configurations:

   | Config | Slowest TU wall | Build wall (3-trial avg) | Perf vs baseline (geomean of 24 dsv3+gptoss bf16 shapes) |
   |---|---:|---:|---:|
   | baseline (all 6 flags on) | 7.66s | 18.1s | reference |
   | `-amdgpu-early-inline-all=false` override | 7.68s | 17.9s | +0.24% (slower by 0.24%) |
   | `-amdgpu-function-calls=true` override | 7.74s | 17.8s | -0.10% (faster by 0.10%) |
   | `--amdgpu-mfma-vgpr-form=false` override | 7.75s | 18.3s | +0.04% |
   | all three off | 7.71s | 17.9s | +0.06% |

   Both build wall and perf differences are within measurement
   noise (~1%). We also tried `-O1`, `--amdgpu-igrouplp-exact-solver=0`,
   and `-greedy-regalloc-eviction-max-iterations=2` (last one
   doesn't exist in ROCm 7.2.2 LLVM and was rejected). None
   reduced the slowest TU's wall by more than measurement
   noise. Conclusion: **the Greedy RA Evict cost on this kid
   is fundamental to the IR's register pressure**, not gated
   by any user-tunable -mllvm flag in this LLVM revision. See
   §7.6.1 for the underlying SGPR-spill / pfk=7 analysis. The
   five flags should stay in the global config because they
   improve perf on smaller kids elsewhere in aiter.

2. **Trim or split heavy splitk kid instantiations** (untried,
   high potential) -- the slowest 4-5 splitk kids (`*_64x96x*`
   and `*_96x64x*` family with `wgpcu1`) eat the entire ninja
   schedule's tail. Empirical sweep of tuned-CSV winners would
   reveal which of these are actually selected for production
   shapes; un-selected kids can be dropped at codegen time,
   removing them from the build entirely. Saving: depends on
   CSV coverage; potentially -3 to -5s end-to-end if the
   slowest 1-2 kids turn out unused.

3. **(LANDED, round 7)** Force unroll on tiled_mma_adaptor's
   MMA-tile loops -- root cause of the §7.6.1 SGPR spill
   blow-up turned out to be `opus.hpp`'s
   `for (I = 0; I < EXPAND_K * EXPAND_M * EXPAND_N; I++)`
   relying on clang's heuristic to unroll. For the worst
   splitk tile (4x6x2 = 48 outer iters x mma_a_len=4 inner)
   the heuristic gave up, the loop ran at runtime, and the
   `a[i_a + j]` reads compiled to N-way s_cselect select
   trees. Adding `#pragma unroll` to the five overloads in
   `tiled_mma_adaptor` (plus the inner extract / insert loops
   in the vector_t path) eliminated the spill problem
   entirely: slowest TU 7.7s -> 1.3s, 8931 spills -> 0,
   ASM 69k lines -> 2k. End-to-end wall 14.4s -> 11.1s.

4. **`ccache` / `sccache` integration** -- reuse `.cuda.o`
   across rebuilds when `gen_instances.py` produces a
   byte-identical TU. Pure infra change, complementary to all
   other items. The first build still pays full freight;
   subsequent rebuilds (e.g. `AITER_REBUILD=1` after a CSV-only
   edit) drop to seconds. This was deferred earlier because
   parsing was the bottleneck and parses don't compose well
   across rebuilds; with parse gone, MCG dominates and MCG
   output is much more cacheable.

5. **Header structure cleanup** (low priority) -- `opus.hpp` is
   3055 lines, parsed by every device TU on its device pass and
   by the fused host TU on its host pass. The fused TU now
   parses it in ~1.3s and that's already off the critical path;
   the per-device TU host pass is even shorter (~0.1s with RTC).
   Only worth doing for cleanliness, not for time.

Practical ceiling on this hardware (post-round-7):

- The new bottleneck is the **pybind TU at 4.7s** (mostly
  pybind11 + libtorch parse, neither easy to remove without
  abandoning the python binding).
- Fused host TU is 2.4s, dispatcher TU 1.8s, slowest device
  TU 1.5s -- the gap between pybind and the next slowest TU
  is ~2.3s, leaving room for ninja schedule / link / Python
  startup (~6s observed).
- If item 2 (CSV-driven kid trimming) lands and removes
  ~5 unused splitk variants: probably **~10s** end-to-end
  (slow TUs are already fast, savings are linear in TU count
  and amortized across MAX_JOBS=102 parallelism).
- Removing the pybind TU entirely (e.g. via a C-API shim
  similar to the dispatcher's torch-free refactor) would
  bring this to ~7-8s but requires wider-scope changes to
  the Python wrapper layer.

Items 1 (-mllvm flag sweep) and 3 (MMA-tile unroll) are
closed -- 1 measured negative, 3 landed in round 7. Items
4 + 5 are infrastructure / cleanliness, not on the wall
critical path.

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

The whole rebuild takes ~11s on dev hardware (128-core,
ROCm 7.2.2; see [§7.6 Compile-time techniques](#76-compile-time-techniques)
for the seven-stage optimization stack that drops it from ~48s).

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
