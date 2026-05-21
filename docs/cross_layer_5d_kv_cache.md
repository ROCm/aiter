# Cross-Layer 5D KV Cache Support in AITER (Prefill + reshape_and_cache)

> Status: implemented and tested for both the prefill *read* path
> (`mha_batch_prefill_func`, AITERKER-116) and the cache *write* path
> (`reshape_and_cache_flash`, which is the second bullet of both AITERKER-116
> *and* AITERKER-117 — they ask the same `reshape_and_cache` question).
> The decode *read* path (`PagedAttention.forward_decode`, AITERKER-117 first
> bullet) is a separate kernel and remains out of scope of this document.

## 1. Ticket context

Tencent's HY3 model on Kunlun XPU uses a "Cross-Layer KV Cache" layout where the
KV cache is allocated as a single 6D physical tensor and exposed per-layer as a
non-contiguous 5D view. They asked AMD to provide the same support in AITER so
HY3 can run on MI-class GPUs through the same vLLM integration. The two source
specs that drive this work are kept in the AITER repo root for reference:

- `Cross-Layer_5D_KV_Cache_Operator_Adaptation_Plan_EN.md`
- `Hover_Cache_EN.md`

The original ticket asks three things:

1. Does `mha_batch_prefill_func` ([aiter/aiter/ops/mha.py](../aiter/ops/mha.py))
   already accept non-contiguous KV cache tensors? If not, make it. — **covered
   in Part A below.**
2. Does `reshape_and_cache` (vLLM's `csrc/cache_kernels.cu`) accept non-contiguous
   KV cache? If not, fix or suggest an alternative. — **covered in Part B
   below.**
3. There is no Tencent-side Python repro yet; we have to validate this on synthetic
   inputs that mirror the doc's section 9.2 numeric example.

The companion "Hover Cache" design (a GPU/CPU/KVStore tiered prefix-cache
offloading connector) is a vLLM Scheduler/Worker concern; it requires no kernel
work by itself and is discussed only briefly in §6.

---

# Part A: Prefill read path (`mha_batch_prefill_func`)

The user-visible call chain we had to teach the new layout to is:

```
mha_batch_prefill_func (Python)
  -> _mha_batch_prefill (Python helper)
    -> mha_batch_prefill (@compile_ops binding, pybind)
      -> aiter::torch_itfs::mha_batch_prefill (C++ wrapper, validates and packs args)
        -> aiter::mha_batch_prefill (dispatch shim)
          -> ck_tile::fmha_batch_prefill (CK Tile FMHA kernel)
```

## 2. What the new layout looks like

The framework allocates a contiguous 6D physical buffer

```
(num_blocks, num_kv_heads, num_layers, 2, page_size, head_dim)
```

and exposes a per-layer 5D logical view

```
shape:   (2, num_blocks, num_kv_heads, page_size, head_dim)
stride:  (B*D, L*2*H*B*D, L*2*B*D, D, 1)
                        where B=page_size, D=head_dim, H=num_kv_heads, L=num_layers
```

This view is intentionally **non-contiguous**. Calling `.contiguous()` per layer
per step would copy the entire KV cache (multiple GB) every forward pass; that's
the option Tencent explicitly rejects. So the operator side has to consume the
view as-is and walk it through stride metadata.

Two properties of the layout make it tractable for CK Tile:

- `stride[-1] == 1` — the head_dim dimension is still innermost-contiguous, so
  CK's 16-byte (`dwordx4`) vector loads along `head_dim` keep working.
- `stride[3] == head_dim` — the page-offset (token-in-page) stride matches what
  the kernel already expects for a linear layout; only the per-head and per-block
  strides change relative to the packed `LINEAR_LAYOUT`.

The full layout zoo we now support in `mha_batch_prefill_func`:

| Layout name                  | Rank | K/V shape                           | Source                                |
|------------------------------|------|-------------------------------------|---------------------------------------|
| `VECTORIZED_LAYOUT`          | 5D   | `[N, H, D/V, B, V]` (swizzled)      | existing (SGLang/vLLM vectorized)     |
| `LINEAR_LAYOUT` (4D)         | 4D   | `[N, B, H, D]`                      | existing (SGLang/vLLM linear)         |
| `LINEAR_LAYOUT` (3D, page=1) | 3D   | `[N, H, D]`                         | existing                              |
| `LINEAR_HEADS_FIRST_LAYOUT`  | 4D   | `[N, H, B, D]` (non-contiguous)     | **new**: Tencent Cross-Layer 5D view  |

The new variant differs from the existing 4D linear by one transposition: heads
sit before page in the dim order, mirroring what falls out of slicing
`logical_6d[layer_idx][kv_idx]`.

## 3. Feasibility verdict and what we did NOT change

CK Tile's `fmha_batch_prefill` kernel already does all its K/V address
arithmetic through user-supplied stride multiplicands
(`stride_k`, `nhead_stride_k`, `batch_stride_k`, and the `page_block_size` page
indirection). The kernel itself does not need to know about the new layout —
giving it the right strides is sufficient.

Concretely, **the following were intentionally left untouched**:

- CK Tile FMHA `batch_prefill` kernel templates, pipeline, codegen.
- `ck_tile::fmha_batch_prefill_args` struct.
- Page-table abstractions (`SGLANG_PAGE_TABLE_1D`, `VLLM_BLOCK_TABLE_2D`).
- The JIT recipe (`aiter/jit/optCompilerConfig.json`) for
  `module_mha_batch_prefill` — same generated kernel instances cover both linear
  layouts because the kernel arithmetic is identical.

The dispatch boundary in `aiter::mha_batch_prefill` collapses
`LINEAR_HEADS_FIRST_LAYOUT` to `LINEAR_LAYOUT` so that the existing generated
kernel instances handle the new strides without any codegen churn.

## 4. Implementation

Seven files were changed in total.

### 4.1 New enum value (CK Tile, header-only)

[`aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp)

Added `LINEAR_HEADS_FIRST_LAYOUT = 2` alongside the existing
`VECTORIZED_LAYOUT = 0` and `LINEAR_LAYOUT = 1`, with a doc comment naming the
Tencent cross-layer origin and explaining that the kernel treats it as
`LINEAR_LAYOUT` at dispatch time (only the AITER wrapper distinguishes them
when extracting strides). **No CK kernel template, codegen, or instance files
were touched.**

### 4.2 AITER C++ wrapper

[`aiter/csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`](../csrc/py_itfs_ck/mha_batch_prefill_kernels.cu)

Three additions:

1. A new branch inside `get_ck_fmha_batch_prefill_args` that reads K/V strides
   positionally for the `[N, H, B, D]` layout:

   - `stride_k = k.stride(2)` — the page (token-in-page) stride, equals `head_dim`.
   - `nhead_stride_k = k.stride(1)` — the per-head stride, equals
     `num_layers * 2 * page_size * head_dim` in the cross-layer view.
   - `batch_stride_k = k.stride(0)` — the per-block stride, equals
     `num_layers * 2 * num_heads * page_size * head_dim`.

   The packed-layout invariants
   (`page_stride >= num_heads * head_stride`,
   `batch_stride >= page_size * page_stride`) are intentionally dropped here —
   they don't hold for the cross-layer view and were the original blocker.

2. **Alignment guards** for the new branch:
   `TORCH_CHECK(nhead_stride_k % k_vector_size == 0)` and the same for
   `nhead_stride_v`, `batch_stride_k`, `batch_stride_v`, with error messages
   that explicitly call out `LINEAR_HEADS_FIRST_LAYOUT; required for 16B
   vectorized loads`. This is the single hardware constraint the new layout
   imposes (CK Tile uses 128-bit vector loads along `head_dim`).

3. The `nhead_stride_k/v` extraction was updated to recognize the new layout
   (head dim sits at index 1, not 2, for heads-first).

The top-level `mha_batch_prefill` function gained an `int kv_layout` parameter
(`-1 = auto`, `0 = vectorized`, `1 = linear`, `2 = linear_heads_first`) plus
matching `CHECK_SHAPE` branches.

### 4.3 Dispatch normalization

[`aiter/csrc/cpp_itfs/mha_fwd_batch_prefill.cu`](../csrc/cpp_itfs/mha_fwd_batch_prefill.cu)

Before constructing `traits`, the shim collapses
`LINEAR_HEADS_FIRST_LAYOUT` → `LINEAR_LAYOUT`. Comment explains why this is
safe: the CK kernel computes `addr = base + head_idx * nhead_stride + page *
batch_stride + tok * stride` and never inspects the enum at runtime, so the
heads-first variant reuses the existing generated kernel instances with
different stride numbers.

### 4.4 Header and pybind

- [`aiter/csrc/include/torch/mha_batch_prefill.h`](../csrc/include/torch/mha_batch_prefill.h) — added `int kv_layout = -1` to the C++ declaration with a doc comment of the four values.
- [`aiter/csrc/include/rocm_ops.hpp`](../csrc/include/rocm_ops.hpp) — added `py::arg("kv_layout") = -1` to the `MHA_BATCH_PREFILL_PYBIND` macro.

### 4.5 Python wrapper

[`aiter/aiter/ops/mha.py`](../aiter/ops/mha.py)

Added module-level constants kept in lock-step with the CK enum:

```python
KV_LAYOUT_AUTO = -1
KV_LAYOUT_VECTORIZED = 0
KV_LAYOUT_LINEAR = 1
KV_LAYOUT_LINEAR_HEADS_FIRST = 2
```

Plumbed `kv_layout` through `cmdGenFunc_mha_batch_prefill`,
`mha_batch_prefill_fake_tensors`, the `@compile_ops`-decorated
`mha_batch_prefill`, and the internal `_mha_batch_prefill` helper.

`mha_batch_prefill_func` grew two new kwargs:

```python
def mha_batch_prefill_func(
    q, k=None, v=None, ...,
    kv_cache=None,                    # 5D [2, N, H, B, D] view
    kv_layout=KV_LAYOUT_AUTO,         # explicit layout override
):
```

Usage from the framework side (the ergonomic path):

```python
# Framework allocates the 6D physical buffer and produces a per-layer 5D view.
kv_cache_view = logical6d[layer_idx]   # shape (2, N, H, B, D), non-contiguous

# Call into the prefill kernel without slicing K/V manually:
out = aiter.mha_batch_prefill_func(
    q,
    cu_seqlens_q=cu_seqlens_q,
    kv_indptr=kv_indptr,
    kv_page_indices=kv_page_indices,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    kv_cache=kv_cache_view,            # auto-slices and forces heads-first layout
    kv_last_page_lens=kv_last_page_lens,
    causal=causal,
)
```

Behavioral guarantees:

- `kv_cache=` is mutually exclusive with explicit `k=`/`v=`.
- When `kv_cache=` is supplied, `kv_layout` is forced to
  `KV_LAYOUT_LINEAR_HEADS_FIRST` unless the caller explicitly says otherwise
  (and is rejected if the override is inconsistent).
- Power users may still pass pre-sliced K and V plus an explicit
  `kv_layout=KV_LAYOUT_LINEAR_HEADS_FIRST` if the framework already does the
  slicing.

### 4.6 Tests

[`aiter/op_tests/test_batch_prefill.py`](../op_tests/test_batch_prefill.py)

Three new tests next to the existing `kv_blockscale` ones:

| Test | What it covers |
|---|---|
| `test_batch_prefill_cross_layer_5d_layout_strides` | Pure metadata check that the per-layer 5D view's strides match the doc section-5.3 formulas `(B*D, L*2*H*B*D, L*2*B*D, D, 1)`. Parametrized over `num_layers ∈ {4, 80}` and `layer_idx ∈ {0, 1}`. |
| `test_batch_prefill_cross_layer_5d_layout_rejects_layout_mismatch` | Validates the wrapper rejects inconsistent inputs (both `kv_cache` and `k/v`, or `kv_layout` mismatch). |
| `test_batch_prefill_cross_layer_5d_layout_matches_contiguous` | Functional GPU test: runs `mha_batch_prefill_func` twice on the same numerical data — once via packed contiguous `[N, B, H, D]` `LINEAR_LAYOUT`, once via the non-contiguous cross-layer `[N, H, B, D]` `LINEAR_HEADS_FIRST_LAYOUT` view sliced from the 6D buffer — and asserts the outputs match. 64 parametrizations: `dtype ∈ {bf16, fp16}` × `causal ∈ {False, True}` × `qo_len ∈ {16, 32}` × `batch_size ∈ {1, 2}` × `GQA ∈ {(1,4), (2,8)}` × `layer_idx ∈ {0, 2}`. |

Helper `_make_cross_layer_5d_view(...)` reproduces the framework-side
construction from sections 1.1/2.2 of the Tencent doc (allocate 6D, permute to
logical 6D, index by `layer_idx` to get the 5D non-contiguous view).

## 5. Verification

Ran on MI-class hardware (`ROCm 7.1.25424`, `torch 2.9.1+rocm7.1.0`,
PyTorch's HIP runtime).

```
test_batch_prefill_cross_layer_5d_layout_strides                4 passed
test_batch_prefill_cross_layer_5d_layout_rejects_layout_mismatch 1 passed
test_batch_prefill_cross_layer_5d_layout_matches_contiguous      64 passed
                                                          ───────────────
                                                          69 / 69 passed
```

Compile cost: the first run JIT-builds the
`mha_batch_prefill_bf16_nlogits_nbias_nmask_nlse_ndropout_nqscale` variant in
~27s; subsequent runs reuse the cached `.so`.

### Migration gotcha

AITER ships pre-built `.so` files for the `module_mha_batch_prefill` variants
under `aiter/jit/`. Because this patch changes the pybind signature of
`mha_batch_prefill` (adds the `kv_layout` argument), **anyone applying this
patch on top of an existing build must force a JIT rebuild** by deleting the
stale `.so` files:

```bash
rm -f aiter/aiter/jit/mha_batch_prefill_*.so \
      aiter/aiter/jit/build/lock_mha_batch_prefill_* \
      aiter/aiter/jit/build/mha_batch_prefill_*/build/lock
```

The JIT cache key isn't perfectly invalidated by source-only edits in this
case; without the manual clean any pre-existing causal-mask variant binary
will raise `TypeError: ... got unexpected positional argument` when the test
suite reaches a causal path. This bit me during initial test runs (see the
"Lessons learned" appendix).

## 6. What about Hover Cache?

The companion `Hover_Cache_EN.md` describes a vLLM Scheduler / Worker
offloading connector (GPU L1 ↔ CPU L2 ↔ Persistent KVStore L3). It is a
purely framework-side design and **requires no kernel work** to land. The only
kernel-side touchpoint is that, after Hover Cache asynchronously restores
prefix blocks from CPU/KVStore into the GPU KV buffer, the prefill kernel
needs to be able to consume whatever GPU-resident view the framework
constructs — which is exactly what the Cross-Layer 5D work in this document
enables. The two features are independent on the AITER side and can be rolled
out in either order.

## 7. Out of scope / follow-up tickets

- **AITERKER-117** — `PagedAttention.forward_decode`
  ([aiter/aiter/paged_attn.py](../aiter/paged_attn.py)). The decode kernel is a
  separate code path and is not touched here. It needs the analogous
  non-contiguous KV support before HY3 can run end-to-end on AMD with the
  cross-layer cache enabled. Useful starting context for whoever picks 117 up:

  - `PagedAttention.forward_decode` calls into `torch.ops.aiter.paged_attention_rocm`,
    which routes to the legacy ROCm paged-attention kernel in
    [aiter/csrc/kernels/attention.cu](../csrc/kernels/attention.cu). The
    wrapper for that path reads `block_size = key_cache.size(3)`, which
    implicitly assumes the x-vectorized 5D cache layout
    `[N, H, D/x, B, x]` — different layout family from Tencent's cross-layer
    `[N, H, B, D]`.
  - **AITER already has two newer decode entry points** that handle the
    heads-first `[N, H, B, D]` ("HND") layout via runtime stride dispatch:
    `aiter.paged_attention_v1` ([attention_v1.cu](../csrc/kernels/attention_v1.cu))
    and `aiter.paged_attention_ragged`
    ([attention_ragged.cu](../csrc/kernels/attention_ragged.cu)).
    Both accept a `kv_cache_layout: str` argument (`"HND"` or `"NHD"`) and read
    the per-head / per-token-in-page strides positionally from
    `key_cache.stride(1)` / `key_cache.stride(2)` rather than hard-coding
    packed values. This means they should already accept the cross-layer
    non-contiguous strides for HND out of the box (innermost head_dim still
    contiguous), but it's worth running a synthetic-input test against the
    packed reference (mirror of `test_batch_prefill_cross_layer_5d_layout_matches_contiguous`)
    to confirm before relying on it.
  - Concrete options for 117:
    1. **Migrate HY3's vLLM integration** to call `paged_attention_v1` or
       `paged_attention_ragged` with `kv_cache_layout="HND"`. This is the
       lower-risk path — no new kernel, and the layout-dispatch wrappers
       already exist. Validation cost: a confidence test as above.
    2. **Extend `paged_attention_rocm` / `attention.cu`** to support HND with
       the same stride-driven address arithmetic. Higher cost (touch the
       legacy kernel) but lets existing `PagedAttention.forward_decode`
       callers opt into cross-layer without changing entry point.
- **`reshape_and_cache`** (vLLM x-vectorized variant in
  `csrc/cache_kernels.cu`). This kernel targets the legacy x-vectorized cache
  layout `[N, H, D/x, B, x]` which is a different layout family from Tencent's
  cross-layer 5D scheme; it is not the right kernel for HY3. The
  `reshape_and_cache_flash` writer (which targets `[N, B, H, D]`) **is**
  addressed in Part B below.
- **Reference Python test from Tencent**: NOTE#2 in the original ticket says
  Tencent will provide one. The synthetic tests in §4.6 and §B.5 are
  sufficient for kernel-level correctness, but the integration test against
  Tencent's actual KV-cache management should be run once the script is
  available.

## 8. Risks and open items

- **Vector-load alignment.** The new alignment guards
  (`nhead_stride_k * elem_size % 16 == 0`) will fire on configurations where
  the cross-layer strides aren't 16-byte-aligned. For the doc's nominal
  parameters (`D=128`, `B=16`, `H=1`, `L=80`, bf16) the per-head stride is
  5,242,880 bytes — well aligned. Pathological combinations (small odd `D`,
  bf16) would trip the check; the resulting error message points the framework
  at this layout and the failing stride.
- **MQA/GQA factor.** `nhead_stride_k = L*2*B*D` is independent of `H`, so the
  per-K-head stride is constant. Address arithmetic uses `nhead_k`-indexed
  strides, so GQA where `nhead_k < nhead_q` works unchanged.
- **Layout disambiguation.** `VECTORIZED_LAYOUT` (5D swizzled) and the new
  `LINEAR_HEADS_FIRST_LAYOUT` (4D heads-first) both originate from "5D" in
  Tencent's terminology but mean different things. The wrapper dispatches by
  `kv_memory_layout` enum, not by tensor rank, to avoid confusion; the public
  Python helper hides this by accepting the 5D `kv_cache=` view and inferring
  the right enum value internally.

## 9. Appendix: lessons learned during this work

Two non-obvious issues surfaced during verification and are worth recording for
future maintenance.

1. **JIT cache invalidation for pybind signature changes.** AITER's JIT
   compares modules by `md_name` and source file list; it does not detect that
   the pybind signature has changed when the only edit is to a source file
   already in the module. After this patch, any cached `.so` from an earlier
   build will be loaded by `torch.utils.cpp_extension` and will raise
   `TypeError` on the first call to `mha_batch_prefill` because `kv_layout`
   isn't in the binary's signature. The fix is the `rm` command in section 5.
   Worth considering: have `compile_ops` hash the pybind binding header into
   the cache key.

2. **`pytest -k` parametrization IDs are confusing.** Pytest renders the
   outermost `@pytest.mark.parametrize` first in the test ID, but for someone
   reading the failure list `dtype0-True-16-1-128-1-4-16-4-0-4` is opaque.
   For long-lived tests it's worth adding `ids=` to the more cryptic
   parametrize calls (e.g. `ids=["causal", "no_causal"]`) so failures are
   self-describing in CI output. Not done in this patch to minimize diff size,
   but recommended as a small follow-up.

---

# Part B: KV cache write path (`reshape_and_cache_flash`)

## B.1 The two candidate writer kernels in AITER

AITER ships two related writer kernels for KV cache, both mirroring vLLM:

| Kernel | Target cache layout | Used by |
|---|---|---|
| `reshape_and_cache` ([aiter/csrc/kernels/cache_kernels.cu](../csrc/kernels/cache_kernels.cu#L175)) | `[N, H, D/x, B, x]` (vLLM x-vectorized) | Legacy vLLM paged-attention pipelines |
| `reshape_and_cache_flash` (same file, L#251) | `[N, B, H, D]` (FlashAttention-style) | vLLM Flash-attn / FlashInfer paths and the prefill kernel covered in Part A |

Tencent's Cross-Layer 5D cache is a **non-x-vectorized** linear cache, so
`reshape_and_cache` (x-vectorized) is not the right writer kernel at all — it
would have to be rewritten end-to-end. `reshape_and_cache_flash` on the other
hand uses the same `[N, ?, ?, D]` family as the prefill kernel and is one
transposition away from the cross-layer `[N, H, B, D]` view. We extended that
one.

## B.2 Why the existing kernel didn't work as-is

The original `reshape_and_cache_flash_kernel` reads `block_stride =
key_cache.stride(0)` from the tensor (so the outermost dim is already
non-contiguous-safe) but hard-codes the inner-dim strides as
`num_heads * head_size` and `head_size`:

```cpp
// Original (packed [N, B, H, D] assumed)
const int64_t tgt_key_value_idx = block_idx * block_stride +
                                  block_offset * num_heads * head_size +
                                  head_idx * head_size + head_offset;
```

Those hard-codes break for the cross-layer view because the B and H dims
swap and the per-head stride embeds the cross-layer factor:

| Stride                | Packed `[N, B, H, D]`  | Cross-layer `[N, H, B, D]` |
|-----------------------|------------------------|----------------------------|
| `cache_page_stride` (per-token-in-page) | `num_heads * head_size` | `head_size`                          |
| `cache_head_stride` (per-kv-head)        | `head_size`             | `num_layers * 2 * page_size * head_size` |

## B.3 What we changed

The kernel is now generic. It takes the two strides as runtime parameters and
the address formula is purely stride-driven:

```cpp
// New: stride-driven, layout-agnostic
const int64_t tgt_key_value_idx = block_idx * block_stride +
                                  block_offset * cache_page_stride +
                                  head_idx * cache_head_stride +
                                  head_offset;
```

The C++ wrapper picks the right strides from the tensor based on a new
`kv_layout` parameter (mirror of the prefill enum). For the legacy default
(`kv_layout = -1` or `1` = `LINEAR_LAYOUT`), the wrapper passes the same
`num_heads * head_size` / `head_size` values the kernel previously hard-coded,
so the existing fast path is bit-exact unchanged. A new branch
(`kv_layout = 2` = `LINEAR_HEADS_FIRST_LAYOUT`) reads the strides positionally
from `key_cache.stride(1)` / `key_cache.stride(2)` for the `[N, H, B, D]`
view, plus the same 16B-alignment guards we added on the prefill side.

Files touched in Part B:

| File | What changed |
|---|---|
| [aiter/csrc/kernels/cache_kernels.cu](../csrc/kernels/cache_kernels.cu) | Kernel takes `cache_page_stride` + `cache_head_stride`; wrapper handles `kv_layout` (new branch for heads-first with alignment guards) |
| [aiter/csrc/include/cache.h](../csrc/include/cache.h) | Added `int kv_layout = -1` to the public declaration |
| [aiter/csrc/include/rocm_ops.hpp](../csrc/include/rocm_ops.hpp) | Pybind binding now uses named `py::arg` form and exposes `kv_layout = -1` |
| [aiter/aiter/ops/cache.py](../aiter/ops/cache.py) | Added `KV_LAYOUT_*` constants and a high-level `reshape_and_cache_flash_func` that mirrors `mha_batch_prefill_func` (accepts a 5D `kv_cache=` view, auto-slices and forces the heads-first layout) |
| [aiter/op_tests/test_reshape_and_cache_cross_layer_5d.py](../op_tests/test_reshape_and_cache_cross_layer_5d.py) | New 26-test file (stride metadata, layout-mismatch rejection, functional equivalence vs packed reference, legacy-path regression) |

The `KV_LAYOUT_*` integer constants are intentionally duplicated in both
`aiter/ops/mha.py` and `aiter/ops/cache.py` (they hold identical values).
Both modules document the canonical source in the CK header so the constants
stay synchronized if anyone touches the enum.

## B.4 Usage

The new high-level helper mirrors `mha_batch_prefill_func` for symmetry:

```python
# Framework allocates the 6D physical KV buffer and produces a per-layer 5D
# non-contiguous view via permute() + index.
kv_cache_view = logical6d[layer_idx]   # shape (2, N, H, B, D), non-contiguous

# Write newly produced K/V tokens into the per-layer cache:
aiter.reshape_and_cache_flash_func(
    key,                                # [num_tokens, num_kv_heads, head_dim]
    value,                              # [num_tokens, num_kv_heads, head_dim]
    slot_mapping=slot_mapping,          # [num_tokens]
    kv_cache=kv_cache_view,             # auto-slices into key_cache/value_cache
    kv_cache_dtype="auto",
)
```

Existing callers of `aiter.reshape_and_cache_flash(key, value, k_cache,
v_cache, slot_mapping, "auto", k_scale, v_scale)` keep working unchanged —
the `kv_layout` argument defaults to `-1` (auto) which routes to the
identical packed-layout fast path the kernel always took. Verified by
`test_reshape_and_cache_flash_legacy_path_unchanged`.

## B.5 Verification

Ran on the same MI-class hardware as Part A.

```
test_reshape_and_cache_flash_cross_layer_5d_strides         8 passed
test_reshape_and_cache_flash_func_rejects_layout_mismatch   1 passed
test_reshape_and_cache_flash_cross_layer_5d_matches_packed 16 passed
test_reshape_and_cache_flash_legacy_path_unchanged          1 passed
                                                     ───────────────
                                                     26 / 26 passed
```

Cost: the `module_cache` JIT compile takes about 30s on a fresh build; reuse
is near-instant from the cached `.so`.

The functional test (`matches_packed`) runs `reshape_and_cache_flash` twice
on the same K/V token data:

1. into a packed `[N, B, H, D]` reference cache (legacy path, `kv_layout = 1`)
2. into the cross-layer non-contiguous `[N, H, B, D]` view sliced from a 6D
   physical buffer (heads-first path, `kv_layout = 2`)

and asserts that, after permuting the cross-layer cache to `[N, B, H, D]` for
comparison, the two are bit-exact equal. It also checks the 6D physical
buffer's *other* layer slots stay all-zero, catching any cross-talk between
layers.

Parametrizations: `dtype ∈ {bf16, fp16}` × `num_tokens ∈ {16, 33}` ×
`num_kv_heads ∈ {1, 2}` × `layer_idx ∈ {0, 2}` × constant `num_blocks=4`,
`page_size=16`, `head_dim=128`, `num_layers=4`.

The same migration gotcha as Part A applies: pre-built `aiter/jit/module_cache*.so`
files from before this patch must be deleted to force a JIT rebuild against
the new pybind signature. Same `rm` recipe, substitute `module_cache` for
`mha_batch_prefill_*`.

## B.6 What is still NOT covered

- The vLLM x-vectorized `reshape_and_cache` kernel (not _flash_) is unchanged.
  Tencent's design doesn't use that layout family, so there's no benefit to
  generalizing it — but if a future Kunlun-XPU pathway insists on the
  x-vectorized cache combined with cross-layer addressing, that kernel would
  need an analogous rewrite.
- Quantized writer variants (`reshape_and_cache_with_pertoken_quant`,
  `reshape_and_cache_with_block_quant`, etc. in the same file) likewise still
  assume packed layout. They're not on the HY3 prefill path, so they're
  intentionally left alone. The same recipe — extend the kernel to take
  per-page and per-head strides, branch the wrapper on `kv_layout` — applies
  cleanly if/when the need arises.

