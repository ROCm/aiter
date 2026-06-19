# `mha_batch_prefill_asm` — FP8 paged causal prefill (gfx942)

Assembly (`.co`) attention **prefill** kernel for paged KV caches with
per-token-per-head FP8 quantization. It is exposed as a standalone op
(`module_mha_batch_prefill_asm`, built without Composable Kernel) and is **not**
auto-selected by the generic `mha_batch_prefill`; callers invoke it explicitly.

```python
import aiter
out = aiter.mha_batch_prefill_asm(...)            # top-level alias
# or
from aiter.ops.mha_batch_prefill_asm import mha_batch_prefill_asm
```

The kernel is a single **varlen** kernel: it derives every per-batch Q base from
`cu_seqlens_q`, so it is correct for any batch size, including `batch == 1`
(a single `[0, S]` segment).

---

## Hard constraints

The call **raises** (no silent fallback) unless all of these hold:

| Requirement | Value |
|---|---|
| GPU arch | `gfx942` (MI300/MI308; the shipped `.co` lives in `hsa/gfx942/fmha_v3_fwd/MI308/`) |
| Head dim | `head_size_q == head_size_v == 128` |
| Page size | `page_block_size == 16` |
| Q/K/V dtype | FP8 E4M3 (`torch.float8_e4m3fnuz` on gfx942) |
| Output dtype | `torch.bfloat16` |
| Masking | causal (bottom-right aligned), always on |
| Attention type | causal prefill. Per-batch KV length (`seqlens_kvcache`) is independent of the Q length: `kv_len >= q_len` is supported (prefix / chunked prefill against a cached KV prefix), not just `kv_len == q_len` self-attention. |

All tensors must be CUDA tensors on the same device. `q`/`out` and the page/scale
tables are contiguous; `k`/`v` may be **non-contiguous strided views** — the kernel
addresses them purely through `stride(0)`/`stride(1)` and the base pointer (see
[Multi-layer KV cache](#multi-layer-kv-cache-all-layers-per-block)).

---

## Signature

```python
mha_batch_prefill_asm(
    q, k, v,
    cu_seqlens_q, kv_indptr, kv_page_indices, seqlens_kvcache,
    out,
    q_descale_per_token, k_descale_per_token, v_descale_per_head,
    batch, num_heads, num_heads_k,
    head_size_q, head_size_v, page_block_size,
    num_total_pages, max_seqlen_q, softmax_scale,
    p_scale=None,
) -> Tensor                      # returns `out` (also written in place)
```

### Symbols used below

| Symbol | Meaning |
|---|---|
| `b` | batch size (`batch`) |
| `hq` / `hk` | query heads (`num_heads`) / KV heads (`num_heads_k`); `hq % hk == 0` (GQA) |
| `d` | head dim = 128 |
| `page` | page block size = 16 |
| `x` | FP8 vector width = 16 (== `page`) |
| `total_q` | `sum(seqlen_q[i] for i in range(b))` (packed Q rows) |
| `num_pages` | total physical pages in the pool (`num_total_pages`) |

---

## Tensor arguments

### Query / output (packed THD)

| Arg | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `[total_q, hq, d]` | fp8 | Q for all batches concatenated along the token axis. Row-major: `stride = (hq*d, d, 1)`. |
| `out` | `[total_q, hq, d]` | bf16 | Pre-allocated; written in place and returned. Same packed layout as `q`. |

### Paged K / V

K and V live in physical page pools indexed through the page table. Per `(page,
head)` each holds `d*page` elements.

| Arg | Shape | Dtype | Layout |
|---|---|---|---|
| `k` | `[num_pages, hk, d/x, page, x]` | fp8 | **vec_k_col_v**: token `t`'s element `e` of head `hk` sits at `[phys, hk, e//x, t%page, e%x]`. |
| `v` | `[num_pages, hk, d, page]` | fp8 | **column / token-minor**: `[phys, hk, e, t%page]`. |

`phys` is the physical page id obtained from the page table (see below). The
kernel reads the physical-block stride from `k.stride(0)`/`v.stride(0)`, the
KV-head stride from `stride(1)`, and the K/V bases from the tensor data pointers,
so the pools need **not** be contiguous: a strided view into a larger buffer is
fine (this is what the [multi-layer KV cache](#multi-layer-kv-cache-all-layers-per-block)
relies on). The only requirement is that the inner `(d/x, page, x)` / `(d, page)`
swizzle is intact per block.

### Page table (SGLang 1D)

| Arg | Shape | Dtype | Meaning |
|---|---|---|---|
| `cu_seqlens_q` | `[b+1]` | int32 | Prefix sum of per-batch `seqlen_q`; `cu_seqlens_q[b] == total_q`. Drives the packed-Q base. |
| `kv_indptr` | `[b+1]` | int32 | Prefix sum of per-batch **page counts** (LTP). Batch `i` owns pages `kv_page_indices[kv_indptr[i] : kv_indptr[i+1]]`. |
| `kv_page_indices` | `[num_pages]` | int32 | Flat list of **physical** page ids (LTD). |
| `seqlens_kvcache` | `[b]` | int32 | Per-batch KV token length. Independent of `seqlen_q`; set equal to it for plain self-attention, or larger for prefix / chunked prefill (`kv_len >= q_len`). |

### Scales (FP8 descales + P scale)

A **descale** is the multiplier that converts the stored FP8 value back to the
real value: `real ≈ fp8.float() * descale`. They are applied per token & head.

| Arg | Shape | Dtype | Meaning |
|---|---|---|---|
| `q_descale_per_token` | `[total_q, hq]` | f32 | One descale per (packed Q token, query head). |
| `k_descale_per_token` | `[num_pages, page, hk]` | f32 | Paged like K: `[phys, t%page, hk]`. |
| `v_descale_per_head` | `[hk]` | f32 | One descale per KV head (shared across tokens & batch). |
| `p_scale` (optional) | `[hq]` | f32 | Per query-head scale applied to softmax probabilities before the FP8 P·V matmul. `None` ⇒ 1.0. |

### Scalars

| Arg | Type | Notes |
|---|---|---|
| `batch`, `num_heads`, `num_heads_k` | int | `num_heads % num_heads_k == 0`. |
| `head_size_q`, `head_size_v` | int | Both 128. |
| `page_block_size` | int | 16. |
| `num_total_pages` | int | Number of physical pages = `k.shape[0]`. |
| `max_seqlen_q` | int | `max(seqlen_q[i])`. |
| `softmax_scale` | float | Usually `1/sqrt(d)`. |

---

## Numerics

For query head `h` (KV head `hk = h // (hq//hk)`):

```
Qd = q.float()  * q_descale_per_token         # per (token, head)
Kd = k.float()  * k_descale_per_token         # per (token, kv-head)
Vd = v.float()  * v_descale_per_head          # per kv-head
S  = (Qd @ Kdᵀ) * softmax_scale               # causal-masked (k <= q)
P  = softmax(S) * p_scale[h]                   # p_scale = 1 if None
out = (fp8(P) @ Vd) / sum(P)                   # P quantized to FP8 for the P·V matmul
```

Output is bf16. Expected accuracy vs an fp32 reference is NRMS ≈ 1–3e-2 (FP8
round-off), rising slowly with sequence length.

---

## Example

```python
import torch, math, itertools
from aiter import dtypes
from aiter.ops.mha_batch_prefill_asm import mha_batch_prefill_asm

dev, FP8 = "cuda", dtypes.fp8                 # torch.float8_e4m3fnuz on gfx942
FP8_MAX = float(torch.finfo(FP8).max)
PAGE = X = 16
hq, hk, d = 8, 1, 128
seqlens = [256, 512]                          # per-batch q == kv (prefill)
b = len(seqlens)
scale = 1.0 / math.sqrt(d)

def quant_rows(x):                            # per-row symmetric fp8 quant
    desc = (x.abs().amax(-1, keepdim=True).clamp(min=1e-6) / FP8_MAX).float()
    return (x / desc).to(FP8), desc.squeeze(-1)

q_list, qdesc_list, k_list, v_list, kdesc_list = [], [], [], [], []
v_desc = (torch.rand(hk, device=dev) * 0.5 + 0.5).float()
for s in seqlens:
    q = torch.randn(s, hq, d, device=dev); k = torch.randn(s, hk, d, device=dev)
    v = torch.randn(s, hk, d, device=dev)
    qf8, qd = quant_rows(q); kf8, kd = quant_rows(k)
    vf8 = (v / v_desc[None, :, None]).to(FP8)
    q_list.append(qf8); qdesc_list.append(qd)
    k_list.append(kf8); v_list.append(vf8); kdesc_list.append(kd)

q_packed = torch.cat(q_list).contiguous()                 # [total_q, hq, d]
qdesc    = torch.cat(qdesc_list).contiguous()             # [total_q, hq]
cu_seqlens_q = torch.tensor([0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=dev)

# Build paged pools (identity page table for simplicity)
ppb = [(s + PAGE - 1) // PAGE for s in seqlens]
num_pages = sum(ppb)
kv_indptr = torch.tensor([0, *itertools.accumulate(ppb)], dtype=torch.int32, device=dev)
kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=dev)
seqlens_kvcache = torch.tensor(seqlens, dtype=torch.int32, device=dev)

k_pool = torch.zeros(num_pages, hk, d // X, PAGE, X, dtype=FP8, device=dev)
v_pool = torch.zeros(num_pages, hk, d, PAGE, dtype=FP8, device=dev)
kdesc_pool = torch.zeros(num_pages, PAGE, hk, dtype=torch.float32, device=dev)
for bi, (kf8, vf8, kd) in enumerate(zip(k_list, v_list, kdesc_list)):
    base = kv_indptr[bi].item()
    for t in range(kf8.shape[0]):
        phys, row = base + t // PAGE, t % PAGE
        k_pool[phys, :, :, row, :] = kf8[t].view(hk, d // X, X)
        v_pool[phys, :, :, row] = vf8[t]
        kdesc_pool[phys, row, :] = kd[t]

p_scale = torch.ones(hq, device=dev)                      # or None
out = torch.empty_like(q_packed, dtype=torch.bfloat16)

mha_batch_prefill_asm(
    q_packed, k_pool, v_pool,
    cu_seqlens_q, kv_indptr, kv_page_indices, seqlens_kvcache,
    out,
    qdesc, kdesc_pool, v_desc,
    b, hq, hk, d, d, PAGE, num_pages, max(seqlens), scale,
    p_scale=p_scale,
)
# out: [total_q, hq, d] bf16
```

---

## Multi-layer KV cache (all-layers-per-block)

The kernel addresses `k`/`v` only through their `stride(0)` (physical-block
stride), `stride(1)` (KV-head stride) and base pointer — it makes no assumption
that the page pool is contiguous. This lets it consume vLLM's **all-layers-per-block**
combined KV cache ([vllm-project/vllm#27742](https://github.com/vllm-project/vllm/issues/27742))
with **no kernel or launcher change**: allocate one buffer that interleaves every
layer (and both K and V) per block, and hand each attention layer a strided view.

Backing buffer, C-contiguous, exactly as in the issue:

```text
(num_blocks, num_kv_heads, num_layers, 2, page * d)
```

The innermost `page * d` chunk keeps the kernel's existing swizzle (`[d/x, page, x]`
for K, `[d, page]` col-major for V) — only the **outer** block/head/layer/(K|V)
strides are new. A single layer's views are then:

| view | shape | stride (elements) | base offset (elements) |
|---|---|---|---|
| `k` | `[num_pages, hk, d/x, PAGE, x]` | `(hk*L2, L2, PAGE*x, x, 1)` | `layer * 2*PAGE*d` |
| `v` | `[num_pages, hk, d, PAGE]`       | `(hk*L2, L2, PAGE, 1)`      | `layer * 2*PAGE*d + PAGE*d` |

where `L2 = num_layers * 2 * PAGE * d`. `stride(0) = hk*L2` is the per-block
stride; `stride(1) = L2` the KV-head stride; V starts exactly one `PAGE*d` tile
after K within the block (the issue's `dim0` / K-vs-V stride).

```python
inner = PAGE * d                         # fp8 elements per (block,head,layer,kv) tile
L2    = num_layers * 2 * inner
blk   = hk * L2                          # per-physical-block stride

# One buffer for ALL layers + K and V (vLLM allocates this once per model).
kv_buf = torch.empty(num_pages * blk, dtype=FP8, device=dev)

def layer_k(layer):                       # K view for `layer`
    return torch.as_strided(kv_buf, (num_pages, hk, d // X, PAGE, X),
                            (blk, L2, PAGE * X, X, 1), layer * 2 * inner)

def layer_v(layer):                       # V view for `layer`
    return torch.as_strided(kv_buf, (num_pages, hk, d, PAGE),
                            (blk, L2, PAGE, 1), layer * 2 * inner + inner)

# Same call as the contiguous case — just pass the strided per-layer views.
mha_batch_prefill_asm(
    q_packed, layer_k(layer), layer_v(layer),
    cu_seqlens_q, kv_indptr, kv_page_indices, seqlens_kvcache,
    out, qdesc, kdesc_pool, v_desc,
    b, hq, hk, d, d, PAGE, num_pages, max(seqlens), scale, p_scale=p_scale,
)
```

The launcher derives the block/head strides from `view.stride(0)`/`stride(1)` and
the K/V bases from the view data pointers, so correctness **and** performance are
identical to the contiguous pools. This is verified bit-for-bit (NRMS unchanged
across layer counts and layer indices, with the non-target layers filled with
noise) in `op_tests/test_batch_prefill.py::test_batch_prefill_asm_combined_kv`
(driven via `run_batch_prefill_asm(..., combined_kv=True, num_kv_layers=N,
kv_layer_idx=L)`).

> **Descales** (`k_descale_per_token`, etc.) are separate inputs and are *not*
> part of the combined buffer; pass them as usual. If a future vLLM layout also
> combines descales per layer, the same strided-view trick applies to them.

---

## Notes & gotchas

- **Packed Q**: there is no batch dimension on `q`/`out`; batches are
  concatenated along `total_q` and delimited by `cu_seqlens_q`.
- **`num_total_pages` must equal `k.shape[0]`** (the SRD bound the kernel uses).
- **Identity vs fragmented page tables**: `kv_page_indices` may map logical pages
  to arbitrary physical pages; the example uses identity for clarity.
- **MI308 only**: the `.co` is deployed under `MI308/`. Other gfx942 parts
  (e.g. MI300X) need the binary placed in the corresponding arch subfolder.
- **No KV append**: this is a prefill kernel; it reads the full per-batch KV
  range, it does not write new tokens into the cache.
- **`kv_len` need not be a multiple of `page` (16).** Only `page_block_size`
  itself must be 16. A batch allocates `ceil(kv_len / 16)` physical pages; the
  last page may be partially filled, and `seqlens_kvcache[i]` gives the exact
  per-batch token count so the kernel masks the unused tail of the final page.
  (The unaligned cases `257`, `300`, `1000`, … are covered by
  `test_batch_prefill_asm_qseqlen_unaligned_256`.) What *is* fixed is `page == 16`
  and the per-block inner swizzle.

---

## Running the tests

All ASM tests are gfx942-only and require the kernel `.co` to be built/deployed
under `hsa/gfx942/fmha_v3_fwd/MI308/` (see `scripts/f8_fmha_prefill/build_qkptph.sh`).

```bash
cd /workspace/aiter

# Multi-layer (all-layers-per-block) combined KV cache — vllm#27742:
pytest op_tests/test_batch_prefill.py -k combined_kv -v

# All ASM qkptph/vph cases (self-attn, varlen, unaligned, combined-KV):
pytest op_tests/test_batch_prefill.py -k batch_prefill_asm -v

# Latency/TFLOPS perf sweep (opt-in):
AITER_ASM_PERF=1 pytest op_tests/test_batch_prefill.py -k batch_prefill_asm_perf -v
```

Or drive it directly from Python (no pytest), e.g. a single combined-KV run:

```python
from aiter.ops.mha_batch_prefill_asm import mha_batch_prefill_asm  # ensure .co is deployed
import op_tests.test_batch_prefill as t

# layer 2 of an 8-layer all-layers-per-block cache, varlen batch:
t.run_batch_prefill_asm([256, 384], use_p_scale=True, seed=37,
                        combined_kv=True, num_kv_layers=8, kv_layer_idx=2)
```

See `op_tests/test_batch_prefill.py` (`run_batch_prefill_asm`,
`test_batch_prefill_asm_*`) for a reference driver, an fp32 correctness check,
and a latency/TFLOPS sweep (`AITER_ASM_PERF=1`).