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
| Attention type | prefill / self-attention (`kv_len == q_len` per batch) |

All tensors must be CUDA tensors, contiguous, on the same device.

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

`phys` is the physical page id obtained from the page table (see below). Both
must be contiguous so the kernel can read strides directly.

### Page table (SGLang 1D)

| Arg | Shape | Dtype | Meaning |
|---|---|---|---|
| `cu_seqlens_q` | `[b+1]` | int32 | Prefix sum of per-batch `seqlen_q`; `cu_seqlens_q[b] == total_q`. Drives the packed-Q base. |
| `kv_indptr` | `[b+1]` | int32 | Prefix sum of per-batch **page counts** (LTP). Batch `i` owns pages `kv_page_indices[kv_indptr[i] : kv_indptr[i+1]]`. |
| `kv_page_indices` | `[num_pages]` | int32 | Flat list of **physical** page ids (LTD). |
| `seqlens_kvcache` | `[b]` | int32 | Per-batch KV token length (for prefill = per-batch `seqlen_q`). |

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

See `op_tests/test_batch_prefill.py` (`run_batch_prefill_asm`,
`test_batch_prefill_asm_*`) for a reference driver, an fp32 correctness check,
and a latency/TFLOPS sweep (`AITER_ASM_PERF=1`).