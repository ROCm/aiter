# AITER BERT Padding & Variable-Length Sequence Guide

This guide documents the padding/unpadding utilities and variable-length sequence handling in AITER, enabling efficient attention computation on batches with different sequence lengths.

---

## Quick Reference

| Use Case | Function | Description |
|----------|---------|-------------|
| **Remove padding** | `unpad_input(hidden_states, attention_mask)` | Batch → packed sequences |
| **Restore padding** | `pad_input(hidden_states, indices, batch, seqlen)` | Packed → batch format |
| **Concatenated sequences** | `unpad_input_for_concatenated_sequences(...)` | SFT-style concatenated samples |
| **Variable-length attention** | `flash_attn_varlen_func(...)` | Flash attention on packed inputs |

---

## 1. Padding / Unpadding Utilities

### `unpad_input`

Removes padding tokens from batch-formatted tensors to create packed sequences:

```python
from aiter.bert_padding import unpad_input

hidden_states_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(
    hidden_states,      # (batch, seqlen, ...) — padded input
    attention_mask,     # (batch, seqlen) — 1=valid, 0=padding
    unused_mask=None,   # (batch, seqlen) — 1=allocated but unused (optional)
)
# hidden_states_unpad: (total_nnz, ...) — packed valid tokens
# indices: (total_nnz,) — indices into flattened input
# cu_seqlens: (batch+1,) — cumulative sequence lengths
# max_seqlen: int — longest sequence
# seqused: (batch,) — tokens selected per batch element
```

### `pad_input`

Restores packed sequences to padded batch format (inverse of `unpad_input`):

```python
from aiter.bert_padding import pad_input

hidden_states = pad_input(
    hidden_states_unpad, # (total_nnz, ...) — packed tokens
    indices,             # (total_nnz,) — from unpad_input
    batch,               # int — batch size
    seqlen,              # int — max sequence length
)
# hidden_states: (batch, seqlen, ...) — zero-padded output
```

### `unpad_input_for_concatenated_sequences`

For supervised fine-tuning where multiple short samples are concatenated:

```python
from aiter.bert_padding import unpad_input_for_concatenated_sequences

hidden_states_unpad, indices, cu_seqlens, max_seqlen = unpad_input_for_concatenated_sequences(
    hidden_states,              # (batch, seqlen, ...)
    attention_mask_in_length,   # (batch, seqlen) — nonzero=length of concat'd sequence
)
```

---

## 2. Variable-Length Flash Attention

After unpadding, use variable-length flash attention:

```python
import aiter

out_unpad = aiter.flash_attn_varlen_func(
    q_unpad,            # (total_q, nheads, headdim)
    k_unpad,            # (total_k, nheads_k, headdim)
    v_unpad,            # (total_k, nheads_k, headdim_v)
    cu_seqlens_q,       # (batch+1,) int32
    cu_seqlens_k,       # (batch+1,) int32
    max_seqlen_q,       # int
    max_seqlen_k,       # int
    causal=True,
    softmax_scale=None,
    # Optional physical padding support:
    cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None,
)
```

**Backends:** CK (primary), FMHA v3 (bf16, hdim 128/192), Triton

---

## 3. End-to-End Example

```python
import torch
import aiter
from aiter.bert_padding import unpad_input, pad_input
from aiter.test_mha_common import generate_qkv

batch_size, seqlen_q, seqlen_k = 4, 512, 512
nheads, d = 32, 128

# Padded inputs
q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(batch_size, seqlen_k, nheads, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(batch_size, seqlen_k, nheads, d, device="cuda", dtype=torch.bfloat16)

# Random padding masks (actual sequence lengths vary)
query_padding_mask = torch.ones(batch_size, seqlen_q, device="cuda", dtype=torch.bool)
key_padding_mask = torch.ones(batch_size, seqlen_k, device="cuda", dtype=torch.bool)
query_padding_mask[0, 300:] = False  # First sequence is 300 tokens
key_padding_mask[0, 300:] = False

# Unpad
q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)

# Run attention on packed sequences (no wasted computation on padding)
out_unpad = aiter.flash_attn_varlen_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True,
)

# Restore to padded format
out = pad_input(out_unpad, indices_q, batch_size, seqlen_q)
```

---

## 4. Implementation Details

### Custom Autograd Functions

The module uses optimized `torch.gather`/`torch.scatter` instead of boolean indexing:

| Function | Forward | Backward |
|----------|---------|----------|
| `IndexFirstAxis` | `torch.gather` | `torch.scatter` |
| `IndexPutFirstAxis` | `torch.scatter` | `torch.gather` |
| `IndexFirstAxisResidual` | Index + residual pass-through | scatter-add |

### Tensor Formats

| Format | Shape | Description |
|--------|-------|-------------|
| **Padded** | `(batch, seqlen, nheads, hdim)` | Standard batch format |
| **Unpadded** | `(total_tokens, nheads, hdim)` | Packed valid tokens |
| **cu_seqlens** | `(batch+1,)` int32 | Cumulative lengths, starts at 0 |

---

## 5. Supported Data Types

| dtype | Notes |
|-------|-------|
| float16 | Fully supported |
| bfloat16 | Preferred for FMHA v3 |
| float8_e4m3fn/fnuz | With descaling parameters |

---

## 6. Source Files

| Component | Path |
|---|---|
| Padding utilities | `aiter/bert_padding.py` |
| Variable-length attention API | `aiter/ops/mha.py` |
| Test utilities (generate_qkv) | `aiter/test_mha_common.py` |
| CK attention kernels | `csrc/include/torch/mha_varlen_fwd.h` |

---

## 7. Test Files

| Test | Path |
|------|------|
| Variable-length MHA | `op_tests/test_mha_varlen.py` |
| Variable-length MHA (FP8) | `op_tests/test_mha_varlen_fp8.py` |
