# gfx1250 MLA v3 ASM decode

Frameworks call `aiter.mla.mla_decode_fwd` and select the KV byte layout with
`kv_layout`. Each layout keeps its own native launcher in `module_mla_asm`:
`token_major` uses the existing `mla_decode_stage1_asm_fwd` dispatch and
`kv_shuffled` uses `mla_ps64_qh128_fp8_asm_fwd`.

| `kv_layout` | Query heads / queries per sequence | KV storage | Metadata | Scheduling |
| --- | --- | --- | --- | --- |
| `"token_major"` (default) | 8 / 1,2,3,4; 16 / 1,2,4; 32,64,128 / 1 | Existing segmented `[P,64,1,576]` FP8 pages | Page-level CSR | Existing split-K and reducer |
| `"kv_shuffled"` | 128 / 1 | Shuffled `[P,1,64,576]` FP8 pages | `seq_lens[B]`, `page_table[B,max_pages]` | One workgroup per sequence, no split-K |

`token_major` is the compatibility name for the existing dispatch. On gfx1250,
that dispatch already consumes segmented page64 storage: each page contains a
512-dimension NoPE plane followed by a 64-dimension RoPE plane. Shuffled storage
tiles each of those planes into 16-token by 16-dimension blocks. Selecting a
layout does not repack the cache. The cache writer must supply the correct bytes.
Other architectures keep the existing default dispatch.

## ATOM shuffled decode

```python
# During initialization: CPU 0-D FP32 scales are accepted.
q_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
kv_scale = torch.tensor(1.25, dtype=torch.float32, device="cpu")

# q: [B,128,576] FP8 E4M3FN, head stride 576 or 768 bytes.
# kv_shuffled: contiguous [P,1,64,576] FP8 E4M3FN.
# out: contiguous [B,128,512] BF16 on q.device.
# context_lens: contiguous GPU int32 [B].
# block_tables: GPU int32 [B,max_pages], unit inner stride.
out, lse = aiter.mla.mla_decode_fwd(
    q, kv_shuffled, out,
    None, None, None, None,  # CSR metadata is unused for this layout.
    max_seqlen_q=1,
    page_size=64,
    kv_layout="kv_shuffled",
    seq_lens=context_lens,
    page_table=block_tables,
    q_scale=q_scale,
    kv_scale=kv_scale,
    return_lse=True,
)
```

The shuffled branch runs before any CSR metadata access. It returns the supplied
`out` tensor and, when requested, an internally allocated FP32 `[B,128]`
natural-log LSE. With `return_lse=False`, it passes a null LSE pointer and returns
`(out, None)`. Empty sequences produce zero output and negative-infinity LSE.
`causal` has no effect when each sequence has one query.

CPU FP32 scales are copied once per `(GPU device, scalar value)` and retained
for reuse. GPU FP32 scalar tensors on Q's device are used directly. Missing
scales mean 1.0. Warm up on the target device with the intended CPU values before
CUDA graph capture; an uncached CPU value during capture raises a clear error.
Replaying a captured graph uses the scale tensors captured during warmup/capture;
changing CPU values requires recapture. Device scales can be updated in place
with the usual stream dependencies. No GPU metadata is copied to the host.

Scales describe the already quantized inputs: the kernel interprets Q as
`q.float() * q_scale` and KV as `kv.float() * kv_scale`. Decode does not quantize
Q. ATOM's shuffled fused writer currently casts Q directly to FP8, so its Q
scale is 1.0. A non-unit Q scale requires matching quantization in the writer.
CPU scale support here applies to decode; Triton KV writers and gathers still
need their scale tensors on the GPU.

Shuffled decode requires gfx1250 B0+, FP8 Q/KV, BF16 O, page64, QH128, and one
query per sequence. Metadata row counts must equal `q.shape[0]` exactly. ATOM
must reconcile `running_bs` and the Q batch before calling; this API does not
silently slice or pad metadata. MTP/speculative multi-query decode, context
parallelism, persistent metadata, KV splitting, and partial-logit requests are
rejected. This branch does not fall back to Triton/Gluon. Used Q/KV values and
GPU scales must be finite.

## Native launchers and compatibility

The two layouts take disjoint metadata (page-level CSR plus split workspaces
versus dense `seq_lens`/`page_table`), so each keeps a launcher whose signature
holds only its own arguments. `mla_decode_fwd` validates and routes in Python;
the `token_major` path is unchanged. `mla_ps64_qh128_fp8_asm_fwd` launches the
shuffled 88-byte ABI and does not allocate workspaces or convert metadata.

`mla_decode_fwd_qh128_asm` remains available for callers with preallocated
O/LSE and explicit GPU scales. Kernel code objects and their ABI layouts are
unchanged.

The page64 benchmark also calls `mla_decode_fwd` for both layouts. Its CLI
defaults to shuffled for QH128 and segmented for other supported head counts;
`--kv-layout token_major` explicitly exercises segmented QH128.

```bash
python -m pytest op_tests/test_mla_decode_pagesize64.py -q
python op_tests/test_mla_decode_pagesize64.py -n 8,1 128,1 --split-kv 1 -b 2 -c 65 1024
python op_tests/test_mla_decode_pagesize64.py -n 128,1 --kv-layout token_major --split-kv 1 3 -b 2 -c 321
```
