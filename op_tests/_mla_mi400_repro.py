"""Standalone repro for the gfx1250 MLA mi400 smoke kernel + GPU sync."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import aiter
print("aiter from:", aiter.__file__, flush=True)
from aiter import dtypes

print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)
print("init cuda...", flush=True)
torch.cuda.init()
print("cuda init OK", flush=True)
device = torch.device("cuda")
print("alloc tiny tensor...", flush=True)
_t = torch.zeros(1, device=device)
print("tiny tensor OK:", _t.device, flush=True)
batch = 1
q_seq_len = 4
kv_seq_len = 578
page_size = 64
num_kv_splits = 1
nhead = 16
nhead_kv = 1
qk_head_dim = 576
v_head_dim = 512
num_pages = (kv_seq_len + page_size - 1) // page_size

q = torch.randn((batch * q_seq_len, nhead, qk_head_dim),
                dtype=torch.bfloat16, device=device).to(dtypes.fp8)
kv_buffer = torch.randn((num_pages, page_size, nhead_kv, qk_head_dim),
                        dtype=torch.bfloat16, device=device).to(dtypes.fp8)
out = torch.empty((batch * q_seq_len, nhead, v_head_dim),
                  dtype=torch.bfloat16, device=device)

qo_indptr = torch.tensor([0, q_seq_len], dtype=torch.int32, device=device)
kv_indptr = torch.tensor([0, kv_seq_len], dtype=torch.int32, device=device)
kv_indices = torch.zeros(num_pages + 4, dtype=torch.int32, device=device)
kv_indices[:num_pages] = torch.arange(num_pages, dtype=torch.int32, device=device)
kv_last_page_lens = torch.tensor([kv_seq_len % page_size],
                                 dtype=torch.int32, device=device)
num_kv_splits_indptr = torch.tensor([0, num_kv_splits],
                                    dtype=torch.int32, device=device)
q_scale = torch.ones((batch,), dtype=torch.float32, device=device)
kv_scale = torch.ones((batch,), dtype=torch.float32, device=device)

print("calling mla_decode_fwd...", flush=True)
attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
    q, kv_buffer, out,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    q_seq_len, page_size, nhead_kv, 1.0 / (qk_head_dim ** 0.5),
    num_kv_splits=num_kv_splits,
    num_kv_splits_indptr=num_kv_splits_indptr,
    q_scale=q_scale, kv_scale=kv_scale,
)
print("launch returned, syncing...", flush=True)
torch.cuda.synchronize()
print("sync OK.", flush=True)
print("out shape:", tuple(out.shape), "dtype:", out.dtype, flush=True)
print("attn_logits is out?:", attn_logits.data_ptr() == out.data_ptr(), flush=True)
print("attn_lse shape:", tuple(attn_lse.shape), "dtype:", attn_lse.dtype, flush=True)

print("copying out to cpu...", flush=True)
out_cpu = out.detach().to("cpu")
print("out_cpu OK", flush=True)
print("out[0,0,:8] =", out_cpu[0, 0, :8].tolist(), flush=True)
print("out[0,0,-8:] =", out_cpu[0, 0, -8:].tolist(), flush=True)

print("copying attn_lse to cpu...", flush=True)
lse_cpu = attn_lse.detach().to("cpu")
print("lse[0,0,0,0] =", lse_cpu[0, 0, 0, 0].item(), flush=True)

print("checking nan/inf via cpu fp32...", flush=True)
out_fp32 = out_cpu.float()
print("nan count:", torch.isnan(out_fp32).sum().item(),
      "inf count:", torch.isinf(out_fp32).sum().item(),
      "min:", out_fp32.min().item(),
      "max:", out_fp32.max().item(),
      "mean:", out_fp32.mean().item(), flush=True)
print("DONE", flush=True)
