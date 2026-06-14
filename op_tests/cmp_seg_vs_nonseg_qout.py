# SPDX-License-Identifier: MIT
"""Decisive offline check: feed the SAME inputs to both
fused_qk_rope_concat_and_cache_mla (non-seg, triton path) and
fused_qk_rope_concat_and_cache_mla_seg (seg path) and diff the q_out pe
segment [512:576]. If they match here, the kernels agree and any serve-time
divergence must come from a different cos/sin cache layout/stride or positions
passed at runtime.
"""
import torch
import aiter
from aiter import dtypes


def cosq(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def compute_cache(seq_len, freqs_dim, dtype, base=10000.0):
    div_term = 1.0 / (base ** (torch.arange(0, freqs_dim, 1).float() / freqs_dim))
    positions = torch.arange(seq_len).float().unsqueeze(1)
    freqs = positions * div_term.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def run(is_neox, cos_width=32, cos_extra_stride=0):
    dev = "cuda"
    torch.set_default_device(dev)
    torch.manual_seed(0)
    KV_LORA, PE, PAGE = 512, 64, 64
    T, H = 1, 128
    max_pos = 128
    pos = torch.tensor([34], dtype=torch.int64, device=dev)

    q_nope = (torch.randn(T, H, KV_LORA, dtype=dtypes.bf16) * 0.1)
    q_pe = (torch.randn(T, H, PE, dtype=dtypes.bf16) * 0.1)
    kv_c = (torch.randn(T, KV_LORA, dtype=dtypes.bf16) * 0.1)
    k_pe = (torch.randn(T, PE, dtype=dtypes.bf16) * 0.1)

    cos32, sin32 = compute_cache(max_pos, PE // 2, dtypes.bf16)
    # Optionally embed cos into a wider buffer to emulate a strided [:, :32] view.
    if cos_extra_stride:
        wide = torch.zeros(max_pos, PE // 2 + cos_extra_stride, dtype=dtypes.bf16)
        wide[:, : PE // 2] = cos32
        cos_cache = wide[:, : PE // 2]
        wide_s = torch.zeros(max_pos, PE // 2 + cos_extra_stride, dtype=dtypes.bf16)
        wide_s[:, : PE // 2] = sin32
        sin_cache = wide_s[:, : PE // 2]
    else:
        cos_cache, sin_cache = cos32, sin32

    q_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)

    # --- non-seg (triton path), interleaved kv cache [nblk, page, kv_lora+pe] ---
    nblk = 4
    kv_ns = torch.zeros(nblk, PAGE, 1, KV_LORA + PE, dtype=dtypes.fp8, device=dev)
    q_out_ns = torch.zeros(T, H, KV_LORA + PE, dtype=dtypes.fp8, device=dev)
    slot = torch.tensor([34], dtype=torch.int64, device=dev)
    aiter.fused_qk_rope_concat_and_cache_mla(
        q_nope, q_pe, kv_c, k_pe,
        kv_ns.view(nblk, PAGE, KV_LORA + PE),
        q_out_ns, slot, k_scale, q_scale, pos,
        cos_cache, sin_cache, is_neox, True,
    )

    # --- seg path, flat kv cache [nblk, page*(kv_lora+pe)] ---
    block_stride = PAGE * KV_LORA + PAGE * PE
    kv_seg = torch.zeros(nblk, block_stride, dtype=dtypes.fp8, device=dev)
    q_out_seg = torch.zeros(T, H, 768, dtype=dtypes.fp8, device=dev)
    aiter.fused_qk_rope_concat_and_cache_mla_seg(
        q_nope, q_pe, kv_c, k_pe, kv_seg, q_out_seg, slot,
        k_scale, q_scale, pos, cos_cache, sin_cache, is_neox,
    )

    ns_pe = q_out_ns[..., KV_LORA:].float()
    sg_pe = q_out_seg[..., KV_LORA : KV_LORA + PE].float()
    ns_nope = q_out_ns[..., :KV_LORA].float()
    sg_nope = q_out_seg[..., :KV_LORA].float()
    print(f"[is_neox={is_neox} cos_stride0={cos_cache.stride(0)}]")
    print(f"  cos(nope) = {cosq(sg_nope, ns_nope):.4f}")
    print(f"  cos(pe)   = {cosq(sg_pe, ns_pe):.4f}")
    print(f"  max|pe diff| = {(sg_pe - ns_pe).abs().max():.4f}")

    # hand reference (neox): out[i]=x_i*c_i - x_{i+32}*s_i ; out[i+32]=x_{i+32}*c_i + x_i*s_i
    half = PE // 2
    c = cos_cache[int(pos[0])].float()  # [32]
    s = sin_cache[int(pos[0])].float()
    xp = q_pe[0, 0].float()  # head 0, [64]
    ref = torch.empty(PE)
    if is_neox:
        ref[:half] = xp[:half] * c - xp[half:] * s
        ref[half:] = xp[half:] * c + xp[:half] * s
    else:
        ref[0::2] = xp[0::2] * c - xp[1::2] * s
        ref[1::2] = xp[1::2] * c + xp[0::2] * s
    print(f"  head0 ref[:6]   = {ref[:6].tolist()}")
    print(f"  head0 ns_pe[:6] = {ns_pe[0,0,:6].tolist()}")
    print(f"  head0 sg_pe[:6] = {sg_pe[0,0,:6].tolist()}")
    print(f"  cos(ns_pe,ref)  = {cosq(ns_pe[0,0], ref):.4f}   cos(sg_pe,ref) = {cosq(sg_pe[0,0], ref):.4f}")
    # which indices of head0 differ between seg and ns?
    d = (sg_pe[0, 0] - ns_pe[0, 0]).abs()
    bad = (d > 1e-3).nonzero().reshape(-1).tolist()
    print(f"  head0 diff idx (seg!=ns): {bad}")
    if bad:
        i0 = bad[0]
        print(f"  at idx {i0}: ns={ns_pe[0,0,i0]:.4f} sg={sg_pe[0,0,i0]:.4f} ref={ref[i0]:.4f}")
    # per-head: count heads where seg differs from ns
    dh = (sg_pe[0] - ns_pe[0]).abs().amax(dim=-1)
    nbad_heads = int((dh > 1e-3).sum())
    print(f"  heads with seg!=ns: {nbad_heads}/{H}  (first bad head idx: {int((dh>1e-3).nonzero()[0]) if nbad_heads else -1})")


if __name__ == "__main__":
    for neox in (True, False):
        run(neox)
    print("\n-- emulate strided cos view (stride0=64, width-32 slice) --")
    run(True, cos_extra_stride=32)
