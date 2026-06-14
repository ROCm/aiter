# SPDX-License-Identifier: MIT
"""Compare the layer-0 decode attention output between the TRITON path (known
good) and the SEG path for the SAME greedy prompt/step, plus an fp32 ground
truth computed from the triton (token-major) KV cache.

Inputs:
  - triton dump: ~/mla_decode_dump/mla_decode_layer000_step0_rank0.pt
      (written by atom.utils.mla_decode_dump.dump_decode_mla; kv_layout=interleaved)
  - seg dump:    ~/mla_decode_dump/seg_decode_OK_layer000_step000.pt
      (written by attention_mla._dump_seg_decode_failure)

Usage:
  python cmp_triton_vs_seg.py <triton_dump.pt> <seg_dump.pt>

What it tells us:
  - cos(triton_o, truth)  -> sanity: triton path is correct (~1).
  - cos(seg_o,    truth)  -> if << 1, the SEG decode output is wrong vs ground
                             truth even though seg's own asm-vs-self-ref is ~1,
                             meaning seg's q_out and/or cache CONTENT is wrong.
  - cos(seg_o, triton_o)  -> direct seg-vs-good comparison.
"""

import sys
import torch


def cosq(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def truth_from_tokenmajor(d, kv_lora=512, pe=64):
    """fp32 ground-truth latent attention from a token-major triton dump."""
    q = d["q"].float()  # [B, nhead, 576] (already roped); triton casts fp8->bf16
    if q.dim() == 3:
        q = q
    kvc = d["kv_compact"].float()  # [U, page_size, 1, 576] token-major
    page_size = kvc.shape[1]
    kv_indptr = d["kv_indptr"].long()
    kv_indices = d["kv_indices"].long()
    kv_last = d["kv_last_page_lens"].long()
    sm_scale = float(d["sm_scale"])
    q_scale = d["q_scale"].float().item() if d["q_scale"] is not None else 1.0
    kv_scale = d["kv_scale"].float().item() if d["kv_scale"] is not None else 1.0
    bs = kv_indptr.numel() - 1
    outs = []
    for i in range(bs):
        p0, p1 = int(kv_indptr[i]), int(kv_indptr[i + 1])
        n_pages = p1 - p0
        last = int(kv_last[i])
        seq = (n_pages - 1) * page_size + last
        pages = kv_indices[p0:p1]
        kv = torch.cat([kvc[pages[j], :, 0, :] for j in range(n_pages)], 0)[:seq] * kv_scale
        k_full = kv  # [seq, 576] token-major: [nope|pe]
        v = kv[:, :kv_lora]
        qi = q[i] * q_scale  # [nhead, 576]
        s = (qi @ k_full.T) * sm_scale
        s = s - s.max(-1, keepdim=True).values
        w = torch.softmax(s, -1)
        outs.append(w @ v)  # [nhead, 512]
    return torch.stack(outs, 0)  # [B, nhead, 512]


def main(triton_path, seg_path):
    dt = torch.load(triton_path, map_location="cpu")
    ds = torch.load(seg_path, map_location="cpu")

    truth = truth_from_tokenmajor(dt)
    to = dt["o_server"].float()[..., : truth.shape[-1]]
    so = ds["o"].float()[..., : truth.shape[-1]]

    print(f"triton dump: o.shape={tuple(dt['o_server'].shape)} kv_indptr={dt['kv_indptr'].tolist()[:4]} last={dt['kv_last_page_lens'].tolist()[:2]}")
    print(f"seg    dump: o.shape={tuple(ds['o'].shape)} kv_indptr={ds['kv_indptr'].tolist()[:4]} last={ds['kv_last_page_lens'].tolist()[:2]}")
    print()
    print(f"cos(triton_o, truth) = {cosq(to, truth):.4f}   (sanity: should be ~1)")
    print(f"cos(seg_o,    truth) = {cosq(so, truth):.4f}   (<<1 => seg q_out/cache CONTENT wrong)")
    print(f"cos(seg_o, triton_o) = {cosq(so, to):.4f}")
    # also compare q content if shapes align
    try:
        qt = dt["q"].float()
        qs = ds["q"].float()
        print(f"\nq shapes: triton {tuple(qt.shape)} {dt['q'].dtype}  seg {tuple(qs.shape)} {ds['q'].dtype}")
        if qt.shape == qs.shape:
            kv_lora = 512
            print(f"cos(seg_q, triton_q)        = {cosq(qs, qt):.4f}   (<<1 => fused_seg q_out wrong)")
            # split into nope (first 512, NO rope) and pe (last 64, roped)
            print(f"cos(seg_q[nope], triton_q[nope]) = {cosq(qs[..., :kv_lora], qt[..., :kv_lora]):.4f}   (nope has no rope; <<1 => quant/layout issue)")
            print(f"cos(seg_q[pe],   triton_q[pe])   = {cosq(qs[..., kv_lora:], qt[..., kv_lora:]):.4f}   (<<1 => rope issue in fused_seg)")
            # per-head cosine over first few heads (detect head scramble)
            qsf, qtf = qs.reshape(-1, qs.shape[-1]), qt.reshape(-1, qt.shape[-1])
            ph = [round(cosq(qsf[h], qtf[h]), 3) for h in range(min(6, qsf.shape[0]))]
            print(f"per-head cos(seg,triton) heads[0:6] = {ph}")
            # absmax sanity
            print(f"absmax: triton_q={qt.abs().max():.2f} seg_q={qs.abs().max():.2f}")
            # --- neox<->gptj layout reconciliation on the pe(64) segment ---
            sp = qs[..., kv_lora:].reshape(-1, 64)   # seg pe
            tp = qt[..., kv_lora:].reshape(-1, 64)    # triton pe
            half = 32
            idx_g2n = torch.empty(64, dtype=torch.long)   # gptj-order -> neox-order
            idx_g2n[:half] = torch.arange(0, 64, 2)
            idx_g2n[half:] = torch.arange(1, 64, 2)
            idx_n2g = torch.empty(64, dtype=torch.long)   # neox-order -> gptj-order
            idx_n2g[0::2] = torch.arange(0, half)
            idx_n2g[1::2] = torch.arange(half, 64)
            print(f"cos(seg_pe, triton_pe[g2n])    = {cosq(sp, tp[:, idx_g2n]):.4f}  (seg=neox, triton=gptj?)")
            print(f"cos(seg_pe, triton_pe[n2g])    = {cosq(sp, tp[:, idx_n2g]):.4f}  (seg=gptj, triton=neox?)")
            print(f"cos(seg_pe[firsthalf], triton_pe[firsthalf]) = {cosq(sp[:, :half], tp[:, :half]):.4f}")
            print(f"cos(seg_pe[2ndhalf],  triton_pe[2ndhalf])  = {cosq(sp[:, half:], tp[:, half:]):.4f}")
        else:
            print(f"q shapes differ: triton {tuple(qt.shape)} vs seg {tuple(qs.shape)}")
    except Exception as e:
        print(f"q compare skipped: {e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
