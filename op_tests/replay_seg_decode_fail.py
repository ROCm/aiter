# SPDX-License-Identifier: MIT
"""Replay a captured failing seg MLA decode step.

Usage:
    python op_tests/replay_seg_decode_fail.py ~/mla_decode_dump/seg_decode_FAIL_layerNNN.pt

Loads the exact kernel inputs saved by attention_mla._dump_seg_decode_failure,
then:
  (a) re-runs aiter.mla.mla_decode_fwd (num_kv_splits=1) to confirm whether the
      asm kernel itself reproduces the NaN/inf on these inputs (kernel bug), and
  (b) unpacks the seg-packed KV to fp32 and computes a manual reference MLA
      attention -> tells us whether the *inputs* (q / kv / metadata) are already
      degenerate (upstream bug) independent of the kernel.
"""

import sys
import torch
import aiter
from aiter import dtypes


def _str2dtype(s):
    return {
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
    }[s]


def unpack_seg_pages(seg_kv_compact, page_size, kv_lora, pe_dim):
    """seg_kv_compact: [U, page_size, 1, kv_lora+pe] contiguous, seg-packed in
    memory as [page_size*kv_lora nope][page_size*pe pe] per page.
    Returns nope [U, page_size, kv_lora], pe [U, page_size, pe] in fp32."""
    U = seg_kv_compact.shape[0]
    flat = seg_kv_compact.reshape(U, -1).to(torch.float32)
    nope = flat[:, : page_size * kv_lora].reshape(U, page_size, kv_lora)
    pe = flat[:, page_size * kv_lora :].reshape(U, page_size, pe_dim)
    return nope, pe


def main(path):
    b = torch.load(path, map_location="cpu")
    dev = "cuda"
    page_size = b["page_size"]
    kv_lora = b["kv_lora_rank"]
    pe_dim = b["qk_rope_head_dim"]
    v_head_dim = b["v_head_dim"]
    nhead = b["padded_num_heads"]
    sm_scale = b["sm_scale"]
    q_scale = b["q_scale"].item() if b["q_scale"] is not None else 1.0
    kv_scale = b["kv_scale"].item() if b["kv_scale"] is not None else 1.0

    print(f"=== replay {path} ===")
    print(
        f"layer={b['layer_num']} nhead={nhead} page_size={page_size} "
        f"kv_lora={kv_lora} pe={pe_dim} vdim={v_head_dim} "
        f"sm_scale={sm_scale:.6f} q_scale={q_scale} kv_scale={kv_scale}"
    )
    print(
        f"kv_indptr={b['kv_indptr'].tolist()} "
        f"kv_last_page_lens={b['kv_last_page_lens'].tolist()} "
        f"max_q_len={b['max_q_len']} q.dtype={b['q_dtype']} kv.dtype={b['kv_dtype']}"
    )

    q = b["q"].to(dev)
    seg_kv = b["seg_kv_compact"].to(dev).contiguous()
    o = torch.zeros_like(b["o"]).to(dev)
    qo_indptr = b["qo_indptr"].to(dev).to(torch.int32)
    kv_indptr = b["kv_indptr"].to(dev).to(torch.int32)
    kv_indices = b["kv_indices_remapped"].to(dev).to(torch.int32)
    kv_last = b["kv_last_page_lens"].to(dev).to(torch.int32)
    q_scale_t = b["q_scale"].to(dev) if b["q_scale"] is not None else None
    kv_scale_t = b["kv_scale"].to(dev) if b["kv_scale"] is not None else None

    # --- input sanity ---
    q_f32 = q.to(torch.float32)
    print(
        f"\n[inputs] q finite={torch.isfinite(q_f32).all().item()} "
        f"q absmax={q_f32.abs().max().item():.3f} "
        f"seg_kv finite={torch.isfinite(seg_kv.to(torch.float32)).all().item()} "
        f"seg_kv absmax={seg_kv.to(torch.float32).abs().max().item():.3f}"
    )

    # --- (a) re-run the asm kernel ---
    try:
        aiter.mla.mla_decode_fwd(
            q,
            seg_kv,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last,
            b["max_q_len"],
            page_size,
            1,
            sm_scale,
            num_kv_splits=1,
            q_scale=q_scale_t,
            kv_scale=kv_scale_t,
        )
        finite = torch.isfinite(o.to(torch.float32)).all().item()
        print(
            f"\n[asm replay] o finite={finite} "
            f"nan={torch.isnan(o.to(torch.float32)).sum().item()} "
            f"absmax={o.to(torch.float32).abs().max().item():.3f}"
        )
    except Exception as e:
        print(f"\n[asm replay] raised: {e}")

    # --- (b) fp32 reference attention from the same inputs ---
    bs = kv_indptr.numel() - 1
    nope, pe = unpack_seg_pages(seg_kv.cpu(), page_size, kv_lora, pe_dim)
    for i in range(bs):
        p0 = int(kv_indptr[i].item())
        p1 = int(kv_indptr[i + 1].item())
        n_pages = p1 - p0
        if n_pages == 0:
            continue
        last = int(kv_last[i].item())
        seq_len = (n_pages - 1) * page_size + last
        # gather keys for this batch in page order
        pages = kv_indices[p0:p1].cpu()
        k_nope = torch.cat([nope[pages[j]] for j in range(n_pages)], dim=0)  # [n_pages*ps, kv_lora]
        k_pe = torch.cat([pe[pages[j]] for j in range(n_pages)], dim=0)
        k_nope = k_nope[:seq_len] * kv_scale
        k_pe = k_pe[:seq_len] * kv_scale
        k_full = torch.cat([k_nope, k_pe], dim=-1)  # [seq, 576]
        # query for this batch (decode_qlen=1 assumed)
        qi = q_f32[i].cpu() * q_scale  # [nhead, 576]
        scores = (qi @ k_full.T) * sm_scale  # [nhead, seq]
        scores = scores - scores.max(dim=-1, keepdim=True).values
        w = torch.softmax(scores, dim=-1)
        ref = w @ k_nope  # [nhead, kv_lora] (MLA value = nope part)
        oi = o[i].to(torch.float32).cpu()[:, :kv_lora]
        cos = torch.nn.functional.cosine_similarity(
            ref.reshape(-1), oi.reshape(-1), dim=0
        ).item()
        print(
            f"\n[fp32 ref] batch={i} seq_len={seq_len} n_pages={n_pages} last={last} "
            f"ref finite={torch.isfinite(ref).all().item()} ref absmax={ref.abs().max().item():.3f} "
            f"cos(asm,ref)={cos:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
