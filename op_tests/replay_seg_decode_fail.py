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

    # Faithfully rebuild the *padded* q the asm actually saw: per-head stride
    # from the bundle (e.g. 768 elems) with the valid 576 copied in and padding
    # zeroed (matching the unit test's split3 layout). The bundle's saved `q`
    # is the contiguous [.,nhead,576] materialization which loses the stride.
    q_saved = b["q"]  # [B, nhead, 576] contiguous
    q_stride = b.get("q_stride", None)
    head_stride = q_stride[1] if q_stride is not None else q_saved.shape[-1]
    Bq, Hq, Dq = q_saved.shape
    if head_stride > Dq:
        q_pad = torch.zeros((Bq, Hq, head_stride), dtype=q_saved.dtype)
        q_pad[..., :Dq] = q_saved
        q = torch.as_strided(
            q_pad.to(dev),
            size=(Bq, Hq, Dq),
            stride=(Hq * head_stride, head_stride, 1),
        )
        print(f"[note] rebuilt padded q: head_stride={head_stride} (valid {Dq})")
    else:
        q = q_saved.to(dev)
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

    # --- (a2) q-sensitivity: does the asm output actually depend on q? ---
    # If o barely changes when we zero / perturb q, the kernel is not using q
    # correctly (q layout/stride mismatch) and collapses to ~uniform averaging.
    o_base = o.clone()
    for tag, qmod in (
        ("q*0", torch.zeros_like(q)),
        ("q*4", (q.to(torch.float32) * 4).clamp(-448, 448).to(q.dtype)),
    ):
        o2 = torch.zeros_like(b["o"]).to(dev)
        try:
            aiter.mla.mla_decode_fwd(
                qmod, seg_kv, o2, qo_indptr, kv_indptr, kv_indices, kv_last,
                b["max_q_len"], page_size, 1, sm_scale,
                num_kv_splits=1, q_scale=q_scale_t, kv_scale=kv_scale_t,
            )
            d = (o2.to(torch.float32) - o_base.to(torch.float32))
            rel = d.abs().max().item() / (o_base.to(torch.float32).abs().max().item() + 1e-9)
            cos = torch.nn.functional.cosine_similarity(
                o2.to(torch.float32).reshape(-1), o_base.to(torch.float32).reshape(-1), dim=0
            ).item()
            print(f"[q-sens] {tag}: max|Δo|/max|o|={rel:.4f} cos(o_mod,o_base)={cos:.4f}")
        except Exception as e:
            print(f"[q-sens] {tag}: raised {e}")
    o = o_base

    # --- (a3) padding sensitivity: does garbage in q's [576:768] padding
    # corrupt the output? If yes, ATOM must zero-init the q_out padding. ---
    if head_stride > Dq:
        for tag, fill in (("pad=0", 0.0), ("pad=20", 20.0), ("pad=rand", None)):
            qp = torch.zeros((Bq, Hq, head_stride), dtype=torch.float32)
            qp[..., :Dq] = q_saved.to(torch.float32)
            if fill is None:
                qp[..., Dq:] = torch.randn((Bq, Hq, head_stride - Dq)) * 20
            else:
                qp[..., Dq:] = fill
            qpv = torch.as_strided(
                qp.clamp(-448, 448).to(q.dtype).to(dev),
                size=(Bq, Hq, Dq),
                stride=(Hq * head_stride, head_stride, 1),
            )
            o3 = torch.zeros_like(b["o"]).to(dev)
            aiter.mla.mla_decode_fwd(
                qpv, seg_kv, o3, qo_indptr, kv_indptr, kv_indices, kv_last,
                b["max_q_len"], page_size, 1, sm_scale,
                num_kv_splits=1, q_scale=q_scale_t, kv_scale=kv_scale_t,
            )
            cos = torch.nn.functional.cosine_similarity(
                o3.to(torch.float32).reshape(-1), o_base.to(torch.float32).reshape(-1), dim=0
            ).item()
            print(f"[pad-sens] {tag}: cos(o,o_base[pad=0])={cos:.4f}")

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
        k_nope = k_nope * kv_scale  # keep full (unmasked) length
        k_pe = k_pe * kv_scale
        # query for this batch (decode_qlen=1 assumed)
        qi = q_f32[i].cpu() * q_scale  # [nhead, 576]
        oi = o[i].to(torch.float32).cpu()[:, :kv_lora]

        def cos_for(n_keys, sc):
            kf = torch.cat([k_nope[:n_keys], k_pe[:n_keys]], dim=-1)
            sc_mat = (qi @ kf.T) * sc
            sc_mat = sc_mat - sc_mat.max(dim=-1, keepdim=True).values
            w = torch.softmax(sc_mat, dim=-1)
            ref = w @ k_nope[:n_keys]
            return torch.nn.functional.cosine_similarity(
                ref.reshape(-1), oi.reshape(-1), dim=0
            ).item()

        print(f"\n[fp32 ref] batch={i} seq_len={seq_len} n_pages={n_pages} last={last}")
        sqrt576 = 1.0 / ((kv_lora + pe_dim) ** 0.5)
        # (1) named scales over the masked seq_len keys
        for name, sc in {
            "passed": sm_scale,
            "1/sqrt(576)": sqrt576,
            "1/sqrt(512)": 1.0 / (kv_lora**0.5),
            "1/sqrt(192)": 1.0 / (192**0.5),
        }.items():
            print(f"    [seq={seq_len:3d}] scale={name:>12s} ({sc:.6f})  cos={cos_for(seq_len, sc):.4f}")
        # (2) fine scale sweep over seq_len keys -> best achievable cos
        best = max(
            ((cos_for(seq_len, s), s) for s in [x / 1000.0 for x in range(2, 200, 2)]),
        )
        print(f"    [seq={seq_len:3d}] BEST cos={best[0]:.4f} @ scale={best[1]:.4f}")
        # (3) does using the WHOLE page (page_size keys) match better? -> would
        # mean the asm ignores last_page_len and reads padding/garbage slots.
        full = min(page_size, k_nope.shape[0])
        bestf = max(
            ((cos_for(full, s), s) for s in [x / 1000.0 for x in range(2, 200, 2)]),
        )
        print(
            f"    [seq={full:3d} (full page)] passed_scale cos={cos_for(full, sm_scale):.4f}  "
            f"1/sqrt576 cos={cos_for(full, sqrt576):.4f}  BEST cos={bestf[0]:.4f} @ scale={bestf[1]:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
