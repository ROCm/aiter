# SPDX-License-Identifier: MIT
"""Compare the seg KV cache content of the SAME physical tokens across two
decode-step dumps. Prefill-written tokens (e.g. 0..5) must be byte-identical
between an early and a later step; if a later decode incremental-write went to
the wrong slot, the earlier tokens get corrupted and will differ here."""

import sys
import torch


def unpack(path, page_size=64, kv_lora=512, pe=64):
    b = torch.load(path, map_location="cpu")
    seg = b["seg_kv_compact"]  # [U, page_size, 1, kv_lora+pe], seg-packed
    last = int(b["kv_last_page_lens"][0].item())
    n_pages = int(b["kv_indptr"][1].item()) - int(b["kv_indptr"][0].item())
    seq = (n_pages - 1) * page_size + last
    flat = seg.reshape(seg.shape[0], -1).to(torch.float32)
    nope = flat[:, : page_size * kv_lora].reshape(seg.shape[0], page_size, kv_lora)
    pep = flat[:, page_size * kv_lora :].reshape(seg.shape[0], page_size, pe)
    return b, nope, pep, seq


def main(p_early, p_late):
    be, nope_e, pe_e, seq_e = unpack(p_early)
    bl, nope_l, pe_l, seq_l = unpack(p_late)
    print(f"early seq={seq_e}  late seq={seq_l}")
    # page 0 assumed (single page case). Compare the first seq_e tokens.
    n = min(seq_e, seq_l)
    nd = (nope_e[0, :n] - nope_l[0, :n]).abs()
    pd = (pe_e[0, :n] - pe_l[0, :n]).abs()
    print(f"comparing first {n} tokens of page 0 (prefill+early-decode tokens):")
    print(f"  nope max|Δ|={nd.max().item():.4f}  mean|Δ|={nd.mean().item():.6f}")
    print(f"  pe   max|Δ|={pd.max().item():.4f}  mean|Δ|={pd.mean().item():.6f}")
    # per-token diff to localize which token drifted
    per_tok = (nope_e[0, :n] - nope_l[0, :n]).abs().amax(dim=-1)
    bad = (per_tok > 1e-3).nonzero().flatten().tolist()
    print(f"  tokens with nope drift > 1e-3: {bad}")
    per_tok_pe = (pe_e[0, :n] - pe_l[0, :n]).abs().amax(dim=-1)
    badpe = (per_tok_pe > 1e-3).nonzero().flatten().tolist()
    print(f"  tokens with pe   drift > 1e-3: {badpe}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
