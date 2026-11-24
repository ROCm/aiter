"""Zero-pad a TP shard's inter dim (raw, unshuffled MoE weights) to a multiple the
kernel tiles take: w1 [E, 2I, H/2] (gate rows, then up rows), w1s [E, 2I, H/32],
w2 [E, H, I/2], w2s [E, H, I/32] -> the same with I = P. Zero weights add exact zeros
(swiglu(0, 0) = 0, and a zero K slice adds nothing), so results are unchanged."""
import torch


def pad_inter(w1, w1s, w2, w2s, I, P):
    if P == I:
        return w1, w1s, w2, w2s
    assert P > I and P % 128 == 0 and I % 32 == 0
    E, _, hb = w1.shape

    def rows(t, fill):
        o = torch.full((E, 2 * P, t.shape[2]), fill, dtype=t.dtype, device=t.device)
        o[:, :I] = t[:, :I]
        o[:, P:P + I] = t[:, I:]
        return o

    def cols(t, n, fill):
        o = torch.full((E, t.shape[1], n), fill, dtype=t.dtype, device=t.device)
        o[:, :, :t.shape[2]] = t
        return o

    return rows(w1, 0), rows(w1s, 127), cols(w2, P // 2, 0), cols(w2s, P // 32, 127)


def pad_w2_scale(w2s, I):
    """w2's e8m0 scales [E, H, I/32] -> [E, H, ceil128(I)/32], new columns 0 (2^-127:
    the kernel's tail k-blocks past I scale to ~0). The n32k4 scale layout needs
    I/32 % 4 == 0; the weights themselves stay at I."""
    n = (I + 127) // 128 * 4
    if w2s.shape[2] == n:
        return w2s
    o = torch.zeros((w2s.shape[0], w2s.shape[1], n), dtype=w2s.dtype, device=w2s.device)
    o[:, :, :w2s.shape[2]] = w2s
    return o
