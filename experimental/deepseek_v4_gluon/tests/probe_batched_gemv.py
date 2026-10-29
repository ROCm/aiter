# SPDX-License-Identifier: MIT
"""Probe of the shared-expert GEMV pattern on the patched Triton: batched
scaled MFMA, wave w over K steps w, w + 8, .. (one 1024-column window a step),
MXFP8 weights [16, K] from global, the MXFP8 x tile and its codes from LDS
through [waves, K window, tokens] views. CPU: compile; GPU (``--run``): against
a torch reference of the dequantized operands."""

import sys

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

MMA = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [8, 1, 1]))
AD = gl.constexpr(gl.DotOperandLayout(0, MMA, 16))
BD = gl.constexpr(gl.DotOperandLayout(1, MMA, 16))
A_LD = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 16, 4], [8, 1, 1], [2, 1, 0]))
X_LD = gl.constexpr(gl.BlockedLayout([1, 16], [16, 4], [8, 1], [1, 0]))
S3 = gl.constexpr(gl.BlockedLayout([1, 1, 4], [1, 16, 4], [8, 1, 1], [2, 1, 0]))
OUT = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))
SH4 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [3, 2, 1, 0]))
SH3 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]))


@gluon.jit
def k_gemv(w, ws, x, xs, out, K: gl.constexpr):
    """out[w][r][n] = sum over wave w's K steps of W[r] . x[n] (scaled)."""
    STEPS: gl.constexpr = K // 1024  # a step: 8 waves x 128 columns
    # the x tile (16 tokens x K fp8) and its codes into LDS as
    # [step][token][wave][128] / [step][token][wave][4]: a step's view is an
    # index, then a permute (a reshape of a sliced view is NYI)
    X4: gl.constexpr = gl.BlockedLayout([1, 1, 1, 16], [1, 2, 4, 8], [1, 8, 1, 1], [3, 2, 1, 0])
    s_ = gl.arange(0, STEPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, X4))))
    t_ = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, X4))))
    w_ = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, X4))))
    k_ = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, X4))))
    s4 = s_[:, None, None, None]
    t4 = t_[None, :, None, None]
    w4 = w_[None, None, :, None]
    xoff = t4 * K + (s4 * 8 + w4) * 128 + k_[None, None, None, :]
    xt = gl.allocate_shared_memory(gl.uint8, [STEPS, 16, 8, 128], SH4, gl.load(x + xoff))
    xt = xt._reinterpret(gl.uint8, [STEPS, 16, 8, 128], SH3)
    C4: gl.constexpr = gl.BlockedLayout([1, 1, 1, 4], [1, 8, 8, 1], [1, 8, 1, 1], [3, 2, 1, 0])
    cs_ = gl.arange(0, STEPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, C4))))
    ct_ = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, C4))))
    cw_ = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, C4))))
    cg_ = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, C4))))
    coff = (ct_[None, :, None, None] * (K // 32) + (cs_[:, None, None, None] * 8
            + cw_[None, None, :, None]) * 4 + cg_[None, None, None, :])  # fmt: skip
    ct = gl.allocate_shared_memory(gl.uint8, [STEPS, 16, 8, 4], SH4, gl.load(xs + coff))
    ct = ct._reinterpret(gl.uint8, [STEPS, 16, 8, 4], SH3)
    wa = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A_LD)))
    ra = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A_LD)))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, A_LD)))
    ws3 = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, S3)))
    rs3 = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, S3)))
    gs3 = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, S3)))
    acc = gl.zeros((8, 16, 16), gl.float32, MMA)
    for i in gl.static_range(STEPS):
        k0 = (8 * i + wa[:, None, None]) * 128
        a = gl.convert_layout(gl.load(w + ra[None, :, None] * K + k0 + ka[None, None, :],
                                      cache_modifier=".cg"), AD)  # fmt: skip
        sa = gl.load(ws + rs3[None, :, None] * 0 + (8 * i + ws3[:, None, None]) * 4
                     + gs3[None, None, :])  # one 32-row block: scale row 0  # fmt: skip
        sa = gl.convert_layout(sa, gl.amd.cdna4.get_mfma_scale_layout(AD, [8, 16, 4]))
        b = xt.index(i).permute((1, 2, 0)).load(BD)
        sb = ct.index(i).permute((1, 0, 2)).load(gl.amd.cdna4.get_mfma_scale_layout(BD, [8, 16, 4]))
        acc = gl.amd.cdna4.mfma_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    o = gl.convert_layout(acc, OUT)
    ow = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, OUT)))
    orr = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, OUT)))
    on = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, OUT)))
    gl.store(out + (ow[:, None, None] * 16 + orr[None, :, None]) * 16 + on[None, None, :], o)


def main():
    K = 4096
    if "--run" not in sys.argv:
        from deepseek_v4_gluon.common.probe import compile_probe

        r = compile_probe(k_gemv, {"w": "*u8", "ws": "*u8", "x": "*u8", "xs": "*u8",
                                   "out": "*fp32", "K": "constexpr"}, {"K": K})  # fmt: skip
        n = r["asm"].count("v_mfma_scale_f32_16x16x128_f8f6f4")
        print(f"compiled: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} scaled_mfma={n}")
        return
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    w = torch.randint(20, 40, (16, K), dtype=torch.uint8, device=dev)
    ws = torch.randint(120, 134, (1, K // 32), dtype=torch.uint8, device=dev)
    x = torch.randint(20, 40, (16, K), dtype=torch.uint8, device=dev)
    xs = torch.randint(120, 134, (16, K // 32), dtype=torch.uint8, device=dev)
    out = torch.empty(8, 16, 16, dtype=torch.float32, device=dev)
    k_gemv[(1,)](w, ws, x, xs, out, K, num_warps=8)
    wf = w.view(torch.float8_e4m3fn).float() * torch.exp2(ws.float() - 127).repeat_interleave(32, 1)
    xf = x.view(torch.float8_e4m3fn).float() * torch.exp2(xs.float() - 127).repeat_interleave(32, 1)
    ref = torch.zeros(8, 16, 16, device=dev)
    for wv in range(8):
        cols = torch.cat([torch.arange((8 * i + wv) * 128, (8 * i + wv + 1) * 128)
                          for i in range(K // 1024)]).to(dev)  # fmt: skip
        ref[wv] = wf[:, cols] @ xf[:, cols].T
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f"run: max rel err {err:.2e} ->", "PASS" if err < 1e-5 else "FAIL")


if __name__ == "__main__":
    main()
