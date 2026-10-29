# SPDX-License-Identifier: MIT
"""The DS V4.1 K1 launch in Gluon: ``layer.build_mono_k1`` (vLLM 8321d11), one
persistent launch a layer and rank, 256 CTAs x 512 threads, S rows:

    seam    slice (160 tasks: the previous layer's reduced output ``pend``
            folded into the residual -- ``FOLD`` off: a settled residual --, the
            layer input, the mix partials), norm (S from CTA 160: RMSNorm ->
            vLLM's MXFP8 -> X8, a flag a wqkv K half), gate (S from CTA 160 + S)
    front   ``front.py``: kt (the 32 CTAs past the wqkv parts), wqkv_a (224
            (row tile, K half) parts), qkv (S), wq_b (every CTA its tasks)

Same launch ABI as FlyDSL's (+ ``health``), the same scratch and tags (K1:
2 e + 1). No epoch marks: K2 ends the launch pair's epoch."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.epoch import load_epoch
from ..common.sync import SMEM_FLAT, T512, deadline
from . import back as B
from . import front as F
from . import seam
from .ffn import first_task, u8


@gluon.jit
def k1(
    res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
    res_out, post_out, comb_out, pre_out, wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos_sin, pos,
    slot, swa, swa_stride, swa_block, q, swa_idx, swa_lens, comp_block, comp_bt, bt_stride,
    topk, t2r, kt, klen, normed_out, qr_out, scratch, epoch, tl, health,
    S: gl.constexpr, TP: gl.constexpr, HIDDEN: gl.constexpr, RATIO: gl.constexpr,
    FOLD: gl.constexpr, INDEX: gl.constexpr, SWA_W: gl.constexpr, LIN: gl.constexpr,
    PMIX: gl.constexpr, X8: gl.constexpr, X8S: gl.constexpr, XRDY: gl.constexpr,
    QKVP: gl.constexpr, QX8: gl.constexpr, QX8S: gl.constexpr, QRDY: gl.constexpr,
):  # fmt: skip
    gl.static_assert(not INDEX, "an index source's K1 (normed / qr out) is not ported yet")
    res_in, pend, post_in, comb_in, pre_in = u8(res_in), u8(pend), u8(post_in), u8(comb_in), u8(pre_in)
    hc_fn, hc_scale, hc_base, attn_w = u8(hc_fn), u8(hc_scale), u8(hc_base), u8(attn_w)
    res_out, post_out, comb_out, pre_out = u8(res_out), u8(post_out), u8(comb_out), u8(pre_out)
    wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s = u8(wqkv), u8(wqkv_s), u8(qn_w), u8(kvn_w), u8(wqb), u8(wqb_s)
    cos_sin, pos, slot, swa, q = u8(cos_sin), u8(pos), u8(slot), u8(swa), u8(q)
    swa_idx, swa_lens, comp_bt, topk, t2r = u8(swa_idx), u8(swa_lens), u8(comp_bt), u8(topk), u8(t2r)
    kt, klen, scratch, epoch, health = u8(kt), u8(klen), u8(scratch), u8(epoch), u8(health)
    HEADS: gl.constexpr = 64 // TP
    SLICES: gl.constexpr = HIDDEN // 32
    RLW: gl.constexpr = seam.pow2(S * 128 + 1)
    tid = gl.arange(0, 512, layout=T512)
    lane = tid % 64
    wave = tid // 64
    bid = gl.program_id(0)
    ep = load_epoch(epoch)
    tag = (ep << 1) + 1 + gl.zeros_like(tid)
    dl = deadline(tid)
    sb = scratch
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    rl = gl.allocate_shared_memory(gl.float32, [RLW], SMEM_FLAT)
    fl = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    # ---- the attention seam
    for task in range(first_task(bid, 0), SLICES, 256):
        seam.stage_slice(res_in, comb_in, post_in, pre_in, hc_fn, res_out, task, tag, sb + LIN,
                         sb + PMIX, red, rl, fl, red, tid, lane, wave, S, HIDDEN, RLW - 1, pend,
                         FOLD)  # fmt: skip
    gl.barrier()
    for t in range(first_task(bid, SLICES), S, 256):
        seam.stage_norm(t, tag, sb + LIN, attn_w, sb, sb + XRDY, red, tid, lane, wave, dl, health,
                        0, HIDDEN, sb + X8, sb + X8S)  # fmt: skip
    for t in range(first_task(bid, SLICES + S), S, 256):
        seam.stage_gate(t, tag, sb + PMIX, red, hc_scale, hc_base, pre_out, post_out, comb_out,
                        tid, lane, wave, dl, health, 0, S, HIDDEN)  # fmt: skip
    gl.barrier()
    # ---- the front
    x_s = gl.allocate_shared_memory(gl.uint8, [65536], B.SH1)
    c_s = gl.allocate_shared_memory(gl.uint8, [2048], B.SH1)
    if bid >= 224:
        # an index source's top-k comes after this launch: its K2 builds the table
        for t in range(bid - 224, S, 32):
            F.stage_kt(t, slot, swa_idx, swa_lens, pos, t2r, topk, comp_bt, bt_stride, comp_block,
                       kt, klen, S, RATIO, SWA_W)  # fmt: skip
    else:
        F.stage_wqkv(bid % 112, bid // 112, wqkv, wqkv_s, sb + X8, sb + X8S, sb + XRDY, sb + QKVP,
                     x_s, c_s, red, tag, tid, dl, health, 0, S)  # fmt: skip
    gl.barrier()
    for t in range(first_task(bid, 0), S, 256):
        F.stage_qkv(t, sb + QKVP, qn_w, kvn_w, pos, slot, cos_sin, swa, swa_stride, swa_block,
                    sb + QX8, sb + QX8S, sb + QRDY, red, tag, tid, lane, wave, dl, health, 0,
                    S)  # fmt: skip
    gl.barrier()
    F.stage_wqb(bid, wqb, wqb_s, pos, cos_sin, sb + QX8, sb + QX8S, sb + QRDY, q, x_s, c_s, red,
                tag, tid, dl, health, 0, S, HEADS)  # fmt: skip


def build_constexprs(s: int, tp: int, ratio: int = 1, fold: bool = True) -> dict:
    """K1's constexprs for width s, TP tp: upstream's own scratch layout. GPU
    image only (upstream imports aiter)."""
    from vllm.models.deepseek_v41.amd.mono import layer as L

    off = {k: v[0] for k, v in L.scratch_layout(s, tp).items()}
    return dict(
        S=s, TP=tp, HIDDEN=5120, RATIO=ratio, FOLD=fold, INDEX=False, SWA_W=128,
        LIN=off["lin"], PMIX=off["pmix"], X8=off["x8"], X8S=off["x8s"], XRDY=off["xrdy"],
        QKVP=off["qkvp"], QX8=off["qx8"], QX8S=off["qx8s"], QRDY=off["qrdy"],
    )  # fmt: skip
