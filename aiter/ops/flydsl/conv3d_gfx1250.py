# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host entry for gfx1250 FlyDSL implicit-GEMM conv3d.

Pre-pads (all padding_modes), converts to NDHWC, pads C/groups to 32, packs
weights, picks a tile, and launches ``compile_conv3d_implicit_gfx1250``.
"""

from __future__ import annotations

import functools
import weakref

import torch

from aiter.ops.flydsl.kernels.conv.conv3d_gfx1250_utils import (
    CGP_ALIGN,
    DEFAULT_BLOCK_ROWS,
    DEFAULT_TILE,
    TILE_K,
    out_extent,
)
from aiter.ops.flydsl.kernels.conv.conv3d_implicit_gfx1250 import (
    compile_conv3d_implicit_gfx1250,
    make_conv3d_implicit_param_gfx1250,
)
from aiter.ops.flydsl.kernels.conv.conv3d_gfx950_utils import _as_stream

# Tuned later; MVP heuristic ladder (tile_m, tile_n, m_warp, n_warp, block_rows).
_TILE_LADDER = (
    (128, 128, 2, 4, 8),
    (128, 64, 2, 2, 8),
    (64, 64, 2, 2, 4),
    (64, 32, 2, 1, 4),
    (32, 32, 1, 1, 2),
)


def _pad_cgp(c_per_group: int) -> int:
    return (c_per_group + CGP_ALIGN - 1) // CGP_ALIGN * CGP_ALIGN


def _as_tuple3(v, name):
    if isinstance(v, int):
        return (v, v, v)
    t = tuple(v)
    if len(t) == 1:
        return t * 3
    assert len(t) == 3, f"{name} must be int or length-3, got {v}"
    return t


def _shape_ncdhw(x, ndhwc):
    if ndhwc:
        n, d, h, w, c = x.shape
    else:
        n, c, d, h, w = x.shape
    return n, c, d, h, w


def _to_ndhwc(x, in_ndhwc):
    if in_ndhwc:
        return x.contiguous()
    # NCDHW → NDHWC
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _host_pad_ndhwc(x_ndhwc, pad_lo, pad_hi, mode):
    """Pad NDHWC on D/H/W. ``mode`` is torch pad mode (constant/reflect/replicate/circular)."""
    pt, ph, pw = pad_lo
    pt2, ph2, pw2 = pad_hi
    if not (pt or ph or pw or pt2 or ph2 or pw2):
        return x_ndhwc
    # F.pad on NDHWC: pad dims from last — (C_lo,C_hi, W_lo,W_hi, H_lo,H_hi, D_lo,D_hi)
    pads = (0, 0, pw, pw2, ph, ph2, pt, pt2)
    if mode == "zeros":
        return torch.nn.functional.pad(x_ndhwc, pads, mode="constant", value=0)
    # reflect/replicate/circular need channels-first for F.pad spatial semantics
    x = x_ndhwc.permute(0, 4, 1, 2, 3)
    x = torch.nn.functional.pad(x, (pw, pw2, ph, ph2, pt, pt2), mode=mode)
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _pad_channels_ndhwc(x_ndhwc, groups, cgp):
    n, d, h, w, c = x_ndhwc.shape
    cg = c // groups
    if cgp == cg:
        return x_ndhwc
    # Pad each group independently: (N,D,H,W,G,Cg) → pad Cg → (N,D,H,W,G*cgp)
    x = x_ndhwc.reshape(n, d, h, w, groups, cg)
    x = torch.nn.functional.pad(x, (0, cgp - cg))
    return x.reshape(n, d, h, w, groups * cgp).contiguous()


_WEIGHT_CACHE = {}


def _evict_weight(key, _ref):
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _WEIGHT_CACHE[key]


def _prep_weight(w, k, kt, kh, kw, wc, cgp):
    """Pack (K, C, T, R, S) → (K, T*R*S*cgp), memoized."""
    anchor = w._base if w._base is not None else w
    key = (w.data_ptr(), cgp)
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype, cgp)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is anchor and ent[2] == stamp:
        return ent[1]
    wsrc = (
        torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cgp - wc))
        if cgp != wc
        else w
    )
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cgp)
    _WEIGHT_CACHE[key] = (
        weakref.ref(anchor, functools.partial(_evict_weight, key)),
        wk,
        stamp,
    )
    return wk


def _pick_tile(npq, k, groups, device, ho, wo):
    """Pick (tile, block_rows) that fills the device and fits Wo."""
    kg = k // groups
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        num_cu = int(get_cu_num()) or 256
    except Exception:
        num_cu = 256

    for tile_m, tile_n, wm, wn, br in _TILE_LADDER:
        cols = tile_m // br
        if cols > wo and wo > 0:
            # Prefer block_cols that don't grossly overshoot Wo.
            continue
        blocks = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        waves = blocks * wm * wn
        if waves >= 2 * num_cu or (tile_m, tile_n) == _TILE_LADDER[-1][:2]:
            return (tile_m, tile_n, wm, wn), br
    return DEFAULT_TILE, DEFAULT_BLOCK_ROWS


def _patch_reuse_ok(st, sh, sw, kh, kw, groups, tile, block_rows, d, h, w, cgp):
    if not (st == sh == sw == 1 and (kh > 1 or kw > 1) and groups == 1):
        return False
    # LDS: 2 patch stages + 2*kh*kw B tiles
    from aiter.ops.flydsl.kernels.conv.conv3d_gfx1250_utils import (
        LDS_PAD,
        make_tile_config_gfx1250,
    )
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes

    cfg = make_tile_config_gfx1250(tile, block_rows)
    pr = block_rows + (kh - 1)
    pc = cfg.block_cols + (kw - 1)
    patch_elems = pr * pc * cfg.lds_k_stride + LDS_PAD
    b_elems = 2 * kh * kw * cfg.lds_b_elems_per_buf
    a_elems = 2 * patch_elems
    bytes_ = 2 * (a_elems + b_elems)
    return bytes_ <= get_lds_capacity_bytes() // 2


def _dispatch(exe, *args, stream=None):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, *getattr(exe, "extra_args", ()), _as_stream(stream))


def run_conv3d_gfx1250(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    stream=None,
    tile=None,
    block_rows=None,
    input_layout="NCDHW",
    output_layout="NCDHW",
    patch_reuse=True,
    **_ignored,
):
    """Run gfx1250 FlyDSL conv3d. Extra kwargs (splitk/wgm/...) are ignored."""
    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape
    groups = int(groups)
    assert c % groups == 0 and k % groups == 0 and wc == c // groups
    st, sh, sw = _as_tuple3(stride, "stride")
    dt, dh, dw = _as_tuple3(dilation, "dilation")

    # Resolve padding → (lo, hi) per axis.
    if isinstance(padding, str):
        if padding == "valid":
            pad_lo = pad_hi = (0, 0, 0)
        elif padding == "same":
            assert st == sh == sw == 1, "padding='same' requires unit stride"
            total = (dt * (kt - 1), dh * (kh - 1), dw * (kw - 1))
            pad_lo = tuple(t // 2 for t in total)
            pad_hi = tuple(t - t // 2 for t in total)
        else:
            raise ValueError(f"bad padding string {padding!r}")
    else:
        p = _as_tuple3(padding, "padding")
        pad_lo = pad_hi = p

    torch_pad_mode = {
        "zeros": "zeros",
        "reflect": "reflect",
        "replicate": "replicate",
        "circular": "circular",
    }[padding_mode]

    x_ndhwc = _to_ndhwc(x, in_ndhwc)
    x_ndhwc = _host_pad_ndhwc(x_ndhwc, pad_lo, pad_hi, torch_pad_mode)
    # After pad, kernel padding is 0; refresh spatial extents.
    n, d, h, w, c = x_ndhwc.shape  # NDHWC
    cgp = _pad_cgp(c // groups)
    x_ndhwc = _pad_channels_ndhwc(x_ndhwc, groups, cgp)
    c_padded = groups * cgp

    do = out_extent(d, 0, dt, kt, st)
    ho = out_extent(h, 0, dh, kh, sh)
    wo = out_extent(w, 0, dw, kw, sw)
    assert min(do, ho, wo) >= 1
    npq = n * do * ho * wo

    if n == 0:
        empty = (0, do, ho, wo, k) if out_ndhwc else (0, k, do, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    w_packed = _prep_weight(weight, k, kt, kh, kw, wc, cgp)
    has_bias = bias is not None
    bias_arg = (
        bias.to(torch.float32).contiguous()
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )

    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_br = (
            max(1, int(block_rows))
            if block_rows is not None
            else DEFAULT_BLOCK_ROWS
        )
        if chosen_tile[0] % chosen_br:
            # Fall back to a divisor of tile_m that keeps cols % 16 == 0.
            for br in (8, 4, 2, 1):
                if chosen_tile[0] % br == 0 and (chosen_tile[0] // br) % 16 == 0:
                    chosen_br = br
                    break
    else:
        chosen_tile, chosen_br = _pick_tile(npq, k, groups, x.device, ho, wo)
        if block_rows is not None:
            chosen_br = max(1, int(block_rows))

    use_patch = bool(patch_reuse) and _patch_reuse_ok(
        st, sh, sw, kh, kw, groups, chosen_tile, chosen_br, d, h, w, cgp
    )

    out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
    y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)

    param = make_conv3d_implicit_param_gfx1250(
        n,
        c_padded,
        d,
        h,
        w,
        k,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        dt,
        dh,
        dw,
        has_bias,
        chosen_tile,
        groups,
        out_ndhwc,
        chosen_br,
        num_buffers=2,
        patch_reuse=use_patch,
    )
    exe = compile_conv3d_implicit_gfx1250(param)
    launch_stream = torch.cuda.current_stream() if stream is None else stream
    _dispatch(exe, y, x_ndhwc, w_packed, bias_arg, stream=launch_stream)
    return y
