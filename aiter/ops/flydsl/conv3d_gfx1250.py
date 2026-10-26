# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host entry for gfx1250 FlyDSL implicit-GEMM conv3d.

Pre-pads (all padding_modes), converts to NDHWC, pads C/groups to 32, packs
weights, picks a tile (tuned table or heuristic), and launches
``compile_conv3d_implicit_gfx1250``.
"""

from __future__ import annotations

import functools
import os
import weakref

import torch

from aiter.ops.flydsl.kernels.conv.conv3d_gfx950_utils import _as_stream
from aiter.ops.flydsl.kernels.conv.conv3d_gfx1250_utils import (
    BF16_BYTES,
    CGP_ALIGN,
    DEFAULT_BLOCK_ROWS,
    DEFAULT_TILE,
    MAX_WAVES,
    out_extent,
)
from aiter.ops.flydsl.kernels.conv.conv3d_implicit_gfx1250 import (
    compile_conv3d_implicit_gfx1250,
    make_conv3d_implicit_param_gfx1250,
)

# Heuristic ladder (tile_m, tile_n, m_warp, n_warp, block_rows).
_TILE_LADDER = (
    (128, 128, 2, 4, 8),
    (128, 128, 2, 2, 8),
    (128, 64, 2, 2, 8),
    (64, 64, 2, 2, 4),
    (64, 32, 2, 1, 4),
    (32, 32, 1, 1, 2),
)

AITER_CONV3D_DYN_HW = os.environ.get("AITER_CONV3D_DYN_HW", "1") not in (
    "0",
    "false",
    "False",
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
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _host_pad_ndhwc(x_ndhwc, pad_lo, pad_hi, mode):
    """Pad NDHWC on D/H/W. ``mode`` is torch pad mode (constant/reflect/replicate/circular)."""
    pt, ph, pw = pad_lo
    pt2, ph2, pw2 = pad_hi
    if not (pt or ph or pw or pt2 or ph2 or pw2):
        return x_ndhwc
    pads = (0, 0, pw, pw2, ph, ph2, pt, pt2)
    if mode == "zeros":
        return torch.nn.functional.pad(x_ndhwc, pads, mode="constant", value=0)
    x = x_ndhwc.permute(0, 4, 1, 2, 3)
    x = torch.nn.functional.pad(x, (pw, pw2, ph, ph2, pt, pt2), mode=mode)
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _pad_channels_ndhwc(x_ndhwc, groups, cgp):
    n, d, h, w, c = x_ndhwc.shape
    cg = c // groups
    if cgp == cg:
        return x_ndhwc
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
        torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cgp - wc)) if cgp != wc else w
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
    except Exception:  # noqa: BLE001 -- no rocminfo (CPU-only host)
        num_cu = 256

    for tile_m, tile_n, wm, wn, br in _TILE_LADDER:
        cols = tile_m // br
        if cols > wo > 0:
            continue
        blocks = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        waves = blocks * wm * wn
        if waves >= 2 * num_cu or (tile_m, tile_n) == _TILE_LADDER[-1][:2]:
            return (tile_m, tile_n, wm, wn), br
    return DEFAULT_TILE, DEFAULT_BLOCK_ROWS


_DROPPED_TILES = set()


def _warn_dropped_tile(tile):
    """Warn once per tile that a tuned row was ignored for exceeding MAX_WAVES."""
    if tile in _DROPPED_TILES:
        return
    _DROPPED_TILES.add(tile)
    from aiter import logger

    logger.warning(
        f"conv3d_gfx1250: tuned tile {tile} exceeds MAX_WAVES={MAX_WAVES}; "
        "using the heuristic tile instead. Re-tune this shape."
    )


def _clamp_num_buffers(num_buffers, tiles_per_split, tile, block_rows):
    """Drop to a legal B that fits LDS and leaves a nonempty prologue."""
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes
    from aiter.ops.flydsl.conv3d_policy_gfx1250 import lds_bytes_tap

    b = max(2, min(8, int(num_buffers)))
    lds_cap = get_lds_capacity_bytes()
    while b > 2 and (
        tiles_per_split < b - 1
        or lds_bytes_tap(tile[0], tile[1], num_buffers=b) > lds_cap
    ):
        b -= 1
    if tiles_per_split < b - 1:
        b = max(2, tiles_per_split + 1)
    del block_rows  # reserved for future LDS accounting that needs cfg
    return b


def _dyn_hw_ok_gfx1250(n, c_padded, d, h, w, k, npq, tile, block_rows, ho, wo):
    """Variable-resolution path: env on, no BIG_IN/OUT, no tile tails."""
    if not AITER_CONV3D_DYN_HW:
        return False
    if npq >= (1 << 31):
        return False
    x_bytes = n * c_padded * d * h * w * BF16_BYTES
    y_bytes = npq * k * BF16_BYTES
    if x_bytes > 0x7FFFFFFF or y_bytes > 0x7FFFFFFF:
        return False
    rows = block_rows
    cols = tile[0] // rows
    if ho % rows or wo % cols:
        return False
    kg_ok = True  # n_tail checked at grid build; dyn_hw also needs no n_tail
    return kg_ok


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
    patch_reuse=True,  # gfx950 API parity; ignored on gfx1250 (no patch path)
    splitk=None,
    wgm=None,  # accepted for API parity; unused on gfx1250
    num_buffers=2,
    variant="bandwidth_bound",
    main_loop_unroll=False,
    **_ignored,
):
    """Run gfx1250 FlyDSL conv3d (TDM+WMMA; optional split-K / dyn_hw / tuned tile)."""
    del _ignored, patch_reuse
    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape
    groups = int(groups)
    assert c % groups == 0 and k % groups == 0 and wc == c // groups
    st, sh, sw = _as_tuple3(stride, "stride")
    dt, dh, dw = _as_tuple3(dilation, "dilation")

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

    # Tuned table lookup (gfx key separates gfx950 / gfx1250 rows).
    tuned_splitk = None
    tuned_br = None
    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_br = (
            max(1, int(block_rows)) if block_rows is not None else DEFAULT_BLOCK_ROWS
        )
        if chosen_tile[0] % chosen_br:
            for br in (8, 4, 2, 1):
                if chosen_tile[0] % br == 0 and (chosen_tile[0] // br) % 16 == 0:
                    chosen_br = br
                    break
    else:
        # Tuned CSV keys are the caller's extents (before host pad).
        n0, c0, d0, h0, w0 = _shape_ncdhw(x, in_ndhwc)
        pt, ph, pw = pad_lo
        tuned_key = (
            n0,
            c0,
            d0,
            h0,
            w0,
            k,
            kt,
            kh,
            kw,
            st,
            sh,
            sw,
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            groups,
            has_bias,
        )
        try:
            from aiter.ops.flydsl.conv_kernels import _lookup_tuned_tile

            hit = _lookup_tuned_tile(tuned_key, x.device)
        except Exception:  # noqa: BLE001 -- bad config table must not break a conv
            hit = None
        if hit is not None and hit[0][2] * hit[0][3] > MAX_WAVES:
            _warn_dropped_tile(hit[0])
            hit = None

        if hit is not None:
            chosen_tile = hit[0]
            tuned_splitk = hit[2]
            tuned_br = hit[3] if len(hit) > 3 else None
            chosen_br = (
                max(1, int(block_rows))
                if block_rows is not None
                else (
                    max(1, int(tuned_br))
                    if tuned_br is not None
                    else DEFAULT_BLOCK_ROWS
                )
            )
            if chosen_tile[0] % chosen_br or (chosen_tile[0] // chosen_br) % 16:
                for br in (8, 4, 2, 1):
                    if chosen_tile[0] % br == 0 and (chosen_tile[0] // br) % 16 == 0:
                        chosen_br = br
                        break
        else:
            chosen_tile, chosen_br = _pick_tile(npq, k, groups, x.device, ho, wo)
            if block_rows is not None:
                chosen_br = max(1, int(block_rows))

    from aiter.ops.flydsl.conv_kernels import _resolve_splitk

    crs = cgp * kt * kh * kw
    sk_arg = splitk if splitk is not None else tuned_splitk
    sk = _resolve_splitk(sk_arg, npq, crs, k, x.device, chosen_tile, groups)

    # 2D-block tails cannot combine with split-K / BIG_OUT.
    rows, cols = chosen_br, chosen_tile[0] // chosen_br
    row_chk = ho % rows != 0 or wo % cols != 0
    n_tail = (k // groups) % chosen_tile[1] != 0
    y_bytes = npq * k * BF16_BYTES
    big_out = y_bytes > 0x7FFFFFFF
    if sk > 1 and (row_chk or n_tail or big_out):
        if splitk is not None and int(splitk) > 1:
            raise AssertionError(
                f"explicit splitk={splitk} incompatible with tails/BIG_OUT "
                f"(row_chk={row_chk}, n_tail={n_tail}, big_out={big_out})"
            )
        sk = 1

    dyn_hw = _dyn_hw_ok_gfx1250(
        n, c_padded, d, h, w, k, npq, chosen_tile, chosen_br, ho, wo
    )
    if dyn_hw and n_tail:
        dyn_hw = False

    # Match GEMM: B-1 K-tiles must fill the prologue; clamp against LDS too.
    k_tiles = crs // 32  # TILE_K == CGP_ALIGN == 32
    tiles_per_split = k_tiles // max(1, sk)
    nb = _clamp_num_buffers(num_buffers, tiles_per_split, chosen_tile, chosen_br)

    if sk > 1:
        y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        kernel_bias = False
        bias_kernel = torch.empty(1, device=x.device, dtype=torch.float32)
    else:
        out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
        y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        kernel_bias = has_bias
        bias_kernel = bias_arg

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
        kernel_bias,
        chosen_tile,
        groups,
        out_ndhwc,
        chosen_br,
        num_buffers=nb,
        splitk=sk,
        dyn_hw=dyn_hw,
        variant=variant,
        main_loop_unroll=main_loop_unroll,
    )
    exe = compile_conv3d_implicit_gfx1250(param)
    launch_stream = torch.cuda.current_stream() if stream is None else stream
    _dispatch(exe, y, x_ndhwc, w_packed, bias_kernel, stream=launch_stream)

    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        if out_ndhwc:
            return y.view(n, do, ho, wo, k).to(torch.bfloat16)
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y
