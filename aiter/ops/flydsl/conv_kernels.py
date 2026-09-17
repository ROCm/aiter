# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Host entry point for the FlyDSL implicit-GEMM convolution.

Everything between a caller's torch tensors and the compiled kernel: the
keyword surface and its validation, layout and padding normalisation, the
weight repack cache, the split-K decision, and the rank dispatch that sends a
call to the 1-D, 2-D or 3-D path. Nothing here emits DSL.

Split out of ``kernels/conv3d_implicit_gfx950.py`` so that module is the kernel and
its compile, the way ``gemm_kernels.py`` sits in front of
``kernels/gemm_a16w16_gfx950.py``.

x: (N, C, D, H, W) bf16 NCDHW by default, weight: (K, C/groups, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16 by default. ``input_layout`` / ``output_layout`` select
NCDHW or NDHWC independently; the GEMM itself is channels-last, so NDHWC input skips the
pre-transpose and NDHWC output is the raw row-major (npq, K) the epilogue produces.
Supports stride, padding (int, per-axis tuple, or torch's "same" / "valid"),
padding_mode, dilation, bias, groups, and split-K.
"""

import functools
import weakref

import torch

from .conv3d_tuned_config import _lookup_tuned_tile, _num_cu, _pick_tile, _pick_wgm
from .kernels.conv3d_gfx950_utils import _as_stream
from .kernels.conv3d_implicit_gfx950 import (
    DEFAULT_TILE,
    LDG_VEC,
    PADDING_MODES,
    SPLITK_MAX_STAGING_BYTES,
    TILE_K,
    compile_conv3d_implicit,
)
from .kernels.conv3d_transpose import (
    TR_MAX_BIG_S,
    TR_VEC,
    compile_transpose_ncdhw_ndhwc,
)


def _dispatch(exe, *args, stream=None):
    """Run a builder's launcher, pre-compiling on first use."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, _as_stream(stream))


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    big = n * c * s > 0x7FFFFFFF
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % TR_VEC == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    if big and s > TR_MAX_BIG_S:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    _dispatch(
        exe, out, x, stream=torch.cuda.current_stream() if stream is None else stream
    )
    return out


_WEIGHT_CACHE = {}


def _pad_channels(c):
    return (c + LDG_VEC - 1) // LDG_VEC * LDG_VEC


LAYOUTS = {
    3: ("NCDHW", "NDHWC"),
    2: ("NCHW", "NHWC"),
    1: ("NCW", "NWC"),
}


def _check_layouts(rank, input_layout, output_layout):
    names = LAYOUTS[rank]
    for what, v in (("input_layout", input_layout), ("output_layout", output_layout)):
        assert v in names, f"{what} must be one of {names}, got {v!r}"


def _shape_ncdhw(x, ndhwc):
    """Unpack a 5-D input in either layout to (n, c, d, h, w)."""
    if ndhwc:
        n, d, h, w, c = x.shape
    else:
        n, c, d, h, w = x.shape
    return n, c, d, h, w


def _pad_spatial(x, ndhwc, pads, mode="constant"):
    """Pad (D, H, W) with torch's (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi) ordering."""
    if mode == "constant":
        return torch.nn.functional.pad(x, ((0, 0) + pads) if ndhwc else pads), ndhwc
    if ndhwc:
        x = x.permute(0, 4, 1, 2, 3)
    return torch.nn.functional.pad(x, pads, mode=mode), False


def _big_in(n, c, groups, d, h, w, pt, ph, pw):
    """Whether the kernel's 64-bit BIG_IN address path would engage for this input."""
    cp = _pad_channels(c // groups) * groups
    return n * cp * (d + 2 * pt) * (h + 2 * ph) * (w + 2 * pw) > 0x7FFFFFFF


def _evict_weight(key, _ref):
    """weakref callback: drop the entry the dead weight was pinning."""
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _WEIGHT_CACHE[key]


def _prep_weight(w, k, kt, kh, kw, c):
    """Pack (K, C, T, R, S) -> (K, T*R*S*Cpad), memoized on the source weight."""
    anchor = w._base if w._base is not None else w
    key = w.data_ptr()
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is anchor and ent[2] == stamp:
        return ent[1]
    cp = _pad_channels(c)
    wsrc = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cp - c)) if cp != c else w
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cp)
    _WEIGHT_CACHE[key] = (
        weakref.ref(anchor, functools.partial(_evict_weight, key)),
        wk,
        stamp,
    )
    return wk


def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE, groups=1):
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if npq * k * 4 > SPLITK_MAX_STAGING_BYTES:
        return 1
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        kg = k // groups
        base = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        if (
            npq < 4096
            or k_tiles < 16
            or kg % tile_n != 0
            or npq % tile_m != 0
            or crs % TILE_K != 0
            or npq * k * 4 > 0x7FFFFFFF
        ):
            sk = 1
        else:
            num_cu = _num_cu(device)
            if base >= (3 * num_cu) // 4:
                sk = 1
            else:
                sk = min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def _as_tuple(v, rank, name):
    if isinstance(v, int):
        return (v,) * rank
    t = tuple(v)
    if len(t) == 1:
        return t * rank
    assert (
        len(t) == rank
    ), f"{name} must be an int or a sequence of 1 or {rank} ints, got {tuple(v)}"
    return t


def _resolve_padding(padding, kernel, stride, dilation):
    """Normalize torch's ``padding`` argument to a (low, high) pair of per-axis triples."""
    if not isinstance(padding, str):
        p = _as_tuple(padding, 3, "padding")
        assert min(p) >= 0, f"negative padding is not supported, got (pt, ph, pw) = {p}"
        return p, p
    if padding == "valid":
        return (0, 0, 0), (0, 0, 0)
    if padding != "same":
        raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
    assert all(
        s == 1 for s in stride
    ), f"padding='same' is not supported for strided convolutions, got stride {tuple(stride)}"
    total = [dl * (kn - 1) for kn, dl in zip(kernel, dilation)]
    return tuple(t // 2 for t in total), tuple(t - t // 2 for t in total)


def _conv3d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    splitk=None,
    stream=None,
    tile=None,
    wgm=None,
    input_layout="NCDHW",
    output_layout="NCDHW",
):
    _check_layouts(3, input_layout, output_layout)

    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape

    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        assert (
            t is None or t.is_cuda
        ), f"flydsl_conv_implicit needs GPU tensors; {name} is on {t.device}"
    assert (
        x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), f"flydsl_conv_implicit is a bf16-only kernel; got x={x.dtype}, weight={weight.dtype}"
    assert bias is None or (bias.dim() == 1 and bias.numel() == k), (
        f"bias must be a 1-D tensor of {k} elements, one per output channel; "
        f"got shape {tuple(bias.shape)}"
    )
    groups = int(groups)
    assert groups >= 1, f"groups must be >= 1, got {groups}"
    assert c % groups == 0, f"in-channels {c} not divisible by groups {groups}"
    assert k % groups == 0, f"out-channels {k} not divisible by groups {groups}"
    assert wc == c // groups, f"weight in-channels {wc} != C/groups = {c // groups}"
    st, sh, sw = _as_tuple(stride, 3, "stride")

    assert (
        min(st, sh, sw) >= 1
    ), f"non-positive stride is not supported, got (st, sh, sw) = {(st, sh, sw)}"
    dt, dh, dw = _as_tuple(dilation, 3, "dilation")
    assert min(dt, dh, dw) >= 1, f"dilation must be >= 1, got {(dt, dh, dw)}"
    pad_lo, pad_hi = _resolve_padding(padding, (kt, kh, kw), (st, sh, sw), (dt, dh, dw))
    pt, ph, pw = pad_lo
    assert (
        padding_mode in PADDING_MODES
    ), f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"

    if padding_mode in ("reflect", "circular"):
        for ax, (p, ext) in enumerate(zip(map(max, pad_lo, pad_hi), (d, h, w))):
            if padding_mode == "reflect":
                assert (
                    p < ext
                ), f"reflect padding {p} must be < input extent {ext} on spatial axis {ax}"
            else:
                assert (
                    p <= ext
                ), f"circular padding {p} must be <= input extent {ext} on spatial axis {ax}"

    # Key into the offline-tuned config table. Captured here, before the padding
    # and channel-padding paths below rewrite n/c/d/h/w, so that it describes the
    # problem the caller asked for and matches the untuned CSV column order.
    # Asymmetric padding cannot be expressed with one value per axis, so those
    # calls fall through to the heuristic rather than matching a wrong row.
    tuned_key = (
        (
            n,
            c,
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
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            groups,
            bias is not None,
        )
        if padding_mode == "zeros" and pad_lo == pad_hi
        else None
    )

    if pad_lo != pad_hi:
        if padding_mode == "zeros":
            x, in_ndhwc = _pad_spatial(
                x, in_ndhwc, (0, pad_hi[2] - pw, 0, pad_hi[1] - ph, 0, pad_hi[0] - pt)
            )
        else:
            x, in_ndhwc = _pad_spatial(
                x,
                in_ndhwc,
                (pw, pad_hi[2], ph, pad_hi[1], pt, pad_hi[0]),
                mode=padding_mode,
            )
            pt = ph = pw = 0
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)

    inline_pad = padding_mode != "zeros" and bool(pt or ph or pw)
    if inline_pad and _big_in(n, c, groups, d, h, w, pt, ph, pw):
        x, in_ndhwc = _pad_spatial(
            x, in_ndhwc, (pw, pw, ph, ph, pt, pt), mode=padding_mode
        )
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
        pt = ph = pw = 0
        inline_pad = False
    pad_mode = padding_mode if inline_pad else "zeros"

    if (
        groups == 1
        and kt == 1
        and kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and pt == 0
        and ph == 0
        and pw == 0
    ):
        wm = weight.reshape(k, c)
        if in_ndhwc:
            y = torch.matmul(x.reshape(n * d * h * w, c), wm.t()).reshape(n, d, h, w, k)
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y if out_ndhwc else y.permute(0, 4, 1, 2, 3).contiguous()
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y.permute(0, 2, 3, 4, 1).contiguous() if out_ndhwc else y

    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    assert (
        min(do, ho, wo) >= 1
    ), f"dilated filter is larger than the padded input: output ({do}, {ho}, {wo})"
    npq = n * do * ho * wo

    if n == 0:
        empty = (0, do, ho, wo, k) if out_ndhwc else (0, k, do, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    cg = c // groups
    cgp = _pad_channels(cg)
    if cgp != cg:
        if in_ndhwc:
            x = torch.nn.functional.pad(
                x.reshape(n, d, h, w, groups, cg), (0, cgp - cg)
            )
            x = x.reshape(n, d, h, w, groups * cgp)
        else:
            x = torch.nn.functional.pad(
                x.reshape(n, groups, cg, d, h, w), (0, 0, 0, 0, 0, 0, 0, cgp - cg)
            )
            x = x.reshape(n, groups * cgp, d, h, w)
    c = groups * cgp
    crs = cgp * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = (
        bias.to(torch.float32).contiguous()
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )

    x_ndhwc = x.contiguous() if in_ndhwc else _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, wc)

    def _run(the_tile, the_wgm=1):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile, groups)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
            y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit(
            n,
            c,
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
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            pad_mode,
            has_bias,
            sk,
            the_tile,
            the_wgm,
            groups,
            out_ndhwc,
        )
        _dispatch(exe, y, x_ndhwc, w_packed, bias_arg, stream=launch_stream)
        return y, sk

    forced_wgm = None if wgm is None else max(1, int(wgm))
    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_wgm = 1 if forced_wgm is None else forced_wgm
    else:
        hit = _lookup_tuned_tile(tuned_key, x.device)
        if hit is not None:
            chosen_tile, chosen_wgm = hit
            if forced_wgm is not None:
                chosen_wgm = forced_wgm
        else:
            chosen_tile = _pick_tile(npq, k, groups, x.device)
            chosen_wgm = (
                _pick_wgm(npq, k, groups, chosen_tile, x.device)
                if forced_wgm is None
                else forced_wgm
            )

    y, sk = _run(chosen_tile, chosen_wgm)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        if out_ndhwc:
            return y.view(n, do, ho, wo, k).to(torch.bfloat16)
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y


def _conv2d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    input_layout="NCHW",
    output_layout="NCHW",
    **kwargs,
):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    _check_layouts(2, input_layout, output_layout)
    sh, sw = _as_tuple(stride, 2, "stride")
    dh, dw = _as_tuple(dilation, 2, "dilation")

    if isinstance(padding, str):
        p3 = padding
    else:
        ph, pw = _as_tuple(padding, 2, "padding")
        p3 = (0, ph, pw)
    k, wc, r, s = weight.shape

    if input_layout == "NHWC":
        n, h, w, c = x.shape
        x5, in5 = x.reshape(n, 1, h, w, c), "NDHWC"
    else:
        n, c, h, w = x.shape
        x5, in5 = x.reshape(n, c, 1, h, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NHWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, sh, sw),
        padding=p3,
        dilation=(1, dh, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NHWC":
        return y5.reshape(y5.shape[0], y5.shape[2], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    input_layout="NCW",
    output_layout="NCW",
    **kwargs,
):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    _check_layouts(1, input_layout, output_layout)
    (sw,) = _as_tuple(stride, 1, "stride")
    (dw,) = _as_tuple(dilation, 1, "dilation")
    if isinstance(padding, str):
        p3 = padding
    else:
        p3 = (0, 0, _as_tuple(padding, 1, "padding")[0])
    k, wc, s = weight.shape
    if input_layout == "NWC":
        n, w, c = x.shape
        x5, in5 = x.reshape(n, 1, 1, w, c), "NDHWC"
    else:
        n, c, w = x.shape
        x5, in5 = x.reshape(n, c, 1, 1, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, 1, sw),
        padding=p3,
        dilation=(1, 1, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NWC":
        return y5.reshape(y5.shape[0], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def flydsl_conv_implicit(
    x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs
):
    """Main implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S).

    ``input_layout`` and ``output_layout`` are independent and named per rank:
    "NCDHW"/"NDHWC", "NCHW"/"NHWC", "NCW"/"NWC". The weight stays KC*, and the batch axis
    leads in both, so an unbatched input works either way. Channels-last is the kernel's
    own layout on both sides: an NDHWC input skips the pre-transpose, and an NDHWC output
    is the (npq, K) index space the GEMM already writes, so it also skips the split-K
    epilogue's transpose. Channels-last output does give up the vectorized store on the
    ``n == 1`` fast path, since a lane's four accumulator values are four M rows and those
    are K apart once channels are innermost.

    ``padding`` takes an int, a per-axis tuple, or one of torch's two strings. "valid" is
    no padding. "same" pads so the output keeps the input's spatial extent, which needs
    ``dilation * (kernel - 1)`` elements per axis and, like torch, is only defined at
    stride 1. That total is normally even and costs nothing beyond an ordinary symmetric
    pad. An even-length filter under odd dilation makes it odd, and torch's rule of
    putting the extra element on the high side then asks for a pad the kernel cannot
    express with one value per axis; that case materializes a padded input first, exactly
    as torch does (it warns about the same copy). ``padding_mode`` applies to "same" too.

    ``dilation`` follows torch semantics: it spaces the filter taps by that factor
    over the input, shrinking the output to
    ``(D + 2*pad - dilation*(T-1) - 1)//stride + 1`` per axis. It costs nothing in the
    GEMM -- the K axis is still C/groups*T*R*S -- it only stretches the im2col gather,
    so a dilated filter reads a wider input footprint per output row and gets less
    reuse out of cache than the same filter undilated.

    ``groups`` follows torch semantics: C and K must both be divisible by it and the
    weight's channel dim is C/groups. Groups map onto the N grid axis, one tile never
    spanning two groups, so efficiency tracks how well K/groups fills TILE_N. Measured
    on gfx950 vs torch/MIOpen, moderate cardinality wins across the board (1.5-2.0x for
    K/groups in [8, 256]). True depthwise (groups == C, so C/groups == 1) is the one
    weak case at ~0.5x: C/groups=1 pads to the gather's 8-wide vector, wasting 7/8 of
    the K axis, while K/groups=1 leaves all but one column of the N tile masked.
    Narrower tiles recover little there -- depthwise wants its own kernel, not this
    single-GEMM mapping.
    """
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError(
            f"flydsl_conv_implicit supports 1D/2D/3D; got filter rank {weight.dim()}"
        )
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    impl = {3: _conv3d_impl, 2: _conv2d_impl, 1: _conv1d_impl}[spatial_rank]
    y = impl(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        **kwargs,
    )
    return y.squeeze(0) if unbatched else y
