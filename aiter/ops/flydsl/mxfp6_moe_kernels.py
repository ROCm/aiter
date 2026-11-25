# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch wrappers of the FlyDSL MXFP6 MoE kernels (gfx950): router, row quantizer,
the two grouped GEMM stages and the combine.

Every wrapper takes contiguous tensors on one device and launches on that device's
current stream, with no host sync. The compiled launchers are cached per argument set
and device index.
"""

from __future__ import annotations

import functools

import torch

from .kernels import mxfp6_moe_aux as _aux
from .kernels import mxfp6_moe_gemm as _gemm
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg
from .mxfp6_moe_policy import H_LAYOUTS, Mxfp6StageConfig, route_tiles
from .mxfp6_moe_quant import FORMATS, kq

# entries (tokens x top-k) above which the router runs multi-block (histogram / scan /
# scatter): one block has the fewest launches, multi-block is ~5x faster at 128k
ROUTE_ONE_BLOCK_MAX = 32768


def _device_index(t: torch.Tensor) -> int:
    """Device index for the launcher caches (a compiled handle does not launch on
    another device)."""
    index = t.device.index
    return index if index is not None else torch.cuda.current_device()


@functools.cache
def _get_compiled_quant_rows(K, fmt, dev):
    return _aux.compile_mxfp6_moe_quant_rows(K, fmt)


@functools.cache
def _get_compiled_combine(H, K, dev):
    return _aux.compile_mxfp6_moe_combine(H, K)


@functools.cache
def _get_compiled_route(E, K, BM, dev):
    return _aux.compile_mxfp6_moe_route(E, K, BM)


@functools.cache
def _get_compiled_route_mb(E, K, BM, dev):
    return _aux.compile_mxfp6_moe_route_mb(E, K, BM)


@functools.cache
def _get_compiled_pad_rows(dev):
    return _gemm.compile_mxfp6_moe_pad_rows()


@functools.cache
def _get_compiled_gemm(
    N,
    K,
    a_fmt,
    b_fmt,
    epi,
    gather,
    FM,
    FN,
    NSTAGE,
    GM,
    NT_WEIGHTS,
    H_TILE_STEP,
    PAD_A,
    WARM,
    BN_SLAB,
    HALF_TILES,
    dev,
):
    return _gemm.compile_mxfp6_moe_gemm(
        N,
        K,
        a_fmt,
        b_fmt,
        epi,
        gather,
        FM,
        FN,
        NSTAGE=NSTAGE,
        GM=GM,
        NT_WEIGHTS=NT_WEIGHTS,
        H_TILE_STEP=H_TILE_STEP,
        PAD_A=PAD_A,
        WARM=WARM,
        BN_SLAB=BN_SLAB,
        HALF_TILES=HALF_TILES,
    )


def flydsl_mxfp6_moe_quant_rows(x: torch.Tensor, fmt: str):
    """x [R, K] bf16 -> ("A" rows [R, K * bits / 8] uint8 in MFMA K order, scales
    [R, KQ, 4] int32), MX `fmt` with numerics = mxfp6_moe_quant.quant_mx."""
    R, K = x.shape
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or K % 128:
        raise ValueError(
            f"x must be contiguous bf16 [R, K] with K a multiple of 128, got "
            f"{x.dtype} {tuple(x.shape)}"
        )
    y = torch.empty(R, K * FORMATS[fmt][3] // 8, dtype=torch.uint8, device=x.device)
    s = torch.empty(R, kq(K), 4, dtype=torch.int32, device=x.device)
    _run_compiled(
        _get_compiled_quant_rows(K, fmt, _device_index(x)),
        ptr_arg(x),
        ptr_arg(y),
        ptr_arg(s),
        R,
        current_stream_arg(x.device),
    )
    return y, s


def flydsl_mxfp6_moe_route(ids: torch.Tensor, experts: int, route_m: int):
    """Counting sort of ids [T, K] int32 (the expert of each token's slot) by expert,
    into tiles of route_m rows.

    Returns ridx [n_tiles * route_m] int32 (the token of each routed row; pad slots ->
    row 0), pos [T, K] int32 (the routed row of each entry), texp [n_tiles + 1] int32
    (tile -> expert, -1 for unused, texp[n_tiles] = number of used tiles) and the
    routed rows n_tiles * route_m, with n_tiles = route_tiles(T * K, experts, route_m).
    The slot order within an expert is arbitrary; no output bit of the op depends on
    it."""
    T_, K = ids.shape
    TK = T_ * K
    n_tiles = route_tiles(TK, experts, route_m)
    device, dev = ids.device, _device_index(ids)
    ridx = torch.empty(n_tiles * route_m, dtype=torch.int32, device=device)
    pos = torch.empty(T_, K, dtype=torch.int32, device=device)
    texp = torch.empty(n_tiles + 1, dtype=torch.int32, device=device)
    if TK <= ROUTE_ONE_BLOCK_MAX:
        rank = torch.empty(TK, dtype=torch.int32, device=device)
        _run_compiled(
            _get_compiled_route(experts, K, route_m, dev),
            ptr_arg(ids.contiguous()),
            ptr_arg(ridx),
            ptr_arg(pos),
            ptr_arg(texp),
            ptr_arg(rank),
            TK,
            n_tiles,
            current_stream_arg(device),
        )
    else:
        NB = (TK + _aux.RCH - 1) // _aux.RCH
        ws = torch.empty(
            2 * NB * experts + experts + 1, dtype=torch.int32, device=device
        )
        cnt = ws[: NB * experts]
        boff = ws[NB * experts : 2 * NB * experts]
        tst = ws[2 * NB * experts :]
        _run_compiled(
            _get_compiled_route_mb(experts, K, route_m, dev),
            ptr_arg(ids.contiguous()),
            ptr_arg(cnt),
            ptr_arg(boff),
            ptr_arg(tst),
            ptr_arg(ridx),
            ptr_arg(pos),
            ptr_arg(texp),
            TK,
            NB,
            n_tiles,
            current_stream_arg(device),
        )
    return ridx, pos, texp, n_tiles * route_m


def flydsl_mxfp6_moe_combine(
    o: torch.Tensor, pos: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """o [M, H] bf16, pos [T, K] int32 rows of o, w [T, K] fp32 -> y [T, H] bf16,
    y[t] = sum_j w[t, j] * o[pos[t, j]] in fixed j order (fp32)."""
    T_, K = w.shape
    H = o.shape[1]
    y = torch.empty(T_, H, dtype=torch.bfloat16, device=o.device)
    _run_compiled(
        _get_compiled_combine(H, K, _device_index(o)),
        ptr_arg(o),
        ptr_arg(pos),
        ptr_arg(w.contiguous()),
        ptr_arg(y),
        T_,
        current_stream_arg(o.device),
    )
    return y


def _pad_rows(a_rows, K, dev, stream):
    """The gathered 6-bit rows re-laid to 128 B per K step ([T][KS][128], 96 B used, pad
    bytes never read), head / tail order, by the pad-rows kernel; by torch where its
    32-bit offsets would wrap (>= 2 GiB); nothing in a compile-only dry run."""
    # rows by bytes: any contiguous shape, as the kernel reads them
    T_, KS_ = a_rows.nbytes // (K * 3 // 4), K // 128
    # 8-B elements: a wide copy
    a_pad = torch.empty((T_, KS_, 16), dtype=torch.int64, device=a_rows.device)
    nch = T_ * KS_ * 8  # 16-B output chunks
    if 0 < nch and nch * 16 < 2**31:
        _run_compiled(
            _get_compiled_pad_rows(dev), ptr_arg(a_rows), ptr_arg(a_pad), nch, stream
        )
    elif type(a_rows).__name__ != "FakeTensor":
        src12 = a_rows.view(torch.int64).reshape(T_, KS_, 12)
        a_pad[:, :, :12].copy_(src12[:, :, [0, 1, 3, 4, 6, 7, 9, 10, 2, 5, 8, 11]])
    return a_pad


def _launch_gemm(
    a_rows,
    a_scales,
    ridx,
    texp,
    b_frags,
    b_scales,
    out,
    out_scales,
    *,
    N,
    K,
    a_fmt,
    b_fmt,
    stage1,
    config,
    h_layout,
):
    """Launch one grouped GEMM: stage 1 (epilogue "swiglu_q" on gathered rows) or
    stage 2 ("store" on the routed rows). Raises ValueError before anything is
    launched; see the stage wrappers."""
    if h_layout not in H_LAYOUTS:
        raise ValueError(f"h_layout {h_layout!r}: use 'tile_step' or 'rows'")
    if config.tile_m % 32 or config.tile_n % 32:
        raise ValueError(
            f"tile {config.tile_m}x{config.tile_n}: rows and columns must be "
            "multiples of 32 (2 x 2 waves of 16x16 fragments)"
        )
    E, m_tiles = b_frags.shape[0], texp.numel() - 1
    bits = FORMATS[a_fmt][3]
    # weights are prepared N-block-major in slabs of bns columns
    # (mxfp6_moe_policy.stage_bn); the tile must cover whole slabs, else it reads the
    # wrong fragments
    bns = 16 * b_frags.shape[3] if b_frags.dim() == 5 else 0
    if (
        not bns
        or b_frags.shape[1] * bns != N
        or b_frags.shape[2] != K // 128
        or config.tile_n % bns
    ):
        raise ValueError(
            "b_frags must be [E][N/BN][K/128][BN/16][1536] with BN dividing the "
            f"{config.tile_n}-wide tile (N {N}, K {K}), got {tuple(b_frags.shape)}"
        )
    # A rows and one expert's weights are addressed by 32-bit buffer offsets (descriptor
    # range 2^32 - 1, grouped-DMA bases up to 3 KB lower): past that they wrap and read
    # wrong rows, so refuse
    if a_rows.nbytes + 4096 > 2**32 or b_frags.nbytes // E + 4096 > 2**32:
        raise ValueError(
            f"A rows ({a_rows.nbytes} B) and one expert's weights "
            f"({b_frags.nbytes // E} B) must stay below 4 GiB - 4 KiB: the kernel "
            "addresses them with 32-bit buffer offsets"
        )
    # every tile of the grid is written whole
    rows, n_out = m_tiles * config.route_m, N // 2 if stage1 else N
    row_bytes = n_out * bits // 8 if stage1 else n_out * 2
    if out.nbytes < rows * row_bytes or (
        stage1 and (out_scales is None or out_scales.nbytes < rows * 16 * kq(n_out))
    ):
        raise ValueError(
            f"out must hold {rows} rows of {row_bytes} B ({out.nbytes} B given)"
            + (f", out_scales {rows} x [{kq(n_out)}][4] int32" if stage1 else "")
        )
    dev = _device_index(out)
    stream = current_stream_arg(out.device)
    pad = config.pad_rows
    exe = _get_compiled_gemm(
        N,
        K,
        a_fmt,
        b_fmt,
        "swiglu_q" if stage1 else "store",
        stage1,
        config.tile_m // 32,
        config.tile_n // 32,
        config.ring,
        config.group_m,
        int(config.nt_weights),
        # a 6-bit h is tile-step-major where h_layout says so: written by stage 1,
        # read by stage 2
        int(bits == 6 and h_layout == "tile_step"),
        int(pad),
        config.warm,
        bns,
        int(config.half_tiles),
        dev,
    )
    if pad:
        a_rows = _pad_rows(a_rows, K, dev, stream)
    try:
        _run_compiled(
            exe,
            ptr_arg(a_rows),
            ptr_arg(a_scales),
            ptr_arg(ridx),
            ptr_arg(texp),
            ptr_arg(b_frags),
            ptr_arg(b_scales),
            ptr_arg(out),
            ptr_arg(out_scales if out_scales is not None else out),
            m_tiles,
            stream,
        )
    except RuntimeError as e:
        if "amdgpu-mfma-vgpr-form" not in str(e):
            raise
        raise RuntimeError(
            _gemm.AGPR_FORM_ERROR.format(f"this FlyDSL's LLVM rejects the option ({e})")
        ) from e


def flydsl_mxfp6_moe_stage1(
    *,
    x_rows: torch.Tensor,
    x_scales: torch.Tensor,
    ridx: torch.Tensor,
    texp: torch.Tensor,
    w1_frags: torch.Tensor,
    w1_scales: torch.Tensor,
    h_rows: torch.Tensor,
    h_scales: torch.Tensor,
    model_dim: int,
    inter_dim: int,
    a_fmt: str,
    w_fmt: str,
    config: Mxfp6StageConfig,
    h_layout: str = "tile_step",
):
    """Stage 1: h = SwiGLU(x[ridx] @ W1^T) per routed row, MX-quantized to a_fmt in
    the epilogue, into h_rows / h_scales in stage 2's activation layout.

    x_rows / x_scales: the quantized token rows (flydsl_mxfp6_moe_quant_rows). ridx,
    texp: the router's ([m_tiles * route_m], [m_tiles + 1], flydsl_mxfp6_moe_route).
    w1_frags [E][2I/BN][H/128][BN/16][1536] uint8 and w1_scales [E][2I/16][KQ][64]
    int32: prepared weights, gate / up fragments interleaved
    (mxfp6_moe_quant.preshuffle_b). h_rows: [>= m_tiles * route_m rows][I * bits / 8]
    uint8, h_scales: [same rows][KQ(I)][4] int32; every tile of the grid is written
    whole.

    h_layout: the layout a 6-bit h is written in; stage 2 must be given the same one
    (the tensors keep no record of it). 8-bit h has one layout.

    Raises ValueError, before anything is launched, for: an unknown h_layout; w1_frags
    not in the layout above with BN dividing config.tile_n; x rows or one expert's
    weights of 4 GiB - 4 KiB or more (32-bit buffer offsets); h_rows or h_scales
    smaller than the grid; a tile that does not fit the LDS."""
    _launch_gemm(
        x_rows,
        x_scales,
        ridx,
        texp,
        w1_frags,
        w1_scales,
        h_rows,
        h_scales,
        N=2 * inter_dim,
        K=model_dim,
        a_fmt=a_fmt,
        b_fmt=w_fmt,
        stage1=True,
        config=config,
        h_layout=h_layout,
    )


def flydsl_mxfp6_moe_stage2(
    *,
    h_rows: torch.Tensor,
    h_scales: torch.Tensor,
    ridx: torch.Tensor,
    texp: torch.Tensor,
    w2_frags: torch.Tensor,
    w2_scales: torch.Tensor,
    out: torch.Tensor,
    model_dim: int,
    inter_dim: int,
    a_fmt: str,
    w_fmt: str,
    config: Mxfp6StageConfig,
    h_layout: str = "tile_step",
):
    """Stage 2: out = h @ W2^T per routed row, bf16 [>= m_tiles * route_m rows][H].

    h_rows / h_scales: stage 1's output, read in the h_layout stage 1 wrote it in.
    w2_frags [E][H/BN][I/128][BN/16][1536] uint8 and w2_scales [E][H/16][KQ(I)][64]
    int32: prepared weights. ridx is not read (the rows are already routed). Raises
    ValueError as flydsl_mxfp6_moe_stage1 does."""
    _launch_gemm(
        h_rows,
        h_scales,
        ridx,
        texp,
        w2_frags,
        w2_scales,
        out,
        None,
        N=model_dim,
        K=inter_dim,
        a_fmt=a_fmt,
        b_fmt=w_fmt,
        stage1=False,
        config=config,
        h_layout=h_layout,
    )
