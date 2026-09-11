# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""FlyDSL Flash Attention fp8 (e4m3fn) forward for gfx950.

Q/K/V are pre-quantized e4m3fn with per-tensor fp32 shape-[1] descales; the
output is bf16. Dense, packed-varlen and split-K.
"""

from __future__ import annotations

import functools

import torch

from aiter.ops.flydsl.kernels.fmha_gfx950.op_combine import (
    dualwave_splitk_workspace_elems,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DUALWAVE_SWP_BLOCK_M,
)

__all__ = ["dualwave_splitk_workspace_elems", "flydsl_flash_attn_fp8_func"]

# Largest flat element count the fp8 C-ABI can address; see the split below.
_FP8_MAX_FLAT_ELEMS = 2**31
# Past this KV length the extra P headroom outweighs the ~0.3% it costs.
_FP8_LONG_SEQ = 4096
_DENSE_LIGHT_CU_FALLBACK = 256

_FP8_AUTOSPLIT_MIN_TILES = 16
_FP8_AUTOSPLIT_MAX_WS_BYTES = 1 << 30
_FP8_BLOCK_N = 64
_FP8_AUTOSPLIT_CANDIDATES = tuple(range(1, 17))
_FP8_AUTOSPLIT_FIXED_TILES = 10.4
_FP8_AUTOSPLIT_CAUSAL_SKEW = 0.75
_FP8_AUTOSPLIT_DENSE_MARGIN = 0.85
# Splitting a causal shape in two doubles the workgroup count, so it only pays
# while either the machine still has room for the second wave or the KV loop is
# long enough to amortise the extra round plus the combine kernel. Measured on
# gfx950/256CU (causal self-attention, D=128, min of 8x120), the sp2-vs-sp1
# crossover tracks kv_tiles / (wgs / num_cu): ~0.70 machine-full at 32 KV tiles,
# ~0.80 at 40, ~0.93 at 48, and past 1.0 from 64 tiles up -- i.e. a constant
# ratio of roughly 50 along the whole boundary.
_FP8_AUTOSPLIT_SPLIT2_OCCUPANCY = 50
_FP8_NARROW_MAX_KV_TILES = 48
_FP8_BATCH_INTERLEAVE_GROUP = 2


def _fp8_rescale_threshold(seqlen_kv: int) -> float:
    return 6.0 if seqlen_kv <= _FP8_LONG_SEQ else 4.0


# Device properties and the shape heuristics below are pure functions of a device
# index resp. a handful of ints, but they run on every call; small shapes are
# entirely host-bound, so memoise them.
@functools.lru_cache(maxsize=16)
def _gpu_arch_cached(index: int | None) -> str:
    try:
        return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001
        return ""


@functools.lru_cache(maxsize=16)
def _num_cu_cached(index: int | None) -> int:
    try:
        return int(torch.cuda.get_device_properties(index).multi_processor_count)
    except Exception:  # noqa: BLE001
        return _DENSE_LIGHT_CU_FALLBACK


def _gpu_arch(device: torch.device) -> str:
    return _gpu_arch_cached(device.index)


def _num_cu(device: torch.device) -> int:
    return _num_cu_cached(device.index)


@functools.lru_cache(maxsize=256)
def _fp8_auto_block_m(
    batch: int, num_heads: int, seqlen_q: int, seqlen_kv: int, num_cu: int
) -> int:
    """Pick BLOCK_M (256 wide / 128 narrow) for an fp8 shape."""
    kv_tiles = -(-seqlen_kv // _FP8_BLOCK_N)
    if kv_tiles > _FP8_NARROW_MAX_KV_TILES:
        return DUALWAVE_SWP_BLOCK_M
    narrow = DUALWAVE_SWP_BLOCK_M // 2
    narrow_wgs = num_heads * -(-seqlen_q // narrow) * batch
    return narrow if narrow_wgs <= num_cu else DUALWAVE_SWP_BLOCK_M


@functools.lru_cache(maxsize=256)
def _fp8_batch_interleave_group(
    batch: int, causal: bool, cross: bool, num_kv_splits: int
) -> int:
    if not causal or cross or num_kv_splits > 1:
        return 1
    g = _FP8_BATCH_INTERLEAVE_GROUP
    return g if batch % g == 0 else 1


@functools.lru_cache(maxsize=512)
def _fp8_auto_kv_splits(
    batch: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    causal: bool,
    num_cu: int,
    block_m: int = DUALWAVE_SWP_BLOCK_M,
) -> int:
    """Pick num_kv_splits by minimising `rounds(s) * (FIXED + tiles/s)`.

    ``block_m`` must be the tile `_fp8_auto_block_m` chose; it sets the workgroup count.
    """
    wgs = num_heads * -(-seqlen_q // block_m) * batch
    kv_tiles = -(-seqlen_kv // _FP8_BLOCK_N)

    if not causal:
        kept = 1.0
    elif seqlen_q <= seqlen_kv:
        kept = max(0.0, 1.0 - (seqlen_q - 1) / (2.0 * seqlen_kv))
    else:
        kept = 0.5 * seqlen_kv / seqlen_q
    if kept < _FP8_AUTOSPLIT_CAUSAL_SKEW:
        if kv_tiles // 2 < _FP8_AUTOSPLIT_MIN_TILES or wgs > num_cu:
            return 1
        # The makespan model below rounds the workgroup count up to whole waves
        # and is blind to causal skew, so this branch hard-codes 2 -- but 2 is
        # only right while the shape leaves enough of the machine idle for the
        # split to fill; a shape already at full occupancy just pays for a second
        # round and a combine launch.
        if 2 * wgs <= num_cu:
            return 2
        if kv_tiles * num_cu < _FP8_AUTOSPLIT_SPLIT2_OCCUPANCY * wgs * batch:
            return 1
        return 2

    def makespan(splits: int) -> float:
        n = wgs * splits
        return (-(-n // num_cu)) * (_FP8_AUTOSPLIT_FIXED_TILES + -(-kv_tiles // splits))

    usable = [
        s
        for s in _FP8_AUTOSPLIT_CANDIDATES
        if s == 1 or kv_tiles // s >= _FP8_AUTOSPLIT_MIN_TILES
    ]
    best = min(usable, key=makespan)
    if best == 1:
        return 1
    return best if makespan(best) <= _FP8_AUTOSPLIT_DENSE_MARGIN * makespan(1) else 1


@functools.lru_cache(maxsize=128)
def _build_fp8(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    rescale_threshold: float,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
    head_dim: int = 128,
    head_dim_v: int | None = None,
    varlen: bool = False,
    cross_seqlen: bool = False,
    num_kv_splits: int = 1,
    block_m: int = 256,
    batch_interleave_group: int = 1,
    return_lse: bool = False,
):
    """Build (and cache) the gfx950 fp8 launcher (dense, packed varlen, or split-K)."""
    from aiter.ops.flydsl.kernels.fmha_gfx950.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module,
    )

    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        causal=causal,
        num_kv_heads=num_kv_heads,
        daz=daz,
        rescale_threshold=rescale_threshold,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
        batch_interleave_group=batch_interleave_group,
        return_lse=return_lse,
    )


def flydsl_flash_attn_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    num_kv_heads: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_kv: int | None = None,
    cross_seqlen: bool | None = None,
    num_kv_splits: int | None = None,
    fp8_block_m: int | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    lse: torch.Tensor | None = None,
    daz: bool = True,
    dualwave_swp_lazy_rescale: bool = True,
    dualwave_swp_setprio: bool = True,
    dualwave_swp_enable_stagger: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the gfx950 DUALWAVE_SWP fp8 flash attention forward.

    Args:
        q: Query tensor, ``torch.float8_e4m3fn``.
           Dense: ``[B, Sq, H, D]`` (BSHD). Varlen: ``[total_q, H, D]`` (packed).
        k: Key tensor. Dense: ``[B, Skv, Hkv, D]``. Varlen: ``[total_kv, Hkv, D]``.
        v: Value tensor, same shape as k except the last dim may be ``Dv != D``.
        causal: Bottom-right aligned causal mask when True.
        num_kv_heads: KV head count for GQA/MQA; defaults to k's head count.
        cu_seqlens_q / cu_seqlens_kv: Int32 ``[B+1]`` cumulative token counts (varlen).
        max_seqlen_q: Maximum per-batch Q seqlen. Required in varlen mode.
        max_seqlen_kv: Maximum per-batch KV seqlen. Required for varlen cross-attn.
        cross_seqlen: Whether seqlen_q and seqlen_kv differ. Required in varlen
            mode; dense mode infers it from ``q.shape[1] != k.shape[1]``.
        num_kv_splits: Split-K factor (seq_len >= 384). ``None`` autotunes it.
        fp8_block_m: Pin the tile height to 128 or 256. ``None`` autotunes it.
        q_descale / k_descale / v_descale: fp32 shape-[1] descales, required.
        out: Optional pre-allocated bf16 output of shape ``q.shape[:-1] + (Dv,)``.
        return_lse: Also return the fp32 log-sum-exp of the softmax logits.
        lse: Optional pre-allocated fp32 LSE buffer; allocated here when None.
            Dense: ``[B, H, Sq]``. Varlen: ``[H, total_q]``.
        daz: Enable denormals-are-zero.
        dualwave_swp_lazy_rescale: Enable lazy online softmax rescale.
        dualwave_swp_setprio: Enable s_setprio scheduling hints.
        dualwave_swp_enable_stagger: Enable wave-group phase stagger.
        stream: CUDA/HIP stream to launch on.

    Returns:
        bf16 output tensor of shape ``q.shape[:-1] + (v.shape[-1],)``, or
        ``(out, lse)`` when ``return_lse``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_fp8_func: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: q/k/v must share device; "
            f"got {q.device}/{k.device}/{v.device}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: q/k/v must share dtype; "
            f"got {q.dtype}/{k.dtype}/{v.dtype}"
        )
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: q/k/v must be float8_e4m3fn, got {q.dtype!r}"
        )

    _auto_splits = num_kv_splits is None
    if _auto_splits:
        num_kv_splits = 1

    # Q/K/V/O are flattened and the C-ABI packs the dynamic dim as int32, so no
    # tensor may reach 2**31. Batch entries are independent and a leading slice
    # stays contiguous, so one launch per entry divides the flat dim at no copy.
    _out_elems = q.numel() // q.shape[-1] * v.shape[-1]
    if max(q.numel(), k.numel(), v.numel(), _out_elems) >= _FP8_MAX_FLAT_ELEMS:
        _packed = cu_seqlens_q is not None or cu_seqlens_kv is not None or q.dim() != 4
        if _packed or q.shape[0] == 1:
            raise NotImplementedError(
                "flydsl_flash_attn_fp8_func: fp8 flattens Q/K/V/O and packs the dynamic "
                f"dim as int32, so no tensor may reach {_FP8_MAX_FLAT_ELEMS} elements; "
                f"got q={q.numel()}, k={k.numel()}, v={v.numel()}, out={_out_elems}. "
                "Shorten the sequence "
                "or use bf16."
            )
        kw = {
            "causal": causal,
            "num_kv_heads": num_kv_heads,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "cross_seqlen": cross_seqlen,
            "num_kv_splits": None if _auto_splits else num_kv_splits,
            "fp8_block_m": fp8_block_m,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
            "return_lse": return_lse,
            "daz": daz,
            "dualwave_swp_lazy_rescale": dualwave_swp_lazy_rescale,
            "dualwave_swp_setprio": dualwave_swp_setprio,
            "dualwave_swp_enable_stagger": dualwave_swp_enable_stagger,
            "stream": stream,
        }
        if out is None:
            # One buffer, a slice per launch: concatenating afterwards would
            # read the parts on the ambient stream while `stream` still runs.
            out = torch.empty(
                q.shape[:-1] + (v.shape[-1],), dtype=torch.bfloat16, device=q.device
            )
        if return_lse and lse is None:
            lse = torch.empty(
                (q.shape[0], q.shape[2], q.shape[1]),
                dtype=torch.float32,
                device=q.device,
            )
        for i in range(q.shape[0]):
            sl = slice(i, i + 1)
            flydsl_flash_attn_fp8_func(
                q[sl].contiguous(),
                k[sl].contiguous(),
                v[sl].contiguous(),
                out=out[sl],
                lse=lse[sl] if return_lse else None,
                **kw,
            )
        return (out, lse) if return_lse else out

    if any(x is None for x in (q_descale, k_descale, v_descale)):
        raise ValueError(
            "flydsl_flash_attn_fp8_func: fp8 requires q_descale, k_descale, and v_descale"
        )
    for name, scale in (
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
    ):
        if not scale.is_cuda:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must be a CUDA tensor"
            )
        if scale.device != q.device:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must be on {q.device}, got {scale.device}"
            )
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must be a shape-[1] float32 tensor"
            )

    varlen = cu_seqlens_q is not None
    if varlen and cu_seqlens_kv is None:
        raise ValueError(
            "flydsl_flash_attn_fp8_func: cu_seqlens_kv required when cu_seqlens_q is given"
        )
    if not varlen and cu_seqlens_kv is not None:
        raise ValueError(
            "flydsl_flash_attn_fp8_func: cu_seqlens_q required when cu_seqlens_kv is given"
        )

    if varlen:
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: varlen q must be 3D [total,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        Hkv = k.shape[1]
        Skv = None
        B = cu_seqlens_q.numel() - 1
        if max_seqlen_q is None:
            raise ValueError(
                "flydsl_flash_attn_fp8_func: max_seqlen_q is required in varlen mode"
            )
        if cross_seqlen is None:
            raise ValueError(
                "flydsl_flash_attn_fp8_func: cross_seqlen is required in varlen mode"
            )
        Sq = int(max_seqlen_q)
        cross = bool(cross_seqlen)
        if cross and max_seqlen_kv is None:
            raise ValueError(
                "flydsl_flash_attn_fp8_func: max_seqlen_kv is required when varlen cross_seqlen=True"
            )
    else:
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        Hkv = k.shape[2]
        cross = Sq != Skv if cross_seqlen is None else bool(cross_seqlen)

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: num_heads ({H}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
    if D < 64 or D % 32 != 0:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: head_dim ({D}) must be >= 64 and a multiple of 32"
        )

    Dv = int(v.shape[-1])
    if k.shape[-1] != D:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: K head_dim ({k.shape[-1]}) must match Q head_dim ({D})"
        )
    if tuple(v.shape[:-1]) != tuple(k.shape[:-1]):
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: V must match K in every dim but the last, got "
            f"v={tuple(v.shape)}, k={tuple(k.shape)}"
        )

    _skv_eff = (int(max_seqlen_kv) if cross else Sq) if varlen else int(Skv)
    _block_m = (
        _fp8_auto_block_m(B, H, Sq, _skv_eff, _num_cu(q.device))
        if fp8_block_m is None
        else int(fp8_block_m)
    )
    if _auto_splits and Sq >= 384:
        _auto = _fp8_auto_kv_splits(
            B, H, Sq, _skv_eff, causal, _num_cu(q.device), block_m=_block_m
        )
        if _auto > 1 and dualwave_splitk_workspace_elems(
            B, H, Sq, _auto, head_dim=Dv
        ) * 4 <= (_FP8_AUTOSPLIT_MAX_WS_BYTES):
            num_kv_splits = _auto

    splitk = num_kv_splits > 1
    if splitk:
        if Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: split-K requires seq_len>=384, got {Sq}"
            )
        ws_elems = dualwave_splitk_workspace_elems(
            B, H, Sq, int(num_kv_splits), head_dim=Dv
        )
        if ws_elems >= _FP8_MAX_FLAT_ELEMS:
            raise NotImplementedError(
                f"flydsl_flash_attn_fp8_func: num_kv_splits={int(num_kv_splits)} needs a "
                f"{ws_elems}-element split-K workspace; the C-ABI packs the dynamic "
                f"dim as int32 so it must stay under {_FP8_MAX_FLAT_ELEMS}"
            )

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        _arch = _gpu_arch(q.device)
        if not _arch.startswith("gfx950"):
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: fp8 requires gfx950, got '{_arch or 'unknown'}'"
            )
        exe = _build_fp8(
            num_heads=H,
            num_kv_heads=num_kv_heads,
            causal=causal,
            rescale_threshold=_fp8_rescale_threshold(_skv_eff),
            daz=daz,
            lazy_rescale=dualwave_swp_lazy_rescale,
            setprio=dualwave_swp_setprio,
            enable_stagger=dualwave_swp_enable_stagger,
            head_dim=D,
            head_dim_v=Dv,
            varlen=varlen,
            cross_seqlen=cross,
            num_kv_splits=int(num_kv_splits),
            block_m=_block_m,
            batch_interleave_group=_fp8_batch_interleave_group(
                B, causal, cross, int(num_kv_splits)
            ),
            return_lse=return_lse,
        )

        _out_shape = tuple(q.shape[:-1]) + (Dv,)
        if out is None:
            out = torch.empty(_out_shape, dtype=torch.bfloat16, device=q.device)
        elif tuple(out.shape) != _out_shape:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: out must be {_out_shape}, got {tuple(out.shape)}"
            )
        elif out.dtype != torch.bfloat16:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: fp8 output must be bf16, got {out.dtype}"
            )
        elif not out.is_contiguous():
            raise ValueError(
                "flydsl_flash_attn_fp8_func: out must be contiguous, got strides "
                f"{tuple(out.stride())} for shape {tuple(out.shape)}"
            )

        # Dense LSE is [B, H, Sq]; varlen is [H, total_q] (aiter's convention).
        if return_lse:
            _lse_shape = (H, int(q.shape[0])) if varlen else (B, H, Sq)
            if lse is None:
                lse = torch.empty(_lse_shape, dtype=torch.float32, device=q.device)
            elif tuple(lse.shape) != _lse_shape:
                raise ValueError(
                    f"flydsl_flash_attn_fp8_func: lse must be {_lse_shape}, got {tuple(lse.shape)}"
                )
            elif lse.dtype != torch.float32:
                raise ValueError(
                    f"flydsl_flash_attn_fp8_func: lse must be float32, got {lse.dtype}"
                )
            elif not lse.is_contiguous():
                raise ValueError(
                    "flydsl_flash_attn_fp8_func: lse must be contiguous, got strides "
                    f"{tuple(lse.stride())} for shape {tuple(lse.shape)}"
                )

        # The fp8 gfx950 module takes flattened Q/K/V/O plus descale kwargs.
        q_flat = q.contiguous().view(-1)
        k_flat = k.contiguous().view(-1)
        v_flat = v.contiguous().view(-1)
        o_flat = out.contiguous().view(-1)

        kwargs = {
            "stream": launch_stream,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }
        if return_lse:
            kwargs["lse"] = lse.view(-1)
            kwargs["lse_stride_h"] = _lse_shape[-1]
        if splitk:
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            if stream is not None:
                _ws.record_stream(launch_stream)
            kwargs["workspace"] = _ws
        if varlen:
            kwargs.update(cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
            if cross:
                kwargs["seq_len_kv"] = int(max_seqlen_kv)
        elif cross:
            kwargs["seq_len_kv"] = Skv
        exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        if stream is not None:
            out.record_stream(launch_stream)
            if return_lse:
                lse.record_stream(launch_stream)

    return (out, lse) if return_lse else out
