# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/host.py
"""Host side of the one-launch MiniMax-M3.1 QKV front chain (``qkv_chain_cooperative``).

``qkv_chain_cooperative_insert`` runs QKV MXFP8 GEMM + q / k / index norm + partial RoPE + NVFP4 K / V / index-K
inserts for 17-64 decode rows in one launch. Epoch counters (each call adds 16 to every counter it uses, so a set
stays a multiple of 16 between calls and needs no re-arm) belong to a device and an eager stream, or to the HIP
capture ID plus stream: separate graphs captured on one stream and replayed concurrently use different storage.
"""

import ctypes
import functools
import struct
import threading

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from . import mxfp8_gemm_decode_splitk as _splitk
from .qkv_chain_cooperative import CNT_STRIDE, HEADS, KS, compile_qkv_cooperative
from .qkv_reduce_insert import compile_qkv_reduce_insert

N, K = 2560, 6144
ROT = 64
COOP_ROWS = (17, 64)

_COUNTER_SETS = 512  # eager streams and retained graph captures per device
_COUNTER_WORDS = HEADS * CNT_STRIDE
_LOCK = threading.Lock()
_POOL: dict = {}
_COUNTERS: dict = {}


SPLIT_ROWS = _splitk.ROWS  # (65, 128): split-3 GEMM launch + reduce-insert launch


def supported(m: int) -> bool:
    return COOP_ROWS[0] <= m <= COOP_ROWS[1]


def supported_split(m: int) -> bool:
    return _splitk.supported(m)


def chain_supported(m: int) -> bool:
    """Rows 17-128 are served by ``qkv_chain_insert``."""
    return supported(m) or supported_split(m)


def code_rows(m: int) -> int:
    """Rows of (16, 16)-layout activation codes ``qkv_chain_insert`` reads at ``m`` rows: pass this as the quantizer's
    block (``mxfp8_quantize_shuffled(x, code_rows(m))``)."""
    return (m + 15) // 16 * 16 if supported(m) else _splitk.code_rows(m)


def tile_rows(m: int) -> int:
    """Row tile of the (16, 16)-layout activation codes this chain reads (rows are padded to it)."""
    return 16


@functools.cache
def _capture_info_api():
    lib = ctypes.CDLL("libamdhip64.so")
    fn = lib.hipStreamGetCaptureInfo
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    fn.restype = ctypes.c_int
    return lib, fn


def _counter_key(device):
    stream = torch.cuda.current_stream(device).cuda_stream
    if not torch.cuda.is_current_stream_capturing():
        return device, ("stream", stream)
    _, fn = _capture_info_api()
    status, capture_id = ctypes.c_int(), ctypes.c_ulonglong()
    rc = fn(ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(capture_id))
    if rc or status.value != 1:
        raise RuntimeError(
            f"hipStreamGetCaptureInfo failed: rc={rc}, status={status.value}"
        )
    return device, ("capture", capture_id.value, stream)


def _counters(device: torch.device) -> torch.Tensor:
    key = _counter_key(device)
    cnt = _COUNTERS.get(key)
    if cnt is not None:
        return cnt
    with _LOCK:
        cnt = _COUNTERS.get(key)
        if cnt is not None:
            return cnt
        pool = _POOL.get(device)
        if pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "qkv_chain_cooperative_insert: first call inside CUDA graph capture; call it once eagerly first"
                )
            pool = torch.zeros(
                _COUNTER_SETS * _COUNTER_WORDS, dtype=torch.int32, device=device
            )
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            ready.synchronize()
            _POOL[device] = pool
        used = sum(1 for d, _ in _COUNTERS if d == device)
        if used >= _COUNTER_SETS:
            raise RuntimeError(
                f"qkv_chain_cooperative_insert: more than {_COUNTER_SETS} counter sets on {device}"
            )
        cnt = _COUNTERS[key] = pool[used * _COUNTER_WORDS : (used + 1) * _COUNTER_WORDS]
        return cnt


def qkv_chain_cooperative_insert(
    activation: torch.Tensor,  # e4m3 codes, (16, 16) layout, rows padded to 16
    activation_scale: torch.Tensor,  # AITER-shuffled E8M0 scales [ceil(rows / 32) * 32, K / 32]
    weight: torch.Tensor,  # [2560, 6144] preshuffled (16, 16) e4m3 codes
    weight_scale: torch.Tensor,  # its AITER-shuffled E8M0 scales
    m: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16
    positions: torch.Tensor,  # [m] int64
    slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [m, 2048] bf16
    index_q_out: torch.Tensor,  # [m, 128] e4m3, row stride free
    *,
    num_heads: int = 16,
    num_kv_heads: int = 1,
    num_index_heads: int = 1,
    eps: float,
    a_row: bool = False,
) -> None:
    """Writes ``q_out``, ``index_q_out`` and the KV / index pages; bitwise the unfused split-3 chain.

    ``a_row``: ``activation`` / ``activation_scale`` are the plain quantizer's row-major ``[m, K]`` codes and
    ``[m, K / 32]`` E8M0 scales (``per_group_quant_hip``) instead of the (16, 16)-layout / shuffled-scale pair.
    """
    assert supported(m), f"qkv chain: {m} rows outside {COOP_ROWS}"
    a_row_scale_ok = not a_row or (
        activation.shape[0] >= m and activation_scale.numel() >= m * (K // 32)
    )
    assert (num_heads, num_kv_heads, num_index_heads) == (16, 1, 1)
    assert (
        tuple(weight.shape) == (N, K)
        and activation.shape[1] == K
        and cos_sin_cache.shape[1] == ROT
    )
    assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
    rows = (
        (m + 15) // 16 * 16
    )  # the kernel reads whole 16-row tiles of codes and 32-row groups of scales
    assert activation.is_contiguous() and weight.is_contiguous()
    if a_row:
        assert a_row_scale_ok, "row-major activation / scales shorter than m rows"
    else:
        assert (
            activation.shape[0] >= rows
        ), f"activation has {activation.shape[0]} rows, the {m}-row tile needs {rows}"
        assert activation_scale.numel() >= (rows + 31) // 32 * 32 * (
            K // 32
        ), "activation scales too short"
    assert weight_scale.numel() == N * (
        K // 32
    ), "weight scales must be the AITER-shuffled [2560, 192] bytes"
    assert (
        positions.numel() >= m
        and slot_mapping.numel() >= m
        and index_slot_mapping.numel() >= m
    )
    for name, t in (
        ("activation_scale", activation_scale),
        ("weight", weight),
        ("weight_scale", weight_scale),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("index_q_norm_weight", index_q_norm_weight),
        ("index_k_norm_weight", index_k_norm_weight),
        ("cos_sin_cache", cos_sin_cache),
        ("positions", positions),
        ("slot_mapping", slot_mapping),
        ("index_slot_mapping", index_slot_mapping),
        ("kv_cache", kv_cache),
        ("index_cache", index_cache),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
        ("q_out", q_out),
        ("index_q_out", index_q_out),
    ):
        # raw pointers: every operand on the activation's device
        assert t.device == activation.device, f"{name} is on {t.device}"
    for name, t in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("index_q_norm_weight", index_q_norm_weight),
        ("index_k_norm_weight", index_k_norm_weight),
    ):
        assert (
            t.dtype == torch.bfloat16 and t.is_contiguous() and t.numel() >= 128
        ), f"{name}: needs 128 contiguous bf16"
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    for name, t in (
        ("positions", positions),
        ("slot_mapping", slot_mapping),
        ("index_slot_mapping", index_slot_mapping),
    ):
        assert (
            t.dtype == torch.int64 and t.is_contiguous() and t.numel() >= m
        ), f"{name}: needs {m} contiguous int64"
    assert k_scale.numel() >= 1 and v_scale.numel() >= 1
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    assert tuple(q_out.shape) == (m, num_heads * 128), f"q_out {tuple(q_out.shape)}"
    assert index_q_out.shape[0] >= m and index_q_out.shape[1] == 128
    assert (
        index_q_out.dtype == torch.float8_e4m3fn
        and index_q_out.stride(1) == 1
        and index_q_out.stride(0) % 4 == 0
    )
    assert (
        kv_cache.shape[1:] == (2, 128, 72)
        and kv_cache.stride(1) == 128 * 72
        and kv_cache.stride(3) == 1
    )
    assert (
        index_cache.shape[1:] == (128, 72)
        and index_cache.stride(1) == 72
        and index_cache.stride(2) == 1
    )
    dev = activation.device
    with torch.cuda.device(dev):
        _launch(
            dev, m, activation, weight, activation_scale, weight_scale, q_out,
            index_q_out, q_norm_weight, k_norm_weight, index_q_norm_weight,
            index_k_norm_weight, cos_sin_cache, positions, slot_mapping,
            index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, eps, a_row,
        )  # fmt: skip


def _launch(
    dev, m, activation, weight, activation_scale, weight_scale, q_out, index_q_out,
    q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
    cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache,
    index_cache, k_scale, v_scale, eps, a_row=False,
):  # fmt: skip
    slab = torch.empty((KS, m, N), dtype=torch.float32, device=dev)
    _run_compiled(
        compile_qkv_cooperative((m + 15) // 16 * 16, a_row),
        activation.data_ptr(),
        weight.data_ptr(),
        activation_scale.data_ptr(),
        weight_scale.data_ptr(),
        slab.data_ptr(),
        _counters(dev).data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        m,
        (
            (m * (K // 32))
            if a_row
            else activation_scale.numel() * activation_scale.element_size()
        ),
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(dev),
        specialization_key=("dev", dev.index),
    )


def qkv_chain_split_insert(
    activation: torch.Tensor,  # e4m3 codes, (16, 16) layout, ``code_rows(m)`` rows
    activation_scale: torch.Tensor,  # AITER-shuffled E8M0 scales [ceil(rows / 32) * 32, K / 32]
    weight: torch.Tensor,  # [2560, 6144] preshuffled (16, 16) e4m3 codes
    weight_scale: torch.Tensor,  # its AITER-shuffled E8M0 scales
    m: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16
    positions: torch.Tensor,  # [m] int64
    slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [m, 2048] bf16
    index_q_out: torch.Tensor,  # [m, 128] e4m3, row stride free
    *,
    num_heads: int = 16,
    num_kv_heads: int = 1,
    num_index_heads: int = 1,
    eps: float,
    a_row: bool = False,
) -> None:
    """65-128 rows in two launches: the split-3 decode GEMM (fp32 K-third slabs) and ``qkv_reduce_insert`` (sum, bf16
    round, norm, RoPE, NVFP4 K / V / index-K inserts). Bitwise the unfused split-3 chain.
    """
    assert supported_split(m), f"qkv split chain: {m} rows outside {SPLIT_ROWS}"
    assert (num_heads, num_kv_heads, num_index_heads) == (16, 1, 1)
    assert cos_sin_cache.shape[1] == ROT
    assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
    assert (
        positions.numel() >= m
        and slot_mapping.numel() >= m
        and index_slot_mapping.numel() >= m
    )
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    assert (
        index_q_out.dtype == torch.float8_e4m3fn
        and index_q_out.stride(1) == 1
        and index_q_out.stride(0) % 4 == 0
    )
    assert (
        kv_cache.shape[1:] == (2, 128, 72)
        and kv_cache.stride(1) == 128 * 72
        and kv_cache.stride(3) == 1
    )
    assert (
        index_cache.shape[1:] == (128, 72)
        and index_cache.stride(1) == 72
        and index_cache.stride(2) == 1
    )
    if a_row:
        # the split-3 GEMM's row-major-activation entry (same results; needs the A_ROW split-3 kernel, #8610)
        slab = _splitk.mxfp8_gemm_decode_splitk_rowmajor(
            activation, activation_scale, weight, weight_scale
        )
    else:
        slab = _splitk.mxfp8_gemm_decode_splitk(
            activation, activation_scale, weight, weight_scale, m
        )
    _run_compiled(
        compile_qkv_reduce_insert(),
        slab.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        m,
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(),
    )


def qkv_chain_insert(*args, **kwargs) -> None:
    """The QKV front chain at 17-128 decode rows: one launch up to 64 rows, two launches above."""
    m = args[4]
    if supported(m):
        return qkv_chain_cooperative_insert(*args, **kwargs)
    return qkv_chain_split_insert(*args, **kwargs)
