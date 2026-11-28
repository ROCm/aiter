# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/host.py
"""Host side of the one-launch MiniMax-M3.1 QKV front chain (``qkv_chain_cooperative``).

``qkv_chain_cooperative_insert`` runs QKV MXFP8 GEMM + q / k / index norm + partial RoPE + NVFP4 K / V / index-K
inserts for 1-64 decode rows in one launch. Epoch counters (each call adds 16 to every counter it uses, so a set
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
COOP_ROWS = (1, 64)

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
    """Rows 1-128 are served by ``qkv_chain_insert``."""
    return supported(m) or supported_split(m)


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


def _validate(
    m, activation, activation_scale, weight, weight_scale, q_norm_weight,
    k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache,
    positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
    k_scale, v_scale, q_out, index_q_out, num_heads, num_kv_heads,
    num_index_heads, eps,
):  # fmt: skip
    """Operand checks shared by both tiers; every raw pointer the kernels get is checked here, as exceptions (not
    ``assert``, which ``python -O`` strips)."""
    if not ((num_heads, num_kv_heads, num_index_heads) == (16, 1, 1)):
        raise ValueError(
            f"qkv front: serves only the MiniMax-M3.1 TP4 head set (16 q, 1 kv, 1 index head), got {(num_heads, num_kv_heads, num_index_heads)}"
        )
    if not (
        tuple(weight.shape) == (N, K)
        and activation.shape[1] == K
        and cos_sin_cache.shape[1] == ROT
    ):
        raise ValueError(
            f"qkv front: serves only N={N}, K={K}, rotary dim {ROT}; got weight {tuple(weight.shape)}, activation K {activation.shape[1]}, cos_sin_cache dim {cos_sin_cache.shape[1]}"
        )
    if not (eps >= 2.0**-126):
        raise ValueError(
            f"qkv front: eps {eps} is below 2**-126 (the kernel's rsqrt skips ocml's denormal scaling)"
        )
    if not (activation.is_contiguous() and weight.is_contiguous()):
        raise ValueError(
            "qkv front: need activation.is_contiguous() and weight.is_contiguous()"
        )
    if not (activation.shape[0] >= m and activation_scale.numel() >= m * (K // 32)):
        raise ValueError("row-major activation / scales shorter than m rows")
    if not (weight_scale.numel() == N * (K // 32)):
        raise ValueError("weight scales must be the AITER-shuffled [2560, 192] bytes")
    if not (
        positions.numel() >= m
        and slot_mapping.numel() >= m
        and index_slot_mapping.numel() >= m
    ):
        raise ValueError(
            "qkv front: need positions.numel() >= m and slot_mapping.numel() >= m and index_slot_mapping.numel() >= m"
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
        if not (t.device == activation.device):
            raise ValueError(f"{name} is on {t.device}")
    for name, t in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("index_q_norm_weight", index_q_norm_weight),
        ("index_k_norm_weight", index_k_norm_weight),
    ):
        if not (t.dtype == torch.bfloat16 and t.is_contiguous() and t.numel() >= 128):
            raise ValueError(f"{name}: needs 128 contiguous bf16")
    if not (cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()):
        raise ValueError(
            "qkv front: need cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()"
        )
    for name, t in (
        ("positions", positions),
        ("slot_mapping", slot_mapping),
        ("index_slot_mapping", index_slot_mapping),
    ):
        if not (t.dtype == torch.int64 and t.is_contiguous() and t.numel() >= m):
            raise ValueError(f"{name}: needs {m} contiguous int64")
    for name, t in (("k_scale", k_scale), ("v_scale", v_scale)):
        # both kernels read the scalar through _gptr(..., T.f32, 4): it must be one f32 (4 bytes),
        # a bf16 scalar would be a 2-byte read reinterpreted as f32
        if not (t.dtype == torch.float32 and t.numel() >= 1):
            raise ValueError(f"qkv front: need {name} float32 with numel() >= 1")
    if not (q_out.dtype == torch.bfloat16 and q_out.is_contiguous()):
        raise ValueError(
            "qkv front: need q_out.dtype == torch.bfloat16 and q_out.is_contiguous()"
        )
    if not (tuple(q_out.shape) == (m, num_heads * 128)):
        raise ValueError(f"q_out {tuple(q_out.shape)}")
    if not (index_q_out.shape[0] >= m and index_q_out.shape[1] == 128):
        raise ValueError(
            "qkv front: need index_q_out.shape[0] >= m and index_q_out.shape[1] == 128"
        )
    if not (
        index_q_out.dtype == torch.float8_e4m3fn
        and index_q_out.stride(1) == 1
        and index_q_out.stride(0) % 4 == 0
    ):
        raise ValueError(
            "qkv front: need index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1 and index_q_out.stride(0) % 4 == 0"
        )
    # page stride(0) must cover the whole page footprint: the kernels index pages by ``slot >> 7``, so an
    # expanded/overlapping page layout (e.g. stride(0) == 0) would make distinct slots alias the same page.
    if not (
        kv_cache.dtype == torch.uint8
        and kv_cache.shape[1:] == (2, 128, 72)
        and kv_cache.stride(0) >= 2 * 128 * 72
        and kv_cache.stride(1) == 128 * 72
        and kv_cache.stride(3) == 1
    ):
        raise ValueError(
            "qkv front: need uint8 kv_cache, kv_cache.shape[1:] == (2, 128, 72), kv_cache.stride(0) >= 2 * 128 * 72 and kv_cache.stride(1) == 128 * 72 and kv_cache.stride(3) == 1"
        )
    if not (
        index_cache.dtype == torch.uint8
        and index_cache.shape[1:] == (128, 72)
        and index_cache.stride(0) >= 128 * 72
        and index_cache.stride(1) == 72
        and index_cache.stride(2) == 1
    ):
        raise ValueError(
            "qkv front: need uint8 index_cache, index_cache.shape[1:] == (128, 72), index_cache.stride(0) >= 128 * 72 and index_cache.stride(1) == 72 and index_cache.stride(2) == 1"
        )


def qkv_chain_cooperative_insert(
    activation: torch.Tensor,  # e4m3 codes [m, K], row-major (per_group_quant_hip)
    activation_scale: torch.Tensor,  # E8M0 scales [m, K / 32], row-major, unshuffled
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
) -> None:
    """Writes ``q_out``, ``index_q_out`` and the KV / index pages; bitwise the unfused split-3 chain. The activation is
    the plain quantizer's output (``per_group_quant_hip``, 1x32 E8M0), no relayout."""
    if not supported(m):
        raise ValueError(f"qkv chain: {m} rows outside {COOP_ROWS}")
    _validate(
        m, activation, activation_scale, weight, weight_scale, q_norm_weight,
        k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache,
        positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
        k_scale, v_scale, q_out, index_q_out, num_heads, num_kv_heads,
        num_index_heads, eps,
    )  # fmt: skip
    dev = activation.device
    with torch.cuda.device(dev):
        _launch(
            dev, m, activation, weight, activation_scale, weight_scale, q_out,
            index_q_out, q_norm_weight, k_norm_weight, index_q_norm_weight,
            index_k_norm_weight, cos_sin_cache, positions, slot_mapping,
            index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, eps,
        )  # fmt: skip


def _launch(
    dev, m, activation, weight, activation_scale, weight_scale, q_out, index_q_out,
    q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
    cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache,
    index_cache, k_scale, v_scale, eps,
):  # fmt: skip
    slab = torch.empty((KS, m, N), dtype=torch.float32, device=dev)
    _run_compiled(
        compile_qkv_cooperative((m + 15) // 16 * 16),
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
        m * (K // 32),
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(dev),
        specialization_key=("dev", dev.index),
    )


def qkv_chain_split_insert(
    activation: torch.Tensor,  # e4m3 codes [m, K], row-major (per_group_quant_hip)
    activation_scale: torch.Tensor,  # E8M0 scales [m, K / 32], row-major, unshuffled
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
    ks: int = _splitk.KS,
    cfg: tuple[int, int, bool, bool] | None = None,
) -> None:
    """65-128 rows in two launches: the split-3 decode GEMM (fp32 K-third slabs) and ``qkv_reduce_insert`` (sum, bf16
    round, norm, RoPE, NVFP4 K / V / index-K inserts). Bitwise the unfused split-3 chain.
    """
    if not supported_split(m):
        raise ValueError(f"qkv split chain: {m} rows outside {SPLIT_ROWS}")
    _validate(
        m, activation, activation_scale, weight, weight_scale, q_norm_weight,
        k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache,
        positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
        k_scale, v_scale, q_out, index_q_out, num_heads, num_kv_heads,
        num_index_heads, eps,
    )  # fmt: skip
    # the split-3 GEMM on row-major activation codes / scales (#8610: ``mxfp8_gemm_decode_splitk`` takes row-major A only)
    dev = activation.device
    with torch.cuda.device(dev):
        slab = _splitk.mxfp8_gemm_decode_splitk(
            activation, activation_scale, weight, weight_scale, cfg=cfg, ks=ks
        )
        _run_compiled(
            compile_qkv_reduce_insert(ks),
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
            torch.cuda.current_stream(dev),
            specialization_key=("dev", dev.index),
        )


def qkv_chain_insert(*args, **kwargs) -> None:
    """The QKV front chain at 1-128 decode rows: one launch up to 64 rows, two launches above."""
    m = args[4]
    if supported(m):
        return qkv_chain_cooperative_insert(*args, **kwargs)
    return qkv_chain_split_insert(*args, **kwargs)
