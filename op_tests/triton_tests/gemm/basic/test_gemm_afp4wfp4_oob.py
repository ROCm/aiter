# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Out-of-bounds regression test for the MXFP4 preshuffle GEMM kernels.

``_gemm_afp4wfp4_preshuffle_kernel`` clamps its tile offsets with the wrong
modulus. Three of its four global reads index a buffer BY ROW, but wrap against
N / M instead of that buffer's row count:

    offs_am  = (pid_m*BSM       + arange(BSM))      % M   # a       has M rows      -- correct
    offs_bn  = (pid_n*(BSN//16) + arange(BSN//16))  % N   # b       has N//16 rows  -- 16x too large
    offs_asn = (pid_n*(BSN//32) + arange(BSN//32))  % N   # b_scale has N//32 rows  -- 32x too large
    offs_asm = (pid_m*(BSM//32) + arange(BSM//32))  % M   # a_scale has ceil(M/32)+ -- 32x too large

Because the modulus is far larger than the row count it never wraps, and the
loads are unmasked, so whenever N is not a multiple of BLOCK_SIZE_N the last
N tile reads past the end of the weight and weight-scale buffers.

DeepSeek-V3/R1 hits this with its fused qkv_a_proj: N = q_lora_rank +
kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64 = 2112, which is not a
multiple of the tuned BLOCK_SIZE_N=128 used for M >= 512. The kernel reads
4 weight rows (224 KiB) and 2 weight-scale rows (14 KiB) past the end.

Detecting this needs care: a plain overrun of a few hundred KiB lands inside
memory the allocator already has mapped, so it is silently absorbed and nothing
fails -- which is exactly why it survived into production and only surfaced as a
rare, load-dependent ``Memory access fault``. ``test_preshuffle_oob_guard_page``
therefore backs the weight with an explicit HIP virtual-memory mapping whose
last byte is the last mapped byte, so any overrun hits an unmapped page and
faults deterministically. The launch runs in a subprocess because a GPU memory
fault aborts the whole process.

Usage:
    pytest op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4_oob.py -q
"""

from __future__ import annotations

import ctypes
import subprocess
import sys

import pytest
import torch

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config
from aiter.ops.triton.utils._triton import arch_info

# (id, M, N, K). The fused qkv_a_proj cases overrun on the unfixed kernel; o_proj
# has N % BLOCK_SIZE_N == 0 and is the control that must pass either way.
CASES = [
    ("qkv_a_proj_prefill", 5125, 2112, 7168),  # 5 x 1025-token prefill
    ("qkv_a_proj_decode", 512, 2112, 7168),  # max_num_seqs decode step
    ("o_proj_prefill", 5125, 7168, 7168),  # control: N is a multiple of 128
]


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _block_sizes(M: int, N: int, K: int) -> tuple[int, int]:
    cfg, _ = _get_config(M, N, K, True)
    bsn = max(cfg["BLOCK_SIZE_N"], 32)
    bsm = max(cfg["BLOCK_SIZE_M"], 32) if M >= 32 else cfg["BLOCK_SIZE_M"]
    return bsm, bsn


# --------------------------------------------------------------------------
# Guard-page launch: run in a subprocess, a GPU fault takes the process with it.
# --------------------------------------------------------------------------

_HIP_PINNED, _HIP_LOC_DEVICE, _HIP_RW, _HIP_GRAN_MIN = 1, 1, 3, 0


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


class _CudaArrayInterface:
    """Minimal wrapper so torch can view externally mapped device memory."""

    def __init__(self, ptr: int, shape: tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


def _alloc_with_trailing_guard(nbytes: int) -> torch.Tensor:
    """Map exactly ``nbytes`` and leave the next granule reserved but unmapped.

    The returned tensor's last byte is the last mapped byte, so a kernel reading
    even one byte past it faults instead of silently hitting neighbouring data.
    """
    hip = ctypes.CDLL("libamdhip64.so.7")

    def ck(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with hipError {rc}")

    prop = _MemAllocationProp()
    prop.type = _HIP_PINNED
    prop.location.type = _HIP_LOC_DEVICE
    prop.location.id = torch.cuda.current_device()

    gran = ctypes.c_size_t()
    ck(
        hip.hipMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), _HIP_GRAN_MIN
        ),
        "hipMemGetAllocationGranularity",
    )
    g = gran.value
    if nbytes % g:
        raise RuntimeError(
            f"buffer of {nbytes} bytes is not a multiple of the {g}-byte "
            "allocation granularity, so its end cannot be a mapping boundary"
        )

    ptr = ctypes.c_void_p()
    ck(
        hip.hipMemAddressReserve(
            ctypes.byref(ptr),
            ctypes.c_size_t(nbytes + g),  # one spare granule, deliberately unmapped
            ctypes.c_size_t(0),
            ctypes.c_void_p(0),
            ctypes.c_ulonglong(0),
        ),
        "hipMemAddressReserve",
    )
    handle = ctypes.c_ulonglong()
    ck(
        hip.hipMemCreate(
            ctypes.byref(handle),
            ctypes.c_size_t(nbytes),
            ctypes.byref(prop),
            ctypes.c_ulonglong(0),
        ),
        "hipMemCreate",
    )
    ck(
        hip.hipMemMap(
            ptr, ctypes.c_size_t(nbytes), ctypes.c_size_t(0), handle,
            ctypes.c_ulonglong(0),
        ),
        "hipMemMap",
    )
    desc = _MemAccessDesc()
    desc.location.type = _HIP_LOC_DEVICE
    desc.location.id = torch.cuda.current_device()
    desc.flags = _HIP_RW
    ck(
        hip.hipMemSetAccess(
            ptr, ctypes.c_size_t(nbytes), ctypes.byref(desc), ctypes.c_size_t(1)
        ),
        "hipMemSetAccess",
    )
    return torch.as_tensor(_CudaArrayInterface(ptr.value, (nbytes,)), device="cuda")


def _guarded_launch(M: int, N: int, K: int) -> None:
    """Run the preshuffle GEMM with the weight ending on a mapping boundary."""
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle

    rows, row_bytes = N // 16, (K // 2) * 16
    bsm, bsn = _block_sizes(M, N, K)
    over = (cdiv(N, bsn) * (bsn // 16) - rows) * row_bytes
    print(
        f"M={M} N={N} K={K} BLOCK_SIZE_N={bsn}: weight has {rows} rows, "
        f"kernel reads {cdiv(N, bsn) * (bsn // 16)} -> {over} bytes past the end",
        flush=True,
    )

    x_fp4 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_scales = torch.randint(
        124, 128, (cdiv(M, 256) * 8, K), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(124, 128, (N // 32, K), dtype=torch.uint8, device="cuda")
    y = torch.empty((cdiv(M, 32) * 32, N), dtype=torch.bfloat16, device="cuda")

    w_flat = _alloc_with_trailing_guard(rows * row_bytes)
    w_flat.fill_(0x22)  # two packed e2m1 1.0 values; any byte pattern is valid fp4
    w_preshuf = w_flat.view(rows, row_bytes)
    torch.cuda.synchronize()

    gemm_afp4wfp4_preshuffle(x_fp4, w_preshuf, x_scales, w_scales, y=y)
    torch.cuda.synchronize()
    print("no fault", flush=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not arch_info.is_fp4_avail(),
    reason="MXFP4 GPU required",
)
@pytest.mark.parametrize(
    "M, N, K", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_preshuffle_oob_guard_page(M: int, N: int, K: int):
    """The kernel must not read past a weight buffer that ends at a page boundary."""
    proc = subprocess.run(
        [sys.executable, __file__, "--child", str(M), str(N), str(K)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        "preshuffle GEMM faulted reading past the weight buffer "
        f"(exit {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _guarded_launch(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
