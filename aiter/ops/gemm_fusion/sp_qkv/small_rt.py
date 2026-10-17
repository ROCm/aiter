# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ctypes runtime for the small-M driver (sm_fusion_driver.cpp). No torch import at module level.
SmBindingV1 mirrors candidates/SM-T1-r18/host/bridge_sm.py (sha256 f243ef11...) field for field."""

from __future__ import annotations

import ctypes as C

OK = 0
SM_MP = 32


class SmBindingV1(C.Structure):
    _fields_ = [
        ("A", C.c_void_p), ("W", C.c_void_p), ("Ascale", C.c_void_p), ("Wscale", C.c_void_p),
        ("qw", C.c_void_p), ("kw", C.c_void_p), ("cos_cache", C.c_void_p), ("sin_cache", C.c_void_p),
        ("cos_stride_pos", C.c_int64), ("positions", C.c_void_p), ("slot_mapping", C.c_void_p),
        ("k_cache", C.c_void_p), ("v_cache", C.c_void_p), ("k_scale", C.c_void_p), ("v_scale", C.c_void_p),
        ("kc_sb", C.c_int64), ("kc_sh", C.c_int64), ("kc_sdx", C.c_int64), ("kc_ss", C.c_int64), ("kc_sx", C.c_int64),
        ("vc_sb", C.c_int64), ("vc_sh", C.c_int64), ("vc_ssc", C.c_int64), ("vc_sd", C.c_int64), ("vc_sx", C.c_int64),
        ("ks_sb", C.c_int64), ("ks_sh", C.c_int64), ("vs_sb", C.c_int64), ("vs_sh", C.c_int64),
        ("q_out", C.c_void_p), ("k_out", C.c_void_p), ("v_out", C.c_void_p),
        ("iqw", C.c_void_p), ("ikw", C.c_void_p), ("index_q_out", C.c_void_p), ("index_cache", C.c_void_p),
        ("index_slot_mapping", C.c_void_p),
        ("M", C.c_uint32), ("T", C.c_uint32), ("block_size", C.c_uint32), ("x_size", C.c_uint32),
        ("strideA", C.c_uint32), ("pad0", C.c_uint32), ("eps", C.c_float), ("fp8_amax", C.c_float),
        ("poll_iterations", C.c_uint64),
    ]


def load_driver(path: str):
    lib = C.CDLL(path)
    vp, u32, u64, i32 = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32
    sig = [
        ("sm_attach_v1", [u32, vp, vp, vp, vp, vp, C.c_char_p, C.c_char_p, C.POINTER(vp)]),
        ("sm_bind_new_v1", [vp, vp, C.POINTER(vp)]), ("sm_launch_blk_v1", [vp, vp, vp, vp, vp]),
        ("sm_blk_free_v1", [vp]), ("sm_info_v1", [vp]), ("sm_poison_v1", [vp, vp]), ("sm_last_hip_error_v1", [vp]),
    ]
    for n, a in sig:
        f = getattr(lib, n)
        f.restype = i32
        f.argtypes = a
    return lib


def info(lib):
    b = (C.c_uint64 * 7)()
    if lib.sm_info_v1(b) != OK:
        raise RuntimeError("sm_info")
    keys = ("rx_bytes", "ctl_bytes", "n", "f", "ran_grid", "ran_threads", "gemm_ntiles")
    return dict(zip(keys, list(b)))


def stream_handle(stream):
    """None -> current torch stream; int; or object with .cuda_stream."""
    if stream is None:
        import torch

        stream = torch.cuda.current_stream()
    return C.c_void_p(stream if isinstance(stream, int) else stream.cuda_stream)


class SmallContext:
    """One compiled kernel pair attached to one owner's Rx+control windows (one epoch domain)."""

    def __init__(self, lib, ctx):
        self.lib, self.ctx = lib, ctx

    def poison(self) -> int:
        v = C.c_uint32()
        if self.lib.sm_poison_v1(self.ctx, C.byref(v)) != OK:
            raise RuntimeError("sm_poison")
        return v.value

    def last_hip_error(self) -> int:
        return self.lib.sm_last_hip_error_v1(self.ctx)

    def bind(self, sb: SmBindingV1):
        blk = C.c_void_p()
        rc = self.lib.sm_bind_new_v1(self.ctx, C.byref(sb), C.byref(blk))
        if rc != OK:
            raise RuntimeError(f"sm_bind_new rc={rc}")
        return blk

    def launch(self, blk, A_ptr: int, W_ptr: int, stream):
        rc = self.lib.sm_launch_blk_v1(self.ctx, blk, C.c_void_p(A_ptr), C.c_void_p(W_ptr), stream)
        if rc != OK:
            raise RuntimeError(f"sm_launch_blk rc={rc}")
