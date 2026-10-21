# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ctypes runtime for the small-M driver (sm_fusion_driver.cpp). No torch import at module level.
SmBindingV1 mirrors candidates/SM-T1-r18/host/bridge_sm.py (sha256 f243ef11...) field for field.
"""

from __future__ import annotations

import ctypes as C

OK = 0
SM_MP = 32


class SmBindingV1(C.Structure):
    _fields_ = [
        ("A", C.c_void_p),
        ("W", C.c_void_p),
        ("Ascale", C.c_void_p),
        ("Wscale", C.c_void_p),
        ("qw", C.c_void_p),
        ("kw", C.c_void_p),
        ("cos_cache", C.c_void_p),
        ("sin_cache", C.c_void_p),
        ("cos_stride_pos", C.c_int64),
        ("positions", C.c_void_p),
        ("slot_mapping", C.c_void_p),
        ("k_cache", C.c_void_p),
        ("v_cache", C.c_void_p),
        ("k_scale", C.c_void_p),
        ("v_scale", C.c_void_p),
        ("kc_sb", C.c_int64),
        ("kc_sh", C.c_int64),
        ("kc_sdx", C.c_int64),
        ("kc_ss", C.c_int64),
        ("kc_sx", C.c_int64),
        ("vc_sb", C.c_int64),
        ("vc_sh", C.c_int64),
        ("vc_ssc", C.c_int64),
        ("vc_sd", C.c_int64),
        ("vc_sx", C.c_int64),
        ("ks_sb", C.c_int64),
        ("ks_sh", C.c_int64),
        ("vs_sb", C.c_int64),
        ("vs_sh", C.c_int64),
        ("q_out", C.c_void_p),
        ("k_out", C.c_void_p),
        ("v_out", C.c_void_p),
        ("iqw", C.c_void_p),
        ("ikw", C.c_void_p),
        ("index_q_out", C.c_void_p),
        ("index_cache", C.c_void_p),
        ("index_slot_mapping", C.c_void_p),
        ("M", C.c_uint32),
        ("T", C.c_uint32),
        ("block_size", C.c_uint32),
        ("x_size", C.c_uint32),
        ("strideA", C.c_uint32),
        ("pad0", C.c_uint32),
        ("eps", C.c_float),
        ("fp8_amax", C.c_float),
        ("poll_iterations", C.c_uint64),
    ]


# ---- R12 (design card r2 D2.3; small/r19, SM_ABI_VERSION 3) ----------------------------------------------------------------------------------------------------
# SmRanArgsV3 is the RAN kernel's by-value kernarg block (one s_load_dwordx16): sizeof == 64 is pinned here and by the r19 header's static_assert. The host driver builds it
# natively; this mirror exists to pin the layout and to unit-test the packing. err/step point into the per-comm global page.
SM_RAN_ARGS_V3_BYTES = 64


class SmRanArgsV3(C.Structure):
    _fields_ = [
        ("blk", C.c_void_p),
        ("Ascale", C.c_void_p),
        ("positions", C.c_void_p),
        ("slot_mapping", C.c_void_p),
        ("err", C.c_void_p),
        ("step", C.c_void_p),
        ("M", C.c_uint32),
        ("rank", C.c_uint32),
        ("t_valid", C.c_uint32),
        ("call_id", C.c_uint32),
    ]


assert C.sizeof(SmRanArgsV3) == SM_RAN_ARGS_V3_BYTES


def load_driver(path: str, abi: int = 2):
    """Bind the small driver. abi 2 = the shipped v1 entry points; abi 3 = the R12 entry points (sm_attach_v3 / sm_launch_blk_v3; NO per-plan poison read). The abi 3
    signatures are those of the r19 host entry points (sm_fusion_driver.cpp under SM_ABI_VERSION >= 3): attach(rank, comm, rx_win, rx_local, ctl_win, ctl_local,
    gpage_local (64 B aligned), gpeers (a device-resident SmGfPeers* from the runtime, or NULL: the driver resolves the 3 peer ERR words with ccoGetPeerPtr),
    gemm_hsaco, ran_hsaco, &ctx); launch(ctx, blk, A, W, strideA, Ascale, positions, slot_mapping, t_valid (<= T), call_id, stream). The per-launch tensors must hold at
    least t_valid int64 (positions, slot_mapping) and M floats (Ascale)."""
    lib = C.CDLL(path)
    vp, u32, i32 = C.c_void_p, C.c_uint32, C.c_int32
    if abi == 3:
        sig3 = [
            (
                "sm_attach_v3",
                [
                    u32,
                    vp,
                    vp,
                    vp,
                    vp,
                    vp,
                    vp,
                    vp,
                    C.c_char_p,
                    C.c_char_p,
                    C.POINTER(vp),
                ],
            ),
            ("sm_bind_new_v1", [vp, vp, C.POINTER(vp)]),
            ("sm_launch_blk_v3", [vp, vp, vp, vp, u32, vp, vp, vp, u32, u32, vp]),
            ("sm_blk_free_v1", [vp]),
            ("sm_free_v1", [vp]),
            ("sm_info_v1", [vp]),
        ]
        for n, a in sig3:
            f = getattr(lib, n)
            f.restype = i32
            f.argtypes = a
        return lib
    sig = [
        (
            "sm_attach_v1",
            [u32, vp, vp, vp, vp, vp, C.c_char_p, C.c_char_p, C.POINTER(vp)],
        ),
        ("sm_bind_new_v1", [vp, vp, C.POINTER(vp)]),
        ("sm_launch_blk_v1", [vp, vp, vp, vp, u32, vp]),
        ("sm_blk_free_v1", [vp]),
        ("sm_free_v1", [vp]),
        ("sm_poison_fast_v1", [vp, vp, C.POINTER(u32)]),
        ("sm_info_v1", [vp]),
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
        self.blks = []

    def poison_fast(self, stream=None) -> int:
        """Cheap per-step host check: 4-byte D2H of the poison word + stream sync (not capture-safe; call outside capture)."""
        v = C.c_uint32()
        rc = self.lib.sm_poison_fast_v1(
            self.ctx, stream if stream is not None else C.c_void_p(None), C.byref(v)
        )
        if rc != OK:
            raise RuntimeError(f"sm_poison_fast rc={rc}")
        return v.value

    def close(self):
        """L-5: free every binding block, then P/modules/context. After a device sync, before comm teardown. Idempotent."""
        if self.ctx:
            for b in self.blks:
                self.lib.sm_blk_free_v1(b)
            self.blks.clear()
            self.lib.sm_free_v1(self.ctx)
            self.ctx = None

    def bind(self, sb: SmBindingV1):
        blk = C.c_void_p()
        rc = self.lib.sm_bind_new_v1(self.ctx, C.byref(sb), C.byref(blk))
        if rc != OK:
            raise RuntimeError(f"sm_bind_new rc={rc}")
        self.blks.append(blk)
        return blk

    def launch_v3(
        self,
        blk,
        A_ptr: int,
        W_ptr: int,
        strideA: int,
        Ascale_ptr: int,
        positions_ptr: int,
        slot_ptr: int,
        t_valid: int,
        call_id: int,
        stream,
    ):
        """R12 launch (card D2.6): x_scale / positions / slot_mapping / t_valid / call_id are PER-LAUNCH kernargs (they are no longer part of the binding), err and step
        come from the global page the driver was attached with."""
        rc = self.lib.sm_launch_blk_v3(
            self.ctx,
            blk,
            C.c_void_p(A_ptr),
            C.c_void_p(W_ptr),
            C.c_uint32(strideA),
            C.c_void_p(Ascale_ptr),
            C.c_void_p(positions_ptr),
            C.c_void_p(slot_ptr),
            C.c_uint32(t_valid),
            C.c_uint32(call_id),
            stream,
        )
        if rc != OK:
            raise RuntimeError(f"sm_launch_blk_v3 rc={rc}")

    def launch(self, blk, A_ptr: int, W_ptr: int, strideA: int, stream):
        rc = self.lib.sm_launch_blk_v1(
            self.ctx,
            blk,
            C.c_void_p(A_ptr),
            C.c_void_p(W_ptr),
            C.c_uint32(strideA),
            stream,
        )
        if rc != OK:
            raise RuntimeError(f"sm_launch_blk rc={rc}")
