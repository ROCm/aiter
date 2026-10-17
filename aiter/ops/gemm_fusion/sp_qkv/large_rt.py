# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ctypes runtime for the large-M driver (lg_fusion_driver.cpp = LG-QSDMA-r11 driver + attach). No torch import at module level."""

from __future__ import annotations

import ctypes as C

REPORT_FIELDS = ("epoch", "poison", "deferred_chunks", "credit_wait_ticks_100MHz", "inline_posts", "quiet_posts",
                 "poison_reason", "epochs_done", "credit_miss", "lock_miss")
POLL_ITERATIONS = 2_000_000  # r11 bridge default


class _Va:  # raw device pointer -> torch via __cuda_array_interface__ (no copy, no ownership)
    def __init__(self, ptr, shape, typestr="<u2"):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr, "data": (int(ptr), False), "version": 3, "strides": None}


def load_driver(path: str):
    lib = C.CDLL(path)
    vp, u32, u64, i32 = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32
    lib.lg_attach_v1.restype = vp
    lib.lg_attach_v1.argtypes = [u32, u32, vp, vp, vp, vp, vp, vp, vp, vp, C.c_char_p, C.c_char_p, u32, C.POINTER(u64)]
    lib.lg_free_v1.restype = i32
    lib.lg_free_v1.argtypes = [vp]
    lib.lg_poison_v1.restype = i32
    lib.lg_poison_v1.argtypes = [vp, vp, C.POINTER(u32)]
    lib.lg_window_bytes_v1.restype = u64
    lib.lg_window_bytes_v1.argtypes = [u32]
    lib.lg_info_v1.restype = i32
    lib.lg_info_v1.argtypes = [vp]
    lib.lg_comm_R_ptr.restype = vp
    lib.lg_comm_R_ptr.argtypes = [vp]
    lib.lg_comm_init_control.restype = i32
    lib.lg_comm_init_control.argtypes = [vp]
    lib.lg_comm_launch.restype = i32
    lib.lg_comm_launch.argtypes = [vp, vp, vp, vp, vp, u32, u64, u64, u32, u32, vp]
    lib.lg_comm_read_report.restype = i32
    lib.lg_comm_read_report.argtypes = [vp, C.POINTER(u64)]
    lib.lg_comm_drain_all.restype = i32
    lib.lg_comm_drain_all.argtypes = [vp, vp]
    return lib


ATTACH_STAGES = {1: "args", 2: "image_hash", 3: "hipSetDevice", 4: "topology", 5: "sdma_queues", 6: "hipModuleLoad", 7: "entry_resolve"}
ATTACH_ENTRIES = ("n1", "n2", "n4", "n2h", "n4h", "n3q", "quiet_ready", "drain_all")


def decode_attach_diag(d) -> str:
    """Human-readable reason for a failed lg_attach_v1 from its diag[8] output (see lg_fusion_driver.cpp D5)."""
    st = int(d[0])
    name = ATTACH_STAGES.get(st, f"unknown({st})")
    s = lambda v: v - (1 << 64) if v >= (1 << 63) else v  # noqa: E731  (values are signed ints stored in u64)
    if st == 1:
        names = ("comm", "devcomm", "S", "R", "C", "Sva", "Rva", "Cva", "hsaco_path", "sha")
        nulls = [n for i, n in enumerate(names) if int(d[1]) >> i & 1]
        return f"stage={name}: NULL args {nulls or 'none (range check)'}, rank={int(d[2])} M={int(d[3])} Q={int(d[4])} (need rank<=3, M%256==0, M<=8192, Q==1)"
    if st == 3 or st == 6:
        return f"stage={name}: hipError={int(d[1])}"
    if st == 4:
        return f"stage={name}: world={s(int(d[1]))} lsaSize={s(int(d[2]))} lsaRank={s(int(d[3]))} myNodeStart={s(int(d[4]))} (need 4/4/rank/0)"
    if st == 5:
        return f"stage={name}: devcomm sdmaNumQueue={s(int(d[1]))} but the r11 kernels need {int(d[2])}"
    if st == 7:
        i = int(d[2])
        return f"stage={name}: hipError={int(d[1])} entry #{i} ({ATTACH_ENTRIES[i] if i < len(ATTACH_ENTRIES) else '?'})"
    return f"stage={name}"


def info(lib):
    b = (C.c_uint64 * 6)()
    if lib.lg_info_v1(b) != 0:
        raise RuntimeError("lg_info")
    return dict(zip(("f", "n", "indexer", "fold", "r_pad", "devcomm_bytes"), list(b)))


class LargeContext:
    def __init__(self, lib, h, M, F, rank):
        self.lib, self.h, self.M, self.F, self.rank = lib, h, M, F, rank

    def init_control(self):
        rc = self.lib.lg_comm_init_control(self.h)
        if rc != 0:
            raise RuntimeError(f"lg_comm_init_control rc={rc}")

    def launch(self, A, W, As, Ws, strideA, puts_per_peer, half_first, stream, budget_us=0):
        rc = self.lib.lg_comm_launch(self.h, C.c_void_p(A), C.c_void_p(W), C.c_void_p(As), C.c_void_p(Ws), strideA,
                                     POLL_ITERATIONS, budget_us, puts_per_peer, half_first, stream)
        if rc != 0:
            raise RuntimeError(f"lg_comm_launch rc={rc}")

    def report(self) -> dict:
        out = (C.c_uint64 * 10)()
        if self.lib.lg_comm_read_report(self.h, out) != 0:
            raise RuntimeError("lg_comm_read_report")
        return dict(zip(REPORT_FIELDS, list(out)))

    def drain_all(self, stream):
        rc = self.lib.lg_comm_drain_all(self.h, stream)
        if rc != 0:
            raise RuntimeError(f"lg_comm_drain_all rc={rc}")

    def recv_ptr(self) -> int:
        return int(self.lib.lg_comm_R_ptr(self.h))

    def poison(self, stream=None) -> int:
        """Cheap per-step host check: 4-byte D2H of the poison word on `stream` (null = default stream) + that stream's sync.
        NOT capture-safe (host sync): call outside graph capture, e.g. once per model step."""
        v = C.c_uint32()
        rc = self.lib.lg_poison_v1(self.h, stream if stream is not None else C.c_void_p(None), C.byref(v))
        if rc != 0:
            raise RuntimeError(f"lg_poison rc={rc}")
        return int(v.value)

    def close(self):
        """L-5: unload the module and free the context (after a device sync, before comm teardown). Idempotent."""
        if self.h:
            self.lib.lg_free_v1(self.h)
            self.h = None
