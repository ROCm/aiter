# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ctypes runtime for the large-M driver (lg_fusion_driver.cpp = LG-QSDMA-r11 driver + attach). No torch import at module level."""

from __future__ import annotations

import ctypes as C

REPORT_FIELDS = (
    "epoch",
    "poison",
    "deferred_chunks",
    "credit_wait_ticks_100MHz",
    "inline_posts",
    "quiet_posts",
    "poison_reason",
    "epochs_done",
    "credit_miss",
    "lock_miss",
)
POLL_ITERATIONS = 2_000_000  # r11 bridge default


class _Va:  # raw device pointer -> torch via __cuda_array_interface__ (no copy, no ownership)
    def __init__(self, ptr, shape, typestr="<u2"):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
        }


# ---- R12 (design card r2 D2.4; large/r12, ABI >= 4) ------------------------------------------------------------------------------------------------------------
# LgCommArgsV3 = LgCommArgsV2 + {step @320, err @328, gpeers @336, call_id @344, pad @348}: 352 bytes. The driver builds it natively from the page attached with
# lg_gpage_attach_v1; Python passes call_id per launch. LgGfPeers = {u32* err[4]}: device-resident table of the 4 ranks' ERR addresses (own entry ignored).
LG_COMM_ARGS_V3_BYTES = 352
LG_COMM_ABI_VERSION_V3 = 4
LG_GF_PEERS_BYTES = 32
LG_CTAG_OFF, LG_ATAG_OFF = (
    3072,
    3104,
)  # in the C window; reset by lg_init_control_v1 (zeroes the whole C image)
CALL_ID_LIMIT = 1 << 31


def load_driver(path: str, abi: int = 2):
    """Bind the large driver. abi 3 (r12 build, LG_COMM_ABI_VERSION >= 4): lg_launch_v1 / lg_poison_v1 are compiled out; lg_gpage_attach_v1 attaches the per-comm ERR/STEP
    page once after lg_attach_v1; lg_launch_v2 = GEMM + quiet_ready (LG_QR_WGS workgroups; the best-effort NaN fill under ERR is folded into it: there is NO separate fill node and the runtime must NOT add one); lg_abi_info_v2 reports the ABI (and the host
    build's LG_QR_WGS) for the runtime's static checks."""
    lib = C.CDLL(path)
    vp, u32, u64, i32 = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32
    if abi == 3:
        lib.lg_attach_v1.restype = vp
        lib.lg_attach_v1.argtypes = [
            u32,
            u32,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            C.c_char_p,
            C.c_char_p,
            u32,
            C.POINTER(u64),
        ]
        lib.lg_gpage_attach_v1.restype = i32
        lib.lg_gpage_attach_v1.argtypes = [vp, vp, vp, vp]
        lib.lg_free_v1.restype = i32
        lib.lg_free_v1.argtypes = [vp]
        lib.lg_window_bytes_v1.restype = u64
        lib.lg_window_bytes_v1.argtypes = [u32]
        lib.lg_info_v1.restype = i32
        lib.lg_info_v1.argtypes = [vp]
        lib.lg_abi_info_v2.restype = i32
        lib.lg_abi_info_v2.argtypes = [C.POINTER(u64)]
        lib.lg_comm_R_ptr.restype = vp
        lib.lg_comm_R_ptr.argtypes = [vp]
        lib.lg_init_control_v1.restype = i32
        lib.lg_init_control_v1.argtypes = [vp]
        lib.lg_launch_v2.restype = i32
        lib.lg_launch_v2.argtypes = [
            vp,
            vp,
            vp,
            vp,
            vp,
            u32,
            u64,
            u64,
            u32,
            u32,
            u32,
            vp,
        ]
        lib.lg_report_v1.restype = i32
        lib.lg_report_v1.argtypes = [vp, C.POINTER(u64)]
        lib.lg_drain_v1.restype = i32
        lib.lg_drain_v1.argtypes = [vp, vp]
        return lib
    lib.lg_attach_v1.restype = vp
    lib.lg_attach_v1.argtypes = [
        u32,
        u32,
        vp,
        vp,
        vp,
        vp,
        vp,
        vp,
        vp,
        vp,
        C.c_char_p,
        C.c_char_p,
        u32,
        C.POINTER(u64),
    ]
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
    # device-scoped wrappers (restore the caller's current device); the bare Phase-1 lg_comm_* entry points are not called from Python
    lib.lg_init_control_v1.restype = i32
    lib.lg_init_control_v1.argtypes = [vp]
    lib.lg_launch_v1.restype = i32
    lib.lg_launch_v1.argtypes = [vp, vp, vp, vp, vp, u32, u64, u64, u32, u32, vp]
    lib.lg_report_v1.restype = i32
    lib.lg_report_v1.argtypes = [vp, C.POINTER(u64)]
    lib.lg_drain_v1.restype = i32
    lib.lg_drain_v1.argtypes = [vp, vp]
    return lib


ATTACH_STAGES = {
    1: "args",
    2: "image_hash",
    3: "hipSetDevice",
    4: "topology",
    5: "sdma_queues",
    6: "hipModuleLoad",
    7: "entry_resolve",
}
ATTACH_ENTRIES = ("n1", "n2", "n4", "n2h", "n4h", "n3q", "quiet_ready", "drain_all")


def decode_attach_diag(d) -> str:
    """Human-readable reason for a failed lg_attach_v1 from its diag[8] output (see lg_fusion_driver.cpp D5)."""
    st = int(d[0])
    name = ATTACH_STAGES.get(st, f"unknown({st})")
    s = lambda v: (v - (1 << 64) if v >= (1 << 63) else v)
    if st == 1:
        names = (
            "comm",
            "devcomm",
            "S",
            "R",
            "C",
            "Sva",
            "Rva",
            "Cva",
            "hsaco_path",
            "sha",
        )
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


def abi_info(lib) -> dict:
    """lg_abi_info_v2: {abi, args_bytes, err_off, step_off, diag_off, ctag_off, atag_off, peers_bytes, qr_wgs} of an ABI >= 4 build (r12d: out[8] = the HOST build's
    LG_QR_WGS, the quiet_ready workgroups its launcher starts)."""
    b = (C.c_uint64 * 9)()
    if lib.lg_abi_info_v2(b) != 0:
        raise RuntimeError("lg_abi_info")
    return dict(
        zip(
            (
                "abi",
                "args_bytes",
                "err_off",
                "step_off",
                "diag_off",
                "ctag_off",
                "atag_off",
                "peers_bytes",
                "qr_wgs",
            ),
            list(b),
        )
    )


def check_abi_info(got: dict, qr_wgs: int) -> None:
    """The runtime's static check of the driver against this module's layout constants and the global page layout (errpage.py), and of the host library's LG_QR_WGS
    against `qr_wgs`, the value the DEVICE image was built with: the launcher starts the quiet_ready workgroups, the image's fill / reset logic assumes the count, and the
    device refuses a mismatching launch (CONFIG), so a stale or hand-built pair is refused here, at attach, with both numbers.
    """
    from aiter.gemm_fusion import errpage

    want = {
        "abi": LG_COMM_ABI_VERSION_V3,
        "args_bytes": LG_COMM_ARGS_V3_BYTES,
        "err_off": errpage.ERR_OFF,
        "step_off": errpage.STEP_OFF,
        "diag_off": errpage.DIAG_OFF,
        "ctag_off": LG_CTAG_OFF,
        "atag_off": LG_ATAG_OFF,
        "peers_bytes": LG_GF_PEERS_BYTES,
    }
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise RuntimeError(f"large driver ABI mismatch (got, expected): {bad}")
    if got.get("qr_wgs") != qr_wgs:
        raise RuntimeError(
            f"large driver qr_wgs mismatch: the host library starts {got.get('qr_wgs')} quiet_ready workgroups but the device image was built with LG_QR_WGS={qr_wgs}"
        )


class LargeContext:
    def __init__(self, lib, h, M, F, rank):
        self.lib, self.h, self.M, self.F, self.rank = lib, h, M, F, rank
        self.budget_us = 0  # per-wait poll budget (us) passed to every launch; 0 = the kernel default (50 ms). Set by the plan from FusionResources.wait_budget_us.

    def init_control(self):
        rc = self.lib.lg_init_control_v1(self.h)
        if rc != 0:
            raise RuntimeError(f"lg_comm_init_control rc={rc}")

    def launch(
        self, A, W, As, Ws, strideA, puts_per_peer, half_first, stream, budget_us=None
    ):
        # budget_us is a kernel ARGUMENT baked into a captured graph's argument block at capture time (see resources.resolve_wait_budget_us)
        rc = self.lib.lg_launch_v1(
            self.h,
            C.c_void_p(A),
            C.c_void_p(W),
            C.c_void_p(As),
            C.c_void_p(Ws),
            strideA,
            POLL_ITERATIONS,
            self.budget_us if budget_us is None else budget_us,
            puts_per_peer,
            half_first,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"lg_comm_launch rc={rc}")

    def gpage_attach(self, err_addr: int, step_addr: int, gpeers_addr: int | None):
        """Attach the per-comm ERR/STEP page once after lg_attach_v1 (device addresses; the peers table may be 0/None: no ERR propagation)."""
        rc = self.lib.lg_gpage_attach_v1(
            self.h,
            C.c_void_p(err_addr),
            C.c_void_p(step_addr),
            C.c_void_p(gpeers_addr or 0),
        )
        if rc != 0:
            raise RuntimeError(f"lg_gpage_attach rc={rc}")

    def launch_v3(
        self,
        A,
        W,
        As,
        Ws,
        strideA,
        puts_per_peer,
        half_first,
        call_id,
        stream,
        budget_us=None,
    ):
        """R12 launch = GEMM + quiet_ready (all in the driver, 2 graph nodes; quiet_ready includes the best-effort NaN fill under ERR; capturable, no sync/alloc). call_id is the 31-bit decision key of the tag
        E = (STEP << 32) | call_id; the driver refuses call_id >= 2^31 (-104) and a launch without an attached page (-110).
        """
        if not 0 <= call_id < CALL_ID_LIMIT:
            raise ValueError(f"call_id {call_id} must be in [0, 2^31)")
        rc = self.lib.lg_launch_v2(
            self.h,
            C.c_void_p(A),
            C.c_void_p(W),
            C.c_void_p(As),
            C.c_void_p(Ws),
            strideA,
            POLL_ITERATIONS,
            self.budget_us if budget_us is None else budget_us,
            puts_per_peer,
            half_first,
            call_id,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"lg_launch_v2 rc={rc}")

    def report(self) -> dict:
        out = (C.c_uint64 * 10)()
        if self.lib.lg_report_v1(self.h, out) != 0:
            raise RuntimeError("lg_comm_read_report")
        return dict(zip(REPORT_FIELDS, list(out)))

    def drain_all(self, stream):
        rc = self.lib.lg_drain_v1(self.h, stream)
        if rc != 0:
            raise RuntimeError(f"lg_comm_drain_all rc={rc}")

    def recv_ptr(self) -> int:
        return int(self.lib.lg_comm_R_ptr(self.h))

    def poison(self, stream=None) -> int:
        """Cheap per-step host check: 4-byte D2H of the poison word on `stream` (null = default stream) + that stream's sync.
        NOT capture-safe (host sync): call outside graph capture, e.g. once per model step.
        """
        v = C.c_uint32()
        rc = self.lib.lg_poison_v1(
            self.h, stream if stream is not None else C.c_void_p(None), C.byref(v)
        )
        if rc != 0:
            raise RuntimeError(f"lg_poison rc={rc}")
        return int(v.value)

    def close(self):
        """L-5: unload the module and free the context (after a device sync, before comm teardown). Idempotent."""
        if self.h:
            self.lib.lg_free_v1(self.h)
            self.h = None
