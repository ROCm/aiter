# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""R12 host API pieces (design card r2, section R2 / D2.1, D2.6, D-7): the per-comm ERR/STEP/DIAG page, the ERR word decoder, zero-copy device views of the page
and the async pinned readback ring. No torch import at module level (torch is used lazily by the default ring ops / views).

Global page ("gf/global"), one per comm, one 4 MiB window (arena: a view; uniform: one more equal-size window), registered by FusionResources before finalize():

  offset 0   u32  ERR   0 = ok. Else reason[7:0] | origin_rank[9:8] | kind[11:10] (1 small, 2 large) | propagated[12]. The first CAS wins.
  offset 8   u64  STEP  rank-uniform forward counter; ATOM is the only writer (add_(1) per forward).
  offset 64  u64[8] DIAG written once by the CAS winner: expected tag, seen tag, flag/arrival value, plan code. Logging only.

Peer addressing for ERR propagation (the winner CASes the 3 peers' ERR words) is done by the DRIVER at attach time from the page's local address (ccoGetPeerPtr, the U2
pattern of the existing drivers); the Python side hands the driver the page's local address and never needs peer pointers.
"""

from __future__ import annotations

from dataclasses import dataclass

GLOBAL_PAGE_BYTES = 4 << 20
SHARE_GLOBAL = "gf.global"  # the share tag every R12 plan's page request carries (one page per comm)
ERR_OFF, STEP_OFF, DIAG_OFF, DIAG_WORDS = 0, 8, 64, 8
READBACK_SLOTS = 32  # R >= 2 (card D-7: e.g. 32); slot = step % R, 4 B each

KIND_SMALL, KIND_LARGE = 1, 2
KIND_NAMES = {0: "none", KIND_SMALL: "small", KIND_LARGE: "large"}
# reason codes (low 8 bits). Small (card D2.3 / D2.5): B-WG poll timeout 0x20 (overshoot is the same reason with its own flag in the card: 0x20|0x100), tag mismatch 0x30,
# contract failures 0x40..0x4f. Large (lg_comm_abi_v3.h, large/r12): the r11 poison codes 1..6 are kept, 8 = launch without an attached page (mirror only), 0x31 / 0x32 =
# credit / arrival tag != E. A test pins this table to the header text.
SMALL_REASONS = {0x20: "B_POLL_TIMEOUT", 0x30: "TAG_MISMATCH"}
LARGE_REASONS = {
    1: "W1_TIMEOUT(credit wait: a peer never began this epoch)",
    2: "W2_TIMEOUT(arrival wait: a peer's puts did not complete)",
    3: "EPOCH_MISMATCH",
    4: "DUP_CHUNK",
    5: "CONFIG(host/device build or plan mismatch)",
    6: "POSTER_WAIT(local GEMM group wait timed out)",
    8: "NO_GPAGE(launch without an attached ERR/STEP page)",
    0x31: "TAG_CREDIT(credit tag != expected E)",
    0x32: "TAG_ARRIVAL(arrival tag != expected E)",
}


@dataclass(frozen=True)
class ErrInfo:
    word: int
    reason: int
    origin_rank: int
    kind: int
    propagated: bool

    @property
    def ok(self) -> bool:
        return self.word == 0

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, f"kind{self.kind}")

    @property
    def reason_name(self) -> str:
        if self.kind == KIND_LARGE:
            return LARGE_REASONS.get(self.reason, f"reason{self.reason:#x}")
        if 0x40 <= self.reason <= 0x4F:
            return f"CONTRACT({self.reason:#x})"
        return SMALL_REASONS.get(self.reason, f"reason{self.reason:#x}")

    def describe(self) -> str:
        if self.ok:
            return "ok"
        return f"{self.kind_name}:{self.reason_name} origin_rank={self.origin_rank} propagated={int(self.propagated)} word={self.word:#x}"


def encode_err(
    reason: int, origin_rank: int, kind: int, propagated: bool = False
) -> int:
    if not (0 <= reason < 256 and 0 <= origin_rank < 4 and 0 <= kind < 4):
        raise ValueError(
            f"err fields out of range: reason={reason} origin_rank={origin_rank} kind={kind}"
        )
    return reason | origin_rank << 8 | kind << 10 | int(bool(propagated)) << 12


def decode_err(word: int) -> ErrInfo:
    w = int(word) & 0xFFFFFFFF
    return ErrInfo(w, w & 0xFF, (w >> 8) & 3, (w >> 10) & 3, bool((w >> 12) & 1))


class DevView:
    """A raw device address as a torch-convertible object (__cuda_array_interface__): no copy, no ownership."""

    def __init__(self, ptr: int, shape, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
        }


class GlobalPage:
    """The ERR/STEP/DIAG page of one comm: addresses, and cached zero-copy device tensors (the SAME tensor objects on every call: ATOM captures the step tick)."""

    def __init__(self, local: int, nbytes: int, device: int):
        if nbytes < DIAG_OFF + 8 * DIAG_WORDS:
            raise ValueError(f"global page of {nbytes} B is smaller than its layout")
        self.local, self.bytes, self.device = int(local), int(nbytes), int(device)
        self._t: dict[str, object] = {}

    @property
    def err_addr(self) -> int:
        return self.local + ERR_OFF

    @property
    def step_addr(self) -> int:
        return self.local + STEP_OFF

    @property
    def diag_addr(self) -> int:
        return self.local + DIAG_OFF

    def _tensor(self, name: str, addr: int, shape, typestr: str):
        t = self._t.get(name)
        if t is None:
            import torch

            t = self._t[name] = torch.as_tensor(
                DevView(addr, shape, typestr), device=f"cuda:{self.device}"
            )
        return t

    def err_tensor(self):
        """int32 [1], a zero-copy view of the local ERR word."""
        return self._tensor("err", self.err_addr, (1,), "<i4")

    def step_tensor(self):
        """int64 [1], a zero-copy view of the STEP counter (ATOM ticks it once per forward with add_(1))."""
        return self._tensor("step", self.step_addr, (1,), "<i8")

    def diag_tensor(self):
        """int64 [8], a zero-copy view of DIAG (logging only)."""
        return self._tensor("diag", self.diag_addr, (DIAG_WORDS,), "<i8")

    def read(self) -> tuple[int, int, list[int]]:
        """(err word, step, diag words): D2H of 4 + 8 + 64 bytes. Needs the device to be quiet (the caller synchronises); NOT for the hot path."""
        err = int(self.err_tensor().cpu()[0]) & 0xFFFFFFFF
        step = int(self.step_tensor().cpu()[0])
        diag = [int(v) & 0xFFFFFFFFFFFFFFFF for v in self.diag_tensor().cpu().tolist()]
        return err, step, diag


class TorchRingOps:
    """Default device ops of the readback ring: a pinned int32 buffer, a 4 B non_blocking D2H copy on the stream, a recorded event."""

    def alloc(self, slots: int):
        import torch

        return torch.zeros(slots, dtype=torch.int32, pin_memory=True)

    def capturing(self) -> bool:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())

    def copy(self, buf, slot: int, src, stream):
        import torch

        s = stream if stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(s):
            buf[slot : slot + 1].copy_(src, non_blocking=True)  # 4 B D2H, no host sync
            ev = torch.cuda.Event()
            ev.record(s)
        return ev

    def read(self, buf, slot: int) -> int:
        return int(buf[slot])


class ReadbackRing:
    """Async pinned ring of the ERR word (card D-7). enqueue(step) issues ONE 4 B D2H into slot step % R in stream order (call it right before sampling, in the compute
    stream, eager: the slot index is a host value and must not be baked into a graph); consume(step) returns the value of THAT step's slot once its copy is complete.
    Neither call synchronises the host (consume(wait=True) is the explicit opt-in)."""

    def __init__(self, slots: int = READBACK_SLOTS, ops=None):
        if slots < 2:
            raise ValueError("the readback ring needs at least 2 slots")
        self.slots = int(slots)
        self.ops = ops if ops is not None else TorchRingOps()
        self.buf = self.ops.alloc(self.slots)
        self._events: list = [None] * self.slots
        self._steps = [-1] * self.slots
        self.enqueued = 0  # one 4 B D2H per enqueue: audited by the G2 trace gate

    def slot_of(self, step_idx: int) -> int:
        return int(step_idx) % self.slots

    def enqueue(self, step_idx: int, src, stream=None) -> int:
        if step_idx < 0:
            raise ValueError("step_idx must be >= 0")
        if self.ops.capturing():
            raise RuntimeError(
                "the ERR readback is a per-step host-indexed copy: never enqueue it inside graph capture"
            )
        slot = self.slot_of(step_idx)
        self._events[slot] = self.ops.copy(self.buf, slot, src, stream)
        self._steps[slot] = int(step_idx)
        self.enqueued += 1
        return slot

    def consume(self, step_idx: int, wait: bool = False) -> int | None:
        """The ERR word captured for `step_idx`, or None if its copy has not completed yet (never blocks unless wait=True). Raises if the slot no longer (or never)
        holds that step: more than R steps passed since the enqueue, or no enqueue happened.
        """
        slot = self.slot_of(step_idx)
        if self._steps[slot] != int(step_idx):
            raise RuntimeError(
                f"readback slot {slot} holds step {self._steps[slot]}, not {step_idx}: enqueued more than {self.slots} steps ago, or never enqueued"
            )
        ev = self._events[slot]
        if not ev.query():
            if not wait:
                return None
            ev.synchronize()
        return int(self.ops.read(self.buf, slot)) & 0xFFFFFFFF

    def reset(self):
        """Forget every pending slot (after the collective reset: the old readings describe a state that no longer exists)."""
        self._events = [None] * self.slots
        self._steps = [-1] * self.slots
