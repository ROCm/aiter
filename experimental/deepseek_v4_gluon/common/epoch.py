# SPDX-License-Identifier: MIT
"""The launch-pair epoch, as the FlyDSL mono layer keeps it
(``attention/back.py``: ``epoch_begin`` / ``epoch_end``; ``layer.py``).

The epoch buffer (i32 words): [0] the epoch e, [EPOCH_MARKS + b] CTA b's mark,
then the MoE's counter slots. A launch reads e once; its hand-offs carry
``2 e + 1`` (K1) or ``2 e + 2`` (K2 / the FFN launch); its peer half is
``(e & 1) * half``. Every CTA marks ``e + 1`` when it starts; the pair's last
launch's CTA 0 waits for every mark, resets what the next pairs must find
cleared, then stores ``e + 1``. Nothing is cleared per step: graph-safe."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .sync import BLOCKS, deadline, fail, store32

EPOCH_MARKS: gl.constexpr = gl.constexpr(4)

_WAIT_EQ = gl.constexpr("\n".join(
    [
        "s_mov_b32 $3, 0",
        "s_mov_b32 $4, 0",
        "v_cmp_ne_u32 vcc, 0, $8",
        "s_and_saveexec_b64 $1, vcc",
        "s_cbranch_execz .Lweq_done${:uid}",
        ".Lweq_retry${:uid}:",
        "global_load_dword $0, $6, off sc1",
        "s_waitcnt vmcnt(0)",
        "v_cmp_ne_u32 vcc, $7, $0",
        "s_cmp_eq_u64 vcc, 0",
        "s_cbranch_scc1 .Lweq_done${:uid}",
        "s_sleep 1",
        "s_add_u32 $3, $3, 1",
        "s_cmp_lt_u32 $3, 256",
        "s_cbranch_scc1 .Lweq_retry${:uid}",
        "s_mov_b32 $3, 0",
        "s_memrealtime $2",
        "s_waitcnt lgkmcnt(0)",
        "v_cmp_ge_u64 vcc, $2, $9",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc0 .Lweq_retry${:uid}",
        "s_mov_b32 $4, 1",
        ".Lweq_done${:uid}:",
        "s_mov_b64 exec, $1",
        "v_mov_b32 $5, $4",
    ]
))
_WAIT_EQ_C = gl.constexpr("=&v,=&s,=&s,=&s,=&s,=&v,v,v,v,v,~{memory},~{vcc},~{scc}")


@gluon.jit
def wait_eq(addr, want, active, dl):
    """Spin until the 32-bit word at ``addr`` equals ``want`` (device scope),
    bounded by ``dl``: (value, failed)."""
    out = gl.inline_asm_elementwise(
        _WAIT_EQ,
        _WAIT_EQ_C,
        [addr.to(gl.uint64), want.to(gl.uint32), active.to(gl.uint32), dl],
        (gl.uint32, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint32),
        False,
        1,
    )
    return out[0], out[5]


@gluon.jit
def load_epoch(epoch):
    """The launch pair's epoch (word 0 of the buffer at address ``epoch``)."""
    return gl.load(epoch.to(gl.pointer_type(gl.int32)))


@gluon.jit
def epoch_begin(epoch, ep, bid, tid):
    """This CTA's mark := ep + 1 (its own word: no contended atomic)."""
    base = epoch.to(gl.uint64) + (EPOCH_MARKS + bid).to(gl.uint64) * 4
    addr = base + gl.zeros_like(tid).to(gl.uint64)
    return store32(addr, ep + 1 + gl.zeros_like(tid), tid == 0)


@gluon.jit
def epoch_end(epoch, ep, bid, tid, health, rank,
              RESET_BASE: gl.constexpr = -1, RESET_SLOTS: gl.constexpr = 0):  # fmt: skip
    """CTA 0, once every CTA marked this epoch (thread u checks CTA u's mark):
    zero counter slot ``(ep + 128) & 255`` of each of ``RESET_SLOTS`` regions of
    256 words from word ``RESET_BASE`` (the MoE's ug queue / down count: their
    last use long done, their next far off), then the epoch := ep + 1. A mark
    missing past the wait bound sets the health word instead and the epoch
    stays (M3's fail-stop)."""
    if bid == 0:
        mine = tid < BLOCKS
        markp = epoch.to(gl.uint64) + (EPOCH_MARKS + tid).to(gl.uint64) * 4
        _, failed = wait_eq(markp, ep + 1 + gl.zeros_like(tid), mine, deadline(tid))
        fail(health, failed, rank, 100, mine)
        gl.barrier()
        if RESET_BASE >= 0:
            slot = (ep + 128) & 255
            for r in gl.static_range(RESET_SLOTS):
                at = epoch.to(gl.uint64) + ((RESET_BASE + r * 256) + slot) * 4
                store32(at + gl.zeros_like(tid).to(gl.uint64), gl.zeros_like(tid),
                        tid == 0)  # fmt: skip
        if gl.max(failed, axis=0) == 0:
            gl.store(epoch.to(gl.pointer_type(gl.int32)), ep + 1)
