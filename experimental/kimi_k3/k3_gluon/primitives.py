"""gfx950 publication primitives and wave-local tagged-payload polling.

Reference: FlyDSL 21a3d1e, k3/kernel.py:1481-1548 (moe_peer_reduce).
The tagged messages are self-contained aligned 64-bit words. There is no
separately stored payload requiring a release/acquire fence. System-scope
relaxed atomic stores lower to coherent *stores*, not remote RMW operations.
Polling is side-effecting ISA with a memory clobber: it cannot be hoisted.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


def _poll_asm(world):
    scratch, saved, pending, iteration, failed = range(world, world + 5)
    addr, tag, active = world + 5, 2 * world + 5, 2 * world + 6
    lines = [
        f"s_mov_b32 ${iteration}, 0",
        f"s_mov_b32 ${failed}, 0",
        f"v_cmp_ne_u32 vcc, 0, ${active}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        "s_cbranch_execz .Lrestore${:uid}",
        ".Lretry${:uid}:",
    ]
    for rank in range(world):
        lines.append(f"global_load_dwordx2 ${rank}, ${addr + rank}, off sc1")
    lines += ["s_waitcnt vmcnt(0)", f"s_mov_b64 ${pending}, 0"]
    for rank in range(world):
        lines += [
            f"v_lshrrev_b64 ${scratch}, 32, ${rank}",
            f"v_cmp_ne_u64 vcc, ${tag}, ${scratch}",
            f"s_or_b64 ${pending}, ${pending}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pending}, 0",
        "s_cbranch_scc1 .Lrestore${:uid}",
        f"s_add_u32 ${iteration}, ${iteration}, 1",
        f"s_cmp_lt_u32 ${iteration}, 100000",
        "s_cbranch_scc1 .Lretry${:uid}",
        f"s_mov_b32 ${failed}, 1",
        ".Lrestore${:uid}:",
        f"s_mov_b64 exec, ${saved}",
    ]
    return "\n".join(lines)


POLL8_ASM = gl.constexpr(_poll_asm(8))
POLL8_CONSTRAINTS = gl.constexpr(
    ",".join(["=&v"] * 9 + ["=&s"] * 4 + ["v"] * 10 + ["~{memory}", "~{vcc}", "~{scc}"])
)


@g.jit
def poll8(local, pair, epoch, active, STRIDE: gl.constexpr):
    base = local.to(gl.uint64) + pair.to(gl.uint64) * 8
    values = gl.inline_asm_elementwise(
        POLL8_ASM,
        constraints=POLL8_CONSTRAINTS,
        args=[
            base,
            base + STRIDE * 8,
            base + STRIDE * 16,
            base + STRIDE * 24,
            base + STRIDE * 32,
            base + STRIDE * 40,
            base + STRIDE * 48,
            base + STRIDE * 56,
            epoch.to(gl.uint64),
            active.to(gl.uint32),
        ],
        dtype=(
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint64,
            gl.uint32,
            gl.uint32,
        ),
        is_pure=False,
        pack=1,
    )
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[12],
    )


@g.jit
def publish_sys(ptr, payload, epoch, mask=None):
    message = payload.to(gl.uint32).to(gl.uint64) | (epoch.to(gl.uint64) << 32)
    gl.atomic_store(ptr, message, mask=mask, sem="relaxed", scope="sys")


@g.jit
def bf16_sum8(a, b, c, d, e, f, h, i):
    lo = gl.full(a.shape, 0.0, gl.float32, a.type.layout)
    hi = gl.full(a.shape, 0.0, gl.float32, a.type.layout)
    words = (a, b, c, d, e, f, h, i)
    for index in gl.static_range(8):
        bits = words[index].to(gl.uint32)
        lo = lo + (bits << 16).to(gl.float32, bitcast=True)
        hi = hi + (bits & 0xFFFF0000).to(gl.float32, bitcast=True)
    return lo, hi


@g.jit
def pack_bf16(lo, hi):
    return lo.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
        hi.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )


@g.jit
def _poll_local_attempt(ptr, epoch, active):
    """Coherent, bounded wave-local polling of a self-contained GPU message."""
    result = gl.inline_asm_elementwise(
        """
        s_mov_b32 $3, 0
        s_mov_b32 $4, 0
        v_cmp_ne_u32 vcc, 0, $7
        s_and_saveexec_b64 $2, vcc
        s_cbranch_execz .Llocal_done${:uid}
        .Llocal_retry${:uid}:
        global_load_dwordx2 $0, $5, off sc1
        s_waitcnt vmcnt(0)
        v_lshrrev_b64 $1, 32, $0
        v_cmp_eq_u64 vcc, $6, $1
        s_cmp_eq_u64 vcc, exec
        s_cbranch_scc1 .Llocal_done${:uid}
        s_add_u32 $3, $3, 1
        s_cmp_lt_u32 $3, 100000
        s_cbranch_scc1 .Llocal_retry${:uid}
        s_mov_b32 $4, 1
        .Llocal_done${:uid}:
        s_mov_b64 exec, $2
        """,
        constraints="=&v,=&v,=&s,=&s,=&s,v,v,v,~{memory},~{vcc},~{scc}",
        args=[ptr.to(gl.uint64), epoch.to(gl.uint64), active.to(gl.uint32)],
        dtype=(gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )
    return result[0].to(gl.uint32), result[4]


@g.jit
def publish_local(ptr, bits, epoch, mask=None):
    word = bits.to(gl.uint32).to(gl.uint64) | (epoch.to(gl.uint64) << 32)
    gl.atomic_store(ptr, word, mask=mask, sem="relaxed", scope="gpu")


POLL4_ASM = gl.constexpr(_poll_asm(4))
POLL4_CONSTRAINTS = gl.constexpr(
    ",".join(["=&v"] * 5 + ["=&s"] * 4 + ["v"] * 6 + ["~{memory}", "~{vcc}", "~{scc}"])
)


@g.jit
def _poll_peers_attempt(
    local, pair, epoch, active, STRIDE: gl.constexpr, WORLD: gl.constexpr
):
    gl.static_assert(WORLD == 4 or WORLD == 8)
    if WORLD == 8:
        a, b, c, d, e, f, h, i, failed = poll8(local, pair, epoch, active, STRIDE)
        words = (a, b, c, d, e, f, h, i)
    else:
        base = local.to(gl.uint64) + pair.to(gl.uint64) * 8
        values = gl.inline_asm_elementwise(
            POLL4_ASM,
            POLL4_CONSTRAINTS,
            [
                base,
                base + STRIDE * 8,
                base + STRIDE * 16,
                base + STRIDE * 24,
                epoch.to(gl.uint64),
                active.to(gl.uint32),
            ],
            (
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint32,
                gl.uint32,
            ),
            False,
            1,
        )
        words = (values[0], values[1], values[2], values[3])
        failed = values[8]
    return words, failed


@g.jit
def wall_clock():
    # The retry condition is CTA-wide. Make its timestamp CTA-uniform too;
    # independent wave timestamps must not diverge around reduction barriers.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lanes = gl.arange(0, 512, layout=layout)
    ticks = gl.inline_asm_elementwise(
        "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
        "=s,v,~{memory}",
        [lanes],
        gl.uint64,
        False,
        1,
    )
    return gl.max(ticks, 0)


def _timed_asm(world):
    scratch, saved, pending, iteration, failed, now = range(world, world + 6)
    addr, tag, active, deadline = world + 6, 2 * world + 6, 2 * world + 7, 2 * world + 8
    lines = [
        f"s_mov_b32 ${iteration}, 0",
        f"s_mov_b32 ${failed}, 0",
        f"v_cmp_ne_u32 vcc, 0, ${active}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        "s_cbranch_execz .Ltimed_done${:uid}",
        ".Ltimed_retry${:uid}:",
    ]
    for rank in range(world):
        lines.append(f"global_load_dwordx2 ${rank}, ${addr + rank}, off sc1")
    lines += ["s_waitcnt vmcnt(0)", f"s_mov_b64 ${pending}, 0"]
    for rank in range(world):
        lines += [
            f"v_lshrrev_b64 ${scratch}, 32, ${rank}",
            f"v_cmp_ne_u64 vcc, ${tag}, ${scratch}",
            f"s_or_b64 ${pending}, ${pending}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pending}, 0",
        "s_cbranch_scc1 .Ltimed_done${:uid}",
        f"s_add_u32 ${iteration}, ${iteration}, 1",
        f"s_cmp_lt_u32 ${iteration}, 100000",
        "s_cbranch_scc1 .Ltimed_retry${:uid}",
        f"s_memrealtime ${now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_le_u64 vcc, ${deadline}, ${now}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Ltimed_fail${:uid}",
        f"s_mov_b32 ${iteration}, 0",
        "s_branch .Ltimed_retry${:uid}",
        ".Ltimed_fail${:uid}:",
        f"s_mov_b32 ${failed}, 1",
        ".Ltimed_done${:uid}:",
        f"s_mov_b64 exec, ${saved}",
    ]
    return "\n".join(lines)


TIMED1 = gl.constexpr(_timed_asm(1))
TIMED4 = gl.constexpr(_timed_asm(4))
TIMED8 = gl.constexpr(_timed_asm(8))
CONSTRAINTS1 = gl.constexpr(
    ",".join(["=&v"] * 2 + ["=&s"] * 5 + ["v"] * 4 + ["~{memory}", "~{vcc}", "~{scc}"])
)
CONSTRAINTS4 = gl.constexpr(
    ",".join(["=&v"] * 5 + ["=&s"] * 5 + ["v"] * 7 + ["~{memory}", "~{vcc}", "~{scc}"])
)
CONSTRAINTS8 = gl.constexpr(
    ",".join(["=&v"] * 9 + ["=&s"] * 5 + ["v"] * 11 + ["~{memory}", "~{vcc}", "~{scc}"])
)


@g.jit
def wave_deadline(ptr, WATCHDOG_TICKS: gl.constexpr):
    # Per-wave register values: no CTA reduction, LDS, or barriers in polling.
    now = gl.inline_asm_elementwise(
        "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
        "=s,v,~{memory}",
        [ptr.to(gl.uint64)],
        gl.uint64,
        False,
        1,
    )
    return now + WATCHDOG_TICKS


@g.jit
def poll_local(ptr, epoch, active, WATCHDOG_TICKS: gl.constexpr = 0):
    if WATCHDOG_TICKS:
        result = gl.inline_asm_elementwise(
            TIMED1,
            CONSTRAINTS1,
            [
                ptr.to(gl.uint64),
                epoch.to(gl.uint64),
                active.to(gl.uint32),
                wave_deadline(ptr, WATCHDOG_TICKS),
            ],
            (
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint64,
                gl.uint32,
                gl.uint32,
                gl.uint64,
            ),
            False,
            1,
        )
        bits, failed = result[0].to(gl.uint32), result[5]
    else:
        bits, failed = _poll_local_attempt(ptr, epoch, active)
    return bits, failed


@g.jit
def poll_peers(
    local,
    pair,
    epoch,
    active,
    STRIDE: gl.constexpr,
    WORLD: gl.constexpr,
    WATCHDOG_TICKS: gl.constexpr = 0,
):
    if WATCHDOG_TICKS:
        base = local.to(gl.uint64) + pair.to(gl.uint64) * 8
        deadline = wave_deadline(base, WATCHDOG_TICKS)
        if WORLD == 8:
            result = gl.inline_asm_elementwise(
                TIMED8,
                CONSTRAINTS8,
                [
                    base,
                    base + STRIDE * 8,
                    base + STRIDE * 16,
                    base + STRIDE * 24,
                    base + STRIDE * 32,
                    base + STRIDE * 40,
                    base + STRIDE * 48,
                    base + STRIDE * 56,
                    epoch.to(gl.uint64),
                    active.to(gl.uint32),
                    deadline,
                ],
                (
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint32,
                    gl.uint32,
                    gl.uint64,
                ),
                False,
                1,
            )
            words = (
                result[0],
                result[1],
                result[2],
                result[3],
                result[4],
                result[5],
                result[6],
                result[7],
            )
            failed = result[12]
        else:
            gl.static_assert(WORLD == 4)
            result = gl.inline_asm_elementwise(
                TIMED4,
                CONSTRAINTS4,
                [
                    base,
                    base + STRIDE * 8,
                    base + STRIDE * 16,
                    base + STRIDE * 24,
                    epoch.to(gl.uint64),
                    active.to(gl.uint32),
                    deadline,
                ],
                (
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint64,
                    gl.uint32,
                    gl.uint32,
                    gl.uint64,
                ),
                False,
                1,
            )
            words = (result[0], result[1], result[2], result[3])
            failed = result[8]
    else:
        words, failed = _poll_peers_attempt(local, pair, epoch, active, STRIDE, WORLD)
    return words, failed


@g.jit
def bf16_sum_peers(words):
    lo = gl.full(words[0].shape, 0.0, gl.float32, words[0].type.layout)
    hi = gl.full(words[0].shape, 0.0, gl.float32, words[0].type.layout)
    for index in gl.static_range(len(words)):
        bits = words[index].to(gl.uint32)
        lo += (bits << 16).to(gl.float32, bitcast=True)
        hi += (bits & 0xFFFF0000).to(gl.float32, bitcast=True)
    return lo, hi
