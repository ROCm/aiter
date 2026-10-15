# SPDX-License-Identifier: MIT
"""gfx950 coherent polling primitives adapted from the K3 Gluon experiment.

Each 64-bit message carries its own epoch in the high word. Assembly is marked
side-effecting with a memory clobber so a compiler cannot cache a semaphore.
These ISA semantics require inspection on the selected Triton build.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


def _peer_poll_asm():
    world = 4
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


POLL4_ASM = gl.constexpr(_peer_poll_asm())
POLL4_CONSTRAINTS = gl.constexpr(
    ",".join(["=&v"] * 5 + ["=&s"] * 4 + ["v"] * 6 + ["~{memory}", "~{vcc}", "~{scc}"])
)


@g.jit
def poll_local(ptr, epoch, active):
    result = gl.inline_asm_elementwise(
        "\n        s_mov_b32 $3, 0\n        s_mov_b32 $4, 0\n        v_cmp_ne_u32 vcc, 0, $7\n        s_and_saveexec_b64 $2, vcc\n        s_cbranch_execz .Llocal_done${:uid}\n        .Llocal_retry${:uid}:\n        global_load_dwordx2 $0, $5, off sc1\n        s_waitcnt vmcnt(0)\n        v_lshrrev_b64 $1, 32, $0\n        v_cmp_eq_u64 vcc, $6, $1\n        s_cmp_eq_u64 vcc, exec\n        s_cbranch_scc1 .Llocal_done${:uid}\n        s_add_u32 $3, $3, 1\n        s_cmp_lt_u32 $3, 100000\n        s_cbranch_scc1 .Llocal_retry${:uid}\n        s_mov_b32 $4, 1\n        .Llocal_done${:uid}:\n        s_mov_b64 exec, $2\n        ",
        constraints="=&v,=&v,=&s,=&s,=&s,v,v,v,~{memory},~{vcc},~{scc}",
        args=[ptr.to(gl.uint64), epoch.to(gl.uint64), active.to(gl.uint32)],
        dtype=(gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )
    return result[4]


@g.jit
def poll_peers4(local, pair, epoch, active, STRIDE: gl.constexpr):
    base = local.to(gl.uint64) + pair.to(gl.uint64) * 8
    result = gl.inline_asm_elementwise(
        POLL4_ASM,
        constraints=POLL4_CONSTRAINTS,
        args=[
            base,
            base + STRIDE * 8,
            base + STRIDE * 16,
            base + STRIDE * 24,
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
            gl.uint32,
            gl.uint32,
        ),
        is_pure=False,
        pack=1,
    )
    return result[0], result[1], result[2], result[3], result[8]
