# SPDX-License-Identifier: MIT
"""Tagged mailbox hand-offs for the DS V4.x Gluon monokernels (gfx950).

The FlyDSL kernels' contract (``vllm/models/deepseek_v41/amd/mono/common/sync.py``),
in Gluon:

- A pair is one aligned 64-bit word, (value | tag << 32), stored at once: a reader
  that sees the tag sees the value. Device-scope hand-offs (this GPU's scratch,
  read by CTAs of other XCDs) store and load with ``sc1``; system-scope ones (the
  uncached peer heap) with ``sc0 sc1`` -- the FlyDSL ``CM_DEV`` / ``CM_SYS``.
- A poll re-loads every pair of a batch until all tags match, in side-effecting
  wave-local asm with a memory clobber (LLVM can neither hoist nor cache it).
  Inactive lanes are masked off through EXEC; their outputs are undefined.
- Every poll is bounded: each ``CHECK_EVERY`` rounds it reads the rank's sticky
  health word and the 100 MHz clock, and gives up once the word is set (another
  wait failed) or its deadline passed. ``failed``: 0 ok, 1 timed out, 2 health.
- Bulk data goes as plain scoped words, then ``publish``: every wave drains its
  stores (``s_waitcnt vmcnt(0)``: a workgroup release does not, outside tgsplit),
  a CTA barrier, then one thread's tagged pair.

Gluon in the stock vLLM image has no ``atomic_load`` / ``atomic_store``, so puts
are asm stores as well: their cache bits are exactly the FlyDSL ones.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# one element a thread: a CTA of 512 threads in 8 waves of 64
BLOCKS: gl.constexpr = gl.constexpr(256)
THREADS: gl.constexpr = gl.constexpr(512)
WAVES: gl.constexpr = gl.constexpr(8)
T512: gl.constexpr = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))
# a flat LDS array, gathered / scattered by per-thread index
SMEM_FLAT: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))

CHECK_EVERY = 256  # poll rounds between health / clock checks
WAIT_TICKS = 1_000_000_000  # 10 s of the 100 MHz clock
POLL_MAX = 12  # pairs a thread polls in one batch (bounds live registers)

_BITS = {False: "sc1", True: "sc0 sc1"}  # device scope, system scope


# ---------------------------------------------------------------- asm builders
def _poll_asm(n: int, system: bool) -> tuple[str, str]:
    """A bounded batched poll of ``n`` pairs. Operands:
    outputs v0..v(n-1) (=&v u64), scratch (=&v u64), saved exec, pending
    (=&s u64), round, failed (=&s u32), now (=&s u64), health value (=&v u32);
    inputs n addresses (v u64), tag (v u64), active (v u32), deadline (v u64),
    health address (v u64)."""
    scratch = n
    saved, pending, rnd, failed, now, hv = range(n + 1, n + 7)
    addr0 = n + 7
    tag, active, deadline, haddr = range(addr0 + n, addr0 + n + 4)
    bits = _BITS[system]
    lines = [
        f"s_mov_b32 ${rnd}, 0",
        f"s_mov_b32 ${failed}, 0",
        f"v_cmp_ne_u32 vcc, 0, ${active}",
        f"s_and_saveexec_b64 ${saved}, vcc",
        "s_cbranch_execz .Lpoll_done${:uid}",
        ".Lpoll_retry${:uid}:",
    ]
    lines += [f"global_load_dwordx2 ${i}, ${addr0 + i}, off {bits}" for i in range(n)]
    lines += ["s_waitcnt vmcnt(0)", f"s_mov_b64 ${pending}, 0"]
    for i in range(n):
        lines += [
            f"v_lshrrev_b64 ${scratch}, 32, ${i}",
            f"v_cmp_ne_u64 vcc, ${tag}, ${scratch}",
            f"s_or_b64 ${pending}, ${pending}, vcc",
        ]
    lines += [
        f"s_cmp_eq_u64 ${pending}, 0",
        "s_cbranch_scc1 .Lpoll_done${:uid}",
        "s_sleep 1",
        f"s_add_u32 ${rnd}, ${rnd}, 1",
        f"s_cmp_lt_u32 ${rnd}, {CHECK_EVERY}",
        "s_cbranch_scc1 .Lpoll_retry${:uid}",
        f"s_mov_b32 ${rnd}, 0",
        f"global_load_dword ${hv}, ${haddr}, off sc0 sc1",
        f"s_memrealtime ${now}",
        "s_waitcnt vmcnt(0) lgkmcnt(0)",
        f"v_cmp_ne_u32 vcc, 0, ${hv}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 .Lpoll_health${:uid}",
        f"v_cmp_ge_u64 vcc, ${now}, ${deadline}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc0 .Lpoll_retry${:uid}",
        f"s_mov_b32 ${failed}, 1",
        "s_branch .Lpoll_done${:uid}",
        ".Lpoll_health${:uid}:",
        f"s_mov_b32 ${failed}, 2",
        ".Lpoll_done${:uid}:",
        f"s_mov_b64 exec, ${saved}",
    ]
    constraints = ",".join(
        ["=&v"] * (n + 1) + ["=&s"] * 5 + ["=&v"] + ["v"] * (n + 4)
        + ["~{memory}", "~{vcc}", "~{scc}"]
    )  # fmt: skip
    return "\n".join(lines), constraints


def _store_asm(width: int, system: bool) -> tuple[str, str]:
    """A masked scoped store of ``width`` dwords (1 or 2): outputs a dummy (=v)
    and the saved exec (=&s); inputs address (v u64), active (v u32), value."""
    op = {1: "global_store_dword", 2: "global_store_dwordx2"}[width]
    asm = "\n".join(
        [
            "v_cmp_ne_u32 vcc, 0, $3",
            "s_and_saveexec_b64 $1, vcc",
            "s_cbranch_execz .Lst_done${:uid}",
            f"{op} $2, $4, off {_BITS[system]}",
            ".Lst_done${:uid}:",
            "s_mov_b64 exec, $1",
            "v_mov_b32 $0, 0",
        ]
    )
    return asm, "=v,=&s,v,v,v,~{memory},~{vcc},~{scc}"


_POLL = {(n, s): _poll_asm(n, s) for n in [*range(1, POLL_MAX + 1), 24] for s in (False, True)}
_STORE = {(w, s): _store_asm(w, s) for w in (1, 2) for s in (False, True)}


@gluon.constexpr_function
def poll_asm(n, system):
    return _POLL[(int(n), bool(system))][0]


@gluon.constexpr_function
def poll_constraints(n, system):
    return _POLL[(int(n), bool(system))][1]


@gluon.constexpr_function
def store_asm(width, system):
    return _STORE[(int(width), bool(system))][0]


@gluon.constexpr_function
def store_constraints(width, system):
    return _STORE[(int(width), bool(system))][1]


# ---------------------------------------------------------------- clock
@gluon.jit
def memrealtime(like):
    """The 100 MHz device clock, one value a lane of ``like``'s layout."""
    return gl.inline_asm_elementwise(
        "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
        "=s,v,~{memory}",
        [like.to(gl.uint32)],
        gl.uint64,
        False,
        1,
    )


@gluon.jit
def deadline(like, TICKS: gl.constexpr = WAIT_TICKS):
    return memrealtime(like) + TICKS


# ---------------------------------------------------------------- stores
@gluon.jit
def store32(addr, value, active, SYSTEM: gl.constexpr = False):
    """``value`` (32-bit) at the 64-bit address ``addr`` where ``active``."""
    out = gl.inline_asm_elementwise(
        store_asm(1, SYSTEM),
        store_constraints(1, SYSTEM),
        [addr.to(gl.uint64), active.to(gl.uint32), value.to(gl.uint32, bitcast=True)],
        (gl.uint32, gl.uint64),
        False,
        1,
    )
    return out[0]


@gluon.jit
def store64(addr, value, active, SYSTEM: gl.constexpr = False):
    out = gl.inline_asm_elementwise(
        store_asm(2, SYSTEM),
        store_constraints(2, SYSTEM),
        [addr.to(gl.uint64), active.to(gl.uint32), value.to(gl.uint64, bitcast=True)],
        (gl.uint32, gl.uint64),
        False,
        1,
    )
    return out[0]


# ---------------------------------------------------------------- scoped loads
# Data another CTA of the same launch wrote (behind a flag the caller already
# saw): device-scope loads (sc1), as FlyDSL's CM_DEV. Gluon's cache modifiers
# give no sc1-only load on this build (buffer_load: none / .cg / .cs -> sc0 nt),
# and a plain cached load may hit a line this XCD cached before the data
# landed (a line holding a ready and a not-yet-written item). n dwordx2 loads
# go out together, then one vmcnt(0).
def _load_asm(n: int, system: bool) -> tuple[str, str]:
    lines = [f"global_load_dwordx2 ${i}, ${n + i}, off {_BITS[system]}" for i in range(n)]
    lines.append("s_waitcnt vmcnt(0)")
    return "\n".join(lines), ",".join(["=&v"] * n + ["v"] * n + ["~{memory}"])


_LOAD = {(n, s): _load_asm(n, s) for n in (1, 2, 4, 8) for s in (False, True)}


@gluon.constexpr_function
def load_asm(n, system):
    return _LOAD[(int(n), bool(system))][0]


@gluon.constexpr_function
def load_constraints(n, system):
    return _LOAD[(int(n), bool(system))][1]


@gluon.jit
def load_dev1(addr, SYSTEM: gl.constexpr = False):
    """8 bytes at ``addr`` (u64)."""
    return gl.inline_asm_elementwise(
        load_asm(1, SYSTEM), load_constraints(1, SYSTEM), [addr.to(gl.uint64)],
        gl.uint64, False, 1,
    )  # fmt: skip


@gluon.jit
def load_dev2(a0, a1, SYSTEM: gl.constexpr = False):
    out = gl.inline_asm_elementwise(
        load_asm(2, SYSTEM), load_constraints(2, SYSTEM),
        [a0.to(gl.uint64), a1.to(gl.uint64)], (gl.uint64, gl.uint64), False, 1,
    )  # fmt: skip
    return out[0], out[1]


@gluon.jit
def load_dev4(a, SYSTEM: gl.constexpr = False):
    """32 contiguous bytes from ``a`` as four u64."""
    a = a.to(gl.uint64)
    out = gl.inline_asm_elementwise(
        load_asm(4, SYSTEM), load_constraints(4, SYSTEM),
        [a, a + 8, a + 16, a + 24], (gl.uint64, gl.uint64, gl.uint64, gl.uint64),
        False, 1,
    )  # fmt: skip
    return out[0], out[1], out[2], out[3]


@gluon.jit
def load_dev8(a, SYSTEM: gl.constexpr = False):
    """64 contiguous bytes from ``a`` as eight u64."""
    a = a.to(gl.uint64)
    out = gl.inline_asm_elementwise(
        load_asm(8, SYSTEM), load_constraints(8, SYSTEM),
        [a, a + 8, a + 16, a + 24, a + 32, a + 40, a + 48, a + 56],
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64,
         gl.uint64, gl.uint64),
        False, 1,
    )  # fmt: skip
    return out


@gluon.jit
def lo32(w):
    return (w & 0xFFFFFFFF).to(gl.uint32)


@gluon.jit
def hi32(w):
    return (w >> 32).to(gl.uint32)


@gluon.jit
def pair_word(bits, tag):
    """(bits | tag << 32): ``bits`` any 32-bit value, carried as its bits."""
    return bits.to(gl.uint32, bitcast=True).to(gl.uint64) | (
        tag.to(gl.uint32).to(gl.uint64) << 32
    )


@gluon.jit
def put(base, i, bits, tag, active, SYSTEM: gl.constexpr = False):
    """Pair ``i`` of the region at byte address ``base`` := (bits, tag)."""
    addr = base.to(gl.uint64) + i.to(gl.uint64) * 8
    return store64(addr, pair_word(bits, tag), active, SYSTEM)


# ---------------------------------------------------------------- polls
# Output tuples: N values (u64), scratch (u64), saved exec, pending (u64),
# round, failed (u32), now (u64), health value (u32). ``failed`` is output N + 4.


@gluon.jit
def poll1(addr, tag, active, dl, health, SYSTEM: gl.constexpr = False):
    """One pair at byte address ``addr``: (value bits u32, failed u32)."""
    out = gl.inline_asm_elementwise(
        poll_asm(1, SYSTEM),
        poll_constraints(1, SYSTEM),
        [addr.to(gl.uint64), tag.to(gl.uint32).to(gl.uint64), active.to(gl.uint32), dl, health.to(gl.uint64)],
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint64, gl.uint32),
        False,
        1,
    )
    return out[0].to(gl.uint32), out[5]


@gluon.jit
def poll2(a0, a1, tag, active, dl, health, SYSTEM: gl.constexpr = False):
    out = gl.inline_asm_elementwise(
        poll_asm(2, SYSTEM),
        poll_constraints(2, SYSTEM),
        [a0.to(gl.uint64), a1.to(gl.uint64), tag.to(gl.uint32).to(gl.uint64), active.to(gl.uint32), dl,
         health.to(gl.uint64)],  # fmt: skip
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint64, gl.uint32),
        False,
        1,
    )
    return out[0].to(gl.uint32), out[1].to(gl.uint32), out[6]


@gluon.jit
def poll4(a0, a1, a2, a3, tag, active, dl, health, SYSTEM: gl.constexpr = False):
    out = gl.inline_asm_elementwise(
        poll_asm(4, SYSTEM),
        poll_constraints(4, SYSTEM),
        [a0.to(gl.uint64), a1.to(gl.uint64), a2.to(gl.uint64), a3.to(gl.uint64), tag.to(gl.uint32).to(gl.uint64),
         active.to(gl.uint32), dl, health.to(gl.uint64)],  # fmt: skip
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint64, gl.uint32),
        False,
        1,
    )
    return out[0].to(gl.uint32), out[1].to(gl.uint32), out[2].to(gl.uint32), out[3].to(gl.uint32), out[8]


@gluon.jit
def poll8(a0, a1, a2, a3, a4, a5, a6, a7, tag, active, dl, health,
          SYSTEM: gl.constexpr = False):  # fmt: skip
    out = gl.inline_asm_elementwise(
        poll_asm(8, SYSTEM),
        poll_constraints(8, SYSTEM),
        [a0.to(gl.uint64), a1.to(gl.uint64), a2.to(gl.uint64), a3.to(gl.uint64), a4.to(gl.uint64), a5.to(gl.uint64),
         a6.to(gl.uint64), a7.to(gl.uint64), tag.to(gl.uint32).to(gl.uint64), active.to(gl.uint32), dl,
         health.to(gl.uint64)],  # fmt: skip
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint64,
         gl.uint32),  # fmt: skip
        False,
        1,
    )
    return (
        out[0].to(gl.uint32), out[1].to(gl.uint32), out[2].to(gl.uint32), out[3].to(gl.uint32),
        out[4].to(gl.uint32), out[5].to(gl.uint32), out[6].to(gl.uint32), out[7].to(gl.uint32), out[12],
    )  # fmt: skip


@gluon.jit
def fail(health, failed, rank, site, active):
    """A timed-out wait (``failed`` == 1) sets the sticky health word
    ``[1, rank, site]`` (later waits then give up at their next check); a wait
    that gave up on the word (2) leaves it."""
    first = active & (failed == 1)
    hb = health.to(gl.uint64)
    store32(hb + 4, rank + gl.zeros_like(failed).to(gl.int32), first, True)
    store32(hb + 8, site + gl.zeros_like(failed).to(gl.int32), first, True)
    return store32(hb, gl.full(failed.shape, 1, gl.int32, failed.type.layout), first,
                   True)  # fmt: skip


# ---------------------------------------------------------------- publish
@gluon.jit
def drain():
    """This wave's stores and loads complete (vmcnt 0)."""
    layout: gl.constexpr = T512
    z = gl.arange(0, 512, layout=layout)
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\nv_mov_b32 $0, $1", "=v,v,~{memory}", [z], gl.int32,
        False, 1,
    )  # fmt: skip


@gluon.jit
def publish(base, i, bits, tag, who, SYSTEM: gl.constexpr = False):
    """The CTA's earlier plain stores made readable by every CTA, then thread
    ``who``'s pair (base, i) := (bits, tag)."""
    drain()
    gl.barrier()
    return put(base, i, bits, tag, who, SYSTEM)


@gluon.jit
def poll6x4(b0, b1, b2, b3, b4, b5, tag, active, dl, health, SYSTEM: gl.constexpr = False):
    """Four consecutive pairs from each of six bases, one round trip (FlyDSL's
    batch of 12 two-pair loads): ((v[b][0..3]) for b in 0..5), failed."""
    out = gl.inline_asm_elementwise(
        poll_asm(24, SYSTEM),
        poll_constraints(24, SYSTEM),
        [b0.to(gl.uint64), b0.to(gl.uint64) + 8, b0.to(gl.uint64) + 16, b0.to(gl.uint64) + 24,
         b1.to(gl.uint64), b1.to(gl.uint64) + 8, b1.to(gl.uint64) + 16, b1.to(gl.uint64) + 24,
         b2.to(gl.uint64), b2.to(gl.uint64) + 8, b2.to(gl.uint64) + 16, b2.to(gl.uint64) + 24,
         b3.to(gl.uint64), b3.to(gl.uint64) + 8, b3.to(gl.uint64) + 16, b3.to(gl.uint64) + 24,
         b4.to(gl.uint64), b4.to(gl.uint64) + 8, b4.to(gl.uint64) + 16, b4.to(gl.uint64) + 24,
         b5.to(gl.uint64), b5.to(gl.uint64) + 8, b5.to(gl.uint64) + 16, b5.to(gl.uint64) + 24,
         tag.to(gl.uint32).to(gl.uint64), active.to(gl.uint32), dl, health.to(gl.uint64)],
        (gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64,
         gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64,
         gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64,
         gl.uint64, gl.uint64, gl.uint64, gl.uint32, gl.uint32, gl.uint64, gl.uint32),
        False,
        1,
    )  # fmt: skip
    return out


@gluon.jit
def load_dev32(addr, SYSTEM: gl.constexpr = False):
    """4 bytes at ``addr`` (u32), device scope (sc1) or system (sc0 sc1)."""
    if SYSTEM:
        out = gl.inline_asm_elementwise("global_load_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)",
                                        "=&v,v,~{memory}", [addr.to(gl.uint64)], gl.uint32, False, 1)
    else:
        out = gl.inline_asm_elementwise("global_load_dword $0, $1, off sc1\ns_waitcnt vmcnt(0)",
                                        "=&v,v,~{memory}", [addr.to(gl.uint64)], gl.uint32, False, 1)
    return out
