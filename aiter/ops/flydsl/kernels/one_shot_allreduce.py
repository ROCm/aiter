# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} exact one-shot (1-stage) all-reduce.

Decode-regime kernel: bf16 in, fp32 accumulate, bf16 out, no codec. One
communication round, no grid-wide barrier and no flags -- each rank pushes its
whole tile into every peer's inbox, and the payload is its own flag (a Lamport
scheme). Inbox slots hold a sentinel value until a peer's data lands. The receiver
polls its 16 B per source until no half-word is the sentinel, reduces the N
copies, and re-arms the slot two rounds ahead.

This kernel trades wire volume for round trips:
(N-1)*S pushed rather than (N-1)*S read, but one serialized fabric traversal.

No LDS is needed: thread ``t``'s 16 B lands at the same offset in every destination,
so it can be pushed straight from registers.

On a cacheable (fine-grained) inbox each wave writes its push back out of L2,
and the re-arm is written through.

Padded builds (a hidden dim with no native row geometry) mask their pad columns
with a per-row buffer descriptor bounded to ``hidden*2`` bytes: pad columns are
out of bounds for that descriptor, so a plain tiled copy gets hardware OOB
handling for free (load->0, store dropped), with no exec-mask split. See
``quick_allreduce_fusions.make_rowbuf_atom_row``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

# The fused epilogue and the row-to-workgroup geometry are shared with the
# quantized schedules.
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    FUSIONS,
    XCHG_LINE_BYTES,
    make_rowbuf_atom_row,
    make_wave_partials,
    pack_bf16,
    padded_row_block,
    padded_row_block_options,
    residual_add,
    rms_local_sumsq,
    rms_rstd,
    row_block,
    row_block_options,
    row_block_supported,
    rstd_from_total,
    scale_by_weight,
    wave_reduce_add,
    xchg_arrivals,
    xchg_clamp_units,
    xchg_contribution,
    xchg_total,
)

# The peer-store/load primitives, the cache-policy table and the inbox-memory
# taxonomy are shared with the quantized kernels verbatim.
from .quick_allreduce_shared import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    SUPPORTED_WORLDS,
    WAVE,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _global_ptr,
    _i32_to_bytes,
    _load_peers,
    _poll_v4i32,
    _release_inbox,
    _store_v4i32_peer,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
    make_hbm_operand,
)

DEFAULT_BLOCK = 256
# Threads per block, which sets the tile width: ``tile = block * atoms * 16 B``.
# It sets the parallelism floor at a given payload.
#
# The trade is per-block fixed cost: every block polls and re-arms its own
# slots, so that work rises by the same factor the block count does.
SUPPORTED_BLOCKS = (64, 128, 256, 512)
# 16 B per thread per atom -- one ``global_store_dwordx4``.
ATOM_BYTES = 16
ATOM_I32 = ATOM_BYTES // 4
DEFAULT_ATOMS = 1
# Atoms per thread per tile. More atoms means a bigger tile, hence fewer blocks
# for a given payload, at the cost of coarser load balance on the last partial
# tile.
SUPPORTED_ATOMS = (1, 2, 4)
DEFAULT_GRID_CAP = 64
# Workgroups one fused token row is split over. 1 is one workgroup per row, the
# unsplit build. The rest are the counts that divide the shipped widths into
# whole waves: 7168 = 14 waves at atoms=1 gives {2, 7, 14}; 4096/8192 give the
# powers of two.
SUPPORTED_SPLITS = (1, 2, 4, 7, 8, 14, 16)
DEFAULT_SPLIT = 1

# Per-``(link, world_size)`` tuning ladder: ``(min_bytes, atoms, grid_cap,
# fanout, block)`` rungs. The host builds one engine per rung and
# selects by payload size at launch. Created from a tuning sweep.
#
# ``atoms`` and ``block`` both scale the tile, and their product is what
# matters to the block count; they are separate knobs because only ``block``
# also changes the workgroup size, and only ``atoms`` also changes how many
# stores one thread has in flight.
#
#   PCIe
#
#     TP2  b256 a1 cap128 -- the 4 KiB tile to 384 KiB, then b512 a4 cap128,
#                            the 32 KiB tile, over the rest of the exact-mode
#                            window, which runs to 64 MiB.
#     TP4  b128 a1 cap128 -- the 2 KiB tile to 96 KiB, then b512 a2 cap64, the
#                            16 KiB tile, to the 1 MiB exact-mode ceiling.
#     TP8  b64 a1 cap64   -- the 1 KiB tile, except b256 a1 cap64, the 4 KiB
#                            tile, from 48 to 192 KiB.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b128/b256 -- the 2 KiB tile to 256 KiB, the most it
#                               covers in one round, then the 4 KiB tile.
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b128 -- one rung over the whole window
ONESHOT_LADDER = {
    ("pcie", 2): (
        (0, 1, 128, "peer", 256),
        (384 << 10, 4, 128, "peer", 512),
    ),
    ("pcie", 4): (
        (0, 1, 128, "peer", 128),
        (96 << 10, 2, 64, "peer", 512),
    ),
    ("pcie", 8): (
        (0, 1, 64, "peer", 64),
        (48 << 10, 1, 64, "peer", 256),
        (192 << 10, 1, 64, "peer", 64),
    ),
    ("xgmi", 2): (
        (0, 1, 128, "peer", 128),
        (256 << 10, 1, 128, "peer", 256),
    ),
    ("xgmi", 4): ((0, 1, 128, "peer", 256),),
    ("xgmi", 8): ((0, 1, 128, "peer", 128),),
}


def oneshot_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)*, or a single default rung if unlisted.

    *link* defaults to ``"pcie"`` so a caller that has not resolved the fabric
    gets the conservative table: its fatter tiles cost throughput on xGMI but
    are never wrong in the sense of failing.
    """
    return ONESHOT_LADDER.get(
        (str(link), int(world_size)),
        (
            (
                0,
                DEFAULT_ATOMS,
                DEFAULT_GRID_CAP,
                DEFAULT_FANOUT,
                DEFAULT_BLOCK,
            ),
        ),
    )


# In the plain schedule ``atoms`` sets the *tile width*: tile = BLOCK*atoms*16 B,
# so a bigger atom count means fewer, fatter tiles for a given payload.
#
# In the fused schedule the tile is pinned to one token row -- or, with
# ``split``, to one 1/split slice of it -- because RMSNorm reduces over the row.
# ``atoms`` therefore sets the *block width* instead -- BLOCK =
# hidden/(split*8*atoms) -- and the tile and the block count are both
# independent of it.
#
# Rungs are ``(min_bytes, atoms, grid_cap, fanout, split)``. Rungs select by
# payload, which at a fixed hidden is the token count, so "split the row at
# M=1, not at M=4" is two rungs.
FUSED_ONESHOT_LADDER = {
    # PCIe: from measurements on MI350P
    ("pcie", 2): ((0, 1, 128, "peer", 1),),
    ("pcie", 4): (
        (0, 2, 64, "peer", 1),
        (64 << 10, 1, 128, "peer", 1),
        (384 << 10, 2, 32, "peer", 1),
    ),
    ("pcie", 8): ((0, 1, 128, "peer", 8), (48 << 10, 1, 32, "peer", 1)),
    # xGMI: from measurements on MI325X.
    # Split (k > 1) wins at TP4/TP8 for small M; TP2 never benefits from splitting
    # the hidden dim.
    ("xgmi", 2): ((0, 1, 128, "peer", 1),),
    ("xgmi", 4): ((0, 1, 128, "peer", 16), (144 << 10, 1, 128, "peer", 1)),
    ("xgmi", 8): ((0, 1, 64, "peer", 16), (56 << 10, 1, 128, "peer", 16)),
}


def fused_oneshot_ladder(world_size: int, link: str = "pcie"):
    """Rungs for *(link, world_size)* under ``fusion="rmsnorm"``."""
    return FUSED_ONESHOT_LADDER.get(
        (str(link), int(world_size)),
        ((0, 1, DEFAULT_GRID_CAP, DEFAULT_FANOUT, DEFAULT_SPLIT),),
    )


def fused_block(hidden: int, atoms: int) -> int:
    """Threads per block for a fused build, or raise saying why hidden dim
    cannot covered.

    One block covers one whole token row -- ``BLOCK * atoms * 8 == hidden`` --
    because the RMSNorm reduction spans the row and a row split across two
    blocks could only be joined with a grid-wide barrier."""

    return row_block(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_hidden_supported(hidden: int, atoms: int = 1) -> bool:
    """Whether a fused build exists for this (hidden, atoms). For host-side gates."""
    return row_block_supported(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_block_options(hidden: int) -> tuple[tuple[int, int], ...]:
    """Every ``(block, atoms)`` a fused build can use at hidden dim, widest first."""
    return row_block_options(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_padded_block_options(hidden: int) -> tuple[tuple[int, int, int], ...]:
    """Every ``(block, atoms, h_pad)`` a padded fused build can use, least pad first."""
    return padded_row_block_options(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_padded_block(hidden: int) -> tuple[int, int, int]:
    """``(block, atoms, h_pad)`` for a padded fused build, least padding first."""
    return padded_row_block(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_split_options(hidden: int) -> tuple[tuple[int, int, int], ...]:
    """Every ``(split, block, atoms)`` a native (unpadded) fused build can use
    at hidden dim, by ascending split, widest block first within one.

    ``split == 1`` is the unsplit build, i.e. ``fused_block_options``. A split
    build needs each ``hidden/split`` slice to be a whole-wave row geometry, and
    ``split * waves`` exchange writers to fit the word's arrival count.
    """
    hidden = int(hidden)
    opts = []
    for split in SUPPORTED_SPLITS:
        if hidden % split:
            continue
        for block, atoms in fused_block_options(hidden // split):
            if split > 1:
                try:
                    xchg_clamp_units(split * (block // WAVE))
                except ValueError:
                    continue
            opts.append((split, block, atoms))
    return tuple(opts)


def fused_atoms_for_block(hidden: int, block: int) -> int:
    """``atoms`` giving *block* threads at hidden dim, or raise naming the legal set."""
    block = int(block)
    opts = fused_block_options(hidden)
    for b, a in opts:
        if b == block:
            return a
    legal = ", ".join(str(b) for b, _ in opts) if opts else "none"
    raise ValueError(
        f"fused hidden={hidden} has no build at block={block}: one workgroup "
        f"covers one token row, so block is pinned to hidden/(8*atoms) for "
        f"atoms in {SUPPORTED_ATOMS} -- the legal blocks here are {legal}"
    )


# Read our own inbox with the caches bypassed: a peer wrote these lines
# microseconds ago and an L1 hit here is a stale hit. Same reasoning as the
# ring kernel's ``_RECV_POLICY``.
_RECV_POLICY = _CM_SC0 | _CM_SC1

# Which axis of the (peer, atom) fanout runs fastest across consecutive stores.
#
# "peer": a thread pushes all its atoms to one destination before moving to the
# next, so a wave hands each destination a contiguous ``block * 16`` B run.
#
# "atom": consecutive stores walk the peers of one atom. On xGMI the native
# packet is 64 B and there is no per-destination run-length benefit to collect,
# so spreading across links sooner can start more of them in parallel.
FANOUT_ORDERS = ("peer", "atom")
DEFAULT_FANOUT = "peer"

# Three inbox buffers rotate by colour. Round ``c`` reads buffer ``c``, and
# re-arms buffer ``c + 2`` (mod 3) with the sentinel. No peer can be writing that
# one: a peer is at most one round ahead, and that round uses ``c + 1``.
INBOX_BUFFERS = 3
# The sentinel is bf16 (and fp16) NaN ``0xFFFF`` in every half-word, so an unused
# inbox is just a ``0xFF`` memset. A sender rewrites a sentinel half-word in its
# own payload to ``0x7FFF``, which is still NaN, so the result is unchanged.
SENTINEL_FILL_BYTE = 0xFF
# Bit 15 of both half-words of an i32, as a signed i32.
_HALF_SIGN_BITS = -0x7FFF8000  # 0x80008000


def _sentinel_mask(atom_i32):
    """Bit 15 of every half-word of *atom_i32* that is the sentinel ``0xFFFF``.

    Branch-free zero test per 16-bit lane on the complement: a half-word of
    ``~x`` is non-zero iff adding ``0x7FFF`` to its low 15 bits, or its own bit
    15, sets bit 15. The low 15 bits are masked first, so no carry crosses into
    the upper half-word.
    """
    x = atom_i32 ^ -1
    nonzero = ((x & 0x7FFF7FFF) + 0x7FFF7FFF) | x
    return (nonzero & _HALF_SIGN_BITS) ^ _HALF_SIGN_BITS


def _canonicalize(atom_i32):
    """*atom_i32* with every sentinel half-word ``0xFFFF`` turned into ``0x7FFF``."""
    return atom_i32 ^ _sentinel_mask(atom_i32)


def _has_sentinel(vec_i32):
    """Whether any half-word of the i32 vector *vec_i32* is still the sentinel."""
    mask = _sentinel_mask(vec_i32)
    acc = mask[0]
    for i in range(1, vec_i32.numel):
        acc = acc | mask[i]
    return acc != 0


class _LdsWindow:
    """``base[offset + i]``: one colour's set of the per-wave LDS partials, in
    the indexing ``block_reduce_add`` uses."""

    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def __getitem__(self, i):
        return self.base[self.offset + i]

    def __setitem__(self, i, value):
        self.base[self.offset + i] = value


def make_one_shot_allreduce_kernel(
    *,
    world_size: int,
    atoms: int = DEFAULT_ATOMS,
    grid: int,
    inbox_memory: str = "uncached",
    fanout: str = DEFAULT_FANOUT,
    rank: int,
    block: int | None = None,
    fusion: str = "none",
    hidden: int | None = None,
    h_pad: int | None = None,
    split: int = DEFAULT_SPLIT,
    debug_slice_delay: int = 0,
):
    """Build the one-shot all-reduce, plain or fused with residual-add + RMSNorm.

    ``split`` (fused only) spreads each token row over that many workgroups,
    each owning a contiguous ``hidden/split`` slice; they join the row's sum of
    squares through one HBM exchange word (see ``_xchg`` in the kernel body).
    ``split=1`` is the one-workgroup-per-row build and compiles none of it.

    ``debug_slice_delay`` is for tests only: when set, the waves of slice 0
    sleep ``debug_slice_delay`` x ``s_sleep 127`` on every third row *after*
    contributing to the exchange and *before* reading it, so their siblings
    complete the row and run ahead into the next colour's word while slice 0
    still reads this one -- the case the exchange's reuse argument is about.
    Only a workgroup that handles several rows in one launch can be overtaken
    (launches on a stream do not overlap), so a test pairs it with a small
    grid cap.
    """
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
    split = int(split)
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")
    if split > 1 and fusion == "none":
        raise ValueError("split is only meaningful for a fused build")
    if debug_slice_delay and split == 1:
        raise ValueError("debug_slice_delay needs a split build")
    if fusion != "none" and hidden is None:
        raise ValueError(f"fusion={fusion!r} requires hidden")
    if fusion == "none" and hidden is not None:
        raise ValueError("hidden is only meaningful for a fused build")
    if fusion == "none" and h_pad is not None:
        raise ValueError("h_pad is only meaningful for a fused build")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if atoms not in SUPPORTED_ATOMS:
        raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if fanout not in FANOUT_ORDERS:
        raise ValueError(f"fanout must be one of {FANOUT_ORDERS}, got {fanout!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]

    fused = fusion == "rmsnorm"
    if fused:
        assert hidden is not None  # rejected above
        # ``h_pad`` is the width the *workgroup* covers; ``hidden`` stays the
        # width the *tensor* has.
        h_pad = hidden if h_pad is None else int(h_pad)
        if h_pad < hidden:
            raise ValueError(f"h_pad={h_pad} is narrower than hidden={hidden}")
        if h_pad != hidden and hidden % ATOM_ELEMS:
            raise ValueError(
                f"a padded fused build needs hidden to be a whole number of "
                f"{ATOM_ELEMS}-element atoms so the pad boundary lands on an "
                f"atom granule, got hidden={hidden}"
            )
        if split > 1:
            # The padded path bounds a per-row descriptor at the true row
            # width; splitting it is not needed by any shipped width.
            if h_pad != hidden:
                raise ValueError(
                    f"split={split} needs a native geometry, but hidden={hidden} "
                    f"runs padded to {h_pad}"
                )
            if hidden % split:
                raise ValueError(f"split={split} does not divide hidden={hidden}")
            if grid % split:
                # A row group is split consecutive workgroups, and every one of
                # them has to be launched or its siblings wait forever.
                raise ValueError(f"grid={grid} is not a multiple of split={split}")
        row_width = fused_block(h_pad // split, atoms)
        if block is not None and int(block) != row_width:
            legal = (
                fused_block_options(h_pad)
                if split == 1
                else [(b, a) for k, b, a in fused_split_options(hidden) if k == split]
            )
            raise ValueError(
                f"fused hidden={hidden} (h_pad={h_pad}, split={split}) at "
                f"atoms={atoms} needs block={row_width}, got block={block}; the "
                f"legal (block, atoms) pairs for this width are {legal}"
            )
        block = row_width
    else:
        block = DEFAULT_BLOCK if block is None else int(block)
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    # Whether any lane of the workgroup sits past the end of the real row.
    padded = fused and h_pad != hidden
    n_waves = block // WAVE
    # Split build: every wave of every slice of a row contributes one partial
    # to the row group's exchange word. Raises if the arrival count overflows.
    n_writers = split * n_waves
    if split > 1:
        xchg_clamp_units(n_writers)
    # Row groups: ``split`` consecutive workgroups share one row.
    n_groups = grid // split
    # Exchange state, split builds only: one word per (colour, group), each on
    # its own line, then this workgroup's last-complete word per colour.
    xchg_prev_off = INBOX_BUFFERS * n_groups * XCHG_LINE_BYTES
    xchg_bytes = xchg_prev_off + grid * INBOX_BUFFERS * 8 if split > 1 else 0
    # Per-wave partials for the block-wide sum of squares, one set per colour.
    # The plain build has no LDS at all, and nor does a split one: its waves
    # reduce through HBM.
    lds_bytes = INBOX_BUFFERS * n_waves * 4 if fused and split == 1 else 0

    tile_bytes = block * atoms * ATOM_BYTES
    tile_i32 = tile_bytes // 4
    # i32 between one token row and the next *in HBM*. The tile the wire and the
    # workgroup see is h_pad wide, but the tensor's rows are still packed at the
    # true width, so these part company exactly when a build is padded. A split
    # build views an (M, hidden) operand as (M*split, hidden/split) -- the same
    # memory -- so its "row" here is one slice.
    row_stride_i32 = (hidden // split // 2) if fused else tile_i32
    data_bytes = INBOX_BUFFERS * grid * world_size * tile_bytes

    # This rank's own index as a trace-time constant. It has to be
    # compile-time: the peer fanout, the poll and the reduce are all unrolled
    # over trace-time peer indices, and "all peers but me" is only expressible
    # there. The cost is one kernel binary per rank -- but a process is one
    # rank, so it compiles exactly one.
    self_rank = int(rank)
    # Peers this rank pushes payload to. Our own contribution never round-trips
    # through our own inbox: it stays in registers. Our own inbox slot is simply
    # never touched -- the wire format is unchanged, the slot is still
    # allocated, and no peer can observe the difference.
    push_peers = [p for p in range(world_size) if p != self_rank]

    # (peer, atom) iteration order for the fanout, unrolled at trace time.
    if fanout == "peer":
        fanout_pairs = [(p, a) for p in push_peers for a in range(atoms)]
    else:
        fanout_pairs = [(p, a) for a in range(atoms) for p in push_peers]

    # LDS for the fused sum-of-squares. Carries the per-wave partial sums
    # only: the 1/hidden, the +eps and the rsqrt all happen afterwards in
    # registers, per thread, so no scale is ever broadcast through LDS.
    #
    # One set per colour, because nothing else separates one tile's reads of
    # the partials from the next tile's writes: there is no barrier between
    # tiles. A wave writes a set again three tiles later, after it has passed
    # the reduction barrier of the two tiles in between, which every wave
    # reaches only once it has finished reading this one.
    _RmsShared = (
        make_wave_partials(INBOX_BUFFERS * n_waves) if fused and split == 1 else None
    )

    # One signature for both modes. The fused-only arguments are present (and
    # passed as zeros) in a plain build rather than being appended to a second
    # kernel. The fused-only code is still elided by ``const_expr(fused)``,
    # only five unused kernargs remain in a plain build. ``xchg_ptr`` is the
    # split build's exchange state, and 0 (never read) in every other build.
    @flyc.kernel(known_block_size=[block, 1, 1])
    def one_shot_allreduce(
        rank: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        n_blocks: fx.Int32,
        res_in_ptr: fx.Int64,
        res_out_ptr: fx.Int64,
        w_ptr: fx.Int64,
        eps: fx.Float32,
        xchg_ptr: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        # The tiled copy is used by both modes. ``hbm_row_layout`` (one
        # atom-wide row) and ``hbm_copy`` are built unconditionally; a padded
        # build addresses each row through a per-row buffer descriptor
        # (``_rowbuf_atom_row``) while an unpadded one slices the whole-tensor
        # buffer tensor (``_hbm_atom_row``). ``hbm_layout`` is the 3-D whole-tensor
        # layout consumed only by the unpadded ``make_hbm_operand``; a padded
        # build binds it to None (its operand never reads it).
        hbm_layout = None
        if const_expr(not padded):
            hbm_layout = fx.make_layout(
                (num_tiles, atoms, block * ATOM_I32),
                (row_stride_i32, block * ATOM_I32, 1),
            )
        hbm_row_layout = fx.make_layout((1, block * ATOM_I32), (block * ATOM_I32, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, block), (1, 1)),
            fx.make_layout((1, ATOM_I32), (1, 1)),
        ).get_slice(tid)

        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)

        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        # A padded build keeps the HBM operands as raw ``Int64`` base pointers so
        # ``_rowbuf_atom_row`` can bound a fresh descriptor per row; an unpadded
        # build wraps them in the whole-tensor buffer tensor.
        _operand = make_hbm_operand(
            padded=padded,
            nbytes=nbytes,
            hbm_i32_ptr=hbm_i32_ptr,
            hbm_layout=hbm_layout,
        )
        # A padded build addresses each row through a per-row buffer descriptor
        # bounded to the true width; ``_rowbuf_atom_row(ptr, tile, atom)`` builds
        # it from the operand's raw base pointer. Unpadded is None (unused).
        _rowbuf_atom_row = (
            make_rowbuf_atom_row(
                atoms_per_row=atoms,
                rows_per_tile=1,
                row_stride_i32=row_stride_i32,
                block=block,
                hidden=hidden,
                hbm_i32_ptr=hbm_i32_ptr,
                hbm_row_layout=hbm_row_layout,
            )
            if padded
            else None
        )

        in_buf = _operand(inp_ptr)
        out_buf = _operand(out_ptr)
        _load_color, _store_color = _color_io(colors_ptr, bid)

        if const_expr(fused):
            # residual in/out are (M, hidden) bf16 exactly like the payload, so
            # they ride the same addressing.
            res_in_buf = _operand(res_in_ptr)
            res_out_buf = _operand(res_out_ptr)
            # The gain is a single (hidden,) row shared by every token. A padded
            # build reads it through the same per-row descriptor at row 0; an
            # unpadded build bounds a one-row buffer tensor at the true width,
            # which is the whole mask for this one operand (nothing lies past it).
            # A split build sees it as (split, hidden/split) and reads row
            # ``part``: its own slice of the gain.
            w_buf = _operand(w_ptr, records=fx.Int64(hidden * 2))

        if const_expr(split > 1):
            # ``split`` consecutive workgroups share one token row: ``group``
            # names the row group, ``part`` this workgroup's slice. The host
            # launches a multiple of ``split`` workgroups, so tile
            # ``bid + i*n_blocks`` is row ``group + i*(n_blocks/split)``, slice
            # ``part`` -- the same row for the whole group at every i -- and
            # ``n_block_tiles`` below is equal across the group:
            # ceil((split*M - split*g - part) / (split*G)) = ceil((M - g) / G)
            # for every 0 <= part < split.
            group = bid // fx.Int32(split)
            part = bid % fx.Int32(split)

        def _slot_i32(buf, src):
            """i32 offset of the wire slot ``[buf][bid][src]``.

            Plain arithmetic rather than ``crd2idx`` on a 3-D layout: at
            ``grid == 1`` the middle mode is unit and gets coalesced away,
            after which a three-coordinate lookup silently returns a wrong
            (negative) index. The ring kernel hit exactly this.
            """
            return (
                buf * fx.Int32(grid * world_size * tile_i32)
                + bid * fx.Int32(world_size * tile_i32)
                + src * fx.Int32(tile_i32)
            )

        def _wrap(color):
            """*color* reduced into ``[0, INBOX_BUFFERS)``, for a *color* below
            twice that."""
            if color >= fx.Int32(INBOX_BUFFERS):
                color = color - fx.Int32(INBOX_BUFFERS)
            return color

        def _hbm_atom_row(buf, tile, atom):
            return fx.make_view(
                fx.get_iter(fx.slice(buf, (tile, atom, None))), hbm_row_layout
            )

        # The row-view function is chosen once at trace time: a padded build
        # addresses each row through a per-row bounded descriptor, an unpadded
        # one slices the whole-tensor buffer tensor. Both return a one-atom-wide
        # row view that ``partition_S/D`` consume, so the copy bodies are uniform.
        _atom_row = _rowbuf_atom_row if padded else _hbm_atom_row

        def _load_rows(buf, tile):
            """This thread's 16 B of each atom of *tile* of *buf*, as raw i32x4."""
            out = []
            for atom in range_constexpr(atoms):
                src = hbm_copy.partition_S(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                out.append(fx.Vector(frag.load()))
            return out

        def _load_tile(tile):
            return _load_rows(in_buf, tile)

        def _store_rows(buf, tile, vals):
            for atom in range_constexpr(atoms):
                dst = hbm_copy.partition_D(_atom_row(buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(vals[atom])
                fx.copy(hbm_copy_atom, frag, dst)

        def _store_tile(tile, vals):
            _store_rows(out_buf, tile, vals)

        def _fanout(buf, my_atoms):
            """Push this thread's atoms into every peer's slot for this rank.

            Thread ``t``'s data lands at the same offset in every destination,
            so it goes straight from registers -- no LDS staging.

            Our own inbox is not among the destinations: our contribution stays
            in registers, which saves 1/N of the stores, the polls and the
            re-arms.

            On a cacheable inbox an ``nt`` push can sit in this XCD's L2 until
            something evicts it, so each wave writes it back with a release
            fence.
            """
            for peer, atom in fanout_pairs:
                _store_v4i32_peer(
                    peer_vec[peer]
                    + _i32_to_bytes(
                        _slot_i32(buf, rank)
                        + fx.Int32(atom * block * ATOM_I32)
                        + tid * fx.Int32(ATOM_I32)
                    ),
                    my_atoms[atom],
                    payload_policy,
                )
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)

        def _recv_i32(buf, src, atom):
            """i32 offset of this thread's 16 B of *atom* in the slot of *src*."""
            return (
                _slot_i32(buf, fx.Int32(src))
                + fx.Int32(atom * block * ATOM_I32)
                + tid * fx.Int32(ATOM_I32)
            )

        def _load_recv(buf, src, atom):
            return _buffer_load(
                inbox, _recv_i32(buf, src, atom), ATOM_I32, fx.Int32, _RECV_POLICY
            )

        def _poll(buf):
            """Receive: this thread's 16 B of every atom of every source,
            keyed ``(src, atom)``, once none of them is the sentinel.

            The first reads all go out before any is tested. Until every slot
            has landed, every slot is then re-read on each pass, together,
            rather than one source at a time: a serial spin would leave the later
            sources' reads stale, and pay one more round trip per source after
            the last one lands. Re-reading a slot that has already landed is
            harmless, since nobody writes it again this round. The re-reads go
            through ``_poll_v4i32``, so the spin cannot be deleted.
            """
            keys = [
                (src, atom) for src in push_peers for atom in range_constexpr(atoms)
            ]
            addrs = [
                peer_vec[rank] + _i32_to_bytes(_recv_i32(buf, src, atom))
                for src, atom in keys
            ]

            def _pack(vecs):
                return fx.Vector.from_elements(
                    [v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32
                )

            got = _pack([_load_recv(buf, src, atom) for src, atom in keys])
            while _has_sentinel(got):
                got = _pack([_poll_v4i32(addr) for addr in addrs])
            return {
                key: fx.Vector.from_elements(
                    [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)],
                    fx.Int32,
                )
                for i, key in enumerate(keys)
            }

        def _rearm(buf):
            """Put the sentinel back in every slot this thread reads, in *buf*.

            On a cacheable inbox the sentinel is written through (``sc0 sc1``).
            An ``nt`` store could leave it dirty in our own L2, where it would
            hide a peer's later push from the poll, and its eventual eviction
            would overwrite that push in memory. The stores are local, so
            writing through costs no fabric traffic.
            """
            sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
            for src in push_peers:
                for atom in range_constexpr(atoms):
                    if const_expr(release_scope is not None):
                        _buffer_store(
                            inbox,
                            _recv_i32(buf, src, atom),
                            sentinel,
                            _CM_SC0 | _CM_SC1,
                        )
                    else:
                        _store_v4i32_peer(
                            peer_vec[rank] + _i32_to_bytes(_recv_i32(buf, src, atom)),
                            sentinel,
                            payload_policy,
                        )

        def _reduce_f32(my_atoms, polled):
            """Sum this thread's atom across all N contributions, in rank order.

            Rank order, not a rotated order: every rank must accumulate in the
            same sequence or the results differ in the last bit across ranks.
            ``cross_device_reduce`` makes the same promise for the same reason.
            Our own contribution comes out of the registers; the peers' come
            from *polled*, ``_poll``'s result.

            Left unrounded: the plain path rounds once in ``_reduce``, and the
            fused path needs the fp32 accumulator for the norm.
            """
            outs = []
            for atom in range_constexpr(atoms):
                acc = None
                for src in range_constexpr(world_size):
                    if const_expr(src == self_rank):
                        raw = my_atoms[atom]
                    else:
                        raw = polled[(src, atom)]
                    v = atom_bf16_to_f32(raw)
                    acc = v if acc is None else acc + v
                outs.append(acc)
            return outs

        def _reduce(my_atoms, polled):
            """The plain path's result: one rounding, at the end of the sum."""
            return [atom_f32_to_bf16(a) for a in _reduce_f32(my_atoms, polled)]

        def _xchg(tile, color, wave_sum, prev_same):
            """Join this wave's sum of squares with every other wave of the row.

            ``n_writers`` waves -- every wave of every slice of the row -- each
            add ``fixed(wave_sum) << 8 | 1`` to the row group's word for this
            colour (``quick_allreduce_fusions.xchg_contribution``), with a
            relaxed agent-scope 64-bit atomic whose result is left *unused*:
            that is the no-return form, and a returning one measured ~0.25 us
            slower. Count and sum move in one atomic op, so a wave that sees the
            count complete sees the sum complete, and no fence orders anything.
            (On gfx950 the release/acquire fences would be ``buffer_wbl2 sc1`` /
            ``buffer_inv sc1``: a whole-L2 writeback and invalidate.) Integer
            addition is order-independent, so every wave on every rank decodes
            the same bits whatever order the writers arrived in.

            The word rotates with the inbox colour, so the group's workgroups,
            which run the same rows from the same colour, agree on it. It is
            never reset. *prev_same* is its value when this colour last
            completed; this call is done when ``cur - prev_same`` counts
            ``n_writers`` arrivals. That value cannot be overtaken: before any
            wave can add to this colour's word again it must pass the other two
            colours' exchanges, which need every wave's contribution there,
            which each wave makes only after it has finished reading this one.

            Returns ``(row sum of squares, cur)``; *cur* becomes ``prev`` for
            this colour's next use.
            """
            word = _global_ptr(
                xchg_ptr
                + fx.Int64(color * fx.Int32(n_groups) + group)
                * fx.Int64(XCHG_LINE_BYTES),
                T.i64,
                8,
            )
            if (tid & fx.Int32(WAVE - 1)) == fx.Int32(0):
                fx.atomic_add(
                    word,
                    xchg_contribution(wave_sum, n_writers),
                    syncscope=rocdl.SyncScope.AgentOneAs,
                )
            if const_expr(debug_slice_delay):
                # Test-only: contributed, not yet read. The siblings complete
                # this row without us and run on into the next colour's word.
                if (part == fx.Int32(0)) & (
                    (tile // fx.Int32(split)) % fx.Int32(3) == fx.Int32(0)
                ):
                    for _ in range_constexpr(debug_slice_delay):
                        rocdl.s_sleep(127)
            # Every lane polls the same address: one request per wave.
            cur = fx.generic_load(
                word,
                dtype=fx.Int64,
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=rocdl.SyncScope.AgentOneAs,
            )
            while xchg_arrivals(cur - prev_same) != fx.Int64(n_writers):
                cur = fx.generic_load(
                    word,
                    dtype=fx.Int64,
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=rocdl.SyncScope.AgentOneAs,
                )
            return xchg_total(cur - prev_same), cur

        def _epilogue(tile, x_atoms, w_atoms, color, sq_lds, accs, prev_same):
            """bf16 round-trip, residual add, RMSNorm, on the reduced *accs*.

            The arithmetic lives in ``quick_allreduce_fusions``, which the mesh
            and ring epilogues share; what stays here is the store placement.
            ``residual_out`` is written *before* the row reduction, so it is in
            flight across the barrier (or, split, the exchange) rather than
            issued behind it -- it has no dependence on the norm.

            Returns the exchange word a split build saw complete (the next
            ``prev`` for this colour), None otherwise.
            """
            accs = residual_add(accs, x_atoms)
            _store_rows(res_out_buf, tile, pack_bf16(accs))
            done = None
            if const_expr(split > 1):
                # The row spans ``split`` workgroups: reduce this wave's share,
                # then join the rest through HBM. 1/hidden is the whole row's.
                wave_sum = wave_reduce_add(rms_local_sumsq([accs]))[0]
                total, done = _xchg(tile, color, wave_sum, prev_same)
                rstd = rstd_from_total(total, eps, hidden)
            else:
                # One block covers one row, so there is a single row to reduce.
                lds = None
                if const_expr(sq_lds is not None):
                    lds = _LdsWindow(sq_lds, color * fx.Int32(n_waves))
                rstds = rms_rstd([accs], eps, hidden, tid=tid, block=block, lds=lds)
                rstd = rstds[0]
            _store_tile(tile, scale_by_weight(accs, rstd, w_atoms))
            return done

        sq_lds = None
        w_atoms = None
        if const_expr(fused):
            # The gain is one row shared by every token, so it is read once here
            # rather than once per token -- a split build reads its own slice.
            if const_expr(split > 1):
                w_atoms = _load_rows(w_buf, part)
            else:
                w_atoms = _load_rows(w_buf, fx.Int32(0))
            if const_expr(n_waves > 1 and split == 1):
                # Allocated once, at kernel scope: SharedAllocator is static, so
                # an allocation reached from inside the tile loop would emit a
                # fresh LDS symbol per trace-time visit.
                sq_lds = fx.SharedAllocator().allocate(_RmsShared).peek().wave.ptr

        def _round(tile, color, raw_atoms, x_atoms, prev_same):
            """One tile, from its input atoms (and, fused, its residual atoms).

            Returns ``(next colour, done)``: *done* is the exchange word a split
            build saw complete, None otherwise. The colour is the buffer index
            itself, kept in [0, 3), so it never wraps out of step with the
            rotation.
            """
            my_atoms = [_canonicalize(v) for v in raw_atoms]
            _fanout(color, my_atoms)
            polled = _poll(color)
            done = None
            if const_expr(fused):
                done = _epilogue(
                    tile,
                    x_atoms,
                    w_atoms,
                    color,
                    sq_lds,
                    _reduce_f32(my_atoms, polled),
                    prev_same,
                )
            else:
                _store_tile(tile, _reduce(my_atoms, polled))
            _rearm(_wrap(color + fx.Int32(2)))
            return _wrap(color + fx.Int32(1)), done

        def _load_residual(tile):
            """The residual atoms of *tile*, None in a plain build.

            Issued before the push on purpose: the residual is plain HBM with
            no dependence on any peer, so this load retires during the poll
            instead of after it.
            """
            if const_expr(fused):
                return _load_rows(res_in_buf, tile)
            return None

        # Stride by the *launched* grid, not the compile-time cap: the host may
        # launch fewer blocks than ``grid``, and striding by the cap would leave
        # every tile above n_blocks unprocessed.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        # The host never launches more blocks than tiles, so every block has a
        # first tile, and its input is read before the colour: the two loads are
        # then in flight together rather than one after the other.
        first_atoms = _load_tile(bid)
        first_x = _load_residual(bid)
        color = _load_color()
        # Split build: the exchange word's value when each colour last
        # completed -- ``prev_0`` for the coming row's colour, ``prev_1`` and
        # ``prev_2`` for the two after it -- rotated every row and persisted per
        # workgroup across launches. Defined in every build because the loop
        # below assigns them (the rewriter carries whatever the body assigns);
        # unsplit, they pass through the loop untouched and fold away.
        prev_0 = fx.Int64(0)
        prev_1 = fx.Int64(0)
        prev_2 = fx.Int64(0)
        if const_expr(split > 1):
            prev_base = (
                xchg_ptr
                + fx.Int64(xchg_prev_off)
                + fx.Int64(bid) * fx.Int64(INBOX_BUFFERS * 8)
            )

            def _prev_word(slot):
                """This workgroup's persisted exchange word for colour *slot*."""
                return _global_ptr(prev_base + fx.Int64(slot) * fx.Int64(8), T.i64, 8)

            prev_0 = fx.generic_load(_prev_word(color), dtype=fx.Int64)
            prev_1 = fx.generic_load(
                _prev_word(_wrap(color + fx.Int32(1))), dtype=fx.Int64
            )
            prev_2 = fx.generic_load(
                _prev_word(_wrap(color + fx.Int32(2))), dtype=fx.Int64
            )
        first = _round(bid, color, first_atoms, first_x, prev_0)
        color = first[0]
        if const_expr(split > 1):
            # The next row uses the next colour.
            prev_0 = prev_1
            prev_1 = prev_2
            prev_2 = first[1]
        for i in range(fx.Int32(1), n_block_tiles, fx.Int32(1)):
            # The last round's re-arm must land before this round's push: a peer
            # that sees the push can be one round further on, writing into the
            # slot just re-armed. Between calls the kernel boundary does this.
            rocdl.s_waitcnt(vmcnt=0)
            tile = bid + i * n_blocks
            step = _round(tile, color, _load_tile(tile), _load_residual(tile), prev_0)
            color = step[0]
            if const_expr(split > 1):
                prev_0 = prev_1
                prev_1 = prev_2
                prev_2 = step[1]
        if const_expr(split > 1):
            slot_1 = _wrap(color + fx.Int32(1))
            slot_2 = _wrap(color + fx.Int32(2))
        if tid == 0:
            _store_color(color)
            if const_expr(split > 1):
                # Every wave decoded the same completed words, so any one
                # thread's copy is the workgroup's.
                fx.generic_store(_prev_word(color), prev_0)
                fx.generic_store(_prev_word(slot_1), prev_1)
                fx.generic_store(_prev_word(slot_2), prev_2)
        gpu.barrier()

    flat_wg = f"{block},{block}"

    @flyc.jit
    def launch_one_shot_allreduce(
        rank: fx.Int32,
        nbytes: fx.Int64,
        num_tiles: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        grid_x: fx.Int32,
        res_in_ptr: fx.Int64,
        res_out_ptr: fx.Int64,
        w_ptr: fx.Int64,
        eps: fx.Float32,
        xchg_ptr: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        one_shot_allreduce(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            grid_x,
            res_in_ptr,
            res_out_ptr,
            w_ptr,
            eps,
            xchg_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    tag = f"ws{world_size}_r{self_rank}_a{atoms}_g{grid}_{inbox_memory}_b{block}"
    if atoms > 1:
        tag += f"_{fanout}"
    if fused:
        # hidden sets BLOCK and the tile, so it belongs in the key just as much
        # as atoms does. h_pad changes both the geometry and which loads carry a
        # mask, so a padded build must not share a key with the plain one.
        tag += f"_rms_h{hidden}"
        if padded:
            tag += f"p{h_pad}"
        if split > 1:
            # split sets the tile and adds the exchange; block alone does not
            # name it (7168 at split=7 and 896 at split=1 differ in both).
            tag += f"_k{split}"
            if debug_slice_delay:
                tag += f"_dly{debug_slice_delay}"
    launch_one_shot_allreduce.func.__name__ = f"launch_one_shot_allreduce_{tag}"
    try:
        one_shot_allreduce.func.__name__ = f"one_shot_allreduce_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_one_shot_allreduce,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": lds_bytes,
        "tile_bytes": tile_bytes,
        # No handshake sector: a wire slot is just the payload.
        "wire_tile_bytes": tile_bytes,
        # Shims for ``quick_allreduce._StEngine``, which is reused verbatim for the IPC
        # inbox and peer table. This schedule has no super-tile (one round per
        # tile, nothing to batch) and no per-rank tile split (every rank sends
        # the whole tile), so the two are 1 and the full tile respectively.
        "super_tile": 1,
        "rank_tile_bytes": tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "atoms": atoms,
        "world_size": world_size,
        "inbox_memory": inbox_memory,
        "fanout": fanout,
        "rank": self_rank,
        # Byte every inbox byte starts as.
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": grid,
        "block": block,
        "fusion": fusion,
        "hidden": hidden,
        "h_pad": h_pad,
        "padded": padded,
        "split": split,
        # Local (never IPC-shared) exchange state for a split build, zeroed at
        # engine build: ``_StEngine`` appends it to its meta allocation.
        "xchg_bytes": xchg_bytes,
    }
