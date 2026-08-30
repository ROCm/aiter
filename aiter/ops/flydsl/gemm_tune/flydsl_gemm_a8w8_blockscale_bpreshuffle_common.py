# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune space for the FlyDSL a8w8 blockscale bpreshuffle GEMM on CDNA.

Companion to ``flydsl_gemm_a8w8_bpreshuffle_common.py`` (the rowwise family). This
module is the single source of truth for the ``kernelName`` string that travels
through a tuned CSV row: the tuner writes it, and
``aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_flydsl`` reads it back with
``parse_kernel_name``. Keep the two here so they cannot drift.

The name carries the tile triple, the two input dtypes and the output dtype, the K
scale-block size and the scheduler, then one field per swept knob:

* ``_w<n>``  -- ``num_waves``, 4 or 8.
* ``_ac<b>`` -- ``use_async_copy``.
* ``_cs<b>`` -- ``use_cshuffle_epilog``.

Only those three vary across ``kernels_list``; the dtypes and the scheduler are
recorded at their defaults so that a name stays self-describing. All three are
optional groups, so a name written before they existed still parses.
``dsrd_depth``, ``waves_per_eu`` and ``stage_a_scales`` are absent from the name
entirely: the tuner sweeps none of them, and the first two follow the arch. Add fields as the sweep grows -- appending to the regex is
safe, changing the existing groups is not.
"""

import re
from dataclasses import dataclass

_DTYPE_SHORT = {
    "fp8": "F8",
    "int8": "I8",
    "bf16": "B16",
    "fp16": "F16",
}
_SHORT_DTYPE = {v: k for k, v in _DTYPE_SHORT.items()}

# Note the correct spelling. The rowwise family ships "bpreshuflle" in every
# tuned CSV it has already written, so that typo cannot be fixed there; no
# block-scale CSV names a FlyDSL kernel yet, so this one starts out clean.
_NAME_RE = re.compile(
    r"^flydsl_blockscale_bpreshuffle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<q_dtype_a>[A-Z0-9]+)_(?P<q_dtype_w>[A-Z0-9]+)_(?P<dtype>[A-Z0-9]+)_"
    r"sbk(?P<scale_block_k>\d+)_"
    r"(?P<scheduler>[a-z0-9]+)"
    r"(?:_w(?P<num_waves>\d+))?"
    r"(?:_ac(?P<use_async_copy>[01]))?"
    r"(?:_cs(?P<use_cshuffle_epilog>[01]))?$"
)


@dataclass(frozen=True)
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    scale_block_k: int = 128
    q_dtype_a: str = "fp8"
    q_dtype_w: str = "fp8"
    dtype: str = "bf16"  # output dtype
    sScheduler: str = "Default"
    num_waves: int = 4
    use_async_copy: bool = False
    use_cshuffle_epilog: bool = False

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        return "_".join(
            [
                "flydsl",
                "blockscale",
                "bpreshuffle",
                "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
                qa,
                qw,
                dt,
                f"sbk{self.scale_block_k}",
                self.sScheduler.lower(),
                f"w{self.num_waves}",
                f"ac{int(self.use_async_copy)}",
                f"cs{int(self.use_cshuffle_epilog)}",
            ]
        )


def parse_kernel_name(kernel_name: str):
    """Inverse of ``kernelInstance.name``; returns a kernelInstance or None."""
    m = _NAME_RE.match(kernel_name or "")
    if m is None:
        return None
    return kernelInstance(
        tile_m=int(m.group("tile_m")),
        tile_n=int(m.group("tile_n")),
        tile_k=int(m.group("tile_k")),
        scale_block_k=int(m.group("scale_block_k")),
        q_dtype_a=_SHORT_DTYPE.get(m.group("q_dtype_a"), m.group("q_dtype_a").lower()),
        q_dtype_w=_SHORT_DTYPE.get(m.group("q_dtype_w"), m.group("q_dtype_w").lower()),
        dtype=_SHORT_DTYPE.get(m.group("dtype"), m.group("dtype").lower()),
        sScheduler=m.group("scheduler").capitalize(),
        num_waves=int(m.group("num_waves") or 4),
        use_async_copy=bool(int(m.group("use_async_copy") or 0)),
        use_cshuffle_epilog=bool(int(m.group("use_cshuffle_epilog") or 0)),
    )


def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for the current target."""
    from aiter.ops.flydsl.utils import get_shared_memory_per_block

    try:
        from aiter.jit.utils.chip_info import get_gfx
    except Exception:  # pragma: no cover - chip_info needs a live runtime
        return get_shared_memory_per_block()
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


def estimated_lds_bytes(
    ki: "kernelInstance",
    use_cshuffle_epilog: bool = False,
    k: int = 0,
    num_waves: int = 4,
    stage_a_scales: bool = True,
) -> int:
    """LDS footprint of one instance, or 0 when the kernel cannot be imported.

    Asks ``gemm_blockscale_preshuffle.plan_lds`` rather than recomputing it: the kernel
    sizes its own ping/pong buffers. A copy of that arithmetic lived here and went stale
    the first time those choices changed, disagreeing with the kernel on 7 of the 14
    candidate tiles. ``use_cshuffle_epilog`` matters: the CShuffle image is
    ``2*tile_m*tile_n``, larger than the plain staging whenever ``tile_n > tile_k``, so a
    tile that fits without it can exceed the limit with it. ``stage_a_scales`` matters for
    the same reason and defaults to **True**, the value every production caller compiles
    with: the staged A-scale slice adds ``2 * (tile_k // scale_block_k) * tile_m * 4``
    bytes, and validating without it can approve kernelIds that would overflow.
    """
    try:
        from aiter.ops.flydsl.kernels.gemm_blockscale_preshuffle import plan_lds
    except Exception:  # pragma: no cover - needs a FlyDSL install
        return 0
    num_k_tiles = (k // ki.tile_k) if k else 1
    if num_k_tiles < 1:
        return 0
    *_, total_bytes = plan_lds(
        tile_m=ki.tile_m,
        tile_n=ki.tile_n,
        tile_k=ki.tile_k,
        num_waves=num_waves,
        num_k_tiles=num_k_tiles,
        use_cshuffle_epilog=use_cshuffle_epilog,
        scale_block_k=ki.scale_block_k,
        stage_a_scales=stage_a_scales,
    )
    return total_bytes


# Mirrors of the kernel's module constants, so the structural checks below still run
# without a FlyDSL install. These are shape-independent and have not moved.
_MFMA_MN = 16
_WAVE = 64


def effective_stage_a_scales(
    tile_m: int,
    tile_k: int,
    scale_block_k: int = 128,
    use_async_copy: bool = False,
    stage_a_scales: bool = True,
    wave: int = _WAVE,
) -> bool:
    """Whether the kernel will actually stage the A scales for this tile.

    Mirrors ``gemm_blockscale_preshuffle`` exactly (``_stage_a_scales = stage_a_scales
    and _scales_async_ok``): staging rides the async global->LDS path, so it is off
    whenever async copy is, and off for tiles whose scale count is not a whole number of
    waves. Validation must ask this rather than the caller's request -- asking the
    request over-counts on gfx942, where async copy is arch-off, and would reject tiles
    that genuinely fit.
    """
    return bool(
        stage_a_scales
        and use_async_copy
        and ((tile_k // scale_block_k) * tile_m) % wave == 0
    )


def tile_is_valid(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    num_waves: int = 4,
    use_cshuffle_epilog: bool = False,
    stage_a_scales: bool = True,
) -> bool:
    """Whether ``compile_blockscale_preshuffle_gemm`` will accept this tile.

    Every rejection here is one the kernel would otherwise make itself, and the reason to
    make it first is the failure mode. The kernel's own guards raise ``ValueError`` rather
    than the ``RuntimeError`` the caller contract specifies, and an over-large tile does
    not raise at all: it compiles, the backend reports ``local memory (N) exceeds limit``,
    and the launch then fails with ``hipErrorIllegalState`` **without raising in Python**.
    The output buffer is left unwritten and the HIP context is poisoned, so the next
    unrelated call fails far from the cause. That is worse than an abort, because it
    returns. Rejecting the tile before anything is compiled turns both into an ordinary
    RuntimeError.

    ``stage_a_scales`` defaults to True to match production; passing False here while
    compiling with True is what let the six over-budget ids through.
    """
    total_threads = num_waves * _WAVE
    bytes_a_per_tile = tile_m * tile_k  # fp8: one byte per element
    if n % tile_n or k % tile_k or tile_k % scale_block_k or k % scale_block_k:
        return False
    # Each wave owns a whole number of 16-wide MFMA tiles along N.
    if tile_n % (num_waves * _MFMA_MN):
        return False
    # A is staged through LDS in 16B or 8B per-thread chunks; a 4B remainder has nowhere
    # to land, and the kernel rejects it rather than silently dropping the bytes.
    if bytes_a_per_tile % total_threads or (bytes_a_per_tile // total_threads) % 8:
        return False
    lds = estimated_lds_bytes(
        kernelInstance(tile_m, tile_n, tile_k, scale_block_k),
        use_cshuffle_epilog=use_cshuffle_epilog,
        k=k,
        num_waves=num_waves,
        stage_a_scales=stage_a_scales,
    )
    return not lds or lds <= max_lds_bytes_for_tune()


def default_use_async_copy(gfx: str = "") -> bool:
    """The global to LDS DMA moves 16 bytes per thread on gfx95x and 4 on gfx942, so
    the wide path pays there and the narrow one does not pay here.

    ``gfx`` names the arch to answer for, defaulting to the running one. The AOT
    prewarm passes a target arch, which is not always the host.
    """
    if not gfx:
        try:
            from aiter.jit.utils.chip_info import get_gfx

            gfx = get_gfx()
        except Exception:  # noqa: BLE001
            return False
    return gfx.startswith("gfx95")


def default_dsrd_depth(gfx: str = "") -> int:
    """Absent from the kernelName and never swept, so the arch decides it.

    More than one ds_read per MFMA group is a small loss on gfx942 at every depth
    tried, and a gain on gfx95x.
    """
    return 3 if default_use_async_copy(gfx) else 1


# ---------------------------------------------------------------------------
# Tuning search space. kernelId indexes the product of the three tables below,
# so they are append-only: inserting anywhere repoints every tuned CSV row.
# ---------------------------------------------------------------------------

# Upstream's candidate list (FlyDSL tests/kernels/test_blockscale_preshuffle_gemm.py
# select_tile_config), extended past its tile_m ceiling of 64.
TILE_CANDIDATES = (
    (16, 64, 256),
    (16, 128, 256),
    (32, 64, 128),
    (32, 64, 256),
    (32, 128, 128),
    (32, 128, 256),
    (64, 64, 128),
    (64, 64, 256),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
    (128, 128, 128),
    (128, 256, 128),
    (256, 128, 128),
)

# The compile flags are further axes over the same tiles. They are appended rather
# than interleaved because kernelId indexes this dict and already-tuned CSV rows must
# keep resolving to the kernel they measured. The nesting is flag-major, then wave,
# then tile, and the all-default flag pair sorts first, so every id an earlier sweep
# could have written still names the same kernel. tile_is_valid rejects what a given
# combination cannot serve (8 waves needs tile_n % 128, and cshuffle roughly doubles
# the LDS footprint), so the appended blocks stay uniform even where they are sparse.
WAVE_CANDIDATES = (4, 8)
# (use_async_copy, use_cshuffle_epilog); (False, False) must stay first.
FLAG_CANDIDATES = ((False, False), (True, False), (False, True), (True, True))

kernels_list = {
    (f_idx * len(WAVE_CANDIDATES) + w_idx) * len(TILE_CANDIDATES)
    + i: kernelInstance(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        num_waves=nw,
        use_async_copy=ac,
        use_cshuffle_epilog=cs,
    )
    for f_idx, (ac, cs) in enumerate(FLAG_CANDIDATES)
    for w_idx, nw in enumerate(WAVE_CANDIDATES)
    for i, (tm, tn, tk) in enumerate(TILE_CANDIDATES)
}
