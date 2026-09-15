# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Codec-level tests for the quick-allreduce wire formats.

Single GPU, no IPC: these cover the codec, while ``test_flydsl_quick_allreduce_int4.py``
covers the schedules that carry it.

Two properties are load-bearing:

* **Memory path equals register path.** Staging the quantized words through
  LDS at their wire offsets and reading them back with ``_codec_load`` must
  reproduce, bit for bit, what handing the same words straight to the
  dequantizer produces. Relocation cannot change a value, so any disagreement
  about where the 2-bit plane or the scale word lives shows up here -- and
  nowhere else, since both sides of the real kernel would be wrong together.
* **The analytic error bound.** E4M3 carries 3 mantissa bits, so the decoded
  extremum is within 1/16 of the true one; on top of that a value can miss by
  half a step, and one near the extreme can land a whole code short. That gives
  ``|err| <= |ext| * (1/16 + 1.05/bias)`` with no free parameters.
"""

from __future__ import annotations

import functools
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

pytest.importorskip("flydsl")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from aiter.ops.flydsl.kernels.quick_allreduce_codec import (
    CODECS,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _scale_from_word,
    scale_slot_of,
    thread_lane,
)
from aiter.ops.flydsl.kernels.quick_allreduce_shared import BLOCK, make_pack_storage

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS or torch.cuda.device_count() < 1,
    reason="quick-allreduce codec needs one gfx942/gfx950 GPU",
)

#: One thread's contribution to a row: 8 bf16 values, 16 B -- the codec's own
#: atom, not the collective's 8-atom tile. A "row" is one such atom for the
#: whole block: BLOCK * 8 bf16 elements.
_ATOM_BF16 = 8
TILE_ELEMS = BLOCK * _ATOM_BF16
_GRID_CAP = 256


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    return fx.inttoptr(ptr_ty, addr_i64)


def make_codec_roundtrip_kernel(codec_name: str, via_memory: bool = True):
    codec = CODECS[codec_name]
    PackStorage = make_pack_storage(codec.rank_tile_i32)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def codec_roundtrip(
        num_rows: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        n_blocks: Int32,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        _wave, lane = thread_lane(tid)
        scale_slot, pair_in_slot = scale_slot_of(tid)

        in_ptr = _global_i32_ptr(inp_ptr)
        out_ptr_g = _global_i32_ptr(out_ptr)

        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        smem_ptr = lds.pack.ptr
        pack = lds.pack.view(
            fx.make_layout((1, codec.rank_tile_i32), (codec.rank_tile_i32, 1))
        )
        row0 = fx.Int32(0)

        def _row_i32_off(row):
            # 4 i32 (16 B) per thread; BLOCK*4 i32 per row.
            return row * fx.Int32(BLOCK * 4) + tid * fx.Int32(4)

        def _load_atom(row):
            v4 = fx.ptr_load(
                in_ptr + _row_i32_off(row), result_type=fx.Vector.make_type(4, fx.Int32)
            )
            return _atom_bf16_to_f16(v4)

        def _store_atom(row, value):
            fx.ptr_store(_atom_f16_to_bf16(value), out_ptr_g + _row_i32_off(row))

        def _get(off):
            return fx.Int32(fx.ptr_load(smem_ptr + off))

        def _write_packet(words, scale_word, is_leader):
            for (off, pred), word in zip(codec.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (row0, off))
            if codec.has_scale:  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale_word,
                        pack,
                        (row0, fx.Int32(codec.scale_i32_off) + scale_slot),
                    )

        def _row_via_memory(row):
            words, scale_word, leader = _codec_quant(codec, _load_atom(row), lane, tid)
            _write_packet(words, scale_word, leader)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            words_in, word_in = _codec_load(codec, _get, tid, scale_slot)
            scale = _scale_from_word(codec, word_in, pair_in_slot)
            _store_atom(row, _codec_dequant(codec, words_in, scale, tid))
            gpu.barrier()

        def _row_in_registers(row):
            words, scale_word, _leader = _codec_quant(codec, _load_atom(row), lane, tid)
            scale = _scale_from_word(codec, scale_word, pair_in_slot)
            _store_atom(row, _codec_dequant(codec, words, scale, tid))

        n_block_rows = (num_rows - bid + n_blocks - fx.Int32(1)) // n_blocks
        for i in range(fx.Int32(0), n_block_rows, fx.Int32(1)):
            row = bid + i * n_blocks
            # Python-level, so exactly one body is traced and neither branch
            # assigns anything the rewriter would have to thread as state.
            if via_memory:
                _row_via_memory(row)
            else:
                _row_in_registers(row)

    flat_wg = f"{BLOCK},{BLOCK}"

    @flyc.jit
    def launch(
        num_rows: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        codec_roundtrip(
            num_rows,
            inp_ptr,
            out_ptr,
            grid_x,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    tag = f"{codec_name}_{'mem' if via_memory else 'reg'}"
    launch.func.__name__ = f"launch_codec_roundtrip_{tag}"
    try:
        codec_roundtrip.func.__name__ = f"codec_roundtrip_{tag}"
    except AttributeError:
        pass
    return launch


@functools.cache
def _engine(codec_name: str, via_memory: bool):
    return [make_codec_roundtrip_kernel(codec_name, via_memory), None]


def codec_roundtrip(
    x: torch.Tensor, codec_name: str, *, via_memory: bool = True
) -> torch.Tensor:
    """Quantize and dequantize *x* with the real kernel codec.

    *x* is bf16 on a GPU with a whole number of :data:`TILE_ELEMS`.
    ``via_memory=False`` skips the LDS staging and keeps the quantized words in
    registers; the two must agree bit for bit.
    """
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("codec_roundtrip needs a bf16 CUDA tensor")
    if x.numel() % TILE_ELEMS:
        raise ValueError(f"numel must be a multiple of {TILE_ELEMS}, got {x.numel()}")
    x = x.contiguous()
    out = torch.empty_like(x)
    num_rows = x.numel() // TILE_ELEMS
    grid_x = max(1, min(num_rows, _GRID_CAP))
    eng = _engine(codec_name, via_memory)
    args = (
        Int32(num_rows),
        Int64(int(x.data_ptr())),
        Int64(int(out.data_ptr())),
        Int32(grid_x),
        Stream(None),
    )
    if eng[1] is None:
        eng[1] = flyc.compile(eng[0], *args)
    else:
        eng[1](*args)
    torch.cuda.synchronize()
    return out


# A quantization group is one E4M3 scale's worth of values: PAIR=2 threads x 8
# values per atom. Thread t owns elements [8t, 8t+8) of an atom row and pairs
# with its xor-1 neighbour, so groups are contiguous 16-element runs -- and an
# atom row is 2048 elements, a multiple of 16, so a flat reshape lines up.
GROUP_ELEMS = 16

# Half-ulp of a 3-bit mantissa: the decoded extremum is within this fraction of
# the true one.
E4M3_REL_SLACK = 1.0 / 16.0

CODEC_NAMES = ("int4", "int6")


def _payload(*, n_tiles: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_tiles * TILE_ELEMS, generator=g, dtype=torch.float32) * scale
    return x.to(device="cuda:0", dtype=torch.bfloat16)


def _groups(t: torch.Tensor) -> torch.Tensor:
    return t.float().reshape(-1, GROUP_ELEMS)


def _signed_extremum(g: torch.Tensor) -> torch.Tensor:
    mx, mn = g.max(-1).values, g.min(-1).values
    return torch.where(mx.abs() > mn.abs(), mx, mn)


def _err_bound(bias: int) -> float:
    """Largest ``|y-x| / |ext|`` the codec may produce, from its definition.

    ``1/16`` for the E4M3 scale, ``0.5/bias`` for rounding, and up to another
    ``0.5/bias`` because the positive end of the range stops one code short of
    the negative end. Rounded up to ``1.05/bias`` for fp16 arithmetic.
    """
    return E4M3_REL_SLACK + 1.05 / bias


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_memory_path_matches_register_path(codec_name):
    """Staging through the wire layout must not change a single bit.

    This is the store/load consistency check. It would catch the two sides
    disagreeing about where the INT6 2-bit plane lives, or a half-swap between
    the threads that share one of its i32 slots.
    """
    x = _payload(n_tiles=2, seed=17)
    through_lds = codec_roundtrip(x, codec_name, via_memory=True)
    in_regs = codec_roundtrip(x, codec_name, via_memory=False)
    mismatch = int((through_lds != in_regs).sum())
    assert mismatch == 0, (
        f"{mismatch}/{x.numel()} elements differ between the staged and "
        "register paths: the wire layout is not round-tripping"
    )


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_error_within_analytic_bound(codec_name):
    x = _payload(n_tiles=4, seed=23)
    y = codec_roundtrip(x, codec_name)
    xg, yg = _groups(x), _groups(y)
    ext = _signed_extremum(xg).abs().clamp_min(1e-20)
    worst = float(((yg - xg).abs().max(-1).values / ext).max())
    bound = _err_bound(CODECS[codec_name].bias)
    assert worst <= bound, f"max |err|/|ext| {worst:.4f} > {bound:.4f}"


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_reconstruction_stays_in_range(codec_name):
    """A saturating codec cannot amplify: nothing may exceed the group extremum."""
    x = _payload(n_tiles=4, seed=29)
    y = codec_roundtrip(x, codec_name)
    xg, yg = _groups(x), _groups(y)
    ext = _signed_extremum(xg).abs().clamp_min(1e-20)
    worst = float((yg.abs().max(-1).values / ext).max())
    assert worst <= 1.0 + E4M3_REL_SLACK + 1e-3, f"max |y|/|ext| {worst:.4f}"


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_group_extremum_keeps_its_sign(codec_name):
    """The extremum drives the scale, so it must survive with its sign intact."""
    x = _payload(n_tiles=2, seed=31)
    xg, yg = _groups(x), _groups(codec_roundtrip(x, codec_name))
    idx = xg.abs().argmax(-1, keepdim=True)
    x_ext = xg.gather(-1, idx).squeeze(-1)
    y_ext = yg.gather(-1, idx).squeeze(-1)
    live = x_ext.abs() > 1e-6
    assert bool((torch.sign(x_ext[live]) == torch.sign(y_ext[live])).all()), (
        "Sign changed"
    )
    rel = float(((y_ext[live] - x_ext[live]).abs() / x_ext[live].abs()).max())
    bound = _err_bound(CODECS[codec_name].bias)
    assert rel <= bound, f"extremum moved {rel:.4f} > {bound:.4f}"


def test_int6_codec_is_more_accurate_than_int4():
    """Two extra bits, on the same input."""
    x = _payload(n_tiles=4, seed=41)
    xf = x.float()
    rms = {
        c: float((codec_roundtrip(x, c).float() - xf).pow(2).mean().sqrt())
        for c in CODEC_NAMES
    }
    ratio = rms["int6"] / rms["int4"]
    assert 0.15 <= ratio <= 0.30, f"int6/int4 RMS ratio {ratio:.3f}, expected ~0.25"


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_degenerate_groups_stay_finite(codec_name):
    """Zero and sub-2^-7 groups drive the encode reciprocal to its ceiling.

    ``1/d`` is materialised as fp16, so without the clamp in ``_codec_quant`` a
    zero-extremum group reaches the codec as Inf and ``0 * Inf`` poisons the
    tile. INT6 has a quarter of INT4's headroom here, hence both codecs.
    """
    x = _payload(n_tiles=2, seed=43).reshape(-1, GROUP_ELEMS)
    x[0::4] = 0.0
    x[1::4] *= 1e-8
    x = x.reshape(-1)
    y = codec_roundtrip(x, codec_name)
    assert bool(torch.isfinite(y.float()).all()), "codec produced NaN or Inf"
    zeros = y.reshape(-1, GROUP_ELEMS)[0::4]
    assert bool((zeros == 0).all()), "an all-zero group did not stay zero"


# The largest representable group-extremum magnitude, and where the exponent
# field pins to its minimum instead of shrinking further.
E4M3_MAX = 480.0
E4M3_FLOOR = 2.0**-7


def _spike_group(ext_value: float, fill: float = 0.02) -> torch.Tensor:
    """One tile whose first group has extremum *ext_value*; every other element is *fill*.

    :func:`_groups` reshapes flat into 16-wide runs, so element 0 is exactly
    this group's extremum as long as *fill* cannot compete with it.
    """
    x = torch.full((TILE_ELEMS,), fill, dtype=torch.float32)
    x[0] = ext_value
    return x.to(device="cuda:0", dtype=torch.bfloat16)


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
@pytest.mark.parametrize("ext", (256.0, 400.0, 479.0, 490.0))
def test_extremum_near_e4m3_ceiling_saturates(codec_name, ext):
    """Right up to the ceiling, the extremum clamps to E4M3_MAX and stays close to true."""
    decoded = float(codec_roundtrip(_spike_group(ext), codec_name)[0])
    assert decoded <= E4M3_MAX + 1.0, decoded
    assert decoded >= min(ext, E4M3_MAX) * 0.9, decoded


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_extremum_past_e4m3_ceiling_saturates_monotonically(codec_name):
    """Past the ceiling the codec clips; it never wraps to a smaller value.

    Saturating the exponent field alone is not enough: with the mantissa left
    at whatever the input produced, 512 used to encode as 256, so an outlier
    decoded *smaller* than the ceiling it passed and the whole group's scale
    collapsed with it. The encoder pins the mantissa to its maximum on
    overflow too, so every extremum at or above the ceiling lands on the same
    value.
    """
    at_ceiling = float(codec_roundtrip(_spike_group(480.0), codec_name)[0])
    assert at_ceiling == pytest.approx(E4M3_MAX, abs=1.0)
    for ext in (500.0, 512.0, 1024.0, 1.0e6):
        decoded = float(codec_roundtrip(_spike_group(ext), codec_name)[0])
        assert decoded == pytest.approx(at_ceiling, abs=1.0), (
            f"{ext} decoded to {decoded}, not the ceiling's {at_ceiling} -- "
            "the overflow clamp has regressed to saturating the exponent alone"
        )


@pytest.mark.parametrize("codec_name", CODEC_NAMES)
def test_extremum_below_e4m3_floor_survives_instead_of_zeroing(codec_name):
    """A sub-floor extremum rounds coarsely; it does not lose the whole group.

    ``0x00`` is the byte reserved for exact zero. A positive extremum small
    enough to encode there used to collide with that sentinel, so the decoder
    read a scale of 0 and every value in the group came back as 0 -- at 1e-3,
    which is nowhere near the edge of either bf16 or fp16. The encoder now
    bumps such a value to ``0x01``, the smallest non-zero magnitude.

    Far enough below the floor it does still underflow to zero, but that is
    the format running out of exponent rather than a sentinel collision.
    """
    decoded = codec_roundtrip(_spike_group(1e-3, fill=0.0), codec_name)
    assert torch.isfinite(decoded.float()).all()
    assert float(decoded[0]) == pytest.approx(1e-3, rel=0.2), (
        f"expected the sub-floor extremum to survive; got {float(decoded[0])}"
    )
    # Exact zero still encodes as exact zero -- the bump must not perturb it.
    zeros = codec_roundtrip(_spike_group(0.0, fill=0.0), codec_name)
    assert float(zeros.float().abs().max()) == 0.0


# fp16 is the passthrough wire format used to test the reduce-scatter/
# all-gather transport in isolation.


def test_fp16_codec_roundtrip_is_identity():
    x = _payload(n_tiles=2, seed=53)
    y = codec_roundtrip(x, "fp16")
    assert torch.equal(x, y), "fp16 passthrough must not alter a single bit"


def test_fp16_codec_memory_path_matches_register_path():
    x = _payload(n_tiles=2, seed=59)
    through_lds = codec_roundtrip(x, "fp16", via_memory=True)
    in_regs = codec_roundtrip(x, "fp16", via_memory=False)
    assert torch.equal(through_lds, in_regs)


def _resolve(monkeypatch, algorithm, world_size, env=None, rs=None, ag=None):
    from aiter.ops.flydsl import quick_allreduce_int4 as host

    # Patch the parsed value rather than os.environ: the variable is read once
    # at import, which is the behaviour under test everywhere else.
    monkeypatch.setattr(host, "AITER_ALL_REDUCE_CODEC", env)
    monkeypatch.setattr(host, "_warned_codecs", set())
    return host._resolve_codecs(host.ALGORITHMS[algorithm], world_size, rs, ag)


@pytest.mark.parametrize(
    "world_size,expected",
    ((2, ("int4", "int4")), (4, ("int4", "int4")), (8, ("int6", "int4"))),
)
def test_ring_codec_defaults_widen_only_at_tp8(monkeypatch, world_size, expected):
    """TP8 is the only world size where INT4 misses the floor."""
    assert _resolve(monkeypatch, "ring", world_size) == expected


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_mesh_is_int4_at_every_world_size(monkeypatch, world_size):
    """The mesh has no separable lap, so the per-N default must not leak into it."""
    assert _resolve(monkeypatch, "mesh", world_size) == ("int4", "int4")


@pytest.mark.parametrize("env,expected", (("int4", "int4"), ("int6", "int6")))
def test_env_override_sets_both_laps(monkeypatch, env, expected):
    """One variable, both laps -- including the all-gather lap, which has no
    other way to reach INT6."""
    assert _resolve(monkeypatch, "ring", 8, env=env) == (expected, expected)


def test_explicit_argument_outranks_the_environment(monkeypatch):
    assert _resolve(monkeypatch, "ring", 8, env="int6", rs="int4") == ("int4", "int6")


def test_env_that_the_schedule_cannot_build_falls_back(monkeypatch):
    """A process-wide variable must not break an unrelated call site."""
    assert _resolve(monkeypatch, "mesh", 8, env="int6") == ("int4", "int4")


def test_explicit_codec_the_schedule_cannot_build_raises(monkeypatch):
    """Unlike the environment: naming it in code is a programming error."""
    with pytest.raises(ValueError, match="rs_codec"):
        _resolve(monkeypatch, "mesh", 8, rs="int6")
