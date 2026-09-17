# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the shape cache in front of the flash_kda kernel launches.

The cache skips Triton's per-call preamble by reusing the kernel it compiled
for a set of shapes. That is only sound while its key asks everything Triton
specialized on, and getting it wrong does not raise -- it runs a kernel
compiled under assumptions the arguments no longer meet, and returns whatever
that produces. So the tests here are about the key, not about speed:

* the cached path returns the same bytes as the ordinary one, across shapes,
  varlen, bias, initial state, and both segmented and not;
* two shapes that need different kernels get different entries, rather than
  the first one's kernel being handed to the second;
* an argument whose alignment differs is treated as different, since 16-byte
  alignment is one of the things the compiler was told it could assume.
"""

import pytest
import torch

from aiter.ops.triton._triton_kernels.chunk_delta_attn import fast_launch
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import flash_kda_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash_kda needs a GPU"
)

device = "cuda"
dtype = torch.bfloat16
K_DIM = 128
LOWER_BOUND = -5.0


def make_inputs(B, T, H, seed=0, bias=False, state=False, varlen=False):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=dt)

    args = {
        "q": rnd(B, T, H, K_DIM),
        "k": rnd(B, T, H, K_DIM),
        "v": rnd(B, T, H, K_DIM),
        "g": rnd(B, T, H, K_DIM, dt=torch.float32),
        "beta": rnd(B, T, H, dt=torch.float32),
        "A_log": rnd(H, dt=torch.float32),
        "dt_bias": rnd(H * K_DIM, dt=torch.float32) if bias else None,
        "initial_state": (rnd(B, H, K_DIM, K_DIM, dt=torch.float32) if state else None),
    }
    if varlen:
        # One packed sequence per batch entry, which is how the varlen path is
        # reached: B collapses to 1 and the bounds carry the split.
        args["q"], args["k"], args["v"], args["g"] = (
            args[n].reshape(1, B * T, H, -1) for n in ("q", "k", "v", "g")
        )
        args["beta"] = args["beta"].reshape(1, B * T, H)
        args["cu_seqlens"] = torch.arange(
            0, B * T + 1, T, device=device, dtype=torch.int32
        )
        if args["initial_state"] is not None:
            args["initial_state"] = args["initial_state"][:B]
    return args


def run(args, **kw):
    return flash_kda_fwd(
        **args,
        scale=K_DIM**-0.5,
        lower_bound=LOWER_BOUND,
        output_final_state=True,
        **kw,
    )


@pytest.mark.parametrize("B,T,H", [(1, 512, 12), (1, 4096, 12), (2, 1024, 4)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_matches_ordinary_path(B, T, H, bias, state):
    args = make_inputs(B, T, H, bias=bias, state=state)
    with fast_launch.bypassed():
        want_o, want_s = run(args)
    got_o, got_s = run(args)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


@pytest.mark.parametrize("chunks_per_seg", [0, 4])
def test_matches_ordinary_path_when_segmented(chunks_per_seg):
    args = make_inputs(1, 4096, 12)
    with fast_launch.bypassed():
        want_o, want_s = run(args, chunks_per_seg=chunks_per_seg)
    got_o, got_s = run(args, chunks_per_seg=chunks_per_seg)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


def test_matches_ordinary_path_varlen():
    args = make_inputs(3, 512, 12, varlen=True)
    with fast_launch.bypassed():
        want_o, want_s = run(args)
    got_o, got_s = run(args)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


def test_a_second_shape_does_not_reuse_the_first_entry():
    small, large = make_inputs(1, 512, 12), make_inputs(1, 1024, 12)
    run(small)  # whatever either shape needs is compiled and cached by now
    run(large)
    with fast_launch.recording() as misses:
        run(small)
        run(large)
    assert misses == [], "a warm shape should not be compiling"

    other = make_inputs(1, 2048, 12)
    with fast_launch.recording() as misses:
        run(other)
    assert misses, "a shape not seen before has to miss, not reuse an entry"


def _wrapped():
    """Every fast_launch-wrapped kernel the pipeline can reach."""
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn import (
        flash_kda_k1,
        flash_kda_k2,
    )
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    candidates = (
        fk._prepare_fast,
        fk._segment_fast,
        fk._seg_scan_fast,
        flash_kda_k1._k1_fast,
        flash_kda_k2.k2_ab_fused_fast,
    )
    return [c for c in candidates if isinstance(c, fast_launch._FastLaunch)]


def test_entries_do_not_pin_their_inputs():
    """An entry outlives the call that made it, so it must not hold its tensors.

    The binder hands back every argument bound together, which is the obvious
    thing to keep for the grid callback and also every q, k, v and workspace of
    whichever call arrived first, none of which any caller can then free.
    """
    for T_ in (256, 512, 1024):
        run(make_inputs(1, T_, 12))

    entries = 0
    for wrapper in _wrapped():
        for entry in wrapper._cache.values():
            entries += 1
            held = [v for part in entry[2:] for v in part.values()]
            tensors = [v for v in held if isinstance(v, torch.Tensor)]
            assert not tensors, f"entry holds {len(tensors)} tensors"
    assert entries, "nothing was cached, so nothing was tested"


def test_the_cache_is_bounded():
    """Past the bound the oldest entry goes, and asking for it again rebuilds it."""
    wrappers = _wrapped()
    for w in wrappers:
        w.clear()
    shapes = [256, 512, 1024, 2048]
    original = fast_launch._MAX_ENTRIES
    fast_launch._MAX_ENTRIES = 2
    try:
        for T_ in shapes:
            run(make_inputs(1, T_, 12))
        # Which kernels a shape reaches depends on the route, so the ones left
        # empty are the ones this configuration does not use.
        filled = [w for w in wrappers if w._cache]
        assert filled, "nothing was cached, so nothing was bounded"
        for w in filled:
            assert len(w._cache) <= 2, f"grew to {len(w._cache)}"

        # The first shape was evicted, so it has to miss -- and still be right.
        args = make_inputs(1, shapes[0], 12)
        with fast_launch.recording() as misses:
            got_o, got_s = run(args)
        assert misses, "an evicted shape should have been rebuilt, not reused"
    finally:
        fast_launch._MAX_ENTRIES = original
    with fast_launch.bypassed():
        want_o, want_s = run(args)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second GPU")
def test_a_second_device_does_not_reuse_the_first_entry():
    """An entry is a module loaded on one device, not just a compiled kernel.

    Nothing in the binder's specialization separates the two, so the entry is
    reachable from the wrong device, where it fails with an invalid device
    ordinal rather than producing wrong numbers.
    """
    B, T, H = 1, 512, 12
    first = make_inputs(B, T, H)
    run(first)  # the entry for this shape is now held against device 0

    torch.cuda.set_device(1)
    try:
        second = {
            k: (v.to("cuda:1") if isinstance(v, torch.Tensor) else v)
            for k, v in first.items()
        }
        with fast_launch.recording() as misses:
            got_o, got_s = run(second)
        assert misses, "a second device has to miss, not reuse device 0's module"
        with fast_launch.bypassed():
            want_o, want_s = run(second)
        assert torch.equal(got_o, want_o)
        assert torch.equal(got_s, want_s)
    finally:
        torch.cuda.set_device(0)


def test_alignment_is_part_of_the_key():
    """A tensor off a 16-byte boundary must not reach a kernel told otherwise."""
    B, T, H = 1, 512, 12
    args = make_inputs(B, T, H)
    run(args)

    # Same shape and dtype, one element into its storage, so the pointer moves
    # by 2 bytes and the divisibility Triton specialized on no longer holds.
    wide = torch.randn(B * T * H * K_DIM + 1, device=device, dtype=dtype)
    assert wide.data_ptr() % 16 == 0
    skewed = wide[1:].view(B, T, H, K_DIM)
    assert skewed.data_ptr() % 16 != 0

    misaligned = dict(args, q=skewed)
    with fast_launch.recording() as misses:
        run(misaligned)
    assert misses, "a differently aligned argument has to miss"

    with fast_launch.bypassed():
        want_o, want_s = run(misaligned)
    got_o, got_s = run(misaligned)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)
