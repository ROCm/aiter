# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Sol-Attn routing/preprocessing: kernel ABI contract and torch.compile traceability.

sol_prepare must be traceable under torch.compile(fullgraph=True). That is a hard requirement
of the entrypoint design, not a nice-to-have: a caller compiling an attention layer around Sol-Attn
must be able to trace straight through the routing, rather than hiding it in an opaque custom op of
its own. Two properties are what make it possible, and both are tested here rather than assumed:

  1. every output shape is a function of the input SHAPES alone, never of the values;
  2. no host-side branch reads device data.

Both were previously violated by the same code: a `counts[-1].item()` read of a token count the host
already knew from the shapes, and an `if empty.any()` guard around an idempotent term. This suite
pins the replacements as EQUIVALENT, not merely traceable.

The ASM kernel is not involved; this is pure Torch/Triton preprocessing and runs on any GPU.
"""

import pytest
import torch
import torch._dynamo

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_TS_KV,
    SOL_ATTN_TS_QO,
    _sol_attn_pool_q,
    _sol_attn_route,
    sol_prepare,
)

BETA = 0.5


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo caches between tests so each gets a clean compile."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, d=128, seed=0):
    """BSHD fp8 operands with a shared smooth positional component.

    Routing only has structure to find when attention concentrates; independent noise produces a
    near-uniform proxy, which would let a broken threshold still look plausible.
    """
    torch.manual_seed(seed)
    walk = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda") / (seqlen_k**0.5)
    traj = walk.cumsum(dim=1)
    traj = traj / traj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    amp = 2.0 * (d**0.5)

    q = torch.randn(batch, seqlen_q, nhead_q, d, device="cuda")
    k = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda")
    v = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda")
    q_traj = (
        torch.nn.functional.interpolate(
            traj.permute(0, 2, 3, 1).reshape(batch * nhead_kv, d, seqlen_k),
            size=seqlen_q,
            mode="linear",
            align_corners=False,
        )
        .reshape(batch, nhead_kv, d, seqlen_q)
        .permute(0, 3, 1, 2)
    )
    q = q + amp * q_traj.repeat_interleave(nhead_q // nhead_kv, dim=2)
    k = k + amp * traj
    to_fp8 = lambda x: (x / x.abs().amax() * torch.finfo(dtypes.fp8).max).to(dtypes.fp8)
    return to_fp8(q), to_fp8(k), to_fp8(v)


def _expected_shapes(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, d=128):
    num_q_tiles = -(-seqlen_q // SOL_ATTN_TS_QO)
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    work_items = batch * nhead_q * num_q_tiles
    bitmap_ds = 4 * (-(-num_kv_blocks // 128))
    return {
        "mean_k": ((batch, num_kv_blocks, nhead_kv, d), dtypes.fp8),
        "mean_v": ((batch, num_kv_blocks, nhead_kv, d), dtypes.fp8),
        "block_bitmap": ((work_items, bitmap_ds), torch.uint32),
        "kv_block_indices": ((work_items * num_kv_blocks,), torch.int32),
        "lut_start": ((work_items,), torch.int32),
        "lut_count": ((work_items,), torch.int32),
        "block_attn_mask": (
            (batch, nhead_q, num_q_tiles, num_kv_blocks),
            torch.bool,
        ),
    }


# Aligned and ragged sequence lengths, MHA and GQA. 9419 is a real Wan video shape and is a
# multiple of neither tile size.
SHAPES = [
    (1, 4096, 4096, 8, 8),
    (1, 4096, 4096, 8, 2),
    (2, 512, 1024, 4, 4),
    (1, 9419, 9419, 5, 5),
    (1, 257, 129, 4, 4),
]


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_output_contract(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """Shapes and dtypes the kernarg ABI depends on, at aligned and ragged sequence lengths."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(q, k, v, BETA)

    for name, (shape, dtype) in _expected_shapes(
        batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
    ).items():
        assert tuple(prep[name].shape) == shape, f"{name} shape"
        assert prep[name].dtype == dtype, f"{name} dtype"
        assert prep[name].is_contiguous(), f"{name} must be contiguous"

    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    assert prep["num_kv_blocks"] == num_kv_blocks
    assert prep["num_q_tiles"] == -(-seqlen_q // SOL_ATTN_TS_QO)
    assert prep["bitmap_Ds"] == 4 * (-(-num_kv_blocks // 128))


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_shapes_do_not_depend_on_values(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """Same shapes for wildly different selections.

    This is the property that lets a caller trace through the routing. beta=-4 selects almost every
    block and beta=4 almost none, so any output whose size tracked the selection would differ.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    dense = sol_prepare(q, k, v, -4.0)
    sparse = sol_prepare(q, k, v, 4.0)

    assert (
        dense["block_attn_mask"].sum() > sparse["block_attn_mask"].sum()
    ), "beta did not change the selection, so this test proves nothing"
    for name in ("mean_k", "mean_v", "block_bitmap", "kv_block_indices", "lut_start"):
        assert (
            dense[name].shape == sparse[name].shape
        ), f"{name} size is value-dependent"


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_lut_and_bitmap_agree_with_mask(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """The two forms the kernel consumes must encode exactly the routed mask.

    The kernel reads selection twice, as a ragged LUT for the exact pass and as a bitmap for the
    approximate pass. A disagreement would double-count or drop a block's mass rather than fail.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(q, k, v, BETA)
    mask = prep["block_attn_mask"]
    num_kv_blocks = mask.shape[-1]
    flat = mask.reshape(-1, num_kv_blocks)

    assert torch.equal(prep["lut_count"], flat.sum(-1, dtype=torch.int32))
    expected_start = (
        torch.cumsum(prep["lut_count"], 0, dtype=torch.int32) - prep["lut_count"]
    )
    assert torch.equal(prep["lut_start"], expected_start)

    # Every row, not a sample of them: nonzero() returns row-major order, so its column indices are
    # exactly the concatenation of the per-row block lists that lut_start and lut_count carve up.
    total = int(prep["lut_count"].sum())
    assert torch.equal(
        prep["kv_block_indices"][:total], flat.nonzero()[:, 1].to(torch.int32)
    ), "the LUT does not list the mask's selected blocks, in row-major order"

    bits = prep["block_bitmap"].to(torch.int64)
    unpacked = (
        (bits.unsqueeze(-1) >> torch.arange(32, device=bits.device))
        .bitwise_and(1)
        .bool()
        .reshape(bits.shape[0], -1)
    )
    assert torch.equal(unpacked[:, :num_kv_blocks], flat), "bitmap disagrees with mask"
    # Padding bits mean "already computed exactly" and are what clips the last tile's overhang.
    assert unpacked[:, num_kv_blocks:].all(), "bitmap padding bits must be set"


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_pooled_kv_matches_an_independent_block_mean(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """mean_k and mean_v must be the per-block mean of the stored values, tail block included.

    Nothing else pins their values. The routing tests consume mean_k rather than deriving it, and the
    kernel comparison feeds the same pooled tensors to both sides, so a pooling error would agree with
    itself everywhere and surface only as an unexplained accuracy gap. The ragged tail is the case
    that matters: dividing a short final block by the full block size scales it down by up to 128x
    while leaving every shape, dtype and contiguity check green.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(q, k, v, BETA)
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)

    for name, source in (("mean_k", k), ("mean_v", v)):
        pooled = prep[name]
        for block in range(num_kv_blocks):
            lo = block * SOL_ATTN_TS_KV
            hi = min(lo + SOL_ATTN_TS_KV, seqlen_k)
            expected = (source[:, lo:hi].float().sum(dim=1) / (hi - lo)).to(
                source.dtype
            )
            assert torch.equal(pooled[:, block], expected), (
                f"{name} block {block} covers tokens {lo}:{hi} ({hi - lo} of "
                f"{SOL_ATTN_TS_KV}) and does not equal their mean"
            )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_every_work_item_selects_at_least_one_block(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """Kernel ABI invariant: the sparse prologue preloads LUT[0] unconditionally."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(q, k, v, 8.0)  # threshold high enough to clear a row
    assert int(prep["lut_count"].min()) >= 1


def test_empty_row_rule_matches_the_guarded_form():
    """The unconditional empty-row term is equivalent to the `if empty.any()` version it replaced.

    Guarding it saved nothing but a cheap elementwise op, and cost a device sync and a graph break.
    A flat proxy is constructed here so that the rule actually fires: with an all-equal proxy row
    nothing exceeds mean + beta * std, so every row is empty and must fall back to its argmax.
    """
    batch, nhead, tiles, blocks, d = 1, 2, 3, 16, 128
    q_mean = torch.ones(batch, tiles, nhead, d, device="cuda")
    k_mean = torch.ones(batch, blocks, nhead, d, device="cuda")
    k_mean[:, 5] += 1e-3  # one block wins the argmax

    selected = _sol_attn_route(q_mean, k_mean, BETA, partial_tail=False)
    assert int(selected.sum(-1).min()) >= 1, "flat proxy must fall back to argmax"
    assert bool(selected[..., 5].all()), "fallback must keep the highest-proxy block"

    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_mean.float())
    guarded = proxy > (
        proxy.mean(-1, keepdim=True)
        + BETA * proxy.std(-1, unbiased=False, keepdim=True)
    )
    empty = ~guarded.any(dim=-1, keepdim=True)
    if bool(empty.any()):  # the host branch that used to be here
        guarded = guarded | (
            empty & torch.nn.functional.one_hot(proxy.argmax(-1), blocks).bool()
        )
    assert torch.equal(selected, guarded)


@pytest.mark.parametrize("seqlen_k", [128, 129, 255, 256, 1024, 9419])
def test_partial_tail_rule_matches_the_device_side_count(seqlen_k):
    """`seqlen_k % BLOCK_N != 0` is exactly the `counts[-1].item() != BLOCK_N` it replaced.

    The token-count tensor was only ever built from the shapes, so reading its last element back to
    the host round-tripped a value the host already had, for a full device sync.
    """
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    pad = num_kv_blocks * SOL_ATTN_TS_KV - seqlen_k
    counts = torch.full(
        (num_kv_blocks,), SOL_ATTN_TS_KV, dtype=torch.int64, device="cuda"
    )
    if pad:
        counts[-1] = SOL_ATTN_TS_KV - pad
    assert (int(counts[-1].item()) != SOL_ATTN_TS_KV) == (
        seqlen_k % SOL_ATTN_TS_KV != 0
    )

    q, k, v = _operands(1, 512, seqlen_k, 4, 4)
    mask = sol_prepare(q, k, v, 8.0)["block_attn_mask"]
    if seqlen_k % SOL_ATTN_TS_KV:
        assert bool(
            mask[..., -1].all()
        ), "a partial tail block must be computed exactly"


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_routing_matches_reference_threshold(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """tau = mean_j(proxy) + beta * population_std_j(proxy), against an independent computation."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(q, k, v, BETA)

    q_mean = _sol_attn_pool_q(q, SOL_ATTN_TS_QO)
    k_rep = prep["mean_k"].float().repeat_interleave(nhead_q // nhead_kv, dim=2)
    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_rep)
    tau = proxy.mean(-1, keepdim=True) + BETA * proxy.std(
        -1, unbiased=False, keepdim=True
    )
    expected = proxy > tau
    if seqlen_k % SOL_ATTN_TS_KV:
        expected[..., -1] = True
    empty = ~expected.any(dim=-1, keepdim=True)
    expected = expected | (
        empty & torch.nn.functional.one_hot(proxy.argmax(-1), proxy.shape[-1]).bool()
    )
    assert torch.equal(prep["block_attn_mask"], expected)


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_compiles_fullgraph_and_matches_eager(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """fullgraph=True raises on any graph break, so compiling at all is the assertion.

    Equality with eager is checked too: a silently different routing would still be a valid mask and
    would only show up much later as an accuracy regression.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    eager = sol_prepare(q, k, v, BETA)
    compiled = torch.compile(sol_prepare, fullgraph=True, dynamic=False)(q, k, v, BETA)

    for name, value in eager.items():
        if isinstance(value, torch.Tensor):
            if name == "kv_block_indices":
                # Only the spans named by lut_start/lut_count are meaningful; the rest of the
                # overallocated buffer is uninitialized in both paths.
                total = int(eager["lut_count"].sum())
                assert torch.equal(compiled[name][:total], value[:total]), name
            else:
                assert torch.equal(compiled[name], value), name
        else:
            assert compiled[name] == value, name


def test_no_graph_breaks_and_routing_is_in_the_graph():
    """Explicit break accounting, so a regression names the break instead of failing obscurely."""
    q, k, v = _operands(1, 4096, 4096, 8, 8)
    explained = torch._dynamo.explain(sol_prepare)(q, k, v, BETA)
    assert (
        explained.graph_break_count == 0
    ), f"graph breaks: {[str(r) for r in explained.break_reasons]}"
    assert explained.graph_count == 1


@pytest.mark.skipif(
    get_gfx() != "gfx950", reason="the MXFP4 and MXFP6 packers are gfx950 kernels"
)
@pytest.mark.parametrize(
    "k_format, v_format, k_variance",
    [
        ("mxfp4", "mxfp4_fp6_p", False),
        ("mxfp4", "mxfp4_fp6_p", True),
        ("mxfp6", "mxfp6_fp6_p", True),
    ],
)
def test_packed_path_compiles_fullgraph_and_matches_eager(
    k_format, v_format, k_variance
):
    """The packed path reaches production packers and rebuilds strided views over raw buffers.

    None of that is obviously traceable -- it calls out to custom ops and lands on torch.as_strided
    for both the K/V views and the V-scale slack -- so the same fullgraph requirement the rest of
    this suite pins for the ordinary path is pinned here. Strides are compared as well as values,
    because a view rebuilt with the right contents and the wrong stride would still feed the kernel
    a wrong descriptor. A packed K's variance goes through the same K packer, so it is pinned too.
    """
    batch, seqlen_k, nhead = 1, 16 * SOL_ATTN_TS_KV, 2
    q = torch.randn(batch, 512, nhead, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nhead, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = {
        "k_source": k,
        "v_source": v,
        "k_packed_format": k_format,
        "v_packed_format": v_format,
        "k_variance": k_variance,
    }

    eager = sol_prepare(q, k, v, BETA, **kwargs)
    compiled = torch.compile(sol_prepare, fullgraph=True, dynamic=False)(
        q, k, v, BETA, **kwargs
    )

    names = ["mean_k", "mean_v", "mean_k_scale", "mean_v_scale"]
    if k_variance:
        names += ["mean_k_var", "mean_k_var_scale"]
        assert eager["mean_k_var"].shape == eager["mean_k"].shape
        assert eager["mean_k_var"].stride() == eager["mean_k"].stride()
        assert eager["mean_k_var_scale"].shape == eager["mean_k_scale"].shape
    for name in names:
        assert eager[name] is not None, name
        assert compiled[name].shape == eager[name].shape, name
        assert compiled[name].stride() == eager[name].stride(), name
        assert torch.equal(compiled[name], eager[name]), name

    # The V-scale gather reads a tile past the image, so the slack has to be real storage.
    scale = eager["mean_v_scale"]
    slack = scale.untyped_storage().size() - scale.numel() * scale.element_size()
    assert slack >= 512, f"only {slack} bytes of slack behind the pooled V scale"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k_packed_format": "mxfp4"}, "k_source and k_packed_format go together"),
        ({"v_source": "v"}, "v_source and v_packed_format go together"),
        ({"k_packed_format": "fp3", "k_source": "k"}, "is not one of"),
        # The FP6-P names describe a V layout, and plain MXFP6 has no V packing of its own here.
        ({"k_packed_format": "mxfp4_fp6_p", "k_source": "k"}, "is not one of"),
        ({"v_packed_format": "mxfp6", "v_source": "v"}, "is not one of"),
        # No Sol-Attn row reads a column-major MXFP4 V.
        ({"v_packed_format": "mxfp4", "v_source": "v"}, "is not one of"),
    ],
)
def test_packed_path_rejects_an_incoherent_request(kwargs, message):
    """A packed operand needs both its source and its format; neither implies the other."""
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    resolved = {
        key: {"k": k, "v": v}.get(value, value) for key, value in kwargs.items()
    }
    with pytest.raises(ValueError, match=message):
        sol_prepare(q, k, v, BETA, **resolved)


@pytest.mark.skipif(
    get_gfx() != "gfx950", reason="the MXFP4 packers are gfx950 kernels"
)
def test_packed_path_rejects_a_stored_scale_it_cannot_pool():
    """A packed operand is quantized again from its source, so a stored scale has no meaning here.

    Accepting one silently would be the bad failure: the pooled tensor would come back correct and
    the argument would simply have been ignored.
    """
    batch, seqlen_k, nhead = 1, 2 * SOL_ATTN_TS_KV, 2
    q = torch.randn(batch, 256, nhead, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nhead, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = torch.full(
        (batch, seqlen_k, nhead, 4), 127, dtype=torch.uint8, device="cuda"
    )

    with pytest.raises(ValueError, match="k_scale does not apply to a packed operand"):
        sol_prepare(q, k, v, BETA, k_scale=scale, k_source=k, k_packed_format="mxfp4")


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_a_supplied_mask_replaces_routing_and_leaves_pooling_alone(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """A supplied selection must reach BOTH consumed forms, and must not disturb the pooled K/V.

    The split matters: pooling reduces the sequence axis and knows nothing about selection, so a
    supplied mask has to change the LUT and the bitmap and nothing else. If it perturbed the pooled
    tensors, timing a Sol-Attn row against a sparse row over one mask would no longer be comparing
    the same approximate work, and a caller supplying a fixed pattern would silently get different
    pooled operands than the routed path builds.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    routed = sol_prepare(q, k, v, BETA)
    num_q_tiles, num_kv_blocks = routed["num_q_tiles"], routed["num_kv_blocks"]

    torch.manual_seed(7)
    supplied = (
        torch.rand(batch, nhead_q, num_q_tiles, num_kv_blocks, device="cuda") > 0.5
    )
    prep = sol_prepare(q, k, v, block_attn_mask=supplied)

    for name in ("mean_k", "mean_v"):
        assert torch.equal(prep[name], routed[name]), f"{name} depends on the selection"

    used = prep["block_attn_mask"]
    flat = used.reshape(-1, num_kv_blocks)
    assert torch.equal(prep["lut_count"], flat.sum(-1, dtype=torch.int32))
    total = int(prep["lut_count"].sum())
    assert torch.equal(
        prep["kv_block_indices"][:total], flat.nonzero()[:, 1].to(torch.int32)
    ), "the LUT does not list the supplied mask's blocks"

    bits = prep["block_bitmap"].to(torch.int64)
    unpacked = (
        (bits.unsqueeze(-1) >> torch.arange(32, device=bits.device))
        .bitwise_and(1)
        .bool()
        .reshape(bits.shape[0], -1)
    )
    assert torch.equal(unpacked[:, :num_kv_blocks], flat), "bitmap ignored the mask"

    # Only the partial-tail column may differ from what was handed in.
    if seqlen_k % SOL_ATTN_TS_KV == 0:
        assert torch.equal(used, supplied)
    else:
        assert used[
            ..., -1
        ].all(), "a short tail block must be forced onto the exact pass"
        assert torch.equal(used[..., :-1], supplied[..., :-1])


def test_a_supplied_mask_and_beta_are_exclusive():
    """Routing and a supplied selection are alternatives, so silently preferring one would hide a
    caller's mistake: passing both usually means the mask was expected to win."""
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="exactly one of beta"):
        sol_prepare(q, k, v, BETA, block_attn_mask=mask)
    with pytest.raises(ValueError, match="exactly one of beta"):
        sol_prepare(q, k, v)


@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.zeros(1, 2, 1, 2, dtype=torch.bool), "block_attn_mask must be"),
        (torch.zeros(1, 2, 2, 3, dtype=torch.bool), "block_attn_mask must be"),
        (torch.zeros(1, 2, 2, 2, dtype=torch.uint8), "must be bool"),
    ],
)
def test_a_supplied_mask_is_shape_and_dtype_checked(mask, message):
    """A wrongly shaped selection would reshape into the bitmap without complaint.

    num_work_items * num_kv_blocks is the only thing the packing arithmetic needs, so a mask that
    got num_q_tiles and num_kv_blocks the wrong way round, or carried a stale tile count, can pack
    to the right total and scramble which block each bit refers to.
    """
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    with pytest.raises(ValueError, match=message):
        sol_prepare(q, k, v, block_attn_mask=mask.cuda())


def test_a_supplied_mask_stays_fullgraph_traceable():
    """The supplied-mask branch must not cost the traceability the routed path is built around."""
    q, k, v = _operands(1, 512, 4 * SOL_ATTN_TS_KV, 2, 2)
    mask = torch.rand(1, 2, 2, 4, device="cuda") > 0.5

    eager = sol_prepare(q, k, v, block_attn_mask=mask)
    compiled = torch.compile(sol_prepare, fullgraph=True, dynamic=False)(
        q, k, v, block_attn_mask=mask
    )
    for name in ("mean_k", "mean_v", "block_bitmap", "lut_start", "lut_count"):
        assert torch.equal(compiled[name], eager[name]), name


# The error router's proxy is not scale invariant. This puts the fixture's fp8 code products at
# logits of a few nats, where the mass and spread terms both matter.
ROUTING_SCALE = 2e-6


def _tile_moments(q, block_m):
    """Per-tile mean and population variance of q, one slice at a time (ragged last tile)."""
    means, variances = [], []
    for start in range(0, q.shape[1], block_m):
        tile = q[:, start : start + block_m].float()
        means.append(tile.mean(dim=1))
        variances.append(tile.var(dim=1, unbiased=False))
    return torch.stack(means, dim=1), torch.stack(variances, dim=1)


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_error_router_matches_reference(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """The error proxy against an independent computation from tile moments and the stored var_k."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_prepare(
        q, k, v, BETA, k_variance=True, router="error", routing_scale=ROUTING_SCALE
    )

    g = nhead_q // nhead_kv
    c = ROUTING_SCALE
    q_mean, q_var = _tile_moments(q, SOL_ATTN_TS_QO)
    k_mean = prep["mean_k"].float().repeat_interleave(g, dim=2)
    k_var = (prep["mean_k_var"].float() * 512.0).repeat_interleave(g, dim=2)
    dot = lambda a, b: torch.einsum("bihd,bjhd->bhij", a, b)
    sig2 = c * c * dot(q_var + q_mean * q_mean, k_var)
    proxy = (
        c * dot(q_mean, k_mean)
        + 0.5 * sig2
        + 0.5 * c * c * dot(q_var, k_mean * k_mean)
        + torch.log(torch.expm1(sig2.clamp(1e-30, 20.0)))
    )
    tau = proxy.mean(-1, keepdim=True) + BETA * proxy.std(
        -1, unbiased=False, keepdim=True
    )
    expected = proxy > tau
    if seqlen_k % SOL_ATTN_TS_KV:
        expected[..., -1] = True
    empty = ~expected.any(dim=-1, keepdim=True)
    expected = expected | (
        empty & torch.nn.functional.one_hot(proxy.argmax(-1), proxy.shape[-1]).bool()
    )
    # The moments are formed differently here (var vs E[q^2] - mean^2), so a block within fp32
    # rounding of tau may land either side.
    mismatched = (prep["block_attn_mask"] != expected).sum().item()
    assert mismatched <= max(1, expected.numel() // 1000), mismatched

    # And it is a different selection from the mean router, or this test proves nothing.
    mean_mask = sol_prepare(q, k, v, BETA)["block_attn_mask"]
    assert not torch.equal(prep["block_attn_mask"], mean_mask)


def test_error_router_leaves_the_default_and_the_pooled_operands_alone():
    q, k, v = _operands(1, 1024, 2048, 4, 2)
    default = sol_prepare(q, k, v, BETA)
    explicit = sol_prepare(q, k, v, BETA, router="mean")
    with_var = sol_prepare(q, k, v, BETA, k_variance=True)
    error = sol_prepare(q, k, v, BETA, router="error", routing_scale=ROUTING_SCALE)
    for name in ("mean_k", "mean_v", "block_bitmap", "block_attn_mask"):
        assert torch.equal(explicit[name], default[name]), name
        assert torch.equal(with_var[name], default[name]), name
    for name in ("mean_k", "mean_v"):
        assert torch.equal(error[name], default[name]), name
    assert default["mean_k_var"] is None and error["mean_k_var"] is None


def test_error_router_takes_a_device_scale():
    """A per-tensor descale lives on the device; passing it must not need a host read."""
    q, k, v = _operands(1, 1024, 2048, 4, 2)
    as_float = sol_prepare(q, k, v, BETA, router="error", routing_scale=ROUTING_SCALE)
    as_tensor = sol_prepare(
        q,
        k,
        v,
        BETA,
        router="error",
        routing_scale=torch.tensor([ROUTING_SCALE], device="cuda"),
    )
    assert torch.equal(as_float["block_attn_mask"], as_tensor["block_attn_mask"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"router": "max"}, "router must be"),
        ({"router": "error"}, "routing_scale goes with"),
        ({"routing_scale": 1.0}, "routing_scale goes with"),
    ],
)
def test_error_router_rejects_an_incoherent_request(kwargs, message):
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    with pytest.raises(ValueError, match=message):
        sol_prepare(q, k, v, BETA, **kwargs)


def _block_variance(x):
    batch, seqlen, heads, d = x.shape
    blocks = x.float().reshape(
        batch, seqlen // SOL_ATTN_TS_KV, SOL_ATTN_TS_KV, heads, d
    )
    return blocks.var(dim=2, unbiased=False)


def test_error_router_rejects_a_supplied_mask():
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="routes from beta"):
        sol_prepare(q, k, v, block_attn_mask=mask, router="error", routing_scale=1.0)


def _e8m0_image(x, low=122, octaves=3):
    """A random E8M0 1x32 scale image for x: a per-group offset, as a channel group's typical
    magnitude sets its exponent, plus per-token jitter. Jitter alone averages out over a tile.
    """
    groups = x.shape[3] // 32
    offset = 2 * torch.arange(groups, device="cuda")
    jitter = torch.randint(0, octaves, (*x.shape[:3], groups), device="cuda")
    return (low + offset + jitter).to(torch.uint8)


def _e8m0_apply(x, scale):
    return x.float() * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=-1)


def test_q_scale_routes_as_the_dequantized_q():
    """An MXFP8 Q's codes are not proportional to Q, so routing has to see it dequantized."""
    q, k, v = _operands(1, 1024, 2048, 4, 2)
    q_scale = _e8m0_image(q)
    with_scale = sol_prepare(q, k, v, BETA, q_scale=q_scale)
    dequantized = sol_prepare(_e8m0_apply(q, q_scale), k, v, BETA)
    assert torch.equal(with_scale["block_attn_mask"], dequantized["block_attn_mask"])
    # A per-group scale that varies really does move the selection, or this proves nothing.
    codes = sol_prepare(q, k, v, BETA)
    assert not torch.equal(with_scale["block_attn_mask"], codes["block_attn_mask"])
    with pytest.raises(ValueError, match="q_scale must be"):
        sol_prepare(q, k, v, BETA, q_scale=q_scale[:, :-1])


def test_error_router_on_a_block_scaled_k_scores_the_dequantized_moments():
    """With k_scale the proxy is built from what the kernel loads: the requantized pooled mean and
    variance, dequantized by their own scales, against the dequantized Q."""
    q, k, v = _operands(1, 1024, 2048, 4, 2)
    q_scale, k_scale = _e8m0_image(q), _e8m0_image(k)
    # The top channel group's scales sit near 1, so the code products keep their magnitude.
    c = ROUTING_SCALE
    prep = sol_prepare(
        q,
        k,
        v,
        BETA,
        k_scale=k_scale,
        q_scale=q_scale,
        k_variance=True,
        router="error",
        routing_scale=c,
    )

    g = q.shape[2] // k.shape[2]
    q_mean, q_var = _tile_moments(_e8m0_apply(q, q_scale), SOL_ATTN_TS_QO)
    k_mean = _e8m0_apply(prep["mean_k"], prep["mean_k_scale"]).repeat_interleave(
        g, dim=2
    )
    k_var = _e8m0_apply(prep["mean_k_var"], prep["mean_k_var_scale"]).repeat_interleave(
        g, dim=2
    )
    dot = lambda a, b: torch.einsum("bihd,bjhd->bhij", a, b)
    sig2 = c * c * dot(q_var + q_mean * q_mean, k_var)
    proxy = (
        c * dot(q_mean, k_mean)
        + 0.5 * sig2
        + 0.5 * c * c * dot(q_var, k_mean * k_mean)
        + torch.log(torch.expm1(sig2.clamp(1e-30, 20.0)))
    )
    tau = proxy.mean(-1, keepdim=True) + BETA * proxy.std(
        -1, unbiased=False, keepdim=True
    )
    expected = proxy > tau
    empty = ~expected.any(dim=-1, keepdim=True)
    expected = expected | (
        empty & torch.nn.functional.one_hot(proxy.argmax(-1), proxy.shape[-1]).bool()
    )
    mismatched = (prep["block_attn_mask"] != expected).sum().item()
    assert mismatched <= max(1, expected.numel() // 1000), mismatched
    assert not torch.equal(
        prep["block_attn_mask"],
        sol_prepare(q, k, v, BETA, k_scale=k_scale, q_scale=q_scale)["block_attn_mask"],
    )
    # Routing needs the variance whether or not the kernel takes it; the outputs follow k_variance.
    no_var = sol_prepare(
        q, k, v, BETA, k_scale=k_scale, q_scale=q_scale, router="error", routing_scale=c
    )
    assert torch.equal(no_var["block_attn_mask"], prep["block_attn_mask"])
    assert no_var["mean_k_var"] is None and no_var["mean_k_var_scale"] is None


def test_k_variance_of_an_integer_k_is_its_codes_over_512_in_e4m3():
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    k_int = (k.float() / 448.0 * 127.0).round().clamp(-127, 127).to(torch.int8)
    prep = sol_prepare(q, k_int, v, BETA, k_variance=True)
    assert prep["mean_k"].dtype == torch.int8
    assert prep["mean_k_var"].dtype == torch.float8_e4m3fn
    assert prep["mean_k_var_scale"] is None
    expected = (_block_variance(k_int) / 512.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        prep["mean_k_var"].float(), expected.float(), rtol=0, atol=0
    )
    # Asking for the variance leaves the routing and the mean alone.
    plain = sol_prepare(q, k_int, v, BETA)
    assert torch.equal(prep["mean_k"], plain["mean_k"])
    assert torch.equal(prep["block_bitmap"], plain["block_bitmap"])


def test_k_variance_of_a_block_scaled_k_is_the_dequantized_block_variance():
    q, k, v = _operands(1, 512, 2 * SOL_ATTN_TS_KV, 2, 2)
    # Per-token, per-group exponents spread across a few octaves, so pooling really has to dequantize.
    k_scale = (
        122 + torch.randint(0, 6, (*k.shape[:3], k.shape[3] // 32), device="cuda")
    ).to(torch.uint8)
    prep = sol_prepare(q, k, v, BETA, k_scale=k_scale, k_variance=True)
    data, scale = prep["mean_k_var"], prep["mean_k_var_scale"]
    assert data.dtype == k.dtype and data.shape == prep["mean_k"].shape
    assert scale.dtype == torch.uint8 and scale.shape == prep["mean_k_scale"].shape
    factor = torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=-1)
    dequantized_k = k.float() * torch.exp2(k_scale.float() - 127.0).repeat_interleave(
        32, dim=-1
    )
    expected = _block_variance(dequantized_k)
    # One e4m3 rounding: within 1/16 of a normal value, and within half the 2^-9 subnormal step of
    # the group's scale for the small values sharing a group with a large one.
    bound = expected / 16.0 + factor * 2.0**-10
    assert ((data.float() * factor - expected).abs() <= bound).all()
    plain = sol_prepare(q, k, v, BETA, k_scale=k_scale)
    assert torch.equal(prep["mean_k"], plain["mean_k"])
    assert torch.equal(prep["mean_k_scale"], plain["mean_k_scale"])
    assert plain["mean_k_var"] is None and plain["mean_k_var_scale"] is None


def test_error_router_compiles_fullgraph_and_matches_eager():
    q, k, v = _operands(1, 9419, 9419, 5, 5)
    scale = torch.tensor([ROUTING_SCALE], device="cuda")
    kwargs = {"k_variance": True, "router": "error", "routing_scale": scale}
    eager = sol_prepare(q, k, v, BETA, **kwargs)
    compiled = torch.compile(sol_prepare, fullgraph=True, dynamic=False)(
        q, k, v, BETA, **kwargs
    )
    for name in (
        "mean_k",
        "mean_v",
        "block_attn_mask",
        "block_bitmap",
        "lut_start",
        "lut_count",
    ):
        assert torch.equal(compiled[name], eager[name]), name
    # Inductor reduces the fp32 variance in another order, so an element on an e4m3 rounding
    # boundary can land one step away (1 in ~47k here).
    a, b = eager["mean_k_var"].float(), compiled["mean_k_var"].float()
    off = a != b
    assert off.sum().item() <= max(1, a.numel() // 1000)
    assert ((a - b).abs() <= 0.125 * torch.maximum(a.abs(), b.abs()))[off].all()
