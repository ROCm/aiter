"""Block-sparse accuracy as the selected KV-tile count grows.

The existing sparse tests all use sequence_k=512, which is four KV tiles. A kernel whose LUT walk
degrades only past that stays green: the shipped mxfp8 sparse object was correct to six tiles and
then fell from cosine 0.998 to 0.925 at seven and 0.765 at thirty-two, and the suite never saw it.
The gap was shape coverage, not the assertion -- a cosine>0.99 bound would have caught it.

Lives in its own module so it can be extended without touching test_mha_v4.py.
"""

import pytest
import torch

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    mha_v4_kv_tile,
    native_fp8_format,
)
from aiter.jit.utils.chip_info import get_gfx
from op_tests.test_mha_v4 import _mha_v4_sparse_co_available

FP8 = native_fp8_format()
_MX_SCALES = {
    "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
}

# (q/k format, v format, kwargs). K takes Q's format, as the manifest rows do.
SPARSE_RECIPES = {
    "i8fp8": (AttentionFormat.INT8, FP8, {}),
    "fp8": (FP8, FP8, {}),
    "mxfp8": (FP8, FP8, _MX_SCALES),
    "f8f6": (FP8, AttentionFormat.MXFP6, {}),
    "f6f8": (AttentionFormat.MXFP6, FP8, {}),
    "f6f4": (AttentionFormat.MXFP6, AttentionFormat.MXFP4, {}),
    "mxfp4": (AttentionFormat.MXFP4, FP8, {}),
    "f4f4": (AttentionFormat.MXFP4, AttentionFormat.MXFP4, {}),
}

# Two tiles is the baseline the growing counts are judged against; the rest span the region the
# suite never covered. 32 tiles is where the shipped mxfp8 object had lost a quarter of its cosine.
BASELINE_TILES = 2
GROWN_TILES = (4, 7, 8, 12, 16, 32)

# Each row carries its own quantization error, so the bar is that accuracy does not DEGRADE with
# tile count. Only a modest absolute floor is applied on top, to catch a row that is broken outright.
MAX_DEGRADATION = 0.01
ABSOLUTE_FLOOR = 0.95

requires_sparse = pytest.mark.skipif(
    get_gfx() != "gfx950" or not _mha_v4_sparse_co_available(),
    reason="gfx950 sorted-sparse MHA v4 code object is not deployed",
)


def _reference(q, k, v):
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (qf.shape[-1] ** -0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).transpose(1, 2)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _run(recipe_name, tiles, heads=5, sequence_q=256, all_true=True):
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    sequence_k = tiles * mha_v4_kv_tile()
    torch.manual_seed(41)
    q = torch.randn((1, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(
        (1, heads, -(-sequence_q // 256), tiles), device="cuda", dtype=torch.bool
    )
    mask[:] = all_true
    if not all_true:
        mask[:, :, :, ::2] = True
    out = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs)
    torch.cuda.synchronize()
    return out, _reference(q, k, v), mask


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
def test_mha_v4_sparse_accuracy_holds_as_tile_count_grows(recipe_name):
    """A row's own small-tile-count accuracy is its baseline; growing the count must not erode it."""
    baseline_out, baseline_ref, _ = _run(recipe_name, BASELINE_TILES)
    baseline = _cosine(baseline_out, baseline_ref)
    assert torch.isfinite(baseline_out).all(), f"{recipe_name}: non-finite at {BASELINE_TILES} tiles"
    assert baseline > ABSOLUTE_FLOOR, f"{recipe_name}: baseline cosine {baseline:.5f}"

    for tiles in GROWN_TILES:
        out, ref, _ = _run(recipe_name, tiles)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        cosine = _cosine(out, ref)
        assert cosine > ABSOLUTE_FLOOR, f"{recipe_name}: cosine {cosine:.5f} at {tiles} tiles"
        assert cosine > baseline - MAX_DEGRADATION, (
            f"{recipe_name}: accuracy degrades with tile count -- "
            f"{baseline:.5f} at {BASELINE_TILES} tiles, {cosine:.5f} at {tiles}"
        )


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
def test_mha_v4_sparse_skipping_lut_holds_as_tile_count_grows(recipe_name):
    """Same sweep with a LUT that actually skips, so the walk has to jump rather than run affine."""
    for tiles in (4, 8, 16, 32):
        out, _, mask = _run(recipe_name, tiles, all_true=False)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        assert mask.sum() > 0
        assert not bool((out == 0).all()), f"{recipe_name}: all-zero output at {tiles} tiles"


@requires_sparse
@pytest.mark.parametrize("recipe_name", ["i8fp8", "fp8", "mxfp8", "f6f8"])
def test_mha_v4_sparse_all_true_lut_matches_dense_bitwise(recipe_name):
    """An all-true LUT selects every tile, so the walk must reduce exactly to the dense one.

    Restricted to the rows whose V stays FP8. The MX-V rows resolve to a different V packing on
    their dense manifest row than on their sparse one, so they cannot be compared bitwise.
    """
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    for tiles in (2, 8, 16):
        sequence_k = tiles * mha_v4_kv_tile()
        torch.manual_seed(41)
        q = torch.randn((1, 256, 5, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, sequence_k, 5, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.ones((1, 5, 1, tiles), device="cuda", dtype=torch.bool)
        sparse = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs)
        dense = mha_v4(q, k, v, q_format, q_format, v_format, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(sparse, dense), (
            f"{recipe_name}: all-true LUT differs from dense at {tiles} tiles "
            f"(cosine {_cosine(sparse, dense):.7f})"
        )
