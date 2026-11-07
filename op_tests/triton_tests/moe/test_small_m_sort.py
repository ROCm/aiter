import pytest
import torch

from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.small_m_sort import small_m_sort


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
@pytest.mark.parametrize("tokens", [1, 7, 8, 17, 32, 33, 63, 64])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("padding", [False, True])
def test_small_m_sort_preserves_routes_and_graph_output(tokens, block_size, padding):
    torch.manual_seed(337)
    ids = (
        torch.randn(tokens, 384, device="cuda", dtype=torch.float32)
        .topk(6, dim=-1)
        .indices.int()
    )
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    storage = torch.full(
        (tokens * 5120 + 32,), 9.0, device="cuda", dtype=torch.bfloat16
    )
    out = storage[16:-16].view(tokens, 5120)
    small_m_sort(ids, weights, 384, 5120, out.dtype, block_size, output=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = small_m_sort(
            ids, weights, 384, 5120, out.dtype, block_size, output=out
        )
    ids.copy_(ids.roll(1, dims=1))
    if padding:
        ids[-1].fill_(-1)
    weights.mul_(0.75)
    out.fill_(float("nan"))
    graph.replay()
    expected = moe_sorting(ids, weights, 384, 5120, out.dtype, block_size)
    extent = int(expected[3][0])
    assert torch.equal(actual[3], expected[3])
    assert torch.equal(actual[0][:extent], expected[0][:extent])
    assert torch.equal(
        actual[2][: extent // block_size], expected[2][: extent // block_size]
    )
    live = expected[0][:extent] != ((6 << 24) | tokens)
    assert torch.equal(actual[1][:extent][live], expected[1][:extent][live])
    assert out.count_nonzero() == 0
    assert (storage[:16] == 9).all() and (storage[-16:] == 9).all()
    assert small_m_sort(ids, weights, 384, 5120, out.dtype, 64) is None


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
@pytest.mark.parametrize("tokens", [0, 65])
def test_small_m_sort_returns_fallback_without_writing_output(tokens):
    ids = torch.zeros(tokens, 6, device="cuda", dtype=torch.int32)
    weights = torch.zeros(tokens, 6, device="cuda", dtype=torch.float32)
    output = torch.full((tokens, 5120), 9, device="cuda", dtype=torch.bfloat16)
    assert small_m_sort(ids, weights, 384, 5120, output.dtype, 16, output) is None
    assert (output == 9).all()


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
def test_small_m_sort_rejects_bad_output_before_writing():
    ids = torch.zeros(8, 6, device="cuda", dtype=torch.int32)
    weights = torch.zeros(8, 6, device="cuda", dtype=torch.float32)
    output = torch.full((8, 5121), 9, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="output must be"):
        small_m_sort(ids, weights, 384, 5120, output.dtype, 16, output)
    assert (output == 9).all()


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
@pytest.mark.parametrize("tokens", [1, 17, 64])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_small_m_sort_invalid_routes(tokens, block_size, all_invalid):
    route = [-1, 384, 2147483647, -2, 0, 383]
    if all_invalid:
        route[-2:] = [384, -1]
    ids = torch.tensor(route, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    weights = torch.arange(tokens * 6, device="cuda", dtype=torch.float32).view(
        tokens, 6
    )
    actual = small_m_sort(ids, weights, 384, 5120, torch.bfloat16, block_size)
    padded = ((tokens + block_size - 1) // block_size) * block_size
    extent = 0 if all_invalid else 2 * padded
    expected_counts = torch.tensor([extent, tokens], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(actual[3], expected_counts, atol=0, rtol=0)
    assert actual[4].count_nonzero() == 0
    if all_invalid:
        return
    token_ids = torch.arange(tokens, device="cuda", dtype=torch.int32)
    for group, slot, expert in [(0, 4, 0), (1, 5, 383)]:
        start = group * padded
        torch.testing.assert_close(
            actual[0][start : start + tokens], (slot << 24) | token_ids, atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual[1][start : start + tokens], weights[:, slot], atol=0, rtol=0
        )
        assert (
            actual[0][start + tokens : start + padded] == ((6 << 24) | tokens)
        ).all()
        assert actual[1][start + tokens : start + padded].count_nonzero() == 0
        assert (
            actual[2][start // block_size : (start + padded) // block_size] == expert
        ).all()


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
@pytest.mark.parametrize("alias", ["ids", "weights"])
def test_small_m_sort_rejects_input_output_alias(alias):
    output = torch.full((8, 5120), 9, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(8, 6, device="cuda", dtype=torch.int32)
    weights = torch.zeros(8, 6, device="cuda", dtype=torch.float32)
    if alias == "ids":
        ids = output.view(torch.int32).reshape(-1)[:48].view(8, 6)
    else:
        weights = output.view(torch.float32).reshape(-1)[:48].view(8, 6)
    with pytest.raises(ValueError, match="output must be"):
        small_m_sort(ids, weights, 384, 5120, output.dtype, 16, output)
    assert (output == 9).all()


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 sorting specialization")
@pytest.mark.parametrize(
    "unsupported",
    ["experts", "topk", "ids_dtype", "weights_dtype", "strides", "output_dtype"],
)
def test_small_m_sort_unsupported_inputs(unsupported):
    topk = 4 if unsupported == "topk" else 6
    experts = 128 if unsupported == "experts" else 384
    ids_dtype = torch.int64 if unsupported == "ids_dtype" else torch.int32
    weights_dtype = torch.float16 if unsupported == "weights_dtype" else torch.float32
    output_dtype = torch.float32 if unsupported == "output_dtype" else torch.bfloat16
    ids = torch.zeros(8, topk, device="cuda", dtype=ids_dtype)
    weights = torch.zeros(8, topk, device="cuda", dtype=weights_dtype)
    if unsupported == "strides":
        ids = ids.t().contiguous().t()
    output = torch.full((8, 5120), 9, device="cuda", dtype=output_dtype)
    assert small_m_sort(ids, weights, experts, 5120, output.dtype, 16, output) is None
    assert (output == 9).all()
