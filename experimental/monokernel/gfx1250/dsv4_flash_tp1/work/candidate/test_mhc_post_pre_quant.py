import os
from pathlib import Path
import pytest
import torch
from aiter.ops.triton.fusions.mhc import mhc_post_pre_quant_gfx1250

GOLDEN = Path(os.environ.get("MHC_GOLDEN_DIR", "/work/golden"))
assert list(GOLDEN.glob("*.pt")), (
    "Generate original baseline fixtures with make_golden.py first"
)


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.pt")), ids=lambda p: p.stem)
def test_original_outputs_and_fp8(path):
    data = torch.load(path, weights_only=True)
    args = tuple(t.cuda() for t in data["args"])
    norm = data["norm"].cuda()
    expected = tuple(t.cuda() for t in data["expected"])
    for _ in range(2):
        actual = mhc_post_pre_quant_gfx1250(
            *args, norm_weight=norm, res_preshuffle=data["shuffled"]
        )
        for got, ref in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        for got, ref in zip(actual[4:], expected[4:]):
            torch.testing.assert_close(
                got.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_post_pre_quant_gfx1250(
            *args, norm_weight=norm, res_preshuffle=data["shuffled"]
        )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for got, ref in zip(captured, expected):
        torch.testing.assert_close(
            got.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
        )
