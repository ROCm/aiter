
import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
    perftest,
)
from aiter import dtypes
import pandas as pd
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

M = 4
N = 8
topk_tokens = 2

logits = torch.randn(M, N, device='cuda', dtype=torch.float32)
rowStarts = torch.arange(0, M*N, N, device='cuda', dtype=torch.int32)
rowEnds = rowStarts + N

topk_indices = torch.empty(
                M, topk_tokens, dtype=torch.int32, device=logits.device
            )
topk_values = torch.empty(
                M, topk_tokens, dtype=logits.dtype, device=logits.device
            )

aiter.topk_per_row(
    logits,
    rowStarts,
    rowEnds,
    topk_indices,
    topk_values,
    logits.shape[0],
    logits.stride(0),
    logits.stride(1),
)
