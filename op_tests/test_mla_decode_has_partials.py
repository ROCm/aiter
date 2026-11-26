# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``mla_decode_fwd(..., has_partials=False)`` on a persistent plan with no
split rows.

A sparse plan built with ``kv_granularity`` >= the row width puts each row in
one work item, so ``reduce_indptr[-1] == 0``. Passing ``has_partials=False``
leaves the split buffers empty and skips the reduce launch. The output (and
the final LSE when requested) must be bitwise equal to the default call, and the reduce dispatch
must not be reached.

CI runs this via ``python3 op_tests/test_mla_decode_has_partials.py`` (also
pytest-collectable).
"""

from unittest.mock import patch

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import get_mla_decode_shape_support

D, DV, TOPK, CTX = 576, 512, 2048, 65536
bf16 = dtypes.bf16


def _case(rows, heads):
    torch.manual_seed(rows * 1000 + heads)
    dev = "cuda"
    kv = torch.randn(CTX, 1, 1, D, dtype=bf16, device=dev)
    q = torch.randn(rows, heads, D, dtype=bf16, device=dev)
    qo = torch.arange(rows + 1, dtype=torch.int32, device=dev)
    kvp = qo * TOPK
    ind = torch.randint(0, CTX, (rows * TOPK,), dtype=torch.int32, device=dev)
    last = torch.ones(rows, dtype=torch.int32, device=dev)
    info = aiter.get_mla_metadata_info_v1(
        rows, 1, heads, bf16, bf16, is_sparse=True, fast_mode=True
    )
    w = [torch.empty(sz, dtype=dt, device=dev) for sz, dt in info]
    aiter.get_mla_metadata_v1(
        qo,
        kvp,
        last,
        heads,
        1,
        True,
        w[0],
        w[2],
        w[1],
        w[3],
        w[4],
        w[5],
        page_size=1,
        kv_granularity=TOPK,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        dtype_q=bf16,
        dtype_kv=bf16,
    )
    assert int(w[3][-1]) == 0, "whole-row plan must have no split rows"
    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    work = {name: buf for name, buf in zip(names, w)}

    def call(**kw):
        o = torch.full((rows, heads, DV), float("nan"), dtype=bf16, device=dev)
        _, lse = aiter.mla.mla_decode_fwd(
            q,
            kv,
            o,
            qo,
            kvp,
            ind,
            last,
            1,
            page_size=1,
            num_kv_splits=16,
            sm_scale=D**-0.5,
            **work,
            **kw,
        )
        torch.cuda.synchronize()
        return o, lse

    return call


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("rows", [64, 256])
def test_has_partials_false_matches_default(rows, heads, return_lse):
    support = get_mla_decode_shape_support(heads, 1, bf16, bf16, return_lse=return_lse)
    if support.supported is not True:
        pytest.skip(f"no persistent bf16 kernel for {heads} heads")
    call = _case(rows, heads)
    o_ref, lse_ref = call(return_lse=return_lse)
    with patch.object(
        aiter.mla,
        "_mla_decode_reduce_v1_dispatch",
        side_effect=AssertionError("reduce launched"),
    ):
        o, lse = call(return_lse=return_lse, has_partials=False)
    assert not o.isnan().any()
    assert torch.equal(o.view(torch.int16), o_ref.view(torch.int16))
    if return_lse:
        assert torch.equal(lse.view(torch.int32), lse_ref.view(torch.int32))
    else:
        assert lse is None and lse_ref is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
