# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ROCm/aiter#5826: BF16 MLA must retain KV byte offsets beyond 4 GiB.

Run on gfx942 with at least 5 GiB free:
    python -m pytest op_tests/test_mla_gfx942_kv_address.py -v

The 16 cases cover persistent and stage-1 decode with 16/128 heads, a row
below/straddling/above 4 GiB, and a 256-token split-KV case above 4 GiB.
Each case first checks low-offset and same-byte rebased-pointer controls.
Only the KV pool is large; queries and attention references stay small.
"""

import pytest
import torch

HEAD_DIM = 576
VALUE_DIM = 512
ROW_BYTES = HEAD_DIM * 2
BOUNDARY_ROW = (1 << 32) // ROW_BYTES
LOW_ROW = 1024
MAX_CONTEXT = 256


@pytest.fixture(scope="module")
def kv_pool():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a ROCm GPU")

    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx942":
        pytest.skip("covers the three gfx942 BF16 decode code objects")
    rows = BOUNDARY_ROW + 1 + MAX_CONTEXT
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < rows * ROW_BYTES + (1 << 30):
        pytest.skip("requires a 4 GiB KV pool plus 1 GiB of workspace")
    # Wrapped addresses land in initialized storage, giving deterministic
    # wrong values instead of depending on allocator contents or a fault.
    return torch.zeros(rows, 1, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")


def _persistent_metadata(aiter, heads, qo_indptr, kv_indptr, last_page_lens, splits):
    sizes = aiter.get_mla_metadata_info_v1(
        1,
        1,
        heads,
        torch.bfloat16,
        torch.bfloat16,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=splits,
        max_split_per_batch=splits,
    )
    wmd, wi, wis, ri, rfm, rpm = (
        torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in sizes
    )
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last_page_lens,
        heads,
        1,
        True,
        wmd,
        wis,
        wi,
        ri,
        rfm,
        rpm,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=splits,
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
    )
    return {
        "work_meta_data": wmd,
        "work_indptr": wi,
        "work_info_set": wis,
        "reduce_indptr": ri,
        "reduce_final_map": rfm,
        "reduce_partial_map": rpm,
    }


@pytest.mark.parametrize("heads", [16, 128])
@pytest.mark.parametrize("persistent", [False, True], ids=["stage1", "persistent"])
@pytest.mark.parametrize(
    "first_page,context,splits",
    [
        (BOUNDARY_ROW - 1, 1, 1),
        (BOUNDARY_ROW, 1, 1),
        (BOUNDARY_ROW + 1, 1, 1),
        (BOUNDARY_ROW + 1, MAX_CONTEXT, 4),
    ],
    ids=["below4g", "straddling4g", "above4g", "above4g-splitkv"],
)
def test_mla_kv_address(kv_pool, heads, persistent, first_page, context, splits):
    import aiter
    from aiter.mla import mla_decode_fwd

    generator = torch.Generator(device="cpu").manual_seed(42)
    q_cpu = torch.randn(1, heads, HEAD_DIM, generator=generator).to(torch.bfloat16)
    kv_cpu = torch.randn(context, HEAD_DIM, generator=generator).to(torch.bfloat16)
    # Compute the golden result on CPU, independently of GPU page addressing.
    scores = (q_cpu[0].float() @ kv_cpu.float().T) * (HEAD_DIM**-0.5)
    reference = (torch.softmax(scores, dim=-1) @ kv_cpu[:, :VALUE_DIM].float())[None]
    q = q_cpu.cuda()
    source = kv_cpu.cuda().view(context, 1, 1, HEAD_DIM)
    kv_pool[LOW_ROW : LOW_ROW + context].copy_(source)
    placed = kv_pool[first_page : first_page + context]
    placed.copy_(source)
    torch.testing.assert_close(placed.cpu().view_as(kv_cpu), kv_cpu, rtol=0, atol=0)

    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, context], dtype=torch.int32, device="cuda")
    last_page_lens = torch.ones(1, dtype=torch.int32, device="cuda")
    kwargs = {"num_kv_splits": splits}
    if persistent:
        kwargs.update(
            _persistent_metadata(
                aiter, heads, qo_indptr, kv_indptr, last_page_lens, splits
            )
        )
    else:
        # Supply both values so the stage-1 heuristic cannot reduce the
        # requested split count for the short regression sequence.
        kwargs["num_kv_splits_indptr"] = torch.tensor(
            [0, splits], dtype=torch.int32, device="cuda"
        )

    for label, pool_view, page in (
        ("low-offset control", kv_pool, LOW_ROW),
        ("same-byte rebased control", placed, 0),
        ("pool-global page indices", kv_pool, first_page),
    ):
        indices = torch.arange(page, page + context, dtype=torch.int32, device="cuda")
        output = torch.full(
            (1, heads, VALUE_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        mla_decode_fwd(
            q,
            pool_view,
            output,
            qo_indptr,
            kv_indptr,
            indices,
            last_page_lens,
            1,
            sm_scale=HEAD_DIM**-0.5,
            **kwargs,
        )
        torch.testing.assert_close(
            output.float().cpu(), reference, atol=2e-2, rtol=2e-2, msg=label
        )
