"""Persistent-attention wrapper dispatch and input-validation checks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import aiter.ops.attention as attention_ops


@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("legacy_flydsl_flag", [False, True])
@pytest.mark.parametrize(
    "partial_indices,indptr,partial_map,expected_reduce_calls",
    [
        ([-1, -1], [0], [], 0),
        ([-1, -1], [0, 0, 0], [], 0),
        ([-1, -1], [0], [17, 23], 0),
        ([-1, 0, 1, -1], [0, 2], [0, 1], 1),
        ([0], [0, 1], [0], 1),
        ([-1, -1], [0, 0, 0], [17, 23], 1),
    ],
    ids=[
        "empty",
        "empty-partials",
        "empty-tiles",
        "mixed",
        "scratch-single",
        "padded-device-fallback",
    ],
)
def test_pa_persistent_reduce_dispatch(
    monkeypatch,
    use_asm,
    legacy_flydsl_flag,
    partial_indices,
    indptr,
    partial_map,
    expected_reduce_calls,
):
    monkeypatch.setattr(
        attention_ops, "get_gfx", Mock(return_value="gfx942" if use_asm else "gfx950")
    )
    device_properties = Mock(
        return_value=SimpleNamespace(
            gcnArchName="gfx950:sramecc+" if use_asm else "gfx942:sramecc+"
        )
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", device_properties)
    monkeypatch.setenv("AITER_PA_PS_REDUCE_FLYDSL", "1" if legacy_flydsl_flag else "0")
    reducers = {"pa_reduce_v1": Mock(), "_pa_ps_reduce_asm": Mock()}
    for name, reducer in reducers.items():
        monkeypatch.setattr(attention_ops, name, reducer)
    output = torch.empty((2, 16, 128), dtype=torch.bfloat16, device="cpu")

    def attention(*arguments, **keywords):
        assert arguments[16] == 0
        arguments[10].fill_(3)

    attention_mock = Mock(side_effect=attention)
    monkeypatch.setattr(attention_ops, "pa_ps_fwd_asm", attention_mock)
    work_info = torch.zeros((len(partial_indices), 8), dtype=torch.int32, device="cpu")
    work_info[:, 1] = torch.tensor(partial_indices, dtype=torch.int32, device="cpu")
    metadata = lambda values: torch.tensor(values, dtype=torch.int32, device="cpu")

    def forbid_data_read(*arguments, **keywords):
        raise AssertionError("Dispatch must not read tensor values on the host")

    with monkeypatch.context() as patch:
        for method in ("item", "tolist", "cpu", "__bool__"):
            patch.setattr(torch.Tensor, method, forbid_data_read)
        logits, final_lse = attention_ops.pa_persistent_fwd(
            output.clone(),
            torch.empty((1, 1, 8, 16, 16), device="cpu"),
            torch.empty((1, 1, 1, 128, 16), device="cpu"),
            output,
            1,
            metadata([0, 1, 2]),
            metadata([0, 1, 2]),
            metadata([0, 0]),
            metadata([16, 16]),
            metadata([0, len(partial_indices)]),
            work_info,
            metadata(indptr),
            torch.zeros((len(indptr) - 1, 2), dtype=torch.int32, device="cpu"),
            metadata(partial_map),
            mask=0,
        )
    assert logits.shape == (len(partial_map), 1, 16, 128)
    assert final_lse.shape == (2, 16)
    assert torch.isnan(final_lse).all()
    torch.testing.assert_close(output, torch.full_like(output, 3))
    attention_mock.assert_called_once()
    if expected_reduce_calls:
        device_properties.assert_called_once_with(output.device)
    else:
        device_properties.assert_not_called()
    selected = "_pa_ps_reduce_asm" if use_asm else "pa_reduce_v1"
    assert reducers[selected].call_count == expected_reduce_calls
    for name, reducer in reducers.items():
        if name != selected:
            reducer.assert_not_called()


@pytest.mark.parametrize(
    "query_heads,kv_heads,mask,message",
    [(17, 2, 1, "divisible"), (8, 0, 1, "positive"), (16, 1, 0, "causal masking")],
)
def test_pa_ps_rejects_invalid_geometry(query_heads, kv_heads, mask, message):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("page16 PS requires gfx950")
    device = torch.device("cuda", torch.cuda.current_device())
    query = torch.empty((1, query_heads, 128), dtype=torch.bfloat16, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=message):
        attention_ops.pa_ps_fwd_asm(
            Q=query,
            K=torch.empty((1, kv_heads, 16, 16, 8), dtype=query.dtype, device=device),
            V=torch.empty((1, kv_heads, 2, 128, 8), dtype=query.dtype, device=device),
            kv_indptr=indptr,
            kv_page_indices=torch.zeros(1, dtype=torch.int32, device=device),
            context_lens=torch.full((1,), 16, dtype=torch.int32, device=device),
            softmax_scale=128**-0.5,
            max_qlen=1,
            K_QScale=None,
            V_QScale=None,
            out_=torch.empty_like(query),
            qo_indptr=indptr,
            work_indptr=torch.zeros(
                torch.cuda.get_device_properties().multi_processor_count + 1,
                dtype=torch.int32,
                device=device,
            ),
            work_info=torch.zeros((1, 8), dtype=torch.int32, device=device),
            splitData=torch.empty((1, 1, query_heads, 128), device=device),
            splitLse=torch.empty((1, 1, query_heads, 1), device=device),
            mask=mask,
            quant_type=attention_ops.QuantType.No,
        )


@pytest.mark.parametrize("query_length,query_tiles", [(1, 1), (9, 2), (17, 3)])
def test_pa_metadata_allocation_query_tiles(monkeypatch, query_length, query_tiles):
    monkeypatch.setattr(torch.cuda, "current_device", Mock(return_value=0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        Mock(return_value=SimpleNamespace(multi_processor_count=256)),
    )
    baseline = attention_ops.get_pa_metadata_info_v1(3, 2)
    shapes = attention_ops.get_pa_metadata_info_v1(
        3, 2, max_seqlen_qo=query_length, num_heads_per_head_k=16
    )
    assert shapes[:2] == baseline[:2]
    assert shapes[2] == ((258 * query_tiles * 2, 8), torch.int32)
    assert shapes[3] == (3 * query_tiles + 1, torch.int32)
    assert shapes[4] == ((3 * query_tiles, 2), torch.int32)
    assert shapes[5] == (258 * query_tiles, torch.int32)
    if query_tiles == 1:
        assert shapes == baseline
