"""Separate boundary fixtures; the original 211-node suite is unchanged."""

import runpy
from pathlib import Path

import pytest
import torch

_original = runpy.run_path(
    str(Path(__file__).with_name("test_fused_bmm_rope_kv_cache.py"))
)


@pytest.mark.parametrize(
    "variant",
    [
        "decode",
        "neox",
        "full-frequencies",
        "negative-slots",
        "query-view",
        "query-view-padded",
        "bf16-output",
        "unsupported-query-stride",
    ],
)
def test_large_prefill_dispatch_boundary(variant):
    T = 11073
    inputs = _original["_make_fp8_bmm_mla_inputs"](
        T,
        q_out_dtype=torch.bfloat16 if variant == "bf16-output" else "fp8",
        num_decode=3 if variant == "decode" else 0,
        is_neox=variant == "neox",
        reuse_freqs=variant != "full-frequencies",
    )
    if variant == "negative-slots":
        inputs["slot_mapping"][0] = -1
        inputs["slot_mapping"][-1] = -1
    if variant in ("query-view", "query-view-padded", "unsupported-query-stride"):
        width = {
            "query-view": 192,
            "query-view-padded": 256,
            "unsupported-query-stride": 320,
        }[variant]
        queries = torch.zeros((T, 8, width), dtype=torch.bfloat16, device="cuda")
        queries[..., :192].copy_(inputs["q_nope"].transpose(0, 1))
        inputs["q_nope"] = queries[..., :192].transpose(0, 1)
        if variant == "query-view-padded":
            queries[..., 192:].copy_(inputs["q_pe"])
            inputs["q_pe"] = queries[..., 192:]
        before = queries.clone()
    expected, expected_cache = _original["_reference_fp8_bmm_mla"](inputs)
    actual = _original["fused_fp8_bmm_rope_cat_and_cache_mla"](**inputs)
    _original["_assert_fp8_bmm_mla_result"](inputs, actual, expected, expected_cache)
    if variant in ("query-view", "query-view-padded", "unsupported-query-stride"):
        _original["_assert_same_tensor_bytes"](queries, before)
