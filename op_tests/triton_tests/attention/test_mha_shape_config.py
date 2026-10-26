# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MHA forward resolver serves a tuned shape file before DEFAULT.json."""

import json
import shutil
from pathlib import Path
from unittest import mock

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention import mha as kernels
from aiter.ops.triton.attention import mha as wrapper
from aiter.ops.triton.utils.config_utils import load_config_json

DEFAULT = (
    Path(kernels.__file__).parents[2]
    / "configs/gfx950/triton/attention/mha/DEFAULT.json"
)
FWD = json.loads(DEFAULT.read_text())["fwd"]
# Real DEFAULT.json entries that DEFAULT.json does not serve this shape, so
# a bucket's tile is a tile the kernel runs and can only come from the file.
SMALL, LARGE = dict(FWD["small_head"]), dict(FWD["pe"])
SHAPE_FILE = "MHA-DQ=128-DV=128-CAUSAL=1.json"


def clear_caches():
    kernels._get_config.cache_clear()
    load_config_json.cache_clear()


@pytest.fixture
def config_dir(tmp_path):
    shutil.copy(DEFAULT, tmp_path)
    table = {"DEFAULT_FALLBACK": True, "SK_LEQ_4096": LARGE, "SK_LEQ_1024": SMALL}
    table["SK_LEQ_4k"] = SMALL
    (tmp_path / SHAPE_FILE).write_text(json.dumps(table))
    clear_caches()
    with mock.patch.object(kernels, "resolve_config_dir", return_value=str(tmp_path)):
        yield tmp_path
    clear_caches()


def resolve(**overrides):
    arguments = {"enable_dropout": False, "dtype": torch.bfloat16, "has_pe": False}
    arguments |= {"head_dim_v": 128, "causal": True, "max_seqlen_q": 2048}
    arguments |= {"max_seqlen_k": 2048, "batch_heads": 64, "head_dim_qk": 128}
    return kernels._get_config(**{**arguments, **overrides})


def default_bucket(**overrides):
    return resolve(**{**overrides, "head_dim_qk": None})


@pytest.mark.parametrize(
    "max_seqlen_k, expected",
    [(1, SMALL), (1024, SMALL), (1025, LARGE), (4096, LARGE)],
)
def test_the_smallest_bucket_that_holds_the_shape_serves_it(
    config_dir, max_seqlen_k, expected
):
    assert default_bucket(max_seqlen_k=max_seqlen_k) != expected
    assert resolve(max_seqlen_k=max_seqlen_k) == expected


def test_a_shape_past_the_largest_bucket_gets_default_json(config_dir):
    assert resolve(max_seqlen_k=4097) == default_bucket(max_seqlen_k=4097)


def test_an_any_bucket_serves_shapes_past_the_largest_bound(config_dir):
    shape_file = config_dir / SHAPE_FILE
    shape_file.write_text(
        json.dumps({**json.loads(shape_file.read_text()), "any": SMALL})
    )
    load_config_json.cache_clear()
    assert resolve(max_seqlen_k=4097) == SMALL


def test_a_bucket_whose_bound_is_not_an_integer_is_skipped(config_dir):
    with mock.patch.object(kernels._LOGGER, "warning") as warning:
        assert resolve(max_seqlen_k=2048) == LARGE
    assert "SK_LEQ_4k" in warning.call_args.args


@pytest.mark.parametrize(
    "overrides",
    [{"causal": False}, {"head_dim_qk": 192, "has_pe": True}],
)
def test_another_head_dim_or_mask_has_no_file(config_dir, overrides):
    assert resolve(**overrides) == default_bucket(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [{"dtype": torch.float16}, {"dtype": torch.float32}, {"enable_dropout": True}],
)
def test_only_bf16_without_dropout_reads_the_shape_file(config_dir, overrides):
    assert resolve(**overrides) == default_bucket(**overrides)


def test_the_public_resolver_serves_the_shape_file(config_dir):
    tile = wrapper.get_mha_fwd_config(
        torch.bfloat16, 128, 128, causal=True, max_seqlen_k=1024, batch_heads=64
    )
    assert tile == SMALL


@pytest.mark.parametrize(
    "dtype, head_dim_qk, is_fp8, has_pe",
    [(torch.bfloat16, 192, False, True), (torch.float8_e4m3fn, 128, True, False)],
)
def test_the_public_resolver_asks_gluon_for_the_entrys_tile(
    dtype, head_dim_qk, is_fp8, has_pe
):
    with (
        mock.patch.object(wrapper.types.arch_info, "is_fp8_avail", return_value=True),
        mock.patch.object(wrapper, "_get_gluon_config", return_value={"t": 1}) as get,
    ):
        tile = wrapper.get_mha_fwd_config(dtype, head_dim_qk, 128, backend="gluon")
    assert tile == {"t": 1}
    get.assert_called_once_with(is_fp8=is_fp8, has_pe=has_pe)


def test_the_public_resolver_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match="Unknown MHA backend"):
        wrapper.get_mha_fwd_config(torch.bfloat16, 128, 128, backend="Gluon")


class _Resolved(Exception):
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_varlen_forward_resolves_its_tile_through_the_public_resolver():
    batch, nhead, seqlen, head_dim = 2, 4, 64, 128
    q, k, v = (
        torch.randn(
            batch * seqlen, nhead, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(3)
    )
    cu_seqlens = torch.arange(
        0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda"
    )
    with (
        mock.patch.object(
            wrapper, "get_mha_fwd_config", side_effect=_Resolved
        ) as resolver,
        pytest.raises(_Resolved),
    ):
        wrapper.flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, causal=True
        )
    resolver.assert_called_once_with(
        torch.bfloat16,
        head_dim,
        head_dim,
        causal=True,
        dropout_p=0.0,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        batch_heads=batch * nhead,
    )
