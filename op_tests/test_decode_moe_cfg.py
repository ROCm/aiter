# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the decode MoE chain's tuning configs (names, legality, candidates) and of the tuned rows that
name them. No GPU."""

import csv
import os

import pytest

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_cfg import (
    ROWS,
    ExpertCfg,
    RouterCfg,
    expert_candidates,
    impl_kernel_name,
    parse_impl_kernel_name,
    router_candidates,
)

ALL_ROWS = (8, 9, 16, 17, 24, 32, 33, 57, 58, 64, 65, 96, 128)
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "aiter", "configs")


@pytest.mark.parametrize("n", ALL_ROWS)
def test_defaults_round_trip_and_are_legal(n):
    e = ExpertCfg.default(n)
    assert e.legal() and ExpertCfg.parse(e.name()) == e
    assert parse_impl_kernel_name(impl_kernel_name(e)) == e
    r = RouterCfg.default(n)
    assert r.fits(n) and RouterCfg.parse(r.name()) == r


@pytest.mark.parametrize("n", (8, 16, 32, 64, 128))
def test_candidates_start_with_default_and_are_unique_and_legal(n):
    ec, rc = expert_candidates(n), router_candidates(n)
    assert ec[0] == ExpertCfg.default(n) and rc[0] == RouterCfg.default(n)
    assert len(set(ec)) == len(ec) and len(set(rc)) == len(rc)
    assert all(c.legal() and ExpertCfg.parse(c.name()) == c for c in ec)
    assert all(c.fits(n) and RouterCfg.parse(c.name()) == c for c in rc)
    # the split-K factor is a tuning field: more than the default's is enumerated
    assert len({c.split_k for c in rc}) > 1
    assert len(expert_candidates(n, full=True)) > len(ec)
    assert len(router_candidates(n, full=True)) > len(rc)


def test_split_k_candidates_cover_every_legal_factor():
    """The full enumeration skips no legal (split_k, block_k) pair at 64 rows."""
    pairs = {(c.split_k, c.block_k) for c in router_candidates(64, full=True)}
    legal = {
        (s, bk)
        for s in range(8, 97, 8)
        for bk in (32, 64, 96, 128)
        for nbw in (1, 2, 4, 8)
        for w in (2, 4, 8)
        if RouterCfg("nblocks", s, bk, w, nbw, 0).fits(64)
    }
    assert legal == pairs and len(pairs) > 7


@pytest.mark.parametrize(
    "bad", ["", "x", "sbm16_skw4", "sbm17_skw4_spf2_stn64_g1pf3_g2tn256_g2pf2_prion"]
)
def test_bad_names_are_rejected(bad):
    with pytest.raises(ValueError):
        ExpertCfg.parse(bad)
    with pytest.raises(ValueError):
        RouterCfg.parse(bad)


def test_tuned_rows_name_valid_configs():
    path = os.path.join(
        CONFIGS, "model_configs", "minimax_m3_tp4_decode_tuned_shared_fmoe.csv"
    )
    rows = list(csv.DictReader(open(path)))
    assert {int(r["token"]) for r in rows} >= {8, 16, 32, 64, 128}
    seen = set()
    for r in rows:
        n = int(r["token"])
        assert ROWS[0] <= n <= ROWS[1]
        key = tuple(
            r[k]
            for k in (
                "gfx",
                "cu_num",
                "token",
                "model_dim",
                "inter_dim",
                "expert",
                "topk",
                "shared_expert_id",
                "act_type",
                "q_dtype_a",
                "gate_mode",
            )
        )
        assert key not in seen
        seen.add(key)
        parse_impl_kernel_name(r["kernelName1"]).legal()
        assert RouterCfg.parse(r["kernelName2"]).fits(n)
