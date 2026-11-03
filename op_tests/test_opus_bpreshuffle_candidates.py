# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU regressions for OPUS bpreshuffle candidate coverage."""

import ast
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def search():
    # These helpers use only kernel metadata. Load them without the tuner's
    # FlyDSL/GPU imports, as in the host-only CO parser tests.
    source = Path(__file__).resolve().parents[1] / (
        "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            in (
                "_opus_kernel_profile",
                "_opus_select_variants",
                "opus_candidates",
            )
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("_OPUS_")
                for target in node.targets
            )
        )
    ]
    namespace = {
        "math": math,
        "re": re,
        "defaultdict": defaultdict,
        "lru_cache": lru_cache,
    }
    exec(  # noqa: S102 - trusted checkout AST with CPU dependencies
        compile(tree, str(source), "exec"), namespace
    )
    return namespace


def _template(name, args):
    return f"{name}<{','.join(str(arg).lower() for arg in args)}>"


def _instance(pipeline, traits, *, bm=16, bn=32, bk=128, wm=1, wn=2, split=1, wg=1):
    return SimpleNamespace(
        B_M=bm,
        B_N=bn,
        B_K=bk,
        T_M=wm,
        T_N=wn,
        BLOCK_SIZE=wm * wn * 64,
        WG_PER_CU=wg,
        bpreshuffle_split_k=split,
        bpreshuffle_co=None,
        bpreshuffle_entry={"pipeline": pipeline, "traits": traits},
    )


def _profile(search, instance):
    geometry = (
        instance.bpreshuffle_entry["pipeline"],
        instance.B_M,
        instance.B_N,
        instance.B_K,
        instance.T_M,
        instance.T_N,
        instance.bpreshuffle_split_k,
    )
    profile, lds, alignment = search["_opus_kernel_profile"](
        geometry,
        instance.bpreshuffle_entry["traits"],
        instance.WG_PER_CU,
        instance.BLOCK_SIZE,
    )
    return geometry, profile, lds, alignment


def _composable(
    *, split=1, policy=(), stages=3, bm=16, bn=32, bk=128, wm=1, wn=2, fixed_k=7168
):
    geometry = _template(
        "opus_gemm_composable_geometry", (bm, bn, wm, wn, bk, stages, fixed_k, split)
    )
    return _template(
        "opus_gemm_composable_traits",
        (geometry, _template("opus_gemm_composable_policy", policy)),
    )


def _install_registry(monkeypatch, registry, legal=None):
    common = ModuleType("csrc.opus_gemm.opus_gemm_common")
    common.get_registry = lambda *args: registry
    common.candidate_kids_for_shape = lambda *args, **kwargs: list(
        registry if legal is None else legal
    )
    monkeypatch.setitem(sys.modules, common.__name__, common)


def test_common_returns_legal_variants_without_legacy_policy_prefilter():
    source = Path(__file__).resolve().parents[1] / "csrc/opus_gemm/opus_gemm_common.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "candidate_kids_for_shape"
    ]
    registry = {
        3: SimpleNamespace(legal=False, bpreshuffle_entry={"tune_eligible": True}),
        2: SimpleNamespace(legal=True, bpreshuffle_entry={"tune_eligible": False}),
        1: SimpleNamespace(legal=True, bpreshuffle_entry=None),
    }
    namespace = {
        "get_registry": lambda *args: registry,
        "canonical_output_dtype": lambda dtype: "bf16_t",
        "a8w8_mxscale_bpreshuffle_supports_shape": lambda inst, *shape: inst.legal,
    }
    exec(  # noqa: S102 - trusted checkout AST with CPU dependencies
        compile(tree, str(source), "exec"), namespace
    )
    assert namespace["candidate_kids_for_shape"](
        "gfx950", 16, 128, 7168, cu_num=256
    ) == [1, 2]


def test_b_direct_keeps_panel_distance_while_large_queue_deduplicates(search):
    profile = search["_opus_kernel_profile"]
    large = ("prefetch_b_vgpr", 128, 256, 256, 1, 4, 1)
    small = ("b_direct", 128, 256, 256, 1, 4, 1)
    large_profiles, small_profiles = [], []
    for distance in (1, 2):
        args = (128, 256, 3, distance)
        large_profiles.append(
            profile(
                large,
                _template(
                    "opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits",
                    args,
                ),
                1,
                256,
            )[0]
        )
        small_profiles.append(
            profile(
                small,
                _template(
                    "opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits", args
                ),
                1,
                256,
            )[0]
        )
    assert large_profiles[0] == large_profiles[1]
    assert small_profiles[0] != small_profiles[1]


@pytest.mark.parametrize(
    "pipeline,base,wrappers",
    [
        (
            "b_direct",
            _template(
                "opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits",
                (16, 32, 3, 1, 0, False, 7168, 2),
            ),
            (
                "opus_gemm_bpreshuffle_bounded_thread_traits",
                "opus_gemm_bpreshuffle_simple_tail_traits",
                "opus_gemm_bpreshuffle_native_c_traits",
            ),
        ),
        ("composable", _composable(), ("opus_gemm_composable_mfirst_traits",)),
    ],
)
def test_wrappers_remain_distinct_execution_profiles(search, pipeline, base, wrappers):
    variants = [base, *(_template(wrapper, (base,)) for wrapper in wrappers)]
    records = [_profile(search, _instance(pipeline, traits)) for traits in variants]
    assert len({record[1] for record in records}) == len(variants)
    assert len({record[2] for record in records}) == 1
    geometry = records[0][0]
    profiles = {kid: record[1] for kid, record in enumerate(records)}
    footprints = {kid: record[2] for kid, record in enumerate(records)}
    assert search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, "secondary"
    ) == set(profiles)


def test_b_direct_primary_covers_joint_routes_and_secondary_keeps_representatives(
    search,
):
    profiles, footprints = {}, {}
    for kid, (schedule, wg, caches, byte_scale) in enumerate(
        product(
            ((2, 1), (3, 1), (3, 2), (4, 2)),
            (1, 2),
            ((0, 0), (0, 2), (1, 17)),
            (False, True),
        )
    ):
        stages, ahead = schedule
        ca, cb = caches
        args = (
            16,
            32,
            stages,
            ahead,
            cb,
            False,
            7168,
            2,
            wg,
            ca,
            2,
            0,
            128,
            True,
            False,
            False,
            True,
            1,
            byte_scale,
        )
        geometry, profile, lds, _ = _profile(
            search,
            _instance(
                "b_direct",
                _template(
                    "opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits", args
                ),
                wg=wg,
            ),
        )
        profiles[kid], footprints[kid] = profile, lds
    selected = search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, "primary"
    )
    for indices in ((2, 3, 8), (8, 9, 4, 18)):
        available = {tuple(p[index] for index in indices) for p in profiles.values()}
        retained = {
            tuple(profiles[kid][index] for index in indices) for kid in selected
        }
        assert retained == available
    assert len(selected) < len(profiles)
    secondary = search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, "secondary"
    )
    assert len(secondary) < len(selected)
    for indices in ((8,), (13, 16, 18), (11, 19)):
        assert {
            tuple(profiles[kid][index] for index in indices) for kid in secondary
        } == {
            tuple(profile[index] for index in indices) for profile in profiles.values()
        }


@pytest.mark.parametrize("mode", ["primary", "secondary"])
def test_b_direct_retains_each_scale_and_output_execution_path(search, mode):
    profiles, footprints, paths = {}, {}, {}
    for kid, (output, byte_scale) in enumerate(product((0, 2), (False, True))):
        args = (
            16,
            32,
            3,
            1,
            0,
            False,
            7168,
            2,
            1,
            0,
            2,
            output,
            128,
            True,
            False,
            False,
            True,
            1,
            byte_scale,
        )
        geometry, profile, lds, _ = _profile(
            search,
            _instance(
                "b_direct",
                _template(
                    "opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits", args
                ),
            ),
        )
        profiles[kid], footprints[kid], paths[kid] = profile, lds, (output, byte_scale)
    selected = search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, mode
    )
    assert {paths[kid] for kid in selected} == set(paths.values())


def test_composable_primary_covers_loading_routes_and_secondary_keeps_mfma_modes(
    search,
):
    profiles, footprints = {}, {}
    for kid, (compact, per_stage, direct_b, normal, cb) in enumerate(
        product((False, True), (False, True), (False, True), (False, True), (0, 2))
    ):
        policy = (compact, per_stage, direct_b, 2, cb, True, True, normal)
        geometry, profile, lds, _ = _profile(
            search, _instance("composable", _composable(policy=policy))
        )
        profiles[kid], footprints[kid] = profile, lds
    selected = search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, "primary"
    )
    for indices in ((9, 10, 11, 16), (13, 18)):
        available = {tuple(p[index] for index in indices) for p in profiles.values()}
        retained = {
            tuple(profiles[kid][index] for index in indices) for kid in selected
        }
        assert retained == available
    assert len(selected) < len(profiles)
    secondary = search["_opus_select_variants"](
        geometry, list(profiles), profiles, footprints, 16, 128, "secondary"
    )
    assert len(secondary) < len(selected)
    for indices in ((10, 11), (11, 16)):
        assert {
            tuple(profiles[kid][index] for index in indices) for kid in secondary
        } == {
            tuple(profile[index] for index in indices) for profile in profiles.values()
        }


@pytest.mark.parametrize("direct_b", [False, True])
def test_composable_deduplicates_only_inactive_b_lookahead(search, direct_b):
    records = [
        _profile(
            search,
            _instance("composable", _composable(policy=(True, False, direct_b, ahead))),
        )
        for ahead in (1, 2)
    ]
    assert (records[0][1] == records[1][1]) is (not direct_b)
    assert records[0][2] == records[1][2]


@pytest.mark.parametrize("split", [1, 2])
def test_composable_keeps_xcd_when_it_changes_the_launch_path(search, split):
    records = [
        _profile(
            search,
            _instance(
                "composable",
                _composable(
                    split=split,
                    policy=(
                        True,
                        False,
                        False,
                        1,
                        0,
                        True,
                        True,
                        True,
                        False,
                        3,
                        1,
                        xcd,
                    ),
                ),
                split=split,
            ),
        )
        for xcd in (False, True)
    ]
    assert (records[0][1] == records[1][1]) is (split > 1)
    assert records[0][2] == records[1][2]


def test_composable_deduplicates_direct_stores_and_preserves_staged_store_policies(
    search,
):
    profiles = [
        _profile(
            search,
            _instance(
                "composable",
                _composable(
                    policy=(True, False, False, 1, 0, True, True, True, False, output)
                ),
            ),
        )[1]
        for output in range(4)
    ]
    assert profiles[0] == profiles[1]
    assert len(set(profiles)) == 3


def test_small_range_covers_launch_axes_without_retaining_geometry_cross_product(
    search, monkeypatch
):
    registry = {}
    for bm, bn, bk, (wm, wn), split in product(
        (16, 32, 64, 96, 128),
        (16, 32, 64, 128),
        (128, 256, 512),
        ((1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2)),
        (1, 2, 4),
    ):
        if bm % (16 * wm) or bn % (16 * wn) or 7168 % (split * bk):
            continue
        instance = _instance(
            "composable",
            _composable(bm=bm, bn=bn, bk=bk, wm=wm, wn=wn, split=split),
            bm=bm,
            bn=bn,
            bk=bk,
            wm=wm,
            wn=wn,
            split=split,
        )
        if _profile(search, instance)[2] + 16 <= 160 * 1024:
            registry[len(registry)] = instance
    _install_registry(monkeypatch, registry)
    chosen = [
        inst
        for kid, inst in search["opus_candidates"](
            16, 768, 7168, cu_num=256, small_m=False
        )
    ]
    assert len(chosen) < len(registry) // 4
    for projection in (
        lambda inst: inst.B_K,
        lambda inst: (inst.T_M, inst.T_N),
        lambda inst: inst.bpreshuffle_split_k,
    ):
        assert {projection(inst) for inst in chosen} == {
            projection(inst) for inst in registry.values()
        }


def test_selection_retains_low_lds_per_split_and_every_legal_co(search, monkeypatch):
    registry, baselines = {}, {}
    for split in (1, 2, 4):
        for compact in (False, True):
            kid = split * 10 + int(compact)
            registry[kid] = _instance(
                "composable",
                _composable(split=split, policy=(compact, False, False)),
                split=split,
            )
        baselines[split] = min(
            (split * 10, split * 10 + 1),
            key=lambda kid: (_profile(search, registry[kid])[2], kid),
        )
    for kid in (3, 1, 2):
        registry[kid] = SimpleNamespace(bpreshuffle_co={"pipeline": "pin_agpr"})
    # An unavailable/illegal CO is outside the supplied legal list.
    legal = [kid for kid in reversed(registry) if kid != 2]
    _install_registry(monkeypatch, registry, legal)
    first = search["opus_candidates"](16, 128, 7168, cu_num=256, small_m=False)
    second = search["opus_candidates"](16, 128, 7168, cu_num=256, small_m=False)
    selected = [kid for kid, _ in first]
    assert first == second
    assert selected == sorted(set(selected))
    assert set(baselines.values()) <= set(selected)
    assert {1, 3} <= set(selected)
    assert 2 not in selected


def test_split_k_contributes_to_grid_supply(search, monkeypatch):
    registry = {
        split: _instance("composable", _composable(split=split), split=split)
        for split in (1, 4)
    }
    _install_registry(monkeypatch, registry)
    modes = {}

    def capture(geometry, kids, *args):
        modes[geometry[-1]] = args[-1]
        return {min(kids)}

    monkeypatch.setitem(search, "_opus_select_variants", capture)
    search["opus_candidates"](16, 1024, 7168, cu_num=256, small_m=False)
    assert modes == {1: "secondary", 4: "primary"}


@pytest.mark.parametrize(
    "pipeline,traits",
    [
        ("unknown", "traits<16,32,3,1>"),
        ("b_direct", "traits<16>"),
        ("composable", "opus_gemm_composable_traits<16,32>"),
        ("lds_panel", "opus_gemm_bpreshuffle_native_c_traits<traits<192,256>>"),
    ],
)
def test_incomplete_or_mismatched_traits_fail_explicitly(search, pipeline, traits):
    with pytest.raises(ValueError, match="OPUS traits"):
        _profile(search, _instance(pipeline, traits))


def test_small_selection_retains_coupled_stage_load_cache_and_mfma_policies(
    search, monkeypatch
):
    registry = {}
    for kid, (stages, direct_b, cache_b, normal) in enumerate(
        product((4, 6), (False, True), (0, 2), (False, True))
    ):
        registry[kid] = _instance(
            "composable",
            _composable(
                bm=32,
                bn=64,
                bk=256,
                stages=stages,
                fixed_k=1536,
                policy=(True, False, direct_b, 2, cache_b, True, True, normal),
            ),
            bm=32,
            bn=64,
            bk=256,
        )
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](192, 1536, 1536, cu_num=256)
    assert {kid for kid, _ in chosen} == set(registry)


def test_small_selection_retains_joint_bk_wave_and_split_route(search, monkeypatch):
    registry = {}
    for kid, (bn, bk, wn, stages) in enumerate(
        ((32, 256, 2, 6), (32, 128, 2, 3), (64, 256, 4, 3))
    ):
        registry[kid] = _instance(
            "composable",
            _composable(
                bn=bn,
                bk=bk,
                wn=wn,
                stages=stages,
                split=4,
                policy=(True, False, False, 1, 2, True, True, False),
            ),
            bn=bn,
            bk=bk,
            wn=wn,
            split=4,
        )
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](4, 768, 7168, cu_num=256)
    assert 0 in {kid for kid, _ in chosen}


def test_small_selection_coalesces_identical_default_expansions(search, monkeypatch):
    defaults = (True, True, False, 1, 0, True, True, True, False, 3, 0, False)
    registry = {
        7: _instance("composable", _composable()),
        3: _instance("composable", _composable(policy=defaults)),
    }
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](16, 128, 7168, cu_num=256)
    assert [kid for kid, _ in chosen] == [3]


@pytest.mark.parametrize("difference", ["wg", "grid"])
def test_small_selection_keeps_distinct_launch_configuration(
    search, monkeypatch, difference
):
    traits = _composable(
        bm=32,
        bn=64,
        bk=256,
        policy=(True, False, False, 1, 0, True, True, False),
    )
    registry = {
        0: _instance("composable", traits, bm=32, bn=64, bk=256),
        1: _instance(
            "composable",
            traits,
            bm=32,
            bn=64,
            bk=256,
            wg=2 if difference == "wg" else 1,
        ),
    }
    if difference == "grid":
        registry[0].bpreshuffle_entry["grid_kind"] = "nm"
        registry[1].bpreshuffle_entry["grid_kind"] = "xcd"
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](256, 1536, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == {0, 1}


def test_small_selection_limits_padding_within_each_wave_layout(search, monkeypatch):
    registry = {}
    for kid, (bm, wm) in enumerate(((16, 1), (32, 1), (128, 1), (128, 4))):
        registry[kid] = _instance(
            "composable",
            _composable(
                bm=bm,
                wm=wm,
                bk=256,
                stages=3 if wm == 4 else 6,
                fixed_k=1536,
                policy=(False, False, False),
            ),
            bm=bm,
            wm=wm,
            bk=256,
        )
    _install_registry(monkeypatch, registry)
    # Short K isolates padding from the steady-loop MFMA work bound.
    chosen = search["opus_candidates"](1, 7168, 1536, cu_num=256)
    assert {kid for kid, _ in chosen} == {0, 3}


def test_small_selection_retains_every_legal_co(search, monkeypatch):
    registry = {
        5: SimpleNamespace(bpreshuffle_co={"pipeline": "pin_agpr"}),
        2: SimpleNamespace(bpreshuffle_co={"pipeline": "b_direct"}),
        1: SimpleNamespace(bpreshuffle_co={"pipeline": "b_direct"}),
    }
    _install_registry(monkeypatch, registry, legal=[5, 2])
    chosen = search["opus_candidates"](16, 128, 7168, cu_num=256)
    assert [kid for kid, _ in chosen] == [2, 5]


def test_selection_uses_small_and_large_ranges_at_m_boundary(search, monkeypatch):
    registry = {
        kid: _instance(
            "prefetch_b_vgpr",
            _template(
                "opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits",
                (48, 128, stages, 2),
            ),
            bm=48,
            bn=128,
        )
        for kid, stages in enumerate((3, 4))
    }
    _install_registry(monkeypatch, registry)

    variants = search["_opus_select_variants"]
    small_calls = []

    def capture(*args, **kwargs):
        small_calls.append(kwargs.get("small_m", False))
        return variants(*args, **kwargs)

    monkeypatch.setitem(search, "_opus_select_variants", capture)
    chosen = search["opus_candidates"](1024, 1536, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == {0, 1}
    assert small_calls and all(small_calls)

    small_calls.clear()
    chosen = search["opus_candidates"](1025, 1536, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == {0, 1}
    assert small_calls and not any(small_calls)


def test_small_resource_range_keeps_coupled_b_direct_policies(search, monkeypatch):
    registry, expected, underfilled = {}, set(), set()
    for stages, wg, caches, output, byte_scale in product(
        (3, 4), (1, 2), ((0, 0), (1, 17)), (0, 2), (False, True)
    ):
        ca, cb = caches
        args = (
            64,
            64,
            stages,
            1 if stages == 3 else 2,
            cb,
            False,
            7168,
            2,
            wg,
            ca,
            2,
            output,
            256,
            True,
            False,
            False,
            True,
            1,
            byte_scale,
        )
        kid = len(registry)
        registry[kid] = _instance(
            "b_direct",
            _template("opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits", args),
            bm=64,
            bn=64,
            bk=256,
            wg=wg,
        )
        # Direct output covers both scale layouts; packed output covers the
        # packed-scale route while retaining joint stage/WG/cache policies.
        if output == 0 or not byte_scale:
            expected.add(kid)
            if stages == 4 or wg == 1:
                underfilled.add(kid)
    registry[len(registry)] = _instance(
        "b_direct",
        _template(
            "opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits",
            (
                16,
                64,
                3,
                1,
                0,
                False,
                7168,
                2,
                1,
                0,
                2,
                0,
                128,
                True,
                False,
                False,
                True,
                1,
                False,
            ),
        ),
        bn=64,
    )
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](224, 7168, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == expected
    # A smaller grid keeps the complete deeper-ring policies, while shallow
    # rings retain their single-workgroup cache and scale combinations.
    chosen = search["opus_candidates"](192, 7168, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == underfilled


def test_small_resource_range_distinguishes_shared_and_direct_b(search, monkeypatch):
    registry = {}
    for kid, (bm, bn, split, direct, wm, bk) in enumerate(
        (
            (32, 128, 1, False, 2, 128),
            (32, 128, 1, True, 2, 128),
            (96, 64, 2, True, 2, 256),
            (96, 64, 1, True, 2, 256),
            (64, 64, 1, True, 1, 512),
        )
    ):
        registry[kid] = _instance(
            "composable",
            _composable(
                bm=bm,
                bn=bn,
                wm=wm,
                wn=2,
                bk=bk,
                stages=4,
                split=split,
                policy=(True, False, direct, 2, 2, True, True, False),
            ),
            bm=bm,
            bn=bn,
            wm=wm,
            wn=2,
            bk=bk,
            split=split,
        )
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](176, 4096, 7168, cu_num=256)
    # The shared-B tile may have E_N=4. The Split-K 2 direct-B record needs
    # M cooperation with E_M=3 and fits a single CU grid; a single partition
    # uses the one-M-wave route.
    # The BK=512 direct-B ring exceeds the small-M live-register range.
    assert {kid for kid, _ in chosen} == {0, 2}


def test_small_resource_range_drops_oversized_shared_m_fragments(search, monkeypatch):
    registry = {}
    for kid, (bm, wm) in enumerate(((32, 1), (128, 1), (128, 4))):
        registry[kid] = _instance(
            "composable",
            _composable(
                bm=bm,
                bn=64,
                wm=wm,
                stages=4,
                policy=(True, False, False, 1, 0, True, True, False),
            ),
            bm=bm,
            bn=64,
            wm=wm,
        )
    _install_registry(monkeypatch, registry)
    # Keep enough steady-loop CU coverage to isolate M-fragment cooperation.
    chosen = search["opus_candidates"](32, 12288, 7168, cu_num=256)
    assert {kid for kid, _ in chosen} == {0, 2}
    saturated = search["opus_candidates"](128, 65536, 7168, cu_num=256)
    assert {kid for kid, _ in saturated} == {0}


def test_small_split_k_range_accounts_for_wave_supply(search, monkeypatch):
    registry = {
        split: _instance(
            "composable",
            _composable(
                bm=32,
                bn=64,
                bk=256,
                split=split,
                policy=(True, False, False, 1, 0, True, True, False),
            ),
            bm=32,
            bn=64,
            bk=256,
            split=split,
        )
        for split in (1, 2, 4)
    }
    _install_registry(monkeypatch, registry)
    saturated = search["opus_candidates"](128, 7168, 7168, cu_num=256)
    underfilled = search["opus_candidates"](32, 7168, 7168, cu_num=256)
    assert {kid for kid, _ in saturated} == {1}
    assert {kid for kid, _ in underfilled} == {4}


def _panel_instance(group_m, panel=16):
    return _instance(
        "lds_panel",
        _template(
            "opus_gemm_bpreshuffle_lds_panel_traits",
            (192, 256, panel, 1536, 4, group_m, 0, 2, False, False, False, False, True),
        ),
        bm=192,
        bn=256,
        wm=4,
    )


def test_small_panel_range_starts_with_ungrouped_m_traversal(search, monkeypatch):
    registry = {group_m: _panel_instance(group_m) for group_m in (0, 4, 8)}
    registry[16] = _panel_instance(0, panel=32)
    registry[32] = _panel_instance(0, panel=64)
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](384, 8192, 1536, cu_num=256)
    assert [kid for kid, _ in chosen] == [0]


def test_small_resource_range_has_a_legal_family_fallback(search, monkeypatch):
    registry = {5: _panel_instance(4), 3: _panel_instance(8)}
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](384, 8192, 1536, cu_num=256)
    assert [kid for kid, _ in chosen] == [3]


def test_small_range_keeps_the_fourth_tile_for_tiny_m_and_wide_n(search, monkeypatch):
    registry = {}
    for bm, bn in product((32, 64, 128), (32, 64, 128)):
        registry[len(registry)] = _instance(
            "composable",
            _composable(
                bm=bm,
                bn=bn,
                bk=256,
                wm=2,
                wn=2,
                policy=(False, False, False, 1, 0, True, True, True),
            ),
            bm=bm,
            bn=bn,
            bk=256,
            wm=2,
            wn=2,
        )
    _install_registry(monkeypatch, registry)
    chosen = search["opus_candidates"](16, 65536, 1536, cu_num=256)
    tiles = {(instance.B_M, instance.B_N) for _, instance in chosen}
    # This existing small-M winner is fourth in its joint layout. Its padding
    # is allowed relative to the minimum M tile for the two-M-wave layout.
    assert (64, 128) in tiles
    assert tiles == {(32, 128), (64, 128)}
