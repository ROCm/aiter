# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
import aiter.fused_moe as fm

# moe_sorting lives in aiter.moev2 (under development, not graduated; owners @qli88,
# see aiter/moev2/README.md); its backend flags are patched there.
import aiter.moev2.segments.sort as sorting
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

BLOCK_SIZE_M = 32
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def test_moe_sorting_opus_host_dispatch_boundaries():
    if get_gfx_runtime() == "gfx942":
        device_id = torch.cuda.current_device()
        expect_multi = {
            (63, 8): False,
            (64, 8): True,
            (63, 16): False,
            (64, 16): True,
            (32, 32): False,
            (33, 32): True,
            (16, 64): False,
            (17, 64): True,
            (8, 128): False,
            (9, 128): True,
            (1, 129): True,
            (1, 257): True,
            (24, 257): True,
            (1, 385): True,
            (1, 512): True,
        }
        for (token, E), multi in expect_multi.items():
            auto_workspace = aiter.moe_sorting_opus_get_workspace_size(
                token, E, 8, 0, device_id
            )
            assert (
                auto_workspace > 0
            ) == multi, f"unexpected auto dispatch for E={E}, M={token}"
            assert (
                aiter.moe_sorting_opus_get_workspace_size(token, E, 8, 1, device_id)
                == 0
            ), f"forced oneshot failed for E={E}, M={token}"
            assert (
                aiter.moe_sorting_opus_get_workspace_size(token, E, 8, 2, device_id) > 0
            ), f"forced multi failed for E={E}, M={token}"
        return

    if get_gfx_runtime() != "gfx950":
        device_id = torch.cuda.current_device()
        assert aiter.moe_sorting_opus_get_workspace_size(1, 384, 8, 0, device_id) == 0
        assert aiter.moe_sorting_opus_get_workspace_size(1, 385, 8, 0, device_id) == 0
        assert aiter.moe_sorting_opus_get_workspace_size(1, 512, 8, 0, device_id) > 0
        return

    device_id = torch.cuda.current_device()
    tokens = (1, 7, 8, 12, 16, 17, 24, 25, 31)
    multi_tokens = {
        127: (),
        128: (17, 24, 25, 31),
        192: (17, 24, 25, 31),
        255: (17, 24, 25, 31),
        256: (8, 12, 16, 17, 24, 25, 31),
        257: (8, 12, 16, 17, 24, 25, 31),
        383: (8, 12, 16, 17, 24, 25, 31),
        384: tokens,
        385: tokens,
    }
    for E, token in itertools.product(multi_tokens, tokens):
        auto_workspace = aiter.moe_sorting_opus_get_workspace_size(
            token, E, 8, 0, device_id
        )
        oneshot_workspace = aiter.moe_sorting_opus_get_workspace_size(
            token, E, 8, 1, device_id
        )
        multi_workspace = aiter.moe_sorting_opus_get_workspace_size(
            token, E, 8, 2, device_id
        )
        expect_auto_multi = token in multi_tokens[E]
        assert (
            auto_workspace > 0
        ) == expect_auto_multi, f"unexpected auto dispatch for E={E}, M={token}"
        assert oneshot_workspace == 0, f"forced oneshot failed for E={E}, M={token}"
        assert multi_workspace > 0, f"forced multi failed for E={E}, M={token}"


def test_moe_sorting_opus_workspace_uses_explicit_device():
    original = torch.cuda.current_device()
    target = (
        (original + 1) % torch.cuda.device_count()
        if torch.cuda.device_count() > 1
        else original
    )
    try:
        workspace = aiter.moe_sorting_opus_get_workspace_size(8, 256, 8, 0, target)
        assert torch.cuda.current_device() == original
        if _device_arch(target) in ("gfx942", "gfx950"):
            assert workspace > 0
    finally:
        torch.cuda.set_device(original)


def test_moe_sorting_opus_workspace_switches_arch_policy():
    arches = [_device_arch(device_id) for device_id in range(torch.cuda.device_count())]
    gfx950 = next((i for i, arch in enumerate(arches) if arch == "gfx950"), None)
    other = next((i for i, arch in enumerate(arches) if arch != "gfx950"), None)
    if gfx950 is None or other is None:
        aiter.logger.warning("arch-switch test requires gfx950 and non-gfx950 devices")
        return

    # M=1 is multi-phase at E=385 on gfx950 only, and at E=256 on gfx942 only.
    E = 256 if arches[other] == "gfx942" else 385
    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(other)
        assert (aiter.moe_sorting_opus_get_workspace_size(1, E, 8, 0, gfx950) > 0) == (
            E == 385
        )
        torch.cuda.set_device(gfx950)
        assert (aiter.moe_sorting_opus_get_workspace_size(1, E, 8, 0, other) > 0) == (
            E == 256
        )
    finally:
        torch.cuda.set_device(original)


def test_moe_sorting_opus_rejects_short_workspace():
    """A multi-phase sort without enough workspace raises instead of faulting."""
    token, E, topk = 1, 257, 9
    topk_ids = torch.arange(topk, dtype=dtypes.i32).repeat(token, 1)
    topk_weights = torch.rand((token, topk), dtype=dtypes.fp32)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        run_torch_moe_sorting(topk_ids, topk_weights, E)
    )
    moe_buf = torch.empty((token, 128), dtype=dtypes.bf16)
    ws_size = aiter.moe_sorting_opus_get_workspace_size(token, E, topk, 2)
    for workspace in (None, torch.empty(ws_size - 1, dtype=torch.uint8)):
        try:
            aiter.moe_sorting_opus_fwd(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf,
                E,
                BLOCK_SIZE_M,
                None,
                None,
                workspace,
                2,
            )
        except RuntimeError as e:
            assert "workspace" in str(e), e
        else:
            raise AssertionError("moe_sorting_opus_fwd ran without enough workspace")


def _device_arch(device_id: int) -> str:
    props = torch.cuda.get_device_properties(device_id)
    return str(getattr(props, "gcnArchName", props.name)).split(":", 1)[0]


def set_moe_sorting_backend(backend: str) -> None:
    """Force which moe_sorting backend `moe_sorting()` dispatches to."""
    if backend == "flydsl":
        sorting._USE_CK_MOE_SORTING = False
        sorting._USE_FLYDSL_MOE_SORTING = True
    elif backend == "opus":
        sorting._USE_CK_MOE_SORTING = False
        sorting._USE_FLYDSL_MOE_SORTING = False
    elif backend == "ck":
        sorting._USE_CK_MOE_SORTING = True
        sorting._USE_FLYDSL_MOE_SORTING = False
    elif backend == "auto":
        pass
    else:
        raise ValueError(f"unknown backend: {backend}")


def run_torch_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
):
    """CPU/torch reference for moe_sorting outputs (not timed, not in the table)."""
    device = topk_ids.device
    m, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    # The GEMM consumers describe this allocation as max_num_m_blocks full
    # blocks. Keep the reference capacity block-aligned so the sorting test also
    # enforces that descriptor/allocation contract.
    max_num_tokens_padded = max_num_m_blocks * block_size
    init_val = topk << 24 | m
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=dtypes.i32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=dtypes.i32, device=device
    )
    num_tokens_post_pad = torch.empty((2), dtype=dtypes.i32, device=device)

    if num_local_tokens is not None:
        topk_ids = topk_ids[: num_local_tokens.item()]

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expert_id in range(num_experts):
        if expert_mask is not None and expert_mask[expert_id] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = token_id.numel()
        sorted_expert_ids_num = (tokens_num + block_size - 1) // block_size
        tokens_num_pad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_id << 24 | token_id
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ]
        sorted_ids_begin = sorted_ids_begin + tokens_num_pad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expert_id - skip_expert_num)
        sorted_expert_ids_begin = sorted_expert_ids_begin + sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = topk_ids.shape[0]

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def _onewg_pair_keys(ids, weights, eids, num_rows, block_size, topk, M, topk_weights):
    """(expert, packed id) of every real pair of a sorted table, after checking the padding rows and weights."""
    rows = ids[:num_rows]
    expert = eids[: num_rows // block_size].repeat_interleave(block_size)
    real = rows != (topk << 24 | M)
    tok, slot = rows[real] & 0x00FFFFFF, rows[real] >> 24
    assert bool((tok < M).all()) and bool(
        (slot < topk).all()
    ), "pad sentinel must be (topk << 24) | M"
    assert torch.equal(
        weights[:num_rows][real], topk_weights[tok.long(), slot.long()]
    ), "sorted_weights must follow their ids"
    return (expert[real].long() << 32) | rows[real].long()


def _check_onewg(gen, topk, block_size, M, case):
    """One (topk, block_size, M, routing case): the one-workgroup table equals the reference up to the pair order
    inside an expert."""
    E = 128
    score = torch.rand(M, E, device="cuda", generator=gen)
    if case == "skewed":
        score[:, :8] += 2.0
    elif case == "one_expert_set":
        score[:, :] = torch.rand(1, E, device="cuda", generator=gen)
    topk_ids = score.topk(topk, dim=1).indices.to(dtypes.i32).contiguous()
    topk_weights = torch.rand(M, topk, device="cuda", generator=gen)
    ref = run_torch_moe_sorting(topk_ids, topk_weights, E, block_size)
    # record what the public unordered branch itself returns (no separate launch)
    import aiter.moev2.segments.sort as sort_mod
    from aiter.ops.flydsl.kernels.moe_sorting_onewg import onewg_supported

    calls = []
    orig = sort_mod._flydsl_moe_sorting_onewg

    def recording(*args, **kwargs):
        calls.append(orig(*args, **kwargs))
        return calls[-1]

    sort_mod._flydsl_moe_sorting_onewg = recording
    try:
        out = moe_sorting(
            topk_ids,
            topk_weights,
            E,
            4096,
            torch.bfloat16,
            block_size,
            accumulate=False,
            unordered=True,
        )
    finally:
        sort_mod._flydsl_moe_sorting_onewg = orig
    used = calls[0] if calls else None
    if get_gfx_runtime() == "gfx950" and onewg_supported(M, E, topk, block_size):
        assert used is not None, "unordered=True did not take the one-workgroup sort"
        assert all(a is b for a, b in zip(out[:4], used)), "result is not its output"
    if (
        used is None
    ):  # topk=8 at the largest M does not fit one workgroup's LDS: normal path, exact
        errs = _compare_moe_sorting_outputs(ref, out, topk, M)
        assert not any(errs.values()), errs
        return False
    ids, w, eids, nv = out[:4]
    assert (
        nv.tolist() == ref[3].tolist()
    ), f"num_valid_ids {nv.tolist()} vs {ref[3].tolist()}"
    n = nv[0].item()
    assert ids.shape == ref[0].shape and eids.shape == ref[2].shape
    assert torch.equal(eids[: n // block_size], ref[2][: n // block_size])
    got = _onewg_pair_keys(ids, w, eids, n, block_size, topk, M, topk_weights)
    want = _onewg_pair_keys(
        ref[0],
        torch.where(ref[0] != (topk << 24 | M), 1.0, 0.0),
        ref[2],
        n,
        block_size,
        topk,
        M,
        torch.ones(M + 1, topk, device="cuda"),
    )
    assert torch.equal(
        got.sort().values, want.sort().values
    ), f"pairs differ: topk={topk} bm={block_size} M={M} {case}"
    assert not (ids[:n] == (topk << 24 | M)).all() or M == 0
    return True


def test_moe_sorting_onewg_table_boundary():
    """Configs whose block -> sorted-row bound fills whole 16 B tiles (the table needs one more row for the invalid
    pairs): topk 1 / 2 at the 64-token tier, topk 3 at 384, topk 6 at 192 used to fail the compile-time assert.
    """
    if get_gfx_runtime() != "gfx950":  # the one-workgroup path is gfx950-only
        return
    gen = torch.Generator(device="cuda").manual_seed(11)
    for topk, block_size, M in (
        (1, 16, 1),
        (1, 32, 64),
        (1, 64, 100),
        (2, 16, 1),
        (2, 32, 33),
        (3, 16, 300),
        (6, 16, 150),
    ):
        assert _check_onewg(gen, topk, block_size, M, "random"), (topk, block_size, M)


def test_moe_sorting_onewg():
    """One-workgroup sort (``moe_sorting(unordered=True)``): same table as the reference up to the pair order
    inside an expert, for E=128 up to 512 tokens."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    E = 128
    for topk, block_size, M, case in itertools.product(
        (4, 8),
        (16, 32, 64),
        (1, 2, 8, 16, 17, 24, 32, 48, 64, 96, 128, 192, 256, 384, 448, 512),
        ("random", "skewed", "one_expert_set"),
    ):
        _check_onewg(gen, topk, block_size, M, case)
    # CUDA-graph capture + replay with fresh routing
    topk_ids = (
        torch.rand(48, E, device="cuda")
        .topk(4, dim=1)
        .indices.to(dtypes.i32)
        .contiguous()
    )
    topk_weights = torch.rand(48, 4, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = moe_sorting(
            topk_ids,
            topk_weights,
            E,
            4096,
            torch.bfloat16,
            32,
            accumulate=False,
            unordered=True,
        )
    for _ in range(3):
        topk_ids.copy_(
            torch.rand(48, E, device="cuda").topk(4, dim=1).indices.to(dtypes.i32)
        )
        g.replay()
        ref = run_torch_moe_sorting(topk_ids, topk_weights, E, 32)
        n = out[3][0].item()
        assert n == ref[3][0].item()
        assert torch.equal(
            _onewg_pair_keys(out[0], out[1], out[2], n, 32, 4, 48, topk_weights)
            .sort()
            .values,
            _onewg_pair_keys(
                ref[0],
                torch.where(ref[0] != (4 << 24 | 48), 1.0, 0.0),
                ref[2],
                n,
                32,
                4,
                48,
                torch.ones(49, 4, device="cuda"),
            )
            .sort()
            .values,
        )


def test_moe_slots_shared_reduce():
    """moe_slots_shared_reduce == fp32 sequential sum of the slots then the shared expert, rounded once to bf16
    (bitwise), against a torch reference."""
    from aiter.ops.flydsl.kernels.moe_slots_shared_reduce import moe_slots_shared_reduce

    for M, topk, H in itertools.product((1, 7, 16, 33, 128, 512), (4, 6), (2048, 6144)):
        parts = (torch.randn(M, topk, H, device="cuda") * 3).bfloat16()
        shared = (torch.randn(M, H, device="cuda") * 3).bfloat16()
        acc = torch.zeros(M, H, device="cuda", dtype=torch.float32)
        for k in range(topk):
            acc += parts[:, k].float()
        ref = (acc + shared.float()).bfloat16()
        out = moe_slots_shared_reduce(parts, shared)
        assert torch.equal(
            out, ref
        ), f"slots+shared reduce mismatch M={M} topk={topk} H={H}"


def test_moe_slots_shared_reduce_dropped_routes():
    """With ``topk_ids``, a slot whose id is outside [0, E) adds zero (even if its row holds NaN); valid slots keep
    the unmasked kernel's bits."""
    from aiter.ops.flydsl.kernels.moe_slots_shared_reduce import moe_slots_shared_reduce

    E = 128
    for M, topk, H in itertools.product((1, 16, 128), (4,), (2048, 6144)):
        parts = (torch.randn(M, topk, H, device="cuda") * 3).bfloat16()
        shared = (torch.randn(M, H, device="cuda") * 3).bfloat16()
        ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device="cuda")
        # all valid: same bits as the unmasked kernel
        assert torch.equal(
            moe_slots_shared_reduce(parts, shared, topk_ids=ids, num_experts=E),
            moe_slots_shared_reduce(parts, shared),
        )
        bad = ids.clone()
        bad[:, 1] = -1
        bad[0, :] = E
        if M > 2:
            bad[2, 3] = 2**20
        valid = (bad >= 0) & (bad < E)
        poisoned = parts.clone()
        poisoned[~valid] = float("nan")
        acc = torch.zeros(M, H, device="cuda", dtype=torch.float32)
        for k in range(topk):
            acc += torch.where(valid[:, k, None], parts[:, k].float(), 0.0)
        ref = (acc + shared.float()).bfloat16()
        out = moe_slots_shared_reduce(poisoned, shared, topk_ids=bad, num_experts=E)
        assert torch.equal(out, ref), f"dropped-route reduce mismatch M={M} H={H}"


def _moe_sorting_roofline(token, topk, E, model_dim, dtype):
    """Crude roofline estimates for a memory-bound sort (not exact kernel traffic)."""
    max_num_tokens_padded = token * topk + E * BLOCK_SIZE_M - topk
    max_num_m_blocks = (max_num_tokens_padded + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    max_num_tokens_padded = max_num_m_blocks * BLOCK_SIZE_M
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    nbytes = (
        token * topk * 4
        + token * topk * 4
        + max_num_tokens_padded * 4
        + max_num_tokens_padded * 4
        + max_num_m_blocks * 4
        + token * model_dim * elem_bytes
    )
    flops = 2 * max_num_tokens_padded
    return flops, nbytes


def _compare_moe_sorting_outputs(ref, out, topk, num_rows, exact_ids=True):
    (
        sorted_ids_a,
        sorted_weights_a,
        sorted_expert_ids_a,
        num_tokens_post_padded_a,
    ) = ref
    (
        sorted_ids_b,
        sorted_weights_b,
        sorted_expert_ids_b,
        num_tokens_post_padded_b,
        _moe_buf,
    ) = out

    assert sorted_ids_b.shape == sorted_ids_a.shape, (
        "sorted_ids capacity must cover every full GEMM block: "
        f"expected {sorted_ids_a.shape}, got {sorted_ids_b.shape}"
    )
    assert sorted_weights_b.shape == sorted_weights_a.shape, (
        "sorted_weights capacity must match sorted_ids: "
        f"expected {sorted_weights_a.shape}, got {sorted_weights_b.shape}"
    )
    assert sorted_expert_ids_b.shape == sorted_expert_ids_a.shape, (
        "sorted_expert_ids capacity mismatch: "
        f"expected {sorted_expert_ids_a.shape}, got {sorted_expert_ids_b.shape}"
    )

    errs = {}
    errs["num_tokens_post_padded"] = checkAllclose(
        num_tokens_post_padded_a,
        num_tokens_post_padded_b,
        atol=0,
        msg="num_tokens_post_padded",
    )
    weight_mask = sorted_ids_a != (topk << 24 | num_rows)
    num_tokens_post_pad = num_tokens_post_padded_a[0].item()
    # Exact: ids are packed (topk << 24 | token), so a relative tolerance would
    # pass a pad id holding the wrong token.
    exact = {"rtol": 0, "atol": 0} if exact_ids else {}
    errs["sorted_ids"] = checkAllclose(
        sorted_ids_a[:num_tokens_post_pad],
        sorted_ids_b[:num_tokens_post_pad],
        msg="sorted_ids",
        **exact,
    )
    errs["sorted_weights"] = checkAllclose(
        sorted_weights_a[weight_mask],
        sorted_weights_b[weight_mask],
        msg="sorted_weights",
    )
    expert_mask = sorted_expert_ids_a != -1
    errs["sorted_expert_ids"] = checkAllclose(
        sorted_expert_ids_a[expert_mask],
        sorted_expert_ids_b[expert_mask],
        msg="sorted_expert_ids",
    )
    return errs


def test_moe_sorting_opus_aux_outputs(dtype, model_dim):
    """Cover auxiliary outputs on one-shot and newly multi-phase auto routes."""
    for token, E, topk in ((7, 32, 5), (8, 256, 8), (1, 384, 6), (1, 385, 7)):
        topk_ids, topk_weights, _, _ = _build_moe_sorting_inputs(
            token,
            model_dim,
            E,
            topk,
            dtype,
            has_expert_mask=False,
            padding_extra=0,
        )
        for block_size in (16, 32, 64, 128):
            # Other shapes go to the fused aux sort, which has no instance for them.
            if not fm._aux_uses_opus(fm.AUX_SORT_OPUS, block_size, token * topk, E):
                continue
            ref = run_torch_moe_sorting(topk_ids, topk_weights, E, block_size)
            out = moe_sorting(
                topk_ids,
                topk_weights,
                E,
                model_dim,
                dtype,
                block_size,
                output_aux=fm.AUX_SORT_OPUS,
            )

            errs = _compare_moe_sorting_outputs(ref, out[:5], topk, token)
            bad = {name: err for name, err in errs.items() if err}
            mismatch = (
                f"Opus auxiliary sort mismatch at M={token}, E={E}, "
                f"block_size={block_size}: {bad}"
            )
            assert not bad, mismatch
            (
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                _,
                _,
                aux_m_indices,
                aux_reverse_sorted,
            ) = out
            expected_capacity = sorted_expert_ids.numel() * block_size
            assert sorted_ids.numel() == expected_capacity
            assert sorted_weights.numel() == expected_capacity
            assert aux_m_indices.numel() == expected_capacity
            assert aux_reverse_sorted.numel() == topk_ids.numel()
            num_tokens_post_pad = out[3][0].item()
            assert torch.equal(
                aux_m_indices[:num_tokens_post_pad],
                sorted_ids[:num_tokens_post_pad] & 0x00FFFFFF,
            )
            topk_slots = torch.arange(
                topk, dtype=topk_ids.dtype, device=topk_ids.device
            )
            expected_sorted_ids = (
                topk_slots.unsqueeze(0) << 24
                | torch.arange(
                    token, dtype=topk_ids.dtype, device=topk_ids.device
                ).unsqueeze(1)
            ).flatten()
            assert torch.equal(
                sorted_ids[aux_reverse_sorted.long()], expected_sorted_ids
            )


def test_moe_sorting_opus_local_ids_large_expert_auto(dtype, model_dim):
    for token, E, topk in ((8, 256, 8), (1, 384, 6), (1, 385, 7)):
        topk_ids, topk_weights, _, _ = _build_moe_sorting_inputs(
            token,
            model_dim,
            E,
            topk,
            dtype,
            has_expert_mask=False,
            padding_extra=0,
        )
        *_, local_topk_ids = moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            dispatch_policy=0,
            return_local_topk_ids=True,
        )
        assert torch.equal(
            topk_ids, local_topk_ids
        ), f"local_topk_ids mismatch at M={token}, E={E}"


def _build_moe_sorting_inputs(
    token,
    model_dim,
    E,
    topk,
    dtype,
    has_expert_mask,
    padding_extra,
):
    input_tensor = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    expert_mask = (
        torch.randint(0, 2, (E,), dtype=topk_ids.dtype, device="cuda")
        if has_expert_mask
        else None
    )
    if padding_extra:
        num_local_tokens = torch.tensor([token], dtype=topk_ids.dtype, device="cuda")
        topk_ids_pad = torch.empty(
            [token + padding_extra, topk], dtype=topk_ids.dtype, device="cuda"
        )
        topk_ids_pad[:token, :] = topk_ids
        topk_ids = topk_ids_pad
    else:
        num_local_tokens = None

    return topk_ids, topk_weights, expert_mask, num_local_tokens


def _apply_routing_case(topk_ids, topk_weights, routing_case):
    token, topk = topk_weights.shape
    if routing_case == "all-empty":
        topk_ids[:token].fill_(-1)
        topk_weights.zero_()
    elif routing_case == "mixed":
        invalid = (
            torch.arange(token * topk, device=topk_ids.device, dtype=torch.int64).view(
                token, topk
            )
            % 2
            == 1
        )
        topk_ids[:token].masked_fill_(invalid, -1)
        topk_weights.masked_fill_(invalid, 0)
    elif routing_case != "valid":
        raise ValueError(f"unknown routing case: {routing_case}")


@benchmark()
def test_moe_sorting(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    has_expert_mask=False,
    padding_extra=0,
    dispatch_policy=0,
    accumulate=True,
    routing_case="valid",
):
    """accumulate=False (with has_expert_mask=False) makes moe_sorting allocate
    a (0,0) moe_buf placeholder instead of the full [M, model_dim] buffer --
    useful for isolating moe_buf-zeroing cost (which scales with the static
    padded capacity M) from the mesh-scan work (which should NOT scale with M
    once tokens_ correctly bounds it) when comparing padding_extra=0 vs >0.
    """
    topk_ids, topk_weights, expert_mask, num_local_tokens = _build_moe_sorting_inputs(
        token,
        model_dim,
        E,
        topk,
        dtype,
        has_expert_mask,
        padding_extra,
    )
    _apply_routing_case(topk_ids, topk_weights, routing_case)

    ref = run_torch_moe_sorting(
        topk_ids,
        topk_weights,
        E,
        BLOCK_SIZE_M,
        expert_mask,
        num_local_tokens,
    )

    candidates = {
        "opus": lambda: moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            accumulate=accumulate,
        ),
        "ck": lambda: moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            accumulate=accumulate,
        ),
    }
    # FlyDSL kernel only supports dispatch_policy=0 today.
    if dispatch_policy == 0:
        candidates["flydsl"] = lambda: moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            accumulate=accumulate,
        )

    flops, nbytes = _moe_sorting_roofline(token, topk, E, model_dim, dtype)
    ret = {"gfx": get_gfx(), "routing case": routing_case}
    failures = {}

    for name, fn in candidates.items():
        set_moe_sorting_backend(name)
        out, us = run_perftest(
            fn,
            num_warmup=1,
            num_iters=2,
        )
        # CK-tile's sorter still pads with the device-side count, so its ids
        # keep the loose comparison: in composable_kernel's
        # moe_sorting_kernel.hpp, moe_align_block_size_kernel_ex and
        # MoeSortingMultiPhaseKernel_P3 / _P23 write `tokens`, which under
        # LocalToken is p_local_tokens[0].
        errs = _compare_moe_sorting_outputs(
            ref, out, topk, topk_ids.shape[0], exact_ids=name != "ck"
        )
        err = max(errs.values()) if errs else 0.0
        bad = {k: v for k, v in errs.items() if v}
        if bad:
            failures[name] = bad

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    if failures:
        raise AssertionError(
            f"moe_sorting mismatch vs CPU reference at token={token}, E={E}, "
            f"topk={topk}, has_expert_mask={has_expert_mask}, "
            f"padding_extra={padding_extra}, dispatch_policy={dispatch_policy}, "
            f"routing_case={routing_case}: "
            f"{failures}"
        )
    return ret


def test_moe_sorting_flydsl_cuda_graph_capture(
    dtype,
    token,
    model_dim,
    E,
    topk,
    has_expert_mask=False,
    routing_case="valid",
):
    """Capture moe_sorting (FlyDSL backend) under a real torch.cuda.graph(),
    then replay with a DIFFERENT num_local_tokens tensor value than was
    captured. Regression test for the .item() capture-safety fix: reading
    num_local_tokens on the host during capture used to raise
    'operation not permitted when stream is capturing'; and even a naive
    host-side guard would still bake a stale capture-time count into the
    graph. This asserts real dynamic behavior survives across replay.
    """
    set_moe_sorting_backend("flydsl")

    capture_tokens = token
    replay_tokens = max(1, token // 2)  # a different dynamic count at replay time
    padding_extra = token  # topk_ids capacity must cover the larger of the two

    topk_ids, topk_weights, expert_mask, num_local_tokens = _build_moe_sorting_inputs(
        capture_tokens,
        model_dim,
        E,
        topk,
        dtype,
        has_expert_mask,
        padding_extra,
    )
    _apply_routing_case(topk_ids, topk_weights, routing_case)
    assert isinstance(num_local_tokens, torch.Tensor)

    def run():
        return moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
        )

    # Warm-up on a side stream (standard torch.cuda.graph() prerequisite):
    # populates the FlyDSL JIT cache before capture, since compiling a new
    # kernel variant mid-capture is itself capture-unsafe.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    num_local_tokens.fill_(capture_tokens)
    with torch.cuda.graph(graph):
        out = run()
    torch.cuda.synchronize()

    # Replay with a different dynamic token count -- the num_local_tokens
    # buffer is captured by reference, so mutating its contents (not the
    # tensor object) is what a real decode step would do between replays.
    num_local_tokens.fill_(replay_tokens)
    graph.replay()
    torch.cuda.synchronize()

    ref = run_torch_moe_sorting(
        topk_ids,
        topk_weights,
        E,
        BLOCK_SIZE_M,
        expert_mask,
        num_local_tokens,
    )
    # num_rows here must be the static padded capacity (topk_ids.shape[0]),
    # matching run_torch_moe_sorting's sentinel (topk << 24 | m) -- NOT the
    # dynamic replay_tokens count.
    errs = _compare_moe_sorting_outputs(ref, out, topk, topk_ids.shape[0])
    bad = {k: v for k, v in errs.items() if v}
    assert not bad, (
        f"moe_sorting cuda-graph replay mismatch (captured token={capture_tokens}, "
        f"replayed token={replay_tokens}, E={E}, topk={topk}, "
        f"has_expert_mask={has_expert_mask}, routing_case={routing_case}): {bad}"
    )


def test_moe_sorting_invalid_topk_ids():
    """Exercise every invalid-ID sorting path against the torch reference."""
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("moe_sorting unsupported on %s; skipping", get_gfx())
        return

    dtype = dtypes.bf16
    model_dim = 4096
    E = 256
    topk = 6

    direct_cases = (
        # Automatic large-token Opus path; P0_v1 is guarded by #4839.
        ("opus", 16384, 0, False),
        # Remaining single-kernel mesh and local-expert-mask accesses.
        ("opus", 8, 1, False),
        ("opus", 8, 1, True),
        # Forced large-token multi-phase regression (P0_v1 -> P1 -> P23).
        ("opus", 2048, 2, False),
        # FlyDSL LDS oneshot and HBM multiphase mesh stores.
        ("flydsl", 8, 0, False),
        ("flydsl", 2048, 0, False),
    )

    for backend, token, dispatch_policy, has_expert_mask in direct_cases:
        for routing_case in ("all-empty", "mixed"):
            topk_ids, topk_weights, expert_mask, _ = _build_moe_sorting_inputs(
                token,
                model_dim,
                E,
                topk,
                dtype,
                has_expert_mask,
                padding_extra=0,
            )
            _apply_routing_case(topk_ids, topk_weights, routing_case)
            ref = run_torch_moe_sorting(
                topk_ids,
                topk_weights,
                E,
                BLOCK_SIZE_M,
                expert_mask,
            )

            set_moe_sorting_backend(backend)
            out = moe_sorting(
                topk_ids,
                topk_weights,
                E,
                model_dim,
                dtype,
                BLOCK_SIZE_M,
                expert_mask,
                None,
                dispatch_policy,
                accumulate=True,
            )
            torch.cuda.synchronize()
            errs = _compare_moe_sorting_outputs(ref, out, topk, token)
            bad = {name: value for name, value in errs.items() if value}
            assert not bad, (
                f"{backend} moe_sorting mismatch for token={token}, "
                f"dispatch_policy={dispatch_policy}, "
                f"has_expert_mask={has_expert_mask}, "
                f"routing_case={routing_case}: {bad}"
            )

    test_moe_sorting_flydsl_cuda_graph_capture(
        dtype,
        token=2048,
        model_dim=model_dim,
        E=E,
        topk=topk,
        has_expert_mask=True,
        routing_case="mixed",
    )


def test_moe_sorting_decode_graph_perf(
    dtype,
    capture_capacity,
    model_dim,
    E,
    topk,
    real_tokens,
    has_expert_mask=False,
    num_iters=50,
):
    """Benchmark steady-state torch.cuda.graph() REPLAY latency

    This is what actual decode looks like: the graph is captured once per
    bucket, then replayed every step with whatever the real dynamic count
    happens to be that step
    """
    real_tokens = min(real_tokens, capture_capacity)
    topk_ids, topk_weights, expert_mask, num_local_tokens = _build_moe_sorting_inputs(
        capture_capacity,
        model_dim,
        E,
        topk,
        dtype,
        has_expert_mask,
        padding_extra=0,  # capture_capacity IS the static M; no extra slack
    )
    if num_local_tokens is None:
        num_local_tokens = torch.tensor(
            [capture_capacity], dtype=topk_ids.dtype, device="cuda"
        )

    ret = {
        "gfx": get_gfx(),
        "E": E,
        "topk": topk,
        "has_expert_mask": has_expert_mask,
        "capture_capacity": capture_capacity,
        "real_tokens": real_tokens,
    }
    backends = ["opus", "ck", "flydsl"]
    for name in backends:
        set_moe_sorting_backend(name)

        def run():
            return moe_sorting(
                topk_ids,
                topk_weights,
                E,
                model_dim,
                dtype,
                BLOCK_SIZE_M,
                expert_mask,
                num_local_tokens,
            )

        # Warm-up on a side stream: populates any JIT cache before capture,
        # since compiling mid-capture is itself capture-unsafe.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # moe_sorting() allocates fresh output tensors on every call
            # (torch.empty(...) inside _flydsl_moe_sorting / _moe_sorting_impl).
            # Keep a live Python reference to what's captured -- CUDA graph
            # semantics require capture-time allocations to stay referenced
            # for the graph's lifetime, or the allocator can reclaim that
            # memory (e.g. for a later unrelated allocation) while replay
            # still writes into it.
            captured_out = run()
        torch.cuda.synchronize()

        # Set the REAL per-step token count, then time steady-state replay.
        num_local_tokens.fill_(real_tokens)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        us = start_event.elapsed_time(end_event) * 1000 / num_iters
        del captured_out

        ret[f"{name} us"] = us

    return ret


def run_invalid_routing_regressions(dtype, model_dim, inter_dim):
    rows = []
    for routing_case in ("all-empty", "mixed"):
        rows.append(
            test_moe_sorting(
                dtype,
                token=2048,
                model_dim=model_dim,
                inter_dim=inter_dim,
                E=256,
                topk=6,
                has_expert_mask=False,
                padding_extra=0,
                dispatch_policy=0,
                accumulate=True,
                routing_case=routing_case,
            )
        )
    return rows


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("moe_sorting unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        help="Data type.\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 7, 31, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Number of tokens.\n    e.g.: -m 64",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[32, 256, 40, 385],
        help="Number of experts (paired with -t).\n    e.g.: -e 32 385",
    )
    parser.add_argument(
        "-md",
        "--model_dim",
        type=int,
        default=4096,
        help="Model dimension.\n    e.g.: -md 7168",
    )
    parser.add_argument(
        "-id",
        "--inter_dim",
        type=int,
        default=4096,
        help="Intermediate dimension (table column only).\n    e.g.: -id 4096",
    )
    parser.add_argument(
        "-t",
        "--topk",
        type=int,
        nargs="*",
        default=[5, 8, 6, 7],
        help="Top-k experts per token (paired with -e).\n    e.g.: -t 5 7",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        nargs="*",
        default=[0, 1000],
        help="Extra padding rows in topk_ids (0 = none).\n    e.g.: -p 0",
    )
    parser.add_argument(
        "-dp",
        "--dispatch_policy",
        type=int,
        nargs="*",
        default=[0, 1],
        help="Dispatch policy.\n    e.g.: -dp 0",
    )
    parser.add_argument(
        "-em",
        "--expert_mask",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
        help="Expert mask.\n    e.g.: -em f",
    )
    parser.add_argument(
        "-accum",
        "--accumulate",
        type=dtypes.str2bool,
        nargs="*",
        default=[True],
        help="Stage2 accumulate mode (False + no expert_mask allocates a (0,0) "
        "moe_buf placeholder instead of the full [M, model_dim] buffer -- "
        "isolates moe_buf-zeroing cost from mesh-scan work).\n    e.g.: -accum f",
    )
    parser.add_argument(
        "-rc",
        "--routing_case",
        choices=["valid", "all-empty", "mixed"],
        nargs="*",
        default=None,
        help="Routing inputs, including invalid -1 sentinel coverage.\n"
        "    e.g.: -rc all-empty mixed\n"
        "    default: valid matrix plus focused all-empty/mixed regressions",
    )
    parser.add_argument(
        "-dg",
        "--decode_graph",
        type=int,
        nargs="*",
        default=None,
        help="Real per-step token counts for a cuda-graph capture/replay decode "
        "benchmark (typical decode batch sizes: 1 2 4 8 16 32 64 128). Each is "
        "run against every -m value as the captured static capacity (real "
        "tokens capped to the capacity). Omit to skip this benchmark.\n"
        "    e.g.: -dg 1 2 4 8 16 32 64 128",
    )
    args = parser.parse_args()

    if len(args.expert) != len(args.topk):
        parser.error("-e/--expert and -t/--topk must have the same length")

    model_configs = list(zip(args.expert, args.topk))
    routing_cases = args.routing_case if args.routing_case is not None else ["valid"]

    test_moe_sorting_opus_host_dispatch_boundaries()
    test_moe_sorting_onewg()
    if get_gfx_runtime() == "gfx950":  # asserts the one-workgroup path is taken
        test_moe_sorting_onewg_table_boundary()
    test_moe_slots_shared_reduce()
    test_moe_slots_shared_reduce_dropped_routes()
    test_moe_sorting_opus_workspace_uses_explicit_device()
    test_moe_sorting_opus_workspace_switches_arch_policy()
    test_moe_sorting_opus_rejects_short_workspace()
    for dtype in args.dtype:
        test_moe_sorting_opus_aux_outputs(dtype, args.model_dim)
        test_moe_sorting_opus_local_ids_large_expert_auto(dtype, args.model_dim)
        df = []
        for (
            padding_extra,
            expert_mask,
            dispatch_policy,
            accumulate,
            routing_case,
            m,
        ) in itertools.product(
            args.padding,
            args.expert_mask,
            args.dispatch_policy,
            args.accumulate,
            routing_cases,
            args.m,
        ):
            for E, topk in model_configs:
                df.append(
                    test_moe_sorting(
                        dtype,
                        m,
                        args.model_dim,
                        args.inter_dim,
                        E,
                        topk,
                        has_expert_mask=expert_mask,
                        padding_extra=padding_extra,
                        dispatch_policy=dispatch_policy,
                        accumulate=accumulate,
                        routing_case=routing_case,
                    )
                )
        if args.routing_case is None:
            df.extend(
                run_invalid_routing_regressions(
                    dtype,
                    args.model_dim,
                    args.inter_dim,
                )
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "moe_sorting summary (markdown):\n%s", df.to_markdown(index=False)
        )

        for expert_mask, m in itertools.product(args.expert_mask, args.m):
            if m < 2:
                continue  # need capture/replay token counts to differ
            for E, topk in model_configs:
                test_moe_sorting_flydsl_cuda_graph_capture(
                    dtype,
                    m,
                    args.model_dim,
                    E,
                    topk,
                    has_expert_mask=expert_mask,
                )
        aiter.logger.info("moe_sorting FlyDSL cuda-graph capture/replay: all passed")

        if args.decode_graph:
            decode_rows = []
            for expert_mask, capacity, real_tokens in itertools.product(
                args.expert_mask, args.m, args.decode_graph
            ):
                for E, topk in model_configs:
                    decode_rows.append(
                        test_moe_sorting_decode_graph_perf(
                            dtype,
                            capacity,
                            args.model_dim,
                            E,
                            topk,
                            real_tokens,
                            has_expert_mask=expert_mask,
                        )
                    )
            decode_df = pd.DataFrame(decode_rows)
            aiter.logger.info(
                "moe_sorting decode graph-replay summary (us, capture_capacity=static M, "
                "real_tokens=actual per-step count):\n%s",
                decode_df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
