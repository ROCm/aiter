# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from dataclasses import dataclass

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
    _mxfp4_a_scale_sorted_shuffled,
    build_v2_inputs,
    gen,
    quant_a,
    v2_stage1_dequant_cosine_err,
    v2_stage1_sorted_ref,
)
from aiter.ops.shuffle import shuffle_weight_a16w4
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import moe_mxfp4_sort
from op_tests.mxmoe_v2_test_utils import (
    logits_diff,
    run_standalone_v2_a8w8,
)

STANDALONE_LOGITS_DIFF_LIMIT = 1e-5


def test_v2_gemm2_compile_rejects_bn64():
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import (
        compile_gemm2_a4w4_port,
    )

    with pytest.raises(AssertionError, match=r"BN in \{128,256,512\}.*BN=64"):
        compile_gemm2_a4w4_port(BN=64)


def test_v2_gemm2_wrapper_rejects_bn_outside_catalog(monkeypatch):
    import aiter.ops.flydsl.kernels.mxmoe_dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "get_g2",
        lambda *_args, **_kwargs: pytest.fail("unsupported BN reached get_g2"),
    )
    with pytest.raises(
        AssertionError, match=r"BN must be one of \(128, 256, 512\), got 384"
    ):
        dispatcher.mxfp4_moe_gemm2(
            inter_sorted_quant=None,
            inter_sorted_shuffled_scale=None,
            w2_u8=None,
            w2_scale_u8=None,
            sorted_expert_ids=None,
            cumsum_tensor=None,
            sorted_token_ids=None,
            sorted_weights=None,
            out=None,
            M_logical=32,
            max_sorted=32,
            NE=1,
            D_HIDDEN=384,
            D_INTER=128,
            topk=1,
            BN=384,
            BK=128,
        )


def test_v2_gemm2_wrapper_rejects_fp16_output_before_compile(monkeypatch):
    import aiter.ops.flydsl.kernels.mxmoe_dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "get_g2",
        lambda *_args, **_kwargs: pytest.fail("fp16 output reached get_g2"),
    )
    with pytest.raises(
        TypeError,
        match=r"FlyDSL v2 GEMM2 supports only torch\.bfloat16 output, got torch\.float16",
    ):
        dispatcher.mxfp4_moe_gemm2(
            inter_sorted_quant=None,
            inter_sorted_shuffled_scale=None,
            w2_u8=None,
            w2_scale_u8=None,
            sorted_expert_ids=None,
            cumsum_tensor=None,
            sorted_token_ids=None,
            sorted_weights=None,
            out=torch.empty((1, 128), dtype=torch.float16, device="cpu"),
            M_logical=1,
            max_sorted=32,
            NE=1,
            D_HIDDEN=128,
            D_INTER=128,
            topk=1,
            BM=32,
            BN=128,
            BK=128,
            a_dtype="fp8",
            b_dtype="fp8",
        )


def test_v2_tuner_skips_fp16_output():
    previous_device = torch.get_default_device()
    try:
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

        info = (
            "gfx950",
            256,
            16,
            256,
            256,
            8,
            2,
            ActivationType.Swiglu,
            dtypes.fp16,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_1x32,
            1,
            False,
        )
        assert FmoeTuner.gen_flydsl_v2_2stages_task(object(), info, [32]) == []
    finally:
        torch.set_default_device(previous_device)


def test_v2_tuner_skips_doweight_stage1():
    previous_device = torch.get_default_device()
    try:
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

        info = (
            "gfx950",
            256,
            16,
            256,
            256,
            8,
            2,
            ActivationType.Swiglu,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_1x32,
            1,
            True,
        )
        assert FmoeTuner.gen_flydsl_v2_2stages_task(object(), info, [32]) == []
    finally:
        torch.set_default_device(previous_device)


def test_v2_gemm2_requires_sorted_weights_before_compile(monkeypatch):
    import aiter.ops.flydsl.kernels.mxmoe_dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "get_g2",
        lambda *_args, **_kwargs: pytest.fail("missing sorted_weights reached get_g2"),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"requires sorted_weights; doweight_stage1=True is not supported",
    ):
        dispatcher.mxfp4_moe_gemm2(
            inter_sorted_quant=None,
            inter_sorted_shuffled_scale=None,
            w2_u8=None,
            w2_scale_u8=None,
            sorted_expert_ids=None,
            cumsum_tensor=None,
            sorted_token_ids=None,
            sorted_weights=None,
            out=torch.empty((1, 128), dtype=torch.bfloat16, device="cpu"),
            M_logical=1,
            max_sorted=32,
            NE=1,
            D_HIDDEN=128,
            D_INTER=128,
            topk=1,
            BM=32,
            BN=128,
            BK=128,
            a_dtype="fp8",
            b_dtype="fp8",
        )


def test_fp8_b_pair_layout_matches_runtime_shuffle():
    E, N, K = 1, 128, 256
    raw = (torch.arange(E * N * K, dtype=torch.int32) % 251).to(torch.uint8)
    weight = raw.view(E, N, K).view(dtypes.fp8)
    shuffled = shuffle_weight_a16w4(weight, 16, False).view(torch.uint8)
    physical = shuffled.view(E, N // 16, K // 64, 4, 16, 16)
    flat = shuffled.view(-1)
    k_halves = 1
    for n0, kt, pair, klane, nlane in (
        (0, 0, 0, 0, 0),
        (3, 0, 1, 2, 7),
        (7, 1, 0, 3, 15),
        (4, 1, 1, 1, 9),
    ):
        base = n0 * 16 * K
        i32_offset = klane * 64 + nlane * 4 + kt * (k_halves * 2 * 256) + pair * 256
        got = flat[base + i32_offset * 4 : base + i32_offset * 4 + 16]
        expected = physical[0, n0, kt * 2 + pair, klane, nlane]
        assert torch.equal(got, expected)
    lo = 0
    hi = 256 * 4
    assert hi - lo == 1024


def test_e8m0_shuffle_word_matches_opsel_mapping():
    N, K = 128, 384
    raw = (torch.arange(N * (K // 32), dtype=torch.int32) % 254).to(torch.uint8)
    raw = raw.view(N, K // 32).view(dtypes.fp8_e8m0)
    shuffled = fp4_utils.e8m0_shuffle(raw).view(torch.uint8)
    physical = shuffled.view(8, 2, 4, 16, 2, 2)
    for group, chunk, klane, nlane, kpack, npack in (
        (0, 0, 0, 0, 0, 0),
        (1, 0, 3, 7, 1, 1),
        (2, 1, 1, 11, 0, 0),
        (3, 1, 3, 15, 0, 1),
    ):
        opsel = 2 * kpack + npack
        word = physical[group, chunk, klane, nlane].reshape(-1)
        row = group * 32 + npack * 16 + nlane
        k_group = chunk * 8 + kpack * 4 + klane
        assert word[opsel] == raw.view(torch.uint8)[row, k_group]


@pytest.mark.l2_device
@pytest.mark.parametrize("BK", [128, 256])
def test_v2_a8w8_value_scale_coupling(BK):
    coupling_limit = 1e-4
    result = run_standalone_v2_a8w8(BM=32, BN=128, BK=BK, K=256, limit=coupling_limit)
    assert not result.out.isnan().any()
    assert logits_diff(result.out, result.ref) <= coupling_limit


@pytest.mark.l2_device
def test_v2_a8w8_coupling_negative_control():
    coupling_limit = 1e-4
    bad = run_standalone_v2_a8w8(
        BM=32,
        BN=128,
        BK=256,
        K=256,
        corrupt_b_scale_roll=1,
        limit=coupling_limit,
    )
    diff = logits_diff(bad.out, bad.ref)
    assert not math.isfinite(diff) or diff > 100 * coupling_limit


@pytest.mark.l2_device
def test_v2_a8w8_k384_does_not_consume_scale_padding():
    limit = STANDALONE_LOGITS_DIFF_LIMIT
    clean = run_standalone_v2_a8w8(BM=32, BN=128, BK=128, K=384, limit=limit)
    poisoned = run_standalone_v2_a8w8(
        BM=32,
        BN=128,
        BK=128,
        K=384,
        poison_scale_padding=True,
        limit=limit,
    )
    assert not poisoned.out.isnan().any()
    assert logits_diff(clean.out, clean.ref) <= limit
    assert logits_diff(poisoned.out, poisoned.ref) <= limit
    torch.testing.assert_close(poisoned.out, clean.out, rtol=0, atol=0)


@pytest.mark.l2_device
@pytest.mark.parametrize("BK", [128, 256])
def test_v2_a8w8_k_pad_weights_read_back_as_zero(BK):
    # has_pad K-skip shrinks only the B-weight buffer to the real K; the A buffer
    # keeps the padded extent (aq_num_records covers K_BYTES). A dropped B bound
    # would therefore multiply live A tail data by live B tail data.
    limit = STANDALONE_LOGITS_DIFF_LIMIT
    result = run_standalone_v2_a8w8(
        BM=32,
        BN=128,
        BK=BK,
        K=2 * BK,
        inter_dim_pad=BK,
        limit=limit,
    )
    assert not result.out.isnan().any()
    assert logits_diff(result.out, result.ref) <= limit
    # Guard against a vacuous test: the tail must carry enough signal that
    # consuming it would be visible.
    assert logits_diff(result.ref_if_pad_consumed, result.ref) > 100 * limit


@pytest.mark.l2_device
def test_v2_a8w8_scale_poison_negative_control():
    limit = STANDALONE_LOGITS_DIFF_LIMIT
    bad = run_standalone_v2_a8w8(
        BM=32,
        BN=128,
        BK=128,
        K=384,
        poison_real_scale=True,
        limit=limit,
    )
    assert bad.out.isnan().any() or logits_diff(bad.out, bad.ref) > 100 * limit


@dataclass
class Stage1ConsistencyCase:
    data: dict
    inputs: dict
    BM: int
    K: int

    def synthetic_sorted_scale(self):
        ref1 = self.data["ref1"].contiguous()
        payload, scale = quant_a(ref1.view(-1, self.K), "fp8")
        del payload
        return _mxfp4_a_scale_sorted_shuffled(
            scale.view(torch.uint8),
            self.inputs["sti"],
            self.inputs["cumsum"],
            self.inputs["max_sorted"],
            self.K,
            BM=self.BM,
            BK=256,
            source_topk=self.data["topk_ids"].shape[1],
        )

    def synthetic_payload_and_scale(self):
        payload, scale = quant_a(self.data["ref1"].contiguous().view(-1, self.K), "fp8")
        payload = payload.view(torch.uint8).view(-1, self.K)
        packed = self.inputs["sti"]
        token = self.data["inp"].shape[0]
        topk = self.data["topk_ids"].shape[1]
        token_ids = packed & 0x00FFFFFF
        slots = (packed >> 24) & 0xFF
        valid = (token_ids < token) & (slots < topk)
        source_rows = token_ids * topk + slots
        sorted_payload = torch.zeros(
            (self.inputs["max_sorted"], self.K),
            dtype=torch.uint8,
            device=payload.device,
        )
        sorted_payload[valid] = payload[source_rows[valid].long()]
        sorted_scale = _mxfp4_a_scale_sorted_shuffled(
            scale.view(torch.uint8),
            packed,
            self.inputs["cumsum"],
            self.inputs["max_sorted"],
            self.K,
            BM=self.BM,
            BK=256,
            source_topk=topk,
        )
        return sorted_payload, sorted_scale

    def run_real_v2_stage1(self):
        from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

        a1_scale_sort = moe_mxfp4_sort(
            self.data["a1_scale"].view(self.data["inp"].shape[0], 1, -1),
            sorted_ids=self.inputs["sti"],
            num_valid_ids=self.inputs["cumsum"],
            token_num=self.data["inp"].shape[0],
            block_size=self.BM,
        )
        result = flydsl_moe_stage1(
            a=self.data["a1_qt"],
            w1=self.data["base"]["w1_qt_shuf"],
            out=self.inputs["isq"],
            sorted_token_ids=self.inputs["sti"],
            sorted_expert_ids=self.inputs["sei"],
            num_valid_ids=self.inputs["cumsum"],
            topk=self.data["topk_ids"].shape[1],
            tile_m=self.BM,
            tile_n=128,
            tile_k=256,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype="fp8",
            act="swiglu",
            w1_scale=self.data["base"]["w1_scale_shuf"],
            a1_scale=a1_scale_sort,
            gate_mode="interleave",
            a_scale_one=False,
            v2_output_layout=True,
        )
        assert isinstance(result, tuple) and len(result) == 2, (
            "v2 stage1 must return (out, out_scale_sorted); "
            "check out_dtype='fp8' and v2_output_layout=True"
        )
        return result

    def _valid_routes_and_offsets(self):
        token = self.data["inp"].shape[0]
        topk = self.data["topk_ids"].shape[1]
        padded_cols = ((self.K // 32) + 7) // 8 * 8
        routes = []
        offsets = []
        for row in range(self.inputs["max_sorted"]):
            packed = int(self.inputs["sti"][row].item())
            token_id = packed & 0x00FFFFFF
            slot = (packed >> 24) & 0xFF
            if token_id >= token or slot >= topk:
                continue
            source_row = token_id * topk + slot
            for k_group in range(self.K // 32):
                d0 = row // 32
                d1 = (row // 16) & 1
                d2 = row & 15
                d3 = k_group // 8
                d4 = (k_group // 4) & 1
                d5 = k_group & 3
                routes.append((source_row, k_group))
                offsets.append(
                    d0 * (padded_cols * 32) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1
                )
        return routes, torch.tensor(
            offsets, dtype=torch.long, device=self.inputs["sti"].device
        )

    def consumed_scale_bytes(self, scale):
        _, offsets = self._valid_routes_and_offsets()
        return scale.view(torch.uint8).view(-1)[offsets]

    def reshuffle_runtime_scale(self, runtime_scale):
        routes, offsets = self._valid_routes_and_offsets()
        topk = self.data["topk_ids"].shape[1]
        logical = torch.zeros(
            self.data["inp"].shape[0] * topk,
            self.K // 32,
            dtype=torch.uint8,
            device=runtime_scale.device,
        )
        physical = runtime_scale.view(torch.uint8).view(-1)
        for (source_row, k_group), offset in zip(routes, offsets.tolist()):
            logical[source_row, k_group] = physical[offset]
        return _mxfp4_a_scale_sorted_shuffled(
            logical,
            self.inputs["sti"],
            self.inputs["cumsum"],
            self.inputs["max_sorted"],
            self.K,
            BM=self.BM,
            BK=256,
            source_topk=topk,
        )

    def required_consumed_bytes(self):
        routes, _ = self._valid_routes_and_offsets()
        return len(routes)

    def poison_k_padding(self, scale):
        poisoned = scale.view(torch.uint8).clone()
        words = poisoned.view(-1, 4)
        chunks = (self.K + 255) // 256
        groups = int(self.inputs["n"]) // 32
        for group in range(groups):
            start = (group * chunks + chunks - 1) * 64
            words[start : start + 64, 2:4] = 0xFF
        return poisoned

    def run_gemm2(self, payload, scale):
        from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

        token = self.data["inp"].shape[0]
        model_dim = self.data["inp"].shape[1]
        topk = self.data["topk_ids"].shape[1]
        out = torch.zeros(
            (token, model_dim), dtype=torch.bfloat16, device=payload.device
        )
        mxfp4_moe_gemm2(
            inter_sorted_quant=payload.view(torch.uint8),
            inter_sorted_shuffled_scale=scale.view(torch.uint8),
            w2_u8=self.inputs["w2u8"],
            w2_scale_u8=self.inputs["w2sc"],
            sorted_expert_ids=self.inputs["sei"],
            cumsum_tensor=self.inputs["cumsum"],
            sorted_token_ids=self.inputs["sti"],
            sorted_weights=self.inputs["swt"],
            out=out,
            M_logical=token,
            max_sorted=self.inputs["max_sorted"],
            NE=self.data["w2_qt"].shape[0],
            D_HIDDEN=model_dim,
            D_INTER=self.K,
            topk=topk,
            BM=self.BM,
            BN=128,
            BK=128,
            a_dtype="fp8",
            b_dtype="fp8",
            epilog="atomic",
            SBM=self.BM,
            n_sorted_padded=self.inputs["n"],
        )
        torch.cuda.synchronize()
        return out


def build_stage1_consistency_case(BM, K):
    with torch.device("cuda"):
        data = gen(
            16,
            256,
            K,
            8,
            2,
            BM,
            adtype="fp8",
            b_dtype="fp8",
            activation=ActivationType.Swiglu,
        )
        inputs = build_v2_inputs(data, 16, 256, K, 8, 2, BM)
    return Stage1ConsistencyCase(data=data, inputs=inputs, BM=BM, K=K)


@pytest.mark.l2_device
def test_v2_a8w8_k384_tuner_and_stage1_scale_layout_match():
    case = build_stage1_consistency_case(BM=32, K=384)
    synthetic = case.synthetic_sorted_scale()
    runtime_payload, runtime_scale = case.run_real_v2_stage1()
    expected_cols = ((384 // 32) + 7) // 8 * 8
    assert runtime_scale.shape[1] == expected_cols, (
        "runtime v2 stage1 scale stride diverged from GEMM2's K/256 ABI; "
        "do not relax this assertion"
    )
    assert synthetic.numel() >= case.required_consumed_bytes()
    assert runtime_scale.numel() >= case.required_consumed_bytes()
    runtime_roundtrip = case.reshuffle_runtime_scale(runtime_scale)
    assert torch.equal(
        case.consumed_scale_bytes(runtime_roundtrip),
        case.consumed_scale_bytes(runtime_scale.view(torch.uint8)),
    )
    synthetic_bytes = case.consumed_scale_bytes(synthetic).to(torch.int16)
    runtime_bytes = case.consumed_scale_bytes(runtime_scale).to(torch.int16)
    assert (synthetic_bytes - runtime_bytes).abs().max() <= 1

    ref = v2_stage1_sorted_ref(
        case.data["ref1"],
        case.data["topk_ids"],
        case.inputs["sti"],
        case.inputs["sei"],
        case.inputs["n"],
        token=case.data["inp"].shape[0],
        inter_dim=case.K,
        bm_s1=case.BM,
        max_sorted=case.inputs["max_sorted"],
    )
    assert (
        v2_stage1_dequant_cosine_err(
            ref,
            runtime_payload.view(torch.uint8).view(case.inputs["max_sorted"], case.K),
            printLog=False,
            inter_dim=case.K,
            adtype="fp8",
        )
        <= 1e-3
    )


@pytest.mark.l2_device
def test_v2_stage1_scale_producers_ignore_k_padding_in_chained_gemm2():
    case = build_stage1_consistency_case(BM=32, K=384)
    runtime_payload, runtime_scale = case.run_real_v2_stage1()
    runtime_clean = case.run_gemm2(runtime_payload, runtime_scale)
    runtime_poisoned = case.run_gemm2(
        runtime_payload, case.poison_k_padding(runtime_scale)
    )
    torch.testing.assert_close(runtime_poisoned, runtime_clean, rtol=0, atol=0)

    synthetic_payload, synthetic_scale = case.synthetic_payload_and_scale()
    synthetic_clean = case.run_gemm2(synthetic_payload, synthetic_scale)
    synthetic_poisoned = case.run_gemm2(
        synthetic_payload, case.poison_k_padding(synthetic_scale)
    )
    torch.testing.assert_close(synthetic_poisoned, synthetic_clean, rtol=0, atol=0)


@pytest.mark.l2_device
def test_v2_a8w8_bm16_standalone_only():
    result = run_standalone_v2_a8w8(
        BM=16, BN=128, BK=128, K=384, limit=STANDALONE_LOGITS_DIFF_LIMIT
    )
    assert logits_diff(result.out, result.ref) <= result.limit


@pytest.mark.l2_device
@pytest.mark.parametrize(
    "BM,BN,epilog",
    [
        pytest.param(64, 256, "atomic", id="bm64_bn256_atomic_nt"),
        pytest.param(128, 256, "reduce", id="bm128_bn256_reduce_nt"),
    ],
)
def test_v2_a8w8_large_tile_nt_branches(BM, BN, epilog):
    limit = STANDALONE_LOGITS_DIFF_LIMIT
    result = run_standalone_v2_a8w8(
        BM=BM,
        BN=BN,
        BK=256,
        K=512,
        epilog=epilog,
        use_nt=True,
        limit=limit,
    )
    assert not result.out.isnan().any()
    assert logits_diff(result.out, result.ref) <= limit
