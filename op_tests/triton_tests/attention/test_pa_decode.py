# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import glob
import itertools
import json
import random

import pytest
import torch
import triton
import triton.language as tl

import aiter.ops.triton.attention.pa_decode as pa_decode_mod
from aiter import logger, pertoken_quant
from aiter.ops.triton._triton_kernels.attention import pa_decode as pa_kernels
from aiter.ops.triton.attention.pa_decode import paged_attention_decode
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def paged_attention_decode_ref(
    output,  # [num_seqs, num_q_heads, head_sz]
    query,  # [num_seqs, num_kv_heads, head_sz],
    k_cache,  # [num_seqs, num_kv_heads, head_sz/x, blk_sz, x]
    v_cache,  # [num_seqs, num_kv_heads, head_sz, blk_sz]
    blk_tables,  # [num_seq, max_num_blks_per_seq]
    ctx_lens,  # [num_seqs]
) -> None:
    num_q_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    head_sz = v_cache.shape[2]
    kv_blk_sz = v_cache.shape[3]

    num_seqs = query.shape[0]
    for s in range(num_seqs):
        q = query[s].unsqueeze(0)
        blk_tbl = blk_tables[s]
        ctx_len = ctx_lens[s]

        keys = []
        values = []
        for j in range(ctx_len):
            blk_number = int(blk_tbl[j // kv_blk_sz])
            blk_offset = j % kv_blk_sz

            k = k_cache[blk_number, :, :, blk_offset, :]
            k = k.reshape(num_kv_heads, head_sz)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v = v_cache[blk_number, :, :, blk_offset]
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_sz**0.5)
        q = q * scale
        attn = torch.einsum("qhd,khd->hqk", q, keys)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk, khd->qhd", attn, values)
        out = out.view(num_q_heads, head_sz)
        output[s].copy_(out, non_blocking=True)


tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}


def input_helper(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks=4,
    random_seed: int = 0,
):
    """Helper function to generate input tensors for paged attention testing."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)

    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype)
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(
            D, 16 // torch.tensor([], dtype=torch.float16, device="cuda").element_size()
        )
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(
            D,
            16 // torch.tensor([], dtype=kv_cache_dtype, device="cuda").element_size(),
        )
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")

    return (
        query,
        output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    )


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (8, 1)])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 16])
@pytest.mark.parametrize("SEQ_LEN", [1, 57, 1024])
@pytest.mark.parametrize("NUM_BLK", [4, 16])
# q_dtype, kv_dtype, compute_type, output_type
# INT8xINT8 -> BF16-> BF16
# FP8xFP8 -> BF16-> FP8
# BF16xINT8 -> BF16-> BF16
# BF16xFP8 -> BF16-> BF16
# BF16xBF16->BF16->BF16
# FP16xFP16->FP16->FP16
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.int8, tl.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.int8, torch.int8, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn(
    B,
    H_Q,
    H_KV,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):

    head_size = 128

    if SEQ_LEN >= 8192 and B >= 16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    num_blocks = NUM_BLK

    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B,
        H_Q,
        H_KV,
        head_size,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )

    attn_scale = 1.0 / (head_size**0.5)

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, head_size, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )

    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


# test_paged_attn stops at SEQ_LEN=1024, a single 1024-token partition, so it
# only ever reaches the V1 kernels. SEQ_LEN=4096 spans four partitions, which
# dispatches to the V2 (partitioned + reduce) kernels; the spy fails the test
# if a dispatch change ever routes these cases back to V1.
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (8, 1)])
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn_v2(
    B, H_Q, H_KV, dtype, kv_cache_dtype, compute_type, output_type, monkeypatch
):
    # paged_attention_decode looks up paged_attn_decode_v2 in its module globals.
    v2_calls = []
    paged_attn_decode_v2 = paged_attention_decode.__globals__["paged_attn_decode_v2"]

    def spy(*args, **kwargs):
        v2_calls.append(True)
        return paged_attn_decode_v2(*args, **kwargs)

    monkeypatch.setitem(paged_attention_decode.__globals__, "paged_attn_decode_v2", spy)

    test_paged_attn(
        B,
        H_Q,
        H_KV,
        KV_BLK_SZ=16,
        SEQ_LEN=4096,
        NUM_BLK=16,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        compute_type=compute_type,
        output_type=output_type,
    )
    assert v2_calls, "SEQ_LEN=4096 was expected to dispatch to the V2 kernels"


@pytest.mark.parametrize("B, H_Q, H_KV", [(1, 8, 1), (4, 16, 16), (64, 16, 16)])
@pytest.mark.parametrize("SEQ_LEN", [2048, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_multi_partition_dispatch(B, H_Q, H_KV, SEQ_LEN, dtype, monkeypatch):
    """Check output and V1/V2 choice for bf16/fp16 caches above 1024 tokens."""
    head_size, kv_blk_sz = 128, 16
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]
    num_blocks = (SEQ_LEN + kv_blk_sz - 1) // kv_blk_sz
    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B, H_Q, H_KV, head_size, kv_blk_sz, SEQ_LEN, dtype, dtype, dtype, num_blocks
    )

    called = []
    for name in ("paged_attn_decode_v1", "paged_attn_decode_v2"):
        fn = getattr(pa_decode_mod, name)
        monkeypatch.setattr(
            pa_decode_mod,
            name,
            lambda *a, _fn=fn, _name=name, **k: (called.append(_name), _fn(*a, **k))[1],
        )

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        1.0 / head_size**0.5,
        int(max_context_len),
        compute_type,
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )

    num_partitions = -(-SEQ_LEN // pa_decode_mod._SEQ_PARTITION_SIZE)
    rule = pa_decode_mod._get_dispatch_config(
        {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]
    )
    if rule is not None:
        expect_v1 = num_partitions <= rule["v1_max_partitions"]
        busy_min = rule.get("v1_min_programs")
        if busy_min is not None and B * H_KV >= busy_min:
            expect_v1 = expect_v1 or num_partitions <= rule["v1_busy_max_partitions"]
    else:
        expect_v1 = SEQ_LEN <= 8192 and (num_partitions == 1 or B * H_Q > 512)
    if get_arch() == "gfx950":
        assert rule is not None and not expect_v1, "gfx950 bf16/fp16 must use V2 here"
    assert called == ["paged_attn_decode_v1" if expect_v1 else "paged_attn_decode_v2"]

    torch_output = torch.zeros(B, H_Q, head_size, dtype=dtype, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("B", [1, 4, 57, 64])
# @pytest.mark.parametrize("H_Q, H_KV", [(1,1), (16, 16), (2,1), (24,4)]) #TODO: GQA failing
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (16, 16)])
@pytest.mark.parametrize("D", [1, 64, 128])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 4, 512])
@pytest.mark.parametrize("SEQ_LEN", [1, 32, 57, 512, 10000])
@pytest.mark.parametrize("NUM_BLK", [4, 32])
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.float16, torch.float16, tl.float16, torch.float16),
        # (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn_per_token_quant(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):
    if D == 128 and KV_BLK_SZ == 512:  # Causes Shared Memory out of resources on Mi300
        pytest.skip("D={128} and KV_BLK_SZ={512} causes shared memory out of resources")

    if SEQ_LEN >= 8192 and B >= 16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    num_blocks = NUM_BLK

    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )

    attn_scale = 1.0 / (D**0.5)

    (
        key_cache_tri_quant,
        k_scale,
    ) = pertoken_quant(
        key_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz
    )
    (
        value_cache_tri_quant,
        v_scale,
    ) = pertoken_quant(
        value_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz
    )

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri_quant,
        value_cache_tri_quant,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    logger.debug(
        "B=%d H_Q=%d, H_KV=%d D=%d, KV_BLK_SZ=%d, SEQ_LEN=%d, NUM_BLK=%d",
        B,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        NUM_BLK,
    )
    logger.debug("query=%s", query)
    logger.debug(
        "key_cache_tri.shape=%s key_cache_tri=%s", key_cache_tri.shape, key_cache_tri
    )
    logger.debug("k_scale.shape=%s k_scale=%s", k_scale.shape, k_scale)
    logger.debug(
        "key_cache_tri_quant.shape=%s key_cache_tri_quant=%s",
        key_cache_tri_quant.shape,
        key_cache_tri_quant,
    )
    logger.debug("v_scale.shape=%s v_scale=%s", v_scale.shape, v_scale)
    logger.debug(
        "value_cache_tri.shape=%s value_cache_tri=%s",
        value_cache_tri.shape,
        value_cache_tri,
    )
    logger.debug(
        "value_cache_tri_quant.shape=%s value_cache_tri_quant=%s",
        value_cache_tri_quant.shape,
        value_cache_tri_quant,
    )
    logger.debug("triton_output=%s", triton_output)
    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    logger.debug("torch_output=%s", torch_output)

    torch.testing.assert_close(triton_output, torch_output, rtol=2.5e-1, atol=2.5e-1)


# Archs that have a PA-DECODE config file. Add an arch here when it gets one.
_ARCHS = ("gfx942", "gfx950", "gfx1250")
_PAGES = (16, 64, 128)
_MAX_NUM_STAGES = 2  # 3 or more hangs the GPU on gfx1250


def _config(arch):
    cfg_dir = resolve_config_dir("attention", "PA-DECODE", backend="triton", arch=arch)
    return load_config_json(f"{cfg_dir}/DEFAULT.json", required=False) or {}


_SPLIT_RULES = [
    pytest.param(rule, id=f"{arch}-{dtype}")
    for arch in _ARCHS
    for dtype, rule in _config(arch).get("dispatch", {}).items()
    if "split_target_programs" in rule
]


@pytest.mark.parametrize("arch", _ARCHS)
def test_pa_decode_config_is_valid(arch):
    config = _config(arch)
    assert config, f"no PA-DECODE config for {arch}"
    for rule in config.get("dispatch", {}).values():
        if "split_target_programs" in rule:
            assert {"split_max_partitions", "split_sizes"} <= set(rule)
            # The default size means "do not split".
            assert pa_decode_mod._SEQ_PARTITION_SIZE in rule["split_sizes"]
    # The wrapper has no built-in launch options, so every arch needs this entry.
    assert "v1_w_dot" in config["launch"]
    for table in config["launch"].values():
        assert "any" in table
        for key, entry in table.items():
            assert key == "any" or key.startswith("BLK_LEQ_")
            assert entry.get("num_stages", 1) <= _MAX_NUM_STAGES


def test_missing_config_gives_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pa_kernels, "resolve_config_dir", lambda *a, **k: str(tmp_path))
    load_config_json.cache_clear()
    try:
        assert pa_kernels._get_dispatch_config("fp16") is None
        assert pa_kernels._get_launch_config("v1_w_dot", _PAGES[0]) is None
    finally:
        load_config_json.cache_clear()


def _cu_override_files(arch):
    cfg_dir = resolve_config_dir("attention", "PA-DECODE", backend="triton", arch=arch)
    return sorted(glob.glob(f"{cfg_dir}/PA-DECODE-CU=*.json"))


_CU_OVERRIDES = [
    pytest.param(arch, path, id=f"{arch}-{path.rsplit('/', 1)[-1]}")
    for arch in _ARCHS
    for path in _cu_override_files(arch)
]


@pytest.mark.parametrize("arch,path", _CU_OVERRIDES)
def test_cu_override_only_replaces_default_keys(arch, path):
    assert int(path.rsplit("=", 1)[-1][: -len(".json")]) > 0
    default, override = _config(arch), load_config_json(path)
    assert set(override) <= {"dispatch", "launch"}
    for dtype, rule in override.get("dispatch", {}).items():
        assert set(rule) <= set(default["dispatch"][dtype])
    for kernel, table in override.get("launch", {}).items():
        assert set(table) <= set(default["launch"][kernel])
        for entry in table.values():
            assert entry.get("num_stages", 1) <= _MAX_NUM_STAGES


def test_cu_override_applies_only_on_matching_cu_count(tmp_path, monkeypatch):
    default = {
        "dispatch": {"fp16": {"v1_min_programs": 2048, "split_max_programs": 1024}},
        "launch": {"v1_w_dot": {"any": {"num_stages": 1}}},
    }
    override = {
        "dispatch": {"fp16": {"v1_min_programs": 256}},
        "launch": {"v1_w_dot": {"any": {"num_stages": 2}}},
    }
    (tmp_path / "DEFAULT.json").write_text(json.dumps(default))
    (tmp_path / "PA-DECODE-CU=96.json").write_text(json.dumps(override))
    monkeypatch.setattr(pa_kernels, "resolve_config_dir", lambda *a, **k: str(tmp_path))
    load_config_json.cache_clear()
    try:
        monkeypatch.setattr(pa_kernels, "get_num_sms", lambda: 96)
        assert pa_kernels._get_dispatch_config("fp16") == {
            "v1_min_programs": 256,
            "split_max_programs": 1024,
        }
        assert pa_kernels._get_launch_config("v1_w_dot", _PAGES[0]) == {"num_stages": 2}
        monkeypatch.setattr(pa_kernels, "get_num_sms", lambda: 256)
        assert pa_kernels._get_dispatch_config("fp16") == default["dispatch"]["fp16"]
        assert pa_kernels._get_launch_config("v1_w_dot", _PAGES[0]) == {"num_stages": 1}
    finally:
        load_config_json.cache_clear()


@pytest.mark.parametrize("rule", _SPLIT_RULES)
def test_split_size_follows_rule(rule):
    default = pa_decode_mod._SEQ_PARTITION_SIZE
    busy = rule.get("split_max_programs", 1 << 20)
    short_seq = rule.get("split_short_seq", 0)
    short_programs = rule.get("split_short_min_programs", 1 << 20)
    programs = [1, busy - 1, busy, 2 * busy]
    seqs = [max(short_seq // 2, 1), max(short_seq, 1), 4 * default]
    for n, seq, page in itertools.product(programs, seqs, _PAGES):
        size = pa_decode_mod._select_seq_partition_size(n, seq, page, rule)
        info = f"programs={n} seq={seq} page={page} size={size}"
        assert size in rule["split_sizes"] and size % page == 0, info
        if n >= busy or (seq < short_seq and n >= short_programs):
            assert size == default, info
        if size == default:
            continue
        assert triton.cdiv(seq, size) <= rule["split_max_partitions"], info
        # No larger size that is allowed would also reach the target.
        for larger in (s for s in rule["split_sizes"] if size < s <= seq):
            reaches = n * triton.cdiv(seq, larger) >= rule["split_target_programs"]
            assert not (reaches and larger % page == 0), info
    # The short-sequence rule stops below its limit. At the limit it does not apply.
    no_short = {k: v for k, v in rule.items() if not k.startswith("split_short")}
    for n, page in itertools.product(programs, _PAGES):
        at_limit = pa_decode_mod._select_seq_partition_size(n, short_seq, page, rule)
        expected = pa_decode_mod._select_seq_partition_size(
            n, short_seq, page, no_short
        )
        assert at_limit == expected, f"programs={n} seq={short_seq} page={page}"


@pytest.mark.parametrize("group_size", [4, 1], ids=["gqa", "mha"])
@pytest.mark.parametrize("rule", _SPLIT_RULES)
def test_dispatch_passes_partition_size(rule, group_size, monkeypatch):
    seq, page, kv_heads = 4 * pa_decode_mod._SEQ_PARTITION_SIZE, _PAGES[-1], 1
    called = []
    monkeypatch.setattr(pa_decode_mod, "_get_dispatch_config", lambda dtype: rule)
    for name in ("v1", "v2"):
        monkeypatch.setattr(
            pa_decode_mod,
            f"paged_attn_decode_{name}",
            lambda *a, _name=name, **k: called.append((_name, a, k)),
        )
    paged_attention_decode(
        output=None,
        query=torch.empty(1, group_size * kv_heads, 1, dtype=torch.float16),
        key_cache=torch.empty(1, kv_heads, page, 1, dtype=torch.float16),
        value_cache=torch.empty(1, kv_heads, page, 1, dtype=torch.float16),
        seq_lens=None,
        block_tables=None,
        attn_scale=1.0,
        max_seq_len=seq,
        compute_type=None,
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )
    # Only the GQA kernels use the selected size.
    expected = pa_decode_mod._SEQ_PARTITION_SIZE
    if group_size > 1:
        expected = pa_decode_mod._select_seq_partition_size(kv_heads, seq, page, rule)
    ((_, args, kwargs),) = called
    assert kwargs["seq_partition_size"] == expected
    assert args[-1] == triton.cdiv(seq, expected)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("KV_BLK_SZ", [_PAGES[0], _PAGES[-1]])
@pytest.mark.parametrize("SEQ_LEN", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_split_output(B, KV_BLK_SZ, SEQ_LEN, dtype):
    """Check GQA decode output when the sequences have different lengths."""
    H_Q, H_KV, head_size = 32, 8, 128
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]
    num_blocks = B * triton.cdiv(SEQ_LEN, KV_BLK_SZ)
    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        _,
        block_tables,
        _,
    ) = input_helper(
        B, H_Q, H_KV, head_size, KV_BLK_SZ, SEQ_LEN, dtype, dtype, dtype, num_blocks
    )
    context_lens = torch.randint(1, SEQ_LEN + 1, (B,), device="cuda")
    context_lens[0] = SEQ_LEN
    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        1.0 / head_size**0.5,
        SEQ_LEN,
        compute_type,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
    )
    torch_output = torch.zeros(B, H_Q, head_size, dtype=dtype, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


_TDM_LENS = [8192, 1, 127, 128, 129, 255, 257, 4000, 7777, 12345]


def _gqa_tdm_case(
    dtype, H_KV, group, lens_list, poison=False, reference=True, head_size=128
):
    """Random paged cache, an fp32 reference and optional NaN/Inf in unused rows."""
    page = 128
    B = len(lens_list)
    max_pages = triton.cdiv(max(lens_list), page)
    num_blocks = B * max_pages + 3
    query = torch.randn(B, H_KV * group, head_size, device="cuda").to(dtype)
    key_cache = torch.randn(num_blocks, H_KV, page, head_size, device="cuda").to(dtype)
    value_cache = torch.randn(num_blocks, H_KV, page, head_size, device="cuda").to(
        dtype
    )
    perm = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    block_tables = perm[: B * max_pages].view(B, max_pages)
    lens = torch.tensor(lens_list, device="cuda")
    expected = None
    if reference:
        expected = torch.empty(
            B, H_KV * group, head_size, dtype=torch.float32, device="cuda"
        )
        for b, n in enumerate(lens_list):
            blocks = block_tables[b, : triton.cdiv(n, page)].long()
            k = key_cache[blocks].permute(1, 0, 2, 3).reshape(H_KV, -1, head_size)
            v = value_cache[blocks].permute(1, 0, 2, 3).reshape(H_KV, -1, head_size)
            q = query[b].float().view(H_KV, group, head_size)
            scores = torch.einsum("hgd,hld->hgl", q, k[:, :n].float())
            probs = (scores / head_size**0.5).softmax(-1)
            expected[b] = torch.einsum("hgl,hld->hgd", probs, v[:, :n].float()).reshape(
                H_KV * group, head_size
            )
    if poison:
        for b, n in enumerate(lens_list):
            if n % page:
                blk = int(block_tables[b, n // page])
                key_cache[blk, :, n % page :] = float("nan")
                value_cache[blk, :, n % page :] = float("inf")
    return query, key_cache, value_cache, block_tables, lens, expected


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, num_warps", [(8, 4, 2), (8, 16, 4), (1, 16, 1), (2, 1, 2)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm(dtype, H_KV, group, num_warps, poison):
    """TDM GQA kernel against an fp32 reference, with partial pages and garbage rows."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS, poison
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output, query, key_cache, value_cache, block_tables, lens, 128**-0.5, num_warps
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, head_size, num_warps",
    [(128, 1, 56, 1), (8, 4, 64, 2), (4, 16, 56, 4)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm_head_sizes(
    dtype, H_KV, group, head_size, num_warps, poison
):
    """Head sizes below the padded tile (56) and 64, including MHA-like groups of 1."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS[:6], poison, head_size=head_size
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        lens,
        head_size**-0.5,
        num_warps,
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


def test_paged_attn_gqa_tdm_empty_sequence():
    """A sequence of length 0 writes zeros and does not disturb its neighbours."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    lens_list = [300, 0, 129]
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        torch.float16, 8, 4, [300, 1, 129]
    )
    lens = torch.tensor(lens_list, device="cuda")
    output = torch.full_like(query, 7.0)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output, query, key_cache, value_cache, block_tables, lens, 128**-0.5, 2
    )
    assert torch.count_nonzero(output[1]) == 0
    for b in (0, 2):
        torch.testing.assert_close(output[b].float(), expected[b], rtol=0, atol=4e-3)


def _tdm_dispatch_calls(
    monkeypatch,
    rule,
    *,
    scale=1.0,
    alibi=None,
    head=128,
    kv=8,
    seqs=2,
    max_seq_len=1024,
):
    calls = []
    monkeypatch.setattr(pa_decode_mod, "_get_dispatch_config", lambda dtype: rule)
    monkeypatch.setattr(pa_decode_mod, "_pa_decode_gqa_tdm_kernel", object())
    monkeypatch.setattr(
        pa_decode_mod,
        "paged_attn_decode_gqa_tdm",
        lambda *a, **k: calls.append(("tdm", k)),
    )
    for name in ("v1", "v2"):
        monkeypatch.setattr(
            pa_decode_mod,
            f"paged_attn_decode_{name}",
            lambda *a, _name=name, **k: calls.append((_name, k)),
        )
    # The launchers are mocked, so CPU tensors are enough.
    paged_attention_decode(
        output=torch.empty(seqs, 4 * kv, head, dtype=torch.float16, device="cpu"),
        query=torch.empty(seqs, 4 * kv, head, dtype=torch.float16, device="cpu"),
        key_cache=torch.empty(1, kv, 128, head, dtype=torch.float16, device="cpu"),
        value_cache=torch.empty(1, kv, 128, head, dtype=torch.float16, device="cpu"),
        seq_lens=None,
        block_tables=None,
        attn_scale=1.0,
        max_seq_len=max_seq_len,
        compute_type=None,
        k_scale=torch.tensor([scale]),
        v_scale=torch.tensor([1.0]),
        alibi_slopes=alibi,
    )
    return [name for name, _ in calls], calls


_TDM_RULE = {
    "v1_max_partitions": 1,
    "tdm_num_warps": 4,
}


def test_dispatch_uses_tdm_when_enabled(monkeypatch):
    names, calls = _tdm_dispatch_calls(monkeypatch, _TDM_RULE)
    assert names == ["tdm"] and calls[0][1] == {
        "num_warps": 4,
        "seq_partition_size": pa_decode_mod._TDM_NO_SPLIT,
        "max_seq_len": 1024,
    }


@pytest.mark.parametrize(
    "rule_change, kwargs",
    [
        ({"tdm_num_warps": None}, {}),
        ({}, {"scale": 2.0}),
        ({}, {"alibi": torch.ones(32)}),
        ({}, {"head": 80}),
    ],
    ids=["disabled", "k_scale", "alibi", "head_size"],
)
def test_dispatch_skips_tdm_when_unsupported(rule_change, kwargs, monkeypatch):
    names, _ = _tdm_dispatch_calls(monkeypatch, {**_TDM_RULE, **rule_change}, **kwargs)
    assert "tdm" not in names


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, head_size, seq_partition_size",
    [(8, 4, 128, 128), (1, 16, 128, 256), (2, 1, 128, 1024), (4, 1, 56, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm_split(
    dtype, H_KV, group, head_size, seq_partition_size, poison
):
    """Split TDM kernel plus the v2 reduce kernel, with partitions past short sequences."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS, poison, head_size=head_size
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        lens,
        head_size**-0.5,
        2,
        seq_partition_size,
        max(_TDM_LENS),
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


@pytest.mark.parametrize("arch", _ARCHS)
def test_tdm_split_config_is_valid(arch):
    for rule in _config(arch).get("dispatch", {}).values():
        if rule.get("tdm_split_target_programs") is None:
            continue
        assert rule["tdm_split_max_partitions"] >= 1
        assert all(s % 128 == 0 for s in rule["tdm_split_sizes"])


_TDM_SPLIT_RULE = {
    "tdm_split_target_programs": 1024,
    "tdm_split_sizes": [4096, 2048, 1024, 512, 256],
    "tdm_split_max_partitions": 64,
}


@pytest.mark.parametrize("max_seq_len", [1, 256, 4000, 8192, 100000])
@pytest.mark.parametrize("programs", [1, 8, 100, 512, 1023, 1024, 5000])
def test_tdm_partition_size_follows_rule(programs, max_seq_len):
    no_split = pa_decode_mod._TDM_NO_SPLIT
    rule = _TDM_SPLIT_RULE
    size = pa_decode_mod._select_tdm_partition_size(programs, max_seq_len, rule)
    info = f"programs={programs} seq={max_seq_len} size={size}"
    if programs >= rule["tdm_split_target_programs"]:
        assert size == no_split, info
    if size == no_split:
        return
    assert size in rule["tdm_split_sizes"] and size < max_seq_len, info
    parts = triton.cdiv(max_seq_len, size)
    assert parts <= rule["tdm_split_max_partitions"], info
    # No larger allowed size would also reach the target.
    for larger in (s for s in rule["tdm_split_sizes"] if size < s < max_seq_len):
        reaches = programs * triton.cdiv(max_seq_len, larger) >= 1024
        assert not reaches, info


def test_tdm_partition_size_is_off_without_target():
    rule = {
        k: v for k, v in _TDM_SPLIT_RULE.items() if k != "tdm_split_target_programs"
    }
    assert pa_decode_mod._select_tdm_partition_size(1, 8192, rule) == (
        pa_decode_mod._TDM_NO_SPLIT
    )


def test_dispatch_passes_tdm_partition_size(monkeypatch):
    # No size reaches the target for 16 programs and 1024 tokens, so the smallest size is used.
    rule = {**_TDM_RULE, **_TDM_SPLIT_RULE}
    _, calls = _tdm_dispatch_calls(monkeypatch, rule)
    assert calls[0][1]["seq_partition_size"] == 256
    assert calls[0][1]["max_seq_len"] == 1024


def test_paged_attn_gqa_tdm_is_repeatable():
    """A race in the LDS ring would make repeated runs differ."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, _ = _gqa_tdm_case(
        torch.float16, 8, 4, [8192] * 128, reference=False
    )
    first = torch.zeros_like(query)
    args = (query, key_cache, value_cache, block_tables, lens, 128**-0.5)
    pa_decode_mod.paged_attn_decode_gqa_tdm(first, *args, 1)
    output = torch.zeros_like(query)
    for _ in range(20):
        output.zero_()
        pa_decode_mod.paged_attn_decode_gqa_tdm(output, *args, 1)
        assert torch.equal(output, first)


_V1_V2_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def _page128_rule(dtype):
    """gfx1250 rule without the Gluon kernel, so the Triton V1/V2 path is chosen."""
    return {
        **_config("gfx1250")["dispatch"][_V1_V2_NAMES[dtype]],
        "tdm_num_warps": None,
    }


# (programs relative to v1_min_programs, partitions relative to v1_busy_max_partitions)
@pytest.mark.parametrize(
    "d_programs, d_parts, expect",
    [(0, 0, "v1"), (0, 1, "v2"), (-1, 0, "v2")],
    ids=["busy_at_limit", "over_partition_limit", "below_program_limit"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dispatch_page128_busy_v1_limits(
    dtype, d_programs, d_parts, expect, monkeypatch
):
    rule = _page128_rule(dtype)
    programs = rule["v1_min_programs"] + d_programs
    parts = rule["v1_busy_max_partitions"] + d_parts
    seq = parts * pa_decode_mod._SEQ_PARTITION_SIZE
    names, _ = _tdm_dispatch_calls(
        monkeypatch, rule, kv=1, seqs=programs, max_seq_len=seq
    )
    assert names == [expect]


@pytest.mark.parametrize(
    "min_programs, seq, expect",
    [(8, 8192, "v1"), (8, 9216, "v2"), (9, 8192, "v2")],
    ids=["busy_v1", "over_partition_limit", "not_busy"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_page128_busy_v1_output(
    dtype, min_programs, seq, expect, monkeypatch
):
    """V1/V2 output on 128-token pages; 8 programs stand in for the 2048-program limit."""
    if get_arch() != "gfx1250":
        pytest.skip("the 128-token launch settings are tuned for gfx1250")
    rule = {k: v for k, v in _page128_rule(dtype).items() if not k.startswith("split_")}
    rule["v1_min_programs"] = min_programs
    assert rule["v1_busy_max_partitions"] == 8
    monkeypatch.setattr(pa_decode_mod, "_get_dispatch_config", lambda _: rule)
    called, launches = [], []
    for name in ("paged_attn_decode_v1", "paged_attn_decode_v2"):
        fn = getattr(pa_decode_mod, name)
        monkeypatch.setattr(
            pa_decode_mod,
            name,
            lambda *a, _fn=fn, _name=name, **k: (called.append(_name), _fn(*a, **k))[1],
        )
    get_launch = pa_decode_mod._get_launch_config
    monkeypatch.setattr(
        pa_decode_mod,
        "_get_launch_config",
        lambda kernel, blk: (launches.append((kernel, blk)), get_launch(kernel, blk))[
            1
        ],
    )
    B, H_KV, group = 4, 2, 4
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, [seq] * B
    )
    output = torch.zeros_like(query)
    paged_attention_decode(
        output,
        query,
        key_cache,
        value_cache,
        lens,
        block_tables,
        128**-0.5,
        seq,
        {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype],
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )
    assert called == [f"paged_attn_decode_{expect}"]
    assert launches == [(f"{expect}_w_dot", 128)]
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)
