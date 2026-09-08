# SPDX-License-Identifier: MIT
"""Safety/correctness checks for the FlashKDA descriptor/raw ABIs."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

flash_kda: Any | None = None
get_module: Any | None = None

_CONTEXT_LDS_PIPELINE_PASS_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_B",
    "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_A",
    "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_REPLAY",
)
_CONTEXT_NW8_ENV = "FLASH_KDA_GFX950_CONTEXT_NW8"
_CONTEXT_DIRECT_TAIL_FIRST_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_TAIL_FIRST"
)
_CONTEXT_DIRECT_NW1_FLAT_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_FLAT_TAIL_FIRST"
)
_CONTEXT_DIRECT_PREFIXLESS_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_PREFIXLESS"
)
_CONTEXT_DIRECT_DENSE_N1_H12_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_DENSE_N1_H12"
)
_CONTEXT_DIRECT_GLOBAL_N1_H12_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_N1_H12"
)
_CONTEXT_DIRECT_GLOBAL_KR_GLL_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_KR_GLL"
)
_CONTEXT_DIRECT_GLOBAL_KQ_GLL_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_KQ_GLL"
)
_CONTEXT_DIRECT_KSPLIT_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT"
)
_CONTEXT_DIRECT_KSPLIT_TAIL_MQK_PREFETCH_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT_TAIL_MQK_PREFETCH"
)
_CONTEXT_DIRECT_KSPLIT_LONG_N1_H12_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT_LONG_N1_H12"
)
_CONTEXT_AFFINE_AB_FUSED_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED"
)
_CONTEXT_AFFINE_AB_STAGE_EARLY_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_STAGE_EARLY"
)
_CONTEXT_EQUAL_DENSE_N4_G64_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_EQUAL_DENSE_N4_G64"
)
_CONTEXT_SCAN_KSPLIT_ENV = "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT"
_CONTEXT_PERSISTENT_ENV = "FLASH_KDA_GFX950_CONTEXT_PERSISTENT"
_CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV = (
    "FLASH_KDA_GFX950_CONTEXT_PERSISTENT_ESTABLISHED_AB"
)
_BT16_DENSE_N1_ALL_FULL_C16_ENV = (
    "FLASH_KDA_GFX950_BT16_DENSE_N1_ALL_FULL_C16"
)
_BT16_DENSE_N1_PADDED_SOLVE_ENV = (
    "FLASH_KDA_GFX950_BT16_DENSE_N1_PADDED_SOLVE"
)
_BT16_DENSE_N1_EARLY_BETA_ENV = (
    "FLASH_KDA_GFX950_BT16_DENSE_N1_EARLY_BETA"
)

_DESCRIPTOR_ABI_ARGUMENTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "out",
    "workspace",
    "A_log",
    "dt_bias",
    "initial_state",
    "final_state",
    "cu_seqlens",
    "scale",
    "lower_bound",
    "has_initial_state",
    "output_final_state",
    "is_varlen",
)
_RAW_V1_ABI_ARGUMENTS = (
    "q_ptr",
    "k_ptr",
    "v_ptr",
    "g_ptr",
    "beta_ptr",
    "out_ptr",
    "workspace_ptr",
    "A_log_ptr",
    "dt_bias_ptr",
    "initial_state_ptr",
    "final_state_ptr",
    "cu_seqlens_ptr",
    "B",
    "T",
    "H",
    "N",
    "workspace_bytes",
    "scale",
    "lower_bound",
    "has_initial_state",
    "output_final_state",
    "is_varlen",
    "state_fp32",
    "device_id",
    "stream_ptr",
)
_RAW_V2_ABI_ARGUMENTS = _RAW_V1_ABI_ARGUMENTS + (
    "max_seqlen_upper_bound",
)
_RAW_V3_ABI_ARGUMENTS = _RAW_V2_ABI_ARGUMENTS + ("H_q",)

# A raw-v2 policy check must exercise the zero-environment production route.
# Save and clear every knob that can override either the K2 backend or the
# captured context topology, then restore the caller's environment verbatim.
_RAW_V2_POLICY_ENV = (
    "FLASH_KDA_K2",
    "FLASH_KDA_GFX950_BT16_K1",
    "FLASH_KDA_GFX950_BT16_FUSED",
    _BT16_DENSE_N1_ALL_FULL_C16_ENV,
    _BT16_DENSE_N1_PADDED_SOLVE_ENV,
    _BT16_DENSE_N1_EARLY_BETA_ENV,
    "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    "FLASH_KDA_GFX950_CONTEXT_DIRECT",
    "FLASH_KDA_GFX950_CONTEXT_AFFINE",
    "FLASH_KDA_GFX950_CONTEXT_HYBRID",
    "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
    _CONTEXT_DIRECT_TAIL_FIRST_ENV,
    _CONTEXT_DIRECT_NW1_FLAT_ENV,
    _CONTEXT_DIRECT_PREFIXLESS_ENV,
    _CONTEXT_DIRECT_DENSE_N1_H12_ENV,
    _CONTEXT_DIRECT_GLOBAL_N1_H12_ENV,
    _CONTEXT_DIRECT_GLOBAL_KR_GLL_ENV,
    _CONTEXT_DIRECT_GLOBAL_KQ_GLL_ENV,
    _CONTEXT_DIRECT_KSPLIT_ENV,
    _CONTEXT_DIRECT_KSPLIT_TAIL_MQK_PREFETCH_ENV,
    _CONTEXT_DIRECT_KSPLIT_LONG_N1_H12_ENV,
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_DENSE_ALL_FULL_C16",
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_PAIRED_STATE_X32",
    "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_WAVE_BARRIER",
    _CONTEXT_NW8_ENV,
    _CONTEXT_AFFINE_AB_FUSED_ENV,
    _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
    _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
    "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
    "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
    "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
    "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
    _CONTEXT_SCAN_KSPLIT_ENV,
    _CONTEXT_PERSISTENT_ENV,
    _CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV,
    "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
    "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
    "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
    "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
    "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
    *_CONTEXT_LDS_PIPELINE_PASS_ENV,
)


def validate_input_shape(
    seq_lens: tuple[int, ...], heads: int, *, packed: bool
) -> None:
    """Validate layout metadata without allocating CPU or GPU tensors."""

    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    if (
        not seq_lens
        or any(length < 0 for length in seq_lens)
        or not any(seq_lens)
    ):
        raise ValueError(
            f"seq_lens must contain a nonnegative, nonempty workload, got {seq_lens}"
        )
    if not packed and any(length != seq_lens[0] for length in seq_lens):
        raise ValueError("dense validation requires equal sequence lengths")
    if not packed and seq_lens[0] == 0:
        raise ValueError("dense validation requires a positive sequence length")


def make_inputs(
    seq_lens: tuple[int, ...],
    heads: int,
    device: torch.device | str = torch.device("cuda"),
    *,
    value_heads: int | None = None,
    packed: bool = True,
    state_dtype: torch.dtype = torch.float32,
    has_initial_state: bool = True,
    output_final_state: bool = True,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Build one deterministic packed or dense KDA input."""

    validate_input_shape(seq_lens, heads, packed=packed)
    value_heads = heads if value_heads is None else value_heads
    if value_heads < heads or value_heads % heads != 0:
        raise ValueError(
            "value_heads must be an integer multiple of heads, got "
            f"heads={heads}, value_heads={value_heads}"
        )
    if state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported state dtype: {state_dtype}")

    device = torch.device(device)
    batch = 1 if packed else len(seq_lens)
    tokens = sum(seq_lens) if packed else seq_lens[0]
    qk_shape = (batch, tokens, heads, 128)
    value_shape = (batch, tokens, value_heads, 128)
    torch.manual_seed(seed)
    state = torch.randn(
        len(seq_lens), value_heads, 128, 128, device=device, dtype=torch.float32
    )
    state.mul_(0.02)
    v_axis = torch.linspace(-0.04, 0.03, 128, device=device).view(1, 1, 128, 1)
    k_axis = torch.linspace(0.02, -0.01, 128, device=device).view(1, 1, 1, 128)
    state.add_(v_axis).add_(0.37 * k_axis)
    state = state.to(state_dtype)

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return {
        "q": torch.randn(qk_shape, device=device, dtype=torch.bfloat16),
        "k": torch.randn(qk_shape, device=device, dtype=torch.bfloat16),
        "v": torch.randn(value_shape, device=device, dtype=torch.bfloat16),
        "g": torch.randn(value_shape, device=device, dtype=torch.bfloat16),
        "beta": torch.randn(
            (batch, tokens, value_heads), device=device, dtype=torch.float32
        ),
        "A_log": torch.empty(value_heads, device=device, dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        "dt_bias": torch.randn(
            value_heads * 128, device=device, dtype=torch.float32
        ),
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, device=device, dtype=torch.int32)
            if packed
            else None
        ),
        "scale": 128**-0.5,
        "lower_bound": -5.0,
        "has_initial_state": has_initial_state,
        "output_final_state": output_final_state,
        "state_dtype": state_dtype,
        "N": len(seq_lens),
        "is_varlen": packed,
    }


def split_packed_tokens(tokens: int) -> tuple[int, ...]:
    """Use multi-sequence packed metadata whenever the token count permits."""

    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    sequence_count = min(tokens, 3)
    base, extra = divmod(tokens, sequence_count)
    return tuple(base + (sequence < extra) for sequence in range(sequence_count))


def check_cpu_parameter_validation() -> None:
    """Prove packed empty entries are legal while empty workloads are not."""

    validate_input_shape((0, 1, 0), 1, packed=True)
    validate_input_shape((1, 1), 1, packed=False)
    invalid = (
        ((), 1, True),
        ((0, 0), 1, True),
        ((1, -1), 1, True),
        ((1, 2), 1, False),
        ((0, 0), 1, False),
        ((1,), 0, True),
    )
    for seq_lens, heads, packed in invalid:
        try:
            validate_input_shape(seq_lens, heads, packed=packed)
        except ValueError:
            continue
        raise AssertionError(
            "input validation accepted "
            f"seq_lens={seq_lens}, heads={heads}, packed={packed}"
        )
    for value_heads in (0, 1, 3):
        try:
            make_inputs(
                (1,), 2, "cpu", value_heads=value_heads, packed=True
            )
        except ValueError:
            continue
        raise AssertionError(
            "input construction accepted invalid GVA geometry: "
            f"H_q=2, H_v={value_heads}"
        )
    ratio4 = make_inputs(
        (1,), 1, "cpu", value_heads=4, packed=True, seed=20260827
    )
    if (
        ratio4["q"].shape[2] != 1
        or ratio4["v"].shape[2] != 4
        or ratio4["initial_state"].shape[1] != 4
    ):
        raise AssertionError("input construction lost ratio-4 GVA geometry")
    print("PASS CPU input-shape and integer-ratio GVA validation")


def check_packed_n1_entry_dense_normalization_static() -> None:
    """Pin descriptor/raw-v3 packed-N1 normalization and its dense route."""

    entry_path = _REPO_ROOT / "csrc/kernels/flash_kda/flash_kda_aiter.cu"
    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    entry = entry_path.read_text()
    policy = policy_path.read_text()
    compact_entry = " ".join(entry.split())
    compact_policy = " ".join(policy.split())

    declaration = "const bool single_sequence_packed = is_varlen && N == 1;"
    if compact_entry.count(declaration) != 2:
        raise AssertionError(
            "descriptor/raw-v3 packed-N1 normalization count changed: "
            f"{compact_entry.count(declaration)}"
        )
    entry_contracts = (
        "const int64_t total_tiles64 = single_sequence_packed ? "
        "(total_tokens + chunk - 1) / chunk :",
        "single_sequence_packed ? nullptr : cu_seqlens_ptr",
        "if(is_varlen && !single_sequence_packed) "
        'raw_check_pointer(cu_seqlens_ptr, "cu_seqlens_ptr", alignof(int32_t));',
        "const int64_t launch_tiles = single_sequence_packed ? "
        "(total_tokens + chunk - 1) / chunk :",
        "single_sequence_packed ? nullptr : "
        "reinterpret_cast<const int32_t*>(cu_seqlens_ptr)",
    )
    for contract in entry_contracts:
        if contract not in compact_entry:
            raise AssertionError(
                "packed-N1 entry normalization lost contract: " + contract
            )

    # Once the adapter has erased packed provenance, policy must see the same
    # compact dense geometry as an explicit B=N=1 input.  Lock both halves of
    # the measured two-node recipe so a future adapter change cannot silently
    # make the full-C16 K1 selector unreachable again.
    route_contracts = (
        "const bool automatic_short_single_direct = p.N == 1 && "
        "(T_seq == 256 || T_seq == 512)",
        "if (direct) { group_chunks = 0;",
        "const dim3 grid = a.is_varlen ? dim3(a.total_tiles, a.H) : "
        "dim3(a.NT, a.N * a.H);",
        "const bool automatic_short_single_nw1_flat = a.N == 1 && "
        "(a.T_seq == 256 || a.T_seq == 512)",
        "const uint64_t direct_flat_blocks_per_sequence = a.H > 0 ? "
        "uint64_t(a.H) * uint64_t(8) : uint64_t(0);",
        "<<<dim3(direct_flat_blocks, 1, 1), 64, 0, a.stream>>>",
        "if (dense_all_full_c16_enabled && a.N == 1 && "
        "(a.T_seq & 15) == 0) "
        "dispatch_flat.template operator()<false, true>();",
    )
    for contract in route_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "packed-N1 dense K1/K2 route lost contract: " + contract
            )

    def normalize(tokens: int, *, packed: bool, sequences: int) -> tuple[bool, int]:
        single_sequence_packed = packed and sequences == 1
        total_tiles = (
            (tokens + 15) // 16
            if single_sequence_packed
            else ((tokens + 15) // 16 + sequences if packed else 0)
        )
        return packed and not single_sequence_packed, total_tiles

    for tokens, expected_tiles in (
        (256, 16),
        (512, 32),
        (1024, 64),
        (2048, 128),
    ):
        normalized_varlen, total_tiles = normalize(
            tokens, packed=True, sequences=1
        )
        if normalized_varlen or total_tiles != expected_tiles:
            raise AssertionError(
                f"packed N=1 T={tokens} did not normalize to dense "
                f"total_tiles={expected_tiles}"
            )
    if normalize(512, packed=True, sequences=2) != (True, 34):
        raise AssertionError("multi-sequence packed fallback was normalized")

    print(
        "PASS static descriptor/raw-v3 packed-N1 dense normalization, compact "
        "K1 grid, prefix-free two-node route, and dense NW1-flat K2 contract"
    )


def check_bt16_dense_n1_all_full_c16_policy_static() -> None:
    """Audit the strict dense-N1 full-C16 K1 template selector."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/k1_kda_bt16_fused_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())

    selector_start = policy.find(
        "static bool bt16_dense_n1_all_full_c16_enabled("
    )
    selector_end = policy.find(
        "static bool bt16_fused_opt_enabled()", selector_start
    )
    if selector_start < 0 or selector_end < 0:
        raise AssertionError("policy is missing the dense-N1 full-C16 selector")
    selector = " ".join(policy[selector_start:selector_end].split())
    selector_contracts = (
        'env_exact( "FLASH_KDA_GFX950_BT16_DENSE_N1_ALL_FULL_C16", "1")',
        "cache_context_operands && a.cache_context_operands",
        "!a.is_varlen && a.N == 1",
        "(a.T_seq == 256 || a.T_seq == 512 || "
        "a.T_seq == 1024 || a.T_seq == 2048)",
        "a.T_seq % WorkspaceSizes::CHUNK == 0",
        "a.NT == a.T_seq / WorkspaceSizes::CHUNK",
        "a.total_tiles == a.NT",
        "fused == Bt16FusedMode::vector_x32",
        "bt16_fused_opt_enabled()",
    )
    for contract in selector_contracts:
        if contract not in selector:
            raise AssertionError(
                "dense-N1 full-C16 selector lost contract: " + contract
            )

    launch_contracts = (
        "bool PACKED_DIRECT_PREFIXLESS, "
        "bool DENSE_N1_ALL_FULL_C16, bool GVA = false>",
        "PACKED_DIRECT_PREFIXLESS, DENSE_N1_ALL_FULL_C16, "
        "PADDED_SOLVE, EARLY_DENSE_BETA, GVA>",
        "if constexpr (DENSE_N1_ALL_FULL_C16) { "
        "const bool padded_solve = "
        "bt16_dense_n1_padded_solve_enabled(); "
        "auto launch_dense = [&]<bool EARLY_DENSE_BETA>() { "
        "if (padded_solve) launch.template operator()< "
        "true, true, EARLY_DENSE_BETA>(); else "
        "launch.template operator()< "
        "true, false, EARLY_DENSE_BETA>(); }; "
        "if (bt16_dense_n1_early_beta_enabled()) "
        "launch_dense.template operator()<true>(); else "
        "launch_dense.template operator()<false>();",
        "if constexpr (DENSE_N1_ALL_FULL_C16) launch_bt16_fused< "
        "VL, false, true, CACHE_CONTEXT_OPERANDS, "
        "PUBLISH_ACTIVATED_BETA, PACKED_DIRECT_PREFIXLESS, true>",
        "dispatch.template operator()<true, true, false>();",
        "dispatch.template operator()<true, false, false>();",
        "else if (dense_n1_all_full_c16) { "
        "dispatch.template operator()<false, false, true>();",
        "dispatch.template operator()<false, false, false>();",
    )
    for contract in launch_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "dense-N1 full-C16 dispatch lost contract: " + contract
            )

    compile_contracts = (
        "static_assert(!DENSE_N1_ALL_FULL_C16 || !VL",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || !EXACT_PREP",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || USE_X32",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || "
        "CACHE_CONTEXT_OPERANDS",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || "
        "PUBLISH_ACTIVATED_BETA",
    )
    for contract in compile_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "dense-N1 full-C16 compile proof lost contract: " + contract
            )

    kernel_contracts = (
        "bool PACKED_DIRECT_PREFIXLESS = false, "
        "bool DENSE_N1_ALL_FULL_C16 = false, "
        "bool PADDED_SOLVE = false, "
        "bool EARLY_DENSE_BETA = false, "
        "bool GVA = false>",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || !VL",
        "static_assert(!DENSE_N1_ALL_FULL_C16 || "
        "!PACKED_DIRECT_PREFIXLESS",
        "if constexpr (DENSE_N1_ALL_FULL_C16) { h = bh; "
        "ht = h * gridDim.x + nt; t0 = nt * C; alen = C;",
        "if (DENSE_N1_ALL_FULL_C16 || vec_m < alen)",
        "(DENSE_N1_ALL_FULL_C16 || row_lane < alen)",
        "(DENSE_N1_ALL_FULL_C16 || tid < alen)",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                "dense-N1 full-C16 kernel lost contract: " + contract
            )

    def selected(
        value: str | None,
        *,
        cached: bool = True,
        cache_requested: bool = True,
        varlen: bool = False,
        sequences: int = 1,
        tokens: int = 256,
        chunks: int = 16,
        total_tiles: int = 16,
        fused: str = "vector_x32",
        fused_opt: bool = True,
    ) -> bool:
        return (
            value == "1"
            and cached
            and cache_requested
            and not varlen
            and sequences == 1
            and tokens in (256, 512)
            and tokens % 16 == 0
            and chunks == tokens // 16
            and total_tiles == chunks
            and fused == "vector_x32"
            and fused_opt
        )

    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected(spelling):
            raise AssertionError(
                f"dense-N1 full-C16 parser accepted fallback {spelling!r}"
            )
    for tokens, chunks in ((256, 16), (512, 32)):
        if not selected("1", tokens=tokens, chunks=chunks, total_tiles=chunks):
            raise AssertionError(
                f"dense-N1 full-C16 selector rejected dense T={tokens}"
            )
    mismatches = (
        {"cached": False},
        {"cache_requested": False},
        {"varlen": True},
        {"sequences": 2},
        {"tokens": 255, "chunks": 15, "total_tiles": 15},
        {"tokens": 1024, "chunks": 64, "total_tiles": 64},
        {"chunks": 15},
        {"total_tiles": 15},
        {"fused": "exact_x16"},
        {"fused": "exact_x32"},
        {"fused_opt": False},
        # The packed-resume equal-dense rewrite presents dense K1 geometry,
        # but its four logical sequences must retain template bit ten false.
        {
            "sequences": 4,
            "tokens": 4096,
            "chunks": 256,
            "total_tiles": 1024,
        },
    )
    for mismatch in mismatches:
        if selected("1", **mismatch):
            raise AssertionError(
                f"dense-N1 full-C16 selector accepted fallback {mismatch}"
            )

    print(
        "PASS static strict dense-N1 full-C16 K1 selector, tenth-template-bit "
        "rollback with eleventh/twelfth bits false, vector-x32 production "
        "guard, and equal-dense N4 exclusion"
    )


def check_bt16_dense_n1_padded_solve_policy_static() -> None:
    """Audit the strict dense-N1 padded-solve LDS specialization."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/k1_kda_bt16_fused_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())

    parser_start = policy.find(
        "static bool bt16_dense_n1_padded_solve_enabled()"
    )
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the padded-solve parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    parser_contracts = (
        'env_exact( "FLASH_KDA_GFX950_BT16_DENSE_N1_PADDED_SOLVE", "1")',
    )
    for contract in parser_contracts:
        if contract not in parser:
            raise AssertionError(
                "padded-solve parser lost contract: " + contract
            )

    policy_contracts = (
        "if constexpr (DENSE_N1_ALL_FULL_C16) { "
        "const bool padded_solve = "
        "bt16_dense_n1_padded_solve_enabled(); "
        "auto launch_dense = [&]<bool EARLY_DENSE_BETA>() { "
        "if (padded_solve) launch.template operator()< "
        "true, true, EARLY_DENSE_BETA>(); else "
        "launch.template operator()< "
        "true, false, EARLY_DENSE_BETA>(); }; "
        "if (bt16_dense_n1_early_beta_enabled()) "
        "launch_dense.template operator()<true>(); else "
        "launch_dense.template operator()<false>(); } else { "
        "const bool opt = bt16_fused_opt_enabled(); if (opt) "
        "launch.template operator()<true, false, false>(); else "
        "launch.template operator()<false, false, false>(); }",
        "static_assert(!PADDED_SOLVE || DENSE_N1_ALL_FULL_C16",
        "static_assert(!PADDED_SOLVE || USE_X32",
        "static_assert(!PADDED_SOLVE || OPT",
        "static_assert(!PADDED_SOLVE || CACHE_CONTEXT_OPERANDS",
        "static_assert(!PADDED_SOLVE || PUBLISH_ACTIVATED_BETA",
        "PACKED_DIRECT_PREFIXLESS, DENSE_N1_ALL_FULL_C16, PADDED_SOLVE, "
        "EARLY_DENSE_BETA, GVA>",
    )
    for contract in policy_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "padded-solve dispatch lost contract: " + contract
            )

    kernel_contracts = (
        "bool DENSE_N1_ALL_FULL_C16 = false, bool PADDED_SOLVE = false, "
        "bool EARLY_DENSE_BETA = false, bool GVA = false>",
        "static_assert(!PADDED_SOLVE || DENSE_N1_ALL_FULL_C16",
        "static_assert(!PADDED_SOLVE || USE_X32",
        "constexpr int SD = PADDED_SOLVE ? D + 4 : D;",
        "__bf16 kd[C * SD]; __bf16 qd[C * SD]; __bf16 ki[C * SD];",
        "sizeof(SolveStorage<D>) == 14400",
        "sizeof(SolveStorage<D + 4>) == 14784",
        "sizeof(SharedStorage<false, D + 4>) == "
        "sizeof(SolveStorage<D + 4>)",
        "sizeof(SharedStorage<true, D + 4>) == "
        "sizeof(ExactPrepStorage)",
        "const int64_t ws_vec_off = int64_t(ht) * TILE_ELEMS + vec_idx;",
        "const int solve_vec_idx = vec_m * SD + vec_d0;",
        "solve.kd + solve_vec_idx",
        "solve.qd + solve_vec_idx",
        "solve.ki + solve_vec_idx",
        "contract_last_x32<D, SD, SD>(solve.qd, solve.ki, lane)",
        "contract_last_x32<D, SD, SD>(solve.kd, solve.ki, lane)",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                "padded-solve kernel lost contract: " + contract
            )
    if compact_kernel.count(
        "contract_last_x32<D, SD, SD>(solve.qd, solve.ki, lane)"
    ) != 2:
        raise AssertionError(
            "padded-solve kernel must have exactly two Qd@Ki X32 call sites"
        )

    def selected(
        value: str | None,
        *,
        full_c16: bool = True,
        use_x32: bool = True,
        fused_opt: bool = True,
        cached: bool = True,
        publish_beta: bool = True,
    ) -> bool:
        return (
            value == "1"
            and full_c16
            and use_x32
            and fused_opt
            and cached
            and publish_beta
        )

    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected(spelling):
            raise AssertionError(
                f"padded-solve parser accepted fallback {spelling!r}"
            )
    if not selected("1"):
        raise AssertionError("padded-solve selector rejected canonical '1'")
    for mismatch in (
        {"full_c16": False},
        {"use_x32": False},
        {"fused_opt": False},
        {"cached": False},
        {"publish_beta": False},
    ):
        if selected("1", **mismatch):
            raise AssertionError(
                f"padded-solve selector accepted fallback {mismatch}"
            )

    print(
        "PASS static strict dense-N1 padded-solve exact opt-in, "
        "eleventh-template-bit rollback with twelfth bit false, compact "
        "workspace ABI, and 14784-byte vector-prep LDS contract"
    )


def check_bt16_dense_n1_early_beta_policy_static() -> None:
    """Audit the strict dense-N1 early-beta scheduling specialization."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/k1_kda_bt16_fused_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())

    parser_start = policy.find(
        "static bool bt16_dense_n1_early_beta_enabled()"
    )
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the early-beta parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    parser_contract = (
        'env_exact( "FLASH_KDA_GFX950_BT16_DENSE_N1_EARLY_BETA", "1")'
    )
    if parser_contract not in parser:
        raise AssertionError(
            "early-beta parser lost exact opt-in contract: "
            + parser_contract
        )

    policy_contracts = (
        "auto launch = [&]<bool OPT, bool PADDED_SOLVE, "
        "bool EARLY_DENSE_BETA>()",
        "static_assert(!EARLY_DENSE_BETA || "
        "(DENSE_N1_ALL_FULL_C16 && !EXACT_PREP && "
        "USE_X32 && OPT && CACHE_CONTEXT_OPERANDS && "
        "PUBLISH_ACTIVATED_BETA)",
        "PACKED_DIRECT_PREFIXLESS, DENSE_N1_ALL_FULL_C16, "
        "PADDED_SOLVE, EARLY_DENSE_BETA, GVA>",
        "if (bt16_dense_n1_early_beta_enabled()) "
        "launch_dense.template operator()<true>(); else "
        "launch_dense.template operator()<false>();",
        "if (opt) launch.template operator()<true, false, false>(); "
        "else launch.template operator()<false, false, false>();",
    )
    for contract in policy_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "early-beta dispatch lost contract: " + contract
            )

    kernel_contracts = (
        "bool PADDED_SOLVE = false, bool EARLY_DENSE_BETA = false, "
        "bool GVA = false>",
        "static_assert(!EARLY_DENSE_BETA || "
        "(!EXACT_PREP && DENSE_N1_ALL_FULL_C16)",
        "float balanced_beta = 0.0f; "
        "if constexpr (EXACT_PREP || EARLY_DENSE_BETA)",
        "if (tid >= 3 * 64 && tid < 3 * 64 + C)",
        "solve.beta[row_lane] = balanced_beta; "
        "if constexpr (PUBLISH_ACTIVATED_BETA) "
        "beta_cache[int64_t(ht) * C + row_lane] = balanced_beta;",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                "early-beta kernel lost contract: " + contract
            )
    if compact_kernel.count(
        "if (tid >= 3 * 64 && tid < 3 * 64 + C)"
    ) != 2:
        raise AssertionError(
            "early-beta kernel must have one wave-3 compute site and one "
            "wave-3 publication site"
        )

    def selected(
        value: str | None,
        *,
        full_c16: bool = True,
        exact_prep: bool = False,
        use_x32: bool = True,
        fused_opt: bool = True,
        cached: bool = True,
        publish_beta: bool = True,
    ) -> bool:
        return (
            value == "1"
            and full_c16
            and not exact_prep
            and use_x32
            and fused_opt
            and cached
            and publish_beta
        )

    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected(spelling):
            raise AssertionError(
                f"early-beta parser accepted fallback {spelling!r}"
            )
    if not selected("1"):
        raise AssertionError("early-beta selector rejected canonical '1'")
    for mismatch in (
        {"full_c16": False},
        {"exact_prep": True},
        {"use_x32": False},
        {"fused_opt": False},
        {"cached": False},
        {"publish_beta": False},
    ):
        if selected("1", **mismatch):
            raise AssertionError(
                f"early-beta selector accepted fallback {mismatch}"
            )

    print(
        "PASS static strict dense-N1 early-beta exact opt-in, "
        "twelfth-template-bit rollback, wave-3 register hoist, and "
        "unchanged beta publication point"
    )


def check_context_forward_policy_static() -> None:
    """Audit production-default U/V parsing without requiring a GPU."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    source = policy_path.read_text()
    default_on_return = (
        "return value == nullptr || "
        "!(value[0] == '0' && value[1] == '\\0');"
    )
    parsers = {
        "context_u_forward_enabled": (
            "FLASH_KDA_GFX950_CONTEXT_U_FORWARD"
        ),
        "context_v_forward_enabled": (
            "FLASH_KDA_GFX950_CONTEXT_V_FORWARD"
        ),
    }
    for function, environment in parsers.items():
        start = source.find(f"static bool {function}()")
        end = source.find("\n    }", start)
        if start < 0 or end < 0:
            raise AssertionError(f"policy is missing {function}")
        body = " ".join(source[start:end].split())
        if f'std::getenv("{environment}")' not in body:
            raise AssertionError(
                f"{function} does not read the expected environment"
            )
        if default_on_return not in body:
            raise AssertionError(
                f"{function} is not default-on/exact-'0'-off"
            )

    def default_on(value: str | None) -> bool:
        return value != "0"

    for enabled in (None, "", "1", "01", "true", "1 ", " 0", "00"):
        if not default_on(enabled):
            raise AssertionError(
                f"static default-on model rejected {enabled!r}"
            )
    if default_on("0"):
        raise AssertionError("static exact-'0' rollback model stayed enabled")
    print("PASS static context U/V default-on, exact-'0'-off parsing")


def check_context_lds_pipeline_pass_policy_static() -> None:
    """Audit strict parsing and per-mode template wiring without a GPU."""

    policy_path = (
        _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    )
    source = policy_path.read_text()
    compact = " ".join(source.split())

    pass_parsers = {
        "context_lds_pipeline_b_enabled": (
            "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_B"
        ),
        "context_lds_pipeline_a_enabled": (
            "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_A"
        ),
        "context_lds_pipeline_replay_enabled": (
            "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_REPLAY"
        ),
    }
    exact_return = (
        "return value != nullptr && value[0] == '1' && "
        "value[1] == '\\0';"
    )
    for function, environment in pass_parsers.items():
        start = source.find(f"static bool {function}()")
        if start < 0:
            raise AssertionError(f"policy is missing {function}")
        end = source.find("\n    }", start)
        if end < 0:
            raise AssertionError(f"cannot delimit policy parser {function}")
        body = " ".join(source[start:end].split())
        if f'std::getenv("{environment}")' not in body:
            raise AssertionError(
                f"{function} does not read the expected environment"
            )
        if exact_return not in body:
            raise AssertionError(f"{function} is not an exact-'1' parser")

    effective_wiring = (
        "const bool pipeline_lds_b = forward_u && forward_v && "
        "(pipeline_lds_global || context_lds_pipeline_b_enabled());",
        "const bool pipeline_lds_a = forward_u && forward_v && "
        "(pipeline_lds_global || context_lds_pipeline_a_enabled());",
        "const bool pipeline_lds_replay = forward_u && forward_v && "
        "(pipeline_lds_global || context_lds_pipeline_replay_enabled());",
    )
    for statement in effective_wiring:
        if statement not in compact:
            raise AssertionError(
                f"policy is missing guarded per-pass wiring: {statement}"
            )

    # KdaContextMode is already part of the kernel template, so each call must
    # consume only its own final LDS_PIPELINE bit.  This reuses the existing
    # true/false specializations instead of introducing a pass-mask template.
    mode_wiring = (
        "KdaContextMode::kAffineB, false, false, VL, false, CONTEXT_NW, 0, "
        "CACHED_OPERANDS, U_FORWARD, V_FORWARD, LDS_PIPELINE_B>",
        "KdaContextMode::kAffineA, false, false, VL, false, CONTEXT_NW, 0, "
        "CACHED_OPERANDS, U_FORWARD, V_FORWARD, LDS_PIPELINE_A>",
        "KdaContextMode::kReplay, HO, FP, VL, false, CONTEXT_NW, 0, "
        "CACHED_OPERANDS, U_FORWARD, V_FORWARD, LDS_PIPELINE_REPLAY>",
        "kHybridDirectMaxChunks, CACHED_OPERANDS, U_FORWARD, V_FORWARD, "
        "LDS_PIPELINE_REPLAY>",
    )
    for specialization in mode_wiring:
        if specialization not in compact:
            raise AssertionError(
                "policy per-mode LDS template wiring changed: "
                f"{specialization}"
            )
    launch_start = compact.find("static void launch_context_parallel")
    direct_start = compact.find("if (direct) {", launch_start)
    direct_end = compact.find("return;", direct_start)
    direct_dispatch = compact[direct_start:direct_end]
    if "pipeline_lds_replay" not in direct_dispatch or any(
        name in direct_dispatch for name in ("pipeline_lds_b", "pipeline_lds_a")
    ):
        raise AssertionError("direct context dispatch is not replay-only")

    def exact_one(value: str | None) -> bool:
        return value == "1"

    for fallback in (None, "", "0", "01", "true", "1 ", " 1"):
        if exact_one(fallback):
            raise AssertionError(
                f"static exact-'1' model accepted fallback {fallback!r}"
            )
    for u_forward in (False, True):
        for v_forward in (False, True):
            for global_pipeline in (False, True):
                for mask in range(8):
                    requested = tuple(
                        bool(mask & bit) for bit in (4, 2, 1)
                    )
                    effective = tuple(
                        u_forward
                        and v_forward
                        and (global_pipeline or selected)
                        for selected in requested
                    )
                    expected = (
                        (True, True, True)
                        if u_forward and v_forward and global_pipeline
                        else requested
                        if u_forward and v_forward
                        else (False, False, False)
                    )
                    if effective != expected:
                        raise AssertionError(
                            "per-pass LDS policy truth-table mismatch for "
                            f"U={u_forward}, V={v_forward}, "
                            f"global={global_pipeline}, mask={mask:03b}"
                        )
    print(
        "PASS static context LDS per-pass exact parser, U/V guard, "
        "and mode-template truth table"
    )


def check_context_nw8_policy_static() -> None:
    """Audit NW8's exact parser, prerequisites, and low-half staging."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/"
        "k2_kda_context_parallel_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())

    parser_start = policy.find("static bool context_nw8_enabled()")
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the context NW8 parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    if 'std::getenv("FLASH_KDA_GFX950_CONTEXT_NW8")' not in parser:
        raise AssertionError("context NW8 parser reads the wrong environment")
    exact_return = (
        "return value != nullptr && value[0] == '1' && "
        "value[1] == '\\0';"
    )
    if exact_return not in parser:
        raise AssertionError("context NW8 parser is not exact-'1'")

    prerequisite = (
        "const bool context_nw8 = context_nw8_enabled() && "
        "cache_context_operands && forward_u && forward_v;"
    )
    if prerequisite not in compact_policy:
        raise AssertionError("context NW8 dispatch lost its cache/U/V guard")
    policy_contracts = (
        "direct_nw == 8 && cache_context_operands && forward_u && forward_v",
        "launch_recurrence.template operator()<8>();",
        "launch_recurrence.template operator()<4>();",
        "8 / CONTEXT_NW",
        "CONTEXT_NW * 64",
        "dim3(a.N * a.H, 8 / NW)",
        "dim3(a.N * a.H, 8 / CONTEXT_NW)",
        "dim3(a.N * a.H, 2)",
    )
    for contract in policy_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                f"context NW8 dispatch is missing contract: {contract}"
            )
    if compact_policy.count("<<<dim3(a.N * a.H,") != 3:
        raise AssertionError(
            "all three direct replay launches must use the 2D "
            "[sequence*head,V-group] grid"
        )

    kernel_contracts = (
        "constexpr int RW = NW <= 4 ? ROW_VECS / NTHREADS : 1;",
        "NW == 1 || NW == 2 || NW == 4 || NW == 8",
        "NW != 8 || ROW_VECS * 2 == NTHREADS",
        "NW != 8 || (CACHED_OPERANDS && U_FORWARD && V_FORWARD)",
        "__syncthreads(); if (has_next) { commit(); __syncthreads(); }",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                f"context NW8 kernel is missing contract: {contract}"
            )
    legacy_grid_decode = (
        "const int global_context = int(blockIdx.x) / H;",
        "const int h = int(blockIdx.x) - global_context * H;",
        "const int v0 = (int(blockIdx.y) * NW + wave) * BV;",
    )
    flat_nw1_isolated_decode = (
        "if constexpr (DIRECT && DIRECT_TAIL_FIRST && NW == 1)",
        "global_context = int(blockIdx.x) / H;",
        "h = int(blockIdx.x) - global_context * H;",
        "v_group = int(blockIdx.y);",
        "const int v0 = (v_group * NW + wave) * BV;",
    )
    if not (
        all(contract in compact_kernel for contract in legacy_grid_decode)
        or all(
            contract in compact_kernel
            for contract in flat_nw1_isolated_decode
        )
    ):
        raise AssertionError(
            "context NW8 kernel lost its established 2D grid decode or "
            "failed to isolate the alternate decode to flat NW1"
        )
    if "blockIdx.z" in kernel:
        raise AssertionError(
            "context replay must not reintroduce the regressing 3D grid ID"
        )
    if kernel.count("if constexpr (NW <= 4)") != 3:
        raise AssertionError(
            "context NW1/2/4 stage/commit paths are not isolated three times"
        )
    if kernel.count("if (tid < ROW_VECS)") != 3:
        raise AssertionError(
            "context NW8 does not guard all three common publications"
        )

    def exact_one(value: str | None) -> bool:
        return value == "1"

    for fallback in (None, "", "0", "01", "true", "1 ", " 1"):
        if exact_one(fallback):
            raise AssertionError(
                f"static NW8 exact parser accepted fallback {fallback!r}"
            )
    if not exact_one("1"):
        raise AssertionError("static NW8 exact parser rejected canonical '1'")
    print(
        "PASS static context NW8 exact parser, cache/U/V fallback, "
        "direct 2D grid/unconditional tail barrier, and 256-of-512 common "
        "staging"
    )


def _check_context_direct_prefixless_policy_static(
    policy: str, context_kernel: str
) -> None:
    """Audit the exact mixed-boundary prefixless/NW1-flat recipe."""

    common_path = (
        _REPO_ROOT / "csrc/kernels/flash_kda/hip_launch_common.cu"
    )
    k1_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/k1_kda_bt16_fused_kernel.hpp"
    )
    common = common_path.read_text()
    k1_kernel = k1_path.read_text()
    compact_common = " ".join(common.split())
    compact_k1 = " ".join(k1_kernel.split())
    compact_context = " ".join(context_kernel.split())

    def cpp_block(source: str, signature: str) -> str:
        start = source.find(signature)
        opening = source.find("{", start)
        if start < 0 or opening < 0:
            raise AssertionError(f"cannot find C++ block {signature!r}")
        depth = 0
        for position in range(opening, len(source)):
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
                if depth == 0:
                    return " ".join(source[start : position + 1].split())
        raise AssertionError(f"cannot delimit C++ block {signature!r}")

    prefixless_guard = cpp_block(
        policy, "static bool context_direct_prefixless_enabled("
    )
    guard_contracts = (
        f'std::getenv( "{_CONTEXT_DIRECT_PREFIXLESS_ENV}")',
        "const bool explicit_prefixless = value != nullptr && "
        "value[0] == '1' && value[1] == '\\0';",
        "const bool automatic_mixed_boundary_prefixless = "
        "value == nullptr && p.cu_seqlens != nullptr && p.N == 16",
        "(p.max_seqlen_upper_bound == 1024 || "
        "p.max_seqlen_upper_bound == 1025)",
        "p.T_total == p.max_seqlen_upper_bound + 15",
        "p.total_tiles == 81",
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr',
        f'std::getenv( "{_CONTEXT_DIRECT_NW1_FLAT_ENV}") == nullptr',
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW") == nullptr',
        "if (!explicit_prefixless && "
        "!automatic_mixed_boundary_prefixless) return false;",
        "default_k2_route == K2DefaultRoute::context_parallel",
        "route.group_chunks == 0 && route.direct_max_chunks == 0",
        'std::getenv("FLASH_KDA_K2") == nullptr',
        "!bt16_k1_disabled()",
        "bt16_fused_mode() != Bt16FusedMode::disabled",
    )
    for contract in guard_contracts:
        if contract not in prefixless_guard:
            raise AssertionError(
                "mixed-boundary prefixless guard changed: " + contract
            )
    for route in ("DIRECT", "AFFINE", "HYBRID"):
        contract = (
            f'std::getenv("FLASH_KDA_GFX950_CONTEXT_{route}") == nullptr'
        )
        if contract not in prefixless_guard:
            raise AssertionError(
                "automatic prefixless host guard no longer rolls back every "
                f"explicit {route} value"
            )

    context_launch = cpp_block(
        policy,
        "static void launch_context_parallel(const ContextParallelLaunch& a)",
    )
    flat_guard_start = context_launch.find(
        "const bool automatic_mixed_boundary_nw1_flat ="
    )
    flat_guard_end = context_launch.find(";", flat_guard_start)
    if flat_guard_start < 0 or flat_guard_end < 0:
        raise AssertionError(
            "policy is missing the mixed-boundary NW1-flat launch guard"
        )
    flat_guard = context_launch[flat_guard_start : flat_guard_end + 1]
    flat_guard_contracts = (
        "const bool automatic_mixed_boundary_nw1_flat = "
        "a.packed_direct_prefixless && a.is_varlen && a.N == 16",
        "a.total_tiles == 81 && "
        "(a.T_seq == 64 || a.T_seq == 65)",
        f'std::getenv( "{_CONTEXT_DIRECT_PREFIXLESS_ENV}") == nullptr',
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr',
        "direct_nw_value == nullptr && nw1_flat_value == nullptr",
    )
    for contract in flat_guard_contracts:
        if contract not in flat_guard:
            raise AssertionError(
                "mixed-boundary NW1-flat guard changed: " + contract
            )
    for route in ("DIRECT", "AFFINE", "HYBRID"):
        contract = (
            f'std::getenv("FLASH_KDA_GFX950_CONTEXT_{route}") == nullptr'
        )
        if contract not in flat_guard:
            raise AssertionError(
                "automatic NW1-flat launch guard no longer rolls back every "
                f"explicit {route} value"
            )

    flat_dispatch_contracts = (
        "const char* nw1_flat_value = std::getenv( "
        f'"{_CONTEXT_DIRECT_NW1_FLAT_ENV}");',
        "requested_nw1_flat_tail_first || "
        "automatic_short_single_nw1_flat || "
        "automatic_mixed_boundary_nw1_flat",
        "use_nw1_flat_tail_first && !use_deep_n4_nw4",
        "if (a.packed_direct_prefixless) "
        "dispatch_flat.template operator()<true, false>();",
        "VL && !PACKED_DIRECT_PREFIXLESS ? a.tile_prefix : nullptr",
    )
    for contract in flat_dispatch_contracts:
        if contract not in context_launch:
            raise AssertionError(
                "mixed-boundary NW1-flat dispatch changed: " + contract
            )

    k1_launch = cpp_block(policy, "static void launch_bt16_k1(")
    k1_contracts = (
        "if (a.packed_direct_prefixless) "
        "dispatch.template operator()<true, true, false>();",
        "dispatch.template operator()<true, false, false>();",
    )
    for contract in k1_contracts:
        if contract not in k1_launch:
            raise AssertionError(
                "fused K1 lost matched prefixless specialization dispatch: "
                + contract
            )

    common_contracts = (
        "use_context_parallel && !is_gva && is_varlen && "
        "!use_context_equal_dense_n4_g64 && "
        "policy.context_direct_prefixless",
        "policy.context_group_chunks == 0 && "
        "policy.context_direct_max_chunks == 0",
        "if (is_varlen && !use_context_direct_prefixless && "
        "!use_context_equal_dense_n4_g64)",
        "use_context_direct_prefixless ? nullptr : "
        "(k1_is_varlen ? tile_prefix : nullptr)",
        "use_context_direct_prefixless ? nullptr : "
        "(context_is_varlen ? tile_prefix : nullptr)",
        "use_context_direct_prefixless, "
        "use_context_equal_dense_n4_g64, context_operands_cached, is_gva, "
        "policy.context_automatic_gva_packed_nw4, "
        "policy.context_automatic_gva_equal_n4_g16};",
    )
    for contract in common_contracts:
        if contract not in compact_common:
            raise AssertionError(
                "common launcher lost matched prefixless graph wiring: "
                + contract
            )

    k1_mapping_contracts = (
        "static_assert(!PACKED_DIRECT_PREFIXLESS || VL",
        "if constexpr (PACKED_DIRECT_PREFIXLESS)",
        "packed_c16_tile_mapping(cu_seqlens, N, gti)",
    )
    for contract in k1_mapping_contracts:
        if contract not in compact_k1:
            raise AssertionError(
                "prefixless fused K1 mapping changed: " + contract
            )
    context_mapping_contracts = (
        "static_assert(!PACKED_DIRECT_PREFIXLESS || "
        "(VL && DIRECT && DIRECT_MAX_CHUNKS == 0)",
        "if constexpr (PACKED_DIRECT_PREFIXLESS)",
        "packed_c16_sequence_mapping(cu_seqlens, seq)",
        "else if (alen == C)",
    )
    for contract in context_mapping_contracts:
        if contract not in compact_context:
            raise AssertionError(
                "prefixless context replay mapping changed: " + contract
            )

    def selected(
        seq_lens: tuple[int, ...],
        bound: int,
        total_tiles: int,
        *,
        prefixless_value: str | None = None,
        direct_nw_value: str | None = None,
        nw1_flat_value: str | None = None,
        route_values: tuple[str | None, str | None, str | None] = (
            None,
            None,
            None,
        ),
        pure_direct: bool = True,
    ) -> tuple[bool, bool]:
        automatic_prefixless = (
            prefixless_value is None
            and len(seq_lens) == 16
            and bound in (1024, 1025)
            and sum(seq_lens) == bound + 15
            and total_tiles == 81
            and direct_nw_value is None
            and nw1_flat_value is None
            and all(value is None for value in route_values)
            and pure_direct
        )
        prefixless = (
            (prefixless_value == "1" or automatic_prefixless)
            and 0 < len(seq_lens) <= 16
            and pure_direct
        )
        launch_t_seq = sum(seq_lens) // len(seq_lens)
        automatic_flat = (
            prefixless
            and prefixless_value is None
            and len(seq_lens) == 16
            and launch_t_seq in (64, 65)
            and total_tiles == 81
            and direct_nw_value is None
            and nw1_flat_value is None
            and all(value is None for value in route_values)
            and pure_direct
        )
        return prefixless, automatic_flat

    # These concrete fixtures share the admitted aggregate signature.  The
    # host guard sees only N/bound/total/tile-upper; general prefixless device
    # mappings, not admission, are responsible for the actual distribution.
    for bound in (1024, 1025):
        for prefill_first in (False, True):
            decodes = (1,) * 15
            seq_lens = (
                (bound,) + decodes
                if prefill_first
                else decodes + (bound,)
            )
            if selected(seq_lens, bound, 81) != (True, True):
                raise AssertionError(
                    "static mixed-boundary production recipe rejected "
                    f"bound={bound}, prefill_first={prefill_first}"
                )

    target = (1,) * 15 + (1025,)
    rollbacks = (
        {"prefixless_value": "0"},
        {"direct_nw_value": "1"},
        {"nw1_flat_value": "1"},
        {"route_values": ("0", None, None)},
        {"route_values": (None, "true", None)},
        {"route_values": (None, None, "01")},
    )
    for override in rollbacks:
        if selected(target, 1025, 81, **override) != (False, False):
            raise AssertionError(
                f"automatic mixed-boundary recipe ignored {override}"
            )
    if selected(target, 1025, 81, prefixless_value="1") != (True, False):
        raise AssertionError(
            "explicit PREFIXLESS must select mapping without auto NW1-flat"
        )

    ordinary = (65,) * 16
    if selected(ordinary, 65, 80) != (False, False):
        raise AssertionError("ordinary packed shape enabled prefixless by default")
    for value in ("", "0", "01", "true", "1 ", " 1"):
        if selected(ordinary, 65, 80, prefixless_value=value)[0]:
            raise AssertionError(
                f"ordinary prefixless parser accepted {value!r}"
            )
    if selected(ordinary, 65, 80, prefixless_value="1") != (True, False):
        raise AssertionError("ordinary pure-direct shape rejected exact opt-in")
    if selected(
        ordinary, 65, 80, prefixless_value="1", pure_direct=False
    )[0]:
        raise AssertionError("prefixless escaped its pure-direct route guard")


def check_context_direct_tail_first_policy_static() -> None:
    """Audit strict opt-in and pure-direct 2D sequence-slot rotation."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/"
        "k2_kda_context_parallel_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())

    parser_start = policy.find(
        "static bool context_direct_tail_first_enabled()"
    )
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the direct tail-first parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    if (
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_DIRECT_TAIL_FIRST")'
        not in parser
    ):
        raise AssertionError("direct tail-first parser reads the wrong env")
    exact_return = (
        "return value != nullptr && value[0] == '1' && "
        "value[1] == '\\0';"
    )
    if exact_return not in parser:
        raise AssertionError("direct tail-first parser is not exact-'1'")

    policy_contracts = (
        "const bool direct_tail_first = "
        "context_direct_tail_first_enabled();",
        "NW == 4 && CACHED_OPERANDS && U_FORWARD && V_FORWARD && "
        "!LDS_PIPELINE",
        "<<<dim3(a.N * a.H, 8 / NW), NW * 64, 0, a.stream>>>",
    )
    for contract in policy_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                f"direct tail-first dispatch is missing contract: {contract}"
            )
    legacy_launches = (
        "launch_kernel.template operator()<true>();",
        "launch_kernel.template operator()<false>();",
    )
    prefixless_aware_launches = (
        "launch_mapping.template operator()<true>();",
        "launch_mapping.template operator()<false>();",
        "launch_kernel.template operator()< DIRECT_TAIL_FIRST, true>();",
        "launch_kernel.template operator()< DIRECT_TAIL_FIRST, false>();",
    )
    if not (
        all(contract in compact_policy for contract in legacy_launches)
        or all(
            contract in compact_policy
            for contract in prefixless_aware_launches
        )
    ):
        raise AssertionError(
            "direct tail-first dispatch lost either its legacy launch pair "
            "or the matched prefixless-aware launch mapping"
        )

    kernel_contracts = (
        "bool DIRECT_TAIL_FIRST = false",
        "static_assert(!DIRECT_TAIL_FIRST || DIRECT",
        "seq = global_context == 0 ? N - 1 : global_context - 1;",
        "seq = launch_seq == 0 ? N - 1 : launch_seq - 1;",
        "local_group = global_context - launch_seq * groups_per_sequence;",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                f"direct tail-first kernel is missing contract: {contract}"
            )
    if "blockIdx.z" in kernel:
        raise AssertionError("direct tail-first must retain the 2D grid")

    def exact_one(value: str | None) -> bool:
        return value == "1"

    for fallback in (None, "", "0", "01", "true", "1 ", " 1"):
        if exact_one(fallback):
            raise AssertionError(
                "static direct tail-first parser accepted fallback "
                f"{fallback!r}"
            )
    if not exact_one("1"):
        raise AssertionError(
            "static direct tail-first parser rejected canonical '1'"
        )
    _check_context_direct_prefixless_policy_static(policy, kernel)
    print(
        "PASS static direct tail-first exact parser, guarded NW4/P0/cache/U/V "
        "dispatch, 2D rotation, and mixed-boundary prefixless/NW1-flat "
        "production/rollback contract"
    )


def _check_context_equal_dense_n4_g64_policy_static(
    policy: str, kernel: str
) -> None:
    """Audit the strict whole-graph packed-equal to dense-N4 candidate."""

    common = (
        _REPO_ROOT / "csrc/kernels/flash_kda/hip_launch_common.cu"
    ).read_text()
    common_abi = (
        _REPO_ROOT / "csrc/kernels/flash_kda/hip_common.hpp"
    ).read_text()
    compact_policy = " ".join(policy.split())
    compact_kernel = " ".join(kernel.split())
    compact_common = " ".join(common.split())
    compact_abi = " ".join(common_abi.split())

    def cpp_block(source: str, signature: str) -> str:
        start = source.find(signature)
        opening = source.find("{", start)
        if start < 0 or opening < 0:
            raise AssertionError(f"cannot find C++ block {signature!r}")
        depth = 0
        for position in range(opening, len(source)):
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
                if depth == 0:
                    return " ".join(source[start : position + 1].split())
        raise AssertionError(f"cannot delimit C++ block {signature!r}")

    capability = cpp_block(
        policy, "static bool context_equal_dense_n4_g64_enabled("
    )
    capability_contracts = (
        f'env_exact( "{_CONTEXT_EQUAL_DENSE_N4_G64_ENV}", "1")',
        "p.H_q != p.H || p.cu_seqlens == nullptr || "
        "p.N != kSequences || p.H <= 0",
        "p.T_total != kSequences * kSequenceTokens",
        "p.max_seqlen_upper_bound != kSequenceTokens",
        "p.total_tiles != kPackedTiles",
        "default_k2_route != K2DefaultRoute::context_parallel",
        "!route.force_context || route.group_chunks != 64",
        "route.direct_max_chunks != 0",
        'std::getenv("FLASH_KDA_K2") != nullptr',
        "bt16_k1_disabled()",
        "bt16_fused_mode() == Bt16FusedMode::disabled",
        '"FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE", "1"',
        '"FLASH_KDA_GFX950_CONTEXT_U_FORWARD", "1"',
        '"FLASH_KDA_GFX950_CONTEXT_V_FORWARD", "1"',
        f'"{_CONTEXT_AFFINE_AB_FUSED_ENV}", "1"',
        f'"{_CONTEXT_AFFINE_AB_STAGE_EARLY_ENV}"',
        '"FLASH_KDA_GFX950_CONTEXT_SCAN_NW", "2"',
        f'"{_CONTEXT_SCAN_KSPLIT_ENV}", "1"',
        "ksplit_selected",
    )
    for contract in capability_contracts:
        if contract not in capability:
            raise AssertionError(
                "equal-dense N4/G64 capability lost contract: " + contract
            )

    policy_wiring = (
        "const bool use_context_equal_dense_n4_g64 = "
        "context_equal_dense_n4_g64_enabled( p, context_route, "
        "default_k2_route);",
        "policy.context_equal_dense_n4_g64 = "
        "use_context_equal_dense_n4_g64;",
    )
    for contract in policy_wiring:
        if contract not in compact_policy:
            raise AssertionError(
                "equal-dense N4/G64 policy wiring changed: " + contract
            )

    abi_contracts = (
        "bool equal_dense_n4_g64 = false;",
        "bool context_equal_dense_n4_g64 = false;",
    )
    for contract in abi_contracts:
        if contract not in compact_abi:
            raise AssertionError(
                "equal-dense N4/G64 aggregate-tail ABI changed: " + contract
            )

    common_contracts = (
        "const bool use_context_equal_dense_n4_g64 = "
        "!is_gva && policy.context_equal_dense_n4_g64 && "
        "use_context_parallel && "
        "is_varlen && N == 4 && H > 0",
        "p.max_seqlen_upper_bound == 4096",
        "int64_t(p.max_seqlen_upper_bound) * int64_t(N) == "
        "int64_t(p.T_total)",
        "policy.context_group_chunks == 64",
        "equal_dense_total_tiles == int64_t(4 * 256)",
        "int64_t(total_tiles) == equal_dense_total_tiles + N",
        "!policy.context_direct_prefixless",
        "policy.launch_context_prefix == nullptr",
        "policy.context_persistent_blocks == 0",
        "if (is_varlen && !use_context_direct_prefixless && "
        "!use_context_equal_dense_n4_g64)",
        "const bool k1_is_varlen = is_varlen && "
        "!use_context_equal_dense_n4_g64;",
        "const int k1_total_tiles = use_context_equal_dense_n4_g64 ? "
        "int(equal_dense_total_tiles) : total_tiles;",
        "const int k1_nt = use_context_equal_dense_n4_g64 ? "
        "equal_dense_nt : NT;",
        "const bool context_is_varlen = is_varlen && "
        "!use_context_equal_dense_n4_g64;",
        "use_context_direct_prefixless, use_context_equal_dense_n4_g64, "
        "context_operands_cached, is_gva, "
        "policy.context_automatic_gva_packed_nw4, "
        "policy.context_automatic_gva_equal_n4_g16};",
    )
    for contract in common_contracts:
        if contract not in compact_common:
            raise AssertionError(
                "equal-dense N4/G64 common graph wiring changed: " + contract
            )

    launch_contracts = (
        "const bool equal_dense_n4_g64 = !a.is_gva && "
        "a.equal_dense_n4_g64 && "
        "!a.is_varlen && a.N == 4 && a.T_seq == 4096 && a.NT == 256 && "
        "a.total_tiles == 1024",
        "packed_automatic_equal_n4_g64 || equal_dense_n4_g64",
        "(!a.is_varlen && a.N == 1) || equal_dense_n4_g64",
        "k2_kda_context_affine_ab_fused_equal_n4_g64_nw4_kernel",
        "k2_kda_context_affine_ab_fused_equal_n4_g64_stage_early_nw4_kernel",
    )
    for contract in launch_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "equal-dense N4/G64 launch dispatch changed: " + contract
            )
    for symbol in launch_contracts[-2:]:
        if policy.count(symbol) != 1 or kernel.count(symbol + "(") != 1:
            raise AssertionError(
                f"equal-dense N4/G64 symbol is not independent: {symbol}"
            )

    kernel_contracts = (
        "bool EQUAL_DENSE_N4 = false, bool STAGE_EARLY = true",
        "!EQUAL_DENSE_N4 || (DENSE && GROUP_CHUNKS == 64)",
        "if constexpr (EQUAL_DENSE_N4)",
        "N != kSequences || T_seq != 4096 || NT != 256",
        "const int seq = global_context / kGroupsPerSequence;",
        "ht_base = (seq * H + h) * NT + first_chunk;",
        "token_base = seq * T_seq;",
        "if constexpr (STAGE_EARLY)",
        "if constexpr (!STAGE_EARLY)",
        "64, true, true, false",
        "64, true, true, true",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                "equal-dense N4/G64 kernel mapping changed: " + contract
            )

    def selected(
        value: str | None,
        *,
        sequences: int = 4,
        bound: int = 4096,
        tokens: int = 16384,
        total_tiles: int = 1028,
        group_chunks: int = 64,
        direct_max_chunks: int = 0,
    ) -> bool:
        return (
            value == "1"
            and sequences == 4
            and bound == 4096
            and tokens == sequences * bound
            and total_tiles == 1028
            and group_chunks == 64
            and direct_max_chunks == 0
        )

    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected(spelling):
            raise AssertionError(
                f"equal-dense parser accepted fallback {spelling!r}"
            )
    if not selected("1"):
        raise AssertionError("equal-dense parser rejected canonical '1'")
    for mismatch in (
        {"sequences": 3},
        {"bound": 4095},
        {"tokens": 16383},
        {"total_tiles": 1024},
        {"group_chunks": 32},
        {"direct_max_chunks": 64},
    ):
        if selected("1", **mismatch):
            raise AssertionError(
                f"equal-dense whole-graph guard accepted {mismatch}"
            )


def check_context_affine_ab_fused_policy_static() -> None:
    """Audit packed/dense-N1 fused B/A parsers and complete fallback guards."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    policy = policy_path.read_text()
    compact = " ".join(policy.split())
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/"
        "k2_kda_context_affine_ab_fused_kernel.hpp"
    )
    kernel = kernel_path.read_text()

    include = '#include "k2_kda_context_affine_ab_fused_kernel.hpp"'
    if include not in policy:
        raise AssertionError("policy does not include the fused affine header")
    parser_start = policy.find(
        "static bool context_affine_ab_fused_enabled(int group_chunks)"
    )
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the affine B/A fused parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    if (
        'std::getenv("FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED")'
        not in parser
    ):
        raise AssertionError("affine B/A fused parser reads the wrong env")
    explicit_return = (
        "if (value != nullptr) return value[0] == '1' && "
        "value[1] == '\\0';"
    )
    default_return = "return group_chunks == 8 || group_chunks == 16;"
    if explicit_return not in parser or default_return not in parser:
        raise AssertionError(
            "affine B/A fused parser lost its exact override or G8/G16 default"
        )

    stage_parser_start = policy.find(
        "static bool context_affine_ab_stage_early_enabled()"
    )
    stage_parser_end = policy.find("\n    }", stage_parser_start)
    if stage_parser_start < 0 or stage_parser_end < 0:
        raise AssertionError("policy is missing the affine B/A stage-early parser")
    stage_parser = " ".join(
        policy[stage_parser_start:stage_parser_end].split()
    )
    if (
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_STAGE_EARLY")'
        not in stage_parser
        or "return value != nullptr && value[0] == '1' && "
        "value[1] == '\\0';" not in stage_parser
    ):
        raise AssertionError(
            "affine B/A stage-early parser is not exact-'1'"
        )

    context_launch_start = compact.find(
        "static void launch_context_parallel(const ContextParallelLaunch& a)"
    )
    if context_launch_start < 0:
        raise AssertionError(
            "policy is missing the established context launcher"
        )
    automatic_guard_start = compact.find(
        "const bool packed_automatic_equal_n4_g64 =", context_launch_start
    )
    automatic_guard_end = compact.find(";", automatic_guard_start)
    if automatic_guard_start < 0 or automatic_guard_end < 0:
        raise AssertionError(
            "policy is missing the automatic packed N4/G64 guard"
        )
    automatic_n4_g64_guard = compact[
        automatic_guard_start : automatic_guard_end + 1
    ]
    automatic_guard_contracts = (
        "packed_automatic_equal_n4_g64 = "
        "!a.is_gva && a.is_varlen && a.N == 4 && a.T_seq == 4096",
        "group_chunks == 64 && direct_max_chunks == 0",
        '!env_exact("FLASH_KDA_GFX950_CONTEXT_DIRECT", "1")',
        '!env_exact("FLASH_KDA_GFX950_CONTEXT_AFFINE", "1")',
        '!env_exact("FLASH_KDA_GFX950_CONTEXT_HYBRID", "1")',
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr',
    )
    for contract in automatic_guard_contracts:
        if contract not in automatic_n4_g64_guard:
            raise AssertionError(
                "affine B/A fusion lost its exact packed N4/G64 guard: "
                f"{contract}"
            )
    effective_guard = (
        "const bool fuse_affine_ab = "
        "(context_affine_ab_fused_enabled(group_chunks) || "
        "(automatic_equal_n4_g64 && std::getenv( "
        '"FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED") == nullptr)) && '
        "!direct && (a.is_varlen || (!a.is_varlen && a.N == 1)) && "
        "!context_nw8 && "
        "cache_context_operands && forward_u && forward_v && "
        "!pipeline_lds_b && !pipeline_lds_a;"
    )
    effective_guard = effective_guard.replace(
        "(a.is_varlen || (!a.is_varlen && a.N == 1))",
        "(a.is_varlen || (!a.is_varlen && a.N == 1) || "
        "equal_dense_n4_g64)",
    )
    if effective_guard not in compact:
        raise AssertionError(
            "affine B/A fusion lost a route/layout/NW/cache/U/V/P0 guard"
        )
    stage_selector = (
        "const bool affine_ab_stage_early = fuse_affine_ab && "
        "context_affine_ab_stage_early_enabled() && "
        "(group_chunks == 64 || (!a.is_varlen && group_chunks == 16));"
    )
    if stage_selector not in compact:
        raise AssertionError(
            "affine B/A stage-early selector lost its fused/G64/dense-G16 guard"
        )
    scan_nw_contracts = (
        "const char* scan_nw_env = std::getenv( "
        '"FLASH_KDA_GFX950_CONTEXT_SCAN_NW");',
        "const bool automatic_gva_scan_nw4 = "
        "automatic_gva_packed_nw4 && scan_nw_env == nullptr",
        "const int scan_nw = scan_nw_env ? std::atoi(scan_nw_env) : "
        "automatic_gva_scan_nw4 ? 4 : 2;",
    )
    for contract in scan_nw_contracts:
        if contract not in compact:
            raise AssertionError(
                "affine scan lost its NW2/GVA-NW4 selection contract: "
                + contract
            )
    for experiment in (
        "SCAN_KSPLIT",
        "SCAN_A_GLL",
        "SCAN_B_PHASED",
    ):
        contract = (
            'std::getenv( "FLASH_KDA_GFX950_CONTEXT_'
            + experiment
            + '") == nullptr'
        )
        if contract not in compact:
            raise AssertionError(
                "automatic GVA NW4 did not yield to explicit scan experiment: "
                + experiment
            )
    compile_guards = (
        "CONTEXT_NW == 4 && CACHED_OPERANDS && U_FORWARD && "
        "V_FORWARD && !LDS_PIPELINE_B && !LDS_PIPELINE_A",
        "if (fuse_affine_ab)",
        "if constexpr (VL)",
    )
    for guard in compile_guards:
        if guard not in compact:
            raise AssertionError(
                f"affine B/A fused compile guard is missing: {guard}"
            )

    packed_symbol = "k2_kda_context_affine_ab_fused_nw4_kernel<"
    dense_symbol = "k2_kda_context_affine_ab_fused_dense_nw4_kernel<"
    if policy.count(packed_symbol) != 2 or policy.count(dense_symbol) != 1:
        raise AssertionError(
            "policy must contain one established packed launch, one nested "
            "persistent reuse, and one dense-N1 fused launch"
        )
    packed_stage_symbol = (
        "k2_kda_context_affine_ab_fused_stage_early_g64_nw4_kernel"
    )
    dense_stage_symbol = (
        "k2_kda_context_affine_ab_fused_dense_stage_early_nw4_kernel<"
    )
    if (
        policy.count(packed_stage_symbol) != 1
        or policy.count(dense_stage_symbol) != 1
    ):
        raise AssertionError(
            "policy must contain one packed-G64 and one dense stage-early launch"
        )
    if compact.count("auto launch_established_fused = [&]()") != 2:
        raise AssertionError(
            "stage-early dispatch lost a packed or dense established rollback"
        )
    if compact.count("if (affine_ab_stage_early)") != 4:
        raise AssertionError(
            "stage-early dispatch must guard packed, dense-N1 and exact-N4 "
            "candidate/rollback symbols"
        )
    persistent_start = policy.find(
        "static void launch_context_parallel_persistent("
    )
    established_start = policy.find(
        "static void launch_context_parallel(", persistent_start
    )
    if persistent_start < 0 or established_start < 0:
        raise AssertionError("cannot delimit context launch policy bodies")
    persistent_body = policy[persistent_start:established_start]
    established_body = policy[established_start:]
    if persistent_body.count(packed_symbol) != 1:
        raise AssertionError(
            "nested persistent graph must reuse exactly one packed fused launch"
        )
    if (
        established_body.count(packed_symbol) != 1
        or established_body.count(dense_symbol) != 1
    ):
        raise AssertionError(
            "established packed/dense fused dispatch multiplicity changed"
        )
    packed_launch_abi = (
        "a.v, beta, a.kd, a.kr, a.gt, a.inv, a.affine_b, a.affine_a, "
        "a.cu_seqlens, a.tile_prefix, a.context_prefix, a.N, "
        "a.total_tiles, a.H"
    )
    dense_launch_abi = (
        "a.v, beta, a.kd, a.kr, a.gt, a.inv, a.affine_b, a.affine_a, "
        "a.T_seq, a.H, a.NT"
    )
    for layout, launch_abi in (
        ("packed", packed_launch_abi),
        ("dense-N1", dense_launch_abi),
    ):
        if launch_abi not in compact:
            raise AssertionError(
                f"affine B/A fused {layout} launch ABI wiring changed"
            )

    stage_early_start = kernel.find(
        "// Strict-opt-in next-stage-early experiment."
    )
    dense_start = kernel.find(
        "k2_kda_context_affine_ab_fused_dense_nw4_kernel("
    )
    if dense_start < 0 or stage_early_start <= dense_start:
        raise AssertionError("dense-N1 fused kernel is missing")
    dense_body = kernel[dense_start:stage_early_start]
    if any(
        token in dense_body
        for token in ("cu_seqlens", "tile_prefix", "context_prefix")
    ):
        raise AssertionError("dense-N1 fused kernel depends on packed metadata")
    dense_compact = " ".join(dense_body.split())
    dense_groups = (
        "GROUP_CHUNKS == 8 || GROUP_CHUNKS == 16 || "
        "GROUP_CHUNKS == 32 || GROUP_CHUNKS == 64 || GROUP_CHUNKS == 128"
    )
    if dense_groups not in dense_compact:
        raise AssertionError(
            "dense-N1 fused kernel lost its G8/G16/G32/G64/G128 guard"
        )

    def source_function(name: str) -> str:
        start = kernel.find(name)
        brace = kernel.find("{", start)
        if start < 0 or brace < 0:
            raise AssertionError(f"cannot find source body for {name}")
        depth = 0
        for index in range(brace, len(kernel)):
            if kernel[index] == "{":
                depth += 1
            elif kernel[index] == "}":
                depth -= 1
                if depth == 0:
                    return kernel[start : index + 1] + "\n"
        raise AssertionError(f"unterminated source body for {name}")

    established_body_hashes = {
        "k2_kda_context_affine_ab_fused_nw4_kernel(": (
            "daef8c2205f5d0230b94ab500da4a0805b6ea9e92a87db50b51479869f526650"
        ),
        "k2_kda_context_affine_ab_fused_dense_nw4_kernel(": (
            "628cf148d15c109e7f91224cb598f43ceec2ddc8ae319fccf2f8ba81bf5b1d72"
        ),
    }
    for name, expected_hash in established_body_hashes.items():
        actual_hash = hashlib.sha256(source_function(name).encode()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(
                f"established affine B/A source body changed for {name}: "
                f"{actual_hash} != {expected_hash}"
            )

    candidate_body = kernel[stage_early_start:]
    candidate_contracts = (
        "k2_kda_context_affine_ab_fused_stage_early_g64_nw4_kernel(",
        "k2_kda_context_affine_ab_fused_dense_stage_early_nw4_kernel(",
        "(!DENSE && GROUP_CHUNKS == 64)",
        "DENSE && (GROUP_CHUNKS == 16 || GROUP_CHUNKS == 64)",
    )
    for contract in candidate_contracts:
        if contract not in candidate_body:
            raise AssertionError(
                f"affine B/A stage-early kernel lost contract: {contract}"
            )
    if (
        candidate_body.count(packed_stage_symbol) != 1
        or candidate_body.count(
            "k2_kda_context_affine_ab_fused_dense_stage_early_nw4_kernel"
        )
        != 1
    ):
        raise AssertionError("stage-early kernel symbols are not independent")

    next_stage = candidate_body.find("stage(ht_next, t0_next, alen_next);")
    b_advance = candidate_body.find(
        "advance.template operator()<false>(breg);", next_stage
    )
    dependency_fence = candidate_body.find("asm volatile(", b_advance)
    a_advance = candidate_body.find(
        "advance.template operator()<true>(areg);", dependency_fence
    )
    first_barrier = candidate_body.find("__syncthreads();", a_advance)
    next_commit = candidate_body.find("commit();", first_barrier)
    second_barrier = candidate_body.find("__syncthreads();", next_commit)
    write_b = candidate_body.find("affine_b[idx] = breg[ktile][i];")
    write_a = candidate_body.find(
        "affine_a[idx] = f32_to_bf16(areg[ktile][i]);", write_b
    )
    if not (
        0 <= next_stage < b_advance < dependency_fence < a_advance
        < first_barrier < next_commit < second_barrier < write_b < write_a
    ):
        raise AssertionError(
            "stage-early kernel changed arithmetic/barrier/commit/writeback order"
        )
    fallback_contracts = (
        "auto launch_affine_b = [&]()",
        "auto launch_affine_a = [&]()",
        "U_FORWARD, V_FORWARD, LDS_PIPELINE_B>",
        "U_FORWARD, V_FORWARD, LDS_PIPELINE_A>",
        "launch_affine_b(); launch_affine_a();",
    )
    for contract in fallback_contracts:
        if contract not in compact:
            raise AssertionError(
                f"affine B/A established fallback changed: {contract}"
            )

    launch_start = context_launch_start
    direct_start = compact.find("if (direct) {", launch_start)
    direct_end = compact.find("dispatch_state_mode", direct_start)
    if "k2_kda_context_affine_ab_fused" in compact[direct_start:direct_end]:
        raise AssertionError("pure direct context dispatch can launch fusion")

    def enabled(
        value: str | None,
        *,
        group_chunks: int,
        direct: bool,
        varlen: bool,
        sequences: int,
        nw8: bool,
        cached: bool,
        u_forward: bool,
        v_forward: bool,
        pipeline: tuple[bool, bool, bool],
        automatic_equal_n4_g64: bool,
    ) -> bool:
        parsed = value == "1" if value is not None else group_chunks in (8, 16)
        return (
            (parsed or (automatic_equal_n4_g64 and value is None))
            and not direct
            and (varlen or (not varlen and sequences == 1))
            and not nw8
            and cached
            and u_forward
            and v_forward
            and not pipeline[0]
            and not pipeline[1]
        )

    base = {
        "group_chunks": 32,
        "direct": False,
        "varlen": True,
        "sequences": 4,
        "nw8": False,
        "cached": True,
        "u_forward": True,
        "v_forward": True,
        "pipeline": (False, False, False),
        "automatic_equal_n4_g64": False,
    }
    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if enabled(spelling, **base):
            raise AssertionError(
                f"static fused parser accepted fallback {spelling!r}"
            )
    if not enabled("1", **base):
        raise AssertionError("static fused parser rejected valid packed layout")
    for group_chunks in (8, 16):
        if not enabled(None, **{**base, "group_chunks": group_chunks}):
            raise AssertionError(
                f"static fused parser rejected default G{group_chunks}"
            )
        if enabled("0", **{**base, "group_chunks": group_chunks}):
            raise AssertionError(
                f"static fused parser ignored exact-0 G{group_chunks} rollback"
            )
    automatic_n4_g64 = {
        **base,
        "group_chunks": 64,
        "automatic_equal_n4_g64": True,
    }
    if not enabled(None, **automatic_n4_g64):
        raise AssertionError(
            "static fused guard rejected automatic packed N4/G64"
        )
    if enabled("0", **automatic_n4_g64):
        raise AssertionError(
            "static fused guard ignored exact-0 packed N4/G64 rollback"
        )

    def stage_early_enabled(
        value: str | None,
        *,
        fused: bool,
        group_chunks: int,
        varlen: bool,
    ) -> bool:
        return (
            fused
            and value == "1"
            and (group_chunks == 64 or (not varlen and group_chunks == 16))
        )

    packed_g64_stage = {
        "fused": True,
        "group_chunks": 64,
        "varlen": True,
    }
    dense_g16_stage = {
        "fused": True,
        "group_chunks": 16,
        "varlen": False,
    }
    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if stage_early_enabled(spelling, **packed_g64_stage):
            raise AssertionError(
                "static stage-early parser accepted fallback "
                f"{spelling!r} for {_CONTEXT_AFFINE_AB_STAGE_EARLY_ENV}"
            )
    if not stage_early_enabled("1", **packed_g64_stage):
        raise AssertionError("static stage-early parser rejected packed G64")
    if not stage_early_enabled("1", **dense_g16_stage):
        raise AssertionError("static stage-early parser rejected dense G16")
    for rejected in (
        {**packed_g64_stage, "fused": False},
        {**dense_g16_stage, "fused": False},
        {**dense_g16_stage, "varlen": True},
        {**packed_g64_stage, "group_chunks": 8},
        {**packed_g64_stage, "group_chunks": 32},
        {**packed_g64_stage, "group_chunks": 128},
    ):
        if stage_early_enabled("1", **rejected):
            raise AssertionError(
                f"static stage-early selector accepted fallback {rejected}"
            )

    def resolved_scan_nw(
        value: str | None, *, automatic_gva_scan_nw4: bool = False
    ) -> int:
        return (
            int(value)
            if value is not None
            else 4 if automatic_gva_scan_nw4 else 2
        )

    if resolved_scan_nw(None) != 2:
        raise AssertionError("static affine scan NW2 default changed")
    if resolved_scan_nw(None, automatic_gva_scan_nw4=True) != 4:
        raise AssertionError("static automatic GVA affine scan NW4 changed")
    for explicit_nw in (1, 2, 4):
        if (
            resolved_scan_nw(
                str(explicit_nw), automatic_gva_scan_nw4=True
            )
            != explicit_nw
        ):
            raise AssertionError(
                "explicit scan NW did not override the automatic GVA NW4"
            )
    dense_n1 = {**base, "varlen": False, "sequences": 1}
    if not enabled("1", **dense_n1):
        raise AssertionError("static fused parser rejected valid dense N=1 layout")
    fallbacks = (
        {"direct": True},
        {"varlen": False, "sequences": 2},
        {"nw8": True},
        {"cached": False},
        {"u_forward": False},
        {"v_forward": False},
        {"pipeline": (True, False, False)},
        {"pipeline": (False, True, False)},
    )
    for override in fallbacks:
        configuration = {**base, **override}
        if enabled("1", **configuration):
            raise AssertionError(
                f"static affine B/A fused guard accepted {override}"
            )
        stage_configuration = {**base, "group_chunks": 64, **override}
        if stage_early_enabled(
            "1",
            fused=enabled("1", **stage_configuration),
            group_chunks=stage_configuration["group_chunks"],
            varlen=stage_configuration["varlen"],
        ):
            raise AssertionError(
                f"static affine B/A stage-early guard accepted {override}"
            )
    if not enabled("1", **{**base, "pipeline": (False, False, True)}):
        raise AssertionError(
            "replay-only LDS pipeline incorrectly disabled affine B/A fusion"
        )
    replay_only_g64 = {
        **base,
        "group_chunks": 64,
        "pipeline": (False, False, True),
    }
    if not stage_early_enabled(
        "1",
        fused=enabled("1", **replay_only_g64),
        group_chunks=64,
        varlen=True,
    ):
        raise AssertionError(
            "replay-only LDS pipeline incorrectly disabled stage-early producer"
        )
    _check_context_equal_dense_n4_g64_policy_static(policy, kernel)
    print(
        "PASS static packed/dense-N1/equal-dense-N4 affine B/A fused and "
        "stage-early selectors, source rollback hashes, launch ABIs, and "
        "whole-graph fallback guards"
    )


def check_context_scan_ksplit_policy_static() -> None:
    """Audit the exact parser, 2-D mapping, precedence, and fallbacks."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    kernel_path = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/"
        "k2_kda_context_affine_scan_ksplit_kernel.hpp"
    )
    policy = policy_path.read_text()
    kernel = kernel_path.read_text()
    compact_policy = " ".join(policy.split())
    original_symbol = "k2_kda_context_affine_scan_ksplit_wg4_kernel("
    prefetch_symbol = (
        "k2_kda_context_affine_scan_ksplit_prefetch_b_g64_wg4_kernel("
    )
    original_start = kernel.find(original_symbol)
    prefetch_start = kernel.find(prefetch_symbol)
    if original_start < 0 or prefetch_start <= original_start:
        raise AssertionError("K-split original/prefetch symbols are missing")
    original_kernel = kernel[original_start:prefetch_start]
    prefetch_kernel = kernel[prefetch_start:]
    compact_kernel = " ".join(original_kernel.split())
    compact_prefetch_kernel = " ".join(prefetch_kernel.split())

    include = '#include "k2_kda_context_affine_scan_ksplit_kernel.hpp"'
    if include not in policy:
        raise AssertionError("policy does not include the K-split scan header")
    parser_start = policy.find("static bool context_scan_ksplit_enabled()")
    parser_end = policy.find("\n    }", parser_start)
    if parser_start < 0 or parser_end < 0:
        raise AssertionError("policy is missing the K-split scan parser")
    parser = " ".join(policy[parser_start:parser_end].split())
    if f'std::getenv("{_CONTEXT_SCAN_KSPLIT_ENV}")' not in parser:
        raise AssertionError("K-split parser reads the wrong environment")
    exact_return = (
        "return value != nullptr && value[0] == '1' && "
        "value[1] == '\\0';"
    )
    if exact_return not in parser:
        raise AssertionError("K-split parser is not exact-'1'")

    prefetch_parser_start = policy.find(
        "static bool context_scan_ksplit_prefetch_b_enabled()"
    )
    prefetch_parser_end = policy.find("\n    }", prefetch_parser_start)
    if prefetch_parser_start < 0 or prefetch_parser_end < 0:
        raise AssertionError("policy is missing the K-split prefetch-b parser")
    prefetch_parser = " ".join(
        policy[prefetch_parser_start:prefetch_parser_end].split()
    )
    if (
        'std::getenv( "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT_PREFETCH_B")'
        not in prefetch_parser
        or exact_return not in prefetch_parser
    ):
        raise AssertionError("K-split prefetch-b parser is not exact-'1'")

    automatic_guard = (
        "const bool automatic_equal_n4_g64_ksplit = "
        "automatic_equal_n4_g64 && scan_nw_env == nullptr && "
        "std::getenv( "
        '"FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT") == nullptr;'
    )
    if automatic_guard not in compact_policy:
        raise AssertionError(
            "K-split lost its exact zero-environment N4/G64 auto guard"
        )
    effective_guard = (
        "const bool scan_ksplit = "
        "(context_scan_ksplit_enabled() || "
        "automatic_equal_n4_g64_ksplit) && "
        "!direct && !hybrid && scan_nw == 2 && !scan_b_stream && "
        "!scan_a_gll && !scan_b_phased && "
        "(a.is_varlen || (!a.is_varlen && a.N == 1) || "
        "equal_dense_n4_g64);"
    )
    if effective_guard not in compact_policy:
        raise AssertionError(
            "K-split lost its route/NW/layout/scan-axis fallback guard"
        )
    launch_guard = "if constexpr (NW == 2 && !TIGHT_VL_GRID)"
    if launch_guard not in compact_policy:
        raise AssertionError("K-split lost its compile-time non-tight NW2 guard")
    launch_symbol = "k2_kda_context_affine_scan_ksplit_wg4_kernel<"
    if policy.count(launch_symbol) != 1:
        raise AssertionError("policy must contain exactly one K-split launch")
    prefetch_launch_symbol = (
        "k2_kda_context_affine_scan_ksplit_prefetch_b_g64_wg4_kernel<"
    )
    if policy.count(prefetch_launch_symbol) != 1:
        raise AssertionError(
            "policy must contain exactly one G64 K-split prefetch-b launch"
        )
    prefetch_policy_contracts = (
        "const bool scan_ksplit_prefetch_b = scan_ksplit && "
        "context_scan_ksplit_prefetch_b_enabled();",
        "if constexpr (GROUP_CHUNKS == 64)",
        "if (scan_ksplit_prefetch_b)",
        "k2_kda_context_affine_scan_ksplit_prefetch_b_g64_wg4_kernel< "
        "HI, HO, FP, VL>",
    )
    for contract in prefetch_policy_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                f"K-split prefetch-b dispatch is missing contract: {contract}"
            )
    launch_contracts = (
        "GROUP_CHUNKS, HI, HO, FP, VL>",
        "<<<dim3(scan_contexts * a.H, 4), 256, 0, a.stream>>>",
        "VL ? a.cu_seqlens : nullptr",
        "VL ? a.context_prefix : nullptr",
        "a.T_seq, a.H, a.NT",
    )
    for contract in launch_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                f"K-split launch ABI is missing contract: {contract}"
            )

    kernel_contracts = (
        "const int vhalf = wave & 1;",
        "const int khalf = wave >> 1;",
        "const int v0 = (int(blockIdx.y) * 2 + vhalf) * BV;",
        "const int k0 = khalf * KHALF;",
        "SMEM_BYTES == 50176",
        "__global__ void __launch_bounds__(256)",
        "const float plo = khalf == 0 ? owned[ktile][i] : remote[i];",
        "const float phi = khalf == 0 ? remote[i] : owned[ktile][i];",
        "hreg[ktile][i] = (plo + phi) + b;",
    )
    for contract in kernel_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                f"K-split kernel mapping changed: missing {contract}"
            )
    if original_kernel.count("__syncthreads();") != 2:
        raise AssertionError(
            "original K-split must retain exactly two barriers per group"
        )

    prefetch_kernel_contracts = (
        "constexpr int GROUP_CHUNKS = 64;",
        "float breg[LOCAL_NKB][4];",
        "breg[ktile][i] = affine_b[idx];",
        "affine_b[idx] = hreg[ktile][i];",
        "hreg[ktile][i] = (plo + phi) + breg[ktile][i];",
    )
    for contract in prefetch_kernel_contracts:
        if contract not in compact_prefetch_kernel:
            raise AssertionError(
                f"K-split prefetch-b kernel changed: missing {contract}"
            )
    if prefetch_kernel.count("__syncthreads();") != 2:
        raise AssertionError(
            "K-split prefetch-b must retain exactly two barriers per group"
        )

    def enabled(
        value: str | None,
        *,
        direct: bool,
        hybrid: bool,
        scan_nw_value: str | None,
        varlen: bool,
        sequences: int,
        tight: bool,
        b_stream: bool,
        a_gll: bool,
        b_phased: bool,
        automatic_equal_n4_g64: bool,
    ) -> bool:
        automatic = (
            automatic_equal_n4_g64
            and scan_nw_value is None
            and value is None
        )
        scan_nw = int(scan_nw_value) if scan_nw_value is not None else 2
        return (
            (value == "1" or automatic)
            and not direct
            and not hybrid
            and scan_nw == 2
            and not tight
            and not b_stream
            and not a_gll
            and not b_phased
            and (varlen or (not varlen and sequences == 1))
        )

    base = {
        "direct": False,
        "hybrid": False,
        "scan_nw_value": None,
        "varlen": False,
        "sequences": 1,
        "tight": False,
        "b_stream": False,
        "a_gll": False,
        "b_phased": False,
        "automatic_equal_n4_g64": False,
    }
    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if enabled(spelling, **base):
            raise AssertionError(
                f"static K-split parser accepted fallback {spelling!r}"
            )
    if not enabled("1", **base):
        raise AssertionError("K-split rejected dense N=1")
    if not enabled("1", **{**base, "varlen": True, "sequences": 4}):
        raise AssertionError("K-split rejected packed pure-affine")
    automatic_n4_g64 = {
        **base,
        "varlen": True,
        "sequences": 4,
        "automatic_equal_n4_g64": True,
    }
    if not enabled(None, **automatic_n4_g64):
        raise AssertionError("K-split rejected automatic packed N4/G64")
    if enabled("0", **automatic_n4_g64):
        raise AssertionError("explicit K-split zero did not disable automatic")
    if not enabled("1", **automatic_n4_g64):
        raise AssertionError("explicit K-split one did not replace automatic")
    if enabled(None, **{**automatic_n4_g64, "scan_nw_value": "2"}):
        raise AssertionError("explicit scan NW did not disable automatic K-split")
    fallbacks = (
        {"direct": True},
        {"hybrid": True, "varlen": True},
        {"scan_nw_value": "1"},
        {"scan_nw_value": "4"},
        {"varlen": False, "sequences": 2},
        {"tight": True, "varlen": True},
        {"b_stream": True},
        {"a_gll": True},
        {"b_phased": True},
    )
    for override in fallbacks:
        if enabled("1", **{**base, **override}):
            raise AssertionError(
                f"static K-split guard accepted fallback {override}"
            )
    print(
        "PASS static affine-scan K-split exact parser, WG4 K/V mapping, "
        "single-axis isolation, and dense-N1/packed-affine fallback guard"
    )


def check_context_persistent_policy_static() -> None:
    """Audit the persistent candidate's complete guard and matched graph."""

    policy_path = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    common_path = _REPO_ROOT / "csrc/kernels/flash_kda/hip_common.hpp"
    launch_path = _REPO_ROOT / "csrc/kernels/flash_kda/hip_launch_common.cu"
    workspace_path = _REPO_ROOT / "csrc/include/flash_kda.h"
    raw_abi_path = _REPO_ROOT / "csrc/kernels/flash_kda/flash_kda_aiter.cu"
    workspace_cpp_path = _REPO_ROOT / "csrc/kernels/flash_kda/hip_workspace.cpp"
    kernel_root = _REPO_ROOT / "csrc/kernels/flash_kda/gfx950"
    fused_path = (
        kernel_root / "k2_kda_context_affine_ab_fused_persistent_kernel.hpp"
    )
    compact_path = (
        kernel_root / "k2_kda_context_hybrid_compact_scan_kernel.hpp"
    )
    replay_path = (
        kernel_root / "k2_kda_context_hybrid_persistent_replay_kernel.hpp"
    )
    policy = policy_path.read_text()
    common = common_path.read_text()
    launch = launch_path.read_text()
    workspace = workspace_path.read_text()
    raw_abi = raw_abi_path.read_text()
    workspace_cpp = workspace_cpp_path.read_text()
    fused = fused_path.read_text()
    compact_kernel = compact_path.read_text()
    replay = replay_path.read_text()
    compact_policy = " ".join(policy.split())
    compact_common = " ".join(common.split())
    compact_launch = " ".join(launch.split())
    compact_workspace = " ".join(workspace.split())
    compact_raw_abi = " ".join(raw_abi.split())
    compact_workspace_cpp = " ".join(workspace_cpp.split())

    def cpp_block(source: str, signature: str) -> str:
        start = source.find(signature)
        if start < 0:
            raise AssertionError(f"missing C++ contract: {signature}")
        opening = source.find("{", start)
        if opening < 0:
            raise AssertionError(f"cannot delimit C++ contract: {signature}")
        depth = 0
        for position in range(opening, len(source)):
            character = source[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return source[start : position + 1]
        raise AssertionError(f"unterminated C++ contract: {signature}")

    includes = (
        '#include "k2_kda_context_affine_ab_fused_persistent_kernel.hpp"',
        '#include "k2_kda_context_hybrid_compact_scan_kernel.hpp"',
        '#include "k2_kda_context_hybrid_persistent_replay_kernel.hpp"',
    )
    for include in includes:
        if policy.count(include) != 1:
            raise AssertionError(
                f"persistent policy must include exactly one {include}"
            )

    exact_helper = " ".join(
        cpp_block(
            policy,
            "static bool env_exact(const char* name, const char* expected)",
        ).split()
    )
    if (
        "const char* value = std::getenv(name);" not in exact_helper
        or "value != nullptr && std::strcmp(value, expected) == 0"
        not in exact_helper
    ):
        raise AssertionError("persistent env_exact helper is not exact-string")
    optional_helper = " ".join(
        cpp_block(
            policy,
            "static bool env_unset_or_exact(",
        ).split()
    )
    if (
        "const char* value = std::getenv(name);" not in optional_helper
        or "value == nullptr || std::strcmp(value, expected) == 0"
        not in optional_helper
    ):
        raise AssertionError(
            "persistent optional parser accepts a noncanonical spelling"
        )
    established_ab_helper = " ".join(
        cpp_block(
            policy,
            "static bool context_persistent_established_ab_enabled()",
        ).split()
    )
    if (
        f'env_exact( "{_CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV}", "1")'
        not in established_ab_helper
    ):
        raise AssertionError(
            "persistent established-AB parser is not exact-string"
        )

    guard = " ".join(
        cpp_block(policy, "static bool context_persistent_enabled(").split()
    )
    if _CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV in guard:
        raise AssertionError(
            "nested established-AB flag changed the parent persistent guard"
        )
    guard_pairs = {
        _CONTEXT_PERSISTENT_ENV: "1",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT": "0",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE": "0",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID": "1",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS": "64",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW": "4",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW": "2",
        _CONTEXT_NW8_ENV: "0",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM": "0",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL": "0",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED": "0",
        _CONTEXT_SCAN_KSPLIT_ENV: "0",
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN": "0",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE": "0",
        _CONTEXT_LDS_PIPELINE_PASS_ENV[0]: "0",
        _CONTEXT_LDS_PIPELINE_PASS_ENV[1]: "0",
        _CONTEXT_LDS_PIPELINE_PASS_ENV[2]: "0",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE": "1",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD": "1",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD": "1",
        _CONTEXT_AFFINE_AB_FUSED_ENV: "1",
    }
    for environment, expected in guard_pairs.items():
        pair = f'"{environment}", "{expected}"'
        if guard.count(pair) != 1:
            raise AssertionError(
                "persistent guard must consume each canonical recipe value "
                f"exactly once: {pair}"
            )
    host_guards = (
        "p.cu_seqlens == nullptr",
        "p.N <= 0",
        "p.H <= 0",
        "device.cu_count <= 0",
        "default_k2_route != K2DefaultRoute::context_parallel",
        "route.group_chunks != kHybridCompactGroupChunks",
        "route.direct_max_chunks != kHybridCompactDirectMaxChunks",
        'std::getenv("FLASH_KDA_K2") != nullptr',
        "context_operand_cache_active()",
    )
    for contract in host_guards:
        if contract not in guard:
            raise AssertionError(
                f"persistent complete host guard is missing {contract}"
            )

    make = " ".join(
        cpp_block(policy, "static HipLaunchPolicy make(").split()
    )
    make_contracts = (
        "const bool use_context_persistent = context_persistent_enabled( "
        "p, device, context_route, default_k2_route);",
        "if (use_context_persistent) { "
        "policy.launch_context_prefix = &launch_context_persistent_prefix; "
        "policy.context_persistent_blocks = device.cu_count; }",
    )
    for contract in make_contracts:
        if contract not in make:
            raise AssertionError(
                f"persistent policy activation is not atomic: {contract}"
            )

    common_contracts = (
        "const int* sequence_worklist; const int* sequence_count;",
        "int context_persistent_blocks;",
        "PersistentPrefixLauncher launch_context_prefix = nullptr; "
        "int context_persistent_blocks = 0;",
    )
    for contract in common_contracts:
        if contract not in compact_common:
            raise AssertionError(
                f"persistent common ABI is missing contract: {contract}"
            )
    if (
        "((4 * N + 5) * int64_t(sizeof(int32_t)) + 127) / 128 * 128"
        not in compact_workspace
    ):
        raise AssertionError(
            "workspace prefix arena does not reserve 3*(N+1)+N+count+counter"
        )
    if (
        "((4 * sequences + 5) * static_cast<wide>(sizeof(int32_t)) + 127) "
        "/ 128 * 128" not in compact_raw_abi
        or "WorkspaceSizes::prefix_bytes(N)" not in compact_workspace_cpp
    ):
        raise AssertionError(
            "descriptor/raw workspace sizing disagrees with the expanded "
            "prefix ABI"
        )

    launch_contracts = (
        "const bool use_context_persistent = use_context_parallel && "
        "!is_gva && is_varlen && policy.launch_context_prefix != nullptr && "
        "policy.context_persistent_blocks > 0;",
        "else if (use_context_persistent)",
        "policy.launch_context_prefix(args);",
        "segment_prefix, sequence_worklist, sequence_count,",
        "use_context_persistent ? policy.context_persistent_blocks : 0",
    )
    for contract in launch_contracts:
        if contract not in compact_launch:
            raise AssertionError(
                f"common launch lost matched prefix/K2 wiring: {contract}"
            )

    prefix_launcher = " ".join(
        cpp_block(
            policy, "static void launch_context_persistent_prefix("
        ).split()
    )
    if (
        "k1_build_tile_prefix_hybrid_g64_compact_kernel "
        "<<<1, 64, 0, a.stream>>>" not in prefix_launcher
        or "a.segment_prefix, a.sequence_worklist, a.sequence_count"
        not in prefix_launcher
    ):
        raise AssertionError("persistent prefix callback ABI changed")

    persistent_launch = " ".join(
        cpp_block(
            policy, "static void launch_context_parallel_persistent("
        ).split()
    )
    stage_contracts = (
        "const dim3 persistent_grid(a.context_persistent_blocks);",
        "const bool use_established_ab = "
        "context_persistent_established_ab_enabled();",
        "dispatch_state_mode<true>(",
        "kHybridCompactDirectMaxChunks, true, true, true, false>",
        "k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel",
        "k2_kda_context_affine_scan_hybrid_g64_compact_grid_stride_nw2_kernel< "
        "HI, HO, FP>",
        "k2_kda_context_replay_hybrid_g64_grid_stride_nw4_kernel<HO, FP>",
    )
    for contract in stage_contracts:
        if contract not in persistent_launch:
            raise AssertionError(
                f"persistent four-stage topology is missing: {contract}"
            )
    established_ab_contracts = (
        "if (use_established_ab) {",
        "a.total_tiles / (kHybridCompactDirectMaxChunks + 1)",
        "int64_t(max_affine_sequences) * "
        "(kHybridCompactGroupChunks - 1)",
        "const int context_upper = std::max(1, int(upper));",
        "const dim3 established_ab_grid(context_upper * a.H, 2);",
        "k2_kda_context_affine_ab_fused_nw4_kernel< "
        "kHybridCompactGroupChunks> "
        "<<<established_ab_grid, 256, 0, a.stream>>>",
        "} else { "
        "k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel "
        "<<<persistent_grid, 256, 0, a.stream>>>",
    )
    for contract in established_ab_contracts:
        if contract not in persistent_launch:
            raise AssertionError(
                "persistent established-AB nested graph is missing: "
                f"{contract}"
            )
    if persistent_launch.count("<<<established_ab_grid") != 1:
        raise AssertionError(
            "persistent established-AB candidate changed its one-task grid"
        )
    if persistent_launch.count("<<<persistent_grid") != 3:
        raise AssertionError(
            "persistent affine stages do not share one capped physical grid"
        )
    if persistent_launch.count("k2_kda_context_parallel_nw4_kernel<") != 1:
        raise AssertionError(
            "persistent direct short/empty owner is missing or duplicated"
        )

    dispatch = " ".join(
        cpp_block(policy, "static void launch_context_parallel(").split()
    )
    fallback_start = dispatch.find("const bool direct = group_chunks == 0;")
    if fallback_start < 0:
        raise AssertionError("established context fallback body is missing")
    persistent_prefix = dispatch[:fallback_start]
    dispatch_contracts = (
        "if (!a.is_gva && a.context_persistent_blocks > 0",
        "a.context_persistent_blocks > 0",
        "a.is_varlen",
        "group_chunks == kHybridCompactGroupChunks",
        "direct_max_chunks == kHybridCompactDirectMaxChunks",
        "a.sequence_worklist != nullptr",
        "a.sequence_count != nullptr",
        "launch_context_parallel_persistent(a); return;",
    )
    for contract in dispatch_contracts:
        if contract not in persistent_prefix:
            raise AssertionError(
                f"persistent K2 gate is missing matched contract: {contract}"
            )
    if any(symbol in dispatch[fallback_start:] for symbol in (
        "affine_ab_fused_persistent_g64_nw4",
        "affine_scan_hybrid_g64_compact_grid_stride_nw2",
        "replay_hybrid_g64_grid_stride_nw4",
    )):
        raise AssertionError("persistent symbols leaked into fallback dispatch")

    kernel_contracts = (
        (
            fused,
            "k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel",
            "context_prefix[N]",
        ),
        (
            compact_kernel,
            "k2_kda_context_affine_scan_hybrid_g64_compact_grid_stride_"
            "nw2_kernel",
            "sequence_count[0]",
        ),
        (
            replay,
            "k2_kda_context_replay_hybrid_g64_grid_stride_nw4_kernel",
            "context_prefix[N]",
        ),
    )
    for source, symbol, exact_count in kernel_contracts:
        if symbol not in source or exact_count not in source:
            raise AssertionError(
                f"persistent kernel lost exact device work count: {symbol}"
            )
        if "task64 += int64_t(gridDim.x)" not in source:
            raise AssertionError(
                f"persistent kernel lost deterministic grid stride: {symbol}"
            )
        if "atomicAdd(" in source or "task_counter" in source:
            raise AssertionError(
                f"persistent kernel introduced reset/counter state: {symbol}"
            )
    prefix_contracts = (
        "k1_build_tile_prefix_hybrid_g64_compact_kernel",
        "context_prefix[0] = 0;",
        "sequence_worklist[position] = seq;",
        "sequence_count[0] = long_count;",
        "chunks > kHybridCompactDirectMaxChunks",
    )
    for contract in prefix_contracts:
        if contract not in compact_kernel:
            raise AssertionError(
                f"persistent compact prefix lost contract: {contract}"
            )

    def exact_or_unset(value: str | None, expected: str) -> bool:
        return value is None or value == expected

    def enabled(
        persistent: str | None,
        *,
        packed: bool = True,
        sequences: int = 9,
        heads: int = 1,
        device_cus: int = 256,
        context_route: bool = True,
        group_chunks: int = 64,
        direct_chunks: int = 64,
        explicit_k2: str | None = None,
        recipe: dict[str, str | None] | None = None,
    ) -> bool:
        values = {
            "direct": "0",
            "affine": "0",
            "hybrid": "1",
            "group": "64",
            "direct_nw": "4",
            "scan_nw": "2",
            "nw8": "0",
            "b_stream": "0",
            "a_gll": "0",
            "b_phased": "0",
            "ksplit": "0",
            "tight": "0",
            "pipeline": "0",
            "pipeline_b": "0",
            "pipeline_a": "0",
            "pipeline_replay": "0",
            "cache": "1",
            "u": "1",
            "v": "1",
            "fused": "1",
        }
        if recipe is not None:
            values.update(recipe)
        off_axes = (
            "direct",
            "affine",
            "nw8",
            "b_stream",
            "a_gll",
            "b_phased",
            "ksplit",
            "tight",
            "pipeline",
            "pipeline_b",
            "pipeline_a",
            "pipeline_replay",
        )
        return (
            persistent == "1"
            and packed
            and sequences > 0
            and heads > 0
            and device_cus > 0
            and context_route
            and group_chunks == 64
            and direct_chunks == 64
            and explicit_k2 is None
            and all(exact_or_unset(values[axis], "0") for axis in off_axes)
            and values["hybrid"] == "1"
            and values["group"] == "64"
            and exact_or_unset(values["direct_nw"], "4")
            and exact_or_unset(values["scan_nw"], "2")
            and exact_or_unset(values["cache"], "1")
            and values["u"] == "1"
            and values["v"] == "1"
            and values["fused"] == "1"
        )

    def selected_ab(
        persistent: str | None,
        established_ab: str | None,
        **configuration,
    ) -> str:
        if not enabled(persistent, **configuration):
            return "established-fallback"
        if established_ab == "1":
            return "established-one-task"
        return "persistent-grid-stride"

    if not enabled("1"):
        raise AssertionError("static persistent model rejected canonical recipe")
    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if enabled(spelling):
            raise AssertionError(
                f"static persistent parser accepted {spelling!r}"
            )
    host_fallbacks = (
        {"packed": False},
        {"sequences": 0},
        {"heads": 0},
        {"device_cus": 0},
        {"context_route": False},
        {"group_chunks": 32},
        {"group_chunks": 128},
        {"direct_chunks": 0},
        {"explicit_k2": "csplit64"},
    )
    for override in host_fallbacks:
        if enabled("1", **override):
            raise AssertionError(
                f"static persistent host guard accepted {override}"
            )
    recipe_fallbacks = {
        "direct": "1",
        "affine": "1",
        "hybrid": "0",
        "group": "32",
        "direct_nw": "8",
        "scan_nw": "4",
        "nw8": "1",
        "b_stream": "1",
        "a_gll": "1",
        "b_phased": "1",
        "ksplit": "1",
        "tight": "1",
        "pipeline": "1",
        "pipeline_b": "1",
        "pipeline_a": "1",
        "pipeline_replay": "1",
        "cache": "0",
        "u": "0",
        "v": "0",
        "fused": "0",
    }
    for axis, value in recipe_fallbacks.items():
        for spelling in (value, "true"):
            if enabled("1", recipe={axis: spelling}):
                raise AssertionError(
                    "static persistent full guard accepted "
                    f"{axis}={spelling!r}"
                )

    # Old environments are graph-identical: without the new exact nested
    # flag, an active persistent recipe still selects the original AB symbol,
    # while every parent fallback remains an established nonpersistent graph.
    for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected_ab("1", spelling) != "persistent-grid-stride":
            raise AssertionError(
                "nested established-AB parser changed the old persistent "
                f"graph for {spelling!r}"
            )
    for persistent in (None, "", "0", "01", "true", "1 ", " 1"):
        if selected_ab(persistent, "1") != "established-fallback":
            raise AssertionError(
                "isolated established-AB flag activated without its parent "
                f"persistent graph: persistent={persistent!r}"
            )
    if selected_ab("1", "1") != "established-one-task":
        raise AssertionError(
            "canonical persistent + established-AB recipe did not compose"
        )
    for override in host_fallbacks:
        if selected_ab("1", "1", **override) != "established-fallback":
            raise AssertionError(
                "nested established-AB flag escaped a parent host guard: "
                f"{override}"
            )
    for axis, value in recipe_fallbacks.items():
        if selected_ab(
            "1", "1", recipe={axis: value}
        ) != "established-fallback":
            raise AssertionError(
                "nested established-AB flag escaped a parent recipe guard: "
                f"{axis}={value!r}"
            )

    def hybrid_context_upper(total_tiles: int, sequences: int) -> int:
        max_affine_sequences = min(sequences, total_tiles // 65)
        return max(1, (total_tiles + max_affine_sequences * 63) // 64)

    if hybrid_context_upper(1026, 8) != 23:
        raise AssertionError(
            "nested established-AB host upper changed for ragged-16k"
        )
    print(
        "PASS static persistent exact parser, complete packed-hybrid G64 "
        "guard, matched prefix/K2 graph, deterministic grids, nested "
        "established-AB isolation/composition, and whole-graph fallback"
    )


def check_gva_whole_route_policy_static() -> None:
    """Pin the complete gfx950 grouped-value producer/consumer handshake."""

    common_abi = (
        _REPO_ROOT / "csrc/kernels/flash_kda/hip_common.hpp"
    ).read_text()
    common_launch = (
        _REPO_ROOT / "csrc/kernels/flash_kda/hip_launch_common.cu"
    ).read_text()
    policy = (
        _REPO_ROOT / "csrc/kernels/flash_kda/gfx950/policy.hpp"
    ).read_text()
    fused_k1 = (
        _REPO_ROOT
        / "csrc/kernels/flash_kda/gfx950/k1_kda_bt16_fused_kernel.hpp"
    ).read_text()
    entry = (
        _REPO_ROOT / "csrc/kernels/flash_kda/flash_kda_aiter.cu"
    ).read_text()
    public_header = (_REPO_ROOT / "csrc/include/flash_kda.h").read_text()
    pybind = (_REPO_ROOT / "csrc/pybind/flash_kda_pybind.cu").read_text()
    python_adapter = (_REPO_ROOT / "aiter/ops/flash_kda.py").read_text()
    compact_abi = " ".join(common_abi.split())
    compact_common = " ".join(common_launch.split())
    compact_policy = " ".join(policy.split())
    compact_k1 = " ".join(fused_k1.split())
    compact_entry = " ".join(entry.split())
    compact_header = " ".join(public_header.split())
    compact_pybind = " ".join(pybind.split())
    compact_adapter = " ".join(python_adapter.split())

    raw_v3_contracts = (
        (compact_header, "void flash_kda_fwd_hip_raw_v3("),
        (compact_entry, "void flash_kda_fwd_hip_raw_v3("),
        (
            compact_entry,
            "raw_check(H >= H_q && H % H_q == 0, "
            '"H_v must be an integer multiple of H_q");',
        ),
        (
            compact_entry,
            "static_cast<int>(H_q), static_cast<int>(H), "
            "static_cast<int>(N)",
        ),
        (
            compact_pybind,
            'm.def("flash_kda_fwd_hip_raw_v3", '
            "&aiter::flash_kda_fwd_hip_raw_v3",
        ),
        (compact_pybind, 'py::arg("max_seqlen_upper_bound"), py::arg("H_q")'),
        (
            compact_adapter,
            '(3, "flash_kda_fwd_hip_raw_v3"), '
            '(2, "flash_kda_fwd_hip_raw_v2"), '
            '(1, "flash_kda_fwd_hip_raw")',
        ),
        (
            compact_adapter,
            "if raw_version == 3: raw_op( *raw_v1_args, "
            "0 if max_seqlen_upper_bound is None else "
            "max_seqlen_upper_bound, num_qk_heads, )",
        ),
    )
    for source, contract in raw_v3_contracts:
        if contract not in source:
            raise AssertionError("GVA raw-v3 ABI chain changed: " + contract)

    # GVA is a whole-route ABI, not a K1-local indexing switch.  Require every
    # capability and resolved policy fact to survive the two aggregate
    # boundaries between architecture policy, common dispatch, and context K2.
    abi_contracts = (
        "int H_q; int H;",
        "int H; int H_q; int NT;",
        "bool is_gva = false;",
        "bool automatic_gva_packed_nw4 = false;",
        "bool automatic_gva_equal_n4_g16 = false;",
        "bool bt16_k1_supports_gva = false;",
        "bool plain_csplit_supports_gva = false;",
        "bool context_automatic_gva_packed_nw4 = false;",
        "bool context_automatic_gva_equal_n4_g16 = false;",
    )
    for contract in abi_contracts:
        if contract not in compact_abi:
            raise AssertionError("GVA common ABI lost contract: " + contract)

    policy_wiring = (
        "bool automatic_gva_packed_nw4; "
        "bool automatic_gva_equal_n4_g16;",
        "policy.context_automatic_gva_packed_nw4 = "
        "context_route.automatic_gva_packed_nw4;",
        "policy.context_automatic_gva_equal_n4_g16 = "
        "context_route.automatic_gva_equal_n4_g16;",
        "policy.bt16_k1_supports_gva = "
        "bt16_fused_mode() != Bt16FusedMode::disabled;",
        "policy.plain_csplit_supports_gva = "
        "policy.bt16_k1_supports_gva && "
        "policy.launch_bt16_k1 != nullptr && "
        "policy.use_bt16_k1_for_plain && "
        "policy.launch_plain_k1 != nullptr;",
    )
    for contract in policy_wiring:
        if contract not in compact_policy:
            raise AssertionError("GVA policy wiring changed: " + contract)

    common_contracts = (
        "const int H_q = p.H_q; const int H = p.H; "
        "const bool is_gva = H_q != H;",
        "const char* k2env = is_gva ? nullptr : getenv(\"FLASH_KDA_K2\");",
        "const bool use_gva_plain_csplit = is_gva && "
        "policy.plain_csplit_supports_gva && "
        "policy.bt16_k1_supports_gva && "
        "policy.launch_bt16_k1 != nullptr && "
        "policy.use_bt16_k1_for_plain && "
        "policy.launch_plain_k1 != nullptr && "
        "!cs_skip_k1_prep && !cs_skip_k1_solve;",
        "policy.default_k2_route == K2DefaultRoute::csplit64 && "
        "(!is_gva || use_gva_plain_csplit);",
        "policy.default_k2_route == K2DefaultRoute::context_parallel && "
        "policy.launch_context_parallel != nullptr && "
        "(!is_gva || (policy.launch_bt16_k1 != nullptr && "
        "policy.bt16_k1_supports_gva));",
        "const bool request_context_operand_cache = "
        "use_context_parallel && !is_gva && "
        "policy.launch_bt16_k1 != nullptr && "
        "policy.bt16_k1_context_operand_cache;",
        "const bool use_csplit_bt16_k1 = use_default_csplit64 && "
        "policy.launch_bt16_k1 != nullptr && "
        "policy.use_bt16_k1_for_plain && "
        "(!is_gva || use_gva_plain_csplit) && "
        "!cs_skip_k1_prep && !cs_skip_k1_solve;",
        "const bool use_default_vsplit_rs = is_gva || "
        "(!k2env && policy.default_k2_route == K2DefaultRoute::vsplit_rs);",
        "context_operands_cached, is_gva, "
        "policy.context_automatic_gva_packed_nw4, "
        "policy.context_automatic_gva_equal_n4_g16};",
    )
    for contract in common_contracts:
        if contract not in compact_common:
            raise AssertionError(
                "GVA common-launch handshake changed: " + contract
            )

    # The fused producer maps every value head back to its shared Q/K head,
    # while all persistent workspaces and K2 launches remain value-head major.
    k1_contracts = (
        "bool DENSE_N1_ALL_FULL_C16, bool GVA = false>",
        "static_assert(!GVA || (!CACHE_CONTEXT_OPERANDS && "
        "!PUBLISH_ACTIVATED_BETA && !PACKED_DIRECT_PREFIXLESS && "
        "!DENSE_N1_ALL_FULL_C16)",
        "if (a.H_q != a.H) { auto launch_gva = [&]<bool VL>()",
        "VL, true, false, false, false, false, false, true>(a, grid);",
        "VL, true, true, false, false, false, false, true>(a, grid);",
        "VL, false, true, false, false, false, false, true>(a, grid);",
        "a.scale, a.gate_scale, a.T_seq, a.H, a.H_q",
        "if constexpr (GVA) { const int hq = h / (H / H_q); "
        "qk_off = (int64_t(t0 + vec_m) * H_q + hq) * D + vec_d0; }",
    )
    for contract in k1_contracts:
        if contract not in compact_policy and contract not in compact_k1:
            raise AssertionError("GVA fused-K1 contract changed: " + contract)

    route_contracts = (
        "const bool is_gva = p.H_q != p.H;",
        "const bool automatic_equal_n4_g64 = "
        "!is_gva && hinted_equal_lengths && p.N == 4",
        "const bool automatic_gva_equal_n4_g16 = "
        "is_gva && hinted_equal_lengths && p.N == 4",
        "hinted_bound == 4096 && p.T_total == 4 * 4096 && "
        "group_env == nullptr && "
        "!force_direct && !force_affine && !force_hybrid;",
        "const bool hinted_n8_g64 = !is_gva && has_length_hint && "
        "p.N == 8 && hinted_bound >= 4096;",
        "requested_g16 || automatic_dense_g16 || "
        "automatic_gva_equal_n4_g16",
        "const bool automatic_gva_packed_nw4 = "
        "is_gva && is_varlen && !direct && !hybrid && "
        "p.N >= 4 && p.N <= 8 && group_env == nullptr && "
        "!force_direct && !force_affine && !force_hybrid;",
        "return {force_context, group_chunks, "
        "hybrid ? kHybridDirectMaxChunks : 0, "
        "automatic_gva_packed_nw4, automatic_gva_equal_n4_g16};",
        "const bool automatic_gva_packed_nw4 = "
        "a.automatic_gva_packed_nw4 && a.is_gva && a.is_varlen && "
        "!direct && !hybrid;",
        "const bool automatic_gva_equal_n4_g16 = "
        "a.automatic_gva_equal_n4_g16 && "
        "automatic_gva_packed_nw4 && group_chunks == 16;",
        "const bool cache_context_operands = "
        "a.context_operands_cached && !a.is_gva;",
        "const bool automatic_gva_disable_b_stream = "
        "automatic_gva_equal_n4_g16 && automatic_gva_scan_nw4 && "
        "scan_b_stream_env == nullptr;",
        "const bool scan_b_stream = !automatic_gva_disable_b_stream && "
        "context_scan_b_stream_enabled(group_chunks);",
    )
    for contract in route_contracts:
        if contract not in compact_policy:
            raise AssertionError("GVA context-route contract changed: " + contract)

    # Metadata-eliding/cache-heavy candidates are deliberately equal-head only.
    # This list is intentionally structural: adding GVA to one of these routes
    # requires a matched producer/consumer proof and a validator update.
    equal_head_only_contracts = (
        "return p.H_q == p.H && p.cu_seqlens != nullptr",
        "if (p.H_q != p.H || p.cu_seqlens == nullptr ||",
        "if (!a.is_gva && a.context_persistent_blocks > 0",
        "const bool direct_dense_n1_h12 = !a.is_gva &&",
        "const bool direct_global_n1_h12 = !a.is_gva &&",
        "const bool direct_ksplit_eligible = !a.is_gva &&",
    )
    for contract in equal_head_only_contracts:
        if contract not in compact_policy:
            raise AssertionError(
                "equal-head-only specialization lost its GVA exclusion: "
                + contract
            )
    common_equal_head_only = (
        "!is_gva && policy.context_equal_dense_n4_g64",
        "use_context_parallel && !is_gva && is_varlen && "
        "!use_context_equal_dense_n4_g64 && "
        "policy.context_direct_prefixless",
        "use_context_parallel && !is_gva && is_varlen && "
        "policy.launch_context_prefix != nullptr",
    )
    for contract in common_equal_head_only:
        if contract not in compact_common:
            raise AssertionError(
                "common equal-head-only graph lost its GVA exclusion: "
                + contract
            )

    def resolve_gva_case(
        *,
        sequences: int,
        tokens_per_sequence: int,
        packed: bool,
        bound: int = 0,
        route: str | None = None,
        group: str | None = None,
        scan_nw: str | None = None,
        scan_ksplit: str | None = None,
        scan_a_gll: str | None = None,
        scan_b_phased: str | None = None,
        scan_b_stream: str | None = None,
    ) -> tuple[int, bool, int, bool, bool]:
        """Model only the automatic GVA facts consumed across policy stages."""

        force_direct = route == "direct"
        force_affine = route == "affine"
        force_hybrid = route == "hybrid"
        hinted_equal = packed and bound > 0 and bound == tokens_per_sequence
        automatic_n4_g16 = (
            hinted_equal
            and sequences == 4
            and bound == 4096
            and tokens_per_sequence == 4096
            and group is None
            and not force_direct
            and not force_affine
            and not force_hybrid
        )
        hybrid = packed and force_hybrid
        direct = not hybrid and force_direct
        if direct:
            group_chunks = 0
        elif automatic_n4_g16:
            group_chunks = 16
        elif group in {"32", "64", "128"}:
            group_chunks = int(group)
        else:
            group_chunks = 64 if hybrid or tokens_per_sequence >= 12288 else 32
        automatic_packed_nw4 = (
            packed
            and not direct
            and not hybrid
            and 4 <= sequences <= 8
            and group is None
            and not force_direct
            and not force_affine
            and not force_hybrid
        )
        automatic_scan_nw4 = (
            automatic_packed_nw4
            and scan_nw is None
            and scan_ksplit is None
            and scan_a_gll is None
            and scan_b_phased is None
        )
        resolved_scan_nw = (
            int(scan_nw)
            if scan_nw is not None
            else 4 if automatic_scan_nw4 else 2
        )
        automatic_disable_b_stream = (
            automatic_n4_g16
            and automatic_packed_nw4
            and automatic_scan_nw4
            and group_chunks == 16
            and scan_b_stream is None
        )
        parsed_b_stream = (
            scan_b_stream == "1"
            if scan_b_stream is not None
            else group_chunks in (8, 16)
        )
        resolved_b_stream = (
            not automatic_disable_b_stream and parsed_b_stream
        )
        return (
            group_chunks,
            automatic_packed_nw4,
            resolved_scan_nw,
            resolved_b_stream,
            automatic_n4_g16,
        )

    # Packed N=1 is normalized to dense before policy construction.  It must
    # retain the established NW2 scan even when the deep-single context route
    # is selected.  The exact 4x4K resume bucket is the G16/NW4/no-B-stream
    # production recipe; N=8 long/ragged stays G32 rather than inheriting the
    # equal-head hinted-G64 graduation.
    cases = (
        (
            "single-8k",
            dict(sequences=1, tokens_per_sequence=8192, packed=False),
            (32, False, 2, False, False),
        ),
        (
            "resume-4x4k",
            dict(
                sequences=4,
                tokens_per_sequence=4096,
                packed=True,
                bound=4096,
            ),
            (16, True, 4, False, True),
        ),
        (
            "long-ragged-n8",
            dict(
                sequences=8,
                tokens_per_sequence=4096,
                packed=True,
                bound=8192,
            ),
            (32, True, 4, False, False),
        ),
    )
    for label, configuration, expected in cases:
        actual = resolve_gva_case(**configuration)
        if actual != expected:
            raise AssertionError(
                f"static GVA {label} route changed: {actual} != {expected}"
            )

    base_n4 = dict(
        sequences=4,
        tokens_per_sequence=4096,
        packed=True,
        bound=4096,
    )
    explicit_routes = {
        "direct": (0, False, 2, False, False),
        "affine": (32, False, 2, False, False),
        "hybrid": (64, False, 2, False, False),
    }
    for route, expected in explicit_routes.items():
        actual = resolve_gva_case(**base_n4, route=route)
        if actual != expected:
            raise AssertionError(
                f"explicit GVA {route} route did not override auto N4: "
                f"{actual} != {expected}"
            )
    for group in ("32", "64", "128"):
        expected = (int(group), False, 2, False, False)
        actual = resolve_gva_case(**base_n4, group=group)
        if actual != expected:
            raise AssertionError(
                f"explicit GVA G{group} did not override auto G16: "
                f"{actual} != {expected}"
            )

    # Any explicit scan-family value, including a rollback "0", suppresses the
    # automatic NW4 fact.  Explicit SCAN_NW remains authoritative; on G16 its
    # explicit path restores the established default streamed-b decision.
    for explicit_nw in ("1", "2", "4"):
        expected = (16, True, int(explicit_nw), True, True)
        actual = resolve_gva_case(**base_n4, scan_nw=explicit_nw)
        if actual != expected:
            raise AssertionError(
                f"explicit GVA scan NW{explicit_nw} lost precedence: "
                f"{actual} != {expected}"
            )
    for axis in ("scan_ksplit", "scan_a_gll", "scan_b_phased"):
        for value in ("0", "1"):
            actual = resolve_gva_case(**base_n4, **{axis: value})
            expected = (16, True, 2, True, True)
            if actual != expected:
                raise AssertionError(
                    "explicit GVA scan-family fallback changed for "
                    f"{axis}={value}: {actual} != {expected}"
                )
    if resolve_gva_case(**base_n4, scan_b_stream="1") != (
        16,
        True,
        4,
        True,
        True,
    ):
        raise AssertionError("explicit GVA B_STREAM=1 did not override auto-off")
    if resolve_gva_case(**base_n4, scan_b_stream="0") != (
        16,
        True,
        4,
        False,
        True,
    ):
        raise AssertionError("explicit GVA B_STREAM=0 did not retain auto-off")

    print(
        "PASS static GVA whole-route capability handshake, grouped K1 mapping, "
        "equal-head specialization exclusions, single/N4/N8 route facts, "
        "automatic NW4/B-stream recipe, and explicit route/group/scan precedence"
    )


def captured_graph_kernel_names(
    graph: torch.cuda.CUDAGraph, device: torch.device
) -> list[str]:
    """Return raw HIP kernel symbols from one retained graph."""

    import ctypes

    class Dim3(ctypes.Structure):
        _fields_ = (
            ("x", ctypes.c_uint),
            ("y", ctypes.c_uint),
            ("z", ctypes.c_uint),
        )

    class HipKernelNodeParams(ctypes.Structure):
        _fields_ = (
            ("block_dim", Dim3),
            ("extra", ctypes.POINTER(ctypes.c_void_p)),
            ("func", ctypes.c_void_p),
            ("grid_dim", Dim3),
            ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
            ("shared_mem_bytes", ctypes.c_uint),
        )

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGraphGetNodes.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    )
    hip.hipGraphGetNodes.restype = ctypes.c_int
    hip.hipGraphNodeGetType.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )
    hip.hipGraphNodeGetType.restype = ctypes.c_int
    hip.hipGraphKernelNodeGetParams.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(HipKernelNodeParams),
    )
    hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
    hip.hipKernelNameRefByPtr.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

    def checked(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with HIP status {status}")

    graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    checked(
        hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
        "hipGraphGetNodes(count)",
    )
    nodes = (ctypes.c_void_p * count.value)()
    checked(
        hip.hipGraphGetNodes(graph_handle, nodes, ctypes.byref(count)),
        "hipGraphGetNodes(nodes)",
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    names = []
    for node in nodes[: count.value]:
        node_type = ctypes.c_int()
        checked(
            hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
            "hipGraphNodeGetType",
        )
        if node_type.value != 0:  # hipGraphNodeTypeKernel
            continue
        params = HipKernelNodeParams()
        checked(
            hip.hipGraphKernelNodeGetParams(node, ctypes.byref(params)),
            "hipGraphKernelNodeGetParams",
        )
        raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
        if raw_name is not None:
            names.append(raw_name.decode(errors="replace"))
    return names


_CONTEXT_PIPELINE_SYMBOL = re.compile(
    r"k2_kda_context_parallel_nw4_kernelI"
    r"Li(?P<group>[0-9]+)E"
    r"LNS[0-9]*_14KdaContextModeE(?P<mode>[0-2])E"
    r"Lb(?P<ho>[01])ELb(?P<fp>[01])E"
    r"Lb(?P<vl>[01])ELb(?P<direct>[01])E"
    r"Li(?P<nw>[0-9]+)ELi[0-9]+E"
    r"Lb(?P<cached>[01])E"
    r"Lb(?P<u_forward>[01])E"
    r"Lb(?P<v_forward>[01])E"
    r"Lb(?P<lds_pipeline>[01])E"
    r"Lb(?P<tail_first>[01])E"
    r"Lb(?P<prefixless>[01])E"
)

_K1_FUSED_SYMBOL = re.compile(
    # Flags ten through thirteen are dense-N1/full-C16, padded solve, early
    # beta, and GVA.  Keep decoding older nine-to-twelve-flag artifacts so retained
    # pre-specialization modules can still be audited during A/B rollback.
    r"k1_kda_bt16_fused_kernelI(?P<flags>(?:Lb[01]E){9,13})"
)


def decode_context_pipeline_roles(
    kernel_names: list[str],
) -> dict[str, dict[str, object]]:
    """Decode hybrid context mode/transport/scheduling template bits."""

    roles: dict[str, dict[str, object]] = {}
    for name in kernel_names:
        match = _CONTEXT_PIPELINE_SYMBOL.search(name)
        if match is None:
            continue
        group = int(match.group("group"))
        mode = int(match.group("mode"))
        if mode == 0:
            role = "affine_b"
        elif mode == 1:
            role = "affine_a"
        elif group == 1:
            role = "hybrid_direct_replay"
        else:
            role = "affine_replay"
        if role in roles:
            raise AssertionError(
                f"captured duplicate context role {role}: {name}"
            )
        roles[role] = {
            "name": name,
            "normalized_name": (
                name[: match.start("lds_pipeline")]
                + name[match.end("lds_pipeline") :]
            ),
            "cached": int(match.group("cached")),
            "u_forward": int(match.group("u_forward")),
            "v_forward": int(match.group("v_forward")),
            "lds_pipeline": int(match.group("lds_pipeline")),
            "tail_first": int(match.group("tail_first")),
            "nw": int(match.group("nw")),
        }
    expected_roles = {
        "hybrid_direct_replay",
        "affine_b",
        "affine_a",
        "affine_replay",
    }
    if set(roles) != expected_roles:
        raise AssertionError(
            "captured hybrid graph did not contain exactly four context "
            f"launch roles: got={roles}, all kernels={kernel_names!r}"
        )
    return roles


def public_call(x: dict[str, Any]):
    return flash_kda.flash_kda_fwd(
        q=x["q"],
        k=x["k"],
        v=x["v"],
        g=x["g"],
        beta=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        scale=x["scale"],
        initial_state=x["initial_state"] if x["has_initial_state"] else None,
        output_final_state=x["output_final_state"],
        lower_bound=x["lower_bound"],
        cu_seqlens=x["cu_seqlens"],
    )


def rejection_reason(x: dict[str, Any]):
    return flash_kda._native_rejection_reason(
        q=x["q"],
        k=x["k"],
        v=x["v"],
        g=x["g"],
        beta=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        initial_state=x["initial_state"] if x["has_initial_state"] else None,
        output_final_state=x["output_final_state"],
        lower_bound=x["lower_bound"],
        state_v_first=True,
        cu_seqlens=x["cu_seqlens"],
    )


def allocate(x: dict[str, Any]):
    batch, tokens, _, _ = x["q"].shape
    value_heads = x["v"].shape[2]
    out = torch.empty_like(x["v"])
    final = torch.empty(
        x["N"],
        value_heads,
        128,
        128,
        device=x["q"].device,
        dtype=x["state_dtype"],
    )
    workspace = torch.empty(
        flash_kda.flash_kda_workspace_size(
            batch * tokens, value_heads, x["N"]
        ),
        device=x["q"].device,
        dtype=torch.uint8,
    )
    return out, final, workspace


def _raw_common_args(
    x: dict[str, Any], out, final, workspace
) -> tuple[Any, ...]:
    """Construct the stable 25-argument prefix shared by raw-v1/v2/v3."""

    B, T, _, _ = x["q"].shape
    H = x["v"].shape[2]
    device = x["q"].device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    args = (
        x["q"].data_ptr(),
        x["k"].data_ptr(),
        x["v"].data_ptr(),
        x["g"].data_ptr(),
        x["beta"].data_ptr(),
        out.data_ptr(),
        workspace.data_ptr(),
        x["A_log"].data_ptr(),
        x["dt_bias"].data_ptr(),
        x["initial_state"].data_ptr() if x["has_initial_state"] else 0,
        final.data_ptr() if x["output_final_state"] else 0,
        x["cu_seqlens"].data_ptr() if x["cu_seqlens"] is not None else 0,
        B,
        T,
        H,
        x["N"],
        workspace.nbytes,
        x["scale"],
        x["lower_bound"],
        x["has_initial_state"],
        x["output_final_state"],
        x["is_varlen"],
        (x["has_initial_state"] or x["output_final_state"])
        and x["state_dtype"] == torch.float32,
        device_index,
        torch.cuda.current_stream(device).cuda_stream,
    )
    if len(args) != len(_RAW_V1_ABI_ARGUMENTS):
        raise AssertionError(
            "validator constructed an invalid raw ABI prefix: "
            f"expected {len(_RAW_V1_ABI_ARGUMENTS)} arguments, got {len(args)}"
        )
    return args


def raw_args(x: dict[str, Any], out, final, workspace) -> tuple[Any, ...]:
    """Construct the equal-head-only raw-v1/v2 argument prefix."""

    q_heads = x["q"].shape[2]
    value_heads = x["v"].shape[2]
    if q_heads != value_heads:
        raise ValueError(
            "raw-v1/v2 cannot represent GVA: "
            f"H_q={q_heads}, H_v={value_heads}"
        )
    return _raw_common_args(x, out, final, workspace)


def raw_call(module, x, out, final, workspace):
    reason = rejection_reason(x)
    if reason is not None:
        raise RuntimeError(
            f"Python admission unexpectedly rejected valid input: {reason}"
        )
    with torch.cuda.device(x["q"].device):
        module.flash_kda_fwd_hip_raw(*raw_args(x, out, final, workspace))


def raw_v2_call(
    module,
    x,
    out,
    final,
    workspace,
    max_seqlen_upper_bound: int,
):
    """Call the additive 26-argument raw ABI with a host route hint."""

    reason = rejection_reason(x)
    if reason is not None:
        raise RuntimeError(
            f"Python admission unexpectedly rejected valid input: {reason}"
        )
    if type(max_seqlen_upper_bound) is not int:
        raise TypeError(
            "validator raw-v2 bound must be a Python int, got "
            f"{max_seqlen_upper_bound!r}"
        )
    args = (*raw_args(x, out, final, workspace), max_seqlen_upper_bound)
    if len(args) != len(_RAW_V2_ABI_ARGUMENTS):
        raise AssertionError(
            "validator constructed an invalid raw-v2 tuple: "
            f"expected {len(_RAW_V2_ABI_ARGUMENTS)} arguments, got {len(args)}"
        )
    with torch.cuda.device(x["q"].device):
        module.flash_kda_fwd_hip_raw_v2(*args)


def raw_v3_call(
    module,
    x,
    out,
    final,
    workspace,
    max_seqlen_upper_bound: int,
):
    """Call the 27-argument GVA ABI with value and Q/K heads separated."""

    reason = rejection_reason(x)
    if reason is not None:
        raise RuntimeError(
            f"Python admission unexpectedly rejected valid input: {reason}"
        )
    if type(max_seqlen_upper_bound) is not int:
        raise TypeError(
            "validator raw-v3 bound must be a Python int, got "
            f"{max_seqlen_upper_bound!r}"
        )
    q_heads = x["q"].shape[2]
    args = (
        *_raw_common_args(x, out, final, workspace),
        max_seqlen_upper_bound,
        q_heads,
    )
    if len(args) != len(_RAW_V3_ABI_ARGUMENTS):
        raise AssertionError(
            "validator constructed an invalid raw-v3 tuple: "
            f"expected {len(_RAW_V3_ABI_ARGUMENTS)} arguments, got {len(args)}"
        )
    with torch.cuda.device(x["q"].device):
        module.flash_kda_fwd_hip_raw_v3(*args)


def _documented_pybind_arguments(operation, symbol: str) -> tuple[str, ...]:
    """Extract named arguments from pybind's runtime signature metadata."""

    documentation = operation.__doc__ or ""
    prefix = f"{symbol}("
    signature = next(
        (
            line.strip()
            for line in documentation.splitlines()
            if line.strip().startswith(prefix)
        ),
        None,
    )
    if signature is None:
        raise AssertionError(f"{symbol} does not expose a pybind signature")
    closing = signature.rfind(")")
    if closing < len(prefix):
        raise AssertionError(f"cannot parse {symbol} signature: {signature!r}")
    parameter_text = signature[len(prefix) : closing]
    return tuple(
        re.findall(
            r"(?:^|, )([A-Za-z_][A-Za-z0-9_]*):",
            parameter_text,
        )
    )


def check_module_abi_surface(module) -> None:
    """Require descriptor and all three additive raw ABI generations."""

    expected = (
        ("flash_kda_fwd_hip", _DESCRIPTOR_ABI_ARGUMENTS),
        ("flash_kda_fwd_hip_raw", _RAW_V1_ABI_ARGUMENTS),
        ("flash_kda_fwd_hip_raw_v2", _RAW_V2_ABI_ARGUMENTS),
        ("flash_kda_fwd_hip_raw_v3", _RAW_V3_ABI_ARGUMENTS),
    )
    for symbol, expected_arguments in expected:
        operation = getattr(module, symbol, None)
        if not callable(operation):
            raise RuntimeError(
                f"module_flash_kda_hip is missing callable {symbol}"
            )
        documented_arguments = _documented_pybind_arguments(
            operation, symbol
        )
        if documented_arguments != expected_arguments:
            raise AssertionError(
                f"{symbol} ABI changed: expected "
                f"{len(expected_arguments)} arguments {expected_arguments!r}, "
                f"got {len(documented_arguments)} {documented_arguments!r}"
            )

    binding = flash_kda._get_raw_pointer_binding()
    if binding is None or binding[1] != 3:
        raise RuntimeError(
            "public FlashKDA adapter did not prefer the available raw-v3 ABI"
        )
    print(
        "PASS module ABI surface: descriptor=17, raw-v1=25, "
        "raw-v2=26, raw-v3=27"
    )


def descriptor_call(x, out, final, workspace):
    """Call the decorated tensor-descriptor ABI without public raw dispatch."""

    reason = rejection_reason(x)
    if reason is not None:
        raise RuntimeError(
            f"Python admission unexpectedly rejected valid input: {reason}"
        )
    initial_arg = x["initial_state"] if x["has_initial_state"] else x["q"]
    final_arg = final if x["output_final_state"] else initial_arg
    cu_arg = x["cu_seqlens"] if x["cu_seqlens"] is not None else x["q"]
    flash_kda.flash_kda_fwd_hip(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        out,
        workspace,
        x["A_log"],
        x["dt_bias"],
        initial_arg,
        final_arg,
        cu_arg,
        x["scale"],
        x["lower_bound"],
        x["has_initial_state"],
        x["output_final_state"],
        x["is_varlen"],
    )


def assert_same(actual, reference, label: str):
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=label)


def assert_bitwise_same(actual, reference, label: str):
    """Compare floating tensors by storage bits, including signed zero/NaNs."""

    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError(
            f"{label}: shape/dtype mismatch: "
            f"{actual.shape}/{actual.dtype} vs "
            f"{reference.shape}/{reference.dtype}"
        )
    integer_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }.get(actual.dtype)
    if integer_dtype is None:
        raise TypeError(f"{label}: unsupported bitwise dtype {actual.dtype}")
    actual_bits = actual.contiguous().view(integer_dtype)
    reference_bits = reference.contiguous().view(integer_dtype)
    if not torch.equal(actual_bits, reference_bits):
        mismatches = int((actual_bits != reference_bits).sum().item())
        raise AssertionError(
            f"{label}: {mismatches}/{actual_bits.numel()} raw elements differ"
        )


def seed_empty_state_bit_patterns(x, seq_lens, device):
    """Poison empty input-state slabs with conversion-sensitive raw bits."""

    if not x["has_initial_state"]:
        return
    if x["state_dtype"] == torch.float32:
        integer_dtype = torch.int32
        unsigned_patterns = (
            0x00000000,  # +0
            0x80000000,  # -0
            0x00000001,  # minimum subnormal
            0x007FFFFF,  # maximum subnormal
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
            0x7FC00001,  # quiet NaN, payload 1
            0x7FC01234,  # quiet NaN, distinct payload
        )
        modulus = 1 << 32
        sign_bit = 1 << 31
    else:
        integer_dtype = torch.int16
        unsigned_patterns = (
            0x0000,  # +0
            0x8000,  # -0
            0x0001,  # minimum BF16 subnormal
            0x007F,  # maximum BF16 subnormal
            0x7F80,  # +Inf
            0xFF80,  # -Inf
            0x7FC1,  # quiet NaN, payload 1
            0x7FE5,  # quiet NaN, distinct payload
        )
        modulus = 1 << 16
        sign_bit = 1 << 15
    signed_patterns = tuple(
        value if value < sign_bit else value - modulus
        for value in unsigned_patterns
    )
    pattern = torch.tensor(
        signed_patterns, device=device, dtype=integer_dtype
    )
    for sequence, length in enumerate(seq_lens):
        if length != 0:
            continue
        slab = x["initial_state"][sequence].view(integer_dtype).reshape(-1)
        repeats = (slab.numel() + pattern.numel() - 1) // pattern.numel()
        slab.copy_(pattern.repeat(repeats)[: slab.numel()])


def preallocated_call(abi: str, module, x, out, final, workspace):
    if abi == "raw":
        raw_call(module, x, out, final, workspace)
    elif abi == "descriptor":
        descriptor_call(x, out, final, workspace)
    else:
        raise ValueError(f"unknown ABI: {abi}")


def check_raw_vs_descriptor(module, x, label: str):
    device = x["q"].device
    initial_copy = x["initial_state"].clone()
    descriptor_out, descriptor_final, descriptor_workspace = allocate(x)
    raw_out, raw_final, raw_workspace = allocate(x)
    with torch.cuda.device(device):
        descriptor_call(x, descriptor_out, descriptor_final, descriptor_workspace)
        raw_call(module, x, raw_out, raw_final, raw_workspace)
        torch.cuda.synchronize(device)

    assert_same(raw_out, descriptor_out, f"{label}: raw/descriptor output mismatch")
    if x["output_final_state"]:
        assert_same(
            raw_final,
            descriptor_final,
            f"{label}: raw/descriptor final-state mismatch",
        )
    assert_same(x["initial_state"], initial_copy, f"{label}: initial state mutated")
    print(f"PASS raw vs descriptor bitwise: {label}")
    return x, descriptor_out, descriptor_final if x["output_final_state"] else None


def check_raw_v3_gva_vs_descriptor(module, device: torch.device) -> None:
    """Exercise raw-v3 GVA for both packed and dense layouts, including ratio 4."""

    cases = (
        ("packed-ratio2", (17, 31), True, 2, 4),
        ("dense-ratio4", (33, 33), False, 1, 4),
    )
    for label, seq_lens, packed, q_heads, value_heads in cases:
        x = make_inputs(
            seq_lens,
            q_heads,
            device,
            value_heads=value_heads,
            packed=packed,
            has_initial_state=True,
            output_final_state=True,
            seed=20260827 + value_heads,
        )
        initial_copy = x["initial_state"].clone()
        descriptor_out, descriptor_final, descriptor_workspace = allocate(x)
        raw_out, raw_final, raw_workspace = allocate(x)
        descriptor_call(x, descriptor_out, descriptor_final, descriptor_workspace)
        # Bound zero preserves the descriptor route and isolates the appended
        # H_q ABI from intentional policy differences caused by a host hint.
        raw_v3_call(module, x, raw_out, raw_final, raw_workspace, 0)
        torch.cuda.synchronize(device)

        assert_same(
            raw_out,
            descriptor_out,
            f"raw-v3 GVA {label} output mismatch",
        )
        assert_same(
            raw_final,
            descriptor_final,
            f"raw-v3 GVA {label} final-state mismatch",
        )
        assert_same(
            x["initial_state"],
            initial_copy,
            f"raw-v3 GVA {label} mutated initial state",
        )
        try:
            raw_args(x, raw_out, raw_final, raw_workspace)
        except ValueError as error:
            if "raw-v1/v2 cannot represent GVA" not in str(error):
                raise
        else:
            raise AssertionError(
                f"legacy raw ABI unexpectedly represented GVA {label}"
            )
        print(f"PASS raw-v3/descriptor GVA bitwise: {label}")


def check_state_layout_matrix(module, device: torch.device, heads: int):
    """Cover all valid HI/HO modes and both state dtypes for both layouts."""

    state_modes = [
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    ]
    layouts = [
        # N=16, short packed sequences select gfx950's direct context route.
        ("packed", (65,) * 16, True),
        # T=257 selects the plain C-split route while retaining dense indexing.
        ("dense", (257, 257), False),
    ]
    for layout_name, seq_lens, packed in layouts:
        for (
            mode_name,
            has_initial_state,
            output_final_state,
            state_dtype,
        ) in state_modes:
            x = make_inputs(
                seq_lens,
                heads,
                device,
                packed=packed,
                state_dtype=state_dtype,
                has_initial_state=has_initial_state,
                output_final_state=output_final_state,
            )
            check_raw_vs_descriptor(module, x, f"{layout_name}/{mode_name}")


def check_forced_hybrid_route(module, device: torch.device, heads: int):
    """Prove the explicit hybrid knob reaches context, not plain C-split.

    Packed (1024, 1025) is deliberately outside the automatic context policy
    while straddling the hybrid direct/affine boundary at 64/65 C16 chunks.
    K1 skip knobs affect only C-split.  Therefore two forced-hybrid launches,
    one with those knobs set and both with poisoned workspaces, must be
    bitwise identical if and only if the final policy really selects context.
    This catches a route regression without adding a production debug ABI.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP forced-hybrid route assertion: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    x = make_inputs((1024, 1025), heads, device, packed=True)
    initial_copy = x["initial_state"].clone()

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def run(*, poison_plain_k1: bool):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_HYBRID"] = "1"
        if poison_plain_k1:
            os.environ["FLASH_KDA_CS_SKIP_K1_PREP"] = "1"
            os.environ["FLASH_KDA_CS_SKIP_K1_SOLVE"] = "1"
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x7F)
        raw_call(module, x, out, final, workspace)
        return out, final

    try:
        ordinary = run(poison_plain_k1=False)
        skip_poisoned = run(poison_plain_k1=True)
        torch.cuda.synchronize(device)
        assert_same(
            skip_poisoned[0], ordinary[0],
            "forced hybrid was overridden by the plain K1 skip route",
        )
        assert_same(
            skip_poisoned[1], ordinary[1],
            "forced hybrid state was overridden by the plain K1 skip route",
        )
        assert_same(
            x["initial_state"], initial_copy,
            "forced hybrid route assertion mutated initial state",
        )
        print("PASS forced hybrid selects context on packed 1024/1025")
    finally:
        restore_env()


def check_context_tight_scan_matrix(
    module, device: torch.device, heads: int
):
    """Prove the strict hybrid scan-grid specialization is bitwise exact.

    The matrix deliberately keeps ``context_upper < N`` for every launch, so
    ``TIGHT_SCAN=1`` cannot silently exercise the legacy N*H grid.  One compact
    workload covers all state modes across G32/G64/G128 and NW1/NW2/NW4;
    additional cases stress repeated filtered-prefix entries, leading/trailing
    empty sequences, the 1024/1025 direct/affine boundary, and an all-direct
    batch whose affine scan is empty.  Finally, a captured tight graph is
    replayed after changing legal cu_seqlens in place while preserving N/T.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context tight-scan A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def assert_tight_active(
        seq_lens: tuple[int, ...], group_chunks: int
    ) -> None:
        # Mirror policy.hpp's metadata-free conservative upper.  This turns a
        # future route/bound change into an explicit test failure instead of
        # allowing OFF/ON to become a silent comparison of the same kernel.
        sequences = len(seq_lens)
        total_tiles = (sum(seq_lens) + 15) // 16 + sequences
        max_affine_sequences = min(sequences, total_tiles // 65)
        context_upper = max(
            1,
            (
                total_tiles
                + max_affine_sequences * (group_chunks - 1)
            )
            // group_chunks,
        )
        if sequences < 9 or context_upper >= sequences:
            raise AssertionError(
                "tight-scan test case is axis-inactive: "
                f"N={sequences}, total_tiles={total_tiles}, "
                f"G={group_chunks}, context_upper={context_upper}"
            )

    def assert_captured_scan_specialization(
        graph: torch.cuda.CUDAGraph, *, expected_tight: bool
    ) -> None:
        """Inspect HIP graph nodes so OFF/ON cannot dispatch the same kernel."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(f"{operation} failed with HIP status {status}")

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )

        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        kernel_names = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:  # hipGraphNodeTypeKernel
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            raw_text = raw_name.decode(errors="replace")
            demangled = raw_text
            try:
                demangled = torch._C._demangle(raw_text)
            except (AttributeError, RuntimeError):
                pass
            name = (
                raw_text
                if demangled == raw_text
                else f"{raw_text}\n{demangled}"
            )
            kernel_names.append(name)

        scan_names = [
            name
            for name in kernel_names
            if "k2_kda_context_affine_scan_nw4_kernel" in name
        ]
        tight_arg = str(expected_tight).lower()
        needle = (
            "k2_kda_context_affine_scan_nw4_kernel"
            f"<64,2,true,true,true,true,{tight_arg},"
        )
        mangled_args = (
            "ILi64ELi2ELb1ELb1ELb1ELb1"
            f"ELb{1 if expected_tight else 0}E"
        )
        compact_name = "".join(scan_names[0].split()) if scan_names else ""
        if len(scan_names) != 1 or (
            needle not in compact_name and mangled_args not in compact_name
        ):
            raise AssertionError(
                "captured graph did not contain the expected context scan "
                f"specialization {needle!r}; scan nodes={scan_names!r}"
            )

    def configure(group_chunks: int, scan_nw: int, tight: bool) -> None:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_HYBRID"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = (
            "1" if tight else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"

    def run(x, group_chunks: int, scan_nw: int, tight: bool):
        configure(group_chunks, scan_nw, tight)
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0xA5)
        raw_call(module, x, out, final, workspace)
        return out, final

    def compare(x, label: str, group_chunks: int, scan_nw: int):
        seq_prefix = x["cu_seqlens"].cpu().tolist()
        seq_lens = tuple(
            end - start for start, end in zip(seq_prefix, seq_prefix[1:])
        )
        assert_tight_active(seq_lens, group_chunks)
        initial_copy = x["initial_state"].clone()
        ordinary = run(x, group_chunks, scan_nw, False)
        tight = run(x, group_chunks, scan_nw, True)
        torch.cuda.synchronize(device)
        for mode_name, tensors in (("legacy", ordinary), ("tight", tight)):
            checked = (tensors[0], tensors[1]) if x["output_final_state"] else (
                tensors[0],
            )
            if any(not bool(torch.isfinite(tensor).all().item()) for tensor in checked):
                raise AssertionError(
                    f"tight scan {label}: {mode_name} produced non-finite data"
                )
        assert_same(
            tight[0], ordinary[0],
            f"tight scan {label}: output mismatch",
        )
        if x["output_final_state"]:
            assert_same(
                tight[1], ordinary[1],
                f"tight scan {label}: final-state mismatch",
            )
        assert_same(
            x["initial_state"], initial_copy,
            f"tight scan {label}: initial state mutated",
        )
        return ordinary

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    # The single 1025-token sequence is affine; repeated prefix entries on
    # either side come from direct/empty sequences and exercise binary search.
    matrix_lens = (1,) * 16 + (1025,) + (0,) * 16

    try:
        for group_chunks in (32, 64, 128):
            for scan_nw in (1, 2, 4):
                for (
                    mode_name,
                    has_initial_state,
                    output_final_state,
                    state_dtype,
                ) in state_modes:
                    x = make_inputs(
                        matrix_lens,
                        heads,
                        device,
                        packed=True,
                        state_dtype=state_dtype,
                        has_initial_state=has_initial_state,
                        output_final_state=output_final_state,
                    )
                    compare(
                        x,
                        f"G{group_chunks}/NW{scan_nw}/{mode_name}",
                        group_chunks,
                        scan_nw,
                    )

        topology_cases = (
            ("leading-long", (1025,) + (0,) * 32),
            ("middle-long", (0,) * 16 + (1025,) + (0,) * 16),
            ("trailing-long", (0,) * 32 + (1025,)),
            ("direct-1024", (1,) * 32 + (1024,)),
            ("affine-1025", (1,) * 32 + (1025,)),
            ("all-short", (0, 1) * 32),
            (
                "multiple-long-repeated-prefix",
                (0,) * 8 + (1025,) + (0,) * 8 + (2049,) + (0,) * 15,
            ),
        )
        for label, seq_lens in topology_cases:
            x = make_inputs(seq_lens, heads, device, packed=True)
            compare(x, label, 64, 2)

        # The following ATOM-shaped case has real affine work and a tight host
        # grid.  Exercise the exact specialization under graph replay and two
        # disjoint streams, not merely through an OFF/ON launch pair.
        graph_lens = (1025,) + (1,) * 32
        graph_x = make_inputs(graph_lens, heads, device, packed=True)
        graph_reference = compare(graph_x, "graph-base", 64, 2)

        # First capture the OFF route solely to make the legacy specialization
        # observable.  Without this graph-node assertion, a broken env parser
        # that always selected either route could still pass every exact A/B.
        configure(64, 2, False)
        off_out, off_final, off_workspace = allocate(graph_x)
        preallocated_call(
            "raw", module, graph_x, off_out, off_final, off_workspace
        )
        torch.cuda.synchronize(device)
        off_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(off_graph):
            preallocated_call(
                "raw", module, graph_x, off_out, off_final, off_workspace
            )
        assert_captured_scan_specialization(
            off_graph, expected_tight=False
        )
        off_graph.reset()

        configure(64, 2, True)
        check_preallocated_graph(
            "raw", module, graph_x, graph_reference[0], graph_reference[1]
        )
        check_preallocated_multistream(
            "raw", module, graph_x, graph_reference[0], graph_reference[1]
        )

        # Capture once with the long sequence first, then move the same 1025
        # tokens into the middle by changing only the persistent metadata.
        # N and total tokens stay fixed, which is the serving graph contract.
        out, final, workspace = allocate(graph_x)
        preallocated_call("raw", module, graph_x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            preallocated_call("raw", module, graph_x, out, final, workspace)
        assert_captured_scan_specialization(graph, expected_tight=True)
        graph.instantiate()

        replay_lens = (1,) * 16 + (1025,) + (1,) * 16
        offsets = [0]
        for length in replay_lens:
            offsets.append(offsets[-1] + length)
        graph_x["cu_seqlens"].copy_(
            torch.tensor(offsets, device=device, dtype=torch.int32)
        )
        changed_reference = run(graph_x, 64, 2, False)
        torch.cuda.synchronize(device)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x5A)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_same(
            out, changed_reference[0],
            "tight captured graph changed-prefix output mismatch",
        )
        assert_same(
            final, changed_reference[1],
            "tight captured graph changed-prefix state mismatch",
        )
        print(
            "PASS context tight-scan OFF/ON, state/G/NW/topology, "
            "graph-metadata replay and two-stream matrix"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_scan_b_stream_matrix(
    module, device: torch.device, heads: int
):
    """Prove the strict streamed-b scan candidate is bitwise exact.

    Dense and packed affine launches cover every state dispatch while the
    G32/NW1, G64/NW2, and G128/NW4 pairings instantiate all supported group
    and wave choices.  A compact hybrid case covers the tight VL mapping.
    Finally, captured kernel names distinguish the exact ``"1"`` opt-in from
    unset, ``"0"``, and representative noncanonical spellings.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context scan streamed-b A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        group_chunks: int,
        scan_nw: int,
        b_stream: str | None,
        *,
        hybrid: bool = False,
        tight: bool = False,
    ) -> None:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ[
            "FLASH_KDA_GFX950_CONTEXT_HYBRID"
            if hybrid
            else "FLASH_KDA_GFX950_CONTEXT_AFFINE"
        ] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = (
            "1" if tight else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = "0"
        if b_stream is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = b_stream

    def run(
        x,
        group_chunks: int,
        scan_nw: int,
        b_stream: str | None,
        *,
        hybrid: bool = False,
        tight: bool = False,
    ):
        configure(
            group_chunks,
            scan_nw,
            b_stream,
            hybrid=hybrid,
            tight=tight,
        )
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0xA5)
        raw_call(module, x, out, final, workspace)
        return out, final

    def compare(
        x,
        label: str,
        group_chunks: int,
        scan_nw: int,
        *,
        hybrid: bool = False,
        tight: bool = False,
    ):
        initial_copy = x["initial_state"].clone()
        established = run(
            x,
            group_chunks,
            scan_nw,
            "0",
            hybrid=hybrid,
            tight=tight,
        )
        streamed = run(
            x,
            group_chunks,
            scan_nw,
            "1",
            hybrid=hybrid,
            tight=tight,
        )
        torch.cuda.synchronize(device)
        for mode_name, tensors in (("established", established),
                                   ("streamed", streamed)):
            checked = tensors if x["output_final_state"] else tensors[:1]
            if any(
                not bool(torch.isfinite(tensor).all().item())
                for tensor in checked
            ):
                raise AssertionError(
                    f"scan streamed-b {label}: {mode_name} produced "
                    "non-finite data"
                )
        assert_same(
            streamed[0],
            established[0],
            f"scan streamed-b {label}: output mismatch",
        )
        if x["output_final_state"]:
            assert_same(
                streamed[1],
                established[1],
                f"scan streamed-b {label}: final-state mismatch",
            )
        assert_same(
            x["initial_state"],
            initial_copy,
            f"scan streamed-b {label}: initial state mutated",
        )
        return established

    def assert_captured_scan_kernel(
        x,
        b_stream: str | None,
        *,
        expected_streamed: bool,
    ) -> None:
        """Capture one launch and identify the selected scan specialization."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(64, 2, b_stream)
        out, final, workspace = allocate(x)
        preallocated_call("raw", module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            preallocated_call("raw", module, x, out, final, workspace)

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )

        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        kernel_names = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:  # hipGraphNodeTypeKernel
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            raw_text = raw_name.decode(errors="replace")
            demangled = raw_text
            try:
                demangled = torch._C._demangle(raw_text)
            except (AttributeError, RuntimeError):
                pass
            kernel_names.append(
                raw_text
                if demangled == raw_text
                else f"{raw_text}\n{demangled}"
            )

        scan_names = [
            name
            for name in kernel_names
            if "k2_kda_context_affine_scan" in name
        ]
        expected_name = (
            "k2_kda_context_affine_scan_b_stream_nw4_kernel"
            if expected_streamed
            else "k2_kda_context_affine_scan_nw4_kernel"
        )
        if len(scan_names) != 1 or expected_name not in scan_names[0]:
            raise AssertionError(
                "captured graph selected the wrong context scan for "
                f"B_STREAM={b_stream!r}; expected {expected_name!r}, "
                f"scan nodes={scan_names!r}"
            )
        graph.reset()

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    layouts = (
        ("dense", (2049, 2049), False),
        ("packed-vl", (0, 1025, 2049), True),
    )

    try:
        for scan_nw, group_chunks in ((1, 32), (2, 64), (4, 128)):
            for layout_name, seq_lens, packed in layouts:
                for (
                    mode_name,
                    has_initial_state,
                    output_final_state,
                    state_dtype,
                ) in state_modes:
                    x = make_inputs(
                        seq_lens,
                        heads,
                        device,
                        packed=packed,
                        state_dtype=state_dtype,
                        has_initial_state=has_initial_state,
                        output_final_state=output_final_state,
                    )
                    compare(
                        x,
                        f"{layout_name}/G{group_chunks}/NW{scan_nw}/"
                        f"{mode_name}",
                        group_chunks,
                        scan_nw,
                    )

        tight_lens = (1,) * 16 + (1025,) + (0,) * 16
        tight_total_tiles = (
            (sum(tight_lens) + 15) // 16 + len(tight_lens)
        )
        tight_max_affine_sequences = min(
            len(tight_lens), tight_total_tiles // 65
        )
        tight_context_upper = max(
            1,
            (
                tight_total_tiles
                + tight_max_affine_sequences * (64 - 1)
            )
            // 64,
        )
        if tight_context_upper >= len(tight_lens):
            raise AssertionError(
                "scan streamed-b tight test is axis-inactive: "
                f"N={len(tight_lens)}, context_upper={tight_context_upper}"
            )
        tight_x = make_inputs(
            tight_lens,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        compare(
            tight_x,
            "packed-tight/G64/NW2/inout-bf16",
            64,
            2,
            hybrid=True,
            tight=True,
        )

        parser_x = make_inputs(
            (2049,),
            heads,
            device,
            packed=False,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        parser_reference = run(parser_x, 64, 2, "0")
        for spelling, expected_streamed in (
            (None, False),
            ("", False),
            ("0", False),
            ("01", False),
            ("true", False),
            ("1 ", False),
            ("1", True),
        ):
            actual = run(parser_x, 64, 2, spelling)
            torch.cuda.synchronize(device)
            assert_same(
                actual[0],
                parser_reference[0],
                f"scan streamed-b parser {spelling!r}: output mismatch",
            )
            assert_same(
                actual[1],
                parser_reference[1],
                f"scan streamed-b parser {spelling!r}: state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                spelling,
                expected_streamed=expected_streamed,
            )
        print(
            "PASS context scan streamed-b OFF/ON, dense/VL/state/G/NW, "
            "tight mapping, and canonical parser matrix"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_scan_a_gll_matrix(
    module, device: torch.device, heads: int
):
    """Prove the NW2 streamed-b A-GLL candidate is bitwise exact.

    Dense, packed-VL, and strict tight-grid launches cover every state mode
    across G32/G64/G128.  Captured graph kernel names prove that only the exact
    ``"1"`` spelling selects A-GLL, that NW1/NW4 retain the established
    streamed-b kernel, and that A-GLL cannot enable streamed-b by itself.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context scan A-GLL A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        group_chunks: int,
        scan_nw: int,
        a_gll: str | None,
        *,
        b_stream: str | None = "1",
        hybrid: bool = False,
        tight: bool = False,
    ) -> None:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ[
            "FLASH_KDA_GFX950_CONTEXT_HYBRID"
            if hybrid
            else "FLASH_KDA_GFX950_CONTEXT_AFFINE"
        ] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = (
            "1" if tight else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = "0"
        if b_stream is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = b_stream
        if a_gll is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL"] = a_gll

    def run(
        x,
        group_chunks: int,
        scan_nw: int,
        a_gll: str | None,
        *,
        b_stream: str | None = "1",
        hybrid: bool = False,
        tight: bool = False,
    ):
        configure(
            group_chunks,
            scan_nw,
            a_gll,
            b_stream=b_stream,
            hybrid=hybrid,
            tight=tight,
        )
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0xA5)
        raw_call(module, x, out, final, workspace)
        return out, final

    def compare(
        x,
        label: str,
        group_chunks: int,
        *,
        hybrid: bool = False,
        tight: bool = False,
    ):
        initial_copy = x["initial_state"].clone()
        established = run(
            x,
            group_chunks,
            2,
            "0",
            hybrid=hybrid,
            tight=tight,
        )
        a_gll = run(
            x,
            group_chunks,
            2,
            "1",
            hybrid=hybrid,
            tight=tight,
        )
        torch.cuda.synchronize(device)
        for mode_name, tensors in (("established", established),
                                   ("A-GLL", a_gll)):
            checked = tensors if x["output_final_state"] else tensors[:1]
            if any(
                not bool(torch.isfinite(tensor).all().item())
                for tensor in checked
            ):
                raise AssertionError(
                    f"scan A-GLL {label}: {mode_name} produced non-finite data"
                )
        assert_same(
            a_gll[0],
            established[0],
            f"scan A-GLL {label}: output mismatch",
        )
        if x["output_final_state"]:
            assert_same(
                a_gll[1],
                established[1],
                f"scan A-GLL {label}: final-state mismatch",
            )
        assert_same(
            x["initial_state"],
            initial_copy,
            f"scan A-GLL {label}: initial state mutated",
        )

    def assert_captured_scan_kernel(
        x,
        scan_nw: int,
        a_gll: str | None,
        *,
        expected_name: str,
        b_stream: str | None = "1",
    ) -> None:
        """Capture one launch and identify the selected scan specialization."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(
            64,
            scan_nw,
            a_gll,
            b_stream=b_stream,
        )
        out, final, workspace = allocate(x)
        preallocated_call("raw", module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            preallocated_call("raw", module, x, out, final, workspace)

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )

        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        kernel_names = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:  # hipGraphNodeTypeKernel
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            raw_text = raw_name.decode(errors="replace")
            demangled = raw_text
            try:
                demangled = torch._C._demangle(raw_text)
            except (AttributeError, RuntimeError):
                pass
            kernel_names.append(
                raw_text
                if demangled == raw_text
                else f"{raw_text}\n{demangled}"
            )

        scan_names = [
            name
            for name in kernel_names
            if "k2_kda_context_affine_scan" in name
        ]
        if len(scan_names) != 1 or expected_name not in scan_names[0]:
            raise AssertionError(
                "captured graph selected the wrong context scan for "
                f"B_STREAM={b_stream!r}, NW={scan_nw}, "
                f"A_GLL={a_gll!r}; expected {expected_name!r}, "
                f"scan nodes={scan_names!r}"
            )
        graph.reset()

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    tight_lens = (1,) * 16 + (1025,) + (0,) * 16
    layouts = (
        ("dense", (2049, 2049), False, False, False),
        ("packed-vl", (0, 1025, 2049), True, False, False),
        ("packed-tight", tight_lens, True, True, True),
    )
    a_gll_name = "k2_kda_context_affine_scan_b_stream_a_gll_nw2_kernel"
    b_stream_name = "k2_kda_context_affine_scan_b_stream_nw4_kernel"
    legacy_name = "k2_kda_context_affine_scan_nw4_kernel"

    try:
        for group_chunks in (32, 64, 128):
            tight_total_tiles = (
                (sum(tight_lens) + 15) // 16 + len(tight_lens)
            )
            tight_max_affine_sequences = min(
                len(tight_lens), tight_total_tiles // 65
            )
            tight_context_upper = max(
                1,
                (
                    tight_total_tiles
                    + tight_max_affine_sequences * (group_chunks - 1)
                )
                // group_chunks,
            )
            if tight_context_upper >= len(tight_lens):
                raise AssertionError(
                    "scan A-GLL tight test is axis-inactive: "
                    f"N={len(tight_lens)}, G={group_chunks}, "
                    f"context_upper={tight_context_upper}"
                )

            for (
                layout_name,
                seq_lens,
                packed,
                hybrid,
                tight,
            ) in layouts:
                for (
                    mode_name,
                    has_initial_state,
                    output_final_state,
                    state_dtype,
                ) in state_modes:
                    x = make_inputs(
                        seq_lens,
                        heads,
                        device,
                        packed=packed,
                        state_dtype=state_dtype,
                        has_initial_state=has_initial_state,
                        output_final_state=output_final_state,
                    )
                    compare(
                        x,
                        f"{layout_name}/G{group_chunks}/NW2/{mode_name}",
                        group_chunks,
                        hybrid=hybrid,
                        tight=tight,
                    )

        parser_x = make_inputs(
            (2049,),
            heads,
            device,
            packed=False,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        parser_reference = run(parser_x, 64, 2, "0")
        for spelling, expected_name in (
            (None, b_stream_name),
            ("", b_stream_name),
            ("0", b_stream_name),
            ("01", b_stream_name),
            ("true", b_stream_name),
            ("1 ", b_stream_name),
            ("1", a_gll_name),
        ):
            actual = run(parser_x, 64, 2, spelling)
            torch.cuda.synchronize(device)
            assert_same(
                actual[0],
                parser_reference[0],
                f"scan A-GLL parser {spelling!r}: output mismatch",
            )
            assert_same(
                actual[1],
                parser_reference[1],
                f"scan A-GLL parser {spelling!r}: state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                2,
                spelling,
                expected_name=expected_name,
            )

        for scan_nw in (1, 4):
            fallback_reference = run(parser_x, 64, scan_nw, "0")
            requested = run(parser_x, 64, scan_nw, "1")
            torch.cuda.synchronize(device)
            assert_same(
                requested[0],
                fallback_reference[0],
                f"scan A-GLL NW{scan_nw} fallback output mismatch",
            )
            assert_same(
                requested[1],
                fallback_reference[1],
                f"scan A-GLL NW{scan_nw} fallback state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                scan_nw,
                "1",
                expected_name=b_stream_name,
            )

        for b_stream in (None, "0"):
            prerequisite_reference = run(
                parser_x,
                64,
                2,
                "0",
                b_stream=b_stream,
            )
            requested = run(
                parser_x,
                64,
                2,
                "1",
                b_stream=b_stream,
            )
            torch.cuda.synchronize(device)
            assert_same(
                requested[0],
                prerequisite_reference[0],
                "scan A-GLL without streamed-b output mismatch",
            )
            assert_same(
                requested[1],
                prerequisite_reference[1],
                "scan A-GLL without streamed-b state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                2,
                "1",
                b_stream=b_stream,
                expected_name=legacy_name,
            )
        print(
            "PASS context scan A-GLL OFF/ON, dense/VL/tight, all state/G, "
            "strict parser, NW fallback, and streamed-b prerequisite matrix"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_scan_b_phased_matrix(
    module, device: torch.device, heads: int
):
    """Prove the HI=false/NW2 two-phase streamed-b scan is exact.

    Dense, packed-VL, and strict tight-grid cases cover every HI=false state
    mode across G32/G64/G128.  Captured graph names also prove the exact parser,
    streamed-b/NW/HI prerequisites, and A-GLL precedence without relying on
    output equality alone to infer which specialization actually launched.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context scan two-phase streamed-b A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        group_chunks: int,
        scan_nw: int,
        b_phased: str | None,
        *,
        b_stream: str | None = "1",
        a_gll: str | None = "0",
        hybrid: bool = False,
        tight: bool = False,
    ) -> None:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ[
            "FLASH_KDA_GFX950_CONTEXT_HYBRID"
            if hybrid
            else "FLASH_KDA_GFX950_CONTEXT_AFFINE"
        ] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = (
            "1" if tight else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = "0"
        if b_stream is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = b_stream
        if a_gll is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL"] = a_gll
        if b_phased is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED"] = b_phased

    def run(
        x,
        group_chunks: int,
        scan_nw: int,
        b_phased: str | None,
        *,
        b_stream: str | None = "1",
        a_gll: str | None = "0",
        hybrid: bool = False,
        tight: bool = False,
    ):
        configure(
            group_chunks,
            scan_nw,
            b_phased,
            b_stream=b_stream,
            a_gll=a_gll,
            hybrid=hybrid,
            tight=tight,
        )
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x5A)
        raw_call(module, x, out, final, workspace)
        return out, final

    def compare(
        x,
        label: str,
        group_chunks: int,
        *,
        hybrid: bool = False,
        tight: bool = False,
    ) -> None:
        initial_copy = x["initial_state"].clone()
        established = run(
            x,
            group_chunks,
            2,
            "0",
            hybrid=hybrid,
            tight=tight,
        )
        phased = run(
            x,
            group_chunks,
            2,
            "1",
            hybrid=hybrid,
            tight=tight,
        )
        torch.cuda.synchronize(device)
        for mode_name, tensors in (("established", established),
                                   ("two-phase", phased)):
            checked = tensors if x["output_final_state"] else tensors[:1]
            if any(
                not bool(torch.isfinite(tensor).all().item())
                for tensor in checked
            ):
                raise AssertionError(
                    f"scan two-phase {label}: {mode_name} produced "
                    "non-finite data"
                )
        assert_same(
            phased[0],
            established[0],
            f"scan two-phase {label}: output mismatch",
        )
        if x["output_final_state"]:
            assert_same(
                phased[1],
                established[1],
                f"scan two-phase {label}: final-state mismatch",
            )
        assert_same(
            x["initial_state"],
            initial_copy,
            f"scan two-phase {label}: initial state mutated",
        )

    def assert_captured_scan_kernel(
        x,
        scan_nw: int,
        b_phased: str | None,
        *,
        expected_name: str,
        b_stream: str | None = "1",
        a_gll: str | None = "0",
    ) -> None:
        """Capture one launch and identify the selected scan specialization."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(
            64,
            scan_nw,
            b_phased,
            b_stream=b_stream,
            a_gll=a_gll,
        )
        out, final, workspace = allocate(x)
        preallocated_call("raw", module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            preallocated_call("raw", module, x, out, final, workspace)

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )

        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        kernel_names = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:  # hipGraphNodeTypeKernel
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            raw_text = raw_name.decode(errors="replace")
            demangled = raw_text
            try:
                demangled = torch._C._demangle(raw_text)
            except (AttributeError, RuntimeError):
                pass
            kernel_names.append(
                raw_text
                if demangled == raw_text
                else f"{raw_text}\n{demangled}"
            )

        scan_names = [
            name
            for name in kernel_names
            if "k2_kda_context_affine_scan" in name
        ]
        if len(scan_names) != 1 or expected_name not in scan_names[0]:
            raise AssertionError(
                "captured graph selected the wrong context scan for "
                f"B_STREAM={b_stream!r}, NW={scan_nw}, "
                f"A_GLL={a_gll!r}, B_PHASED={b_phased!r}; "
                f"expected {expected_name!r}, scan nodes={scan_names!r}"
            )
        graph.reset()

    hi_false_modes = (
        ("none", False, torch.float32),
        ("out-fp32", True, torch.float32),
        ("out-bf16", True, torch.bfloat16),
    )
    tight_lens = (1,) * 16 + (1025,) + (0,) * 16
    layouts = (
        ("dense", (2049, 2049), False, False, False),
        ("packed-vl", (0, 1025, 2049), True, False, False),
        ("packed-tight", tight_lens, True, True, True),
    )
    phased_name = (
        "k2_kda_context_affine_scan_b_stream_b_phased_nw2_kernel"
    )
    a_gll_name = "k2_kda_context_affine_scan_b_stream_a_gll_nw2_kernel"
    b_stream_name = "k2_kda_context_affine_scan_b_stream_nw4_kernel"
    legacy_name = "k2_kda_context_affine_scan_nw4_kernel"

    try:
        for group_chunks in (32, 64, 128):
            tight_total_tiles = (
                (sum(tight_lens) + 15) // 16 + len(tight_lens)
            )
            tight_max_affine_sequences = min(
                len(tight_lens), tight_total_tiles // 65
            )
            tight_context_upper = max(
                1,
                (
                    tight_total_tiles
                    + tight_max_affine_sequences * (group_chunks - 1)
                )
                // group_chunks,
            )
            if tight_context_upper >= len(tight_lens):
                raise AssertionError(
                    "scan two-phase tight test is axis-inactive: "
                    f"N={len(tight_lens)}, G={group_chunks}, "
                    f"context_upper={tight_context_upper}"
                )

            for (
                layout_name,
                seq_lens,
                packed,
                hybrid,
                tight,
            ) in layouts:
                for (
                    mode_name,
                    output_final_state,
                    state_dtype,
                ) in hi_false_modes:
                    x = make_inputs(
                        seq_lens,
                        heads,
                        device,
                        packed=packed,
                        state_dtype=state_dtype,
                        has_initial_state=False,
                        output_final_state=output_final_state,
                    )
                    compare(
                        x,
                        f"{layout_name}/G{group_chunks}/NW2/{mode_name}",
                        group_chunks,
                        hybrid=hybrid,
                        tight=tight,
                    )

        parser_x = make_inputs(
            (2049,),
            heads,
            device,
            packed=False,
            state_dtype=torch.float32,
            has_initial_state=False,
            output_final_state=True,
        )
        parser_reference = run(parser_x, 64, 2, "0")
        for spelling, expected_name in (
            (None, b_stream_name),
            ("", b_stream_name),
            ("0", b_stream_name),
            ("01", b_stream_name),
            ("true", b_stream_name),
            ("1 ", b_stream_name),
            ("1", phased_name),
        ):
            actual = run(parser_x, 64, 2, spelling)
            torch.cuda.synchronize(device)
            assert_same(
                actual[0],
                parser_reference[0],
                f"scan two-phase parser {spelling!r}: output mismatch",
            )
            assert_same(
                actual[1],
                parser_reference[1],
                f"scan two-phase parser {spelling!r}: state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                2,
                spelling,
                expected_name=expected_name,
            )

        for scan_nw in (1, 4):
            fallback_reference = run(parser_x, 64, scan_nw, "0")
            requested = run(parser_x, 64, scan_nw, "1")
            torch.cuda.synchronize(device)
            assert_same(
                requested[0],
                fallback_reference[0],
                f"scan two-phase NW{scan_nw} fallback output mismatch",
            )
            assert_same(
                requested[1],
                fallback_reference[1],
                f"scan two-phase NW{scan_nw} fallback state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                scan_nw,
                "1",
                expected_name=b_stream_name,
            )

        for state_dtype in (torch.float32, torch.bfloat16):
            hi_x = make_inputs(
                (2049,),
                heads,
                device,
                packed=False,
                state_dtype=state_dtype,
                has_initial_state=True,
                output_final_state=True,
            )
            hi_reference = run(hi_x, 64, 2, "0")
            hi_requested = run(hi_x, 64, 2, "1")
            torch.cuda.synchronize(device)
            assert_same(
                hi_requested[0],
                hi_reference[0],
                f"scan two-phase HI/{state_dtype} output fallback mismatch",
            )
            assert_same(
                hi_requested[1],
                hi_reference[1],
                f"scan two-phase HI/{state_dtype} state fallback mismatch",
            )
            assert_captured_scan_kernel(
                hi_x,
                2,
                "1",
                expected_name=b_stream_name,
            )

        a_gll_reference = run(
            parser_x, 64, 2, "0", a_gll="1"
        )
        a_gll_requested = run(
            parser_x, 64, 2, "1", a_gll="1"
        )
        torch.cuda.synchronize(device)
        assert_same(
            a_gll_requested[0],
            a_gll_reference[0],
            "scan two-phase A-GLL precedence output mismatch",
        )
        assert_same(
            a_gll_requested[1],
            a_gll_reference[1],
            "scan two-phase A-GLL precedence state mismatch",
        )
        assert_captured_scan_kernel(
            parser_x,
            2,
            "1",
            a_gll="1",
            expected_name=a_gll_name,
        )

        a_gll_fallback = run(
            parser_x, 64, 2, "1", a_gll="true"
        )
        torch.cuda.synchronize(device)
        assert_same(
            a_gll_fallback[0],
            parser_reference[0],
            "scan two-phase noncanonical A-GLL output mismatch",
        )
        assert_same(
            a_gll_fallback[1],
            parser_reference[1],
            "scan two-phase noncanonical A-GLL state mismatch",
        )
        assert_captured_scan_kernel(
            parser_x,
            2,
            "1",
            a_gll="true",
            expected_name=phased_name,
        )

        for b_stream in (None, "0", "true"):
            prerequisite_reference = run(
                parser_x,
                64,
                2,
                "0",
                b_stream=b_stream,
            )
            requested = run(
                parser_x,
                64,
                2,
                "1",
                b_stream=b_stream,
            )
            torch.cuda.synchronize(device)
            assert_same(
                requested[0],
                prerequisite_reference[0],
                "scan two-phase without streamed-b output mismatch",
            )
            assert_same(
                requested[1],
                prerequisite_reference[1],
                "scan two-phase without streamed-b state mismatch",
            )
            assert_captured_scan_kernel(
                parser_x,
                2,
                "1",
                b_stream=b_stream,
                expected_name=legacy_name,
            )
        print(
            "PASS context scan two-phase streamed-b OFF/ON, dense/VL/tight, "
            "all HI=false state/G, strict parser, NW/HI prerequisites, and "
            "A-GLL precedence matrix"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_scan_ksplit_matrix(
    module, device: torch.device, heads: int
) -> None:
    """A/B the WG4 K64+K64 affine scan with numerical error gates."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context affine-scan K-split A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        *,
        route: str,
        group_chunks: int,
        ksplit: str | None,
        scan_nw: int = 2,
        tight: str = "0",
        b_stream: str = "0",
        a_gll: str = "0",
        b_phased: str = "0",
    ) -> None:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ[_CONTEXT_NW8_ENV] = "0"
        os.environ[_CONTEXT_AFFINE_AB_FUSED_ENV] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = b_stream
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL"] = a_gll
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED"] = b_phased
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = tight
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = "0"
        for name in _CONTEXT_LDS_PIPELINE_PASS_ENV:
            os.environ[name] = "0"
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if ksplit is not None:
            os.environ[_CONTEXT_SCAN_KSPLIT_ENV] = ksplit

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x5B)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_relative_rms(
        actual: torch.Tensor, reference: torch.Tensor, label: str
    ) -> float:
        actual_f = actual.float()
        reference_f = reference.float()
        if not bool(torch.isfinite(actual_f).all().item()):
            raise AssertionError(f"{label}: candidate contains non-finite data")
        difference_rms = torch.sqrt(
            torch.mean(torch.square(actual_f - reference_f))
        )
        reference_rms = torch.sqrt(torch.mean(torch.square(reference_f)))
        relative_rms = float(
            (difference_rms / reference_rms.clamp_min(1.0e-12)).item()
        )
        if relative_rms > 1.0e-4:
            raise AssertionError(
                f"{label}: relative RMS {relative_rms:.6e} exceeds 1e-4"
            )
        return relative_rms

    def assert_candidate(
        actual, reference, label: str
    ) -> tuple[float, float | None]:
        output_error = assert_relative_rms(
            actual[0], reference[0], f"{label} output"
        )
        state_error = None
        if reference[1] is not None:
            if actual[1] is None:
                raise AssertionError(f"{label}: candidate omitted final state")
            state_error = assert_relative_rms(
                actual[1], reference[1], f"{label} state"
            )
        return output_error, state_error

    def stress_inputs(x, stress: str) -> None:
        with torch.no_grad():
            if stress in {"correlated", "both"}:
                # Equal Q/K plus a correlated V component stresses coherent
                # low-rank state updates instead of only random cancellation.
                x["k"].copy_(x["q"])
                correlated_v = 0.75 * x["q"].float() + 0.25 * x["v"].float()
                x["v"].copy_(correlated_v.to(torch.bfloat16))
            if stress in {"weak-gate", "both"}:
                x["g"].mul_(0.03125)
                x["beta"].mul_(0.125)

    def capture_scan_records(x, **configuration):
        """Capture and replay one graph, returning affine-scan topology."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, x, out, final, workspace)
        graph.replay()
        torch.cuda.synchronize(device)

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        records = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            name = raw_name.decode(errors="replace")
            if "k2_kda_context_affine_scan" not in name:
                continue
            records.append(
                {
                    "name": name,
                    "block": (
                        params.block_dim.x,
                        params.block_dim.y,
                        params.block_dim.z,
                    ),
                    "grid": (
                        params.grid_dim.x,
                        params.grid_dim.y,
                        params.grid_dim.z,
                    ),
                    "shared": params.shared_mem_bytes,
                }
            )
        graph.reset()
        return records

    def assert_one_scan(
        records,
        *,
        ksplit: bool,
        group_chunks: int,
        expected_grid_x: int,
        label: str,
    ) -> None:
        if len(records) != 1:
            raise AssertionError(
                f"{label}: expected one affine scan node, got {records!r}"
            )
        record = records[0]
        has_ksplit = "affine_scan_ksplit_wg4_kernel" in record["name"]
        if has_ksplit != ksplit:
            raise AssertionError(
                f"{label}: K-split selection={has_ksplit}, expected {ksplit}: "
                f"{record['name']!r}"
            )
        if ksplit and re.search(
            rf"ksplit_wg4_kernelI(?:Li)?{group_chunks}E", record["name"]
        ) is None:
            raise AssertionError(
                f"{label}: selected the wrong G specialization: "
                f"{record['name']!r}"
            )
        expected_block = (256 if ksplit else 128, 1, 1)
        expected_grid = (expected_grid_x, 4, 1)
        if record["block"] != expected_block:
            raise AssertionError(
                f"{label}: block={record['block']}, expected {expected_block}"
            )
        if record["grid"] != expected_grid:
            raise AssertionError(
                f"{label}: grid={record['grid']}, expected {expected_grid}"
            )
        if record["shared"] != 0:
            raise AssertionError(
                f"{label}: expected static LDS only, got {record['shared']} B"
            )

    cases = (
        (
            "dense-g32-fresh-correlated",
            (2049,),
            False,
            32,
            False,
            False,
            torch.float32,
            "correlated",
        ),
        (
            "dense-g64-resume-fp32-weak-gate",
            (4097,),
            False,
            64,
            True,
            True,
            torch.float32,
            "weak-gate",
        ),
        (
            "dense-g128-resume-bf16-correlated-weak-gate",
            (8193,),
            False,
            128,
            True,
            True,
            torch.bfloat16,
            "both",
        ),
        (
            "packed-g64-fresh-empty-bf16",
            (0, 1025, 2049),
            True,
            64,
            False,
            True,
            torch.bfloat16,
            "correlated",
        ),
        (
            "packed-g32-resume-fp32",
            (1025, 1537),
            True,
            32,
            True,
            True,
            torch.float32,
            "weak-gate",
        ),
    )

    try:
        for (
            label,
            seq_lens,
            packed,
            group_chunks,
            has_initial_state,
            output_final_state,
            state_dtype,
            stress,
        ) in cases:
            x = make_inputs(
                seq_lens,
                heads,
                device,
                packed=packed,
                state_dtype=state_dtype,
                has_initial_state=has_initial_state,
                output_final_state=output_final_state,
            )
            stress_inputs(x, stress)
            initial_copy = x["initial_state"].clone()
            common = {
                "route": "affine",
                "group_chunks": group_chunks,
            }
            reference = run(x, ksplit="0", **common)
            candidate = run(x, ksplit="1", **common)
            torch.cuda.synchronize(device)
            output_error, state_error = assert_candidate(
                candidate, reference, label
            )
            assert_same(
                x["initial_state"],
                initial_copy,
                f"{label}: initial state mutated",
            )
            state_text = (
                "n/a" if state_error is None else f"{state_error:.3e}"
            )
            print(
                f"PASS context K-split {label}: "
                f"output_rms={output_error:.3e}, state_rms={state_text}"
            )
            del x, reference, candidate, initial_copy
            torch.cuda.empty_cache()

        graph_x = make_inputs(
            (2049,),
            heads,
            device,
            packed=False,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        p0_records = capture_scan_records(
            graph_x,
            route="affine",
            group_chunks=64,
            ksplit="0",
        )
        assert_one_scan(
            p0_records,
            ksplit=False,
            group_chunks=64,
            expected_grid_x=heads,
            label="K-split P0 graph",
        )
        candidate_records = capture_scan_records(
            graph_x,
            route="affine",
            group_chunks=64,
            ksplit="1",
        )
        assert_one_scan(
            candidate_records,
            ksplit=True,
            group_chunks=64,
            expected_grid_x=heads,
            label="K-split selected graph",
        )

        fallback_cases = (
            (
                "noncanonical",
                graph_x,
                {"route": "affine", "group_chunks": 64, "ksplit": "true"},
            ),
            (
                "NW4",
                graph_x,
                {
                    "route": "affine",
                    "group_chunks": 64,
                    "ksplit": "1",
                    "scan_nw": 4,
                },
            ),
            (
                "B-stream-axis",
                graph_x,
                {
                    "route": "affine",
                    "group_chunks": 64,
                    "ksplit": "1",
                    "b_stream": "1",
                },
            ),
            (
                "A-GLL-axis",
                graph_x,
                {
                    "route": "affine",
                    "group_chunks": 64,
                    "ksplit": "1",
                    "a_gll": "1",
                },
            ),
            (
                "B-phased-axis",
                graph_x,
                {
                    "route": "affine",
                    "group_chunks": 64,
                    "ksplit": "1",
                    "b_phased": "1",
                },
            ),
            (
                "dense-N2",
                make_inputs(
                    (1025, 1025),
                    heads,
                    device,
                    packed=False,
                    has_initial_state=True,
                    output_final_state=True,
                ),
                {"route": "affine", "group_chunks": 64, "ksplit": "1"},
            ),
            (
                "hybrid",
                make_inputs(
                    (1, 1, 1, 1, 1, 1, 1, 1, 1025),
                    heads,
                    device,
                    packed=True,
                    has_initial_state=True,
                    output_final_state=True,
                ),
                {"route": "hybrid", "group_chunks": 64, "ksplit": "1"},
            ),
        )
        for label, x, requested_options in fallback_cases:
            reference_options = {**requested_options, "ksplit": "0"}
            reference = run(x, **reference_options)
            requested = run(x, **requested_options)
            torch.cuda.synchronize(device)
            assert_same(
                requested[0], reference[0],
                f"K-split {label} fallback output mismatch",
            )
            if x["output_final_state"]:
                assert_same(
                    requested[1], reference[1],
                    f"K-split {label} fallback state mismatch",
                )
            records = capture_scan_records(x, **requested_options)
            if any(
                "affine_scan_ksplit_wg4_kernel" in record["name"]
                for record in records
            ):
                raise AssertionError(
                    f"K-split {label} fallback reached candidate: {records!r}"
                )
            print(f"PASS context K-split {label} bitwise fallback/graph")

        stream_x = make_inputs(
            (1025,),
            heads,
            device,
            packed=False,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        stream_reference = run(
            stream_x,
            route="affine",
            group_chunks=32,
            ksplit="0",
        )
        torch.cuda.synchronize(device)
        configure(
            route="affine",
            group_chunks=32,
            ksplit="1",
        )
        buffers_a = allocate(stream_x)
        buffers_b = allocate(stream_x)
        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        stream_a.wait_stream(torch.cuda.current_stream(device))
        stream_b.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream_a):
            raw_call(module, stream_x, *buffers_a)
        with torch.cuda.stream(stream_b):
            raw_call(module, stream_x, *buffers_b)
        torch.cuda.current_stream(device).wait_stream(stream_a)
        torch.cuda.current_stream(device).wait_stream(stream_b)
        torch.cuda.synchronize(device)
        for stream_label, result in (
            ("stream-a", buffers_a),
            ("stream-b", buffers_b),
        ):
            assert_candidate(
                result,
                stream_reference,
                f"K-split {stream_label}",
            )
        print("PASS context K-split graph replay and two-stream isolation")
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_plain_beta_cache_matrix(module, device: torch.device, heads: int):
    """A/B the gfx950 plain activated-beta producer/consumer handshake."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP plain beta-cache A/B: gfx950 only")
        return

    state_modes = [
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    ]
    layouts = [
        # N=4 and average T>256 stay on plain C-split.  The zero-length entry
        # and three different tails cover duplicate prefix offsets and every
        # partial C16/BT64 boundary without changing packed token order.
        ("packed-ragged-empty", (0, 257, 513, 1025), True),
        ("dense-tail", (257, 257), False),
    ]
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_FUSED_K1",
        "FLASH_KDA_GFX950_PLAIN_BETA_CACHE",
        "FLASH_KDA_GFX950_SCAN_DECAY_CACHE",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16",
        "FLASH_KDA_GFX950_SIN_FRAGMENT",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(*, cache: bool, fused: bool):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = (
            "vector_x32" if fused else "0"
        )
        os.environ["FLASH_KDA_GFX950_FUSED_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_PLAIN_BETA_CACHE"] = "1" if cache else "0"
        os.environ["FLASH_KDA_GFX950_SCAN_DECAY_CACHE"] = "0"

    def run(
        x,
        *,
        cache: bool,
        fused: bool,
        final_poison: float | None = None,
    ):
        configure(cache=cache, fused=fused)
        out, final, workspace = allocate(x)
        if x["output_final_state"] and final_poison is not None:
            final.fill_(final_poison)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None

    try:
        for layout_name, seq_lens, packed in layouts:
            for (
                mode_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                seed_empty_state_bit_patterns(x, seq_lens, device)
                initial_copy = x["initial_state"].clone()
                uncached_out, uncached_final = run(
                    x, cache=False, fused=True, final_poison=3.25
                )
                cached_out, cached_final = run(
                    x, cache=True, fused=True, final_poison=-4.5
                )
                torch.cuda.synchronize(device)
                label = f"plain beta cache {layout_name}/{mode_name}"
                assert_same(cached_out, uncached_out, f"{label} output mismatch")
                if output_final_state:
                    assert_bitwise_same(
                        cached_final, uncached_final, f"{label} state mismatch"
                    )
                    for sequence, length in enumerate(seq_lens):
                        if length != 0:
                            continue
                        expected = (
                            initial_copy[sequence]
                            if has_initial_state
                            else torch.zeros_like(initial_copy[sequence])
                        )
                        assert_bitwise_same(
                            uncached_final[sequence],
                            expected,
                            f"{label} uncached empty-state mismatch",
                        )
                        assert_bitwise_same(
                            cached_final[sequence],
                            expected,
                            f"{label} cached empty-state mismatch",
                        )
                assert_bitwise_same(
                    x["initial_state"], initial_copy, f"{label} input mutated"
                )
                print(f"PASS bitwise {label}")

        # The cache contract must remain false when the callback falls back to
        # the legacy split producer, even if the cache knob itself is enabled.
        fallback = make_inputs((257, 513), heads, device, packed=True)
        fallback_off = run(fallback, cache=False, fused=False)
        fallback_on = run(fallback, cache=True, fused=False)
        torch.cuda.synchronize(device)
        assert_same(
            fallback_on[0], fallback_off[0],
            "disabled fused producer consumed stale beta output",
        )
        assert_same(
            fallback_on[1], fallback_off[1],
            "disabled fused producer consumed stale beta state",
        )
        print("PASS fused-producer beta-cache handshake fallback")

        # The plain-only knob must not alter the existing context operand cache.
        context = make_inputs((65,) * 16, 1, device, packed=True)
        context_off = run(context, cache=False, fused=True)
        context_on = run(context, cache=True, fused=True)
        torch.cuda.synchronize(device)
        assert_same(
            context_on[0], context_off[0], "context output changed by plain knob"
        )
        assert_same(
            context_on[1], context_off[1], "context state changed by plain knob"
        )
        print("PASS context cache isolated from plain beta-cache knob")
    finally:
        restore_env()


def check_forced_csplit_empty_state_matrix(
    module, device: torch.device, heads: int
):
    """Prove every explicit C-split scan leaves the empty-state identity."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP forced C-split empty-state matrix: gfx950 only")
        return

    # Leading, interior, and trailing empty owners ensure duplicate prefix
    # entries cannot accidentally alias a neighbouring sequence's state slab.
    seq_lens = (0, 257, 0, 513, 1025, 0)
    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    variants = (
        ("csplit", "csplit", None),
        ("csplit-mw2", "csplit", 2),
        ("csplit-mw4", "csplit", 4),
        ("csplit-mw8", "csplit", 8),
        ("csplit32", "csplit32", None),
        ("csplit64", "csplit64", None),
        ("csplit64nw8", "csplit64nw8", None),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SCAN_MW",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_BV",
        "FLASH_KDA_OUT_BV",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_FUSED_K1",
        "FLASH_KDA_GFX950_PLAIN_BETA_CACHE",
        "FLASH_KDA_GFX950_SCAN_DECAY_CACHE",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16",
        "FLASH_KDA_GFX950_SIN_FRAGMENT",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    try:
        for variant_name, k2, scan_mw in variants:
            for name in controlled_env:
                os.environ.pop(name, None)
            os.environ["FLASH_KDA_K2"] = k2
            if scan_mw is not None:
                os.environ["FLASH_KDA_CS_SCAN_MW"] = str(scan_mw)

            for mode_name, has_input, has_output, state_dtype in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=True,
                    state_dtype=state_dtype,
                    has_initial_state=has_input,
                    output_final_state=has_output,
                )
                seed_empty_state_bit_patterns(x, seq_lens, device)
                initial_copy = x["initial_state"].clone()
                out, final, workspace = allocate(x)
                out.fill_(float("nan"))
                final.fill_(3.25)
                workspace.fill_(0xA7)
                raw_call(module, x, out, final, workspace)
                torch.cuda.synchronize(device)
                label = f"forced {variant_name}/{mode_name}"
                if not bool(torch.isfinite(out).all().item()):
                    raise AssertionError(f"{label}: output contains non-finite data")
                if has_output:
                    for sequence, length in enumerate(seq_lens):
                        if length != 0:
                            continue
                        expected = (
                            initial_copy[sequence]
                            if has_input
                            else torch.zeros_like(initial_copy[sequence])
                        )
                        assert_bitwise_same(
                            final[sequence],
                            expected,
                            f"{label}: empty-state[{sequence}] mismatch",
                        )
                assert_bitwise_same(
                    x["initial_state"], initial_copy, f"{label}: input mutated"
                )
                print(f"PASS raw-bit {label}")

        # Capture a common NW4 C-split graph with a leading empty sequence,
        # then move that empty owner to the trailing slot without changing N,
        # total tokens, or any captured pointer.  This proves the prefix
        # kernel reads cu_seqlens at replay time rather than baking ownership.
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_K2"] = "csplit64"
        captured_lens = (0, 257, 513, 1025)
        replay_lens = (257, 513, 1025, 0)
        graph_x = make_inputs(
            captured_lens,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        graph_out, graph_final, graph_workspace = allocate(graph_x)
        raw_call(module, graph_x, graph_out, graph_final, graph_workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, graph_x, graph_out, graph_final, graph_workspace)
        names = captured_graph_kernel_names(graph, device)
        if sum("k1_build_tile_prefix" in name for name in names) != 1:
            raise AssertionError(
                f"forced csplit64 graph has wrong prefix topology: {names!r}"
            )
        if sum(
            "k2_kda_csplit_bt64_scan_kernel" in name for name in names
        ) != 1 or any("k2_kda_context" in name for name in names):
            raise AssertionError(
                f"forced csplit64 graph reached the wrong K2 route: {names!r}"
            )
        graph.instantiate()

        offsets = [0]
        for length in replay_lens:
            offsets.append(offsets[-1] + length)
        graph_x["cu_seqlens"].copy_(
            torch.tensor(offsets, device=device, dtype=torch.int32)
        )
        reference_out, reference_final, reference_workspace = allocate(graph_x)
        raw_call(
            module,
            graph_x,
            reference_out,
            reference_final,
            reference_workspace,
        )
        torch.cuda.synchronize(device)
        graph_out.fill_(float("nan"))
        graph_final.fill_(3.25)
        graph_workspace.fill_(0x39)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_same(
            graph_out,
            reference_out,
            "forced csplit64 changed-prefix graph output mismatch",
        )
        assert_bitwise_same(
            graph_final,
            reference_final,
            "forced csplit64 changed-prefix graph state mismatch",
        )
        graph.reset()
        print("PASS forced csplit64 same-N/same-token changed-prefix graph replay")

        # The prefix kernel has no global storage.  Two calls sharing read-only
        # inputs/cu_seqlens but using disjoint final/workspace buffers must be
        # race-free on independent streams.
        stream_x = make_inputs(
            seq_lens,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        stream_reference = allocate(stream_x)
        raw_call(module, stream_x, *stream_reference)
        torch.cuda.synchronize(device)
        check_preallocated_multistream(
            "raw",
            module,
            stream_x,
            stream_reference[0],
            stream_reference[1],
        )
        print(
            "PASS forced csplit/csplit32/csplit64/csplit64nw8 and "
            "CS_SCAN_MW=2/4/8 empty-state/graph/stream matrix"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_plain_decay_cache_matrix(module, device: torch.device, heads: int):
    """A/B the strict-opt-in gfx950 plain suffix-decay handshake."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP plain suffix-decay cache A/B: gfx950 only")
        return

    state_modes = [
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    ]
    layouts = [
        # Duplicate packed prefix offsets plus C16/BT64 tails exercise the
        # segment-major cache index and zero-filled missing gt components.
        ("packed-ragged-empty", (0, 257, 513, 1025), True),
        ("dense-tail", (257, 257), False),
    ]
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_FUSED_K1",
        "FLASH_KDA_GFX950_PLAIN_BETA_CACHE",
        "FLASH_KDA_GFX950_SCAN_DECAY_CACHE",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16",
        "FLASH_KDA_GFX950_SIN_FRAGMENT",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        decay: bool | None,
        bt16_k1: bool = True,
        fused_postprep: bool = True,
        k2: str | None = None,
        skip_prep: bool = False,
        skip_solve: bool = False,
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1" if bt16_k1 else "0"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_FUSED_K1"] = (
            "1" if fused_postprep else "0"
        )
        # Exercise the supported combined-cache specialization throughout.
        os.environ["FLASH_KDA_GFX950_PLAIN_BETA_CACHE"] = "1"
        if decay is not None:
            os.environ["FLASH_KDA_GFX950_SCAN_DECAY_CACHE"] = (
                "1" if decay else "0"
            )
        if k2 is not None:
            os.environ["FLASH_KDA_K2"] = k2
        if skip_prep:
            os.environ["FLASH_KDA_CS_SKIP_K1_PREP"] = "1"
        if skip_solve:
            os.environ["FLASH_KDA_CS_SKIP_K1_SOLVE"] = "1"

    def run(
        x,
        *,
        decay: bool | None,
        bt16_k1: bool = True,
        fused_postprep: bool = True,
        k2: str | None = None,
        skip_prep: bool = False,
        skip_solve: bool = False,
        zero_workspace: bool = False,
    ):
        configure(
            decay=decay,
            bt16_k1=bt16_k1,
            fused_postprep=fused_postprep,
            k2=k2,
            skip_prep=skip_prep,
            skip_solve=skip_solve,
        )
        out, final, workspace = allocate(x)
        if zero_workspace:
            workspace.zero_()
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    try:
        for layout_name, seq_lens, packed in layouts:
            for (
                mode_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                unset = run(x, decay=None)
                rollback = run(x, decay=False)
                cached = run(x, decay=True)
                torch.cuda.synchronize(device)
                label = f"plain suffix decay {layout_name}/{mode_name}"
                assert_result_same(
                    rollback, unset, f"{label} unset-vs-zero rollback"
                )
                assert_result_same(cached, rollback, f"{label} off-vs-on")
                assert_same(
                    x["initial_state"], initial_copy, f"{label} input mutated"
                )
                print(f"PASS bitwise {label}")

        fallback = make_inputs((257, 513), heads, device, packed=True)
        isolation_cases = (
            # Callback present but the fused postprep publisher rolls back.
            ("postprep fallback", {"fused_postprep": False}),
            # Fused postprep runs without PRE_SOLVED, so publication is illegal.
            ("BT16 producer fallback", {"bt16_k1": False}),
            # Explicit plain selection must retain architecture-neutral stages.
            ("explicit plain", {"k2": "csplit64"}),
            # K6 may use cs_segment_a for unrelated data and must ignore bit 2.
            ("explicit K6", {"k2": "csplit64rtpk6bk32"}),
        )
        for label, options in isolation_cases:
            off = run(fallback, decay=False, **options)
            on = run(fallback, decay=True, **options)
            torch.cuda.synchronize(device)
            assert_result_same(on, off, f"plain decay cache {label}")
            print(f"PASS plain suffix-decay isolation: {label}")

        # Skip-stage diagnostics can consume arbitrary ordinary K1 workspace.
        # Zero identical workspaces so this check isolates only whether the
        # decay knob incorrectly redirects P3 to stale cs_segment_a bytes.
        skip_off = run(
            fallback,
            decay=False,
            skip_prep=True,
            skip_solve=True,
            zero_workspace=True,
        )
        skip_on = run(
            fallback,
            decay=True,
            skip_prep=True,
            skip_solve=True,
            zero_workspace=True,
        )
        torch.cuda.synchronize(device)
        assert_result_same(skip_on, skip_off, "plain decay cache skip-stage")
        print("PASS plain suffix-decay isolation: skip-stage diagnostics")

        context = make_inputs((65,) * 16, 1, device, packed=True)
        context_off = run(context, decay=False)
        context_on = run(context, decay=True)
        torch.cuda.synchronize(device)
        assert_result_same(
            context_on, context_off, "context changed by plain decay knob"
        )
        print("PASS context isolated from plain suffix-decay cache knob")
    finally:
        restore_env()


def check_plain_postprep_opt_matrix(module, device: torch.device, heads: int):
    """Prove the gfx950 postprep store/load/fragment axes bit-exact."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP plain postprep schedule A/B: gfx950 only")
        return

    dead_axis = "FLASH_KDA_GFX950_POSTPREP_DEAD_STORES"
    merged_axis = "FLASH_KDA_GFX950_POSTPREP_MERGED_LOADS"
    fragment_axis = "FLASH_KDA_GFX950_POSTPREP_FRAGMENT_FORWARD"
    state_modes = [
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    ]
    layouts = [
        # Keep average T above the automatic plain-route threshold while the
        # modulo-64 tails hit every branch that changes store ownership:
        # pair-only (17), BT64 zero-chunk3 (33), chunk3 (49), and full (63).
        ("packed-ragged-branch-tails", (0, 337, 353, 369, 383), True),
        ("dense-tail33", (289, 289), False),
        ("dense-tail49", (305, 305), False),
    ]
    schedules = [
        ("base", "0", "0", "0"),
        ("dead", "1", "0", "0"),
        ("merged", "0", "1", "0"),
        ("both", "1", "1", "0"),
        ("forward", "1", "1", "1"),
    ]
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_FUSED_K1",
        "FLASH_KDA_GFX950_FUSED_K1_PADDED",
        "FLASH_KDA_GFX950_PLAIN_BETA_CACHE",
        "FLASH_KDA_GFX950_SCAN_DECAY_CACHE",
        dead_axis,
        merged_axis,
        fragment_axis,
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16",
        "FLASH_KDA_GFX950_SIN_FRAGMENT",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        dead: str | None,
        merged: str | None,
        fragment: str | None,
        beta_cache: bool,
        decay_cache: bool,
        bt16_k1: bool = True,
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = (
            "1" if bt16_k1 else "0"
        )
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_FUSED_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_FUSED_K1_PADDED"] = "1"
        os.environ["FLASH_KDA_GFX950_PLAIN_BETA_CACHE"] = (
            "1" if beta_cache else "0"
        )
        os.environ["FLASH_KDA_GFX950_SCAN_DECAY_CACHE"] = (
            "1" if decay_cache else "0"
        )
        if dead is not None:
            os.environ[dead_axis] = dead
        if merged is not None:
            os.environ[merged_axis] = merged
        if fragment is not None:
            os.environ[fragment_axis] = fragment

    def run(
        x,
        *,
        dead: str | None,
        merged: str | None,
        fragment: str | None,
        beta_cache: bool,
        decay_cache: bool,
        bt16_k1: bool = True,
    ):
        configure(
            dead=dead,
            merged=merged,
            fragment=fragment,
            beta_cache=beta_cache,
            decay_cache=decay_cache,
            bt16_k1=bt16_k1,
        )
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    try:
        # Exercise every factorial schedule under all four independent
        # activated-beta/suffix-decay cache contracts.  In particular,
        # beta-only is the current production configuration.
        cache_modes = (
            ("uncached", False, False),
            ("beta-only-production", True, False),
            ("decay-only", False, True),
            ("both-cached", True, True),
        )
        for cache_name, beta_cache, decay_cache in cache_modes:
            for layout_name, seq_lens, packed in layouts:
                for (
                    mode_name,
                    has_initial_state,
                    output_final_state,
                    state_dtype,
                ) in state_modes:
                    x = make_inputs(
                        seq_lens,
                        heads,
                        device,
                        packed=packed,
                        state_dtype=state_dtype,
                        has_initial_state=has_initial_state,
                        output_final_state=output_final_state,
                    )
                    initial_copy = x["initial_state"].clone()
                    reference = run(
                        x,
                        dead="0",
                        merged="0",
                        fragment="0",
                        beta_cache=beta_cache,
                        decay_cache=decay_cache,
                    )
                    candidates = []
                    for schedule_name, dead, merged, fragment in schedules[1:]:
                        candidates.append(
                            (
                                schedule_name,
                                run(
                                    x,
                                    dead=dead,
                                    merged=merged,
                                    fragment=fragment,
                                    beta_cache=beta_cache,
                                    decay_cache=decay_cache,
                                ),
                            )
                        )
                    torch.cuda.synchronize(device)
                    label = (
                        f"postprep {cache_name}/{layout_name}/{mode_name}"
                    )
                    for schedule_name, candidate in candidates:
                        assert_result_same(
                            candidate,
                            reference,
                            f"{label}/{schedule_name}",
                        )
                    assert_same(
                        x["initial_state"],
                        initial_copy,
                        f"{label} input mutated",
                    )
                    print(f"PASS bitwise {label} five schedules")

        # Only the exact spelling "1" may select a new specialization.  Test
        # each parser independently so an accidental permissive parser cannot
        # be hidden by the other axes.
        parser_x = make_inputs((257, 513), heads, device, packed=True)
        parser_reference = run(
            parser_x,
            dead="0",
            merged="0",
            fragment="0",
            beta_cache=False,
            decay_cache=False,
        )
        parser_candidates = []
        for axis_name in ("dead", "merged", "fragment"):
            for value in (None, "0", "true", "01", "1 "):
                options = {"dead": "0", "merged": "0", "fragment": "0"}
                options[axis_name] = value
                parser_candidates.append(
                    (
                        axis_name,
                        "unset" if value is None else value,
                        run(
                            parser_x,
                            dead=options["dead"],
                            merged=options["merged"],
                            fragment=options["fragment"],
                            beta_cache=False,
                            decay_cache=False,
                        ),
                    )
                )
        torch.cuda.synchronize(device)
        for axis_name, value, candidate in parser_candidates:
            assert_result_same(
                candidate,
                parser_reference,
                f"postprep noncanonical {axis_name}={value}",
            )
        print("PASS strict canonical postprep environment parsing")

        # Fragment forwarding is defined only on top of the canonical merged
        # dead-store schedule.  A canonical fragment request with either
        # prerequisite absent must retain that exact non-fragment dispatch.
        for label, dead, merged in (
            ("base", "0", "0"),
            ("dead", "1", "0"),
            ("merged", "0", "1"),
        ):
            nonfragment = run(
                parser_x,
                dead=dead,
                merged=merged,
                fragment="0",
                beta_cache=False,
                decay_cache=False,
            )
            fragment_requested = run(
                parser_x,
                dead=dead,
                merged=merged,
                fragment="1",
                beta_cache=False,
                decay_cache=False,
            )
            torch.cuda.synchronize(device)
            assert_result_same(
                fragment_requested,
                nonfragment,
                f"postprep fragment prerequisite isolation/{label}",
            )
        print("PASS postprep fragment forwarding requires both prerequisites")

        # Disabling the solved producer enters the fused kernel's PRE_SOLVED=0
        # fallback.  All diagnostic knobs must be compile-time forced off.
        fallback = make_inputs((257, 513), heads, device, packed=True)
        fallback_off = run(
            fallback,
            dead="0",
            merged="0",
            fragment="0",
            beta_cache=False,
            decay_cache=False,
            bt16_k1=False,
        )
        fallback_on = run(
            fallback,
            dead="1",
            merged="1",
            fragment="1",
            beta_cache=False,
            decay_cache=False,
            bt16_k1=False,
        )
        torch.cuda.synchronize(device)
        assert_result_same(
            fallback_on,
            fallback_off,
            "postprep PRE_SOLVED=0 fallback",
        )
        print("PASS postprep schedule isolation: PRE_SOLVED=0 fallback")

        # A short, high-N packed workload takes the context route and must not
        # observe plain-postprep controls at all.
        context = make_inputs((65,) * 16, 1, device, packed=True)
        context_off = run(
            context,
            dead="0",
            merged="0",
            fragment="0",
            beta_cache=False,
            decay_cache=False,
        )
        context_on = run(
            context,
            dead="1",
            merged="1",
            fragment="1",
            beta_cache=False,
            decay_cache=False,
        )
        torch.cuda.synchronize(device)
        assert_result_same(
            context_on,
            context_off,
            "context changed by plain postprep schedule knobs",
        )
        print("PASS context isolated from plain postprep schedule knobs")

        # Capture and concurrent streams cover lifetime and cross-CTA hazards
        # for the complete forward-on schedule under the production beta-only
        # cache contract.
        graph_x = make_inputs(
            (0, 257, 513, 1025),
            min(heads, 2),
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        graph_reference = run(
            graph_x,
            dead="1",
            merged="1",
            fragment="1",
            beta_cache=True,
            decay_cache=False,
        )
        torch.cuda.synchronize(device)
        configure(
            dead="1",
            merged="1",
            fragment="1",
            beta_cache=True,
            decay_cache=False,
        )
        check_preallocated_graph(
            "raw",
            module,
            graph_x,
            graph_reference[0],
            graph_reference[1],
        )
        check_preallocated_multistream(
            "raw",
            module,
            graph_x,
            graph_reference[0],
            graph_reference[1],
        )
        print("PASS production-cache postprep forward-on graph/two-stream schedule")
    finally:
        restore_env()


def check_plain_internal_layout_matrix(module, device: torch.device, heads: int):
    """Prove the compact X16 exchange and matched cs_sin layout bit-exact."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP plain internal-layout A/B: gfx950 only")
        return

    state_modes = [
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    ]
    layouts = [
        ("packed-ragged-empty", (0, 257, 513, 1025), True),
        ("dense-tail", (257, 257), False),
    ]
    modes = (
        ("base", "0", "0"),
        ("x16-exchange", "1", "0"),
        ("fragment-sin", "0", "1"),
        ("x16-fragment", "1", "1"),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_FUSED_K1",
        "FLASH_KDA_GFX950_PLAIN_BETA_CACHE",
        "FLASH_KDA_GFX950_SCAN_DECAY_CACHE",
        "FLASH_KDA_GFX950_REPLAY_DECAY_CACHE",
        "FLASH_KDA_GFX950_SCAN_DB",
        "FLASH_KDA_GFX950_SCAN_TB3",
        "FLASH_KDA_GFX950_PARTIAL_PAD",
        "FLASH_KDA_GFX950_SCAN_REGB_X32",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG",
        "FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16",
        "FLASH_KDA_GFX950_SIN_FRAGMENT",
        "FLASH_KDA_GFX950_OUT_X32",
        "FLASH_KDA_GFX950_OUT_GLL",
        "FLASH_KDA_GFX950_OUT_GLL_SIN",
        "FLASH_KDA_GFX950_OUT_GLL_TB3",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        x16: str | None,
        fragment: str | None,
        scan_db: str = "0",
        out_x32: str = "0",
        out_gll: str = "1",
        explicit_k2: str | None = None,
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ.update(
            {
                "FLASH_KDA_GFX950_CSPLIT64_MIN_T": "256",
                "FLASH_KDA_GFX950_BT16_K1": "1",
                "FLASH_KDA_GFX950_BT16_FUSED": "vector_x32",
                "FLASH_KDA_GFX950_FUSED_K1": "1",
                "FLASH_KDA_GFX950_PLAIN_BETA_CACHE": "1",
                "FLASH_KDA_GFX950_SCAN_DECAY_CACHE": "1",
                "FLASH_KDA_GFX950_REPLAY_DECAY_CACHE": "1",
                "FLASH_KDA_GFX950_SCAN_DB": scan_db,
                "FLASH_KDA_GFX950_SCAN_TB3": "0",
                "FLASH_KDA_GFX950_PARTIAL_PAD": "1",
                "FLASH_KDA_GFX950_SCAN_REGB_X32": "0",
                "FLASH_KDA_GFX950_SCAN_STATE_XCHG": "0",
                "FLASH_KDA_GFX950_OUT_X32": out_x32,
                "FLASH_KDA_GFX950_OUT_GLL": out_gll,
                "FLASH_KDA_GFX950_OUT_GLL_SIN": "1",
                "FLASH_KDA_GFX950_OUT_GLL_TB3": "1",
            }
        )
        if x16 is not None:
            os.environ["FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16"] = x16
        if fragment is not None:
            os.environ["FLASH_KDA_GFX950_SIN_FRAGMENT"] = fragment
        if explicit_k2 is not None:
            os.environ["FLASH_KDA_K2"] = explicit_k2

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    try:
        for layout_name, seq_lens, packed in layouts:
            for (
                mode_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                results = {
                    name: run(x, x16=x16, fragment=fragment)
                    for name, x16, fragment in modes
                }
                torch.cuda.synchronize(device)
                reference = results["base"]
                for name, result in results.items():
                    for tensor_name, tensor in (
                        ("output", result[0]),
                        ("state", result[1]),
                    ):
                        if tensor is not None and not bool(
                            torch.isfinite(tensor).all().item()
                        ):
                            raise RuntimeError(
                                f"plain internal layout {layout_name}/"
                                f"{mode_name}/{name}: non-finite {tensor_name}"
                            )
                    assert_result_same(
                        result,
                        reference,
                        f"plain internal layout {layout_name}/{mode_name}/{name}",
                    )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"plain internal layout {layout_name}/{mode_name} input mutated",
                )
                print(
                    "PASS bitwise plain internal layout "
                    f"{layout_name}/{mode_name}"
                )

        probe = make_inputs(
            (257, 513),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        base = run(probe, x16=None, fragment=None)
        for value in ("0", "01", "true", "1 "):
            candidate = run(probe, x16=value, fragment=value)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                base,
                f"noncanonical internal-layout value {value!r}",
            )
        print("PASS strict canonical plain internal-layout environment parsing")

        # Fragment-major cs_sin is legal only when its private producer and
        # consumer are selected as one policy decision.  These fallbacks prove
        # that forcing another consumer or a multi-arena scan cannot mismatch
        # the internal workspace layout.
        isolation = (
            ("x32 replay", {"out_x32": "1"}),
            ("common replay", {"out_gll": "0"}),
            ("multi-arena scan", {"scan_db": "1"}),
            ("explicit common K2", {"explicit_k2": "csplit64"}),
        )
        for label, options in isolation:
            off = run(probe, x16="0", fragment="0", **options)
            on = run(probe, x16="1", fragment="1", **options)
            torch.cuda.synchronize(device)
            assert_result_same(on, off, f"plain internal-layout {label}")
            print(f"PASS matched internal-layout fallback: {label}")

        graph_x = make_inputs(
            (257, 513, 1025),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        reference = run(graph_x, x16="1", fragment="1")
        configure(x16="1", fragment="1")
        out, final, workspace = allocate(graph_x)
        raw_call(module, graph_x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            raw_call(module, graph_x, out, final, workspace)
        for replay in range(2):
            out.fill_(float("nan"))
            final.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert_same(
                out,
                reference[0],
                f"plain internal-layout graph replay {replay} output mismatch",
            )
            assert_same(
                final,
                reference[1],
                f"plain internal-layout graph replay {replay} state mismatch",
            )
        print("PASS plain internal-layout graph capture/replay")

        configure(x16="1", fragment="1")
        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        buffers_a = allocate(graph_x)
        buffers_b = allocate(graph_x)
        stream_a.wait_stream(torch.cuda.current_stream(device))
        stream_b.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream_a):
            raw_call(module, graph_x, *buffers_a)
        with torch.cuda.stream(stream_b):
            raw_call(module, graph_x, *buffers_b)
        torch.cuda.current_stream(device).wait_stream(stream_a)
        torch.cuda.current_stream(device).wait_stream(stream_b)
        torch.cuda.synchronize(device)
        for label, buffers in (("stream-a", buffers_a), ("stream-b", buffers_b)):
            assert_same(
                buffers[0],
                reference[0],
                f"plain internal-layout {label} output mismatch",
            )
            assert_same(
                buffers[1],
                reference[1],
                f"plain internal-layout {label} state mismatch",
            )
        print("PASS concurrent plain internal-layout two-stream calls")
    finally:
        restore_env()


def expect_rejection(label: str, operation):
    try:
        operation()
    except (ValueError, RuntimeError) as error:
        print(f"PASS reject {label}: {type(error).__name__}: {error}")
        return
    raise AssertionError(f"raw ABI accepted invalid {label}")


def replay_and_check(graph, out, final, reference_out, reference_final, label: str):
    device = out.device
    for replay in range(2):
        out.fill_(float("nan"))
        if final is not None:
            final.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        assert_same(out, reference_out, f"{label} replay {replay} output mismatch")
        if reference_final is not None:
            assert final is not None
            assert_same(
                final,
                reference_final,
                f"{label} replay {replay} final-state mismatch",
            )


def check_preallocated_graph(abi, module, x, reference_out, reference_final):
    device = x["q"].device
    out, final, workspace = allocate(x)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        preallocated_call(abi, module, x, out, final, workspace)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        preallocated_call(abi, module, x, out, final, workspace)
    replay_and_check(
        graph,
        out,
        final if x["output_final_state"] else None,
        reference_out,
        reference_final,
        f"{abi} preallocated graph",
    )
    print(f"PASS {abi} preallocated graph capture/replay")


def check_public_graph(x, reference_out, reference_final):
    """Capture allocations as well as launches, including the local workspace."""

    device = x["q"].device
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        warm_out, warm_final = public_call(x)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    assert_same(warm_out, reference_out, "public graph warmup output mismatch")
    if reference_final is not None:
        assert warm_final is not None
        assert_same(warm_final, reference_final, "public graph warmup state mismatch")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_final = public_call(x)
    replay_and_check(
        graph,
        graph_out,
        graph_final,
        reference_out,
        reference_final,
        "allocation-owning public graph",
    )
    print("PASS allocation-owning public graph capture/replay")


def check_preallocated_multistream(
    abi, module, x, reference_out, reference_final
):
    device = x["q"].device
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    buffers_a = allocate(x)
    buffers_b = allocate(x)
    stream_a.wait_stream(torch.cuda.current_stream(device))
    stream_b.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream_a):
        preallocated_call(abi, module, x, *buffers_a)
    with torch.cuda.stream(stream_b):
        preallocated_call(abi, module, x, *buffers_b)
    torch.cuda.current_stream(device).wait_stream(stream_a)
    torch.cuda.current_stream(device).wait_stream(stream_b)
    torch.cuda.synchronize(device)
    for name, (out, final, _) in (("stream_a", buffers_a), ("stream_b", buffers_b)):
        assert_same(out, reference_out, f"{abi} {name} output mismatch")
        if reference_final is not None:
            assert_same(final, reference_final, f"{abi} {name} state mismatch")
    print(f"PASS concurrent {abi} two-stream calls with disjoint workspaces")


def check_public_multistream(x, reference_out, reference_final):
    device = x["q"].device
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    stream_a.wait_stream(torch.cuda.current_stream(device))
    stream_b.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream_a):
        result_a = public_call(x)
    with torch.cuda.stream(stream_b):
        result_b = public_call(x)
    torch.cuda.current_stream(device).wait_stream(stream_a)
    torch.cuda.current_stream(device).wait_stream(stream_b)
    torch.cuda.synchronize(device)
    for name, (out, final) in (("stream_a", result_a), ("stream_b", result_b)):
        assert_same(out, reference_out, f"public {name} output mismatch")
        if reference_final is not None:
            assert final is not None
            assert_same(final, reference_final, f"public {name} state mismatch")
    print("PASS concurrent allocation-owning public two-stream calls")


def check_context_affine_ab_fused_matrix(
    module, device: torch.device, heads: int
):
    """A/B the packed NW4/P0 fused affine B+A producer."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context affine B/A fusion A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        _CONTEXT_PERSISTENT_ENV,
        _CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        *,
        route: str,
        fused: str | None,
        group_chunks: int = 64,
        nw8: str = "0",
        operand_cache: str = "1",
        u_forward: str = "1",
        v_forward: str = "1",
        global_pipeline: str = "0",
        pipeline_mask: str = "000",
    ) -> None:
        if len(pipeline_mask) != 3 or any(
            bit not in "01" for bit in pipeline_mask
        ):
            raise ValueError(f"invalid B/A/replay mask: {pipeline_mask!r}")
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ[_CONTEXT_NW8_ENV] = nw8
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = "2"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = operand_cache
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = u_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = v_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = global_pipeline
        for name, value in zip(
            _CONTEXT_LDS_PIPELINE_PASS_ENV, pipeline_mask, strict=True
        ):
            os.environ[name] = value
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if fused is not None:
            os.environ[_CONTEXT_AFFINE_AB_FUSED_ENV] = fused

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0xA7)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str) -> None:
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            if actual[1] is None:
                raise AssertionError(f"{label} omitted final state")
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    def capture_kernel_records(x, **configuration):
        """Capture raw names plus launch dimensions for every kernel node."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, x, out, final, workspace)

        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        records = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            records.append(
                {
                    "name": raw_name.decode(errors="replace"),
                    "block": (
                        params.block_dim.x,
                        params.block_dim.y,
                        params.block_dim.z,
                    ),
                    "grid": (
                        params.grid_dim.x,
                        params.grid_dim.y,
                        params.grid_dim.z,
                    ),
                    "shared": params.shared_mem_bytes,
                    "has_params": bool(params.kernel_params),
                    "has_extra": bool(params.extra),
                }
            )
        graph.reset()
        return records

    def producer_kind(name: str) -> str | None:
        if (
            "k2_kda_context_affine_ab_fused_nw4_kernel" in name
            or "k2_kda_context_affine_ab_fused_dense_nw4_kernel" in name
        ):
            return "fused"
        match = _CONTEXT_PIPELINE_SYMBOL.search(name)
        if match is None:
            return None
        mode = int(match.group("mode"))
        if mode == 0:
            return "affine_b"
        if mode == 1:
            return "affine_a"
        return None

    def normalized_records(records, omitted: set[str]):
        return sorted(
            (
                record["name"],
                record["block"],
                record["grid"],
                record["shared"],
            )
            for record in records
            if producer_kind(record["name"]) not in omitted
        )

    def assert_fused_topology(
        established,
        fused,
        *,
        group_chunks: int,
        dense: bool = False,
        label: str,
    ) -> None:
        established_b = [
            record
            for record in established
            if producer_kind(record["name"]) == "affine_b"
        ]
        established_a = [
            record
            for record in established
            if producer_kind(record["name"]) == "affine_a"
        ]
        fused_nodes = [
            record
            for record in fused
            if producer_kind(record["name"]) == "fused"
        ]
        residual_producers = [
            record
            for record in fused
            if producer_kind(record["name"]) in {"affine_b", "affine_a"}
        ]
        if (
            len(established_b) != 1
            or len(established_a) != 1
            or len(fused_nodes) != 1
            or residual_producers
        ):
            raise AssertionError(
                f"{label}: B+A did not become exactly one fused node: "
                f"B={established_b}, A={established_a}, fused={fused_nodes}, "
                f"residual={residual_producers}"
            )
        node = fused_nodes[0]
        symbol = (
            "affine_ab_fused_dense_nw4_kernel"
            if dense
            else "affine_ab_fused_nw4_kernel"
        )
        if re.search(rf"{symbol}I(?:Li)?{group_chunks}E", node["name"]) is None:
            raise AssertionError(
                f"{label}: fused graph selected the wrong G specialization: "
                f"{node['name']!r}"
            )
        if node["block"] != (256, 1, 1) or node["grid"][1:] != (2, 1):
            raise AssertionError(
                f"{label}: fused launch block/grid is wrong: {node}"
            )
        if node["shared"] != 0 or not (
            node["has_params"] or node["has_extra"]
        ):
            raise AssertionError(
                f"{label}: fused graph kernel ABI fields are wrong: {node}"
            )
        for standalone in (established_b[0], established_a[0]):
            if standalone["block"] != node["block"]:
                raise AssertionError(
                    f"{label}: fused block differs from standalone producer"
                )
            if standalone["grid"] != node["grid"]:
                raise AssertionError(
                    f"{label}: fused grid differs from standalone producer"
                )
        established_rest = normalized_records(
            established, {"affine_b", "affine_a"}
        )
        fused_rest = normalized_records(fused, {"fused"})
        if established_rest != fused_rest:
            raise AssertionError(
                f"{label}: fusion changed hybrid-direct/scan/replay nodes"
            )

    def assert_fallback_graph(reference, candidate, label: str) -> None:
        fused_nodes = [
            record
            for record in candidate
            if producer_kind(record["name"]) == "fused"
        ]
        if fused_nodes:
            raise AssertionError(f"{label}: unexpectedly launched fusion")
        if normalized_records(reference, set()) != normalized_records(
            candidate, set()
        ):
            raise AssertionError(
                f"{label}: fallback graph differs from established graph"
            )

    state_modes = (
        ("fresh", False, torch.float32),
        ("resume-fp32", True, torch.float32),
        ("resume-bf16", True, torch.bfloat16),
    )
    positive_cases = (
        ("packed-affine-g32", (0, 511, 512, 513, 1025), True, "affine", 32),
        ("packed-affine-g64", (0, 1023, 1024, 1025, 2049), True, "affine", 64),
        ("packed-affine-g128", (0, 2047, 2048, 2049, 4097), True, "affine", 128),
        # Public/raw single-sequence packed calls normalize to these exact
        # dense B=1 layouts.  Exercise every compiled group specialization at
        # its two-group boundary so the candidate cannot silently fall back.
        ("dense-n1-g32", (513,), False, "affine", 32),
        ("dense-n1-g64", (1025,), False, "affine", 64),
        ("dense-n1-g128", (2049,), False, "affine", 128),
        # This is the actual public/K3 shape: the caller supplies packed N=1,
        # then the raw adapter canonicalizes it to the dense layout before
        # policy dispatch.  Keep it distinct from the direct dense ABI cases.
        ("packed-n1-normalized-g64", (1025,), True, "affine", 64),
        (
            "hybrid-ragged",
            (0, 1, 16, 1024, 1025, 2049, 65, 513, 1537),
            True,
            "hybrid",
            64,
        ),
    )

    try:
        for case_name, seq_lens, packed, route, group_chunks in positive_cases:
            for state_name, has_initial_state, state_dtype in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=True,
                )
                initial_copy = x["initial_state"].clone()
                options = {
                    "route": route,
                    "group_chunks": group_chunks,
                }
                established = run(x, fused="0", **options)
                candidate = run(x, fused="1", **options)
                torch.cuda.synchronize(device)
                for mode_name, result in (
                    ("standalone", established),
                    ("fused", candidate),
                ):
                    if not bool(torch.isfinite(result[0]).all().item()):
                        raise RuntimeError(
                            f"affine B/A {mode_name} {case_name}/{state_name} "
                            "produced non-finite output"
                        )
                    if result[1] is None or not bool(
                        torch.isfinite(result[1]).all().item()
                    ):
                        raise RuntimeError(
                            f"affine B/A {mode_name} {case_name}/{state_name} "
                            "produced non-finite state"
                        )
                assert_result_same(
                    candidate,
                    established,
                    f"affine B/A fused {case_name}/{state_name}",
                )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"affine B/A fused {case_name}/{state_name} input mutated",
                )
                print(
                    f"PASS bitwise affine B/A fused {case_name}/{state_name}"
                )

        graph_x = make_inputs(
            (0, 1023, 1024, 1025, 2049),
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        graph_options = {"route": "affine", "group_chunks": 64}
        established_graph = capture_kernel_records(
            graph_x, fused="0", **graph_options
        )
        fused_graph = capture_kernel_records(
            graph_x, fused="1", **graph_options
        )
        assert_fused_topology(
            established_graph,
            fused_graph,
            group_chunks=64,
            label="packed affine G64",
        )

        dense_graph_x = make_inputs(
            (1025,),
            heads,
            device,
            packed=False,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        dense_graph_options = {"route": "affine", "group_chunks": 64}
        dense_established_graph = capture_kernel_records(
            dense_graph_x, fused="0", **dense_graph_options
        )
        dense_fused_graph = capture_kernel_records(
            dense_graph_x, fused="1", **dense_graph_options
        )
        assert_fused_topology(
            dense_established_graph,
            dense_fused_graph,
            group_chunks=64,
            dense=True,
            label="dense N1 affine G64",
        )

        packed_n1_graph_x = make_inputs(
            (1025,),
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        packed_n1_established_graph = capture_kernel_records(
            packed_n1_graph_x, fused="0", **dense_graph_options
        )
        packed_n1_fused_graph = capture_kernel_records(
            packed_n1_graph_x, fused="1", **dense_graph_options
        )
        assert_fused_topology(
            packed_n1_established_graph,
            packed_n1_fused_graph,
            group_chunks=64,
            dense=True,
            label="packed N1 normalized-to-dense affine G64",
        )

        hybrid_x = make_inputs(
            (0, 1, 16, 1024, 1025, 2049, 65, 513, 1537),
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        hybrid_options = {"route": "hybrid", "group_chunks": 64}
        hybrid_established = capture_kernel_records(
            hybrid_x, fused="0", **hybrid_options
        )
        hybrid_fused = capture_kernel_records(
            hybrid_x, fused="1", **hybrid_options
        )
        assert_fused_topology(
            hybrid_established,
            hybrid_fused,
            group_chunks=64,
            label="packed hybrid G64",
        )
        print(
            "PASS packed/dense-N1 affine B/A fused graph producer and "
            "unchanged consumers"
        )

        parser_reference = run(graph_x, fused=None, **graph_options)
        reference_graph = capture_kernel_records(
            graph_x, fused=None, **graph_options
        )
        for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
            candidate = run(graph_x, fused=spelling, **graph_options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                parser_reference,
                f"affine B/A fused parser {spelling!r}",
            )
            candidate_graph = capture_kernel_records(
                graph_x, fused=spelling, **graph_options
            )
            assert_fallback_graph(
                reference_graph,
                candidate_graph,
                f"affine B/A fused parser {spelling!r}",
            )
        print("PASS strict canonical affine B/A fused environment parsing")

        prerequisite_cases = (
            ("NW8", {"nw8": "1"}),
            ("operand-cache", {"operand_cache": "0"}),
            ("U-forward", {"u_forward": "0"}),
            ("V-forward", {"v_forward": "0"}),
            ("global-pipeline", {"global_pipeline": "1"}),
            ("B-pipeline", {"pipeline_mask": "100"}),
            ("A-pipeline", {"pipeline_mask": "010"}),
        )
        for prerequisite, overrides in prerequisite_cases:
            options = {**graph_options, **overrides}
            reference = run(graph_x, fused="0", **options)
            candidate = run(graph_x, fused="1", **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                reference,
                f"affine B/A fused missing {prerequisite}",
            )
            reference_nodes = capture_kernel_records(
                graph_x, fused="0", **options
            )
            candidate_nodes = capture_kernel_records(
                graph_x, fused="1", **options
            )
            assert_fallback_graph(
                reference_nodes,
                candidate_nodes,
                f"affine B/A fused missing {prerequisite}",
            )

        # Replay owns a separate kernel-local LDS arena, so its double-buffer
        # specialization is orthogonal to the fused producer.  Prove that the
        # exact replay pipeline keeps one fused B/A node and leaves every
        # consumer otherwise unchanged.
        replay_pipeline_options = {
            **graph_options,
            "pipeline_mask": "001",
        }
        replay_pipeline_reference = run(
            graph_x, fused="0", **replay_pipeline_options
        )
        replay_pipeline_candidate = run(
            graph_x, fused="1", **replay_pipeline_options
        )
        torch.cuda.synchronize(device)
        assert_result_same(
            replay_pipeline_candidate,
            replay_pipeline_reference,
            "affine B/A fused with replay pipeline",
        )
        assert_fused_topology(
            capture_kernel_records(
                graph_x, fused="0", **replay_pipeline_options
            ),
            capture_kernel_records(
                graph_x, fused="1", **replay_pipeline_options
            ),
            group_chunks=64,
            label="packed affine G64 with replay pipeline",
        )

        dense_x = make_inputs(
            (1025, 1025),
            heads,
            device,
            packed=False,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        dense_options = {"route": "affine", "group_chunks": 64}
        dense_reference = run(dense_x, fused="0", **dense_options)
        dense_candidate = run(dense_x, fused="1", **dense_options)
        torch.cuda.synchronize(device)
        assert_result_same(
            dense_candidate, dense_reference, "dense N>1 affine B/A fallback"
        )
        assert_fallback_graph(
            capture_kernel_records(dense_x, fused="0", **dense_options),
            capture_kernel_records(dense_x, fused="1", **dense_options),
            "dense N>1 affine B/A fallback",
        )

        direct_x = make_inputs(
            (0, 1, 17, 65, 257),
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        direct_options = {"route": "direct", "group_chunks": 64}
        direct_reference = run(direct_x, fused="0", **direct_options)
        direct_candidate = run(direct_x, fused="1", **direct_options)
        torch.cuda.synchronize(device)
        assert_result_same(
            direct_candidate,
            direct_reference,
            "pure direct affine B/A fallback",
        )
        assert_fallback_graph(
            capture_kernel_records(direct_x, fused="0", **direct_options),
            capture_kernel_records(direct_x, fused="1", **direct_options),
            "pure direct affine B/A fallback",
        )
        print(
            "PASS affine B/A fused dense-N>1/direct/NW8/cache/U/V/B/A "
            "pipeline fallbacks and replay-pipeline composition"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_nw8_matrix(module, device: torch.device, heads: int):
    """A/B the guarded 512-thread context recurrence against NW4."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context NW8 A/B: gfx950 only")
        return

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    # The three full-state cases cover direct empty/VL handling, ragged
    # affine maps, and the mixed direct/affine route.  Compact extra cases
    # exercise dense metadata and both nondefault affine group sizes.
    cases = (
        (
            "direct-packed-empty",
            (0, 1, 16, 17, 65, 257),
            True,
            "direct",
            64,
            state_modes,
        ),
        (
            "affine-packed-ragged-g64",
            (0, 1023, 1024, 1025, 2049),
            True,
            "affine",
            64,
            state_modes,
        ),
        (
            "hybrid-packed-boundary",
            (0, 1, 1024, 1025, 2049, 65, 513, 1537, 257),
            True,
            "hybrid",
            64,
            state_modes,
        ),
        (
            "affine-dense-g32",
            (513, 513),
            False,
            "affine",
            32,
            (state_modes[0], state_modes[-1]),
        ),
        (
            "affine-packed-g128",
            (0, 2049, 4097),
            True,
            "affine",
            128,
            (state_modes[-2], state_modes[-1]),
        ),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def configure(
        *,
        route: str,
        nw8: str | None,
        tail_first: str | None = None,
        direct_nw: str = "4",
        group_chunks: int = 64,
        operand_cache: str = "1",
        u_forward: str = "1",
        v_forward: str = "1",
        global_pipeline: str = "0",
        pipeline_mask: str = "000",
    ) -> None:
        if len(pipeline_mask) != 3 or any(
            bit not in "01" for bit in pipeline_mask
        ):
            raise ValueError(f"invalid B/A/replay mask: {pipeline_mask!r}")
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = direct_nw
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = "2"
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = operand_cache
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = u_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = v_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = global_pipeline
        for name, value in zip(
            _CONTEXT_LDS_PIPELINE_PASS_ENV, pipeline_mask, strict=True
        ):
            os.environ[name] = value
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if nw8 is not None:
            os.environ[_CONTEXT_NW8_ENV] = nw8
        if tail_first is not None:
            os.environ[_CONTEXT_DIRECT_TAIL_FIRST_ENV] = tail_first

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x6D)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str) -> None:
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            if actual[1] is None:
                raise AssertionError(f"{label} omitted final state")
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    def capture_context_topology(x, **configuration):
        """Return name/NW/grid/block for every captured recurrence node."""

        import ctypes

        class Dim3(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("z", ctypes.c_uint),
            )

        class HipKernelNodeParams(ctypes.Structure):
            _fields_ = (
                ("block_dim", Dim3),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("grid_dim", Dim3),
                ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
                ("shared_mem_bytes", ctypes.c_uint),
            )

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGraphGetNodes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        )
        hip.hipGraphGetNodes.restype = ctypes.c_int
        hip.hipGraphNodeGetType.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        hip.hipGraphNodeGetType.restype = ctypes.c_int
        hip.hipGraphKernelNodeGetParams.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(HipKernelNodeParams),
        )
        hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
        hip.hipKernelNameRefByPtr.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

        def checked(status: int, operation: str) -> None:
            if status != 0:
                raise RuntimeError(
                    f"{operation} failed with HIP status {status}"
                )

        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, x, out, final, workspace)
        graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        checked(
            hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
            "hipGraphGetNodes(count)",
        )
        nodes = (ctypes.c_void_p * count.value)()
        checked(
            hip.hipGraphGetNodes(
                graph_handle, nodes, ctypes.byref(count)
            ),
            "hipGraphGetNodes(nodes)",
        )
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(device).cuda_stream
        )
        records = []
        for node in nodes[: count.value]:
            node_type = ctypes.c_int()
            checked(
                hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
                "hipGraphNodeGetType",
            )
            if node_type.value != 0:
                continue
            params = HipKernelNodeParams()
            checked(
                hip.hipGraphKernelNodeGetParams(
                    node, ctypes.byref(params)
                ),
                "hipGraphKernelNodeGetParams",
            )
            raw_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if raw_name is None:
                continue
            name = raw_name.decode(errors="replace")
            match = _CONTEXT_PIPELINE_SYMBOL.search(name)
            if match is None:
                continue
            records.append(
                {
                    "name": name,
                    "group": int(match.group("group")),
                    "mode": int(match.group("mode")),
                    "direct": int(match.group("direct")),
                    "tail_first": int(match.group("tail_first")),
                    "nw": int(match.group("nw")),
                    "block": (
                        params.block_dim.x,
                        params.block_dim.y,
                        params.block_dim.z,
                    ),
                    "grid": (
                        params.grid_dim.x,
                        params.grid_dim.y,
                        params.grid_dim.z,
                    ),
                }
            )
        graph.reset()
        return records

    def assert_topology(
        x,
        *,
        expected_nw: int,
        expected_nodes: int,
        expected_tail_first: int = 0,
        label: str,
        **configuration,
    ) -> None:
        records = capture_context_topology(x, **configuration)
        if len(records) != expected_nodes:
            raise AssertionError(
                f"{label}: expected {expected_nodes} context nodes, "
                f"got {records!r}"
            )
        expected_block = (expected_nw * 64, 1, 1)
        for record in records:
            if "k2_kda_context_parallel_nw4_kernel" not in record["name"]:
                raise AssertionError(
                    f"{label}: unexpected context kernel {record['name']!r}"
                )
            if record["nw"] != expected_nw:
                raise AssertionError(
                    f"{label}: symbol NW={record['nw']}, "
                    f"expected {expected_nw}"
                )
            if record["tail_first"] != expected_tail_first:
                raise AssertionError(
                    f"{label}: symbol tail_first={record['tail_first']}, "
                    f"expected {expected_tail_first}"
                )
            if record["block"] != expected_block:
                raise AssertionError(
                    f"{label}: block={record['block']}, "
                    f"expected {expected_block}"
                )
            if record["direct"]:
                expected_grid = (x["N"] * heads, 8 // expected_nw, 1)
                if record["grid"] != expected_grid:
                    raise AssertionError(
                        f"{label}: direct grid={record['grid']}, "
                        f"expected {expected_grid}"
                    )
            else:
                expected_grid_y = 8 // expected_nw
                if (
                    record["grid"][1] != expected_grid_y
                    or record["grid"][2] != 1
                ):
                    raise AssertionError(
                        f"{label}: affine grid={record['grid']}, expected "
                        f"grid.y={expected_grid_y}, grid.z=1"
                    )

    try:
        for (
            case_name,
            seq_lens,
            packed,
            route,
            group_chunks,
            selected_states,
        ) in cases:
            for (
                state_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in selected_states:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                options = {
                    "route": route,
                    "group_chunks": group_chunks,
                }
                nw4 = run(x, nw8="0", **options)
                nw8 = run(x, nw8="1", **options)
                torch.cuda.synchronize(device)
                for mode_name, result in (("NW4", nw4), ("NW8", nw8)):
                    tensors = result[:2] if output_final_state else result[:1]
                    for tensor in tensors:
                        if tensor is not None and not bool(
                            torch.isfinite(tensor).all().item()
                        ):
                            raise RuntimeError(
                                f"context {mode_name} {case_name}/{state_name} "
                                "produced non-finite data"
                            )
                assert_result_same(
                    nw8, nw4, f"context NW8 {case_name}/{state_name}"
                )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"context NW8 {case_name}/{state_name} input mutated",
                )
                print(f"PASS bitwise context NW8 {case_name}/{state_name}")

        # B-only, A-only, replay-only, all-pass, and legacy-global P1 prove
        # that the 512-thread recurrence composes with every pipeline bit.
        pipeline_x = make_inputs(
            (0, 1025, 2049),
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        for global_pipeline, mask in (
            ("0", "100"),
            ("0", "010"),
            ("0", "001"),
            ("0", "111"),
            ("1", "000"),
        ):
            options = {
                "route": "affine",
                "group_chunks": 64,
                "global_pipeline": global_pipeline,
                "pipeline_mask": mask,
            }
            nw4 = run(pipeline_x, nw8="0", **options)
            nw8 = run(pipeline_x, nw8="1", **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                nw8,
                nw4,
                f"context NW8 pipeline global={global_pipeline}/mask={mask}",
            )
        print("PASS context NW8 with B/A/replay-specific and global pipelines")

        parser_x = make_inputs(
            (0, 1, 17, 65, 257),
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        parser_options = {"route": "direct", "group_chunks": 64}
        parser_reference = run(parser_x, nw8=None, **parser_options)
        for spelling, expected_nw in (
            (None, 4),
            ("", 4),
            ("0", 4),
            ("01", 4),
            ("true", 4),
            ("1 ", 4),
            (" 1", 4),
            ("1", 8),
        ):
            candidate = run(parser_x, nw8=spelling, **parser_options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                parser_reference,
                f"context NW8 parser {spelling!r}",
            )
            assert_topology(
                parser_x,
                nw8=spelling,
                expected_nw=expected_nw,
                expected_nodes=1,
                label=f"context NW8 parser {spelling!r}",
                **parser_options,
            )
        print("PASS strict canonical context NW8 parser and 512-thread graph")

        # Any missing prerequisite must select the established NW4 launch,
        # even when either NW8 control requests the larger workgroup.
        affine_fallback_x = make_inputs(
            (0, 1025, 2049), heads, device, packed=True
        )
        for prerequisite, overrides in (
            ("operand-cache", {"operand_cache": "0"}),
            ("U-forward", {"u_forward": "0"}),
            ("V-forward", {"v_forward": "0"}),
        ):
            options = {**parser_options, **overrides}
            reference = run(parser_x, nw8="0", **options)
            candidate = run(parser_x, nw8="1", **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                reference,
                f"context NW8 missing {prerequisite}",
            )
            assert_topology(
                parser_x,
                nw8="1",
                expected_nw=4,
                expected_nodes=1,
                label=f"context NW8 missing {prerequisite}",
                **options,
            )

            affine_options = {
                "route": "affine",
                "group_chunks": 64,
                **overrides,
            }
            assert_topology(
                affine_fallback_x,
                nw8="1",
                expected_nw=4,
                expected_nodes=3,
                label=f"context affine NW8 missing {prerequisite}",
                **affine_options,
            )

            direct_options = {
                **options,
                "direct_nw": "8",
            }
            direct_candidate = run(
                parser_x, nw8=None, **direct_options
            )
            torch.cuda.synchronize(device)
            assert_result_same(
                direct_candidate,
                reference,
                f"context DIRECT_NW=8 missing {prerequisite}",
            )
            assert_topology(
                parser_x,
                nw8=None,
                expected_nw=4,
                expected_nodes=1,
                label=f"context DIRECT_NW=8 missing {prerequisite}",
                **direct_options,
            )
        print("PASS context NW8 cache/U/V prerequisite fallback to NW4")

        direct_nw8 = run(
            parser_x, nw8=None, direct_nw="8", **parser_options
        )
        torch.cuda.synchronize(device)
        assert_result_same(
            direct_nw8, parser_reference, "context DIRECT_NW=8"
        )
        assert_topology(
            parser_x,
            nw8=None,
            direct_nw="8",
            expected_nw=8,
            expected_nodes=1,
            label="context DIRECT_NW=8",
            **parser_options,
        )

        # The candidate changes only independent sequence-to-CTA ownership.
        # Exercise packed/dense metadata and all public state modes, requiring
        # raw storage-bit identity rather than merely numeric equality.
        tail_cases = (
            ("packed-n1", (1025,), True),
            ("packed-ragged", (0, 1, 65, 257, 1025), True),
            ("dense", (257, 257, 257, 257), False),
        )
        for case_name, seq_lens, packed in tail_cases:
            for (
                state_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                tail_x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = tail_x["initial_state"].clone()
                options = {"route": "direct", "nw8": "0"}
                reference = run(tail_x, tail_first="0", **options)
                candidate = run(tail_x, tail_first="1", **options)
                torch.cuda.synchronize(device)
                label = f"context tail-first {case_name}/{state_name}"
                assert_bitwise_same(
                    candidate[0], reference[0], f"{label} output"
                )
                if output_final_state:
                    if candidate[1] is None or reference[1] is None:
                        raise AssertionError(f"{label} omitted final state")
                    assert_bitwise_same(
                        candidate[1], reference[1], f"{label} final state"
                    )
                assert_bitwise_same(
                    tail_x["initial_state"],
                    initial_copy,
                    f"{label} input state mutated",
                )
                print(f"PASS bitwise {label}")

        tail_parser_reference = run(
            parser_x, nw8="0", tail_first=None, **parser_options
        )
        for spelling, expected_tail_first in (
            (None, 0),
            ("", 0),
            ("0", 0),
            ("01", 0),
            ("true", 0),
            ("1 ", 0),
            (" 1", 0),
            ("1", 1),
        ):
            candidate = run(
                parser_x,
                nw8="0",
                tail_first=spelling,
                **parser_options,
            )
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                tail_parser_reference,
                f"context tail-first parser {spelling!r}",
            )
            assert_topology(
                parser_x,
                nw8="0",
                tail_first=spelling,
                expected_nw=4,
                expected_nodes=1,
                expected_tail_first=expected_tail_first,
                label=f"context tail-first parser {spelling!r}",
                **parser_options,
            )
        print("PASS strict canonical context tail-first parser and 2D graph")

        # Exact "1" is inert unless the actual specialization is the pure
        # direct NW4/P0 cached-operands U+V-forwarding kernel.
        for prerequisite, overrides in (
            ("NW4", {"direct_nw": "2"}),
            ("NW4-from-NW1", {"direct_nw": "1"}),
            ("operand-cache", {"operand_cache": "0"}),
            ("U-forward", {"u_forward": "0"}),
            ("V-forward", {"v_forward": "0"}),
            ("P0", {"global_pipeline": "1"}),
            ("NW8", {"nw8": "1"}),
        ):
            options = {**parser_options, "nw8": "0", **overrides}
            reference = run(parser_x, tail_first="0", **options)
            candidate = run(parser_x, tail_first="1", **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                reference,
                f"context tail-first missing {prerequisite}",
            )
            expected_nw = 8 if prerequisite == "NW8" else int(
                options.get("direct_nw", "4")
            )
            assert_topology(
                parser_x,
                tail_first="1",
                expected_nw=expected_nw,
                expected_nodes=1,
                expected_tail_first=0,
                label=f"context tail-first missing {prerequisite}",
                **options,
            )
        print("PASS context tail-first prerequisite fallback")

        affine_x = make_inputs(
            (0, 1025, 2049), heads, device, packed=True
        )
        hybrid_x = make_inputs(
            (0, 1, 1024, 1025, 2049, 65, 513, 1537, 257),
            heads,
            device,
            packed=True,
        )
        for route, x, expected_nodes in (
            ("affine", affine_x, 3),
            ("hybrid", hybrid_x, 4),
        ):
            assert_topology(
                x,
                route=route,
                nw8="0",
                tail_first="1",
                expected_nw=4,
                expected_nodes=expected_nodes,
                expected_tail_first=0,
                label=f"context tail-first {route} fallback",
            )
        print("PASS context tail-first pure-direct route isolation")

        # Capture the multi-pass routes too: affine has B/A/replay, while
        # hybrid adds its independently guarded direct replay launch.
        assert_topology(
            affine_x,
            route="affine",
            nw8="1",
            expected_nw=8,
            expected_nodes=3,
            label="context NW8 affine graph",
        )
        assert_topology(
            hybrid_x,
            route="hybrid",
            nw8="1",
            expected_nw=8,
            expected_nodes=4,
            label="context NW8 hybrid graph",
        )
        print("PASS context NW8 direct/affine/hybrid graph topology")
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_context_u_forward_matrix(module, device: torch.device, heads: int):
    """A/B the bit-exact gfx950 context U-fragment register forwarding."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context U-forward A/B: gfx950 only")
        return

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    cases = (
        ("direct-packed-nw1", (0, 65, 257, 1025), True, "direct", 1, 64),
        ("direct-packed-nw2", (0, 65, 257, 1025), True, "direct", 2, 64),
        ("direct-packed-nw4", (0, 65, 257, 1025), True, "direct", 4, 64),
        ("direct-dense", (257, 257), False, "direct", 4, 64),
        ("affine-packed-g32", (0, 1025, 2049), True, "affine", 4, 32),
        ("affine-packed-g64", (0, 1025, 2049), True, "affine", 4, 64),
        ("affine-packed-g128", (0, 1025, 2049), True, "affine", 4, 128),
        ("affine-dense", (2049, 2049), False, "affine", 4, 64),
        (
            "hybrid-packed",
            (0, 1, 1024, 1025, 2049, 1, 65, 513, 1537),
            True,
            "hybrid",
            4,
            64,
        ),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        route: str,
        forward: str | None,
        direct_nw: int = 4,
        group_chunks: int = 64,
        operand_cache: bool = True,
        v_forward: str | None = "1",
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = str(direct_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = "2"
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = (
            "1" if operand_cache else "0"
        )
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if forward is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = forward
        if v_forward is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = v_forward

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    try:
        # Exercise U forwarding against both V transports.  With V=0 this
        # preserves the original u0v0/u1v0 rollback matrix; with V=1 it covers
        # the new production-default u0v1/u1v1 pair across every route/state.
        for v_forward, case in (
            (v_forward, case)
            for v_forward in ("0", "1")
            for case in cases
        ):
            (
                case_name,
                seq_lens,
                packed,
                route,
                direct_nw,
                group_chunks,
            ) = case
            for (
                state_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                options = {
                    "route": route,
                    "direct_nw": direct_nw,
                    "group_chunks": group_chunks,
                    "v_forward": v_forward,
                }
                label = f"{case_name}/{state_name}/v{v_forward}"
                off = run(x, forward="0", **options)
                on = run(x, forward="1", **options)
                torch.cuda.synchronize(device)
                for value_name, tensor in (
                    ("off-output", off[0]),
                    ("off-state", off[1]),
                    ("on-output", on[0]),
                    ("on-state", on[1]),
                ):
                    if tensor is not None and not bool(
                        torch.isfinite(tensor).all().item()
                    ):
                        raise RuntimeError(
                            f"context U-forward {label}: "
                            f"non-finite {value_name}"
                        )
                assert_result_same(
                    on,
                    off,
                    f"context U-forward {label}",
                )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"context U-forward {label} input mutated",
                )
                print(f"PASS bitwise context U-forward {label}")

        probe = make_inputs(
            (1, 1024, 1025, 2049, 65, 1, 513, 1537, 257),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        options = {"route": "hybrid", "direct_nw": 4, "group_chunks": 64}
        off = run(probe, forward="0", **options)
        on = run(probe, forward="1", **options)
        unset = run(probe, forward=None, **options)
        assert_result_same(unset, on, "context U-forward unset default")
        for value in ("", "01", "true", "1 "):
            candidate = run(probe, forward=value, **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                on,
                f"context U-forward noncanonical value {value!r}",
            )
        assert_result_same(off, on, "context U-forward exact-zero rollback")
        print("PASS default-on context U-forward environment parsing")

        # The producer-side activated operands change only where beta/decay
        # originate; forwarding the rounded U fragment must be exact for both
        # cached and raw operand paths.
        raw_off = run(probe, forward="0", operand_cache=False, **options)
        raw_on = run(probe, forward="1", operand_cache=False, **options)
        torch.cuda.synchronize(device)
        assert_result_same(raw_on, raw_off, "context U-forward uncached operands")
        print("PASS bitwise context U-forward uncached operands")
    finally:
        restore_env()


def check_context_v_forward_matrix(module, device: torch.device, heads: int):
    """A/B the bit-exact gfx950 context vnew-fragment forwarding."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context V-forward A/B: gfx950 only")
        return

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    cases = (
        ("direct-packed-nw1", (0, 65, 257, 1025), True, "direct", 1, 64),
        ("direct-packed-nw2", (0, 65, 257, 1025), True, "direct", 2, 64),
        ("direct-packed-nw4", (0, 65, 257, 1025), True, "direct", 4, 64),
        ("direct-dense", (257, 257), False, "direct", 4, 64),
        ("affine-packed-g32", (0, 1025, 2049), True, "affine", 4, 32),
        ("affine-packed-g64", (0, 1025, 2049), True, "affine", 4, 64),
        ("affine-packed-g128", (0, 1025, 2049), True, "affine", 4, 128),
        ("affine-dense", (2049, 2049), False, "affine", 4, 64),
        (
            "hybrid-packed",
            (0, 1, 1024, 1025, 2049, 1, 65, 513, 1537),
            True,
            "hybrid",
            4,
            64,
        ),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        route: str,
        forward: str | None,
        direct_nw: int = 4,
        group_chunks: int = 64,
        operand_cache: bool = True,
        u_forward: str = "0",
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = str(direct_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = "2"
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = "0"
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = (
            "1" if operand_cache else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = u_forward
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if forward is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = forward

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    try:
        for (
            case_name,
            seq_lens,
            packed,
            route,
            direct_nw,
            group_chunks,
        ) in cases:
            for (
                state_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                options = {
                    "route": route,
                    "direct_nw": direct_nw,
                    "group_chunks": group_chunks,
                }
                off = run(x, forward="0", **options)
                on = run(x, forward="1", **options)
                torch.cuda.synchronize(device)
                for value_name, tensor in (
                    ("off-output", off[0]),
                    ("off-state", off[1]),
                    ("on-output", on[0]),
                    ("on-state", on[1]),
                ):
                    if tensor is not None and not bool(
                        torch.isfinite(tensor).all().item()
                    ):
                        raise RuntimeError(
                            f"context V-forward {case_name}/{state_name}: "
                            f"non-finite {value_name}"
                        )
                assert_result_same(
                    on,
                    off,
                    f"context V-forward {case_name}/{state_name}",
                )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"context V-forward {case_name}/{state_name} input mutated",
                )
                print(f"PASS bitwise context V-forward {case_name}/{state_name}")

        probe = make_inputs(
            (1, 1024, 1025, 2049, 65, 1, 513, 1537, 257),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        options = {"route": "hybrid", "direct_nw": 4, "group_chunks": 64}
        off = run(probe, forward="0", **options)
        on = run(probe, forward="1", **options)
        unset = run(probe, forward=None, **options)
        assert_result_same(unset, on, "context V-forward unset default")
        for value in ("", "01", "true", "1 "):
            candidate = run(probe, forward=value, **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                on,
                f"context V-forward noncanonical value {value!r}",
            )
        assert_result_same(off, on, "context V-forward exact-zero rollback")
        print("PASS default-on context V-forward environment parsing")

        # The V fragment must remain exact both with U forwarding enabled and
        # when K1 does not publish activated beta/decay operands.
        for operand_cache in (True, False):
            off = run(
                probe,
                forward="0",
                u_forward="1",
                operand_cache=operand_cache,
                **options,
            )
            on = run(
                probe,
                forward="1",
                u_forward="1",
                operand_cache=operand_cache,
                **options,
            )
            torch.cuda.synchronize(device)
            label = "cached" if operand_cache else "uncached"
            assert_result_same(
                on,
                off,
                f"context U+V-forward {label} operands",
            )
        print("PASS bitwise context U+V-forward cached/uncached operands")
    finally:
        restore_env()


def check_context_lds_pipeline_matrix(
    module, device: torch.device, heads: int
):
    """A/B the bit-exact U+V-forward dual-LDS context pipeline."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context LDS-pipeline A/B: gfx950 only")
        return

    state_modes = (
        ("none", False, False, torch.float32),
        ("out-fp32", False, True, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("inout-fp32", True, True, torch.float32),
        ("inout-bf16", True, True, torch.bfloat16),
    )
    cases = (
        # Odd/even chunk counts exercise both arena parities, while one C16
        # proves that P1 omits the otherwise-useless final CTA barrier.
        (
            "direct-packed-nw1",
            (0, 1, 16, 17, 32, 33, 65, 257),
            True,
            "direct",
            1,
            64,
        ),
        (
            "direct-packed-nw2",
            (0, 1, 16, 17, 32, 33, 65, 257),
            True,
            "direct",
            2,
            64,
        ),
        (
            "direct-packed-nw4",
            (0, 1, 16, 17, 32, 33, 65, 257),
            True,
            "direct",
            4,
            64,
        ),
        ("direct-dense", (33, 33), False, "direct", 4, 64),
        (
            "affine-packed-g32",
            (0, 511, 512, 513, 1025),
            True,
            "affine",
            4,
            32,
        ),
        (
            "affine-packed-g64",
            (0, 1023, 1024, 1025, 2049),
            True,
            "affine",
            4,
            64,
        ),
        (
            "affine-packed-g128",
            (0, 2047, 2048, 2049, 4097),
            True,
            "affine",
            4,
            128,
        ),
        ("affine-dense", (1025, 1025), False, "affine", 4, 64),
        (
            "hybrid-packed",
            (0, 1, 16, 17, 1024, 1025, 2049, 65, 513),
            True,
            "hybrid",
            4,
            64,
        ),
    )
    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env():
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        route: str,
        pipeline: str | None,
        pipeline_b: str | None = None,
        pipeline_a: str | None = None,
        pipeline_replay: str | None = None,
        direct_nw: int = 4,
        group_chunks: int = 64,
        operand_cache: bool = True,
        u_forward: str = "1",
        v_forward: str = "1",
        tight_scan: str = "0",
    ):
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = str(direct_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = "2"
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = tight_scan
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = (
            "1" if operand_cache else "0"
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = u_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = v_forward
        route_name = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_name is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_name] = "1"
        if pipeline is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = pipeline
        for name, value in zip(
            _CONTEXT_LDS_PIPELINE_PASS_ENV,
            (pipeline_b, pipeline_a, pipeline_replay),
            strict=True,
        ):
            if value is not None:
                os.environ[name] = value

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        return out, final if x["output_final_state"] else None, workspace

    def assert_result_same(actual, reference, label: str):
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            assert actual[1] is not None
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    def pass_options(mask: str) -> dict[str, str]:
        if len(mask) != 3 or any(bit not in "01" for bit in mask):
            raise ValueError(f"invalid B/A/replay pipeline mask: {mask!r}")
        return {
            "pipeline_b": mask[0],
            "pipeline_a": mask[1],
            "pipeline_replay": mask[2],
        }

    def capture_context_roles(
        x,
        *,
        mask: str,
        global_pipeline: str = "0",
        u_forward: str = "1",
        v_forward: str = "1",
    ) -> dict[str, dict[str, object]]:
        configure(
            route="hybrid",
            pipeline=global_pipeline,
            u_forward=u_forward,
            v_forward=v_forward,
            direct_nw=4,
            group_chunks=64,
            **pass_options(mask),
        )
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, x, out, final, workspace)
        names = captured_graph_kernel_names(graph, device)
        roles = decode_context_pipeline_roles(names)
        graph.reset()
        return roles

    try:
        for (
            case_name,
            seq_lens,
            packed,
            route,
            direct_nw,
            group_chunks,
        ) in cases:
            for (
                state_name,
                has_initial_state,
                output_final_state,
                state_dtype,
            ) in state_modes:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=output_final_state,
                )
                initial_copy = x["initial_state"].clone()
                options = {
                    "route": route,
                    "direct_nw": direct_nw,
                    "group_chunks": group_chunks,
                }
                p0 = run(x, pipeline="0", **options)
                p1 = run(x, pipeline="1", **options)
                torch.cuda.synchronize(device)
                for value_name, tensor in (
                    ("p0-output", p0[0]),
                    ("p0-state", p0[1]),
                    ("p1-output", p1[0]),
                    ("p1-state", p1[1]),
                ):
                    if tensor is not None and not bool(
                        torch.isfinite(tensor).all().item()
                    ):
                        raise RuntimeError(
                            f"context LDS pipeline {case_name}/{state_name}: "
                            f"non-finite {value_name}"
                        )
                assert_result_same(
                    p1,
                    p0,
                    f"context LDS pipeline {case_name}/{state_name}",
                )
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    f"context LDS pipeline {case_name}/{state_name} "
                    "input mutated",
                )
                print(
                    "PASS bitwise context LDS pipeline "
                    f"{case_name}/{state_name}"
                )

        probe = make_inputs(
            (0, 1, 16, 17, 1024, 1025, 2049, 65, 513),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        options = {"route": "hybrid", "direct_nw": 4, "group_chunks": 64}
        unset = run(probe, pipeline=None, **options)
        for value in ("0", "01", "true", "1 "):
            candidate = run(probe, pipeline=value, **options)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate,
                unset,
                f"context LDS-pipeline noncanonical value {value!r}",
            )
        enabled = run(probe, pipeline="1", **options)
        torch.cuda.synchronize(device)
        assert_result_same(enabled, unset, "context LDS-pipeline canonical 1")
        print("PASS strict canonical context LDS-pipeline environment parsing")

        # Exercise every independent pass mask on dense, ordinary varlen, and
        # tight hybrid metadata for both fresh and resumed state.  All modes
        # preserve arithmetic order; the transport schedule must be bitwise.
        pass_cases = (
            (
                "dense",
                (1025, 1025),
                False,
                "affine",
                "0",
            ),
            (
                "varlen",
                (0, 511, 1025, 2049),
                True,
                "affine",
                "0",
            ),
            (
                "tight-hybrid",
                (0, 1, 16, 17, 1024, 1025, 2049, 65, 513),
                True,
                "hybrid",
                "1",
            ),
        )
        pass_states = (
            ("fresh", False, torch.float32),
            ("resume", True, torch.bfloat16),
        )
        for case_name, seq_lens, packed, route, tight_scan in pass_cases:
            for state_name, has_initial_state, state_dtype in pass_states:
                x = make_inputs(
                    seq_lens,
                    heads,
                    device,
                    packed=packed,
                    state_dtype=state_dtype,
                    has_initial_state=has_initial_state,
                    output_final_state=True,
                )
                initial_copy = x["initial_state"].clone()
                reference = None
                for mask_value in range(8):
                    mask = f"{mask_value:03b}"
                    candidate = run(
                        x,
                        route=route,
                        pipeline="0",
                        group_chunks=64,
                        tight_scan=tight_scan,
                        **pass_options(mask),
                    )
                    torch.cuda.synchronize(device)
                    if not bool(torch.isfinite(candidate[0]).all().item()):
                        raise RuntimeError(
                            "context LDS pass pipeline non-finite output: "
                            f"{case_name}/{state_name}/{mask}"
                        )
                    if candidate[1] is None or not bool(
                        torch.isfinite(candidate[1]).all().item()
                    ):
                        raise RuntimeError(
                            "context LDS pass pipeline non-finite state: "
                            f"{case_name}/{state_name}/{mask}"
                        )
                    if reference is None:
                        reference = candidate
                    else:
                        assert_result_same(
                            candidate,
                            reference,
                            "context LDS pass pipeline "
                            f"{case_name}/{state_name}/{mask}",
                        )
                assert reference is not None
                assert_same(
                    x["initial_state"],
                    initial_copy,
                    "context LDS pass pipeline "
                    f"{case_name}/{state_name} input mutated",
                )
                print(
                    "PASS bitwise context LDS pass factorial "
                    f"{case_name}/{state_name}/000..111"
                )

        # Unset, zero, and noncanonical per-pass values all retain mask 000.
        pass_unset = run(probe, pipeline="0", **options)
        pass_keywords = ("pipeline_b", "pipeline_a", "pipeline_replay")
        for keyword in pass_keywords:
            for value in ("0", "01", "true", "1 "):
                candidate = run(
                    probe,
                    pipeline="0",
                    **options,
                    **{keyword: value},
                )
                torch.cuda.synchronize(device)
                assert_result_same(
                    candidate,
                    pass_unset,
                    f"context LDS {keyword} fallback {value!r}",
                )
        print("PASS strict canonical context LDS per-pass environment parsing")

        # Inspect all four context graph launches.  Removing the final mangled
        # LDS_PIPELINE bool must make every role byte-identical across masks;
        # its value alone follows B/A/replay, with both replay launches sharing
        # the replay bit.  This makes accidental cross-pass dispatch visible.
        graph_probe = make_inputs(
            (1, 16, 1024, 1025, 2049, 65, 513, 1537, 257),
            heads,
            device,
            packed=True,
            has_initial_state=True,
            output_final_state=True,
        )
        role_bit = {
            "affine_b": 0,
            "affine_a": 1,
            "hybrid_direct_replay": 2,
            "affine_replay": 2,
        }
        graph_reference = capture_context_roles(graph_probe, mask="000")
        for mask_value in range(8):
            mask = f"{mask_value:03b}"
            roles = capture_context_roles(graph_probe, mask=mask)
            for role, expected_index in role_bit.items():
                actual = roles[role]
                expected = int(mask[expected_index])
                if (
                    actual["cached"] != 1
                    or actual["u_forward"] != 1
                    or actual["v_forward"] != 1
                    or actual["lds_pipeline"] != expected
                ):
                    raise AssertionError(
                        "captured context template bits mismatch for "
                        f"mask={mask}, role={role}: {actual}"
                    )
                if (
                    actual["normalized_name"]
                    != graph_reference[role]["normalized_name"]
                ):
                    raise AssertionError(
                        "context pass mask changed a kernel template field "
                        f"other than LDS_PIPELINE: mask={mask}, role={role}"
                    )
        global_roles = capture_context_roles(
            graph_probe, mask="000", global_pipeline="1"
        )
        if any(
            role["lds_pipeline"] != 1 for role in global_roles.values()
        ):
            raise AssertionError(
                "legacy global LDS pipeline no longer enables every pass"
            )
        for u_forward, v_forward in (("0", "0"), ("1", "0"), ("0", "1")):
            inactive_roles = capture_context_roles(
                graph_probe,
                mask="111",
                u_forward=u_forward,
                v_forward=v_forward,
            )
            if any(
                role["lds_pipeline"] != 0
                for role in inactive_roles.values()
            ):
                raise AssertionError(
                    "per-pass LDS pipeline bypassed U/V guard for "
                    f"U={u_forward}, V={v_forward}: {inactive_roles}"
                )
        print(
            "PASS captured context LDS 000..111 changes only each mode's "
            "final template bit"
        )

        # P1 is deliberately not instantiated for the other U/V transports.
        # Exercise their host dispatch guards as well as the kernel-side
        # static_assert; ELF symbol auditing separately proves only 11/P1 was
        # emitted.
        for u_forward, v_forward in (("0", "0"), ("1", "0"), ("0", "1")):
            transport_options = {
                **options,
                "u_forward": u_forward,
                "v_forward": v_forward,
            }
            inactive_p0 = run(probe, pipeline="0", **transport_options)
            inactive_p1 = run(probe, pipeline="1", **transport_options)
            torch.cuda.synchronize(device)
            assert_result_same(
                inactive_p1,
                inactive_p0,
                f"context LDS-pipeline inactive U={u_forward}/V={v_forward}",
            )
        print("PASS context LDS-pipeline restricted to U=V=true")

        uncached_p0 = run(
            probe, pipeline="0", operand_cache=False, **options
        )
        uncached_p1 = run(
            probe, pipeline="1", operand_cache=False, **options
        )
        torch.cuda.synchronize(device)
        assert_result_same(
            uncached_p1,
            uncached_p0,
            "context LDS-pipeline uncached operands",
        )
        print("PASS bitwise context LDS-pipeline uncached operands")
    finally:
        restore_env()


def check_context_persistent_matrix(
    module, device: torch.device, heads: int
) -> None:
    """Gate the packed-hybrid G64 persistent topology as one opt-in.

    The candidate is deliberately indivisible: its compact prefix, fused
    affine producer, compact scan, and affine replay must either all appear or
    all fall back.  Besides bitwise OFF/ON comparisons, captured graphs make
    that topology observable for every prerequisite.  The metadata-replay
    case keeps N and total tokens fixed while changing an all-short batch into
    a mixed prefill batch, which is the serving contract the device worklist
    is intended to support.
    """

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context persistent topology A/B: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
        _CONTEXT_PERSISTENT_ENV,
        _CONTEXT_PERSISTENT_ESTABLISHED_AB_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}

    def restore_env() -> None:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure(
        *,
        persistent: str | None,
        bt16_k1: str = "1",
        bt16_fused: str = "vector_x32",
        route: str = "hybrid",
        route_value: str = "1",
        group_chunks: int | str = 64,
        direct_nw: str = "4",
        nw8: str = "0",
        fused: str = "1",
        scan_nw: int | str = 2,
        b_stream: str = "0",
        a_gll: str = "0",
        b_phased: str = "0",
        ksplit: str = "0",
        tight: str = "0",
        operand_cache: str = "1",
        u_forward: str = "1",
        v_forward: str = "1",
        global_pipeline: str = "0",
        pipeline_mask: str = "000",
        explicit_k2: str | None = None,
    ) -> None:
        if len(pipeline_mask) != 3 or any(
            bit not in "01" for bit in pipeline_mask
        ):
            raise ValueError(f"invalid B/A/replay mask: {pipeline_mask!r}")
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = bt16_k1
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = bt16_fused
        os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
            group_chunks
        )
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = direct_nw
        os.environ[_CONTEXT_NW8_ENV] = nw8
        os.environ[_CONTEXT_AFFINE_AB_FUSED_ENV] = fused
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM"] = b_stream
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL"] = a_gll
        os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED"] = b_phased
        os.environ[_CONTEXT_SCAN_KSPLIT_ENV] = ksplit
        os.environ["FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN"] = tight
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = operand_cache
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = u_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = v_forward
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = global_pipeline
        for name, value in zip(
            _CONTEXT_LDS_PIPELINE_PASS_ENV, pipeline_mask, strict=True
        ):
            os.environ[name] = value
        route_env = {
            "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "hybrid": "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        }.get(route)
        if route_env is None:
            raise ValueError(f"unknown context route: {route}")
        os.environ[route_env] = route_value
        if explicit_k2 is not None:
            os.environ["FLASH_KDA_K2"] = explicit_k2
        if persistent is not None:
            os.environ[_CONTEXT_PERSISTENT_ENV] = persistent

    def run(x, **configuration):
        configure(**configuration)
        out, final, workspace = allocate(x)
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        workspace.fill_(0x6D)
        raw_call(module, x, out, final, workspace)
        return (
            out,
            final if x["output_final_state"] else None,
            workspace,
        )

    def assert_result_same(actual, reference, label: str) -> None:
        assert_same(actual[0], reference[0], f"{label} output mismatch")
        if reference[1] is not None:
            if actual[1] is None:
                raise AssertionError(f"{label} omitted final state")
            assert_same(actual[1], reference[1], f"{label} state mismatch")

    def capture_kernel_names(x, **configuration) -> list[str]:
        configure(**configuration)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(module, x, out, final, workspace)
        names = captured_graph_kernel_names(graph, device)
        graph.reset()
        return names

    persistent_symbols = (
        "k1_build_tile_prefix_hybrid_g64_compact_kernel",
        "k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel",
        "k2_kda_context_affine_scan_hybrid_g64_compact_"
        "grid_stride_nw2_kernel",
        "k2_kda_context_replay_hybrid_g64_grid_stride_nw4_kernel",
    )

    def persistent_nodes(names: list[str]) -> list[str]:
        return [
            name
            for name in names
            if any(symbol in name for symbol in persistent_symbols)
        ]

    def assert_active_topology(names: list[str], label: str) -> None:
        for symbol in persistent_symbols:
            matches = [name for name in names if symbol in name]
            if len(matches) != 1:
                raise AssertionError(
                    f"{label}: expected one {symbol}, got {matches!r}; "
                    f"all nodes={names!r}"
                )

        prefix_nodes = [
            name for name in names if "k1_build_tile_prefix" in name
        ]
        if len(prefix_nodes) != 1 or persistent_symbols[0] not in prefix_nodes[0]:
            raise AssertionError(
                f"{label}: persistent prefix did not wholly replace the "
                f"established prefix: {prefix_nodes!r}"
            )

        legacy_fused = [
            name
            for name in names
            if "k2_kda_context_affine_ab_fused_nw4_kernel" in name
        ]
        legacy_scan = [
            name
            for name in names
            if "k2_kda_context_affine_scan_nw4_kernel" in name
        ]
        legacy_context_roles = []
        direct_roles = []
        for name in names:
            match = _CONTEXT_PIPELINE_SYMBOL.search(name)
            if match is None:
                continue
            mode = int(match.group("mode"))
            group = int(match.group("group"))
            if mode == 2 and group == 1:
                direct_roles.append(name)
            else:
                legacy_context_roles.append(name)
        if len(direct_roles) != 1:
            raise AssertionError(
                f"{label}: direct short/empty owner count changed: "
                f"{direct_roles!r}"
            )
        if legacy_fused or legacy_scan or legacy_context_roles:
            raise AssertionError(
                f"{label}: persistent graph retained legacy affine nodes: "
                f"fused={legacy_fused!r}, scan={legacy_scan!r}, "
                f"roles={legacy_context_roles!r}"
            )

    def assert_fallback_graph(
        reference: list[str], candidate: list[str], label: str
    ) -> None:
        leaked = persistent_nodes(candidate)
        if leaked:
            raise AssertionError(
                f"{label}: partial persistent topology leaked: {leaked!r}"
            )
        if sorted(candidate) != sorted(reference):
            raise AssertionError(
                f"{label}: fallback graph differs from the established "
                f"graph\nreference={sorted(reference)!r}\n"
                f"candidate={sorted(candidate)!r}"
            )

    def compare_off_on(x, label: str, **configuration):
        initial_copy = x["initial_state"].clone()
        reference = run(x, persistent="0", **configuration)
        candidate = run(x, persistent="1", **configuration)
        torch.cuda.synchronize(device)
        for route_name, result in (
            ("established", reference),
            ("persistent", candidate),
        ):
            checked = result[:2] if result[1] is not None else result[:1]
            if any(
                not bool(torch.isfinite(tensor).all().item())
                for tensor in checked
            ):
                raise AssertionError(
                    f"{label}: {route_name} produced non-finite data"
                )
        assert_result_same(candidate, reference, label)
        assert_same(
            x["initial_state"],
            initial_copy,
            f"{label}: initial state mutated",
        )
        return reference

    def compare_fallback(x, label: str, **configuration) -> None:
        reference = compare_off_on(x, label, **configuration)
        del reference
        reference_names = capture_kernel_names(
            x, persistent="0", **configuration
        )
        candidate_names = capture_kernel_names(
            x, persistent="1", **configuration
        )
        assert_fallback_graph(reference_names, candidate_names, label)

    boundary_lens = (0, 1, 16, 65, 1024, 1025, 2049, 513, 1537)
    all_short_64 = (256,) * 64
    # Same N=64 and total=16,384 as all_short_64, but one sequence crosses
    # the 64-C16 direct threshold.  Host average-length routing cannot tell
    # these two batches apart.
    mixed_64 = (1025,) + (244,) * 50 + (243,) * 13
    if sum(all_short_64) != sum(mixed_64):
        raise AssertionError("persistent metadata replay totals diverged")

    try:
        state_modes = (
            ("none", False, False, torch.float32),
            ("out-fp32", False, True, torch.float32),
            ("out-bf16", False, True, torch.bfloat16),
            ("in-fp32", True, False, torch.float32),
            ("in-bf16", True, False, torch.bfloat16),
            ("inout-fp32", True, True, torch.float32),
            ("inout-bf16", True, True, torch.bfloat16),
        )
        for (
            mode_name,
            has_initial_state,
            output_final_state,
            state_dtype,
        ) in state_modes:
            x = make_inputs(
                boundary_lens,
                heads,
                device,
                packed=True,
                state_dtype=state_dtype,
                has_initial_state=has_initial_state,
                output_final_state=output_final_state,
            )
            compare_off_on(x, f"persistent boundary/{mode_name}")
        print("PASS persistent boundary all seven state dispatches bitwise")

        topology_cases = (
            ("batch64-all-short", all_short_64),
            ("batch64-one-long-same-budget", mixed_64),
            ("empty-and-single-long", (0,) * 4 + (1025,) + (1,) * 4),
            (
                "multiple-long-ragged",
                (0, 1024, 1025, 1, 2049, 65, 3073, 16, 513),
            ),
        )
        for label, seq_lens in topology_cases:
            x = make_inputs(
                seq_lens,
                heads,
                device,
                packed=True,
                state_dtype=torch.bfloat16,
                has_initial_state=True,
                output_final_state=True,
            )
            compare_off_on(x, f"persistent {label}")
        print(
            "PASS persistent all-short/empty/1024-1025/single-long/"
            "multiple-long topology cases bitwise"
        )

        graph_x = make_inputs(
            boundary_lens,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        active_names = capture_kernel_names(graph_x, persistent="1")
        assert_active_topology(active_names, "persistent active graph")
        print("PASS persistent active graph contains exactly four new stages")

        reference_names = capture_kernel_names(graph_x, persistent="0")
        reference_result = run(graph_x, persistent="0")
        for spelling in (None, "", "0", "01", "true", "1 ", " 1"):
            candidate_result = run(graph_x, persistent=spelling)
            torch.cuda.synchronize(device)
            assert_result_same(
                candidate_result,
                reference_result,
                f"persistent parser {spelling!r}",
            )
            candidate_names = capture_kernel_names(
                graph_x, persistent=spelling
            )
            assert_fallback_graph(
                reference_names,
                candidate_names,
                f"persistent parser {spelling!r}",
            )
        print("PASS persistent exact-'1' parser and full-graph fallback")

        dense_x = make_inputs(
            (1025, 1025),
            heads,
            device,
            packed=False,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        direct_x = make_inputs(
            (0, 1, 16, 65, 257, 513, 1024, 33, 9),
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        affine_x = make_inputs(
            (0, 1025, 2049, 1537),
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        tight_x = make_inputs(
            (1,) * 16 + (1025,) + (0,) * 16,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        fresh_x = make_inputs(
            boundary_lens,
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=False,
            output_final_state=True,
        )
        fallback_cases = (
            ("dense", dense_x, {"route": "affine"}),
            ("pure-direct", direct_x, {"route": "direct"}),
            ("pure-affine", affine_x, {"route": "affine"}),
            (
                "noncanonical-hybrid-route",
                graph_x,
                {"route_value": "true"},
            ),
            ("G32", graph_x, {"group_chunks": 32}),
            ("G128", graph_x, {"group_chunks": 128}),
            ("noncanonical-G64", graph_x, {"group_chunks": "64 "}),
            ("noncanonical-direct-NW4", graph_x, {"direct_nw": "04"}),
            ("noncanonical-scan-NW2", graph_x, {"scan_nw": "02"}),
            ("NW8", graph_x, {"nw8": "1"}),
            ("noncanonical-NW8", graph_x, {"nw8": "true"}),
            ("missing-fused-AB", graph_x, {"fused": "0"}),
            ("noncanonical-fused-AB", graph_x, {"fused": "true"}),
            ("scan-NW1", graph_x, {"scan_nw": 1}),
            ("scan-NW4", graph_x, {"scan_nw": 4}),
            ("scan-B-stream", graph_x, {"b_stream": "1"}),
            (
                "scan-A-GLL",
                graph_x,
                {"b_stream": "1", "a_gll": "1"},
            ),
            (
                "scan-B-phased",
                fresh_x,
                {"b_stream": "1", "b_phased": "1"},
            ),
            ("scan-K-split", graph_x, {"ksplit": "1"}),
            ("tight-scan", tight_x, {"tight": "1"}),
            ("missing-operand-cache", graph_x, {"operand_cache": "0"}),
            ("disabled-BT16-K1", graph_x, {"bt16_k1": "0"}),
            ("disabled-BT16-fusion", graph_x, {"bt16_fused": "0"}),
            ("missing-U-forward", graph_x, {"u_forward": "0"}),
            ("missing-V-forward", graph_x, {"v_forward": "0"}),
            (
                "noncanonical-U-forward",
                graph_x,
                {"u_forward": "true"},
            ),
            (
                "noncanonical-V-forward",
                graph_x,
                {"v_forward": "true"},
            ),
            (
                "noncanonical-operand-cache",
                graph_x,
                {"operand_cache": "true"},
            ),
            ("explicit-K2", graph_x, {"explicit_k2": "csplit64"}),
            (
                "global-LDS-pipeline",
                graph_x,
                {"global_pipeline": "1"},
            ),
            (
                "noncanonical-global-LDS-pipeline",
                graph_x,
                {"global_pipeline": "true"},
            ),
            ("B-LDS-pipeline", graph_x, {"pipeline_mask": "100"}),
            ("A-LDS-pipeline", graph_x, {"pipeline_mask": "010"}),
            (
                "replay-LDS-pipeline",
                graph_x,
                {"pipeline_mask": "001"},
            ),
        )
        for label, x, options in fallback_cases:
            compare_fallback(x, f"persistent missing prerequisite/{label}", **options)
        print(
            "PASS persistent dense/direct/affine/G/NW/cache/U/V/fused/scan/"
            "pipeline full-graph fallback matrix"
        )

        # Capture with every sequence direct, then move the same 1,025 tokens
        # into one sequence without changing N, total tokens, or any pointer.
        replay_x = make_inputs(
            all_short_64,
            heads,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        configure(persistent="1")
        replay_out, replay_final, replay_workspace = allocate(replay_x)
        raw_call(
            module,
            replay_x,
            replay_out,
            replay_final,
            replay_workspace,
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(
                module,
                replay_x,
                replay_out,
                replay_final,
                replay_workspace,
            )
        assert_active_topology(
            captured_graph_kernel_names(graph, device),
            "persistent changed-prefix graph",
        )
        graph.instantiate()

        offsets = [0]
        for length in mixed_64:
            offsets.append(offsets[-1] + length)
        replay_x["cu_seqlens"].copy_(
            torch.tensor(offsets, device=device, dtype=torch.int32)
        )
        changed_reference = run(replay_x, persistent="0")
        torch.cuda.synchronize(device)
        replay_out.fill_(float("nan"))
        replay_final.fill_(float("nan"))
        replay_workspace.fill_(0x93)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_same(
            replay_out,
            changed_reference[0],
            "persistent changed-prefix graph output mismatch",
        )
        assert_same(
            replay_final,
            changed_reference[1],
            "persistent changed-prefix graph state mismatch",
        )
        graph.reset()
        print(
            "PASS persistent graph replay after same-N/same-token prefix change"
        )

        stream_x = make_inputs(
            mixed_64,
            heads,
            device,
            packed=True,
            state_dtype=torch.float32,
            has_initial_state=True,
            output_final_state=True,
        )
        stream_reference = run(stream_x, persistent="0")
        torch.cuda.synchronize(device)
        configure(persistent="1")
        check_preallocated_graph(
            "raw",
            module,
            stream_x,
            stream_reference[0],
            stream_reference[1],
        )
        check_preallocated_multistream(
            "raw",
            module,
            stream_x,
            stream_reference[0],
            stream_reference[1],
        )
        print("PASS persistent graph and disjoint-workspace two-stream calls")
    finally:
        restore_env()


def check_context_graph_stream_matrix(module, device: torch.device):
    """Exercise graph replay and disjoint streams across gfx950 context modes."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP context graph/stream matrix: gfx950 only")
        return

    controlled_env = (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX950_BT16_K1",
        "FLASH_KDA_GFX950_BT16_FUSED",
        "FLASH_KDA_GFX950_CSPLIT64_MIN_T",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS",
        "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW",
        _CONTEXT_DIRECT_TAIL_FIRST_ENV,
        _CONTEXT_NW8_ENV,
        _CONTEXT_AFFINE_AB_FUSED_ENV,
        _CONTEXT_AFFINE_AB_STAGE_EARLY_ENV,
        _CONTEXT_EQUAL_DENSE_N4_G64_ENV,
        "FLASH_KDA_GFX950_CONTEXT_SCAN_NW",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL",
        "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED",
        _CONTEXT_SCAN_KSPLIT_ENV,
        "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN",
        "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE",
        "FLASH_KDA_GFX950_CONTEXT_U_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_V_FORWARD",
        "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE",
        *_CONTEXT_LDS_PIPELINE_PASS_ENV,
    )
    previous_env = {name: os.environ.get(name) for name in controlled_env}
    cases = (
        (
            "affine-single-8k",
            (8192,),
            "FLASH_KDA_GFX950_CONTEXT_AFFINE",
        ),
        (
            "direct-n16",
            (65,) * 16,
            "FLASH_KDA_GFX950_CONTEXT_DIRECT",
        ),
        (
            "hybrid-1024-1025",
            (1024, 1025) + (1195,) * 7,
            "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        ),
    )

    try:
        for name in controlled_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_BT16_K1"] = "1"
        os.environ["FLASH_KDA_GFX950_BT16_FUSED"] = "vector_x32"
        os.environ["FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_U_FORWARD"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_V_FORWARD"] = "1"

        route_env = (
            "FLASH_KDA_GFX950_CONTEXT_DIRECT",
            "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            "FLASH_KDA_GFX950_CONTEXT_HYBRID",
        )
        for label, seq_lens, force_env in cases:
            for name in route_env:
                os.environ.pop(name, None)
            os.environ[force_env] = "1"
            x = make_inputs(seq_lens, 1, device, packed=True)
            p0_reference = None
            schedules = [
                ("P0", "0", "0"),
                ("P1", "1", "0"),
            ]
            if force_env == "FLASH_KDA_GFX950_CONTEXT_DIRECT":
                schedules.insert(1, ("tail-first-P0", "0", "1"))
            for schedule, pipeline, tail_first in schedules:
                os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = pipeline
                os.environ[_CONTEXT_DIRECT_TAIL_FIRST_ENV] = tail_first
                x, reference_out, reference_final = check_raw_vs_descriptor(
                    module, x, f"context-{label}-{schedule}"
                )
                if p0_reference is None:
                    p0_reference = (reference_out, reference_final)
                else:
                    assert_same(
                        reference_out,
                        p0_reference[0],
                        f"context {label} graph P0/{schedule} output mismatch",
                    )
                    if p0_reference[1] is not None:
                        assert reference_final is not None
                        assert_same(
                            reference_final,
                            p0_reference[1],
                            f"context {label} graph P0/{schedule} state mismatch",
                        )
                check_preallocated_graph(
                    "raw", module, x, reference_out, reference_final
                )
                check_preallocated_multistream(
                    "raw", module, x, reference_out, reference_final
                )
                print(
                    "PASS context graph/stream route: "
                    f"{label}/{schedule}"
                )
            del x, reference_out, reference_final, p0_reference
            torch.cuda.empty_cache()

        # Capture tail-first with the long sequence last, then move the same
        # tokens to sequence zero.  N, total tokens, pointers, grid, and graph
        # topology stay fixed; only cu_seqlens contents change between replays.
        changed_prefix_x = make_inputs(
            (1, 1, 1, 1025),
            1,
            device,
            packed=True,
            state_dtype=torch.bfloat16,
            has_initial_state=True,
            output_final_state=True,
        )
        for name in route_env:
            os.environ.pop(name, None)
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT"] = "1"
        os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = "4"
        os.environ["FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE"] = "0"
        os.environ[_CONTEXT_DIRECT_TAIL_FIRST_ENV] = "1"
        graph_out, graph_final, graph_workspace = allocate(changed_prefix_x)
        raw_call(
            module,
            changed_prefix_x,
            graph_out,
            graph_final,
            graph_workspace,
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            raw_call(
                module,
                changed_prefix_x,
                graph_out,
                graph_final,
                graph_workspace,
            )
        names = captured_graph_kernel_names(graph, device)
        direct_records = [
            match
            for name in names
            if (match := _CONTEXT_PIPELINE_SYMBOL.search(name)) is not None
        ]
        if len(direct_records) != 1 or any(
            int(direct_records[0].group(field)) != value
            for field, value in (
                ("mode", 2),
                ("direct", 1),
                ("nw", 4),
                ("lds_pipeline", 0),
                ("tail_first", 1),
            )
        ):
            raise AssertionError(
                "tail-first changed-prefix graph captured the wrong context "
                f"specialization: {names!r}"
            )
        graph.instantiate()

        changed_offsets = (0, 1025, 1026, 1027, 1028)
        changed_prefix_x["cu_seqlens"].copy_(
            torch.tensor(changed_offsets, device=device, dtype=torch.int32)
        )
        reference_out, reference_final, reference_workspace = allocate(
            changed_prefix_x
        )
        raw_call(
            module,
            changed_prefix_x,
            reference_out,
            reference_final,
            reference_workspace,
        )
        torch.cuda.synchronize(device)
        graph_out.fill_(float("nan"))
        graph_final.fill_(float("nan"))
        graph_workspace.fill_(0xA7)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_bitwise_same(
            graph_out,
            reference_out,
            "tail-first changed-prefix graph output",
        )
        assert_bitwise_same(
            graph_final,
            reference_final,
            "tail-first changed-prefix graph final state",
        )
        graph.reset()
        print(
            "PASS tail-first graph replay after same-N/same-token prefix "
            "rotation"
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_raw_v2_zero_equivalence(module, x, label: str):
    """Prove raw-v2 bound zero preserves both legacy ABI results."""

    x, reference_out, reference_final = check_raw_vs_descriptor(
        module, x, label
    )
    initial_copy = x["initial_state"].clone()
    v2_out, v2_final, v2_workspace = allocate(x)
    raw_v2_call(module, x, v2_out, v2_final, v2_workspace, 0)
    torch.cuda.synchronize(x["q"].device)
    assert_same(
        v2_out,
        reference_out,
        f"{label}: raw-v2(bound=0)/legacy output mismatch",
    )
    if reference_final is not None:
        assert_same(
            v2_final,
            reference_final,
            f"{label}: raw-v2(bound=0)/legacy final-state mismatch",
        )
    assert_same(
        x["initial_state"],
        initial_copy,
        f"{label}: raw-v2(bound=0) mutated initial state",
    )
    print(f"PASS raw-v2 bound=0 vs raw-v1/descriptor bitwise: {label}")
    return x, reference_out, reference_final


def _assert_context_route_topology(
    kernel_names: list[str],
    expected: tuple[tuple[int, int, int], ...],
    label: str,
) -> None:
    """Check (group, mode, NW) context roles in a retained HIP graph."""

    actual = []
    for name in kernel_names:
        match = _CONTEXT_PIPELINE_SYMBOL.search(name)
        if match is None:
            continue
        actual.append(
            (
                int(match.group("group")),
                int(match.group("mode")),
                int(match.group("nw")),
            )
        )
    if sorted(actual) != sorted(expected):
        raise AssertionError(
            f"{label}: expected context (group, mode, NW) roles "
            f"{sorted(expected)!r}, got {sorted(actual)!r}; "
            f"all kernels={kernel_names!r}"
        )


def _assert_mixed_boundary_direct_topology(
    kernel_names: list[str],
    *,
    prefixless: bool,
    nw: int,
    flat: bool,
    label: str,
) -> None:
    """Require the matched K1/prefix/K2 graph for one boundary recipe."""

    expected_nodes = 2 if prefixless else 3
    if len(kernel_names) != expected_nodes:
        raise AssertionError(
            f"{label}: expected exactly {expected_nodes} kernel nodes, got "
            f"{kernel_names!r}"
        )

    prefix_nodes = [
        name for name in kernel_names if "k1_build_tile_prefix" in name
    ]
    expected_prefix_nodes = 0 if prefixless else 1
    if len(prefix_nodes) != expected_prefix_nodes:
        raise AssertionError(
            f"{label}: expected {expected_prefix_nodes} prefix nodes, got "
            f"{prefix_nodes!r}; all kernels={kernel_names!r}"
        )

    k1_names = [
        name for name in kernel_names if "k1_kda_bt16_fused_kernel" in name
    ]
    if len(k1_names) != 1:
        raise AssertionError(
            f"{label}: expected one fused K1 node, got {k1_names!r}; "
            f"all kernels={kernel_names!r}"
        )
    k1_match = _K1_FUSED_SYMBOL.search(k1_names[0])
    if k1_match is None:
        raise AssertionError(
            f"{label}: cannot decode fused K1 template flags: {k1_names[0]!r}"
        )
    k1_flags = tuple(
        int(value) for value in re.findall(r"Lb([01])E", k1_match["flags"])
    )
    if len(k1_flags) not in (9, 10, 11, 12, 13) or k1_flags[0] != 1:
        raise AssertionError(
            f"{label}: fused K1 is not the packed "
            "nine-through-thirteen-flag ABI: "
            f"flags={k1_flags!r}, name={k1_names[0]!r}"
        )
    if bool(k1_flags[8]) != prefixless:
        raise AssertionError(
            f"{label}: fused K1 prefixless={k1_flags[8]}, expected "
            f"{int(prefixless)}: {k1_names[0]!r}"
        )
    if len(k1_flags) >= 10 and k1_flags[9] != 0:
        raise AssertionError(
            f"{label}: packed fused K1 selected dense-N1/full-C16: "
            f"flags={k1_flags!r}, name={k1_names[0]!r}"
        )
    if len(k1_flags) >= 11 and k1_flags[10] != 0:
        raise AssertionError(
            f"{label}: packed fused K1 selected dense-N1 padded solve: "
            f"flags={k1_flags!r}, name={k1_names[0]!r}"
        )
    if len(k1_flags) >= 12 and k1_flags[11] != 0:
        raise AssertionError(
            f"{label}: packed fused K1 selected dense-N1 early beta: "
            f"flags={k1_flags!r}, name={k1_names[0]!r}"
        )
    if len(k1_flags) >= 13 and k1_flags[12] != 0:
        raise AssertionError(
            f"{label}: equal-head route selected the GVA K1 specialization: "
            f"flags={k1_flags!r}, name={k1_names[0]!r}"
        )

    context = []
    for name in kernel_names:
        match = _CONTEXT_PIPELINE_SYMBOL.search(name)
        if match is not None:
            context.append((name, match))
    if len(context) != 1:
        raise AssertionError(
            f"{label}: expected one direct K2 node, got {context!r}; "
            f"all kernels={kernel_names!r}"
        )
    name, match = context[0]
    expected_fields = {
        "group": 1,
        "mode": 2,
        "vl": 1,
        "direct": 1,
        "nw": nw,
        "cached": 1,
        "u_forward": 1,
        "v_forward": 1,
        "lds_pipeline": 0,
        "tail_first": int(flat),
        "prefixless": int(prefixless),
    }
    mismatches = {
        field: (int(match.group(field)), expected)
        for field, expected in expected_fields.items()
        if int(match.group(field)) != expected
    }
    if mismatches:
        raise AssertionError(
            f"{label}: direct K2 specialization mismatch {mismatches!r}: "
            f"{name!r}"
        )


def _capture_raw_v2_kernel_names(
    module,
    x,
    max_seqlen_upper_bound: int,
) -> list[str]:
    """Capture one raw-v2 call and return its HIP kernel symbols."""

    device = x["q"].device
    out, final, workspace = allocate(x)
    raw_v2_call(
        module,
        x,
        out,
        final,
        workspace,
        max_seqlen_upper_bound,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            raw_v2_call(
                module,
                x,
                out,
                final,
                workspace,
                max_seqlen_upper_bound,
            )
        return captured_graph_kernel_names(graph, device)
    finally:
        graph.reset()


def check_raw_v2_invalid_bounds(
    module, device: torch.device, heads: int
) -> None:
    """Reject raw-v2 bounds outside the packed and dense contracts."""

    packed = make_inputs((3, 4), heads, device, packed=True)
    dense = make_inputs((7, 7), heads, device, packed=False)

    def reject(x, bound: int, label: str) -> None:
        out, final, workspace = allocate(x)
        expect_rejection(
            label,
            lambda: raw_v2_call(
                module, x, out, final, workspace, bound
            ),
        )

    # Packed total=7/N=2 permits only zero or a positive hint in [4, 7].
    for bound, label in (
        (-1, "raw-v2 packed negative bound"),
        (3, "raw-v2 packed bound below ceil(total/N)"),
        (8, "raw-v2 packed bound above total"),
        (1 << 31, "raw-v2 packed bound above native int ABI"),
    ):
        reject(packed, bound, label)

    # Dense validation is deliberately against per-sequence T=7, not B*T.
    for bound, label in (
        (6, "raw-v2 dense bound below T"),
        (8, "raw-v2 dense bound above T"),
        (14, "raw-v2 dense bound equal B*T instead of T"),
    ):
        reject(dense, bound, label)
    print("PASS raw-v2 packed/dense invalid-bound rejection matrix")


def check_raw_v2_hint_policy_matrix(
    module, device: torch.device, heads: int
) -> None:
    """Validate gfx950's three production hint routes and bucket replay."""

    if flash_kda._device_arch(device) != "gfx950":
        print("SKIP raw-v2 hint route/graph matrix: gfx950 only")
        return

    previous_env = {
        name: os.environ.get(name) for name in _RAW_V2_POLICY_ENV
    }

    def configure(
        *,
        route: str | None = None,
        group_chunks: int | None = None,
        direct_nw: int | str | None = None,
        nw1_flat: str | None = None,
        prefixless: str | None = None,
        fused: str | None = None,
        scan_nw: int | None = None,
        ksplit: str | None = None,
    ) -> None:
        for name in _RAW_V2_POLICY_ENV:
            os.environ.pop(name, None)
        if route is not None:
            route_environment = {
                "direct": "FLASH_KDA_GFX950_CONTEXT_DIRECT",
                "affine": "FLASH_KDA_GFX950_CONTEXT_AFFINE",
            }.get(route)
            if route_environment is None:
                raise ValueError(f"unsupported raw-v2 reference route: {route}")
            os.environ[route_environment] = "1"
        if group_chunks is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS"] = str(
                group_chunks
            )
        if direct_nw is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_DIRECT_NW"] = str(
                direct_nw
            )
        if nw1_flat is not None:
            os.environ[_CONTEXT_DIRECT_NW1_FLAT_ENV] = nw1_flat
        if prefixless is not None:
            os.environ[_CONTEXT_DIRECT_PREFIXLESS_ENV] = prefixless
        if fused is not None:
            os.environ[_CONTEXT_AFFINE_AB_FUSED_ENV] = fused
        if scan_nw is not None:
            os.environ["FLASH_KDA_GFX950_CONTEXT_SCAN_NW"] = str(scan_nw)
        if ksplit is not None:
            os.environ[_CONTEXT_SCAN_KSPLIT_ENV] = ksplit

    direct_topology = ((1, 2, 4),)
    direct_nw1_topology = ((1, 2, 1),)
    pure_affine_g64_topology = (
        (64, 0, 4),
        (64, 1, 4),
        (64, 2, 4),
    )
    fused_affine_g64_topology = ((64, 2, 4),)
    cases = (
        (
            "batch-16x1k-direct",
            (1024,) * 16,
            1024,
            False,
            "direct",
            None,
            direct_topology,
            False,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "mixed-15d-prefill-1025-direct",
            (1,) * 15 + (1025,),
            1025,
            False,
            "direct",
            None,
            direct_nw1_topology,
            True,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "n8-ragged-16k-pure-affine-g64",
            (127, 255, 511, 1023, 2047, 3073, 4095, 5253),
            5253,
            False,
            "affine",
            64,
            pure_affine_g64_topology,
            False,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "n4-resume-4x4k-pure-affine-g64",
            (4096,) * 4,
            4096,
            True,
            "affine",
            64,
            fused_affine_g64_topology,
            False,
            64,
            64,
            "1",
            2,
            "1",
        ),
    )

    try:
        for (
            label,
            seq_lens,
            bound,
            has_initial_state,
            reference_route,
            reference_group,
            expected_topology,
            expected_prefixless_flat,
            expected_fused_group,
            expected_ksplit_group,
            reference_fused,
            reference_scan_nw,
            reference_ksplit,
        ) in cases:
            x = make_inputs(
                seq_lens,
                heads,
                device,
                packed=True,
                has_initial_state=has_initial_state,
            )
            initial_copy = x["initial_state"].clone()

            # The forced raw-v1 path and descriptor are an independent result
            # oracle for the exact route selected by the zero-env raw-v2 hint.
            configure(
                route=reference_route,
                group_chunks=reference_group,
                fused=reference_fused,
                scan_nw=reference_scan_nw,
                ksplit=reference_ksplit,
            )
            _, reference_out, reference_final = check_raw_vs_descriptor(
                module, x, f"raw-v2 {label} forced reference"
            )

            configure()
            hinted_out, hinted_final, hinted_workspace = allocate(x)
            raw_v2_call(
                module,
                x,
                hinted_out,
                hinted_final,
                hinted_workspace,
                bound,
            )
            torch.cuda.synchronize(device)
            assert_same(
                hinted_out,
                reference_out,
                f"raw-v2 {label} hinted-route output mismatch",
            )
            if reference_final is not None:
                assert_same(
                    hinted_final,
                    reference_final,
                    f"raw-v2 {label} hinted-route state mismatch",
                )
            assert_same(
                x["initial_state"],
                initial_copy,
                f"raw-v2 {label} mutated initial state",
            )

            kernel_names = _capture_raw_v2_kernel_names(
                module, x, bound
            )
            _assert_context_route_topology(
                kernel_names,
                expected_topology,
                f"raw-v2 {label}",
            )
            if expected_prefixless_flat:
                _assert_mixed_boundary_direct_topology(
                    kernel_names,
                    prefixless=True,
                    nw=1,
                    flat=True,
                    label=f"raw-v2 {label}",
                )

                # Ordering is invisible to the aggregate host guard but is
                # consumed by both prefixless device mappings.  Retain all
                # four measured boundary layouts in the graph contract while
                # keeping this as one top-level result/PASS case.
                production_shapes = (
                    ("decode-first-1024", (1,) * 15 + (1024,), 1024),
                    ("prefill-first-1024", (1024,) + (1,) * 15, 1024),
                    ("decode-first-1025", (1,) * 15 + (1025,), 1025),
                    ("prefill-first-1025", (1025,) + (1,) * 15, 1025),
                )
                for variant, variant_lens, variant_bound in production_shapes:
                    if variant_lens == seq_lens and variant_bound == bound:
                        continue
                    configure()
                    variant_x = make_inputs(
                        variant_lens,
                        heads,
                        device,
                        packed=True,
                        has_initial_state=has_initial_state,
                    )
                    variant_names = _capture_raw_v2_kernel_names(
                        module, variant_x, variant_bound
                    )
                    _assert_context_route_topology(
                        variant_names,
                        direct_nw1_topology,
                        f"raw-v2 mixed-boundary {variant}",
                    )
                    _assert_mixed_boundary_direct_topology(
                        variant_names,
                        prefixless=True,
                        nw=1,
                        flat=True,
                        label=f"raw-v2 mixed-boundary {variant}",
                    )

                # Each explicit scheduling/mapping axis must break the
                # indivisible zero-environment pair.  PREFIXLESS=0 restores
                # NW4; explicit DIRECT_NW=1 retains 2-D NW1; explicit
                # NW1_FLAT=1 retains the flat schedule but restores the
                # prefix node and non-prefixless K1/K2 mappings.
                rollback_recipes = (
                    (
                        "prefixless-0",
                        {"prefixless": "0"},
                        4,
                        False,
                    ),
                    (
                        "explicit-direct-nw1",
                        {"direct_nw": "1"},
                        1,
                        False,
                    ),
                    (
                        "explicit-nw1-flat",
                        {"nw1_flat": "1"},
                        1,
                        True,
                    ),
                )
                for rollback, environment, rollback_nw, rollback_flat in (
                    rollback_recipes
                ):
                    configure(**environment)
                    rollback_names = _capture_raw_v2_kernel_names(
                        module, x, bound
                    )
                    _assert_context_route_topology(
                        rollback_names,
                        ((1, 2, rollback_nw),),
                        f"raw-v2 mixed-boundary rollback {rollback}",
                    )
                    _assert_mixed_boundary_direct_topology(
                        rollback_names,
                        prefixless=False,
                        nw=rollback_nw,
                        flat=rollback_flat,
                        label=(
                            "raw-v2 mixed-boundary rollback " + rollback
                        ),
                    )
                configure()
            if expected_fused_group is not None:
                fused_symbol = "k2_kda_context_affine_ab_fused_nw4_kernel"
                fused_names = [
                    name for name in kernel_names if fused_symbol in name
                ]
                if len(fused_names) != 1 or re.search(
                    rf"{fused_symbol}I(?:Li)?{expected_fused_group}E",
                    fused_names[0],
                ) is None:
                    raise AssertionError(
                        f"raw-v2 {label}: expected one fused G"
                        f"{expected_fused_group} producer, got "
                        f"{fused_names!r}; all kernels={kernel_names!r}"
                    )
            if expected_ksplit_group is not None:
                ksplit_symbol = (
                    "k2_kda_context_affine_scan_ksplit_wg4_kernel"
                )
                ksplit_names = [
                    name for name in kernel_names if ksplit_symbol in name
                ]
                if len(ksplit_names) != 1 or re.search(
                    rf"ksplit_wg4_kernelI(?:Li)?{expected_ksplit_group}E",
                    ksplit_names[0],
                ) is None:
                    raise AssertionError(
                        f"raw-v2 {label}: expected one G"
                        f"{expected_ksplit_group} K-split scan, got "
                        f"{ksplit_names!r}; all kernels={kernel_names!r}"
                    )
            print(f"PASS raw-v2 hinted route/result: {label}")

        # The bound is a capture-time host bucket promise.  Change only the
        # device prefix within that fixed bucket and replay the retained graph;
        # N, total tokens, pointers, and the 1K host bound all stay unchanged.
        configure()
        batch_x = make_inputs(
            (512,) * 16,
            heads,
            device,
            packed=True,
            has_initial_state=False,
        )
        batch_bound = 1024
        graph_out, graph_final, graph_workspace = allocate(batch_x)
        raw_v2_call(
            module,
            batch_x,
            graph_out,
            graph_final,
            graph_workspace,
            batch_bound,
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(graph):
                raw_v2_call(
                    module,
                    batch_x,
                    graph_out,
                    graph_final,
                    graph_workspace,
                    batch_bound,
                )
            _assert_context_route_topology(
                captured_graph_kernel_names(graph, device),
                direct_topology,
                "raw-v2 static-bucket captured graph",
            )
            graph.instantiate()

            changed_lens = (1024,) * 8 + (0,) * 8
            if (
                len(changed_lens) != batch_x["N"]
                or sum(changed_lens) != batch_x["q"].shape[1]
                or max(changed_lens) > batch_bound
            ):
                raise AssertionError(
                    "invalid raw-v2 static-bucket replay fixture"
                )
            offsets = [0]
            for length in changed_lens:
                offsets.append(offsets[-1] + length)
            batch_x["cu_seqlens"].copy_(
                torch.tensor(offsets, device=device, dtype=torch.int32)
            )

            reference_out, reference_final, reference_workspace = allocate(
                batch_x
            )
            raw_v2_call(
                module,
                batch_x,
                reference_out,
                reference_final,
                reference_workspace,
                batch_bound,
            )
            torch.cuda.synchronize(device)
            graph_out.fill_(float("nan"))
            graph_final.fill_(float("nan"))
            graph_workspace.fill_(0xA7)
            graph.replay()
            torch.cuda.synchronize(device)
            assert_same(
                graph_out,
                reference_out,
                "raw-v2 static-bucket graph replay output mismatch",
            )
            assert_same(
                graph_final,
                reference_final,
                "raw-v2 static-bucket graph replay state mismatch",
            )
            print(
                "PASS raw-v2 static 1K bucket graph changed-prefix replay"
            )
        finally:
            graph.reset()
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_invalid_metadata(module, x):
    out, final, workspace = allocate(x)
    args = list(raw_args(x, out, final, workspace))
    expect_rejection(
        "null q pointer", lambda: module.flash_kda_fwd_hip_raw(0, *args[1:])
    )
    too_small = list(args)
    too_small[16] = workspace.nbytes - 1
    expect_rejection(
        "short workspace", lambda: module.flash_kda_fwd_hip_raw(*too_small)
    )
    invalid_scale = list(args)
    invalid_scale[17] = 0.0
    expect_rejection(
        "nonpositive scale", lambda: module.flash_kda_fwd_hip_raw(*invalid_scale)
    )
    bad_dense_relation = list(args)
    bad_dense_relation[21] = False
    bad_dense_relation[12] = 2
    expect_rejection(
        "dense N != B", lambda: module.flash_kda_fwd_hip_raw(*bad_dense_relation)
    )


def check_two_devices(module, tokens: int, heads: int):
    if torch.cuda.device_count() < 2:
        print("SKIP multi-GPU: expose at least two GPUs")
        return
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    x0 = make_inputs((tokens,), heads, device0)
    x1 = make_inputs((tokens,), heads, device1)
    check_raw_vs_descriptor(module, x0, "cuda:0")
    check_raw_vs_descriptor(module, x1, "cuda:1")
    print("PASS correct launches after cuda:0 -> cuda:1 switching")

    with torch.cuda.device(device1):
        out1, final1, workspace1 = allocate(x1)
        args1 = raw_args(x1, out1, final1, workspace1)
    with torch.cuda.device(device0):
        expect_rejection(
            "active-device mismatch",
            lambda: module.flash_kda_fwd_hip_raw(*args1),
        )

    with torch.cuda.device(device0):
        out0, final0, workspace0 = allocate(x0)
        args0 = list(raw_args(x0, out0, final0, workspace0))
    with torch.cuda.device(device1):
        # Legacy/default stream handles may be the universal null handle, so use
        # a real non-default stream whose owning device is unambiguous.
        foreign_stream = torch.cuda.Stream(device=device1)
        args0[-1] = foreign_stream.cuda_stream
    with torch.cuda.device(device0):
        expect_rejection(
            "foreign-device stream",
            lambda: module.flash_kda_fwd_hip_raw(*args0),
        )
    print("PASS multi-GPU device/stream rejection")


def main():
    global flash_kda, get_module
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    check_cpu_parameter_validation()
    check_packed_n1_entry_dense_normalization_static()
    check_bt16_dense_n1_all_full_c16_policy_static()
    check_bt16_dense_n1_padded_solve_policy_static()
    check_bt16_dense_n1_early_beta_policy_static()
    check_context_forward_policy_static()
    check_context_lds_pipeline_pass_policy_static()
    check_context_nw8_policy_static()
    check_context_direct_tail_first_policy_static()
    check_context_affine_ab_fused_policy_static()
    check_context_scan_ksplit_policy_static()
    check_context_persistent_policy_static()
    check_gva_whole_route_policy_static()
    if args.tokens <= 0 or args.heads <= 0:
        raise ValueError("--tokens and --heads must be positive")
    if args.cpu_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")

    import aiter.ops.flash_kda as flash_kda_module
    from aiter.jit.core import get_module as get_jit_module

    flash_kda = flash_kda_module
    get_module = get_jit_module

    device = torch.device("cuda:0")
    # Enter through the decorated descriptor op first.  Under AITER_REBUILD=1
    # this ensures the raw symbol fetched below belongs to the just-built
    # module instead of a stale extension loaded before the rebuild.
    warmup = make_inputs((min(args.tokens, 128),), min(args.heads, 2), device)
    warmup_buffers = allocate(warmup)
    descriptor_call(warmup, *warmup_buffers)
    torch.cuda.synchronize(device)
    module = get_module(flash_kda.MD_NAME)
    check_module_abi_surface(module)
    check_raw_v3_gva_vs_descriptor(module, device)

    x = make_inputs(split_packed_tokens(args.tokens), args.heads, device)
    x, reference_out, reference_final = check_raw_v2_zero_equivalence(
        module, x, "packed-primary"
    )
    check_raw_v2_invalid_bounds(module, device, min(args.heads, 2))
    check_raw_v2_hint_policy_matrix(module, device, min(args.heads, 2))
    check_state_layout_matrix(module, device, min(args.heads, 2))
    check_forced_hybrid_route(module, device, min(args.heads, 2))
    check_context_tight_scan_matrix(module, device, min(args.heads, 2))
    check_context_scan_b_stream_matrix(module, device, min(args.heads, 2))
    check_context_scan_a_gll_matrix(module, device, min(args.heads, 2))
    check_context_scan_b_phased_matrix(module, device, min(args.heads, 2))
    check_context_scan_ksplit_matrix(module, device, min(args.heads, 2))
    check_plain_beta_cache_matrix(module, device, min(args.heads, 2))
    check_forced_csplit_empty_state_matrix(
        module, device, min(args.heads, 2)
    )
    check_plain_decay_cache_matrix(module, device, min(args.heads, 2))
    check_plain_postprep_opt_matrix(module, device, min(args.heads, 2))
    check_plain_internal_layout_matrix(module, device, min(args.heads, 2))
    check_context_affine_ab_fused_matrix(module, device, min(args.heads, 2))
    check_context_nw8_matrix(module, device, min(args.heads, 2))
    check_context_u_forward_matrix(module, device, min(args.heads, 2))
    check_context_v_forward_matrix(module, device, min(args.heads, 2))
    check_context_lds_pipeline_matrix(module, device, min(args.heads, 2))
    check_context_persistent_matrix(module, device, min(args.heads, 2))
    check_context_graph_stream_matrix(module, device)

    # The public checks below must use the production raw fast path.  The
    # descriptor path was exercised directly above and must not be mistaken
    # for a public-call reference again.
    if flash_kda._get_raw_pointer_op() is None:
        raise RuntimeError("allocation-owning public wrapper did not find raw ABI")
    for abi in ("raw", "descriptor"):
        check_preallocated_graph(abi, module, x, reference_out, reference_final)
    check_public_graph(x, reference_out, reference_final)
    for abi in ("raw", "descriptor"):
        check_preallocated_multistream(
            abi, module, x, reference_out, reference_final
        )
    check_public_multistream(x, reference_out, reference_final)
    check_invalid_metadata(module, x)
    check_two_devices(module, min(args.tokens, 512), args.heads)
    print("ALL FLASHKDA ABI CHECKS PASSED")


if __name__ == "__main__":
    main()
