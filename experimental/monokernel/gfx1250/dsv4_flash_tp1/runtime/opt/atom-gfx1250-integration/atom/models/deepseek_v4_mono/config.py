# SPDX-License-Identifier: MIT
"""Measured deployment and row contracts; no GPU or DSL imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    hidden: int
    tp: int
    heads: int
    q_rank: int
    experts: int
    max_rows: int


FLASH = Profile("flash_tp1", 4096, 1, 64, 1024, 256, 16)
PRO = Profile("pro_tp4", 7168, 4, 128, 1536, 384, 64)


def select_profile(hf, tp):
    for profile in (FLASH, PRO):
        if (
            getattr(hf, "hidden_size", None),
            tp,
            getattr(hf, "num_attention_heads", None),
            getattr(hf, "q_lora_rank", None),
            getattr(hf, "n_routed_experts", None),
        ) == (
            profile.hidden,
            profile.tp,
            profile.heads,
            profile.q_rank,
            profile.experts,
        ):
            return profile
    return None


def geometry_refusal(hf, tp):
    if select_profile(hf, tp) is None:
        return "only measured Flash TP1 and Pro TP4 geometries"
    expected = {
        "hc_mult": 4,
        "head_dim": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "hc_sinkhorn_iters": 20,
    }
    for field, value in expected.items():
        if getattr(hf, field, None) != value:
            return f"{field} must be {value}"
    return None


def step_supported(rows, profile, *, prefill, embeds=False, ubatches=False):
    return not (prefill or embeds or ubatches) and 0 < rows <= profile.max_rows


def config_refusal(config, *, arch):
    import os

    from atom.mono.runtime.deployment import common_refusals, first_refusal
    from atom.utils import envs

    if not envs.ATOM_MONO_ENABLE or not envs.ATOM_DSV4_MONO:
        return "switched off"
    if arch.split(":", 1)[0] != "gfx1250":
        return "not gfx1250"
    hf = config.hf_config
    why = geometry_refusal(hf, config.tensor_parallel_size)
    if why:
        return why
    # Sharding is fully specified by the measured profile. The shared deployment
    # validator still owns EP/DP/PP/CP/TBO/plugin/weight-update/compile refusals.
    checks = [
        (os.getenv("V4_FORCE_UE8M0_QUANT", "0") != "1", "forced UE8M0 round-trip")
    ]
    checks += common_refusals(config, lambda tp: None, (1, 4)) + [
        (not config.enable_dp_attention, "DP attention"),
        (config.speculative_config is None, "speculative decoding"),
        (config.kv_cache_dtype == "bf16", "KV cache must be bf16"),
        (config.index_cache_dtype == "fp8", "index cache must be fp8"),
        (envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE, "FP8 weight preshuffle off"),
    ]
    return first_refusal(checks)
