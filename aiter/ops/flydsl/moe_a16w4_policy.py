"""Validated paired tiles for the explicit clamped A16W4 MoE path."""

import csv
from pathlib import Path
from typing import NamedTuple


class A16W4Tiles(NamedTuple):
    block_m: int
    stage1_n: int
    stage1_k: int
    stage2_n: int
    stage2_k: int
    reuse_lds: bool = False


def _load_tuned_tiles(path: Path) -> dict[tuple, A16W4Tiles]:
    entries = {}
    try:
        source = path.open(newline="", encoding="utf-8")
    except FileNotFoundError:
        return entries
    with source:
        for row in csv.DictReader(source):
            if (
                row["act"] != "silu_clamped"
                or row["activation_bf16"] != "1"
                or row["w_dtype"] != "fp4"
                or row["w_layout"] != "standard"
                or row["reuse_lds"] not in ("0", "1")
            ):
                raise ValueError("Unsupported A16W4 CSV semantics")
            key = (
                row["gfx"],
                *(
                    int(row[field])
                    for field in (
                        "cu_num",
                        "token",
                        "hidden_dim",
                        "inter_dim",
                        "logical_inter",
                        "experts",
                        "topk",
                    )
                ),
                float(row["swiglu_limit"]),
            )
            if key in entries:
                raise ValueError(f"Duplicate A16W4 CSV key: {key}")
            entries[key] = A16W4Tiles(
                *(
                    int(row[field])
                    for field in (
                        "block_m",
                        "stage1_n",
                        "stage1_k",
                        "stage2_n",
                        "stage2_k",
                    )
                ),
                row["reuse_lds"] == "1",
            )
    return entries


_TUNED_TILES = _load_tuned_tiles(
    Path(__file__).resolve().parents[2] / "configs" / "tuned_flydsl_moe_a16w4.csv"
)


def get_a16w4_tuned_tiles(
    token_num: int,
    model_dim: int,
    inter_dim: int,
    logical_inter_dim: int,
    experts: int,
    topk: int,
    *,
    act: str,
    activation_bf16: bool,
    swiglu_limit: float,
    w_dtype: str = "fp4",
    w_layout: str = "standard",
) -> A16W4Tiles | None:
    """Return a validated tile pair, or None to preserve the caller's fallback.

    Entries assume separated weights, no bias, BF16 inputs/intermediates/output,
    stage1 b_nt=2/xcd=0 and stage2 b_nt=0/xcd=1, without a waves_per_eu cap.
    Lookup uses host metadata only; it does not inspect routing tensors or time
    candidates during inference.
    """
    if (
        act != "silu_clamped"
        or not activation_bf16
        or w_dtype != "fp4"
        or w_layout != "standard"
    ):
        return None
    if not float(swiglu_limit) > 0.0:
        raise ValueError("silu_clamped requires swiglu_limit > 0")

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    key = (
        get_gfx_runtime(),
        get_cu_num(),
        token_num,
        model_dim,
        inter_dim,
        logical_inter_dim,
        experts,
        topk,
        float(swiglu_limit),
    )
    return _TUNED_TILES.get(key)
