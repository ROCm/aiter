# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

_DTYPE_MAP = {
    "torch.float8_e4m3fn": "f8",
    "torch.float8_e4m3fnuz": "f8",
    "torch.bfloat16": "b16",
    "torch.float16": "f16",
    "torch.int8": "i8",
    "torch.float4_e2m1fn_x2": "fp4x2",
}

_QUANT_ALIAS = {"per_128x128": "per_1x128"}


def _build_moe_variant(
    aiter_csrc_dir: str,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    activation: str,
    quant_type: str,
    mul_routed_weight_stage: int,
    preshuffle_mode: bool,
    is_splitk: bool,
) -> Tuple[str, list]:
    parts = [
        "module_moe_ck2stages",
        a_dtype,
        b_dtype,
        "preshuffle_on" if preshuffle_mode else "preshuffle_off",
        c_dtype,
        activation,
        quant_type,
        f"mulWeightStage{mul_routed_weight_stage}",
    ]
    if is_splitk:
        parts.append("splitk")

    flags = ""
    if preshuffle_mode and b_dtype == "fp4x2":
        flags += " --preshuffle"
    if is_splitk:
        flags += " --issplitk"

    md_name = "_".join(parts)
    blob_gen_cmd = [
        (
            f"{aiter_csrc_dir}/ck_gemm_moe_2stages_codegen/gen_instances.py "
            f"-a {a_dtype} -b {b_dtype} -c {c_dtype} -q {quant_type} "
            f"-act {activation} -m {mul_routed_weight_stage}{flags} -w {{}}"
        )
    ]
    return md_name, blob_gen_cmd


def _normalize_dtype(dtype: str) -> str:
    return _DTYPE_MAP.get(dtype, dtype)


def _normalize_enum_str(s: str) -> str:
    """'QuantType.per_1x128' -> 'per_1x128', 'ActivationType.Silu' -> 'silu'.

    Same pattern as moe_op.py: ``str(enum).split(".")[-1].lower()``.
    """
    return s.split(".")[-1].lower() if "." in s else s.lower()


def _normalize_quant_type(quant_type: str) -> str:
    q = _normalize_enum_str(quant_type)
    return _QUANT_ALIAS.get(q, q)


def _normalize_activation(activation: str) -> str:
    return _normalize_enum_str(activation)


def _infer_preshuffle_modes(b_dtype: str, quant_type: str) -> List[bool]:
    """Infer preshuffle modes based on runtime behavior.

    - fp4x2: may or may not be pre-shuffled -> both off and on
    - no-quant (b16/f16): always pre-shuffled at runtime -> preshuffle_on only
    - other quantized types: always shuffled -> preshuffle_on only
    """
    if b_dtype == "fp4x2":
        return [False, True]
    return [True]


def _should_include_splitk(row: Dict, quant_type: str) -> bool:
    """splitk only applies to f8/f8 per_1x128 (blockscale) kernels."""
    if quant_type != "per_1x128":
        return False
    ksplit = row.get("ksplit")
    if not ksplit:
        return False
    try:
        return int(float(ksplit)) > 1
    except (TypeError, ValueError):
        return False


def _get_mul_weight_stage(row: Dict) -> int:
    value = row.get("doweight_stage1")
    if not value:
        return 2
    v = str(value).strip().lower()
    if v in ("1", "true"):
        return 1
    return 2


def _get_tuned_fmoe_rows() -> List[Dict]:
    configs_dir = Path(__file__).resolve().parents[2] / "configs"
    model_configs_dir = configs_dir / "model_configs"
    tuned_paths = [configs_dir / "tuned_fmoe.csv"]
    tuned_paths.extend(sorted(model_configs_dir.glob("*tuned_fmoe*.csv")))

    rows: List[Dict] = []
    for path in tuned_paths:
        if not path.exists():
            continue
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def get_moe_ck2stages_prebuild_variants(aiter_csrc_dir: str) -> List[Dict]:
    seen: Set[Tuple] = set()
    results: List[Dict] = []
    for row in _get_tuned_fmoe_rows():
        c_dtype = _normalize_dtype(row["dtype"])
        a_dtype = _normalize_dtype(row.get("q_dtype_a") or row["dtype"])
        b_dtype = _normalize_dtype(row.get("q_dtype_w") or row["dtype"])
        quant_type = _normalize_quant_type(row["q_type"])
        activation = _normalize_activation(row["act_type"])
        mul_weight_stage = _get_mul_weight_stage(row)
        need_splitk = _should_include_splitk(row, quant_type)

        for preshuffle in _infer_preshuffle_modes(b_dtype, quant_type):
            for splitk in [False, True] if need_splitk else [False]:
                key = (
                    a_dtype,
                    b_dtype,
                    c_dtype,
                    quant_type,
                    activation,
                    mul_weight_stage,
                    preshuffle,
                    splitk,
                )
                if key in seen:
                    continue
                seen.add(key)
                md_name, blob_gen_cmd = _build_moe_variant(
                    aiter_csrc_dir=aiter_csrc_dir,
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    c_dtype=c_dtype,
                    activation=activation,
                    quant_type=quant_type,
                    mul_routed_weight_stage=mul_weight_stage,
                    preshuffle_mode=preshuffle,
                    is_splitk=splitk,
                )
                results.append({"md_name": md_name, "blob_gen_cmd": blob_gen_cmd})
    return results
