import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as fused_moe_mod
from aiter.fused_moe import fused_topk, torch_moe, torch_moe_stage1, torch_moe_stage2
from aiter.ops.quant import get_torch_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.test_common import run_perftest
from aiter.utility.fp4_utils import e8m0_shuffle


torch.set_default_device("cuda")


@dataclass
class QuantSpec:
    name: str
    q_type: QuantType
    q_dtype_a: torch.dtype
    q_dtype_w: torch.dtype
    q_type2: Optional[QuantType] = None
    q_dtype_a2: Optional[torch.dtype] = None
    q_dtype_w2: Optional[torch.dtype] = None

    @property
    def stage2_q_type(self) -> QuantType:
        return self.q_type if self.q_type2 is None else self.q_type2

    @property
    def stage2_q_dtype_a(self) -> torch.dtype:
        return self.q_dtype_a if self.q_dtype_a2 is None else self.q_dtype_a2

    @property
    def stage2_q_dtype_w(self) -> torch.dtype:
        return self.q_dtype_w if self.q_dtype_w2 is None else self.q_dtype_w2

    @property
    def is_hybrid(self) -> bool:
        return (
            self.q_type != self.stage2_q_type
            or self.q_dtype_a != self.stage2_q_dtype_a
            or self.q_dtype_w != self.stage2_q_dtype_w
        )


@dataclass
class CaseSpec:
    case_name: str
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    activation: ActivationType
    dtype: torch.dtype
    use_g1u1: bool
    doweight_stage1: bool
    csv_quant: Optional[QuantSpec] = None


QUANT_PRESETS: Dict[str, QuantSpec] = {
    "int8": QuantSpec(
        name="int8",
        q_type=QuantType.per_Token,
        q_dtype_a=dtypes.i8,
        q_dtype_w=dtypes.i8,
    ),
    "fp8": QuantSpec(
        name="fp8",
        q_type=QuantType.per_Token,
        q_dtype_a=dtypes.fp8,
        q_dtype_w=dtypes.fp8,
    ),
    "fp4": QuantSpec(
        name="fp4",
        q_type=QuantType.per_1x32,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=dtypes.fp4x2,
    ),
}


EVAL_GLOBALS = {
    "__builtins__": {},
    "torch": torch,
    "QuantType": QuantType,
    "ActivationType": ActivationType,
    "dtypes": dtypes,
    "aiter": aiter,
}


def safe_eval(value):
    if not isinstance(value, str):
        return value
    expr = value.strip()
    if not expr:
        return expr
    return eval(expr, EVAL_GLOBALS, {})


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    sv = str(v).strip().lower()
    if sv in {"1", "true", "t", "yes", "y"}:
        return True
    if sv in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported bool value: {v}")


def parse_activation(v) -> ActivationType:
    if isinstance(v, ActivationType):
        return v
    if v is None:
        return ActivationType.Gelu
    sv = str(v).strip()
    low = sv.lower()
    alias = {
        "gelu": ActivationType.Gelu,
        "silu": ActivationType.Silu,
        "swiglu": ActivationType.Swiglu,
    }
    if low in alias:
        return alias[low]
    obj = safe_eval(sv)
    if isinstance(obj, ActivationType):
        return obj
    raise ValueError(f"Unsupported activation: {v}")


def parse_dtype(v) -> torch.dtype:
    if isinstance(v, torch.dtype):
        return v
    if v is None:
        return torch.bfloat16
    sv = str(v).strip()
    alias = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if sv.lower() in alias:
        return alias[sv.lower()]
    obj = safe_eval(sv)
    if isinstance(obj, torch.dtype):
        return obj
    raise ValueError(f"Unsupported dtype: {v}")


def parse_quant_from_row(row) -> Optional[QuantSpec]:
    required = {"q_type", "q_dtype_a", "q_dtype_w"}
    if not required.issubset(set(row.index)):
        return None

    def _normalize_qtype(qt):
        if qt == QuantType.per_128x128:
            return QuantType.per_1x128
        return qt

    def _has_valid(col: str) -> bool:
        return col in row.index and pd.notna(row[col]) and str(row[col]).strip() != ""

    def _infer_name(q_type, q_dtype_a, q_dtype_w) -> str:
        if (
            q_type == QuantType.per_1x32
            and q_dtype_a == dtypes.fp4x2
            and q_dtype_w == dtypes.fp4x2
        ):
            return "fp4"
        if q_dtype_a == dtypes.i8 and q_dtype_w == dtypes.i8:
            return "int8"
        if q_dtype_a in {dtypes.fp8, torch.float8_e4m3fnuz, torch.float8_e4m3fn} and q_dtype_w in {
            dtypes.fp8,
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        }:
            return "fp8"
        return f"custom({q_type},{q_dtype_a},{q_dtype_w})"

    q_type = safe_eval(str(row["q_type"]))
    q_type = _normalize_qtype(q_type)
    q_dtype_a = safe_eval(str(row["q_dtype_a"]))
    q_dtype_w = safe_eval(str(row["q_dtype_w"]))

    if _has_valid("q_type2"):
        q_type2 = _normalize_qtype(safe_eval(str(row["q_type2"])))
    else:
        q_type2 = q_type
    q_dtype_a2 = safe_eval(str(row["q_dtype_a2"])) if _has_valid("q_dtype_a2") else q_dtype_a
    q_dtype_w2 = safe_eval(str(row["q_dtype_w2"])) if _has_valid("q_dtype_w2") else q_dtype_w

    s1_name = _infer_name(q_type, q_dtype_a, q_dtype_w)
    s2_name = _infer_name(q_type2, q_dtype_a2, q_dtype_w2)
    if s1_name == s2_name:
        name = s1_name
    else:
        name = f"{s1_name}->{s2_name}"
    return QuantSpec(
        name=name,
        q_type=q_type,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type2=q_type2,
        q_dtype_a2=q_dtype_a2,
        q_dtype_w2=q_dtype_w2,
    )


def load_cases_from_csv(csv_path: str, max_cases: int, quant_from_csv: bool) -> List[CaseSpec]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if max_cases > 0:
        df = df.head(max_cases)
    cases: List[CaseSpec] = []
    for idx, row in df.iterrows():
        cases.append(
            CaseSpec(
                case_name=f"csv#{idx}",
                token=int(row["token"]),
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
                expert=int(row["expert"]),
                topk=int(row["topk"]),
                activation=parse_activation(row.get("act_type", "ActivationType.Gelu")),
                dtype=parse_dtype(row.get("dtype", "torch.bfloat16")),
                use_g1u1=parse_bool(row.get("use_g1u1", 0)),
                doweight_stage1=parse_bool(row.get("doweight_stage1", 0)),
                csv_quant=parse_quant_from_row(row) if quant_from_csv else None,
            )
        )
    return cases


def build_cli_case(args) -> CaseSpec:
    return CaseSpec(
        case_name="cli_case",
        token=args.token,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        expert=args.expert,
        topk=args.topk,
        activation=parse_activation(args.activation),
        dtype=parse_dtype(args.dtype),
        use_g1u1=parse_bool(args.use_g1u1),
        doweight_stage1=parse_bool(args.doweight_stage1),
        csv_quant=None,
    )


def quantize_weight(weight: torch.Tensor, q_type: QuantType, q_dtype_w: torch.dtype):
    torch_quant = get_torch_quant(q_type)
    weight_qt, weight_scale = torch_quant(weight, quant_dtype=q_dtype_w)
    if q_dtype_w == dtypes.fp4x2:
        weight_qt = weight_qt.view(weight.shape[0], weight.shape[1], weight.shape[2] // 2)
    else:
        weight_qt = weight_qt.view(weight.shape)
    return weight_qt, weight_scale


def quantize_activation_stage1(hidden: torch.Tensor, quant: QuantSpec):
    if quant.q_type == QuantType.per_1x128:
        token, model_dim = hidden.shape
        a1_qt, a1_scale = aiter.pertoken_quant(
            hidden.view(token, -1, 128), quant_dtype=quant.q_dtype_a
        )
        return a1_qt.view(token, model_dim), a1_scale.squeeze(-1)
    torch_quant = get_torch_quant(quant.q_type)
    return torch_quant(hidden, quant_dtype=quant.q_dtype_a)


def generate_case_data(
    case: CaseSpec,
    quant_specs: List[QuantSpec],
    seed: int,
    device: str = "cuda",
):
    """Generate raw + quantized data for one MoE case.

    Similar to tuner generate_data_2stages flow:
      1) build raw tensors
      2) build routing(topk)
      3) materialize quant tensors/scales for each quant mode
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    hidden = torch.randn((case.token, case.model_dim), dtype=case.dtype, device=device) / 10
    w1_shape = (
        (case.expert, case.inter_dim * 2, case.model_dim)
        if case.use_g1u1
        else (case.expert, case.inter_dim, case.model_dim)
    )
    w1 = torch.randn(w1_shape, dtype=case.dtype, device=device) / 10
    w2 = torch.randn((case.expert, case.model_dim, case.inter_dim), dtype=case.dtype, device=device) / 10
    score = torch.randn((case.token, case.expert), dtype=case.dtype, device=device)

    topk_weight, topk_ids = fused_topk(hidden, score, case.topk, True)
    topk_weight = topk_weight.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    quant_data = {}
    for quant in quant_specs:
        w1_qt, w1_scale = quantize_weight(w1, quant.q_type, quant.q_dtype_w)
        w2_qt, w2_scale = quantize_weight(w2, quant.stage2_q_type, quant.stage2_q_dtype_w)
        a1_qt, a1_scale = quantize_activation_stage1(hidden, quant)
        quant_data[quant.name] = {
            "quant": quant,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
            "a1_qt": a1_qt,
            "a1_scale": a1_scale,
        }

    return {
        "seed": seed,
        "raw": {
            "hidden": hidden,
            "w1": w1,
            "w2": w2,
            "score": score,
            "topk_weight": topk_weight,
            "topk_ids": topk_ids,
        },
        "quant_data": quant_data,
    }


def prepare_fused_weights(
    case: CaseSpec,
    quant: QuantSpec,
    w1_qt,
    w2_qt,
    w1_scale,
    w2_scale,
):
    def _prepare_stage_weight(
        weight_qt,
        weight_scale,
        weight_dtype: torch.dtype,
        is_stage2: bool,
    ):
        if weight_dtype == dtypes.fp4x2:
            if is_stage2:
                return (
                    shuffle_weight_a16w4(weight_qt, 16, False),
                    shuffle_scale_a16w4(weight_scale, case.expert, False)
                    if weight_scale is not None
                    else None,
                )
            return (
                shuffle_weight(weight_qt, (16, 16)),
                e8m0_shuffle(weight_scale) if weight_scale is not None else None,
            )
        return shuffle_weight(weight_qt, (16, 16)), weight_scale

    w1_qt_fused, w1_scale_fused = _prepare_stage_weight(
        w1_qt, w1_scale, quant.q_dtype_w, is_stage2=False
    )
    w2_qt_fused, w2_scale_fused = _prepare_stage_weight(
        w2_qt, w2_scale, quant.stage2_q_dtype_w, is_stage2=True
    )

    return w1_qt_fused, w2_qt_fused, w1_scale_fused, w2_scale_fused


def torch_moe_reference(
    case: CaseSpec,
    quant: QuantSpec,
    hidden: torch.Tensor,
    w1_qt: torch.Tensor,
    w2_qt: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    a1_qt: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
):
    # Prefer framework torch_moe reference for int8/fp8 paths.
    # fp4/per_1x32 keeps stage1/2 path because torch_moe does not directly
    # consume fp4x2 packed weights + e8m0 scales.
    if (
        not quant.is_hybrid
        and quant.q_type in {QuantType.per_Token, QuantType.per_Tensor}
        and quant.q_dtype_w != dtypes.fp4x2
    ):
        return torch_moe(
            hidden,
            w1_qt,
            w2_qt,
            topk_weight,
            topk_ids,
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            activation=case.activation,
        )

    if a1_qt is None or a1_scale is None:
        a1_qt, a1_scale = quantize_activation_stage1(hidden, quant)
    ref_stage1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weight,
        topk_ids,
        dtype=case.dtype,
        activation=case.activation,
        quant_type=quant.q_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=case.doweight_stage1,
    )

    a2_quant_dtype = quant.stage2_q_dtype_a
    if quant.stage2_q_type == QuantType.per_1x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            ref_stage1.view(ref_stage1.shape[0], -1, 128), quant_dtype=a2_quant_dtype
        )
    else:
        torch_quant = get_torch_quant(quant.stage2_q_type)
        a2_qt, a2_scale = torch_quant(ref_stage1, quant_dtype=a2_quant_dtype)
    a2_qt = a2_qt.view(ref_stage1.shape[0], ref_stage1.shape[1], -1)

    expected_n = case.inter_dim * 2 if case.use_g1u1 else case.inter_dim
    # torch_moe_stage2 uses w1 only for shape inference. For fp4-packed stage2 w2,
    # keep w1's K dim packed (model_dim//2); otherwise use unpacked model_dim.
    expected_k = (
        case.model_dim // 2 if quant.stage2_q_dtype_w == dtypes.fp4x2 else case.model_dim
    )
    if w1_qt.shape[1] == expected_n and w1_qt.shape[2] == expected_k:
        w1_stage2_ref = w1_qt
    else:
        w1_stage2_ref = torch.empty(
            (case.expert, expected_n, expected_k),
            dtype=w2_qt.dtype,
            device=w2_qt.device,
        )

    return torch_moe_stage2(
        a2_qt,
        w1_stage2_ref,
        w2_qt,
        topk_weight,
        topk_ids,
        dtype=case.dtype,
        quant_type=quant.stage2_q_type,
        a2_scale=a2_scale,
        w2_scale=w2_scale,
        doweight=not case.doweight_stage1,
    )


def run_fused_moe(
    case: CaseSpec,
    quant: QuantSpec,
    hidden: torch.Tensor,
    w1_qt_fused: torch.Tensor,
    w2_qt_fused: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale_fused: Optional[torch.Tensor],
    w2_scale_fused: Optional[torch.Tensor],
):
    return fused_moe_mod.fused_moe(
        hidden,
        w1_qt_fused,
        w2_qt_fused,
        topk_weight,
        topk_ids,
        activation=case.activation,
        quant_type=quant.q_type,
        doweight_stage1=case.doweight_stage1,
        w1_scale=w1_scale_fused,
        w2_scale=w2_scale_fused,
        q_type2=quant.stage2_q_type,
        q_dtype_a2=quant.stage2_q_dtype_a,
        q_dtype_w2=quant.stage2_q_dtype_w,
        dtype=case.dtype,
    )


def check_result(ref_out, test_out, atol=1.0, rtol=0.05, pass_pct=95.0):
    delta = (ref_out.float() - test_out.float()).abs()
    max_delta = float(delta.max().item())
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = float(close_mask.float().mean().item() * 100)
    cos = float(
        torch.nn.functional.cosine_similarity(
            ref_out.reshape(1, -1).float(), test_out.reshape(1, -1).float()
        ).item()
    )
    return {
        "pass": pct_close >= pass_pct,
        "max_delta": max_delta,
        "pct_close": pct_close,
        "cos": cos,
    }


def _build_quant_pair(stage1_quant: str, stage2_quant: str) -> QuantSpec:
    q1 = QUANT_PRESETS[stage1_quant]
    q2 = QUANT_PRESETS[stage2_quant]
    name = stage1_quant if stage1_quant == stage2_quant else f"{stage1_quant}->{stage2_quant}"
    return QuantSpec(
        name=name,
        q_type=q1.q_type,
        q_dtype_a=q1.q_dtype_a,
        q_dtype_w=q1.q_dtype_w,
        q_type2=q2.q_type,
        q_dtype_a2=q2.q_dtype_a,
        q_dtype_w2=q2.q_dtype_w,
    )


def expand_quant_list(case: CaseSpec, selected_quant: List[str], stage2_quant: str) -> List[QuantSpec]:
    if case.csv_quant is not None:
        csv_stage1_name = case.csv_quant.name.split("->", 1)[0]
        if selected_quant and case.csv_quant.name in selected_quant:
            return [case.csv_quant]
        if selected_quant and csv_stage1_name not in selected_quant:
            return []
        return [case.csv_quant]
    resolved_stage2 = stage2_quant if stage2_quant != "same" else None
    return [
        _build_quant_pair(stage1_q, stage1_q if resolved_stage2 is None else resolved_stage2)
        for stage1_q in selected_quant
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified qMOE test: int8/fp8/fp4, functional + performance."
    )
    parser.add_argument("--csv", type=str, default="", help="CSV test case file.")
    parser.add_argument(
        "--quant-from-csv",
        action="store_true",
        help="Use q_type/q_dtype_a/q_dtype_w from CSV when available.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--quant",
        type=str,
        nargs="+",
        choices=["int8", "fp8", "fp4"],
        default=["int8", "fp8", "fp4"],
        help="Quant modes when CSV does not provide quant fields.",
    )
    parser.add_argument(
        "--quant2",
        type=str,
        choices=["same", "int8", "fp8", "fp4"],
        default="same",
        help="Stage2 quant mode when CSV does not provide q_type2/q_dtype_*2.",
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=["functional", "perf", "both"],
        default="both",
        help="Run correctness test, perf test, or both.",
    )
    parser.add_argument("--token", type=int, default=20480)
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=1536)
    parser.add_argument("--expert", type=int, default=400)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--use-g1u1", type=int, default=0)
    parser.add_argument("--doweight-stage1", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--pass-pct", type=float, default=95.0)
    parser.add_argument(
        "--fmoe-config",
        type=str,
        default="",
        help="Optional config csv path for fused_moe lookup; if empty and --csv has hybrid columns, use --csv.",
    )
    return parser.parse_args()


def _configure_fmoe_config(args):
    cfg_path = args.fmoe_config
    if not cfg_path and args.csv:
        df_head = pd.read_csv(args.csv, nrows=1)
        hybrid_cols = {"q_type2", "q_dtype_a2", "q_dtype_w2"}
        if hybrid_cols.issubset(set(df_head.columns)):
            cfg_path = args.csv

    if not cfg_path:
        return
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"--fmoe-config path not found: {cfg_path}")

    os.environ["AITER_CONFIG_FMOE"] = cfg_path
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0"

    from aiter.jit.core import AITER_CONFIGS

    AITER_CONFIGS.get_config_file.cache_clear()
    if hasattr(fused_moe_mod, "get_2stage_cfgs"):
        fused_moe_mod.get_2stage_cfgs.cache_clear()
    if hasattr(fused_moe_mod, "cfg_2stages"):
        fused_moe_mod.cfg_2stages = None
    print(f"[CFG] AITER_CONFIG_FMOE={cfg_path}")


def main():
    args = parse_args()
    _configure_fmoe_config(args)
    if args.csv:
        cases = load_cases_from_csv(args.csv, args.max_cases, args.quant_from_csv)
        if not cases:
            raise RuntimeError(f"No valid cases loaded from {args.csv}")
    else:
        cases = [build_cli_case(args)]

    results = []
    global_idx = 0
    for case in cases:
        quant_list = expand_quant_list(case, args.quant, args.quant2)
        if not quant_list:
            print(f"[SKIP] {case.case_name}: quant filter excludes this case.")
            continue
        seed = args.seed + global_idx
        global_idx += 1
        case_data = generate_case_data(case, quant_list, seed, device="cuda")
        hidden = case_data["raw"]["hidden"]
        topk_weight = case_data["raw"]["topk_weight"]
        topk_ids = case_data["raw"]["topk_ids"]

        for quant in quant_list:
            case_tag = (
                f"{case.case_name} | {quant.name} | "
                f"shape=({case.token},{case.model_dim},{case.inter_dim},E={case.expert},topk={case.topk})"
            )
            print(f"\n{'=' * 90}\n[RUN] {case_tag}\n{'=' * 90}")
            try:
                q_data = case_data["quant_data"][quant.name]
                w1_qt = q_data["w1_qt"]
                w2_qt = q_data["w2_qt"]
                w1_scale = q_data["w1_scale"]
                w2_scale = q_data["w2_scale"]
                a1_qt = q_data["a1_qt"]
                a1_scale = q_data["a1_scale"]

                if args.run in {"functional", "both"}:
                    ref_out = torch_moe_reference(
                        case,
                        quant,
                        hidden,
                        w1_qt,
                        w2_qt,
                        topk_weight,
                        topk_ids,
                        w1_scale,
                        w2_scale,
                        a1_qt=a1_qt,
                        a1_scale=a1_scale,
                    )

                selected_layout = (
                    "ck_preshuffle"
                    if (quant.q_dtype_w == dtypes.fp4x2 or quant.stage2_q_dtype_w == dtypes.fp4x2)
                    else "default"
                )
                w1_fused, w2_fused, w1_scale_fused, w2_scale_fused = prepare_fused_weights(
                    case,
                    quant,
                    w1_qt,
                    w2_qt,
                    w1_scale,
                    w2_scale,
                )
                fused_out = run_fused_moe(
                    case,
                    quant,
                    hidden,
                    w1_fused,
                    w2_fused,
                    topk_weight,
                    topk_ids,
                    w1_scale_fused,
                    w2_scale_fused,
                )
                perf_us = None
                selected_weights = (w1_fused, w2_fused, w1_scale_fused, w2_scale_fused)

                if args.run in {"perf", "both"}:
                    w1_fused, w2_fused, w1_scale_fused, w2_scale_fused = selected_weights
                    fused_out, perf_us = run_perftest(
                        run_fused_moe,
                        case,
                        quant,
                        hidden,
                        w1_fused,
                        w2_fused,
                        topk_weight,
                        topk_ids,
                        w1_scale_fused,
                        w2_scale_fused,
                        num_warmup=args.warmup,
                        num_iters=args.iters,
                    )
                    print(f"[PERF] e2e fused_moe: {perf_us:.3f} us")

                if args.run in {"functional", "both"}:
                    stat = check_result(
                        ref_out,
                        fused_out,
                        atol=args.atol,
                        rtol=args.rtol,
                        pass_pct=args.pass_pct,
                    )
                    print(
                        f"[FUNC] pass={stat['pass']} "
                        f"max_delta={stat['max_delta']:.4f} "
                        f"close={stat['pct_close']:.2f}% cos={stat['cos']:.6f}"
                    )
                    results.append(
                        {
                            "case": case_tag,
                            "status": "PASS" if stat["pass"] else "FAIL",
                            "max_delta": stat["max_delta"],
                            "close_pct": stat["pct_close"],
                            "cos": stat["cos"],
                            "perf_us": perf_us,
                        }
                    )
                else:
                    results.append(
                        {
                            "case": case_tag,
                            "status": "PERF_ONLY",
                            "max_delta": None,
                            "close_pct": None,
                            "cos": None,
                            "perf_us": perf_us,
                        }
                    )
            except Exception as ex:
                print(f"[ERROR] {case_tag}: {ex}")
                import traceback

                traceback.print_exc()
                results.append(
                    {
                        "case": case_tag,
                        "status": "ERROR",
                        "max_delta": None,
                        "close_pct": None,
                        "cos": None,
                        "perf_us": None,
                    }
                )

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for item in results:
        perf = "N/A" if item["perf_us"] is None else f"{item['perf_us']:.3f} us"
        cos = "N/A" if item["cos"] is None else f"{item['cos']:.6f}"
        if item["status"] in {"PASS", "FAIL"}:
            print(
                f"{item['status']:>5} | {perf:>12} | close={item['close_pct']:.2f}% "
                f"| max_delta={item['max_delta']:.4f} | cos={cos} | {item['case']}"
            )
        else:
            print(f"{item['status']:>9} | {perf:>12} | cos={cos} | {item['case']}")

    has_bad = any(item["status"] in {"FAIL", "ERROR"} for item in results)
    if has_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
