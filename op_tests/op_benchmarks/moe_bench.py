import argparse
import csv
from dataclasses import dataclass
import datetime as _dt
import os
import platform
import socket
import sys
import uuid
from pathlib import Path
from textwrap import dedent

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import run_perftest


def require_rocm_amd() -> None:
    """Fail fast unless running on ROCm (AMD)."""

    hip_ver = getattr(torch.version, "hip", "")
    if not hip_ver:
        raise RuntimeError(
            "This benchmark is AMD/ROCm-only. Detected torch without ROCm (torch.version.hip is empty)."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is required for this benchmark (torch.cuda.is_available() is False).")


@dataclass(frozen=True)
class MoeCase:
    stage1_m: int
    stage1_n: int
    stage1_k: int
    topk: int
    expert: int
    stage2_m: int
    stage2_n: int
    stage2_k: int


STAGE1_SHAPES = [
    (16384, 6144, 3072, 4, 128),
    (256, 6144, 3072, 4, 128),
    (128, 6144, 3072, 4, 128),
    (64, 6144, 3072, 4, 128),
    (48, 6144, 3072, 4, 128),
    (32, 6144, 3072, 4, 128),
    (16, 6144, 3072, 4, 128),
    (8, 6144, 3072, 4, 128),
    (4, 6144, 3072, 4, 128),
    (2, 6144, 3072, 4, 128),
    (1, 6144, 3072, 4, 128),
    (8192, 6144, 3072, 4, 128),
]

STAGE2_SHAPES = [
    (65536, 3072, 3072, 4, 128),
    (1024, 3072, 3072, 4, 128),
    (512, 3072, 3072, 4, 128),
    (256, 3072, 3072, 4, 128),
    (192, 3072, 3072, 4, 128),
    (128, 3072, 3072, 4, 128),
    (64, 3072, 3072, 4, 128),
    (32, 3072, 3072, 4, 128),
    (16, 3072, 3072, 4, 128),
    (8, 3072, 3072, 4, 128),
    (4, 3072, 3072, 4, 128),
    (32768, 3072, 3072, 4, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark cktile MoE flatmm (2-stage) with a fixed shape sweep.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Activation dtype for hidden/a2/moe_out.",
    )
    parser.add_argument(
        "--rep",
        dest="rep",
        type=int,
        default=10,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        type=int,
        default=5,
        help="Number of warmup iterations.",
    )
    # Backward-compatible aliases (hidden)
    parser.add_argument(
        "--num-iters",
        dest="rep",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-warmup",
        dest="warmup",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=-1,
        help="Run one shape case by index (0-based). Default: -1 (run all).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default="",
        help="Optional CSV output path for benchmark results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "tflops"],
        default="tflops",
        help="Console metric to display (CSV always includes time and TFLOPS).",
    )
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="Include device/run metadata columns in CSV output.",
    )
    return parser.parse_args()


def pick_dtype(dtype_name: str):
    return dtypes.bf16 if dtype_name == "bf16" else dtypes.fp16


def tflops(m: int, n: int, k: int, us: float) -> float:
    if us <= 0:
        return 0.0
    flop = 2.0 * m * n * k
    return flop / (us * 1e-6) / 1e12


def _total_tflops(case: MoeCase, total_us: float) -> float:
    if total_us <= 0:
        return 0.0
    flop1 = 2.0 * case.stage1_m * case.stage1_n * case.stage1_k
    flop2 = 2.0 * case.stage2_m * case.stage2_n * case.stage2_k
    return (flop1 + flop2) / (total_us * 1e-6) / 1e12


def build_cases():
    if len(STAGE1_SHAPES) != len(STAGE2_SHAPES):
        raise ValueError("Stage1/Stage2 shape list lengths must match.")
    cases = []
    for s1, s2 in zip(STAGE1_SHAPES, STAGE2_SHAPES):
        s1_m, s1_n, s1_k, topk1, expert1 = s1
        s2_m, s2_n, s2_k, topk2, expert2 = s2
        if topk1 != topk2 or expert1 != expert2:
            raise ValueError(f"Mismatched topk/expert between {s1} and {s2}.")
        if s2_m != s1_m * topk1:
            raise ValueError(f"Expected stage2 M == stage1 M * topk, got {s1} and {s2}.")
        if s2_n != s1_k:
            raise ValueError(f"Expected stage2 N == stage1 K, got {s1} and {s2}.")
        cases.append(MoeCase(s1_m, s1_n, s1_k, topk1, expert1, s2_m, s2_n, s2_k))
    return cases


def run_case(case: MoeCase, dtype, num_iters: int, num_warmup: int, seed: int):
    # PyTorch uses the torch.cuda API and the "cuda" device string for ROCm as well.
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    token = case.stage1_m
    model_dim = case.stage1_k
    inter_dim = case.stage2_k

    hidden_states = torch.randn((token, model_dim), dtype=dtype, device=device)
    w1_fp = torch.randn((case.expert, case.stage1_n, model_dim), dtype=dtype, device=device)
    w2_fp = torch.randn((case.expert, model_dim, inter_dim), dtype=dtype, device=device)

    topk_ids = torch.randint(
        low=0,
        high=case.expert,
        size=(token, case.topk),
        dtype=dtypes.i32,
        device=device,
    )
    topk_weights = torch.rand((token, case.topk), dtype=dtypes.fp32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    block_m = 16 if token < 2048 else 32 if token < 16384 else 64

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=case.expert,
        model_dim=model_dim,
        moebuf_dtype=dtype,
        block_size=block_m,
    )

    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_qt, w1_scale = quant(w1_fp, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = quant(w2_fp, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(case.expert, case.stage1_n, model_dim // 2)
    w2_qt = w2_qt.view(case.expert, model_dim, inter_dim // 2)
    w1_qt = shuffle_weight_a16w4(w1_qt, 16, True)
    w2_qt = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale = shuffle_scale_a16w4(w1_scale, case.expert, True)
    w2_scale = shuffle_scale_a16w4(w2_scale, case.expert, False)

    bias1 = torch.zeros((case.expert, case.stage1_n), dtype=dtypes.fp32, device=device)
    bias2 = torch.zeros((case.expert, model_dim), dtype=dtypes.fp32, device=device)

    stage1_out = torch.empty((token, case.topk, inter_dim), dtype=dtype, device=device)
    stage2_in, stage1_us = run_perftest(
        cktile_moe_stage1,
        hidden_states,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        stage1_out,
        case.topk,
        block_m=block_m,
        a1_scale=None,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias1=bias1,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    if stage2_in is None:
        stage2_in = stage1_out

    moe_out = torch.empty((token, model_dim), dtype=dtype, device=device)
    _, stage2_us = run_perftest(
        cktile_moe_stage2,
        stage2_in,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        case.topk,
        w2_scale=w2_scale,
        a2_scale=None,
        block_m=block_m,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias2=bias2,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    return stage1_us, stage2_us, block_m


def _safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def collect_run_metadata() -> dict:
    """Collect lightweight, dashboard-friendly run metadata.

    Keep this robust across CUDA + ROCm, and avoid any heavy queries.
    """

    ts_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    meta = {
        "run_id": str(uuid.uuid4()),
        "ts_utc": ts_utc,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", ""),
        "aiter": getattr(aiter, "__version__", ""),
        "torch_hip": getattr(torch.version, "hip", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
    }

    if torch.cuda.is_available() and getattr(torch.version, "hip", ""):
        try:
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            meta.update(
                {
                    "device_type": "rocm",
                    "device_index": int(dev_idx),
                    "device_name": torch.cuda.get_device_name(dev_idx),
                    "device_total_mem_bytes": int(_safe_getattr(props, "total_memory", 0) or 0),
                    "device_multi_processor_count": int(
                        _safe_getattr(props, "multi_processor_count", 0) or 0
                    ),
                    "device_gcn_arch_name": str(_safe_getattr(props, "gcnArchName", "") or ""),
                }
            )
        except Exception:
            meta.update({"device_type": "rocm"})
    else:
        meta.update({"device_type": "none"})

    return meta


def _read_text_if_exists(path: str) -> str:
    try:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def collect_rocm_stack_versions() -> dict:
    """Best-effort ROCm stack version strings for the legend block.

    There isn't a single stable Python API for all of these, so we probe common sources.
    Missing values are left empty.
    """

    rocm_version = (
        getattr(torch.version, "hip", "")
        or os.environ.get("ROCM_VERSION", "")
        or _read_text_if_exists("/opt/rocm/.info/version")
        or _read_text_if_exists("/opt/rocm/.info/version-dev")
    )

    # These are often available via packages; keep best-effort and empty otherwise.
    rocblas_version = os.environ.get("ROCBLAS_VERSION", "")
    hipblaslt_version = os.environ.get("HIPBLASLT_VERSION", "")

    return {
        "torch_version": getattr(torch, "__version__", ""),
        "rocm_version": rocm_version,
        "rocblas_version": rocblas_version,
        "hipblaslt_version": hipblaslt_version,
    }


def _infer_aiter_meta_dir() -> Path | None:
    env_meta = os.environ.get("AITER_META_DIR", "").strip()
    if env_meta:
        return Path(env_meta)

    # In the packaged egg, aiter_meta often sits next to the aiter package dir:
    #   .../amd_aiter-...egg/aiter/__init__.py
    #   .../amd_aiter-...egg/aiter_meta/
    try:
        egg_root = Path(aiter.__file__).resolve().parent.parent
        cand = egg_root / "aiter_meta"
        if cand.exists():
            return cand
    except Exception:
        pass
    return None


def _has_moe_sorting_ck_headers() -> bool:
    """Return True if the CK tile moe_sorting headers are available.

    The common failure mode with pip/egg installs is that `aiter_meta/3rdparty/...`
    isn't present, causing moe_sorting JIT to fail with `moe_sorting_api.hpp` missing.
    """

    # Possible roots:
    # - AITER_META_DIR points to a repo checkout's `aiter_meta/`
    # - CK_3RDPARTY_DIR points to a CK checkout root
    meta_dir = _infer_aiter_meta_dir()
    ck_3rdparty = os.environ.get("CK_3RDPARTY_DIR", "").strip()
    ck_dir = Path(ck_3rdparty) if ck_3rdparty else None

    # Directory that failed in your log
    rel_ck_example = Path("example/ck_tile/13_moe_sorting")
    rel_meta_ck_example = Path("3rdparty/composable_kernel") / rel_ck_example

    candidates: list[Path] = []
    if meta_dir is not None:
        candidates.append(meta_dir / rel_meta_ck_example)
    if ck_dir is not None:
        candidates.append(ck_dir / rel_ck_example)

    for d in candidates:
        if d.exists():
            # header can be under this dir or an include/ subdir depending on layout
            try:
                if (d / "moe_sorting_api.hpp").exists():
                    return True
                for p in d.rglob("moe_sorting_api.hpp"):
                    if p.is_file():
                        return True
            except Exception:
                continue

    return False


def _dtype_nbytes(dtype_name: str) -> int:
    # bf16/fp16 activations: 2 bytes
    return 2


def estimate_moe_flops_and_bytes(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype_name: str,
    stage1_n: int,
) -> tuple[float, float]:
    """Return (flops, bytes) for reporting throughput/bandwidth/AI.

    FLOPs model:
    - If stage1_n == 2*intermediate_size, assume gated FFN (SwiGLU-style):
      flop/token/expert = 2*H*(2*I) + 2*I*H = 6*H*I
    - Otherwise assume non-gated: flop/token/expert = 2*H*stage1_n + 2*I*H

    Bytes model (best-effort, aligned with ck_tile fp4 weight storage):
    - Weight bytes assume fp4 packed as fp4x2 => 1 byte per 2 weights.
      * W1 has shape (E, stage1_n, H) weights, packed => E*stage1_n*H/2 bytes
      * W2 has shape (E, H, I) weights, packed => E*H*I/2 bytes
    - Activation bytes include hidden input/output and intermediates (bf16/fp16).
    """

    H = float(hidden_size)
    I = float(intermediate_size)
    M = float(num_tokens)
    T = float(top_k)
    E = float(num_experts)

    gated = (stage1_n == 2 * intermediate_size)
    if gated:
        flops = (6.0 * H * I) * (M * T)
    else:
        flops = (2.0 * H * float(stage1_n) + 2.0 * I * H) * (M * T)

    act_bytes = float(_dtype_nbytes(dtype_name))
    # activations: hidden in/out and intermediate tensors
    bytes_hidden_in = M * H * act_bytes
    bytes_hidden_out = M * H * act_bytes
    bytes_stage1_out = M * T * I * act_bytes
    bytes_stage2_in = bytes_stage1_out
    bytes_total_act = bytes_hidden_in + bytes_hidden_out + bytes_stage1_out + bytes_stage2_in

    # weights packed fp4x2 (uint8)
    bytes_w1 = E * float(stage1_n) * H / 2.0
    bytes_w2 = E * H * I / 2.0
    bytes_total = bytes_total_act + bytes_w1 + bytes_w2

    return flops, bytes_total


def main():
    args = parse_args()
    require_rocm_amd()
    dtype = pick_dtype(args.dtype)

    run_meta = collect_run_metadata() if args.with_meta else {}
    write_report = bool(args.output)
    report_to_stdout = write_report and args.output.strip() == "-"
    emit_human = not report_to_stdout

    cases = build_cases()
    if args.case_index >= len(cases):
        raise ValueError(f"--case-index {args.case_index} out of range [0, {len(cases)-1}].")
    if args.case_index >= 0:
        selected = [(args.case_index, cases[args.case_index])]
    else:
        selected = list(enumerate(cases))

    if emit_human:
        print("==== MoE flatmm benchmark (cktile 2-stage) ====")
        print(
            "config: "
            f"dtype={args.dtype} (quantized to mxfp4), rep={args.rep}, "
            f"warmup={args.warmup}, seed={args.seed}, cases={len(selected)}"
        )
        print(
            "idx | M tokens | K model_dim | N ffn_dim | E experts | topk | "
            "time_ms | tflops_total | stage1_tflops | stage2_tflops"
        )
    rows = []

    for idx, case in selected:
        stage1_us, stage2_us, block_m = run_case(
            case,
            dtype=dtype,
            num_iters=args.rep,
            num_warmup=args.warmup,
            seed=args.seed + idx,
        )
        stage1_tflops = tflops(case.stage1_m, case.stage1_n, case.stage1_k, stage1_us)
        stage2_tflops = tflops(case.stage2_m, case.stage2_n, case.stage2_k, stage2_us)
        total_us = stage1_us + stage2_us
        total_ms = total_us / 1000.0
        total_tflops = _total_tflops(case, total_us)

        hidden_size = int(case.stage1_k)
        intermediate_size = int(case.stage2_k)
        num_tokens = int(case.stage1_m)
        num_experts = int(case.expert)
        top_k = int(case.topk)

        flops, total_bytes = estimate_moe_flops_and_bytes(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            dtype_name=args.dtype,
            stage1_n=int(case.stage1_n),
        )
        secs = float(total_us) * 1e-6
        throughput_tfs = (flops / secs / 1e12) if secs > 0 else 0.0
        bandwidth_gbs = (total_bytes / secs / 1e9) if secs > 0 else 0.0
        arithmetic_intensity = (flops / total_bytes) if total_bytes > 0 else 0.0

        row = {
            **run_meta,
            "moe_impl": "ck_tile_moe",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "top_k": top_k,
            "elapsed_time_us": float(total_us),
            "throughput_tfs": float(throughput_tfs),
            "bandwidth_gbs": float(bandwidth_gbs),
            "arithmetic_intensity": float(arithmetic_intensity),
            "num_warmups": int(args.warmup),
            "num_iterations": int(args.rep),
            # Keep a few extra diagnostic fields
            "idx": idx,
            "dtype": args.dtype,
            "block_m": int(block_m),
            "stage1_us": float(stage1_us),
            "stage2_us": float(stage2_us),
            "stage1_tflops": float(stage1_tflops),
            "stage2_tflops": float(stage2_tflops),
            "seed": int(args.seed + idx),
        }
        rows.append(row)
        if emit_human:
            if args.metric == "time":
                metric_val = total_ms
                metric_str = f"{metric_val:>7.3f}"
            else:
                metric_val = total_tflops
                metric_str = f"{metric_val:>11.2f}"

            print(
                f"{idx:>3} | {case.stage1_m:>8} | {case.stage1_k:>11} | {case.stage1_n:>9} | "
                f"{case.expert:>9} | {case.topk:>4} | "
                f"{total_ms:>7.3f} | {total_tflops:>12.2f} | "
                f"{stage1_tflops:>12.2f} | {stage2_tflops:>12.2f}"
            )

    if write_report:
        out_is_stdout = args.output.strip() == "-"
        out_f = sys.stdout if out_is_stdout else open(args.output, "w", encoding="utf-8", newline="")
        try:
            v = collect_rocm_stack_versions()
            out_f.write(">>>>>>>>>> Legend Start <<<<<<<<<<\n")
            out_f.write(f"Torch Version: {v.get('torch_version','')}\n")
            out_f.write(f"rocm Version: {v.get('rocm_version','')}\n")
            out_f.write(f"rocBlas Version: {v.get('rocblas_version','')}\n")
            out_f.write(f"hipblasLT Version: {v.get('hipblaslt_version','')}\n")
            out_f.write(
                "Keys: moe_impl,hidden_size,num_tokens,num_experts,num_groups,expert_mask,"
                "activation_method,quant_method,inter_dim,topk_group\n"
            )
            out_f.write(">>>>>>>>>> Legend Ends  <<<<<<<<<<\n")

            header = [
                "moe_impl",
                "hidden_size",
                "intermediate_size",
                "num_tokens",
                "num_experts",
                "top_k",
                "elapsed_time(us)",
                "throughput(TF/s)",
                "bandwidth(GB/s)",
                "arithmetic_intensity",
                "num_warmups",
                "num_iterations",
            ]
            out_f.write(",".join(header) + "\n")

            for r in rows:
                line = [
                    r["moe_impl"],
                    str(r["hidden_size"]),
                    str(r["intermediate_size"]),
                    str(r["num_tokens"]),
                    str(r["num_experts"]),
                    str(r["top_k"]),
                    f"{r['elapsed_time_us']:.3f}",
                    f"{r['throughput_tfs']:.6f}",
                    f"{r['bandwidth_gbs']:.3f}",
                    f"{r['arithmetic_intensity']:.3f}",
                    str(r["num_warmups"]),
                    str(r["num_iterations"]),
                ]
                out_f.write(",".join(line) + "\n")

        finally:
            if not out_is_stdout:
                out_f.close()

        if emit_human and not out_is_stdout:
            print(f"wrote report: {args.output}")


if __name__ == "__main__":
    main()
