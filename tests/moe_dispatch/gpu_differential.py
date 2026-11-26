# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU differential of one migration path against the legacy MoE code.

Each case runs on the same inputs with AITER_MOEV2=off and AITER_MOEV2=<path>: the path
must run the call (not decline it), launch the same kernels on equal inputs (exact, but
floating-point tensors within 1 ulp or legacy's own spread), give the same output up to
the GEMM2 atomics, and log the same dispatch lines on a first call (see
`differential.run_differential`). Besides plain calls, the cases cover the inputs
`claims` accepts (`_VARIANTS`); where legacy raises on one (`_RAISES`), v2 must run the
call and raise the same error.

    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt
    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt --synth-table

Shipped tables have per-tensor rows for gfx942 only. `--synth-table` copies their CK rows
to the running arch in a temporary AITER_CONFIG_FMOE, so the tuned CK branch runs on a
gfx950 too. ASM and FlyDSL rows are not copied: their gfx942 configs give garbage on
gfx950 (legacy and v2 alike), so those branches need a gfx942 for this check.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# (token, model_dim, inter_dim, experts, topk, activation, doweight_stage1, out dtype,
#  ep [, variant]): a variant names one of `_VARIANTS` (or "g1u0"), an input `claims`
# accepts beyond the plain call.
_A8W8_PT_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 64, 256, 1024, 4096)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 32, 512)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16, 512)),
    *((m, 4096, 512, 64, 8, "Silu", False, "bf16", ep, v)
      for m, ep in ((16, False), (512, True))
      for v in ("num_local_tokens", "num_local_tokens_fp8_input")),
    *((m, 4096, 512, 64, 8, "Silu", False, "bf16", False, v)
      for v in ("a1_scale", "a1_a2_scales", "fp8_input", "output", "noncontiguous",
                "block_m_32", "block_m_128", "g1u0", "swiglu_limit_0", "swiglu_limit_inf")
      for m in (16, 512)),
    # Mixtral-8x7B (TP1) as vLLM calls it: swiglu_limit=0.0 on every call.
    *((m, 4096, 14336, 8, 2, "Silu", False, "bf16", False, "swiglu_limit_0") for m in (4, 256)),
    # vLLM passes swiglu_limit=0.0 with Gelu too; legacy leaves it unnormalized there.
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "swiglu_limit_0") for m in (32, 512)),
)  # fmt: skip

_A8W8_PT_GFX942_CU80 = (
    # A shipped gfx942-cu80 row selects the FlyDSL whole-graph implementation.
    (1, 4096, 192, 192, 8, "Silu", False, "bf16", False),
)  # fmt: skip


def _make_a8w8_pt_call(
    token, model_dim, inter_dim, E, topk, act, doweight, out, ep, variant=None
):
    """Inputs as op_tests/test_moe_2stage.py builds them for `-q 1` (per-tensor fp8),
    except that the weights are scaled by 1/sqrt(K); then `variant` applied."""
    import torch

    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(token * 7919 + model_dim + inter_dim + E + topk)
    dtype = {"bf16": dtypes.bf16, "fp16": dtypes.fp16}[out]
    device = "cuda"
    hidden = torch.randn((token, model_dim), dtype=dtype, device=device)
    # Scaled so each GEMM keeps values near 1: unscaled, the fp16 outputs reach the
    # fp16 limit, and whether a sum overflows then depends on the order of the GEMM2
    # atomics.
    gate_up = 1 if variant == "g1u0" else 2
    w1 = torch.randn((E, inter_dim * gate_up, model_dim), dtype=dtype, device=device)
    w1 /= model_dim**0.5
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device=device)
    w2 /= inter_dim**0.5
    w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=dtypes.fp8)
    w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=dtypes.fp8)
    w1_qt = shuffle_weight(w1_qt.view(w1.shape), layout=(16, 16))
    w2_qt = shuffle_weight(w2_qt.view(w2.shape), layout=(16, 16))
    global_E = 2 * E if ep else E
    score = torch.randn((token, global_E), dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(hidden, score, topk, True)
    expert_mask = None
    if ep:
        expert_mask = torch.zeros(global_E, dtype=dtypes.i32, device=device)
        expert_mask[:E] = 1
    call = {
        "hidden_states": hidden,
        "w1": w1_qt,
        "w2": w2_qt,
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        "expert_mask": expert_mask,
        "activation": ActivationType.__members__[act],
        "quant_type": QuantType.per_Tensor,
        "doweight_stage1": doweight,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }
    if variant not in (None, "g1u0"):
        _VARIANTS[variant](call)
    return call


def _a1_scale(call):
    import aiter
    from aiter import QuantType, dtypes

    quant = aiter.get_torch_quant(QuantType.per_Tensor)
    return quant(call["hidden_states"], quant_dtype=dtypes.fp8)


def _set_a1_scale(call):
    call["a1_scale"] = _a1_scale(call)[1]


def _set_a1_a2_scales(call):
    import torch

    _set_a1_scale(call)
    # The GEMM1 outputs stay near 1 (the weights are scaled by 1/sqrt(K)); a fixed
    # scale that keeps them inside the fp8 range.
    call["a2_scale"] = torch.full((1,), 8 / 224, dtype=torch.float32, device="cuda")


def _set_fp8_input(call):
    hidden = call["hidden_states"]
    call["dtype"] = hidden.dtype
    call["hidden_states"], call["a1_scale"] = _a1_scale(call)


def _set_num_local_tokens(call):
    import torch

    token = call["hidden_states"].shape[0]
    call["num_local_tokens"] = torch.tensor(
        [token - token // 4], dtype=torch.int32, device="cuda"
    )


def _set_output(call):
    import torch

    call["output"] = torch.empty_like(call["hidden_states"])


def _set_noncontiguous(call):
    import torch

    hidden = call["hidden_states"]
    wide = torch.zeros(
        (hidden.shape[0], 2 * hidden.shape[1]), dtype=hidden.dtype, device="cuda"
    )
    wide[:, : hidden.shape[1]] = hidden
    call["hidden_states"] = wide[:, : hidden.shape[1]]


def _set_num_local_tokens_fp8_input(call):
    _set_fp8_input(call)
    _set_num_local_tokens(call)


_VARIANTS = {
    "a1_scale": _set_a1_scale,
    "a1_a2_scales": _set_a1_a2_scales,
    "fp8_input": _set_fp8_input,
    "num_local_tokens": _set_num_local_tokens,
    "num_local_tokens_fp8_input": _set_num_local_tokens_fp8_input,
    "output": _set_output,
    "noncontiguous": _set_noncontiguous,
    "block_m_32": lambda call: call.update(block_size_M=32),
    "block_m_128": lambda call: call.update(block_size_M=128),
    "swiglu_limit_0": lambda call: call.update(swiglu_limit=0.0),
    "swiglu_limit_inf": lambda call: call.update(swiglu_limit=float("inf")),
}
# Variants legacy raises on: v2 must run them and raise the same error. The 2-stage
# path quantizes per tensor before GEMM2 (and before GEMM1 from bf16) with
# `per_tensor_quant_hip`, which takes no row count.
_RAISES = {
    "num_local_tokens": "num_rows is not supported for per_tensor_quant_hip",
    "num_local_tokens_fp8_input": "num_rows is not supported for per_tensor_quant_hip",
}


def _synth_rows(table_file, gfx, cu_num, limit):
    """gfx942 per-tensor fp8 rows rewritten for (gfx, cu_num); the case tuples."""
    import pandas as pd

    from aiter.moev2.dispatch import kernel_names

    df = pd.read_csv(table_file)
    src = df[
        (df["gfx"] == "gfx942")
        & (df["cu_num"] == 80)
        & (df["q_type"] == "QuantType.per_Tensor")
        & (df["q_dtype_w"].str.contains("float8"))
        & (df["run_1stage"] == 0)
    ].copy()

    def portable(name):
        parsed = kernel_names.try_parse(name)
        return parsed is None or parsed.family == kernel_names.CK2STAGES

    src = src[src["kernelName1"].map(portable) & src["kernelName2"].map(portable)]
    fp8 = "torch.float8_e4m3fn" if gfx == "gfx950" else "torch.float8_e4m3fnuz"
    src["gfx"], src["cu_num"] = gfx, cu_num
    src["q_dtype_a"] = src["q_dtype_w"] = fp8
    # Spread the sample over kernel families, block_m, ksplit and token counts.
    src = src.sort_values(["kernelName1", "block_m", "ksplit", "token"])
    step = max(1, len(src) // limit)
    sample = src.iloc[::step].head(limit)
    cases = []
    for r in sample.itertuples():
        act = str(r.act_type).split(".")[-1]
        out = "bf16" if "bfloat16" in str(r.dtype) else "fp16"
        cases.append(
            (int(r.token), int(r.model_dim), int(r.inter_dim), int(r.expert),
             int(r.topk), act, bool(r.doweight_stage1), out, False)
        )  # fmt: skip
    return src, cases


def cases(path_name, synth_cases=(), gfx=None, cu_num=None):
    if path_name != "a8w8_pt":
        raise SystemExit(f"no GPU cases for path {path_name}")
    selected = list(_A8W8_PT_DEFAULT)
    if (gfx, cu_num) == ("gfx942", 80):
        selected.extend(_A8W8_PT_GFX942_CU80)
    return selected + list(synth_cases)


def _clear_dispatch_caches():
    import aiter.fused_moe as fm
    from aiter.moev2.migration import a8w8_pt

    fm.get_2stage_cfgs.cache_clear()
    a8w8_pt._dispatch.cache_clear()
    a8w8_pt._resolve.cache_clear()


def _raised(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def _v2_outcome(path_name, call):
    """What the path did with `call`: "not claimed", "declined", "raised" or "ran"."""
    import dataclasses

    import aiter.fused_moe as fm
    from aiter.moev2 import paths
    from moe_dispatch.differential import moev2_mode

    path = paths.registered()[path_name]
    outcome = ["not claimed"]

    def run(c):
        outcome[0] = "raised"
        out = path.run(c)
        outcome[0] = "declined" if out is paths.DECLINED else "ran"
        return out

    paths._REGISTRY[path_name] = dataclasses.replace(path, run=run)
    try:
        with moev2_mode(path_name):
            _raised(lambda: fm.fused_moe(**call))
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    return outcome[0]


def run_case(path_name, case):
    """Return a problem string, or None when legacy and v2 agree."""
    import torch

    import aiter.fused_moe as fm
    from moe_dispatch.differential import captured_logs, moev2_mode, run_differential

    call = _make_a8w8_pt_call(*case)
    with moev2_mode("off"), captured_logs():
        legacy_error = _raised(lambda: fm.fused_moe(**call))  # warm-up: JIT logs here
    torch.cuda.synchronize()
    _clear_dispatch_caches()
    with moev2_mode("off"), captured_logs() as legacy_logs:
        legacy_error = _raised(lambda: fm.fused_moe(**call))
    _clear_dispatch_caches()
    with moev2_mode(path_name), captured_logs() as v2_logs:
        v2_error = _raised(lambda: fm.fused_moe(**call))
    if legacy_logs != v2_logs:
        return f"{case}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    expected = _RAISES.get(case[9] if len(case) > 9 else None)
    if expected is not None:
        if not (legacy_error and legacy_error == v2_error and expected in v2_error):
            return (
                f"{case}: expected both to raise {expected!r}; "
                f"legacy raises {legacy_error}; v2 raises {v2_error}"
            )
        _clear_dispatch_caches()
        outcome = _v2_outcome(path_name, call)
        return None if outcome == "raised" else f"{case}: v2 {outcome}, not raised"
    if legacy_error or v2_error:
        return (
            f"{case}: expected successful execution; "
            f"legacy raises {legacy_error}; v2 raises {v2_error}"
        )
    try:
        run_differential(path_name, call)
    except AssertionError as exc:
        return f"{case}: {exc}"
    return None


def _child(args):
    import torch

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    synth_cases = json.loads(os.environ.get("_MOEV2_SYNTH_CASES", "[]"))
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    todo = cases(args.path, [tuple(c) for c in synth_cases], gfx=gfx, cu_num=cu_num)
    print(f"{gfx} cu{cu_num} torch {torch.__version__}: {len(todo)} cases")
    problems = []
    for case in todo:
        problem = run_case(args.path, case)
        print(("FAIL " if problem else "ok   ") + str(case), flush=True)
        if problem:
            problems.append(problem)
    for line in problems:
        print("  " + line)
    print(f"{len(problems)} problems")
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True)
    parser.add_argument("--synth-table", action="store_true")
    parser.add_argument("--synth-limit", type=int, default=48)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    if args.child or not args.synth_table:
        return _child(args)

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    rows, synth = _synth_rows(
        AITER_CONFIGS.AITER_CONFIG_FMOE_FILE, gfx, cu_num, args.synth_limit
    )
    table = pathlib.Path(tempfile.mkdtemp(prefix="moev2_gpu_")) / "tuned_fmoe.csv"
    rows.to_csv(table, index=False)
    print(f"synthesized {len(rows)} per-tensor rows for {gfx} cu{cu_num} in {table}")
    env = dict(
        os.environ, AITER_CONFIG_FMOE=str(table), _MOEV2_SYNTH_CASES=json.dumps(synth)
    )
    cmd = [sys.executable, __file__, "--path", args.path, "--child"]
    return subprocess.run(cmd, env=env, cwd=_REPO, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
