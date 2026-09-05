# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end regression of exact-kid ``opus_bmm`` vs torch.bmm.

Usage:
    python3 op_tests/test_opus_a16w16_gemm.py --kid KID [-m M -n N -k K -b B]
    python3 op_tests/test_opus_a16w16_gemm.py --csv_file <shape_csv>

    # opus-only sweep in CUDA-graph mode, golden-checked:
    python3 op_tests/test_opus_a16w16_gemm.py --opus_sweep -n 2048 -k 7168
"""

import argparse
import sys
from pathlib import Path

import pytest
import torch

# Skip on unsupported arch via the same probe opus uses at import time.
from aiter.ops.opus._arch import _detect_arch, _device_arch_and_cu
from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan

_arch_ok, _detected_gfx = _detect_arch({"gfx950", "gfx942", "gfx1250"})

from aiter.ops.opus import opus_bmm, opus_gemm
from aiter.test_common import checkAllclose, run_perftest

_DEFAULT_TUNED_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv"
)


def _torch_ref(A: torch.Tensor, B: torch.Tensor, out_dtype):
    # A: [batch, M, K], B: [N, K] or [batch, N, K] -> bmm.
    # run_torch computes in fp32 then casts to match the opus path.
    if B.dim() == 2:
        return torch.einsum("bmk,nk->bmn", A.float(), B.float()).to(out_dtype)
    return torch.bmm(A.float(), B.float().transpose(-1, -2)).to(out_dtype)


def _make_b(batch: int, N: int, K: int) -> torch.Tensor:
    """Build the physical dense ``[batch, N, K]`` weight contract."""
    B2D = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    return B2D.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _run_exact_a16w16(
    A: torch.Tensor,
    B: torch.Tensor,
    Y: torch.Tensor,
    *,
    kid: int,
    split_k: int,
    use_graph: bool,
):
    kwargs = {"kid": kid, "split_k": split_k}
    if not use_graph:
        return run_perftest(opus_bmm, A, B, Y, **kwargs)

    arch, cu_num = _device_arch_and_cu(A.device)
    plan = _get_cached_a16w16_launch_plan(
        arch,
        A.shape[1],
        B.shape[1],
        A.shape[2],
        A.shape[0],
        cu_num,
        False,
        A.dtype,
        Y.dtype,
        kid,
        split_k,
    )
    workspace = (
        torch.empty(
            plan.workspace_spec.shape,
            dtype=plan.workspace_spec.dtype,
            device=A.device,
        )
        if plan.workspace_spec is not None
        else None
    )
    kwargs["workspace"] = workspace

    opus_bmm(A, B, Y, **kwargs)
    current = torch.cuda.current_stream(A.device)
    side = torch.cuda.Stream(device=A.device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        opus_bmm(A, B, Y, **kwargs)
    side.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        opus_bmm(A, B, Y, **kwargs)
    current.wait_stream(side)
    _, us = run_perftest(graph.replay, use_cuda_event=True)
    return Y, us


def run_a16w16_case(
    batch: int,
    M: int,
    N: int,
    K: int,
    *,
    kid: int,
    split_k: int = 0,
    out_dtype=torch.bfloat16,
    use_graph: bool = False,
):
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
    B = _make_b(batch, N, K)
    Y = torch.empty((batch, M, N), device="cuda", dtype=out_dtype)

    ref = _torch_ref(A, B, out_dtype)

    Y, us = _run_exact_a16w16(
        A,
        B,
        Y,
        kid=kid,
        split_k=split_k,
        use_graph=use_graph,
    )

    err = checkAllclose(
        Y,
        ref,
        msg=f"a16w16 b={batch} m={M} n={N} k={K}",
        rtol=0.1,
        atol=0.5,
    )
    flops = 2.0 * batch * M * N * K
    tflops = flops / us / 1e6
    print(
        f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} "
        f"| {us:.1f}us | {tflops:.2f} TFLOPs | err={err}"
    )
    return err


def load_shapes_from_csv(csv_path, *, default_kid=None, default_split_k=0):
    import pandas as pd

    df = pd.read_csv(csv_path)
    kid_column = next(
        (name for name in ("kernelId", "solidx", "kid") if name in df), None
    )
    split_column = next((name for name in ("splitK", "split_k") if name in df), None)
    if kid_column is None and default_kid is None:
        raise ValueError(
            "exact-kid CSV sweep needs a kernelId/solidx/kid column or --kid"
        )
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            (
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                int(default_kid if default_kid is not None else row[kid_column]),
                (
                    int(row[split_column])
                    if split_column is not None
                    else int(default_split_k)
                ),
            )
        )
    return list(dict.fromkeys(rows))


def run_a16w16_csv_sweep(
    csv_path: str,
    batch: int = 1,
    *,
    kid: int | None = None,
    split_k: int = 0,
    out_dtype=torch.bfloat16,
    use_graph: bool = False,
):
    shapes = load_shapes_from_csv(csv_path, default_kid=kid, default_split_k=split_k)
    return _run_a16w16_sweep(
        shapes,
        source=csv_path,
        batch=batch,
        out_dtype=out_dtype,
        use_graph=use_graph,
    )


def _run_a16w16_sweep(
    shapes,
    *,
    source: str,
    batch: int,
    out_dtype: torch.dtype,
    use_graph: bool,
):
    print(f"\n{'=' * 80}")
    mode = "graph" if use_graph else "eager"
    print(
        f"a16w16 sweep from {source}: {len(shapes)} unique shapes, "
        f"batch={batch}, mode={mode}"
    )
    print("=" * 80)
    passed = failed = 0
    for M, N, K, row_kid, row_split_k in shapes:
        tag = (
            f"a16w16 b={batch} M={M} N={N} K={K} "
            f"kid={row_kid} split_k={row_split_k}"
        )
        try:
            A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
            B = _make_b(batch, N, K)
            Y = torch.empty((batch, M, N), device="cuda", dtype=out_dtype)
            ref = _torch_ref(A, B, out_dtype)
            Y, us = _run_exact_a16w16(
                A,
                B,
                Y,
                kid=row_kid,
                split_k=row_split_k,
                use_graph=use_graph,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = 2.0 * batch * M * N * K / us / 1e6
            print(f"[PASS] {tag} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(shapes)}")
    return failed == 0


def load_opus_sweep_shapes(csv_path: str, *, N: int, K: int, out_dtype):
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
        "libtype",
        "solidx",
        "splitK",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"OPUS sweep CSV is missing columns {missing}")

    arch, cu_num = _device_arch_and_cu(torch.device("cuda"))

    def is_false(column):
        return df[column].astype(str).str.strip().str.lower().isin(("false", "0"))

    rows = df[
        df["gfx"].astype(str).str.lower().eq(arch)
        & df["cu_num"].eq(cu_num)
        & df["N"].eq(N)
        & df["K"].eq(K)
        & df["libtype"].astype(str).str.lower().eq("opus")
        & df["dtype"].astype(str).eq(str(torch.bfloat16))
        & df["outdtype"].astype(str).eq(str(out_dtype))
        & is_false("bias")
        & is_false("scaleAB")
        & is_false("bpreshuffle")
    ]
    shapes = [
        (int(row.M), int(row.N), int(row.K), int(row.solidx), int(row.splitK))
        for row in rows.itertuples(index=False)
    ]
    if not shapes:
        raise ValueError(
            f"no OPUS tuned rows for gfx={arch}, cu_num={cu_num}, N={N}, K={K}, "
            f"outdtype={out_dtype} in {csv_path}"
        )
    return list(dict.fromkeys(shapes))


def run_a16w16_opus_sweep(
    csv_path: str,
    *,
    N: int,
    K: int,
    out_dtype: torch.dtype,
):
    shapes = load_opus_sweep_shapes(csv_path, N=N, K=K, out_dtype=out_dtype)
    return _run_a16w16_sweep(
        shapes,
        source=csv_path,
        batch=1,
        out_dtype=out_dtype,
        use_graph=True,
    )


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(props.gcnArchName).split(":", 1)[0].lower()


def _assert_matches_golden(actual, A, B, bias=None):
    golden = A.float() @ B.float().transpose(-1, -2)
    if bias is not None:
        golden = golden + bias.float()
    # BF16 output has one final rounding; fp32 output is normally much tighter.
    atol = 0.5 if actual.dtype == torch.bfloat16 else 0.05
    rtol = 0.03 if actual.dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual.float(), golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("kid", "M", "N", "K", "split_k"),
    ((200, 64, 64, 512, 2), (1400, 192, 256, 128, 0)),
)
def test_gfx950_logical_2d_gemm_matches_torch(kid, M, N, K, split_k):
    """The public GEMM adapter adds only a no-copy batch-one raw view."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    torch.manual_seed(0x2D950 + kid)
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    actual = opus_gemm(A, B, Y, kid=kid, split_k=split_k)
    torch.cuda.synchronize()
    assert actual is Y
    _assert_matches_golden(actual, A, B)


def test_gfx950_batch_first_bmm_matches_torch():
    """The public BMM contract preserves a real batch dimension."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    torch.manual_seed(0xB950)
    A = torch.randn((2, 192, 128), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((2, 256, 128), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((2, 192, 256), device="cuda", dtype=torch.bfloat16)
    actual = opus_bmm(A, B, Y, kid=1400)
    torch.cuda.synchronize()
    assert actual is Y
    _assert_matches_golden(actual, A, B)


@pytest.mark.parametrize(
    ("arch", "kid", "M", "N", "K", "split_k", "out_dtype"),
    [
        ("gfx950", 200, 64, 64, 512, 2, torch.bfloat16),
        ("gfx950", 200, 64, 64, 512, 2, torch.float32),
        ("gfx942", 10200, 128, 128, 512, 2, torch.float32),
        ("gfx942", 10210, 128, 128, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.float32),
    ],
)
def test_split_k_matches_torch_golden(arch, kid, M, N, K, split_k, out_dtype):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    torch.manual_seed(8192 + kid)
    A = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, M, N), device="cuda", dtype=out_dtype)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=kid,
        split_k=split_k,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B)


@pytest.mark.parametrize("kid", (1400, 6400))
def test_gfx950_mono_fp32_overwrites_poisoned_output(kid):
    """Regress the ordinary and 4G-safe mono FP32 physical-store paths."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")

    torch.manual_seed(0x950000 + kid)
    A = torch.randn((1, 192, 128), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 256, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.full((1, 192, 256), 12345.0, device="cuda", dtype=torch.float32)

    actual = opus_bmm(
        A,
        B,
        out,
        kid=kid,
    )
    torch.cuda.synchronize()

    assert actual is out
    assert int((actual != 12345.0).sum().item()) == actual.numel()
    _assert_matches_golden(actual, A, B)


def test_gfx950_bias_dtype_rules_and_numerics():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    A = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), device="cuda", dtype=torch.bfloat16)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=200,
        bias=bias,
        split_k=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)

    with pytest.raises(RuntimeError, match="bias dtype must match Y dtype"):
        opus_bmm(
            A,
            B,
            Y,
            kid=200,
            bias=bias.float(),
            split_k=2,
        )


def test_gfx942_workspace_kid_rejects_bias_without_framework_fallback():
    if _runtime_arch() != "gfx942":
        pytest.skip("requires gfx942 hardware")
    A = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 256, 4096), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 128, 256), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="rejects bias on split-K kernels"):
        opus_bmm(
            A,
            B,
            Y,
            kid=10201,
            bias=bias,
            split_k=2,
        )


def test_gfx1250_bf16_output_accepts_fp32_bias():
    if _runtime_arch() != "gfx1250":
        pytest.skip("requires gfx1250 hardware")
    A = torch.randn((1, 16, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 32, 512), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 16, 32), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((32,), device="cuda", dtype=torch.float32)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=20000,
        bias=bias,
        split_k=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)


def test_global_a16_stale_opus_row_keeps_framework_fallback(monkeypatch):
    import aiter.tuned_gemm as tuned

    warnings = []
    key = (
        "gfx942",
        304,
        32,
        256,
        1024,
        False,
        str(torch.bfloat16),
        str(torch.bfloat16),
        False,
        False,
    )
    row = {"libtype": "opus", "solidx": 200, "splitK": 2, "kernelName": ""}
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: {key: row})
    monkeypatch.setattr(tuned, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: 304)
    monkeypatch.setattr(tuned, "_opus_launch", object())
    monkeypatch.setattr(
        tuned.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        config = tuned.get_GEMM_A16W16_config(
            32,
            256,
            1024,
            False,
            str(torch.bfloat16),
            str(torch.bfloat16),
        )
    finally:
        tuned.get_GEMM_A16W16_config.cache_clear()

    assert (config["libtype"], config["solidx"]) == ("torch", 0)
    assert len(warnings) == 1
    assert "kid=200, splitK=2" in warnings[0]


def _capture_shape_driven_opus_launch(monkeypatch, *, arch, tuned_config):
    from aiter.ops import opus
    from aiter.ops.opus import gemm_op_a16w16, policy

    calls = []

    def capture(operation):
        def fake_launch(XQ, WQ, Y, **kwargs):
            calls.append((operation, XQ, WQ, Y, kwargs))
            return Y

        return fake_launch

    monkeypatch.setattr(
        gemm_op_a16w16,
        "_device_arch_and_cu",
        lambda _device: (arch, {"gfx950": 256, "gfx942": 304, "gfx1250": 80}[arch]),
    )
    monkeypatch.setattr(
        policy,
        "lookup_a16w16_opus_config",
        lambda **_kwargs: tuned_config,
    )
    monkeypatch.setattr(gemm_op_a16w16, "_launch_a16w16_gemm", capture("gemm"))
    monkeypatch.setattr(gemm_op_a16w16, "_launch_a16w16_bmm", capture("bmm"))
    return opus, calls


def _a16_policy_args(arch, M, N, K):
    return {
        "arch": arch,
        "M": M,
        "N": N,
        "K": K,
        "batch": 1,
        "cu_num": {"gfx950": 256, "gfx942": 304, "gfx1250": 80}[arch],
        "has_bias": False,
        "input_dtype": torch.bfloat16,
        "output_dtype": torch.bfloat16,
    }


def _mock_a16w16_policy_csv(monkeypatch, read_csv):
    from types import SimpleNamespace

    from aiter.ops.opus import policy

    monkeypatch.setattr(
        policy,
        "AITER_CONFIGS",
        SimpleNamespace(AITER_CONFIG_GEMM_BF16_FILE="test.csv"),
    )
    monkeypatch.setattr(policy.pd, "read_csv", read_csv)
    policy._load_a16w16_opus_tuned.cache_clear()
    return policy


@pytest.mark.parametrize("case", ("missing", "empty", "partial", "unreadable"))
def test_a16w16_policy_loader_bad_csv_is_a_miss(monkeypatch, case):
    import pandas as pd

    warnings = []
    errors = {
        "missing": FileNotFoundError(),
        "empty": pd.errors.EmptyDataError("empty CSV"),
        "unreadable": pd.errors.ParserError("bad CSV"),
    }

    def read_csv(_path):
        if case in errors:
            raise errors[case]
        return pd.DataFrame({"libtype": ["opus"], "solidx": [200]})

    policy = _mock_a16w16_policy_csv(monkeypatch, read_csv)
    monkeypatch.setattr(
        policy.logger,
        "warning",
        lambda *args, **_kwargs: warnings.append(args),
    )
    try:
        assert policy._load_a16w16_opus_tuned() == {}
        assert bool(warnings) == (case in ("partial", "unreadable"))
    finally:
        policy._load_a16w16_opus_tuned.cache_clear()


def test_a16w16_policy_loader_skips_malformed_kid_and_splitk_rows(monkeypatch):
    import pandas as pd

    from aiter.ops.opus import policy

    key = (
        "gfx950",
        256,
        128,
        64,
        512,
        False,
        "torch.bfloat16",
        "torch.bfloat16",
        False,
        False,
    )
    columns = (
        *policy._A16W16_TUNED_KEY_COLUMNS,
        "libtype",
        "solidx",
        "splitK",
        "us",
    )
    rows = [
        (*key, "opus", "not-a-kid", 2, 1.0),
        (*key, "opus", 200, "not-a-split", 1.5),
        (*key, "opus", 200, -1, 1.75),
        (*key, "opus", "200", "2", 2.0),
    ]
    policy = _mock_a16w16_policy_csv(
        monkeypatch, lambda _path: pd.DataFrame(rows, columns=columns)
    )
    try:
        configs = policy._load_a16w16_opus_tuned()
        assert [(row["solidx"], row["splitK"]) for row in configs.values()] == [
            (200, 2)
        ]
    finally:
        policy._load_a16w16_opus_tuned.cache_clear()


@pytest.mark.parametrize(
    ("arch", "shape", "expected_kid"),
    (
        ("gfx950", (128, 64, 512), 1200),
        ("gfx942", (32, 256, 1024), 10300),
        ("gfx1250", (32, 128, 512), 20007),
    ),
)
def test_a16w16_heuristic_baseline_kid(arch, shape, expected_kid):
    from aiter.ops.opus.policy import resolve_a16w16_heuristic_candidate

    M, N, K = shape
    plan = resolve_a16w16_heuristic_candidate(**_a16_policy_args(arch, M, N, K))
    assert plan.resolved_kid == expected_kid


def test_shape_driven_opus_selection_and_rank_route(monkeypatch):
    from aiter.ops.opus import gemm_op_a16w16, policy

    tuned = [None]
    opus, calls = _capture_shape_driven_opus_launch(
        monkeypatch, arch="gfx950", tuned_config=None
    )
    monkeypatch.setattr(policy, "lookup_a16w16_opus_config", lambda **_kwargs: tuned[0])
    A = torch.empty((128, 512), device="meta", dtype=torch.bfloat16)
    B = torch.empty((64, 512), device="meta", dtype=torch.bfloat16)

    opus.gemm_a16w16_opus(A, B)
    tuned[0] = {"solidx": 200, "splitK": 2}
    opus.gemm_a16w16_opus(A, B)
    opus.gemm_a16w16_opus(A, B, kernelId=206, splitK=3)
    tuned[0] = None
    opus.gemm_a16w16_opus(
        A.unsqueeze(0).expand(2, -1, -1).contiguous(),
        B.unsqueeze(0).expand(2, -1, -1).contiguous(),
    )

    assert [(op, args["kid"], args["split_k"]) for op, *_, args in calls] == [
        ("gemm", 1200, 0),
        ("gemm", 200, 2),
        ("gemm", 206, 3),
        ("bmm", 1200, 0),
    ]
    tuned[0] = {"solidx": -1, "splitK": 0}
    monkeypatch.setattr(
        policy,
        "resolve_a16w16_heuristic_candidate",
        lambda **_kwargs: pytest.fail("a present tuned row must not use heuristic"),
    )
    monkeypatch.setattr(
        gemm_op_a16w16,
        "_launch_a16w16_gemm",
        lambda _XQ, _WQ, _Y, **kwargs: (_ for _ in ()).throw(
            ValueError(f"unknown OPUS kid {kwargs['kid']}")
        ),
    )
    with pytest.raises(ValueError, match="unknown OPUS kid -1"):
        opus.gemm_a16w16_opus(A, B)


def test_legacy_a16w16_tune_routes_to_family_executor(monkeypatch):
    from aiter.ops import deepgemm
    from aiter.ops.opus import gemm_op_a16w16

    calls = []

    def capture(XQ, WQ, Y, bias=None, **kwargs):
        calls.append((XQ, WQ, Y, bias, kwargs))
        return Y

    monkeypatch.setattr(gemm_op_a16w16, "_execute_a16w16", capture)
    monkeypatch.setattr(
        gemm_op_a16w16,
        "_launch_a16w16_gemm",
        lambda *_args, **_kwargs: pytest.fail("compatibility used public GEMM route"),
    )
    monkeypatch.setattr(
        gemm_op_a16w16,
        "_launch_a16w16_bmm",
        lambda *_args, **_kwargs: pytest.fail("GEMM compatibility used BMM"),
    )
    XQ = torch.empty((1, 8, 16), device="meta", dtype=torch.bfloat16)
    WQ = torch.empty((1, 32, 16), device="meta", dtype=torch.bfloat16)
    Y = torch.empty((1, 8, 32), device="meta", dtype=torch.bfloat16)

    assert gemm_op_a16w16.opus_gemm_a16w16_tune(XQ, WQ, Y, 206, 3) is Y
    assert gemm_op_a16w16.opus_gemm_a16w16_tune(XQ, WQ, Y, 207, splitK=4) is Y
    with pytest.warns(DeprecationWarning, match="has moved"):
        assert deepgemm.opus_gemm_a16w16_tune(XQ, WQ, Y, 208, 5) is Y
    assert [
        (tuple(xq.shape), tuple(wq.shape), tuple(y.shape), bias, kwargs)
        for xq, wq, y, bias, kwargs in calls
    ] == [
        ((1, 8, 16), (1, 32, 16), (1, 8, 32), None, {"kid": 206, "split_k": 3}),
        ((1, 8, 16), (1, 32, 16), (1, 8, 32), None, {"kid": 207, "split_k": 4}),
        ((1, 8, 16), (1, 32, 16), (1, 8, 32), None, {"kid": 208, "split_k": 5}),
    ]


@pytest.mark.parametrize("arch", ("gfx950", "gfx1250"))
def test_shape_driven_opus_heuristic_rejects_over_4g_shape(arch):
    from aiter.ops.opus.policy import resolve_a16w16_heuristic_candidate

    with pytest.raises(RuntimeError, match="refuses >4 GiB shape"):
        resolve_a16w16_heuristic_candidate(**_a16_policy_args(arch, 1, 1, 1 << 31))


def test_gfx950_heuristic_does_not_try_secondary_kids(monkeypatch):
    from aiter.ops.opus import policy

    attempted = []
    monkeypatch.setattr(
        policy,
        "select_a16w16_heuristic_kid",
        lambda **_kwargs: 1200,
    )

    def reject_candidate(**kwargs):
        attempted.append(kwargs["kid"])

    monkeypatch.setattr(policy, "_resolve_a16w16_candidate", reject_candidate)
    plan = policy.resolve_a16w16_heuristic_candidate(
        **_a16_policy_args("gfx950", 128, 64, 512)
    )

    assert plan is None
    assert attempted == [1200]


if __name__ == "__main__":
    if not _arch_ok:
        print(
            "[skip] test_opus_a16w16_gemm requires "
            f"gfx950/gfx942/gfx1250 (detected {_detected_gfx!r})"
        )
        sys.exit(0)
    if len(sys.argv) == 1:
        sys.exit(pytest.main([__file__]))
    parser = argparse.ArgumentParser(
        description="End-to-end exact-kid test for aiter.ops.opus.opus_bmm"
    )
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("--kid", type=int, default=None)
    parser.add_argument("--split-k", type=int, default=0)
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Optional CSV with M,N,K columns. When given, skips the "
            "single-shape test and runs a full sweep instead."
        ),
    )
    parser.add_argument(
        "--opus_sweep",
        action="store_true",
        help=(
            "Run the CUDA-graph-mode opus_gemm sweep (golden-checked) over "
            "the M values whose tuned winner is opus for the given N/K in the "
            "tuned CSV (default: dsv4_bf16_tuned_gemm.csv)."
        ),
    )
    parser.add_argument(
        "--tuned_csv",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Tuned GEMM CSV used by --opus_sweep to pick opus shapes. "
            "Defaults to the shipped dsv4_bf16_tuned_gemm.csv."
        ),
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use CUDA-graph mode for the single-shape / --csv_file paths too.",
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.opus_sweep and args.csv_file is not None:
        parser.error("--opus_sweep and --csv_file are mutually exclusive")
    if args.tuned_csv is not None and not args.opus_sweep:
        parser.error("--tuned_csv requires --opus_sweep")

    if args.opus_sweep:
        ok = run_a16w16_opus_sweep(
            args.tuned_csv or str(_DEFAULT_TUNED_CSV),
            N=args.n,
            K=args.k,
            out_dtype=out_dtype,
        )
        if not ok:
            sys.exit(1)
    elif args.csv_file is not None:
        ok = run_a16w16_csv_sweep(
            args.csv_file,
            batch=args.batch,
            kid=args.kid,
            split_k=args.split_k,
            out_dtype=out_dtype,
            use_graph=args.graph,
        )
        if not ok:
            sys.exit(1)
    else:
        if args.kid is None:
            parser.error("--kid is required for a single-shape run")
        k_eff = max(args.k, 128)
        run_a16w16_case(
            args.batch,
            args.m,
            args.n,
            k_eff,
            kid=args.kid,
            split_k=args.split_k,
            out_dtype=out_dtype,
            use_graph=args.graph,
        )
