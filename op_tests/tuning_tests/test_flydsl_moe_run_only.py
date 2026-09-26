# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cold AOT/RUN_ONLY regressions for the baseline FP8 whole-graph MoE path.

This is a collected integration test, not the op/performance sweep. Explicitly
select it on two reserved, visible gfx942 GPUs with the same CU count and set
AITER_TEST_EXPECTED_GFX=gfx942. Missing GPUs or the wrong architecture fail;
they are not silently skipped. Visibility and GPU reservation are the caller's
responsibility. Only logical devices 0 and 1 are used.

Every case starts with an empty private FlyDSL cache, compiles a real temporary
CSV through parse_csv/compile_one_config in one process, then calls the public
fused_moe API in a fresh RUN_ONLY process. HIP sorting and per-token quantization
may build in a private AITER_JIT_DIR; they are outside FlyDSL's RUN_ONLY contract.
No runtime warmup, fake timings, compact tasks, new Down paths, or fused clears
are substituted for the production implementation.
"""

import csv
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from op_tests.tuning_tests.test_flydsl_moe_cache import (
    _assert_lightweight_imports,
    _forbid_cuda_access,
    _whole_graph_row,
    _write_csv,
)

_ROOT = Path(__file__).resolve().parents[2]
_PREFIX = "impl__flydsl_gfx942__"
_DECODE = "16_16_16_False"
_PREFILL = "64_128_128_True"
_RESULT_MARKER = "FLYDSL_MOE_RUN_ONLY_RESULT "
_COMPILE_MARKER = "FLYDSL_MOE_AOT_RESULT "
_MISS_MARKER = "FLYDSL_MOE_EXPECTED_CACHE_MISS "
_COMPILE_TIMEOUT = 1200
_RUN_TIMEOUT = 600


def _require_gfx942_devices():
    """Fail closed when an explicitly scheduled architecture test cannot run."""
    import torch

    expected = os.environ.get("AITER_TEST_EXPECTED_GFX")
    if expected != "gfx942":
        raise AssertionError(
            "Set AITER_TEST_EXPECTED_GFX=gfx942 for this baseline-only GPU test; "
            f"got {expected!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise AssertionError(
            "Two reserved, visible gfx942 GPUs are required for cuda:0 -> cuda:1 "
            "-> cuda:0 with an independently selected ambient device"
        )
    properties = [torch.cuda.get_device_properties(index) for index in (0, 1)]
    for index, props in enumerate(properties):
        actual = props.gcnArchName.split(":", 1)[0]
        if actual != expected:
            raise AssertionError(f"cuda:{index}: expected {expected}, got {actual}")
        if props.multi_processor_count <= 0:
            raise AssertionError(f"cuda:{index}: invalid CU count")
    if properties[0].multi_processor_count != properties[1].multi_processor_count:
        raise AssertionError("The cross-device regression needs matching CU counts")
    return properties[0].multi_processor_count


def _read_row(config_file):
    with open(config_file, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"Expected one isolated config row, got {len(rows)}")
    return rows[0]


def _cache_snapshot(directory):
    """Check that RUN_ONLY neither creates nor rewrites compiled artifacts."""
    result = {}
    for path in sorted(Path(directory).rglob("*.pkl")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(directory))] = (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            digest.hexdigest(),
        )
    return result


def _quantize_fp8(value, *, per_token):
    """Independent torch quantization; no aiter/FlyDSL reference helpers."""
    import torch

    value = value.float()
    amax = value.abs().amax(dim=-1, keepdim=True) if per_token else value.abs().amax()
    scale = amax / torch.finfo(torch.float8_e4m3fnuz).max
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = (value / safe_scale).clamp(-240.0, 240.0).to(torch.float8_e4m3fnuz)
    # A zero input reconstructs as zero for both the HIP per-token scale=1
    # convention and the baseline per-tensor scale=0 convention.
    return quantized, safe_scale.float()


def _make_inputs(row, device, *, zero_input=False):
    import torch

    from aiter.ops.shuffle import shuffle_weight

    batch, model_dim, inter_dim, experts, topk = (
        int(row[name]) for name in ("token", "model_dim", "inter_dim", "expert", "topk")
    )
    per_token = row["q_type"] == "QuantType.per_Token"
    swiglu = row["act_type"] == "ActivationType.Swiglu"
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = (
        torch.randn(
            (batch, model_dim), generator=generator, dtype=torch.bfloat16, device=device
        )
        * 0.5
    )
    if zero_input:
        hidden.zero_()
    w1 = torch.empty(
        (experts, 2 * inter_dim, model_dim), dtype=torch.float8_e4m3fnuz, device=device
    )
    w2 = torch.empty(
        (experts, model_dim, inter_dim), dtype=torch.float8_e4m3fnuz, device=device
    )
    scale1 = torch.empty(
        (experts, 2 * inter_dim if per_token else 1, 1),
        dtype=torch.float32,
        device=device,
    )
    scale2 = torch.empty(
        (experts, model_dim if per_token else 1, 1),
        dtype=torch.float32,
        device=device,
    )
    # Generate one expert at a time to bound temporary FP32 storage on the real
    # model shapes. Swiglu deliberately crosses both 7.0 and 1.5 clamp bounds.
    gateup_gain = (20.0 if swiglu else 2.0) / math.sqrt(model_dim)
    down_gain = 0.25 / math.sqrt(inter_dim)
    for expert in range(experts):
        raw1 = (
            torch.randn(
                (2 * inter_dim, model_dim),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            * gateup_gain
        )
        raw2 = (
            torch.randn(
                (model_dim, inter_dim),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            * down_gain
        )
        q1, s1 = _quantize_fp8(raw1, per_token=per_token)
        q2, s2 = _quantize_fp8(raw2, per_token=per_token)
        w1[expert].copy_(q1)
        w2[expert].copy_(q2)
        scale1[expert].copy_(s1)
        scale2[expert].copy_(s2)
    tokens = torch.arange(batch, dtype=torch.int32, device=device)
    ranks = torch.arange(topk, dtype=torch.int32, device=device)
    ids = ((3 * tokens[:, None] + ranks[None, :]) % experts).contiguous()
    routing = (
        torch.rand(
            (batch, topk), generator=generator, dtype=torch.float32, device=device
        )
        + 0.25
    )
    routing /= routing.sum(dim=-1, keepdim=True)
    shuffled1 = shuffle_weight(w1, (16, 16))
    shuffled2 = shuffle_weight(w2, (16, 16))
    shuffled1.is_shuffled = shuffled2.is_shuffled = True
    return {
        "hidden": hidden,
        "w1": w1,
        "w2": w2,
        "shuffled1": shuffled1,
        "shuffled2": shuffled2,
        "scale1": scale1,
        "scale2": scale2,
        "ids": ids,
        "routing": routing,
    }


def _torch_reference(row, inputs, swiglu_limit):
    """Match legacy FP8 numerics using independent torch matmuls and routing.

    Decode consumes BF16 activations internally despite the FP8 request dtype.
    Prefill quantizes both activation matrices. Each stage stores BF16, and
    routed Down contributions are rounded before the top-k reduction.
    """
    import torch
    from torch.nn import functional

    batch, inter_dim, model_dim, experts, topk = (
        int(row[name]) for name in ("token", "inter_dim", "model_dim", "expert", "topk")
    )
    prefill = row["kernelName1"].endswith("_True")
    per_token = row["q_type"] == "QuantType.per_Token"
    hidden = inputs["hidden"].float()
    if prefill:
        q, scale = _quantize_fp8(hidden, per_token=per_token)
        hidden = q.float() * scale
    middle = torch.empty(
        (batch, topk, inter_dim), dtype=torch.bfloat16, device=hidden.device
    )
    # This CPU routing list is only for the torch reference; production receives
    # the original device-side ids and runs its real HIP sorting implementation.
    ids_cpu = inputs["ids"].cpu()
    positions = [
        tuple(
            index.to(hidden.device)
            for index in (ids_cpu == expert).nonzero(as_tuple=True)
        )
        for expert in range(experts)
    ]
    clamp_hits = 0
    limit = 7.0 if swiglu_limit is None or swiglu_limit == 0 else float(swiglu_limit)
    for expert, (token, rank) in enumerate(positions):
        if token.numel() == 0:
            continue
        weight = inputs["w1"][expert].float() * inputs["scale1"][expert]
        gate, up = (hidden[token] @ weight.t()).chunk(2, dim=-1)
        if row["act_type"] == "ActivationType.Swiglu":
            clamp_hits += int(((gate > limit) | (up.abs() > limit)).sum().item())
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            activated = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        else:
            activated = functional.silu(gate) * up
        middle[token, rank] = activated.bfloat16()
    down_input = middle.float()
    if prefill:
        q, scale = _quantize_fp8(
            down_input.reshape(batch * topk, inter_dim), per_token=per_token
        )
        down_input = (q.float() * scale).reshape(batch, topk, inter_dim)
    contributions = torch.empty(
        (batch, topk, model_dim), dtype=torch.bfloat16, device=hidden.device
    )
    for expert, (token, rank) in enumerate(positions):
        if token.numel() == 0:
            continue
        weight = inputs["w2"][expert].float() * inputs["scale2"][expert]
        down = down_input[token, rank] @ weight.t()
        contributions[token, rank] = (
            down * inputs["routing"][token, rank, None]
        ).bfloat16()
    return contributions.float().sum(dim=1).bfloat16(), clamp_hits


def _compare(output, reference, label):
    import torch

    from aiter.test_common import checkAllclose

    actual, expected = output.float(), reference.float()
    if not torch.isfinite(actual).all().item():
        raise AssertionError(f"{label}: nonfinite output")
    if not torch.isfinite(expected).all().item():
        raise AssertionError(f"{label}: nonfinite independent torch reference")
    if expected.count_nonzero().item() and not actual.count_nonzero().item():
        raise AssertionError(f"{label}: the kernel produced no output")
    error = (actual - expected).square().sum()
    denominator = (actual.square() + expected.square()).sum().clamp_min(1e-30)
    logits_diff = (error / denominator).item()
    if not math.isfinite(logits_diff) or logits_diff > 0.01:
        raise AssertionError(f"{label}: logits_diff={logits_diff} exceeds 0.01")
    # Both checks are mandatory. checkAllclose logs/returns a ratio and does not
    # itself fail ordinary mismatches; do not accept its return value blindly.
    ratio = float(
        checkAllclose(
            actual,
            expected,
            rtol=0.03,
            atol=0.02,
            tol_err_ratio=0.05,
            catastrophic_check=True,
            msg=label,
        )
    )
    if not math.isfinite(ratio) or ratio > 0.05:
        raise AssertionError(f"{label}: checkAllclose err_ratio={ratio} exceeds 0.05")
    return {"logits_diff": logits_diff, "err_ratio": ratio}


def _compile_in_fresh_process(config_file, swiglu_limit):
    from aiter.aot.flydsl.moe import compile_one_config, parse_csv

    _assert_lightweight_imports()
    if os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") != "0":
        raise AssertionError("AOT must be the only process permitted to compile")
    before = {
        key: os.environ.get(key)
        for key in ("COMPILE_ONLY", "ARCH", "FLYDSL_GPU_ARCH", "CU_NUM")
    }
    # Keep inherited package/staging imports, then forbid all CUDA access in
    # the real parser/preloader rather than pretending no GPU is available.
    with _forbid_cuda_access():
        jobs = parse_csv(config_file)
        if len(jobs) != 1 or jobs[0]["stage"] != "whole_graph":
            raise AssertionError(f"Expected one real whole-graph CSV job: {jobs}")
        job = jobs[0]
        if swiglu_limit is not None:
            job = {**job, "swiglu_limit": swiglu_limit}
        result = compile_one_config(**job)
        _assert_lightweight_imports()
    after = {key: os.environ.get(key) for key in before}
    if before != after:
        raise AssertionError(f"AOT leaked environment overrides: {before} -> {after}")
    if result["compile_time"] is None or result["compile_arch"] != "gfx942":
        raise AssertionError(f"AOT failed: {result}")
    if not _cache_snapshot(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]):
        raise AssertionError("AOT returned success without any persistent artifacts")
    print(_COMPILE_MARKER + json.dumps(result), flush=True)


def _run_in_fresh_process(config_file, limits, *, expect_miss=False, zero_input=False):
    import torch

    cu_num = _require_gfx942_devices()
    if os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") != "1":
        raise AssertionError("Runtime must not silently JIT missing AOT artifacts")
    if os.environ.get("COMPILE_ONLY", "0") != "0":
        raise AssertionError("RUN_ONLY must execute kernels, not compile-only no-ops")
    row = _read_row(config_file)
    if row["gfx"] != "gfx942" or int(row["cu_num"]) != cu_num:
        raise AssertionError("The CSV target must match both actual devices")
    if not row["kernelName1"].startswith(_PREFIX):
        raise AssertionError(
            "The test must select the existing whole-graph implementation"
        )
    fused = importlib.import_module("aiter.fused_moe")
    backend = importlib.import_module("aiter.ops.flydsl.fused_moe_gfx942")
    activation = (
        backend.ActivationType.Swiglu
        if row["act_type"] == "ActivationType.Swiglu"
        else backend.ActivationType.Silu
    )
    quant_type = (
        backend.QuantType.per_Token
        if row["q_type"] == "QuantType.per_Token"
        else backend.QuantType.per_Tensor
    )
    if backend._get_compiled_kernel.cache_info().currsize:
        raise AssertionError("The runtime process must start without host launchers")
    cache_before = _cache_snapshot(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
    seen = set()
    targets = (0,) if expect_miss else (0, 1, 0)
    for target in targets:
        device = torch.device("cuda", target)
        with torch.cuda.device(device), torch.no_grad():
            inputs = _make_inputs(row, device, zero_input=zero_input)
            stream = torch.cuda.Stream(device=device)
            default_stream = torch.cuda.default_stream(device)
            if stream.cuda_stream == default_stream.cuda_stream:
                raise AssertionError("The regression requires a nondefault stream")
            for limit in limits:
                reference, clamp_hits = _torch_reference(row, inputs, limit)
                if (
                    activation == backend.ActivationType.Swiglu
                    and not zero_input
                    and clamp_hits == 0
                ):
                    raise AssertionError(
                        "Swiglu data did not exercise finite clamp semantics"
                    )
                output = torch.empty_like(inputs["hidden"])
                before = backend._get_compiled_kernel.cache_info()
                stream.wait_stream(torch.cuda.current_stream(device))
                ambient = 1 - target
                with torch.cuda.stream(stream):
                    # A missing write or launch is observable; the public API
                    # must also return this exact model-owned output buffer.
                    output.fill_(float("nan"))
                    torch.cuda.set_device(ambient)
                    try:
                        result = fused.fused_moe(
                            inputs["hidden"],
                            inputs["shuffled1"],
                            inputs["shuffled2"],
                            inputs["routing"],
                            inputs["ids"],
                            activation=activation,
                            quant_type=quant_type,
                            w1_scale=inputs["scale1"],
                            w2_scale=inputs["scale2"],
                            swiglu_limit=limit,
                            output=output,
                        )
                        if expect_miss:
                            raise AssertionError(
                                "RUN_ONLY accepted an uncompiled specialization"
                            )
                        if torch.cuda.current_device() != ambient:
                            raise AssertionError(
                                "fused_moe leaked the input device into its caller"
                            )
                        if (
                            result is not output
                            or result.data_ptr() != output.data_ptr()
                        ):
                            raise AssertionError(
                                "fused_moe did not preserve output buffer identity"
                            )
                        if result.device != device:
                            raise AssertionError(
                                "fused_moe returned output on the ambient device"
                            )
                        stream.synchronize()
                    except RuntimeError as error:
                        # This is a negative assertion, not an error demotion:
                        # only FlyDSL's named RUN_ONLY cache-miss error passes.
                        message = str(error)
                        if (
                            not expect_miss
                            or "FLYDSL_RUNTIME_RUN_ONLY=1" not in message
                            or "AOT cache" not in message
                        ):
                            raise
                        if "launch_batch1" not in message:
                            raise AssertionError(
                                f"Unexpected missing auxiliary: {message}"
                            ) from error
                        if (
                            _cache_snapshot(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
                            != cache_before
                        ):
                            raise AssertionError(
                                "RUN_ONLY wrote artifacts while reporting a miss"
                            )
                        print(
                            _MISS_MARKER
                            + json.dumps({"limit": limit, "error": message}),
                            flush=True,
                        )
                        return
                    finally:
                        torch.cuda.set_device(target)
                # Order the torch checks after nondefault-stream output, too;
                # synchronizing the host alone is not the stream-order contract.
                torch.cuda.current_stream(device).wait_stream(stream)
                after = backend._get_compiled_kernel.cache_info()
                if (after.hits + after.misses) - (before.hits + before.misses) != 2:
                    raise AssertionError(
                        "The public API did not execute both whole-graph stages"
                    )
                signature = (target, limit)
                if signature in seen and after.misses != before.misses:
                    raise AssertionError(
                        "Returning to a device rebuilt its cached launchers"
                    )
                seen.add(signature)
                label = (
                    f"{row['kernelName1']} B{row['token']} cuda:{target} limit={limit}"
                )
                metrics = _compare(output, reference, label)
                print(
                    _RESULT_MARKER
                    + json.dumps(
                        {
                            "device": target,
                            "ambient_device": ambient,
                            "stream": stream.cuda_stream,
                            "limit": limit,
                            "clamp_hits": clamp_hits,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
            del inputs
    cache_after = _cache_snapshot(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
    if cache_before != cache_after:
        raise AssertionError("RUN_ONLY created or rewrote FlyDSL artifacts")


def _worker_main():
    """Private subprocess entry; importing this test never starts GPU work."""
    phase, path, options_json = sys.argv[1:]
    options = json.loads(options_json)
    if phase == "compile":
        _compile_in_fresh_process(path, options["limit"])
    elif phase == "run":
        _run_in_fresh_process(path, **options)
    else:
        raise ValueError(f"Unknown test worker phase: {phase}")


def _run_worker(phase, config_file, env, **options):
    command = [
        sys.executable,
        "-c",
        "from op_tests.tuning_tests.test_flydsl_moe_run_only import _worker_main; _worker_main()",
        phase,
        str(config_file),
        json.dumps(options),
    ]
    timeout = _COMPILE_TIMEOUT if phase == "compile" else _RUN_TIMEOUT
    process = subprocess.Popen(
        command,
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill only this test's process group, including compiler children.
            # A stuck compiler/HIP call must not outlive the test timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise AssertionError(
                    f"{phase} did not exit within 10s of SIGKILL"
                ) from error
            raise AssertionError(
                f"{phase} timed out after {timeout}s\n{stdout[-12000:]}\n{stderr[-12000:]}"
            ) from None
        if process.returncode != 0:
            raise AssertionError(
                f"{phase} exited with {process.returncode}\n{stdout[-20000:]}\n{stderr[-20000:]}"
            )
    finally:
        # Avoid Popen.__exit__'s unbounded wait if HIP is stuck during shutdown.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.stdout.close()
        process.stderr.close()
    return stdout


class TestFlydslMoeRunOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cu_num = _require_gfx942_devices()

    def _model_row(self, model, *, batch, inter_dim, experts):
        names = {
            "qwen": "qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv",
            "hunyuan": "hunyuan3_fp8_per_tensor_tuned_fmoe.csv",
            "minimax": "minimax_m3_fp8_ptpc_tuned_fmoe.csv",
        }
        path = _ROOT / "aiter" / "configs" / "model_configs" / names[model]
        with path.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["gfx"] == "gfx942"
                and int(row["token"]) == batch
                and int(row["inter_dim"]) == inter_dim
                and int(row["expert"]) == experts
                and row["kernelName1"].startswith(_PREFIX)
            ]
        self.assertEqual(len(rows), 1, f"Missing/ambiguous baseline shape in {path}")
        # Preserve published dimensions, top-k, activation, quantization, and
        # kernel tiles. Only retarget the row to the verified physical CU count.
        return {**rows[0], "cu_num": str(self.cu_num)}

    def _exercise(
        self,
        row,
        *,
        compile_limit=None,
        limits=(None,),
        expect_miss=False,
        empty_cache=False,
        zero_input=False,
    ):
        row = {**row, "gfx": "gfx942", "cu_num": str(self.cu_num)}
        self.assertIn(row["q_type"], ("QuantType.per_Token", "QuantType.per_Tensor"))
        self.assertIn(row["act_type"], ("ActivationType.Silu", "ActivationType.Swiglu"))
        self.assertEqual(row["q_dtype_w"], "torch.float8_e4m3fnuz")
        config = row["kernelName1"].removeprefix(_PREFIX)
        self.assertIn(config, (_DECODE, _PREFILL, "64_128_256_True", "64_256_128_True"))
        with tempfile.TemporaryDirectory(prefix="aiter_moe_run_only_") as directory:
            root = Path(directory)
            cache = root / "flydsl_cache"
            cache.mkdir()
            self.assertEqual(list(cache.iterdir()), [])
            config_file = root / "moe.csv"
            _write_csv(config_file, [row])
            env = dict(os.environ)
            for key in (
                "COMPILE_ONLY",
                "MOE_PREFILL_TILE_K",
                "HSA_OVERRIDE_GFX_VERSION",
            ):
                env.pop(key, None)
            env.update(
                PYTHONPATH=str(_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
                PYTHONDONTWRITEBYTECODE="1",
                FLYDSL_RUNTIME_CACHE_DIR=str(cache),
                FLYDSL_RUNTIME_ENABLE_CACHE="1",
                FLYDSL_RUNTIME_RUN_ONLY="1",
                FLYDSL_DUMP_IR="0",
                ARCH="gfx942",
                FLYDSL_GPU_ARCH="gfx942",
                GPU_ARCHS="gfx942",
                CU_NUM=str(self.cu_num),
                AITER_CONFIG_FMOE=str(config_file),
                AITER_JIT_DIR=str(root / "aiter_jit"),
                TRITON_CACHE_DIR=str(root / "triton_cache"),
                AITER_ONLINE_TUNE="0",
                AITER_BYPASS_TUNE_CONFIG="0",
                AITER_AOT_IMPORT="0",
                AITER_TRITON_ONLY="0",
                AITER_USE_CK_MOE_SORTING="1",
                AITER_USE_FLYDSL_MOE_SORTING="0",
                AITER_MOE_SORT_BACKEND="auto",
                AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED="0",
                AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED="0",
            )
            if not empty_cache:
                compile_env = {
                    **env,
                    "AITER_AOT_IMPORT": "1",
                    "COMPILE_ONLY": "1",
                    "FLYDSL_RUNTIME_RUN_ONLY": "0",
                }
                output = _run_worker(
                    "compile", config_file, compile_env, limit=compile_limit
                )
                self.assertEqual(output.count(_COMPILE_MARKER), 1, output)
                self.assertTrue(_cache_snapshot(cache), output)
            before = _cache_snapshot(cache)
            output = _run_worker(
                "run",
                config_file,
                env,
                limits=list(limits),
                expect_miss=expect_miss,
                zero_input=zero_input,
            )
            self.assertEqual(_cache_snapshot(cache), before)
            if expect_miss:
                self.assertEqual(output.count(_MISS_MARKER), 1, output)
                self.assertNotIn(_RESULT_MARKER, output)
            else:
                records = [
                    json.loads(line[len(_RESULT_MARKER) :])
                    for line in output.splitlines()
                    if line.startswith(_RESULT_MARKER)
                ]
                self.assertEqual(len(records), 3 * len(limits), output)
                self.assertEqual(
                    [record["device"] for record in records],
                    [device for device in (0, 1, 0) for _ in limits],
                )
                for record in records:
                    self.assertNotEqual(record["device"], record["ambient_device"])
                    self.assertLessEqual(record["logits_diff"], 0.01)
                    self.assertLessEqual(record["err_ratio"], 0.05)
            print(output, flush=True)

    def test_small_batch1_ptpc_silu(self):
        self._exercise(
            _whole_graph_row(kernelName1=_PREFIX + _DECODE, block_m=16, token=1)
        )

    def test_small_sorted_decode_ptpc_silu(self):
        self._exercise(
            _whole_graph_row(kernelName1=_PREFIX + _DECODE, block_m=16, token=2)
        )

    def test_small_prefill_ptpc_silu(self):
        self._exercise(_whole_graph_row())

    def test_prefill_preserves_odd_256_column_multiples(self):
        self._exercise(_whole_graph_row(model_dim=768))

    def test_small_batch1_per_tensor_swiglu_legacy_defaults(self):
        self._exercise(
            _whole_graph_row(
                kernelName1=_PREFIX + _DECODE,
                block_m=16,
                token=1,
                q_type="QuantType.per_Tensor",
                act_type="ActivationType.Swiglu",
            ),
            limits=(None, 0.0, 7.0),
        )

    def test_small_sorted_decode_per_tensor_swiglu(self):
        self._exercise(
            _whole_graph_row(
                kernelName1=_PREFIX + _DECODE,
                block_m=16,
                token=16,
                q_type="QuantType.per_Tensor",
                inter_dim=192,
                act_type="ActivationType.Swiglu",
            )
        )

    def test_small_prefill_per_tensor_silu(self):
        self._exercise(_whole_graph_row(q_type="QuantType.per_Tensor", inter_dim=192))

    def test_small_prefill_ptpc_swiglu_nondefault_limit(self):
        self._exercise(
            _whole_graph_row(act_type="ActivationType.Swiglu"),
            compile_limit=1.5,
            limits=(1.5,),
        )

    def test_small_prefill_per_tensor_swiglu_nondefault_limit(self):
        self._exercise(
            _whole_graph_row(
                q_type="QuantType.per_Tensor",
                act_type="ActivationType.Swiglu",
                inter_dim=192,
            ),
            compile_limit=1.5,
            limits=(1.5,),
        )

    def test_per_tensor_zero_input_stays_finite(self):
        self._exercise(_whole_graph_row(q_type="QuantType.per_Tensor"), zero_input=True)

    def test_qwen_batch1_ptpc_silu(self):
        self._exercise(self._model_row("qwen", batch=1, inter_dim=128, experts=257))

    def test_qwen_prefill_ptpc_silu(self):
        self._exercise(self._model_row("qwen", batch=1024, inter_dim=128, experts=257))

    def test_hunyuan_sorted_decode_per_tensor_silu(self):
        self._exercise(self._model_row("hunyuan", batch=16, inter_dim=192, experts=192))

    def test_hunyuan_prefill_per_tensor_silu(self):
        self._exercise(
            self._model_row("hunyuan", batch=512, inter_dim=192, experts=192)
        )

    def test_minimax_batch1_ptpc_swiglu(self):
        self._exercise(self._model_row("minimax", batch=1, inter_dim=384, experts=128))

    def test_minimax_prefill_ptpc_swiglu_nondefault_limit(self):
        self._exercise(
            self._model_row("minimax", batch=1024, inter_dim=384, experts=128),
            compile_limit=1.5,
            limits=(1.5,),
        )

    def test_run_only_refuses_an_empty_flydsl_cache(self):
        self._exercise(
            _whole_graph_row(kernelName1=_PREFIX + _DECODE, block_m=16, token=1),
            expect_miss=True,
            empty_cache=True,
        )

    def test_swiglu_nondefault_limit_requires_its_own_aot_specialization(self):
        self._exercise(
            _whole_graph_row(
                kernelName1=_PREFIX + _DECODE,
                block_m=16,
                token=1,
                act_type="ActivationType.Swiglu",
            ),
            compile_limit=None,
            limits=(1.5,),
            expect_miss=True,
        )


if __name__ == "__main__":
    unittest.main()
