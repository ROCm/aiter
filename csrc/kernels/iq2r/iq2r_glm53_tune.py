# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune GLM-5.3 IQ2R MoE launch choices per token count.

Reads shapes from ``aiter/configs/iq2r_glm53_untuned.csv``, checks and times
every candidate (gate kernel and grid, down kernel and grid, prefill down
chunks) on synthetic packed weights on the current GPU, and writes the
fastest to the tuned CSV read by :func:`aiter.iq2r_glm53.iq2r_glm53_config`.

    python3 csrc/kernels/iq2r/iq2r_glm53_tune.py               # untuned shapes
    python3 csrc/kernels/iq2r/iq2r_glm53_tune.py --all         # retune them all
    python3 csrc/kernels/iq2r/iq2r_glm53_tune.py --run_config  # time the CSV

Candidates are timed by CUDA graph replay, as serving runs decode batches.
Each call in the graph reads its own copy of the layer with its own routing,
and one replay reads at least ``ROTATE_BYTES`` of expert weights, so every
call starts with its experts out of the 256 MB MALL, as in serving, where the
other layers run between two calls to one layer.
"""

from __future__ import annotations

import functools
import itertools
import os
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
import torch

import aiter.iq2r_glm53 as glm53
from aiter.iq2r_glm53 import (
    EXPERTS,
    HIDDEN,
    MAX_DECODE_TOKENS,
    TOPK,
    IQ2RGlm53Config,
    IQ2RGlm53Workspace,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack,
)
from aiter.jit import core
from aiter.ops.iq2r import iq2r_encode_device, iq2r_materialize_device
from aiter.ops.iq2r_encoder import iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2RMetadata
from aiter.utility.base_tuner import TunerCommon

CONFIG_ENV = "AITER_CONFIG_IQ2R_GLM53"
KEYS = ["gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk"]
RESULTS = ["gate_kernel", "gate_grid", "down_kernel", "down_grid", "down_chunks", "us"]
ROTATE_BYTES = 1 << 30  # expert weights read per graph replay: 4x the MALL


def synthetic_weights(intermediate_size: int, device: str):
    """Three encoded random experts tiled over all 257, with their dense weights.

    Returns ``(packed, gate, down, pick)``: expert ``e`` uses ``gate[pick[e]]``.
    """
    codebook = iq2r_initial_codebook(device)

    def matrix(n: int, k: int):
        records = [
            iq2r_encode_device(
                torch.randn((n, k), device=device) * 0.02,
                torch.ones((k,), device=device),
                codebook,
                exponent_radius=8,
            )
            for _ in range(3)
        ]
        dense = [
            iq2r_materialize_device(d.view(1, -1), a.view(1, -1), IQ2RMetadata(n, k))
            for d, a in records
        ]
        data, aux = (torch.stack(parts)[pick] for parts in zip(*records))
        return data, aux, dense

    pick = torch.arange(EXPERTS, device=device) % 3
    gate, gate_aux, gate_dense = matrix(2 * intermediate_size, HIDDEN)
    down, down_aux, down_dense = matrix(HIDDEN, intermediate_size)
    packed = iq2r_glm53_pack(
        gate, gate_aux, down, down_aux, intermediate_size=intermediate_size
    )
    return packed, gate_dense, down_dense, pick


def routing(tokens: int, device: str):
    """Random top-8 routes plus the shared expert 256.

    Every route weighs at least 0.5, so a dropped route fails the check.
    """
    routed = torch.rand((tokens, EXPERTS - 1), device=device).topk(TOPK - 1).indices
    shared = torch.full((tokens, 1), EXPERTS - 1, dtype=routed.dtype, device=device)
    ids = torch.cat((routed, shared), 1).to(torch.int32).contiguous()
    weights = 0.5 + 0.5 * torch.rand((tokens, TOPK), device=device)
    weights[:, -1] = 1.0  # shared expert
    return weights, ids


def mxfp8(values: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize to MXFP8 E4M3 with one E8M0 scale per 32 values."""
    blocks = values.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127))
    scale = torch.exp2(exponent)
    quantized = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
    return quantized.flatten(-2)


def dense_reference(hidden, topk_weights, topk_ids, gate, down, pick):
    """MXFP8 activations, SiLU(gate) * up in BF16, BF16 route outputs (FP32 sum)."""
    a1 = mxfp8(hidden)
    output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    route_expert = pick[topk_ids.long()]
    for expert, (gate_weight, down_weight) in enumerate(zip(gate, down)):
        weight = (topk_weights * (route_expert == expert)).sum(dim=-1, keepdim=True)
        gate_up = a1 @ gate_weight.T
        act = (torch.nn.functional.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]).to(
            torch.bfloat16
        )
        route = (mxfp8(act) @ down_weight.T).to(torch.bfloat16)
        output += weight * route.float()
    return output


def dense_mismatch(out: torch.Tensor, expected: torch.Tensor) -> str | None:
    """Why ``out`` fails the op_test's bounds against the dense reference, or None.

    The intermediate is requantized to FP8, so the bounds allow for neighbouring
    FP8 values; the element-wise check against other candidates is the tight one.
    """
    if not torch.isfinite(out).all():
        return "non-finite output"
    out = out.float()
    rmse = ((out - expected).square().mean() / expected.square().mean()).sqrt().item()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten(), expected.flatten(), dim=0
    ).item()
    if rmse < 3e-2 and cosine > 0.9995:
        return None
    return f"relative RMSE {rmse:.3g}, cosine {cosine:.5f} against the dense reference"


def poison(output, workspace) -> None:
    """NaN-fill every buffer a launch writes, so a candidate that skips part of
    one cannot pass with an earlier candidate's values."""
    output.fill_(float("nan"))
    workspace.route_output.fill_(float("nan"))
    for values, scales in (
        (workspace.route_input_fp8, workspace.route_input_scales),
        (workspace.intermediate_fp8, workspace.intermediate_scales),
    ):
        values.view(torch.uint8).fill_(0x7F)  # E4M3 NaN
        scales.fill_(255)  # E8M0 NaN


def candidates(tokens: int, intermediate_size: int):
    tp8 = intermediate_size == 256
    if tokens == 1:
        yield IQ2RGlm53Config("decode", 2, "route9", 4, 1)
        return
    if tokens <= MAX_DECODE_TOKENS:
        gates = ("decode", "nobarrier")
        downs = [("route9", 4)] if tokens in (2, 4) else []
        down_kernels = ("packed", "ordered", "single") if tp8 else ("packed", "single")
        downs += list(itertools.product(down_kernels, (2, 4, 8)))
        for gate, grid in itertools.product(gates, (1, 2, 4)):
            for down, down_grid in downs:
                yield IQ2RGlm53Config(gate, grid, down, down_grid, 1)
    if tokens > 16:
        for grid, chunks in itertools.product((1, 2, 4), (1, 2)):
            yield IQ2RGlm53Config("prefill", grid, "prefill", 0, chunks)


def time_us(calls, warmup: int, iters: int) -> float:
    """Time per call of a CUDA graph that runs ``calls`` in turn.

    The graph is replayed until at least ``iters`` calls have run; serving
    captures decode batches, so host launch overhead (several launches on the
    prefill path) is excluded.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(-(-warmup // len(calls))):
        graph.replay()
    replays = -(-iters // len(calls))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * len(calls))


class IQ2RGlm53Tuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{core.AITER_ROOT_DIR}/aiter/configs/iq2r_glm53_untuned.csv",
        "tune_file": f"{core.AITER_CONFIG_IQ2R_GLM53}",
        "config_env_name": CONFIG_ENV,
        "warmup": 50,  # calls
        "iters": 500,  # calls
    }

    def __init__(self):
        super().__init__("iq2r_glm53_tuned", KEYS, RESULTS, __doc__.splitlines()[0])
        self._intermediate_size = None
        self._layers = []  # copies of one synthetic packed layer
        self._dense = None  # (gate, down, pick) of the layer
        self._weight_bytes = {}  # (tokens, intermediate) -> expert bytes per call

    def _setup_specific_arguments(self):
        pass

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            shapes = self.get_untuned_gemm_list(args.untune_file)
            shapes = shapes[
                (shapes.gfx == self.get_gfx()) & (shapes.cu_num == self.get_cu_num())
            ]
            self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
            tuned = set(self.tunedf[self.keys].itertuples(index=False, name=None))
            new = [
                key not in tuned
                for key in shapes[self.keys].itertuples(index=False, name=None)
            ]
            self.untunedf = shapes.loc[new, self.keys].reset_index(drop=True)
        unsupported = (
            (self.untunedf.model_dim != HIDDEN)
            | (self.untunedf.expert != EXPERTS)
            | (self.untunedf.topk != TOPK)
            | ~self.untunedf.inter_dim.isin(glm53.INTERMEDIATE_SIZES)
        )
        if unsupported.any():
            raise ValueError(
                "the GLM-5.3 tuner covers hidden 6144, 257 experts, top-9 and "
                f"inter_dim {glm53.INTERMEDIATE_SIZES}"
            )

    def _layer(self, intermediate_size: int, copies: int = 1):
        """The first ``copies`` copies of the synthetic layer for one width."""
        if intermediate_size != self._intermediate_size:
            self._layers, self._dense = [], None
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            packed, gate, down, pick = synthetic_weights(intermediate_size, "cuda")
            self._intermediate_size = intermediate_size
            self._layers, self._dense = [packed], (gate, down, pick)
        while len(self._layers) < copies:
            self._layers.append(tuple(t.clone() for t in self._layers[0]))
        return self._layers[:copies]

    def _shape(self, tokens: int, intermediate_size: int):
        """Inputs, dense reference and rotated calls for one shape."""
        (packed,) = self._layer(intermediate_size)
        hidden = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        routes = [routing(tokens, "cuda")]
        expert_bytes = sum(t.nbytes for t in packed) // EXPERTS
        read = [routes[0][1].unique().numel() * expert_bytes]
        while sum(read) < ROTATE_BYTES:
            routes.append(routing(tokens, "cuda"))
            read.append(routes[-1][1].unique().numel() * expert_bytes)
        self._weight_bytes[(tokens, intermediate_size)] = sum(read) / len(read)
        layers = self._layer(intermediate_size, len(routes))
        weights, ids = routes[0]
        return SimpleNamespace(
            hidden=hidden,
            weights=weights,
            ids=ids,
            packed=packed,
            output=torch.empty_like(hidden),
            workspace=IQ2RGlm53Workspace.allocate(
                tokens, intermediate_size, device="cuda"
            ),
            expected=dense_reference(hidden, weights, ids, *self._dense),
            calls=[(layer, *route) for layer, route in zip(layers, routes)],
        )

    @staticmethod
    def _run(shape, config, layer, weights, ids):
        iq2r_glm53_moe_out(
            shape.hidden, *layer, weights, ids, shape.output, shape.workspace, config
        )

    def _time(self, shape, config, args) -> float:
        calls = [functools.partial(self._run, shape, config, *c) for c in shape.calls]
        return time_us(calls, args.warmup, args.iters)

    def tune(self, untunedf, tunedf, args):
        results = []
        for key in untunedf[self.keys].itertuples(index=False, name=None):
            tokens, intermediate_size = int(key[2]), int(key[4])
            shape = self._shape(tokens, intermediate_size)
            label = f"M={tokens} I={intermediate_size}"
            # Every candidate must match the dense reference, and match
            # element-wise the first candidate that did (down variants may sum
            # the nine routes in a different order).
            reference = None
            tolerance = 0.0
            for config in candidates(tokens, intermediate_size):
                info = (key, *(getattr(config, f) for f in RESULTS[:-1]))
                poison(shape.output, shape.workspace)
                try:
                    self._run(shape, config, shape.packed, shape.weights, shape.ids)
                    torch.cuda.synchronize()
                    problem = dense_mismatch(shape.output, shape.expected)
                except (RuntimeError, ValueError) as error:
                    problem = str(error)
                if (
                    problem is None
                    and reference is not None
                    and not torch.allclose(
                        shape.output, reference, rtol=1e-2, atol=tolerance
                    )
                ):
                    problem = "differs from the first matching candidate"
                if problem is not None:
                    print(f"{label} {self.getKernelName(config)}: skipped, {problem}")
                    results.append((info, self.INVALID_TIME, self.INVLAID_ERR_RATIO))
                    continue
                if reference is None:
                    reference = shape.output.clone()
                    tolerance = 1e-2 * reference.float().abs().max().item()
                us = self._time(shape, config, args)
                tflops, tbps = self.calculate((info, us, 0.0))
                print(
                    f"{label} {self.getKernelName(config)}: {us:.2f} us, "
                    f"{tflops:.1f} TFLOPS, {tbps:.2f} TB/s"
                )
                results.append((info, us, 0.0))
            del shape
        return results

    def getKernelName(self, kernel_id):
        """Readable name of an :class:`IQ2RGlm53Config`."""
        config = kernel_id
        if config.gate_kernel == "prefill":
            return f"prefill gate/{config.gate_grid}, {config.down_chunks} chunk(s)"
        return (
            f"{config.gate_kernel} gate/{config.gate_grid}, "
            f"{config.down_kernel} down/{config.down_grid}"
        )

    def calculate(self, results, inbpe=2, outbpe=2):
        """TFLOPS and TB/s; bytes are the experts' weights each call reads
        (averaged over the rotation) plus the BF16 input and output."""
        info, us, _ = results
        tokens, intermediate_size = int(info[0][2]), int(info[0][4])
        if us <= 0:
            return 0.0, 0.0
        flops = 6 * tokens * TOPK * HIDDEN * intermediate_size
        nbytes = self._weight_bytes.get((tokens, intermediate_size), 0) + (
            tokens * HIDDEN * (inbpe + outbpe)
        )
        return flops / us / 1e6, nbytes / us / 1e6

    def result_to_df(self, rets):
        rows = [(*key, *choice, round(us, 2)) for (key, *choice), us, _ in rets]
        return pd.DataFrame(rows, columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        failed = results.us == self.INVALID_TIME
        self.failed = pd.concat([self.failed, results[failed]], ignore_index=True)
        self.success = pd.concat([self.success, results[~failed]], ignore_index=True)
        tuned = self.update_tunedf(self.get_tuned_gemm_list(file), results[~failed])
        tuned.to_csv(file, index=False)

    def _run_lookup(self, tokens: int, intermediate_size: int, args):
        """Check and time the launch choice the CSV lookup makes."""
        try:
            shape = self._shape(tokens, intermediate_size)
            poison(shape.output, shape.workspace)
            self._run(shape, None, shape.packed, shape.weights, shape.ids)
            problem = dense_mismatch(shape.output, shape.expected)
            us = self._time(shape, None, args)
        except (RuntimeError, ValueError) as error:
            return self.INVALID_TIME, f"error: {error}"
        return us, "ok" if problem is None else f"mismatch: {problem}"

    def run_config(self, args):
        results = []
        for key in self.untunedf[self.keys].itertuples(index=False, name=None):
            tokens, intermediate_size = int(key[2]), int(key[4])
            label = f"M={tokens} I={intermediate_size}"
            us, status = self._run_lookup(tokens, intermediate_size, args)
            config = glm53.iq2r_glm53_config(tokens, intermediate_size)
            print(f"{label} {self.getKernelName(config)}: {us:.2f} us, {status}")
            results.append({"shape": label, "e2e_us": us, "status": status})
        return results

    def _clear_op_caches(self):
        core.AITER_CONFIGS.get_config_file.cache_clear()
        glm53.iq2r_glm53_config.cache_clear()
        glm53._TUNED.clear()

    def _set_config_env_for_run_config(self, args, config_file=None):
        """Point the lookup at ``config_file``. The CSV is read in Python, so
        unlike the base class this rebuilds no JIT module."""
        old_val = os.environ.get(CONFIG_ENV)
        os.environ[CONFIG_ENV] = config_file or self.get_out_file(args.tune_file)
        self._clear_op_caches()
        return old_val, core.AITER_REBUILD

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()


if __name__ == "__main__":
    tuner = IQ2RGlm53Tuner()
    tuner.run(tuner.parse_args())
