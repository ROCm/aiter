# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune native MXFP8 / E8M0 block128 GEMM, including native B preshuffle.

python csrc/gemm_mxfp8/gemm_mxfp8_tune.py -i shapes.csv -o tuned.csv
python csrc/gemm_mxfp8/gemm_mxfp8_tune.py -i shapes.csv -o tuned.csv --run_config
"""

from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIG_GEMM_MXFP8
from aiter.ops.flydsl.gemm_mxfp8 import (
    CONFIG_KEYS,
    flydsl_mxfp8_gemm,
    flydsl_mxfp8_kernel_name,
    get_flydsl_mxfp8_configs,
    get_flydsl_mxfp8_kernel_params,
)
from aiter.ops.gemm_op_mxfp8 import MXFP8_KEYS
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner


def output_dtype(name):
    return {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[str(name)]


def generate_data(
    m, n, k, dtype, bias, scale_block, bpreshuffle, scale_a_transposed, device="cuda"
):
    torch.manual_seed(0)
    a = torch.empty(m, k, device=device).uniform_(-1, 1).to(torch.float8_e4m3fn)
    b = torch.empty(n, k, device=device).uniform_(-1, 1).to(torch.float8_e4m3fn)
    sa = torch.randint(
        124, 129, (m, k // scale_block), device=device, dtype=torch.uint8
    )
    sb = torch.randint(
        124,
        129,
        (n if scale_block == 32 else (n + 127) // 128, k // scale_block),
        device=device,
        dtype=torch.uint8,
    )
    if scale_a_transposed:
        sa = sa.t().contiguous().t()
    return {
        "a": a,
        "b": shuffle_weight(b) if bpreshuffle else b,
        "original_b": b,
        "sa": sa,
        "sb": sb,
        "bias": torch.randn(n, device=device, dtype=dtype) if bias else None,
        "out": torch.randn(m, n, device=device, dtype=dtype),
    }


def reference(a, b, sa, sb, bias, dtype, scale_block):
    x = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(scale_block, 1)
    s = torch.exp2(sb.float() - 127)
    if scale_block == 128:
        s = s.repeat_interleave(128, 0)[: b.shape[0]]
    w = b.float() * s.repeat_interleave(scale_block, 1)
    y = x @ w.t()
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype)


def run_kernel(
    a, b, sa, sb, out, bias, config, scale_block, bpreshuffle, scale_a_transposed
):
    return flydsl_mxfp8_gemm(
        a,
        b,
        sa,
        sb,
        out=out,
        bias=bias,
        config=config,
        scale_block=scale_block,
        bpreshuffle=bpreshuffle,
        scale_a_transposed=scale_a_transposed,
    )


def make_tasks(row, run_kwargs):
    m, n, k = (int(row[x]) for x in ("M", "N", "K"))
    dtype = output_dtype(row["outdtype"])
    bias, block, bp, sat = (
        row[x] for x in ("bias", "scale_block", "bpreshuffle", "scale_a_transposed")
    )
    configs = get_flydsl_mxfp8_configs(m, n, k, dtype, bias, block, bp, sat)
    tasks = []
    for kid, config in enumerate(configs):
        name = flydsl_mxfp8_kernel_name(
            config,
            out_dtype=dtype,
            has_bias=bias,
            scale_block=block,
            bpreshuffle=bp,
            scale_a_transposed=sat,
        )
        tasks.append(
            (
                (tuple(row[x] for x in MXFP8_KEYS), kid, config["split_k"], name),
                generate_data,
                (m, n, k, dtype, bias, block, bp, sat),
                run_kernel,
                (["a", "b", "sa", "sb", "out", "bias"], config, block, bp, sat),
                dict(run_kwargs),
                reference,
                (["a", "original_b", "sa", "sb", "bias"], dtype, block),
                {},
                None,
                0.03,
                0.1,
                None,
                None,
                ("out",),
            )
        )
    return tasks


def screen_tasks(tasks, topk):
    """Rank the full space with graph events; mp_tuner remeasures finalists."""
    _, gen_data, gen_args, _, _, _, ref_fn, ref_args, _, *_ = tasks[0]
    data = gen_data(*gen_args)
    ref = ref_fn(*(data[k] for k in ref_args[0]), *ref_args[1:])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ranked = []
    with torch.cuda.stream(stream):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        for task in tasks:
            _info, _, _, fn, args, *_ = task
            inputs = tuple(data[k] for k in args[0]) + tuple(args[1:])
            data["out"].fill_(float("nan"))
            # Compile/load and initialize split-K buffers before capture.
            y = fn(*inputs)
            if not torch.allclose(y, ref, rtol=0.03, atol=0.1):
                continue
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(4):
                    fn(*inputs)
            graph.replay()
            start.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            if torch.allclose(y, ref, rtol=0.03, atol=0.1):
                ranked.append((start.elapsed_time(end), task))
    # Reserve finalists per split/slice regime so a hot-cache screen does not
    # eliminate an entire reduction family before rotating-buffer profiling.
    selected = {}
    for _, task in sorted(ranked, key=lambda item: item[0]):
        c = task[4][1]
        regime = (c["split_k"] > 1, c["k_waves"] > 1)
        bucket = selected.setdefault(regime, [])
        if len(bucket) < topk:
            bucket.append(task)
    finalists = [task for bucket in selected.values() for task in bucket]
    logger.info(
        "MXFP8 screen %s: %d candidates -> %d finalists",
        tasks[0][0][0][2:5],
        len(tasks),
        len(finalists),
    )
    if not finalists:
        raise ValueError("No MXFP8 candidate passed screening")
    return finalists


def check_splitk_stability(results):
    """Reject order-sensitive BF16 atomic reductions, outside timing."""
    checked = []
    data_key, data, ref = None, None, None
    for info, us, error in results:
        keys, _kid, split_k, name = info
        if us > 0 and error == 0 and split_k > 1 and keys[5] == "torch.bfloat16":
            if data_key != keys:
                _gfx, _cu, m, n, k, dtype, bias, block, bp, sat = keys
                data = generate_data(m, n, k, output_dtype(dtype), bias, block, bp, sat)
                ref = reference(
                    data["a"],
                    data["original_b"],
                    data["sa"],
                    data["sb"],
                    data["bias"],
                    output_dtype(dtype),
                    block,
                )
                data_key = keys
            params = get_flydsl_mxfp8_kernel_params(name, split_k)
            config = {key: params[key] for key in CONFIG_KEYS}
            for _ in range(16):
                data["out"].fill_(float("nan"))
                y = run_kernel(
                    data["a"],
                    data["b"],
                    data["sa"],
                    data["sb"],
                    data["out"],
                    data["bias"],
                    config,
                    block,
                    bp,
                    sat,
                )
                if not torch.allclose(y, ref, rtol=0.03, atol=0.1):
                    error = 1.0
                    break
        checked.append((info, us, error))
    return checked


class GemmMXFP8Tuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_GEMM_MXFP8,
        "untune_file": "aiter/configs/mxfp8_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_MXFP8",
        "errRatio": 0.0,
    }

    def __init__(self):
        super().__init__(
            "GemmMXFP8Tuner",
            MXFP8_KEYS,
            [
                "libtype",
                "kernelId",
                "splitK",
                "us",
                "kernelName",
                "errRatio",
                "tflops",
                "bw",
            ],
            "gfx950 FlyDSL MXFP8 GEMM tuner (native or preshuffled weights)",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--screen-topk",
            type=int,
            default=0,
            help="Graph-screen the full space; profile this many finalists per "
            "split/slice regime (0 profiles every candidate).",
        )

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_mxfp8 import _load_mxfp8_configs, get_mxfp8_config

        _load_mxfp8_configs.cache_clear()
        get_mxfp8_config.cache_clear()

    def get_untuned_gemm_list(self, path):
        df = pd.read_csv(path)
        defaults = {
            "outdtype": "torch.bfloat16",
            "bias": False,
            "scale_block": 32,
            "bpreshuffle": False,
            "scale_a_transposed": False,
        }
        for key, value in defaults.items():
            if key not in df:
                df[key] = value
        for _, row in df.iterrows():
            output_dtype(row["outdtype"])
            if row["scale_block"] not in (32, 128):
                raise ValueError("scale_block must be 32 or 128")
            if row["scale_a_transposed"] and row["scale_block"] == 32:
                raise ValueError("1x32 scales must be row-major")
        return df.drop_duplicates().reset_index(drop=True)

    def tune(self, untunedf, tunedf, args):
        tasks, groups = [], []
        for _, row in untunedf.iterrows():
            shape_tasks = make_tasks(
                row, {"num_warmup": args.warmup, "num_iters": args.iters}
            )
            if not shape_tasks:
                raise ValueError(f"No MXFP8 kernels support {row.to_dict()}")
            if args.screen_topk:
                shape_tasks = screen_tasks(shape_tasks, args.screen_topk)
            tasks.extend(shape_tasks)
            groups.append((len(shape_tasks), ()))
        results = mp_tuner(
            tasks,
            groups,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        return check_splitk_stability(results)

    def calculate(self, results, bpes=(1, 1, 2)):
        dtype = output_dtype(results[0][0][5])
        return super().calculate(
            results, bpes=(1, 1, 4 if dtype == torch.float32 else 2)
        )

    def result_to_df(self, results):
        df = super().result_to_df(results)
        df["libtype"] = "flydsl"
        return df[self.columns]

    def run_config(self, args):
        from aiter.ops.gemm_op_mxfp8 import gemm_mxfp8
        from aiter.test_common import run_perftest

        results = []
        for _, row in self.untunedf.iterrows():
            m, n, k = (int(row[x]) for x in ("M", "N", "K"))
            dtype = output_dtype(row["outdtype"])
            bias, block, bp, sat = (
                row[x]
                for x in ("bias", "scale_block", "bpreshuffle", "scale_a_transposed")
            )
            d = generate_data(m, n, k, dtype, bias, block, bp, sat)
            y, us = run_perftest(
                gemm_mxfp8,
                d["a"],
                d["b"],
                d["sa"],
                d["sb"],
                out=d["out"],
                bias=d["bias"],
                dtype=dtype,
                scale_block=block,
                bpreshuffle=bp,
                scale_a_transposed=sat,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            ref = reference(
                d["a"], d["original_b"], d["sa"], d["sb"], d["bias"], dtype, block
            )
            torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)
            results.append(
                {
                    "shape": f"{m},{n},{k},sb={block},bp={bp}",
                    "e2e_us": us,
                    "status": "ok",
                }
            )
        return results


if __name__ == "__main__":
    tuner = GemmMXFP8Tuner()
    tuner.run(tuner.parse_args(), False)
