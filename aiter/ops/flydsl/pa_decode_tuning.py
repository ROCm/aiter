# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Native FlyDSL autotuning and cached work-plan preparation for PA decode.

Run with: python -m aiter.ops.flydsl.pa_decode_tuning --help
"""

import argparse
import os
from contextlib import contextmanager
from copy import copy
from functools import cache
from hashlib import sha256
from pathlib import Path

import torch
from flydsl.autotune import Config, autotune

from aiter.ops.flydsl.kernels.pa_decode import implementation_cache_tag
from aiter.ops.flydsl.kernels.pa_decode_plan import PADecodePlan, plan_pa_decode
from aiter.ops.flydsl.pa_decode import pa_decode

_PA_DECODE_CONFIGS_PATH = Path(__file__).parent / "configs" / "pa"


@cache
def _tuning_cache_tag():
    """Invalidate tuning results when either kernels or tuning policy changes."""
    digest = sha256(implementation_cache_tag().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


class _PADecodeAutotuneResources:
    """Prepare candidate plans and graphs outside FlyDSL's timed replay."""

    def __init__(self, query, args, options, max_partitions):
        self.query = query
        self.args = args
        self.options = options
        self.max_partitions = max_partitions
        self.outputs = {}
        self.graphs = {}
        self.plans = {}
        self.workspaces = {}
        self.reference = None

    def prepare(self, workgroup_budget):
        if workgroup_budget in self.graphs:
            return
        key, value, lengths, table, scale, query_length = self.args
        if key.dtype != torch.bfloat16:
            if (self.options["key_scale"] is None) != (
                self.options["value_scale"] is None
            ):
                raise ValueError(
                    "key_scale and value_scale must either both be provided or both be None"
                )
            for name in ("key_scale", "value_scale"):
                kv_scale = self.options[name]
                if not isinstance(kv_scale, torch.Tensor):
                    # Scalar host-to-device copies must precede graph capture.
                    self.options[name] = torch.tensor(
                        [1.0 if kv_scale is None else float(kv_scale)],
                        dtype=torch.float32,
                        device=self.query.device,
                    )
        plan = plan_pa_decode(
            lengths,
            key.shape[1],
            max_partitions=self.max_partitions,
            workgroup_budget=workgroup_budget,
            sliding_window=self.options["sliding_window"],
            query_length=query_length,
        )
        rows = query_length * self.query.shape[1] // key.shape[1]
        scalar_shape = (key.shape[1], plan.capacity, rows)
        output = torch.full_like(
            self.query, float("nan"), memory_format=torch.contiguous_format
        )
        workspace = {
            "exp_sums": torch.empty(
                scalar_shape, dtype=torch.float32, device=output.device
            ),
            "max_logits": torch.empty(
                scalar_shape, dtype=torch.float32, device=output.device
            ),
            "temporary_output": torch.empty(
                (*scalar_shape, self.query.shape[-1]),
                dtype=output.dtype,
                device=output.device,
            ),
        }

        def launch():
            pa_decode(
                output,
                self.query,
                key,
                value,
                lengths,
                table,
                scale,
                query_length,
                work_plan=plan,
                **self.options,
                **workspace,
            )

        # Eager compilation and graph capture precede native autotune timing.
        stream = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream(device=output.device)
        capture_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            launch()
            with torch.cuda.graph(graph, stream=capture_stream):
                launch()
        stream.wait_stream(capture_stream)
        graph.replay()
        stream.synchronize()
        if not torch.isfinite(output).all().item():
            raise ArithmeticError("PA autotune candidate produced nonfinite output")
        self.outputs[workgroup_budget] = output
        self.graphs[workgroup_budget] = graph
        # Graph replay uses the original pointers, so retain all backing tensors.
        self.plans[workgroup_budget] = plan
        self.workspaces[workgroup_budget] = workspace
        if self.reference is None:
            self.reference = output.clone()
        else:
            torch.testing.assert_close(output, self.reference, atol=5e-3, rtol=5e-3)


def _pa_decode_configs(*args, resources, num_cu, **kwargs):
    # Keep a validated 2*CU baseline; deduplicate identical plan capacities.
    baseline = 2 * num_cu
    resources.prepare(baseline)
    batch = resources.args[2].numel()
    kv_heads = resources.args[0].shape[1]
    capacities = {}
    for budget in [baseline, 128, 256, 512, 1024, 2048, 4096]:
        capacity = min(
            batch * resources.max_partitions,
            max(batch, (budget + kv_heads - 1) // kv_heads),
        )
        capacities.setdefault(capacity, Config(workgroup_budget=budget))
    return list(capacities.values())


def _pa_decode_default(*args, num_cu, **kwargs):
    return Config(workgroup_budget=2 * num_cu)


@contextmanager
def _validate_pa_decode_config(arguments):
    resources = arguments["resources"]
    budget = arguments["workgroup_budget"]
    resources.prepare(budget)
    resources.outputs[budget].fill_(float("nan"))
    yield
    torch.cuda.current_stream().synchronize()
    torch.testing.assert_close(
        resources.outputs[budget], resources.reference, atol=5e-3, rtol=5e-3
    )


@autotune(
    configs=_pa_decode_configs,
    key=[
        "num_seqs",
        "num_kv_heads",
        "query_group_size",
        "head_dim",
        "block_size",
        "query_length",
        "query_dtype",
        "kv_dtype",
        "per_token_kv",
        "trans_v",
        "sink_dtype",
        "sliding_window",
        "max_partitions",
        "num_cu",
        "implementation",
    ],
    default=_pa_decode_default,
    validate_hook=_validate_pa_decode_config,
)
def _pa_decode_autotuner(
    num_seqs,
    num_kv_heads,
    query_group_size,
    head_dim,
    block_size,
    query_length,
    query_dtype,
    kv_dtype,
    per_token_kv,
    trans_v,
    sink_dtype,
    sliding_window,
    max_partitions,
    num_cu,
    implementation,
    resources,
    *,
    workgroup_budget,
):
    resources.graphs[workgroup_budget].replay()


@cache
def _get_pa_decode_autotuner(arch, config_dir=None):
    """Keep native PA tuning results in the target architecture's directory."""
    if config_dir is None and "FLYDSL_AUTOTUNE_CACHE_DIR" in os.environ:
        return _pa_decode_autotuner
    config_root = _PA_DECODE_CONFIGS_PATH if config_dir is None else config_dir
    tuner = copy(_pa_decode_autotuner)
    tuner._cache_file = config_root / arch / "_pa_decode_autotuner.json"
    # Each architecture owns its native in-memory and disk caches. Discard the
    # template's default-directory entries before loading the selected file.
    tuner.cache = {}
    tuner._artifact_cache = {}
    tuner._load_disk_cache()
    return tuner


def prepare_pa_decode_plan(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_length: int,
    *,
    compute_type: torch.dtype = torch.bfloat16,
    key_scale: torch.Tensor | float | None = None,
    value_scale: torch.Tensor | float | None = None,
    sinks: torch.Tensor | None = None,
    sliding_window: int = 0,
    max_partitions: int | None = None,
    max_context_length: int | None = None,
    tune: bool = False,
    config_dir: str | Path | None = None,
) -> PADecodePlan:
    """Select a work budget using native FlyDSL autotune, then build a plan.

    Call outside graph capture with the actual decode tensors/options. Native
    cache hits and misses with the 2*CU default do not launch candidates.
    Set FLYDSL_AUTOTUNE=1 to prepare candidate workspaces/graphs and search with
    FlyDSL's default benchmark and fastest-config selection. Candidates must
    agree with the 2*CU baseline before timing.
    Results are saved to configs/pa/<arch>/_pa_decode_autotuner.json under the
    FlyDSL operator package. Set FLYDSL_AUTOTUNE_CACHE_DIR before importing this
    module to override the native cache directory.

    ``tune=True`` searches even when a result is cached and saves the winner.
    ``config_dir`` selects a root containing per-architecture cache directories
    and takes precedence over the environment's cache-directory override.

    Cache by attention geometry and execution modes. Tensor strides, total KV
    cache pages, block-table padding, and max_context_length do not partition
    the tuning cache; max_context_length still bounds kernel scheduling.

    Retain the returned plan for decode/graph replay. Refresh it with
    plan_pa_decode(..., plan=plan) when context lengths change; decode never
    invokes autotune. Workspace sizes follow the returned plan.capacity.
    """
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_pa_decode_plan must run before graph capture")
        properties = torch.cuda.get_device_properties(query.device)
        num_cu = properties.multi_processor_count
        limit = num_cu if max_partitions is None else max_partitions
        # Validate plan geometry even when native autotune takes the default.
        default_plan = plan_pa_decode(
            context_lengths,
            key_cache.shape[1],
            max_partitions=limit,
            sliding_window=sliding_window,
            query_length=query_length,
        )
        options = {
            "compute_type": compute_type,
            "key_scale": key_scale,
            "value_scale": value_scale,
            "sinks": sinks,
            "sliding_window": default_plan.sliding_window,
            "max_context_length": max_context_length,
        }
        resources = _PADecodeAutotuneResources(
            query,
            (
                key_cache,
                value_cache,
                context_lengths,
                block_tables,
                softmax_scale,
                query_length,
            ),
            options,
            limit,
        )

        arch = properties.gcnArchName.split(":", 1)[0]
        config_dir = None if config_dir is None else Path(config_dir)
        tuner = _get_pa_decode_autotuner(arch, config_dir)
        tuning_args = {
            "num_seqs": context_lengths.numel(),
            "num_kv_heads": key_cache.shape[1],
            "query_group_size": query.shape[1] // key_cache.shape[1],
            "head_dim": query.shape[-1],
            "block_size": key_cache.shape[3],
            "query_length": query_length,
            "query_dtype": str(query.dtype),
            "kv_dtype": str(key_cache.dtype),
            "per_token_kv": (
                isinstance(key_scale, torch.Tensor) and key_scale.numel() > 1
            ),
            "trans_v": value_cache.ndim == 5,
            "sink_dtype": None if sinks is None else str(sinks.dtype),
            "sliding_window": default_plan.sliding_window,
            "max_partitions": limit,
            "num_cu": num_cu,
            "implementation": _tuning_cache_tag(),
            "resources": resources,
        }
        if tune:
            # Search this shape without changing process-wide environment or
            # the cached tuner's default policy. Preserve other shapes on disk.
            search = copy(tuner)
            search.cache = dict(tuner.cache)
            search._artifact_cache = dict(tuner._artifact_cache)
            search.default = None
            search.cache.pop(search._make_key((), tuning_args), None)
            config = search.resolve_config(**tuning_args)
            tuner.cache.update(search.cache)
        else:
            config = tuner.resolve_config(**tuning_args)
        budget = config.kwargs.get("workgroup_budget")
        if (
            set(config.all_kwargs()) != {"workgroup_budget"}
            or type(budget) is not int
            or budget < 1
            or config.compiler_opts()
            or config.pre_hook is not None
        ):
            raise ValueError("Invalid PA workgroup budget in FlyDSL autotune cache")
        if budget == 2 * num_cu:
            return default_plan
        return plan_pa_decode(
            context_lengths,
            key_cache.shape[1],
            max_partitions=limit,
            workgroup_budget=budget,
            sliding_window=default_plan.sliding_window,
            query_length=query_length,
        )


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune a FlyDSL PA decode shape and save its native configuration."
    )
    parser.add_argument("-b", "--batch-size", type=_positive_int, default=1)
    parser.add_argument("--num-query-heads", type=_positive_int, default=16)
    parser.add_argument("--num-kv-heads", type=_positive_int, default=1)
    parser.add_argument("--head-dim", type=_positive_int, default=128)
    parser.add_argument("-q", "--query-length", type=_positive_int, default=1)
    parser.add_argument(
        "-s",
        "--context-length",
        type=_positive_int,
        nargs="+",
        default=[4096],
        help="Lengths include query tokens; supply one for all rows or one per row.",
    )
    parser.add_argument("--block-size", type=int, choices=(16, 64, 128), default=128)
    parser.add_argument("--kv-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--query-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--scale-mode",
        choices=("per-tensor", "per-token", "none"),
        help="Default: per-token for FP8; none for BF16.",
    )
    parser.add_argument("--trans-v", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sliding-window", type=int, default=0)
    parser.add_argument(
        "--sink-dtype", choices=("none", "bf16", "fp16", "fp32"), default="none"
    )
    parser.add_argument("--max-partitions", type=_positive_int)
    parser.add_argument("--device", type=int, default=0, help="Visible GPU index.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save to DIR/<arch>/_pa_decode_autotuner.json; overrides cache env vars.",
    )
    args = parser.parse_args(argv)
    if args.batch_size > 4096:
        parser.error("--batch-size must be at most 4096")
    if args.num_query_heads % args.num_kv_heads:
        parser.error("--num-query-heads must be divisible by --num-kv-heads")
    if not 64 <= args.head_dim <= 1024 or args.head_dim % 64:
        parser.error("--head-dim must be a multiple of 64 in [64, 1024]")
    if args.kv_dtype == "fp8" and args.head_dim != 64 and args.head_dim % 128:
        parser.error("FP8 --head-dim must be 64 or a multiple of 128")
    if args.scale_mode is None:
        args.scale_mode = "per-token" if args.kv_dtype == "fp8" else "none"
    if args.kv_dtype == "bf16" and (
        args.query_dtype != "bf16" or not args.trans_v or args.scale_mode != "none"
    ):
        parser.error("BF16 KV requires BF16 queries, --trans-v 1 and --scale-mode none")
    if len(args.context_length) == 1:
        args.context_length *= args.batch_size
    elif len(args.context_length) != args.batch_size:
        parser.error("--context-length must contain one length or --batch-size lengths")
    if any(
        length < args.query_length or length >= 2**31 for length in args.context_length
    ):
        parser.error("context lengths must include all query tokens and fit in int32")
    if args.sliding_window < -1:
        parser.error("--sliding-window must be -1, 0 or positive")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    return args


def _quantize_kv(tensor, dtype, scale_mode):
    if scale_mode == "none":
        return tensor.to(dtype), None
    values = tensor.float()
    maximum = torch.finfo(dtype).max
    if scale_mode == "per-token":
        scale = values.abs().amax(dim=-1, keepdim=True)
    else:
        scale = values.abs().amax().reshape(1)
    scale = (scale / maximum).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (values / scale).clamp(-maximum, maximum).to(dtype)
    return quantized, scale.contiguous()


def _make_tuning_inputs(args, device, kv_dtype):
    """Construct the requested shape with logical, vectorized K/V layouts."""
    torch.manual_seed(args.seed)
    query_dtype = torch.bfloat16 if args.query_dtype == "bf16" else torch.float16
    page, dim, heads = args.block_size, args.head_dim, args.num_kv_heads
    max_length = max(args.context_length)
    pages_per_seq = (max_length + page - 1) // page
    pages = args.batch_size * pages_per_seq
    query = torch.empty(
        (args.batch_size * args.query_length, args.num_query_heads, dim),
        dtype=query_dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (pages, heads, page, dim), dtype=torch.bfloat16, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty_like(key).uniform_(-0.5, 0.5)
    key_scale = value_scale = None
    if args.kv_dtype == "fp8":
        key, key_scale = _quantize_kv(key, kv_dtype, args.scale_mode)
        value, value_scale = _quantize_kv(value, kv_dtype, args.scale_mode)
    vector = 8 if args.kv_dtype == "bf16" else 16
    key_cache = (
        key.reshape(pages, heads, page, dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, heads, page // vector, vector, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if args.trans_v
        else value.permute(0, 1, 3, 2).contiguous()
    )
    context = torch.tensor(args.context_length, dtype=torch.int32, device=device)
    table = torch.randperm(pages, dtype=torch.int32, device=device).reshape(
        args.batch_size, pages_per_seq
    )
    sinks = None
    if args.sink_dtype != "none":
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            args.sink_dtype
        ]
        sinks = torch.empty(args.num_query_heads, dtype=dtype, device=device).uniform_(
            -2, 2
        )
    return (
        query,
        key_cache,
        value_cache,
        context,
        table,
        dim**-0.5,
        args.query_length,
    ), {
        "compute_type": kv_dtype,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": args.sliding_window,
        "max_partitions": args.max_partitions,
        "max_context_length": max_length,
    }


def main(argv=None):
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("PA decode tuning requires a ROCm GPU")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"visible GPU index {args.device} is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch not in ("gfx942", "gfx950"):
        raise RuntimeError(f"PA decode tuning only supports gfx942/gfx950, got {arch}")
    if (
        args.max_partitions is not None
        and args.max_partitions > properties.multi_processor_count
    ):
        raise ValueError(
            f"--max-partitions must be at most {properties.multi_processor_count}"
        )
    kv_dtype = torch.bfloat16
    if args.kv_dtype == "fp8":
        kv_dtype = torch.float8_e4m3fn if arch == "gfx950" else torch.float8_e4m3fnuz
    print(
        f"Tuning {arch}: batch={args.batch_size}, Q heads={args.num_query_heads}, "
        f"KV heads={args.num_kv_heads}, dim={args.head_dim}, "
        f"query length={args.query_length}, max context={max(args.context_length)}, "
        f"page={args.block_size}, KV={args.kv_dtype}"
    )
    arguments, options = _make_tuning_inputs(args, device, kv_dtype)
    plan = prepare_pa_decode_plan(
        *arguments, **options, tune=True, config_dir=args.output_dir
    )
    cache_file = _get_pa_decode_autotuner(arch, args.output_dir)._cache_file
    print(f"Saved configuration to {cache_file}")
    print(
        f"Selected plan capacity={plan.capacity}, max partitions={plan.max_partitions}"
    )


if __name__ == "__main__":
    main()
