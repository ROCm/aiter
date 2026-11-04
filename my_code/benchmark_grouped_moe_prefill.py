#!/usr/bin/env python3
"""Measure per-rank grouped expert compute using the actual EP tuning row.

This is a single-GPU compute fixture, not a distributed EP benchmark. The
existing op test supplies routing, quantization, reference, tolerance, and
torch.profiler timing. Only configuration lookup is adapted: it uses the
global receive bound and local expert count of the requested EP deployment.
GEMM1 input preparation precedes the measured GEMM1 callable. By default,
the tuned CSV selects the input layout, including the small-batch fallback.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


def _find_grouped_config_with_padding(find, pad, query):
    """Preserve raw token rows and support callers that pad before lookup."""
    row = find(**query)
    if row is None:
        padded_tokens = pad(query["token_num"])
        if padded_tokens != query["token_num"]:
            query = dict(query, token_num=padded_tokens)
            row = find(**query)
    return row, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--global-experts", type=int, required=True)
    parser.add_argument("--ep-world-size", type=int, required=True)
    parser.add_argument(
        "--tokens", type=int, required=True, help="Effective tokens per source rank"
    )
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--inter-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--scenario", choices=("kernel", "verify"), default="kernel")
    parser.add_argument(
        "--a-preshuffle",
        choices=("csv", "0", "1"),
        default="csv",
        help="Use the production CSV input layout, or explicitly override it",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.global_experts > 0 and args.ep_world_size > 0
    assert args.global_experts % args.ep_world_size == 0
    local_experts = args.global_experts // args.ep_world_size
    assert 0 < args.topk <= local_experts and args.tokens > 0
    repo = args.repo_root.resolve()
    config = args.config.resolve()
    assert config.is_file()
    os.environ["AITER_CONFIG_GROUPED_FMOE"] = str(config)
    if args.a_preshuffle == "csv":
        os.environ.pop("AITER_GROUPED_A4W4_A_PRESHUFFLE", None)
    else:
        os.environ["AITER_GROUPED_A4W4_A_PRESHUFFLE"] = args.a_preshuffle
    sys.path.insert(0, str(repo))

    import torch

    from aiter import ActivationType
    from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped
    from aiter.ops.flydsl import grouped_moe_gfx1250 as moe

    test_path = repo / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location(
        "existing_grouped_moe_prefill_test", test_path
    )
    test = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = test
    spec.loader.exec_module(test)
    selected, launches, routes = {}, [], {}
    find = moe._find_grouped_config
    dispatch = grouped._run_compiled
    make_topk = test._make_topk

    def ep_compute_config(**query):
        receive_bound = args.ep_world_size * args.tokens
        production_query = dict(
            query,
            token_num=receive_bound,
            experts=local_experts,
            topk=-1,
            ep_fused=True,
        )
        row, production_query = _find_grouped_config_with_padding(
            find, moe._get_padded_m, production_query
        )
        assert row is not None, "The requested EP workload has no tuned row"
        selected.update(
            receive_bound=receive_bound,
            lookup_token=production_query["token_num"],
            row=dict(row),
        )
        # Keep CSV fallbacks intact unless the command explicitly requests a
        # layout. Apply the same override to revisions without the env switch.
        return (
            row
            if args.a_preshuffle == "csv"
            else dict(row, a_preshuffle=args.a_preshuffle)
        )

    def capture_topk(hidden, experts, topk):
        ids, weights = make_topk(hidden, experts, topk)
        flat = ids.detach().cpu().reshape(-1).long()
        counts = torch.bincount(flat[flat >= 0], minlength=experts)
        assert int(counts.sum()) == args.tokens * args.topk
        routes.update(
            effective_tokens=args.tokens,
            physical_tokens=hidden.shape[0],
            valid_routes=int(counts.sum()),
            counts=counts.tolist(),
            ids_sha256=hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
            weights_sha256=hashlib.sha256(
                weights.detach().cpu().numpy().tobytes()
            ).hexdigest(),
        )
        return ids, weights

    def capture_dispatch(launcher, *values, **kwargs):
        bound = inspect.signature(launcher.func).bind(*values)
        bound.apply_defaults()
        params = bound.arguments
        fields = (
            "N",
            "K",
            "tile_m",
            "tile_n",
            "tile_k",
            "m_warp",
            "n_warp",
            "num_buffers",
            "cluster_m",
            "cluster_n",
            "i32_m",
            "a_is_fp4",
            "a_preshuffled",
            "row_major_ascale",
            "stage1_act",
            "stage1_quant_out",
            "quant_payload_preshuffled",
            "quant_wmma_rep",
            "next_stage_prefetch",
            "tdm_b_th",
            "tdm_as_in_prologue",
            "lds_soa_load_interleave",
            "rolled_drain",
            "persistent_workers",
            "enable_ep_scatter",
            "explicit_vgpr_partition",
            "m_align",
        )
        record = {name: int(params[name]) for name in fields if name in params}
        # Initial dispatch order also identifies GEMM1 when activation is
        # performed by the following quantizer instead of the GEMM epilogue.
        record["stage"] = len(launches) + 1
        assert not record["enable_ep_scatter"]
        if args.a_preshuffle != "csv":
            assert record["a_preshuffled"] == int(args.a_preshuffle)
        launches.append(record)
        result = dispatch(launcher, *values, **kwargs)
        # Restore before the op test creates timed partials or invokes its timer.
        if len(launches) == 2:
            grouped._run_compiled = dispatch
        return result

    moe._find_grouped_config = ep_compute_config
    grouped._run_compiled = capture_dispatch
    test._make_topk = capture_topk
    try:
        metrics = test.run_moe(
            "a4w4",
            experts=local_experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            activation=ActivationType.Silu,
            swiglu_limit=args.swiglu_limit,
            use_bias=False,
            check_aot_cache=False,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            data_init="uniform",
            scale_init="auto",
            kernel_bench=args.scenario == "kernel",
            bench=False,
            raise_on_fail=True,
        )
    finally:
        moe._find_grouped_config = find
        grouped._run_compiled = dispatch
        test._make_topk = make_topk
    assert {entry["stage"] for entry in launches} == {1, 2} and len(launches) == 2
    assert metrics["passed"] and metrics["logits_diff"] < 0.01
    report = {
        "pipeline": (
            "single_gpu_preshuffled_compute"
            if launches[0]["a_preshuffled"]
            else "single_gpu_compute_csv_fallback"
        ),
        "a_preshuffle_mode": args.a_preshuffle,
        "ep4_runtime_validated": False,
        "global_experts": args.global_experts,
        "local_experts": local_experts,
        "ep_world_size": args.ep_world_size,
        "tokens": args.tokens,
        "topk": args.topk,
        "model_dim": args.model_dim,
        "inter_dim": args.inter_dim,
        "pin_requested": os.environ.get("AITER_FLYDSL_EXPLICIT_VGPR_PARTITION"),
        "expert_balance": os.environ.get("AITER_MOE_EXPERT_BALANCE"),
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "scenario": args.scenario,
        "routes": routes,
        "config_lookup": selected,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "timer_sha256": hashlib.sha256(
            (repo / "aiter/test_common.py").read_bytes()
        ).hexdigest(),
        "launches": launches,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key.startswith(("gemm1_", "gemm2_"))
            or key in ("logits_diff", "rel_l2", "passed")
        },
    }
    module_path = Path(grouped.__file__).resolve()
    assert module_path.is_relative_to(repo)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("GROUPED_PREFILL_RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
