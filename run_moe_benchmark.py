"""Benchmark grouped MoE token sizes in a named container from the Linux host.

Select exactly one workload group: --dsr1, --dsv41_flash, or --dsv4_pro.
Routing defaults to non-balanced; --bln selects balanced routing. Add
--fused_sha_exp to include one shared expert for every token, increasing the
total experts and topk by one. --tokens selects one or more token counts;
omitting it runs all seventeen default counts. Use --container dsr1_ep4 to
override the default hyg_fyd1 container.
Docker access is detected automatically; sudo -n is used when direct Docker
access is denied. Every Docker command runs through the host gpu-lock, including
metadata probes. GPU workers run in the foreground without timeouts.

Pin is requested as 1 for three fresh processes, each calling the canonical
op test directly with all token counts after --tokens. APRE, tiles, and clusters use
the repository defaults. The original unit test, timer, and accuracy gate are
preserved. GPU idleness is managed by the operator; this driver does not poll
GPU status. Use --dry-run to inspect the plan without Docker or GPU execution.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path

TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    4096,
    6144,
    8192,
    16384,
)
WORKLOAD_GROUPS = {
    "dsr1": (64, 8, 7168, 2048),
    "dsv41_flash": (96, 6, 5120, 2304),
    "dsv4_pro": (96, 6, 7168, 3072),
}
TEST = "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
METRICS = tuple(
    f"gemm{stage}_{suffix}"
    for stage in (1, 2)
    for suffix in ("us", "TFLOPS", "Bandwidth (GB/s)")
)
ENV = {
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_GROUPED_DEBUG": "0",
    "ENABLE_CK": "0",
    "FLYDSL_DUMP_IR": "0",
    "AITER_LOG_MORE": "0",
    "AITER_FORCE_GFX1250": "1",
    "GPU_ARCHS": "gfx1250",
    "CU_NUM": "256",
    "AITER_MOE_NUM_EXPERT_ACTIVATED": "0",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
    "AITER_FLYDSL_EXPLICIT_VGPR_PARTITION": "1",
}

# Run the canonical CLI through runpy so branches without --output-json can
# still emit full-precision rows. Observe each eager dispatch, restore it before
# timing, and count active experts only after the original test has finished.
WORKER = r"""
import inspect, json, os, pathlib, runpy, sys
task = json.loads(sys.argv[1])
pathlib.Path(task["pid_file"]).write_text(str(os.getpid()), encoding="utf-8")
for name in tuple(os.environ):
    if name.startswith(("AITER_TDM_", "AITER_FLYDSL_", "AITER_CONFIG_")) or name in (
        "AITER_GROUPED_A4W4_A_PRESHUFFLE", "FLYDSL_RUNTIME_RUN_ONLY",
        "AITER_DISABLE_GROUPED_A8W4", "AITER_FORCE_A8W4", "AITER_SELECT",
        "LDS_SOA_LOAD_INTERLEAVE", "AITER_GROUPED_DEEPGEMM_CONTIGUOUS",
    ):
        os.environ.pop(name, None)
os.environ.update(task["env"])
sys.path.insert(0, task["repo"])
sys.argv = [task["test"], *task["args"]]
from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped
from aiter import test_common
original_run = grouped._run_compiled
original_perf = test_common.run_perftest
launches = {}
audited = False

def observe(launcher, *args, **kwargs):
    key = kwargs["specialization_key"]
    stage = 1 if key[1][1] else 2
    launches[f"gemm{stage}"] = {"N": args[7], "specialization_key": key}
    return original_run(launcher, *args, **kwargs)

def audited_perf(*args, **kwargs):
    global audited
    if not audited:
        grouped._run_compiled = original_run
        assert set(launches) == {"gemm1", "gemm2"}, launches
        audited = True
    assert kwargs["testGraph"] is False
    assert kwargs["num_warmup"] == 5 and kwargs["num_iters"] == 101
    return original_perf(*args, **kwargs)

namespace = runpy.run_path(task["test"], run_name="grouped_moe_pin_benchmark")
scope = namespace["main"].__globals__
original_summary = scope["summarize"]
original_moe = scope["run_moe"]
moe_signature = inspect.signature(original_moe)
original_topk = scope["_make_topk"]
work_metrics = scope["_gemm_work_metrics"]
supports_active = "active_experts" in inspect.signature(work_metrics).parameters
records = []
current = None

def capture_topk(*args, **kwargs):
    result = original_topk(*args, **kwargs)
    current["topk_ids"] = result[0]
    return result

def capture_moe(*args, **kwargs):
    global current, launches, audited
    bound = moe_signature.bind(*args, **kwargs)
    bound.apply_defaults()
    settings = bound.arguments
    current = {}
    launches = {}
    audited = False
    grouped._run_compiled = observe
    test_common.run_perftest = audited_perf
    scope["_make_topk"] = capture_topk
    try:
        metrics = original_moe(*args, **kwargs)
    finally:
        grouped._run_compiled = original_run
        test_common.run_perftest = original_perf
        scope["_make_topk"] = original_topk
    assert audited and set(launches) == {"gemm1", "gemm2"}, launches
    ids = current["topk_ids"]
    active = ids[ids >= 0].unique().numel()
    truthy = ("1", "true", "True", "yes", "YES")
    work = work_metrics(
        **({"active_experts": active} if supports_active else {"experts": active}),
        **{name: settings[name] for name in ("tokens", "topk", "model_dim", "inter_dim", "data_format")},
        shared_experts=settings.get("shared_experts", 0),
        fuse_gemm1_act=os.environ.get("AITER_FLYDSL_DISABLE_GEMM1_ACT", "").strip() not in truthy,
        fuse_gemm1_requant=os.environ.get("AITER_FLYDSL_DISABLE_GEMM1_REQUANT", "").strip() not in truthy,
    )
    metrics["active_experts"] = active
    record = {"tokens": settings["tokens"], "active_experts": active,
              "launch_audit": launches}
    for name in ("gemm1", "gemm2"):
        elapsed_us = metrics[name + "_us"]
        flops, nbytes = work[name]
        metrics[name + "_bytes"] = nbytes
        metrics[name + "_tflops"] = flops / elapsed_us / 1e6
        metrics[name + "_bandwidth_gbs"] = nbytes / elapsed_us / 1e3
        record[name + "_bytes"] = nbytes
    records.append(record)
    print("MOE_PIN_LAUNCH_AUDIT=" + json.dumps(
        {"tokens": settings["tokens"], "launches": launches}, default=str), flush=True)
    return metrics

def record(rows):
    assert len(rows) == len(records), (len(rows), len(records))
    output_rows = []
    for row, captured in zip(rows, records):
        assert row["tokens"] == captured["tokens"]
        output_rows.append({**row, **captured})
    pathlib.Path(task["result"]).write_text(
        json.dumps(output_rows, indent=2, default=str) + "\n", encoding="utf-8")
    pathlib.Path(task["audit"]).write_text(
        json.dumps(records[0]["launch_audit"] if len(records) == 1 else records,
                   indent=2, default=str) + "\n", encoding="utf-8")
    print("MOE_PIN_METRICS=" + json.dumps(output_rows, default=str), flush=True)
    return original_summary(rows)

scope["run_moe"] = capture_moe
scope["summarize"] = record
namespace["main"]()
"""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def capture(command, repo, timeout=20):
    options = {
        "cwd": repo,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if Path(command[0]).name == "gpu-lock":
        process = subprocess.Popen(command, **options, start_new_session=True)
        try:
            output, _ = process.communicate()
        except BaseException:
            wait_for_worker(process, drain=True)
            raise
        return subprocess.CompletedProcess(command, process.returncode, output)
    return subprocess.run(command, **options, timeout=timeout, check=False)


def checked(command, repo):
    result = capture(command, repo)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n{result.stdout}"
        )
    return result.stdout.strip()


def detect_docker_command(repo):
    gpu_lock = shutil.which("gpu-lock")
    if not gpu_lock:
        raise RuntimeError("Host gpu-lock is required for every Docker command.")
    probe = ["info", "--format", "{{.ServerVersion}}"]
    command = [gpu_lock, "docker"]
    result = capture(command + probe, repo)
    if result.returncode == 0:
        return command
    if "permission denied" not in result.stdout.lower():
        raise RuntimeError(f"Cannot access Docker: {result.stdout.strip()}")
    if not shutil.which("sudo"):
        raise RuntimeError("Docker access requires sudo, but sudo is unavailable.")
    command = [gpu_lock, "sudo", "-n", "docker"]
    result = capture(command + probe, repo)
    if result.returncode:
        raise RuntimeError(
            "Non-interactive sudo could not access Docker. If a password is "
            "required, run sudo -v before rerunning this script.\n"
            f"{result.stdout.strip()}"
        )
    return command


def test_args(
    tokens, experts, *, topk=8, shared_experts=0, model_dim=7168, inter_dim=2048
):
    token_values = [tokens] if isinstance(tokens, int) else list(tokens)
    args = [
        "--scenario",
        "kernel",
        "--data-format",
        "a4w4",
        "--experts",
        str(experts),
        "--tokens",
        *map(str, token_values),
        "--topk",
        str(topk),
        "--model-dim",
        str(model_dim),
        "--inter-dim",
        str(inter_dim),
        "--act",
        "silu",
        "--no-bias",
        "--no-check-aot-cache",
        "--warmup",
        "5",
        "--iters",
        "101",
        "--seed",
        "0",
        "--data-init",
        "uniform",
        "--scale-init",
        "auto",
    ]
    if shared_experts:
        args += ["--shared-experts", str(shared_experts)]
    return args


def make_plan(
    experts, *, tokens=TOKENS, topk=8, shared_experts=0, model_dim=7168, inter_dim=2048
):
    workload = {
        "experts": experts,
        "topk": topk,
        "shared_experts": shared_experts,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
    }
    return [
        {
            **workload,
            "tokens": list(tokens),
            "pin": 1,
            "repeat": repeat,
            "args": test_args(tokens, **workload),
        }
        for repeat in range(1, 4)
    ]


def wait_for_worker(process, *, drain=False):
    # Keep the host Docker client and its lock alive until the worker exits.
    while process.poll() is None:
        try:
            if drain:
                process.communicate()
            else:
                process.wait()
        except KeyboardInterrupt:
            print("Waiting for the current GPU worker to exit normally.", flush=True)


def validate_row(
    row,
    tokens,
    experts,
    *,
    topk=8,
    shared_experts=0,
    model_dim=7168,
    inter_dim=2048,
    source="result",
):
    expected = {
        "data_format": "a4w4",
        "act": "silu",
        "experts": experts,
        "tokens": tokens,
        "topk": topk,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "seed": 0,
        "data_init": "uniform",
        "scale_init": "auto",
    }
    if (
        row.get("pass") is not True
        or not 0 <= row.get("logits_diff", float("inf")) < 0.01
    ):
        raise RuntimeError(f"The original MoE correctness gate failed: {source}")
    if (
        any(row.get(k) != v for k, v in expected.items())
        or row.get("shared_experts", 0) != shared_experts
    ):
        raise RuntimeError(f"Unexpected workload: {source}")
    active = row.get("active_experts")
    if type(active) is not int or not topk <= active <= min(
        experts, tokens * (topk - shared_experts) + shared_experts
    ):
        raise RuntimeError(
            f"Missing or invalid actual active expert count: {source}; sync the op test too"
        )
    for key in METRICS:
        value = row.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RuntimeError(f"Invalid metric {key}={value}: {source}")
    for stage in (1, 2):
        data_bytes = row.get(f"gemm{stage}_bytes")
        bw = row[f"gemm{stage}_Bandwidth (GB/s)"]
        us = row[f"gemm{stage}_us"]
        if not isinstance(data_bytes, (int, float)) or not math.isclose(
            bw * us * 1e3, data_bytes, rel_tol=1e-12
        ):
            raise RuntimeError(
                f"Inconsistent GEMM{stage} byte estimate or bandwidth: {source}"
            )
    return row


def validate_result(path, tokens, experts, **kwargs):
    # Retained for the single-case configuration-screening helpers.
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one result: {path}")
    return validate_row(rows[0], tokens, experts, source=str(path), **kwargs)


def aggregate(
    samples,
    experts,
    routing,
    *,
    tokens=TOKENS,
    topk=8,
    shared_experts=0,
    model_dim=7168,
    inter_dim=2048,
):
    rows = []
    for token in tokens:
        selected = [r for r in samples if r["tokens"] == token and r["pin"] == 1]
        row = {
            "experts": experts,
            "routing": routing,
            "topk": topk,
            "shared_experts": shared_experts,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "tokens": token,
            "pin": 1,
            "samples": len(selected),
        }
        counts = {r["active_experts"] for r in selected}
        if len(counts) > 1:
            raise RuntimeError(
                f"Routing changed across fixed-seed repeats: T={token}, pin=1"
            )
        row["active_experts"] = next(iter(counts)) if len(selected) == 3 else None
        row.update(
            {
                key: (
                    statistics.median(r[key] for r in selected)
                    if len(selected) == 3
                    else None
                )
                for key in METRICS
            }
        )
        row.update(
            {
                f"gemm{stage}_us_samples": [r[f"gemm{stage}_us"] for r in selected]
                for stage in (1, 2)
            }
        )
        rows.append(row)
    return rows


def save_summary(root, state):
    experts, routing = state["experts"], state["routing"]
    topk, shared_experts = state["topk"], state["shared_experts"]
    token_values = state["tokens"]
    model_dim, inter_dim = state["model_dim"], state["inter_dim"]
    rows = aggregate(
        state["samples"],
        experts,
        routing,
        tokens=token_values,
        topk=topk,
        shared_experts=shared_experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
    )
    write_json(
        root / "summary.json",
        {
            "status": state["status"],
            "experts": experts,
            "topk": topk,
            "shared_experts": shared_experts,
            "group": state["group"],
            "tokens": token_values,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "routing": routing,
            "medians": rows,
        },
    )
    fields = [
        "experts",
        "routing",
        "topk",
        "shared_experts",
        "model_dim",
        "inter_dim",
        "tokens",
        "pin",
        "samples",
        "active_experts",
        *METRICS,
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    shared_tag = f"/shared{shared_experts}" if shared_experts else ""
    workload_tag = f"E{experts}/topk{topk}{shared_tag}/M{model_dim}/I{inter_dim}"
    lines = [
        f"# {workload_tag}, {routing}: pin1 benchmark",
        "",
        f"Workload group: `{state['group']}`; token counts: `{token_values}`.",
        (
            f"Status: `{state['status']}`; completed processes: {state['completed_processes']}/{len(state['plan'])}; "
            f"accepted case samples: {len(state['samples'])}/{len(token_values) * len(state['plan'])}."
        ),
        f"Host repository: `{state['repo']}`; container: `{state['container']}`.",
        f"Branch: `{state['branch']}`; HEAD: `{state['commit']}`.",
        "",
        "Original kernel timer: `run_perftest(testGraph=False)`; warmup=5, iters=101.",
        f"Requested pin=1 has three fresh-process samples; each process runs all {len(token_values)} selected --tokens values. Routing is {routing}, seed=0.",
        "APRE, tiles, and clusters use the current repository defaults. Inherited tuning overrides are cleared.",
        "GPU idleness is managed by the operator; this driver performs no GPU status checks or idle polling.",
        "AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1 is set explicitly; actual specialization keys are in *_launch.json.",
        (
            f"Expert/topk totals include {shared_experts} trailing shared experts; routed experts={experts-shared_experts}, routed topk={topk-shared_experts}. "
            "Shared outputs have unit weight outside routed renormalization."
        ),
        "All metrics below are **pin1 medians**; '-' denotes fewer than three accepted samples.",
        "BW uses logical quantized input/weight payloads, E8M0 scales and the actual fused output format (BF16 when unfused).",
        "Weight payloads/scales count only experts present in the actual routing; inactive experts are excluded.",
        "Active counts are collected after timing. Effective BW is not measured HBM traffic; TFLOPS/timing are unchanged.",
        "",
        "| Workload | Active experts | GEMM1 us | GEMM2 us | GEMM1 TFLOPS | GEMM1 BW GB/s | GEMM2 TFLOPS | GEMM2 BW GB/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_tokens = {r["tokens"]: r for r in rows}
    for tokens in token_values:
        row = by_tokens[tokens]
        cells = ["-" if row["active_experts"] is None else str(row["active_experts"])]
        for key in (
            METRICS[0],
            METRICS[3],
            METRICS[1],
            METRICS[2],
            METRICS[4],
            METRICS[5],
        ):
            value = row[key]
            cells.append("-" if value is None else f"{value:.6f}")
        lines.append(
            f"| E{experts}/T{tokens}/topk{topk}{shared_tag}/M{model_dim}/I{inter_dim}, {routing} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Individual latency samples",
        "",
        "| T | GEMM1 pin1 us (three samples) | GEMM2 pin1 us (three samples) |",
        "|---|---|---|",
    ]
    for tokens in token_values:
        cells = [
            "("
            + ", ".join(
                f"{v:.6f}" for v in by_tokens[tokens][f"gemm{stage}_us_samples"]
            )
            + ")"
            for stage in (1, 2)
        ]
        lines.append(f"| {tokens} | " + " | ".join(cells) + " |")
    if state.get("error"):
        lines += ["", f"Stopped: {state['error']}"]
    lines += [
        "",
        "Raw sweep stdout and canonical JSON results: `*_tokens_pin*_repeat*.log/json`; exact command arrays: `*_command.json`.",
        "Per-case full-precision result copies: `*_t*_pin*_repeat*.json`; eager launch audits: `*_launch.json`.",
        "Progress: `run.log`; parameters, source hashes, samples: `metadata.json`.",
        "Source diff: `source_diff.patch`.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--container",
        default="hyg_fyd1",
        help="Running Docker container name or ID (default: hyg_fyd1).",
    )
    group_args = parser.add_mutually_exclusive_group(required=True)
    for name, (experts, topk, model_dim, inter_dim) in WORKLOAD_GROUPS.items():
        group_args.add_argument(
            f"--{name}",
            dest="group",
            action="store_const",
            const=name,
            help=f"E{experts}/topk{topk}/shared0/M{model_dim}/I{inter_dim}.",
        )
    parser.add_argument(
        "--bln",
        action="store_true",
        help="Use balanced routing (default: non-balanced).",
    )
    parser.add_argument(
        "--fused_sha_exp",
        action="store_true",
        help="Add one shared expert with unit weight for every token; total experts and topk increase by one.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(TOKENS),
        help="Positive, unique token counts to run, in the given order (default: all seventeen counts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job plan; do not use Docker or GPU.",
    )
    args = parser.parse_args()
    if not args.container.strip():
        parser.error("--container must be a nonempty container name or ID")
    if any(token <= 0 for token in args.tokens):
        parser.error("--tokens values must be positive integers")
    if len(set(args.tokens)) != len(args.tokens):
        parser.error("--tokens values must be unique")
    experts, topk, model_dim, inter_dim = WORKLOAD_GROUPS[args.group]
    shared_experts = int(args.fused_sha_exp)
    experts += shared_experts
    topk += shared_experts
    workload = {
        "experts": experts,
        "topk": topk,
        "shared_experts": shared_experts,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
    }
    repo = args.repo.resolve()
    routing = "balanced" if args.bln else "non-balanced"
    environment = {
        "PYTHONPATH": str(repo),
        **ENV,
        "AITER_MOE_EXPERT_BALANCE": "true" if args.bln else "false",
    }
    plan = make_plan(tokens=args.tokens, **workload)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "repo": str(repo),
                    "container": args.container,
                    "group": args.group,
                    **workload,
                    "routing": routing,
                    "tokens": args.tokens,
                    "workload_count": len(args.tokens),
                    "worker_count": len(plan),
                    "case_sample_count": len(args.tokens) * len(plan),
                    "test_command": ["python3", "-u", TEST],
                    "environment": environment,
                    "pin_environment": "AITER_FLYDSL_EXPLICIT_VGPR_PARTITION",
                    "apre_policy": "repository default",
                    "gpu_idle_check": False,
                    "gpu_access": "host gpu-lock for every Docker command",
                    "gpu_job_timeout": None,
                    "workers": plan,
                },
                indent=2,
            )
        )
        return 0
    if os.name != "posix" or not shutil.which("docker"):
        parser.error("Run this script on the Linux host with Docker available.")
    revision = checked(["git", "rev-parse", "HEAD"], repo)
    branch = checked(["git", "branch", "--show-current"], repo)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    shared_tag = "_shared1" if shared_experts else ""
    log_dir = f"{args.group}{shared_tag}_{routing.replace('-', '_')}_pin_benchmark_logs"
    root = repo / "my_code" / log_dir / f"{stamp}_{revision[:8]}"
    root.mkdir(parents=True)
    config_paths = [
        "aiter/configs/tuned_grouped_fmoe.csv",
        *(
            path.relative_to(repo).as_posix()
            for path in sorted(
                (repo / "aiter/configs/model_configs").glob("*tuned_grouped_fmoe*.csv")
            )
            if path.is_file() and "untuned" not in path.name
        ),
    ]
    paths = [
        TEST,
        "aiter/test_common.py",
        *config_paths,
        "aiter/ops/flydsl/grouped_moe_gfx1250.py",
        "aiter/ops/flydsl/grouped_gemm_mxfp4.py",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py",
    ]

    def source_hashes():
        return {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in paths
        }

    state = {
        "repo": str(repo),
        "container": args.container,
        "commit": revision,
        "branch": branch,
        "group": args.group,
        **workload,
        "tokens": args.tokens,
        "routing": routing,
        "environment": environment,
        "plan": plan,
        "apre_policy": "repository default",
        "timer": "original run_perftest, testGraph=False",
        "source_sha256": source_hashes(),
        "status": "running",
        "samples": [],
        "gpu_job_timeout": None,
        "gpu_access": "host gpu-lock for every Docker command",
        "gpu_idle_check": False,
        "completed_processes": 0,
        "invocation": "canonical multi-token op-test CLI via runpy",
    }

    def log(message):
        line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with (root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def save():
        write_json(root / "metadata.json", state)
        save_summary(root, state)

    try:
        shutil.copyfile(Path(__file__), root / "driver.py")
        for name in config_paths:
            destination = root / Path(name).relative_to("aiter/configs")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, destination)
        diff = capture(["git", "diff", "HEAD", "--", *paths], repo)
        if diff.returncode:
            raise RuntimeError(f"Cannot record the source diff: {diff.stdout}")
        (root / "source_diff.patch").write_text(
            diff.stdout, encoding="utf-8", newline="\n"
        )
        save()
        log(
            f"Log directory: {root}; group={args.group}; {len(plan)} direct canonical processes, {len(args.tokens)} token cases per process; "
            f"{len(args.tokens) * len(plan)} case samples in total."
        )
        docker_command = detect_docker_command(repo)
        state["docker_command_prefix"] = docker_command
        log(f"Docker command: {' '.join(docker_command)}")
        save()
        inspection = checked(
            [
                *docker_command,
                "inspect",
                "--format",
                '{"id":{{json .Id}},"image":{{json .Config.Image}},"running":{{json .State.Running}}}',
                args.container,
            ],
            repo,
        )
        state["container_info"] = json.loads(inspection)
        if not state["container_info"]["running"]:
            raise RuntimeError(f"Container {args.container} is not running")
        probe = (
            "import importlib.metadata as m,json,os,shutil,sys; from pathlib import Path; "
            "repo=Path(sys.argv[1]); root=Path(sys.argv[2]); "
            f"assert (repo/{TEST!r}).is_file(); "
            "(root/'container_environment.json').write_text("
            "json.dumps({'packages':{k:m.version(k) for k in ('flydsl','torch')},"
            "'override_names':[k for k in os.environ if "
            "k.startswith(('AITER_TDM_','AITER_FLYDSL_','AITER_CONFIG_')) or "
            "k in ('AITER_GROUPED_A4W4_A_PRESHUFFLE','FLYDSL_RUNTIME_RUN_ONLY',"
            "'AITER_DISABLE_GROUPED_A8W4','AITER_FORCE_A8W4','AITER_SELECT',"
            "'LDS_SOA_LOAD_INTERLEAVE','AITER_GROUPED_DEEPGEMM_CONTIGUOUS')]},indent=2)+'\\n')"
        )
        checked(
            [
                *docker_command,
                "exec",
                args.container,
                "python3",
                "-c",
                probe,
                str(repo),
                str(root),
            ],
            repo,
        )
        checked(
            [
                *docker_command,
                "exec",
                args.container,
                "chmod",
                "-R",
                "a+rwX",
                str(root),
            ],
            repo,
        )
        container_environment = json.loads(
            (root / "container_environment.json").read_text(encoding="utf-8")
        )
        state["packages"] = container_environment["packages"]
        state["cleared_container_overrides"] = sorted(
            container_environment["override_names"]
        )
        save()
        for job in plan:
            if (
                source_hashes() != state["source_sha256"]
                or checked(["git", "rev-parse", "HEAD"], repo) != revision
            ):
                raise RuntimeError("Repository source or HEAD changed during this run")
            pin, repeat = job["pin"], job["repeat"]
            label = f"e{experts}_tokens_pin{pin}_repeat{repeat}"
            task = {
                "repo": str(repo),
                "test": TEST,
                "args": job["args"],
                "env": state["environment"],
                "result": str(root / f"{label}.json"),
                "audit": str(root / f"{label}_launch.json"),
                "pid_file": str(root / f"{label}.pid"),
            }
            command = [
                *docker_command,
                "exec",
                "-w",
                str(repo),
                args.container,
                "env",
            ]
            for name in state["cleared_container_overrides"]:
                command += ["-u", name]
            command += [
                f"{name}={value}" for name, value in sorted(task["env"].items())
            ]
            command += ["python3", "-u", "-c", WORKER, json.dumps(task)]
            write_json(root / f"{label}_command.json", command)
            log(
                f"Running E{experts}/topk{topk}/shared{shared_experts}/M{model_dim}/I{inter_dim}, {routing}; "
                f"all tokens={job['tokens']}; pin={pin}, repeat={repeat}/3."
            )
            with (root / f"{label}.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    rc = process.wait()
                except BaseException:
                    wait_for_worker(process)
                    raise
            if rc:
                raise RuntimeError(f"{label} failed (returncode={rc}); see {label}.log")
            checked(
                [
                    *docker_command,
                    "exec",
                    args.container,
                    "chmod",
                    "-R",
                    "a+rwX",
                    str(root),
                ],
                repo,
            )
            results = json.loads(Path(task["result"]).read_text(encoding="utf-8"))
            if [row.get("tokens") for row in results] != args.tokens:
                raise RuntimeError(
                    f"Missing, duplicate or unexpected token results: {label}"
                )
            for result in results:
                tokens = result["tokens"]
                row = validate_row(result, tokens, **workload, source=task["result"])
                launches = row.pop("launch_audit")
                if set(launches) != {"gemm1", "gemm2"}:
                    raise RuntimeError(
                        f"Missing eager launch audits: {label}, T={tokens}"
                    )
                case_label = f"e{experts}_t{tokens}_pin{pin}_repeat{repeat}"
                write_json(root / f"{case_label}.json", [row])
                write_json(root / f"{case_label}_launch.json", launches)
                state["samples"].append(
                    {
                        "label": case_label,
                        "batch": label,
                        "pin": pin,
                        "repeat": repeat,
                        "routing": routing,
                        **row,
                    }
                )
                log(
                    f"{case_label}: GEMM1={row['gemm1_us']:.6f} us; GEMM2={row['gemm2_us']:.6f} us."
                )
            state["completed_processes"] += 1
            save()
        if (
            source_hashes() != state["source_sha256"]
            or checked(["git", "rev-parse", "HEAD"], repo) != revision
        ):
            raise RuntimeError("Repository source or HEAD changed during this run")
        state["status"] = "complete"
        save()
        log(
            f"Complete: {state['completed_processes']}/{len(plan)} processes, "
            f"{len(state['samples'])}/{len(args.tokens) * len(plan)} case samples; "
            f"medians and individual samples: {root / 'summary.md'}"
        )
        write_json(
            root / "artifact_sha256.json",
            {
                f.relative_to(root)
                .as_posix(): hashlib.sha256(f.read_bytes())
                .hexdigest()
                for f in sorted(root.rglob("*"))
                if f.is_file() and f.name != "artifact_sha256.json"
            },
        )
        return 0
    # Preserve run artifacts for all failures, including interruptions.
    except BaseException as exc:  # noqa: BLE001
        state["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        state["error"] = f"{type(exc).__name__}: {exc}"
        save()
        log(
            f"Stopped; all available logs retained: {state['error']}; summary: {root / 'summary.md'}"
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
