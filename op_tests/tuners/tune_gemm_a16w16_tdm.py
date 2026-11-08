# SPDX-License-Identifier: MIT
"""AITER offline TDM split-K tuner: GemmCommonTuner ranking/CSV + run_perftest."""

import json
import math
import random
import statistics
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch

from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm
from aiter.ops.flydsl.gemm_a16w16_tdm_policy import (
    KEYS,
    SCHEMA,
    TdmTunedPolicy,
    candidates,
    contract,
    precision_limit,
)
from aiter.test_common import run_perftest
from aiter.utility.base_tuner import GemmCommonTuner


class TdmGemmTuner(GemmCommonTuner):
    SUPPORTS_POWER_SELECT = False
    ARG_DEFAULTS: ClassVar[dict] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "mp": 1,
        "errRatio": 0.004,
        "warmup": 3,
        "iters": 16,
    }

    def _setup_specific_arguments(self):
        self.parser.set_defaults(mp=1)
        self.parser.add_argument(
            "--shapes",
            nargs="+",
            default=["256,256,8192"],
            help="M,N,K triples, or use -i with M,N,K columns",
        )
        self.parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
        self.parser.add_argument("--layout", choices=("nt", "nn"), default="nt")
        self.parser.add_argument("--bias", action="store_true")
        self.parser.add_argument(
            "--allow-atomic",
            action="store_true",
            help="permit rounded nondeterministic packed split-K",
        )
        self.parser.add_argument("--workspace-mib", type=int, default=256)
        self.parser.add_argument("--splits", default="1,2,4,8")
        self.parser.add_argument(
            "--exhaustive",
            action="store_true",
            help="disable grid pruning, not legality/memory checks",
        )
        self.parser.add_argument("--rounds", type=int, default=3)
        self.parser.add_argument("--shortlist", type=int, default=5)
        self.parser.add_argument("--seed", type=int, default=139)

    def pre_process(self, args):
        if args.mp != 1 or args.compare or args.run_config or args.all:
            self.parser.error(
                "this exact-contract tuner supports --mp 1; no --compare/--run_config/--all"
            )
        if not args.tune_file:
            self.parser.error(
                "-o/--tune_file is required (no default table is modified)"
            )
        if (
            args.rounds < 1
            or args.shortlist < 1
            or args.workspace_mib < 0
            or args.iters < 1
        ):
            self.parser.error("invalid timing or memory limit")
        if not math.isfinite(args.errRatio) or args.errRatio <= 0:
            self.parser.error("--errRatio must be a positive finite relative-L2 bound")
        if not torch.cuda.is_available() or self.get_gfx() != "gfx1250":
            raise RuntimeError("exclusive gfx1250 GPU required")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("offline tuning cannot run inside graph capture")
        if args.allow_atomic and torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("--allow-atomic conflicts with deterministic algorithms")
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        self.limit = precision_limit(str(self.dtype), args.allow_atomic, args.errRatio)
        self.splits = tuple(int(x) for x in args.splits.split(","))
        self.budget = args.workspace_mib << 20
        if args.untune_file:
            shapes = [
                tuple(int(row[k]) for k in ("M", "N", "K"))
                for _, row in pd.read_csv(args.untune_file).iterrows()
            ]
        else:
            shapes = [tuple(map(int, text.split(","))) for text in args.shapes]
        rows = []
        for shape in dict.fromkeys(shapes):
            if len(shape) != 3 or min(shape) <= 0:
                raise ValueError("shapes must be positive M,N,K triples")
            m, n, k = shape
            lda = (k + 7) // 8 * 8
            ldb = lda if args.layout == "nt" else (n + 7) // 8 * 8
            if (
                max(
                    m * lda,
                    (n if args.layout == "nt" else k) * ldb,
                    m * ((n + 7) // 8 * 8),
                )
                * 2
                >= 2**31
            ):
                raise ValueError("shape exceeds the kernel's tensor footprint limit")
            rows.append(
                dict(
                    zip(
                        KEYS,
                        (
                            self.get_gfx(),
                            self.get_cu_num(),
                            m,
                            n,
                            k,
                            SCHEMA,
                            str(self.dtype),
                            args.layout,
                            lda,
                            ldb,
                            (n + 7) // 8 * 8,
                            int(args.bias),
                            int(args.allow_atomic),
                            self.budget,
                            format(self.limit, ".12g"),
                        ),
                    )
                )
            )
        self.untunedf = pd.DataFrame(rows, columns=KEYS)
        if Path(args.tune_file).exists():
            TdmTunedPolicy(args.tune_file)  # reject incompatible or corrupt tables
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        # Re-tune requested contracts, merge them with untouched rows on output.
        self.audit = []
        self.audit_path = Path(str(args.tune_file) + ".audit.json")
        for p in (Path(args.tune_file), self.audit_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    def result_to_csv(self, resultdf, file, concat=False):
        # This tuner deliberately remeasures requested keys. Preserve all other
        # contracts and replace these keys, even when the new timing is slower.
        return super().result_to_csv(resultdf, file, concat=False)

    def getKernelName(self, kernel_id):
        return str(kernel_id)

    def _save_audit(self):
        self.audit_path.write_text(json.dumps(self.audit, indent=2) + "\n")

    def tune(self, untunedf, tunedf, args):
        results = []
        for _, r in untunedf.iterrows():
            m, n, k = (int(r[x]) for x in ("M", "N", "K"))
            key = tuple(r[x] for x in KEYS)
            cs = candidates(
                m,
                n,
                k,
                layout=args.layout,
                cu_num=int(r["cu_num"]),
                allow_atomic=args.allow_atomic,
                max_workspace_bytes=self.budget,
                splits=self.splits,
                exhaustive=args.exhaustive,
            )
            if not cs:
                raise RuntimeError(f"no legal configs for {m, n, k}")
            a = torch.empty(m, int(r["lda"]), device="cuda", dtype=self.dtype)[:, :k]
            if args.layout == "nt":
                b = torch.empty(n, int(r["ldb"]), device="cuda", dtype=self.dtype)[
                    :, :k
                ].t()
            else:
                b = torch.empty(k, int(r["ldb"]), device="cuda", dtype=self.dtype)[
                    :, :n
                ]
            out = torch.empty(m, int(r["ldc"]), device="cuda", dtype=self.dtype)[:, :n]
            bias = (
                torch.empty(n, device="cuda", dtype=self.dtype) if args.bias else None
            )
            assert (
                tuple(
                    contract(
                        a,
                        b,
                        out,
                        bias,
                        args.layout,
                        args.allow_atomic,
                        self.budget,
                        args.errRatio,
                    ).values()
                )
                == key
            )
            refs = []

            def inputs(seed, a=a, b=b, bias=bias):
                torch.manual_seed(seed)
                a.uniform_(-0.5, 0.5)
                b.uniform_(-0.5, 0.5)
                if bias is not None:
                    bias.uniform_(-0.25, 0.25)

            # The oracle uses FP32, TF32 disabled, no tested partial/reduce path.
            for seed in (args.seed, args.seed + 1):
                inputs(seed)
                ref = (a.float() @ b.float()).to(self.dtype)
                if bias is not None:
                    ref = (ref.float() + bias.float()).to(self.dtype)
                refs.append(ref)

            def error(y, ref):
                if not torch.isfinite(y).all():
                    return math.inf
                delta = y.float() - ref.float()
                rms = ref.float().square().mean().sqrt().item()
                relative = delta.norm().item() / max(ref.float().norm().item(), 1e-20)
                limit = self.limit
                if relative > limit or delta.abs().max().item() > max(
                    0.005, rms * (0.4 if args.allow_atomic else 0.05)
                ):
                    return math.inf
                return relative

            def call(c, a=a, b=b, out=out, bias=bias):
                return gemm_a16w16_tdm(
                    a, b, out, bias, layout=args.layout, **c.kwargs()
                )

            viable = []
            # Only constrained compile errors may be skipped; GPU/runtime errors
            # stop the tuner rather than continuing on a potentially poisoned device.
            for i, c in enumerate(cs):
                record = {
                    "shape": [m, n, k],
                    "dtype": str(self.dtype),
                    "config": c.kwargs(),
                    "workspace_bytes": c.workspace_bytes(m, n),
                }
                graph = None
                try:
                    inputs(args.seed)
                    try:
                        call(c)
                    except (ValueError, NotImplementedError) as exc:
                        record["rejected"] = str(exc)
                        continue
                    torch.cuda.synchronize()
                    e0 = error(out, refs[0])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(8):
                            call(c)
                    inputs(args.seed + 1)
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    e1 = error(out, refs[1])
                    err = max(e0, e1)
                    if not math.isfinite(err):
                        record["rejected"] = "numerical/changed-input gate"
                        continue
                    # AITER timer, captures complete call including workspace reduction.
                    _, us = run_perftest(
                        graph.replay,
                        num_warmup=args.warmup,
                        num_iters=args.iters,
                        use_cuda_event=True,
                        num_rotate_args=1,
                    )
                    us /= 8
                    if not math.isfinite(us) or us <= 0:
                        record["rejected"] = "invalid timing"
                        continue
                    record.update(screen_us=us, relative_l2=err)
                    viable.append([i, c, us, err, record])
                finally:
                    if graph is not None:
                        graph.reset()
                    self.audit.append(record)
                    self._save_audit()
            if not viable:
                raise RuntimeError(f"all candidates failed for {m, n, k}")

            # Retain fastest candidates plus fastest representative of every
            # reduction family, including no split. Randomize confirmation order.
            ranked = sorted(viable, key=lambda x: x[2])
            shortlist = ranked[: args.shortlist]
            for family in ("none", "workspace", "atomic"):
                group = [
                    x
                    for x in ranked
                    if ("none" if x[1].split_k == 1 else x[1].split_k_policy) == family
                ]
                if group and group[0] not in shortlist:
                    shortlist.append(group[0])
            samples = {x[0]: [] for x in shortlist}
            rng = random.Random(args.seed)
            inputs(args.seed)
            for _ in range(args.rounds):
                order = list(shortlist)
                rng.shuffle(order)
                for i, c, _, _, record in order:
                    call(c)
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph):
                            for _ in range(8):
                                call(c)
                        graph.replay()
                        torch.cuda.synchronize()
                        if not math.isfinite(error(out, refs[0])):
                            raise RuntimeError("confirmation numerical gate failed")
                        _, us = run_perftest(
                            graph.replay,
                            num_warmup=args.warmup,
                            num_iters=args.iters,
                            use_cuda_event=True,
                            num_rotate_args=1,
                        )
                        if not math.isfinite(us) or us <= 0:
                            raise RuntimeError("invalid confirmation timing")
                        samples[i].append(us / 8)
                    finally:
                        graph.reset()
            for i, c, _, err, record in shortlist:
                us = statistics.median(samples[i])
                record["confirmation_us"] = samples[i]
                record["median_us"] = us
                results.append(((key, i, c.split_k, c.encode()), us, err))
            self._save_audit()
            print(
                f"{m}x{n}x{k}: {len(cs)} candidates, {len(viable)} valid, {len(shortlist)} confirmed",
                flush=True,
            )
        return results


def main():
    tuner = TdmGemmTuner(
        "TdmGemmTuner",
        key=KEYS,
        description=__doc__ + "\n--errRatio is relative L2 for this tuner.",
    )
    args = tuner.parse_args()
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        tuner.run(args)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


if __name__ == "__main__":
    main()
