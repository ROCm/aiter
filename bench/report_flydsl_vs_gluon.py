#!/usr/bin/env python3
"""Report for run_flydsl_vs_gluon.sh: FlyDSL+HERD vs Triton(gluon)+HERD (+ FlyDSL no-HERD baseline).

Usage:
  report_flydsl_vs_gluon.py --results DIR [--flydsl-herd NAME] [--flydsl-noherd NAME] [--triton-herd NAME] [--out FILE]

DIR holds one sub-directory per config, each written by run_config.sh (run1..run4/bench.log,
server.log, run_config.log, flydsl_cache/). The FlyDSL no-HERD baseline is optional.
Only reads text files; does not touch the GPU.
"""
import argparse
import re
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

# (label, regex, higher_is_better)
METRICS = [
    ("Successful requests", r"Successful requests:\s+([\d.]+)", None),
    ("Benchmark duration (s)", r"Benchmark duration \(s\):\s+([\d.]+)", False),
    ("Total input tokens", r"Total input tokens:\s+([\d.]+)", None),
    ("Total generated tokens", r"Total generated tokens:\s+([\d.]+)", None),
    ("Request throughput (req/s)", r"Request throughput \(req/s\):\s+([\d.]+)", True),
    ("Output token throughput (tok/s)", r"Output token throughput \(tok/s\):\s+([\d.]+)", True),
    ("Total token throughput (tok/s)", r"Total Token throughput \(tok/s\):\s+([\d.]+)", True),
    ("Concurrency", r"Concurrency:\s+([\d.]+)", None),
    ("Mean TTFT (ms)", r"Mean TTFT \(ms\):\s+([\d.]+)", False),
    ("Median TTFT (ms)", r"Median TTFT \(ms\):\s+([\d.]+)", False),
    ("P99 TTFT (ms)", r"P99 TTFT \(ms\):\s+([\d.]+)", False),
    ("Mean TPOT (ms)", r"Mean TPOT \(ms\):\s+([\d.]+)", False),
    ("Median TPOT (ms)", r"Median TPOT \(ms\):\s+([\d.]+)", False),
    ("P99 TPOT (ms)", r"P99 TPOT \(ms\):\s+([\d.]+)", False),
    ("Mean ITL (ms)", r"Mean ITL \(ms\):\s+([\d.]+)", False),
    ("Median ITL (ms)", r"Median ITL \(ms\):\s+([\d.]+)", False),
    ("P99 ITL (ms)", r"P99 ITL \(ms\):\s+([\d.]+)", False),
    ("Mean E2EL (ms)", r"Mean E2EL \(ms\):\s+([\d.]+)", False),
    ("Median E2EL (ms)", r"Median E2EL \(ms\):\s+([\d.]+)", False),
    ("P99 E2EL (ms)", r"P99 E2EL \(ms\):\s+([\d.]+)", False),
]
INTEGER_METRICS = {"Successful requests", "Total input tokens", "Total generated tokens"}
KEY_METRICS = [
    "Output token throughput (tok/s)",
    "Benchmark duration (s)",
    "Mean TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Mean TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean E2EL (ms)",
]
STATUS = {"run1": "cold", "run2": "clean", "run3": "clean", "run4": "clean"}
RUN_LABELS = {"run1": "冷启动", "run2": "预热后", "run3": "预热后", "run4": "预热后"}
CLEAN_RUNS = [r for r, s in STATUS.items() if s == "clean"]

ENV_TABLE = {
    "FlyDSL+HERD": ["0", "1", "1(原脚本, FlyDSL 路径不读)", "serve_dsv4_tp1_flydsl_herd.sh"],
    "FlyDSL 无HERD": ["0", "未设(0)", "1(原脚本, FlyDSL 路径不读)", "serve_dsv4_tp1.sh"],
    "Triton+HERD": ["1", "未设(0)", "1", "serve_dsv4_tp1_triton_herd.sh"],
}


def disp_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def pad(s, width, align="l"):
    gap = width - disp_width(s)
    return " " * gap + s if align == "r" else s + " " * gap


def render_table(headers, rows, align):
    widths = [max(disp_width(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells):
        return "  ".join(pad(str(c), widths[i], align[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


def num(v, label):
    if v is None:
        return "-"
    return f"{v:,.0f}" if label in INTEGER_METRICS else f"{v:,.2f}"


def signed(v, label):
    return f"{v:+,.0f}" if label in INTEGER_METRICS else f"{v:+,.2f}"


def pct(a, b):
    return (b - a) / a * 100


def verdict(a, b, hib):
    if hib is None or not a or b is None:
        return "-"
    p = pct(a, b)
    if abs(p) < 0.5:
        return "持平"
    return "更好" if ((p > 0) == hib) else "更差"


def parse_bench(path):
    text = path.read_text(errors="ignore")
    out = {}
    for label, rx, _ in METRICS:
        m = re.search(rx, text)
        out[label] = float(m.group(1)) if m else None
    return out


class Config:
    def __init__(self, root, name, label):
        self.dir = root / name
        self.name = name
        self.label = label
        self.runs = {r: parse_bench(self.dir / r / "bench.log") for r in STATUS if (self.dir / r / "bench.log").exists()}
        self.clean = self._agg(CLEAN_RUNS)
        self.cold = self._agg(["run1"])
        self.log = (self.dir / "run_config.log").read_text(errors="ignore") if (self.dir / "run_config.log").exists() else ""
        self.server = (self.dir / "server.log").read_text(errors="ignore") if (self.dir / "server.log").exists() else ""

    def _agg(self, keep):
        out = {}
        for label, _, _ in METRICS:
            vals = [self.runs[r][label] for r in keep if r in self.runs and self.runs[r][label] is not None]
            out[label] = (statistics.mean(vals) if vals else None, statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        return out

    def field(self, pattern):
        m = re.search(pattern, self.log)
        return re.sub(r"/(aiter|atom)/__init__\.py$", "", m.group(1).strip()) if m else "?"

    def herd_route_tokens(self):
        return sorted({int(t) for t in re.findall(r"FlyDSL HERD route: tokens=(\d+)", self.server)})

    def herd_kernels(self):
        d = self.dir / "flydsl_cache"
        if not d.exists():
            return []
        return sorted({re.sub(r"_[0-9a-f]{32}$", "", p.name) for p in d.iterdir() if "herd" in p.name})

    def triton_moe(self):
        return "Triton MoE experts:" in self.server

    def moe_buckets(self):
        return sorted({int(t) for t in re.findall(r"grouped CSV config selected token=(\d+)", self.server)})


def diff_rows(base, new):
    rows = []
    for label, _, hib in METRICS:
        a, b = base[label][0], new[label][0]
        if a is None or b is None:
            continue
        d = b - a
        rows.append([label, num(a, label), num(b, label), signed(d, label), f"{pct(a, b):+.2f}%", verdict(a, b, hib)])
    return rows


def bench_params(cfg):
    p = cfg.dir / "run2" / "bench.log"
    if not p.exists():
        return "?"
    text = p.read_text(errors="ignore")
    get = {k: (re.search(rf"{k}=(\d+)", text) or [None, "?"])[1] for k in ("random_input_len", "random_output_len", "max_concurrency", "num_prompts")}
    return (f"ISL={get['random_input_len']}, OSL={get['random_output_len']}, "
            f"CONC={get['max_concurrency']}, {get['num_prompts']} 请求")


def write(root, names, out_path):
    fh = Config(root, names["flydsl_herd"], "FlyDSL+HERD")
    th = Config(root, names["triton_herd"], "Triton+HERD")
    fn = Config(root, names["flydsl_noherd"], "FlyDSL 无HERD") if names.get("flydsl_noherd") else None
    if fn is not None and not fn.runs:
        fn = None
    configs = [c for c in (fh, fn, th) if c is not None]
    base = fn or fh
    o, t, f = "Output token throughput (tok/s)", "Mean TPOT (ms)", "Mean TTFT (ms)"
    date = datetime.fromtimestamp((fh.dir / "run1" / "bench.log").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    out = []
    add = out.append
    add("FlyDSL MoE vs Triton/gluon MoE e2e 对比 (HERD 模式), DeepSeek-V4-Flash")
    add("=" * 72)
    add(f"数据时间  : {date}")
    add(f"GPU       : {fh.field(r'gfx=(\S+)')} x1, TP1")
    add(f"aiter     : {fh.field(r'aiter=(\S+)')}  commit {fh.field(r'aiter=\S+ commit=(\S+)')}")
    add(f"ATOM      : {fh.field(r'atom=(\S+)')}  commit {fh.field(r'atom=\S+ commit=(\S+)')}, "
        f"moe.py 有改动(ATOM#2713 补丁): {fh.field(r'moe_py_modified=(\S+)')}")
    add(f"版本      : {fh.field(r'versions: (.*)')}")
    add(f"模型      : {fh.field(r'model=(\S+)')}")
    add(f"bench     : bench_dsv4.sh ({bench_params(fh)})")
    add(f"结果目录  : {root}")
    add("")
    add(render_table(
        ["配置", "ATOM_USE_TRITON_MOE", "AITER_FLYDSL_USE_HERD", "AITER_TRITON_USE_HERD", "启动脚本", "结果子目录"],
        [[c.label, *ENV_TABLE[c.label], c.name] for c in configs],
        ["l", "c", "c", "c", "l", "l"],
    ))
    add("  其余环境变量与 /app/scripts/dsv4/serve_dsv4.sh 相同, -tp 4 改为 -tp 1; ATOM_USE_TRITON_MOE_DECODE=1 来自原脚本")
    add("")
    add("FlyDSL HERD 生效的前提")
    add("  aiter#6527: fused_moe(herd_topk=K) 接收每个 token 的 K+1 个候选, 在 gfx1250 grouped MoE 的 route kernel 里")
    add("              丢掉 batch 内最冷门的那个, 再对保留的 K 个 renormalize 并乘 routed_scaling_factor")
    add("  ATOM#2713 : decode 步 (token 数 16..128) 用 Top-(K+1) 路由并给 fused_moe 传 herd_topk; 不打这个补丁,")
    add("              AITER_FLYDSL_USE_HERD=1 在 gfx1250 上没有任何效果 (aiter 的自动 HERD 只在 gfx950 上开)")
    add("")
    add("路径与 HERD 校验 (来自 server 日志和 FlyDSL kernel 缓存)")
    for c in configs:
        toks = c.herd_route_tokens()
        decode = [b for b in c.moe_buckets() if b <= 512]
        add(f"  {pad(c.label, 14)} HERD route 的 decode token 数: {toks if toks else '无'}; "
            f"HERD kernel: {c.herd_kernels() if c.herd_kernels() else '无'}; "
            f"decode MoE: {'Triton/gluon' if c.triton_moe() and not decode else 'FlyDSL'}")
    add("")

    add("结论 (稳态, 预热后 run2-4 均值)")
    if fn:
        add(f"  HERD 收益 (FlyDSL+HERD vs FlyDSL 无HERD): 输出吞吐 {fn.clean[o][0]:,.2f} -> {fh.clean[o][0]:,.2f} tok/s "
            f"({pct(fn.clean[o][0], fh.clean[o][0]):+.2f}%), Mean TPOT {fn.clean[t][0]:.2f} -> {fh.clean[t][0]:.2f} ms "
            f"({pct(fn.clean[t][0], fh.clean[t][0]):+.2f}%), Mean TTFT {fn.clean[f][0]:.2f} -> {fh.clean[f][0]:.2f} ms "
            f"({pct(fn.clean[f][0], fh.clean[f][0]):+.2f}%)")
    add(f"  FlyDSL+HERD vs Triton+HERD: 输出吞吐 {fh.clean[o][0]:,.2f} vs {th.clean[o][0]:,.2f} tok/s "
        f"(FlyDSL {pct(th.clean[o][0], fh.clean[o][0]):+.2f}%), Mean TPOT {fh.clean[t][0]:.2f} vs {th.clean[t][0]:.2f} ms "
        f"(FlyDSL {pct(th.clean[t][0], fh.clean[t][0]):+.2f}%)")
    add("")

    sec = 0
    sections = "一二三四五六七"
    if fn:
        add("")
        add(f"{sections[sec]}、稳态对比: FlyDSL+HERD vs FlyDSL 无HERD (|delta%|<0.5% 记为持平)")
        sec += 1
        add("-" * 72)
        add(render_table(["Metric", "FlyDSL 无HERD", "FlyDSL+HERD", "delta", "delta%", "HERD 结论"],
                         diff_rows(fn.clean, fh.clean), ["l", "r", "r", "r", "r", "l"]))
        add("")
    add("")
    add(f"{sections[sec]}、稳态对比: FlyDSL+HERD vs Triton+HERD (delta 以 Triton+HERD 为基准)")
    sec += 1
    add("-" * 72)
    add(render_table(["Metric", "Triton+HERD", "FlyDSL+HERD", "delta", "delta%", "FlyDSL 结论"],
                     diff_rows(th.clean, fh.clean), ["l", "r", "r", "r", "r", "l"]))
    add("")

    add("")
    add(f"{sections[sec]}、关键指标汇总 (稳态均值, 括号内为相对 {base.label} 的变化)")
    sec += 1
    add("-" * 72)
    rows = []
    for label in KEY_METRICS:
        row = [label]
        for c in configs:
            v = c.clean[label][0]
            row.append(num(v, label) if c is base else f"{num(v, label)} ({pct(base.clean[label][0], v):+.2f}%)")
        rows.append(row)
    add(render_table(["Metric", *[c.label for c in configs]], rows, ["l", *["r"] * len(configs)]))
    add("")

    add("")
    add(f"{sections[sec]}、冷启动首轮 (起 server 后第一次 bench, 含首次 JIT 编译, 不代表稳态性能)")
    sec += 1
    add("-" * 72)
    add(render_table(["Metric", *[f"{c.label} run1" for c in configs]],
                     [[label, *[num(c.cold[label][0], label) for c in configs]] for label in KEY_METRICS],
                     ["l", *["r"] * len(configs)]))
    add("")

    add("")
    add(f"{sections[sec]}、逐轮原始数据 (run1=冷启动; run2-4=预热后)")
    sec += 1
    for c in configs:
        run_names = list(c.runs)
        headers = ["Metric", *[f"{r}({RUN_LABELS[r]})" for r in run_names], "预热后均值", "预热后std"]
        rows = []
        for label, _, _ in METRICS:
            mean, std = c.clean[label]
            rows.append([label, *[num(c.runs[r][label], label) for r in run_names], num(mean, label), f"{std:,.2f}"])
        add("")
        add(f"[{c.label}]  {c.dir}")
        add(render_table(headers, rows, ["l", *["r"] * (len(headers) - 1)]))
    add("")

    add("")
    add(f"{sections[sec]}、测试过程与注意事项")
    add("-" * 72)
    add("  流程: 每个配置独立起 server -> run1 冷启动 -> 预热 (k 个并发 1024-token 请求, k=1..16) -> run2-4 -> 停 server")
    add("  预热后 3 轮期间新编译的 kernel 数 (应为 0):")
    for c in configs:
        add(f"        {pad(c.label, 14)} FlyDSL 新 kernel {c.field(r'new FlyDSL kernels during clean runs: (\d+)')}, "
            f"Triton 缓存新增 {c.field(r'new Triton cache entries during clean runs: (\d+)')}")
    add("  精度: HERD 会改变路由 (top-(K+1) 里丢掉 batch 内最冷门的 expert), 采用前需要跑精度评测;")
    add("        FlyDSL HERD 不作用于 DSV4 前 3 层 hash 层; Triton MoE 分支在 decode 时不走 hash 查表")
    add("")
    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory holding the per-config result dirs")
    ap.add_argument("--flydsl-herd", default="flydsl_herd")
    ap.add_argument("--flydsl-noherd", default="flydsl_noherd", help="optional baseline; '' to skip")
    ap.add_argument("--triton-herd", default="triton_herd")
    ap.add_argument("--out", help="report path (default: <results>/flydsl_vs_gluon.txt)")
    args = ap.parse_args()
    root = Path(args.results).resolve()
    write(root,
          {"flydsl_herd": args.flydsl_herd, "flydsl_noherd": args.flydsl_noherd, "triton_herd": args.triton_herd},
          args.out or root / "flydsl_vs_gluon.txt")
