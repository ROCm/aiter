#!/usr/bin/env python3
"""Generate comparison charts for Confluence from results CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).resolve().parent
RESULTS = BENCH_DIR / "results"
OUT = BENCH_DIR / "confluence_assets"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_scenario(
    rows: list[dict],
    *,
    x_col: str,
    title: str,
    out_path: Path,
    xlabel: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, batch in zip(axes, ("2048", "4096")):
        sub = [r for r in rows if r["batch"] == batch]
        sub.sort(key=lambda r: int(r[x_col]))
        x = [int(r[x_col]) for r in sub]
        ck = [float(r["ck_pr_6764(ms)"]) for r in sub]
        jax = [float(r["jax_unfused(ms)"]) for r in sub]
        ax.plot(x, ck, "o-", label="ck_pr_6764", linewidth=2, markersize=5)
        ax.plot(x, jax, "s-", label="jax_unfused", linewidth=2, markersize=5)
        ax.set_title(f"batch={batch}")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("ms (mean step time)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    specs = [
        ("scenario1", "fwd", "s_kv", "Scenario 1 forward — max logical length P", "P (max seq len)"),
        ("scenario1", "bwd", "s_kv", "Scenario 1 backward — uniform P", "P"),
        ("scenario2", "fwd", "s_kv", "Scenario 2 forward — sq=1, max KV len P", "P (max KV len)"),
        ("scenario2", "bwd", "s_kv", "Scenario 2 backward — s_q=1, s_kv=P", "P"),
        ("scenario3_4", "fwd", "s_q", "Scenario 3+4 forward — fixed self-attn", "seq len"),
        ("scenario3_4", "bwd", "s_q", "Scenario 3+4 backward — fixed self-attn", "seq len"),
    ]
    for scen, kind, xcol, title, xlab in specs:
        rows = load_csv(RESULTS / scen / f"{kind}.csv")
        plot_scenario(
            rows,
            x_col=xcol,
            title=title,
            out_path=OUT / f"{scen}_{kind}.png",
            xlabel=xlab,
        )
        print("wrote", OUT / f"{scen}_{kind}.png")


if __name__ == "__main__":
    main()
