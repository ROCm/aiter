"""Clock-normalised summary of the replicated runs.

Reads rep<gpu>.log (raw output of run_replicated.sh) or replicated_gpu<gpu>.jsonl
(the committed copy in results/; *.log is gitignored).

Each GPU's effective shader clock is derived from its raw-WMMA reference time
(WMMA cycles per instruction are fixed), anchored to one measured clock
(shader_clock on one GPU). Times are then reported as cycles and as projected
microseconds at the rated clock. Only clock-bound work projects this way;
memory latency does not scale with the shader clock.

  python3 analyze_replicated.py --logs <dir> --ref-gpu 0 --ref-mhz 65 --rated-mhz 2400
"""

import argparse
import json
import os
import statistics as st

ORDER = [
    "baseline",
    "compute_only",
    "no_store",
    "stage_no_global_store",
    "load_only",
    "store_only",
    "skeleton",
    "epi_wait_exact",
    "store_mode1",
    "bk128_sm1",
    "bk128_sm1_compute_only",
    "mla_baseline",
    "mla_compute_only",
    "mla_no_store",
    "mla_epi_wait_exact",
    "mla_bk128_sm1",
]


def records(logs, g):
    for name in (f"rep{g}.log", f"replicated_gpu{g}.jsonl"):
        path = os.path.join(logs, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                line = line.removeprefix("RESULT ")
                if line.startswith("{"):
                    yield json.loads(line)
        return
    raise FileNotFoundError(f"no rep{g}.log or replicated_gpu{g}.jsonl in {logs}")


def load(logs, gpus):
    rows, refs = {}, {}
    for g in gpus:
        wmma = []
        for r in records(logs, g):
            if "kind" in r:
                wmma.append(r["us"])
            else:
                rows.setdefault(r["config"].get("X_TAG"), {})[g] = r
        refs[g] = st.mean(wmma)
    return rows, refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--ref-gpu", type=int, default=0)
    p.add_argument(
        "--ref-mhz", type=float, required=True, help="measured shader MHz on --ref-gpu"
    )
    p.add_argument("--rated-mhz", type=float, default=2400.0)
    a = p.parse_args()

    rows, refs = load(a.logs, a.gpus)
    mhz = {g: a.ref_mhz * refs[a.ref_gpu] / refs[g] for g in refs}
    print("raw-WMMA reference us:", {g: round(v, 1) for g, v in refs.items()})
    print("effective shader MHz:", {g: round(v, 1) for g, v in mhz.items()})
    print(
        f"{'variant':26s} {'us per GPU':30s} {'x baseline':>12s} {'kcycles':>9s} {'us @rated':>11s}"
    )
    for tag in ORDER:
        d = rows.get(tag)
        if not d:
            continue
        base = rows["mla_baseline" if tag.startswith("mla_") else "baseline"]
        gs = sorted(d)
        ratio = [d[g]["us"] / base[g]["us"] for g in gs]
        cyc = [d[g]["us"] * mhz[g] for g in gs]
        rated = [c / a.rated_mhz for c in cyc]
        print(
            f"{tag:26s} {[round(d[g]['us']) for g in gs]!s:30s} "
            f"{st.mean(ratio):6.2f}±{st.pstdev(ratio):.2f} {st.mean(cyc) / 1e3:9.1f} "
            f"{st.mean(rated):7.1f}±{st.pstdev(rated):.1f}"
        )
    # wmma_peak --quick runs 120 iterations = 10x the kernel's per-wave WMMA count
    wm = [refs[g] / 10 * mhz[g] for g in refs]
    print(
        f"{'raw WMMA, kernel count':26s} {'':30s} {'':>12s} {st.mean(wm) / 1e3:9.1f} {st.mean(wm) / a.rated_mhz:7.1f}"
    )


if __name__ == "__main__":
    main()
