# SPDX-License-Identifier: MIT
"""Minimal first-touch protocol for the page-4 QSA logits kernel (gfx1250).

The v0 kernel's first GPU dispatch produced no output for 40 minutes and left
``amdgpu ... VM_PAGE_FAULT`` in dmesg, so the next GPU window must not repeat
"one big run, no output".  This driver runs a ladder of steps, **each in its own
process, with a hard external timeout (default 60 s)**, and prints

    STEP <name> START
    STEP <name> OK | FAIL | TIMEOUT | SKIP

with ``flush=True`` for every step, plus a tiny ``ft_<name>.json`` per step and a
``first_touch_summary.json`` at the end.  A step that hangs is therefore a
*labelled* hang with its own log file, and everything before it is proven good.

Steps, in order (see ``FIRST_TOUCH.md``):

    env                    versions + device enumeration (no tensor op)
    dev                    one trivial GPU tensor op (device health)
    compile                COMPILE_ONLY build of both dtypes (no dispatch)
    a_1row_1req_4col_bf16  1 row, 1 request, one 4-column page, bf16
    b_1row_1req_8col_bf16  2 pages / 8 columns
    c_4rows_1req_16col_bf16  4 rows (grid.y) / 16 columns
    d0_fp8_scale0          fp8 cache, per-token scale = 0  -> zeros in the window
    d1_fp8_scale1          fp8 cache, per-token scale = 1  -> golden match
    e_I5_min_decode_bf16   the existing I5 smoke case (identical to run_gpu.py)

Usage::

    python3 -u first_touch.py                 # run the whole ladder
    python3 -u first_touch.py --list
    python3 -u first_touch.py --step a_1row_1req_4col_bf16   # one worker (internal)
    python3 -u first_touch.py --dry-run       # validate the driver, touch no GPU
    python3 -u first_touch.py --timeout 45 --out /work/results/g2/builder/first_touch

Exit codes: 0 every step OK/SKIP; 2 a step FAILed; 3 a step TIMEOUT (device
suspect: stop, do not retry, check dmesg).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAGE = 4
HEADS = 4
HDIM = 128
GOLDEN_DIR = os.environ.get(
    "QSA_GOLDEN_DIR", "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline")

RC_OK, RC_FAIL, RC_SKIP, RC_TIMEOUT = 0, 2, 3, 4

STEPS = [
    ("env", "versions + device enumeration (no tensor op)"),
    ("dev", "one trivial GPU tensor op (device health)"),
    ("compile", "COMPILE_ONLY build of both dtypes (no dispatch, no device)"),
    ("a_1row_1req_4col_bf16", "1 row, 1 request, one 4-column page, bf16"),
    ("b_1row_1req_8col_bf16", "1 row, 1 request, 2 pages / 8 columns, bf16"),
    ("c_4rows_1req_16col_bf16", "4 rows (grid.y) x 16 columns, bf16"),
    ("d0_fp8_scale0", "fp8 cache, per-token scale = 0"),
    ("d1_fp8_scale1", "fp8 cache, per-token scale = 1"),
    ("e_I5_min_decode_bf16", "the existing I5 smoke case (run_gpu.py --stage smoke)"),
]
STEP_NAMES = [s[0] for s in STEPS]


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------
def _wlog(msg):
    print(msg, flush=True)


def _write_json(out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ft_{name}.json")
    payload = dict(payload)
    payload.setdefault("step", name)
    payload["pid"] = os.getpid()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def _golden():
    if GOLDEN_DIR not in sys.path:
        sys.path.insert(0, GOLDEN_DIR)
    import qsa_golden as G
    import qsa_shapes as S
    return G, S


def _tiny_fixture(rows, width, fmt, seed=0, scale_value=None, reqs=1):
    """Deterministic CPU fixture: `width` logical pages, one request.

    `pos = ctx - 1` with `ctx = 16 * width` puts the whole table inside the
    visible window (`visible = columns`), so every column is compared.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    pages = max(4, width)
    q = ((torch.rand((rows, HEADS, HDIM), generator=g) * 2 - 1)).to(torch.bfloat16)
    raw = (torch.rand((pages, PAGE, 1, HDIM), generator=g) * 2 - 1) * 3.0
    if fmt == "bf16":
        cache = raw.to(torch.bfloat16)
        scale = None
    else:
        if scale_value is not None:
            scale = torch.full((pages, PAGE, 1), float(scale_value), dtype=torch.float32)
            if float(scale_value) > 0:
                cache = (raw / float(scale_value)).to(torch.float8_e4m3fn)
            else:
                cache = raw.to(torch.float8_e4m3fn)
        else:
            s = raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
            cache = (raw / s.unsqueeze(-1)).to(torch.float8_e4m3fn)
            scale = s.to(torch.float32)
    table = torch.arange(width, dtype=torch.int32).reshape(1, width) % pages
    ctx_len = 16 * width
    token_to_request = torch.zeros(rows, dtype=torch.int32)
    if reqs > 1:
        token_to_request = (torch.arange(rows, dtype=torch.int32) % reqs)
        table = table.repeat(reqs, 1)
    query_positions = torch.full((rows,), ctx_len - 1, dtype=torch.int32)
    context_lens = torch.full((table.shape[0],), ctx_len, dtype=torch.int32)
    return {
        "q": q, "compressed_k_cache": cache, "compressed_k_scale": scale,
        "page_table": table, "token_to_request": token_to_request,
        "query_positions": query_positions, "context_lens": context_lens,
    }


def _run_kernel_case(name, data, out_dir, max_columns=None, note=""):
    """Dispatch one case, synchronise, compare against the CPU golden."""
    import torch

    G, S = _golden()
    from _load import load_kernel_module

    _wlog(f"STEP {name} fixture-cpu")
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], data["page_table"],
        data["token_to_request"], data["query_positions"], data["context_lens"],
        compress_ratio=S.COMPRESS_RATIO, max_columns=max_columns,
        compressed_k_scale=data["compressed_k_scale"])
    columns = int(gold.shape[1])
    _wlog(f"STEP {name} golden-cpu columns={columns} visible={gold_vis.tolist()}")

    if not torch.cuda.is_available():
        _wlog(f"STEP {name} SKIP no CUDA")
        _write_json(out_dir, name, {"ok": False, "skip": True, "reason": "no CUDA device"})
        return RC_SKIP

    K, how = load_kernel_module()
    _wlog(f"STEP {name} kernel-module {how}")

    dev = torch.device("cuda")
    d = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    torch.cuda.synchronize()
    _wlog(f"STEP {name} dispatch")
    t0 = time.time()
    out, out_vis = K.qsa_paged_mqa_logits_page4(
        d["q"], d["compressed_k_cache"], d["page_table"], d["token_to_request"],
        d["query_positions"], d["context_lens"], compress_ratio=PAGE,
        max_columns=columns, compressed_k_scale=d["compressed_k_scale"])
    _wlog(f"STEP {name} dispatched ({time.time() - t0:.1f}s) -> synchronize")
    torch.cuda.synchronize()
    _wlog(f"STEP {name} synchronized")

    out_cpu = out.float().cpu()
    vis_cpu = out_vis.cpu()
    res = G.compare_logits(out_cpu, vis_cpu, gold, gold_vis, valid)
    outside = {
        "outside_count": int((~valid).sum().item()),
        "nan_kernel": int(torch.isnan(out_cpu).sum().item()),
        "outside_not_neg_inf": int((~torch.isinf(out_cpu[~valid])).sum().item())
        if bool((~valid).any()) else 0,
    }
    payload = {"ok": bool(res.get("pass")), "note": note, "dtype": data["fmt"],
               "rows": int(data["q"].shape[0]), "columns": columns,
               "kernel_module": how, "golden": {k: v for k, v in res.items()},
               "outside": outside, "seconds": round(time.time() - t0, 2)}
    _write_json(out_dir, name, payload)
    if res.get("pass"):
        _wlog(f"STEP {name} compare PASS max_abs={res.get('max_abs_err')}")
        return RC_OK
    _wlog(f"STEP {name} compare FAIL {json.dumps(res, default=str)[:600]}")
    return RC_FAIL


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_env(out_dir, args):
    import torch

    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available())}
    try:
        import flydsl
        info["flydsl"] = getattr(flydsl, "__version__", "?")
        info["flydsl_file"] = getattr(flydsl, "__file__", "?")
    except Exception as exc:  # noqa: BLE001
        info["flydsl_error"] = f"{type(exc).__name__}: {exc}"
    if info["cuda_available"]:
        try:
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu": p.name, "arch": getattr(p, "gcnArchName", None),
                         "cu": p.multi_processor_count,
                         "mem_GB": round(p.total_memory / 1e9, 1)})
        except Exception as exc:  # noqa: BLE001
            info["device_props_error"] = f"{type(exc).__name__}: {exc}"
    _wlog("STEP env " + json.dumps(info, default=str))
    _write_json(out_dir, "env", {"ok": True, **info})
    return RC_OK


def step_dev(out_dir, args):
    """The post-mortem's "first GPU tensor op hangs" probe, on its own."""
    import torch

    if not torch.cuda.is_available():
        _write_json(out_dir, "dev", {"ok": False, "skip": True, "reason": "no CUDA device"})
        _wlog("STEP dev SKIP no CUDA")
        return RC_SKIP
    t0 = time.time()
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    y = x.sum().item()
    torch.cuda.synchronize()
    _wlog(f"STEP dev torch.zeros(64).sum() -> {y} in {time.time() - t0:.2f}s")
    _write_json(out_dir, "dev", {"ok": True, "sum": y, "seconds": round(time.time() - t0, 3)})
    return RC_OK


def step_compile(out_dir, args):
    """Both specializations must build before anything is dispatched."""
    os.environ.setdefault("COMPILE_ONLY", "1")
    import compile_only

    rc = compile_only.main()
    _write_json(out_dir, "compile", {"ok": rc == 0, "detail": "see compile_only.json"})
    return RC_OK if rc == 0 else RC_FAIL


def _deal_with(fmt, name, rows, width, out_dir, scale_value=None, note=""):
    data = _tiny_fixture(rows, width, fmt, scale_value=scale_value)
    data["fmt"] = fmt
    return _run_kernel_case(name, data, out_dir, note=note)


def step_a(out_dir, args):
    return _deal_with("bf16", "a_1row_1req_4col_bf16", 1, 1, out_dir)


def step_b(out_dir, args):
    return _deal_with("bf16", "b_1row_1req_8col_bf16", 1, 2, out_dir)


def step_c(out_dir, args):
    return _deal_with("bf16", "c_4rows_1req_16col_bf16", 4, 4, out_dir)


def step_d0(out_dir, args):
    return _deal_with("fp8", "d0_fp8_scale0", 4, 4, out_dir, scale_value=0.0,
                      note="k = E4M3 * 0 -> every in-window logit is 0.0")


def step_d1(out_dir, args):
    return _deal_with("fp8", "d1_fp8_scale1", 4, 4, out_dir, scale_value=1.0,
                      note="k = E4M3 * 1 -> golden match")


def step_e(out_dir, args):
    """Exactly the first case of ``run_gpu.py --stage smoke``."""
    import torch

    G, S = _golden()
    case = S.make_case("I5_min_decode_bf16", "decode", rows=4, requests=1,
                       max_ctx=256, index_dtype="bf16", table_width=16, cache_pages=64)
    data = G.build_case(case, device="cpu", seed=0)
    data["fmt"] = "bf16"
    return _run_kernel_case("e_I5_min_decode_bf16", data, out_dir,
                            max_columns=case.columns)


WORKERS = {
    "env": step_env, "dev": step_dev, "compile": step_compile,
    "a_1row_1req_4col_bf16": step_a,
    "b_1row_1req_8col_bf16": step_b,
    "c_4rows_1req_16col_bf16": step_c,
    "d0_fp8_scale0": step_d0,
    "d1_fp8_scale1": step_d1,
    "e_I5_min_decode_bf16": step_e,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_bounded(cmd, env, timeout, log_path):
    """Run `cmd` with a hard wall-clock timeout; kill the whole process group."""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=HERE)
        t0 = time.time()
        try:
            rc = proc.wait(timeout=timeout)
            return rc, round(time.time() - t0, 2), False
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.wait(timeout=5)
                    break
                except Exception:  # noqa: BLE001
                    continue
            return None, round(time.time() - t0, 2), True


def drive(steps, timeout, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": [],
               "timeout_s": timeout, "out_dir": os.path.abspath(out_dir)}
    worst = RC_OK
    for name in steps:
        desc = dict(STEPS).get(name, "")
        log_path = os.path.join(out_dir, f"ft_{name}.log")
        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        print(f"STEP {name} START  ({desc})", flush=True)
        t0 = time.time()
        rc, secs, timed_out = _run_bounded(
            [sys.executable, "-u", os.path.abspath(__file__), "--step", name,
             "--out", out_dir], env, timeout, log_path)
        json_path = os.path.join(out_dir, f"ft_{name}.json")
        payload = None
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:  # noqa: BLE001
                payload = None
        if timed_out:
            status = "TIMEOUT"
            worst = max(worst, RC_TIMEOUT)
        elif rc == RC_SKIP:
            status = "SKIP"
        elif rc == RC_OK or rc == 0:
            status = "OK"
        else:
            status = "FAIL"
            worst = max(worst, RC_FAIL)
        print(f"STEP {name} {status}  ({secs:.1f}s, rc={rc}"
              f"{', json=missing' if payload is None else ''})", flush=True)
        summary["steps"].append({
            "name": name, "desc": desc, "status": status, "rc": rc,
            "seconds": secs, "log": log_path, "json": payload,
        })
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["worst"] = ("TIMEOUT" if worst >= RC_TIMEOUT else
                        "FAIL" if worst >= RC_FAIL else "OK")
    with open(os.path.join(out_dir, "first_touch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("\nfirst-touch summary: %s  (%s)" % (summary["worst"],
                                               os.path.join(out_dir, "first_touch_summary.json")),
          flush=True)
    for s in summary["steps"]:
        print("  %-26s %-8s %6.1fs" % (s["name"], s["status"], s["seconds"]), flush=True)
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=STEP_NAMES, default=None,
                    help="run one worker step in this process (used by the driver)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="hard per-step timeout in seconds (default 60)")
    ap.add_argument("--out", default=os.environ.get("QSA_FT_OUT", os.path.join(HERE, "ft_out")))
    ap.add_argument("--only", default=None,
                    help="comma separated step names for the driver")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="drive the ladder once with a short timeout; touches no GPU "
                         "by itself (each step skips cleanly when there is no CUDA)")
    args = ap.parse_args(argv)

    if args.list:
        for name, desc in STEPS:
            print(f"{name:26s} {desc}")
        return RC_OK

    if args.step:
        fn = WORKERS[args.step]
        try:
            return fn(args.out, args)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            _write_json(args.out, args.step,
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-4000:]})
            _wlog(f"STEP {args.step} FAIL {type(exc).__name__}: {exc}")
            return RC_FAIL

    steps = STEP_NAMES
    if args.only:
        steps = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in steps if s not in STEP_NAMES]
        if unknown:
            raise SystemExit(f"unknown steps: {unknown}")
    timeout = 20.0 if args.dry_run else args.timeout
    out_dir = args.out or os.path.join(HERE, "ft_out")
    print(f"first-touch ladder: {len(steps)} steps, {timeout:.0f}s hard timeout each")
    print(f"output dir: {os.path.abspath(out_dir)}")
    worst = drive(steps, timeout, out_dir)
    if args.dry_run:
        # a dry run proves the driver bookkeeping, not the GPU
        print("dry-run: driver bookkeeping completed; GPU steps were skipped/failed "
              "as expected on this machine")
        return RC_OK
    return worst


if __name__ == "__main__":
    sys.exit(main())
