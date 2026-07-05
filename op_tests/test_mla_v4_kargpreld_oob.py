# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Guard-page out-of-bounds (OOB) memory detector for the gfx1250 v4 'nm' MLA
decode asm kernel (mla_decode_fwd_v4_nm), generalizing the single-buffer /
single-shape probe in test_mla_v4_nm.py:

  test_mla_v4_nm.py  -> only gqa=128 (tg_idx=1), only the `qrope` buffer.
  THIS file          -> ALL tested kernel variants  {(16,4),(64,1),(128,1)}
                        x ALL device buffers the kernel reads AND writes.

Principle (same as test_mla_v4_nm.py's guard page, made general):
  ROCm 7.x has no compute-sanitizer, so we synthesize a guard page. Every
  buffer the kernel touches is re-allocated so its DATA ends exactly on a
  2 MiB boundary (the tail of its own tight, non-caching allocation). Any
  read OR write even one byte past a buffer's logical end crosses into the
  next (unmapped) page and raises a GPU memory-access fault. This catches the
  class of "OOB but numerically silent / non-faulting on small tensors" bugs
  that accuracy / perf / plain-crash checks all miss.

  Because a GPU fault is unrecoverable (it would abort the whole process /
  pytest session), each kernel launch runs in a SUBPROCESS. A clean exit 0 +
  "COMPLETED no fault" means no OOB; a nonzero exit / "Memory access fault" /
  "HSA_STATUS_ERROR" means the kernel over-ran a buffer.

Buffers guarded (every device pointer handed to mla_decode_v4_asm):
  reads : q, qrope, kv_buffer, kvrope, qo_indptr, kv_indptr,
          kv_page_indices, kv_last_page_lens, split_indptr, sink
  writes: logits, attn_lse, output

Usage (run INSIDE the ff_mla container, gfx1250):
  # default edge-focused sweep, all buffers guarded at once:
  ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld_oob.py

  # on any fault, re-run per-buffer to name the culprit buffer:
  ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld_oob.py --localize

  # widen / narrow the sweep:
  ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld_oob.py \
      --variant qh128-q1-16mx4-64nx1-np -b 256 -c 61 131 323 --split-kv 1 2

  # run the FULL test_mla_v4_kargpreld.py grid (slow: one subprocess/combo):
  ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld_oob.py --full
"""

import argparse
import itertools
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import aiter
import aiter.mla  # noqa: F401  (explicit; main no longer auto-imports submodules)
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

import test_mla_v4_kargpreld as T

torch.set_default_device("cuda")

SUPPORTED_GFX = T.SUPPORTED_GFX  # ["gfx1250"]

# 2 MiB: the ROCm coarse-grain allocation granularity. Padding every buffer up
# to a whole number of 2 MiB pages and placing its data at the tail puts the
# logical end on a page boundary, so the first OOB byte lands in the (separate,
# almost-always-unmapped) next allocation and faults.
_PAGE = 2 * 1024 * 1024

# Every device pointer mla_decode_v4_asm receives (see aiter/mla.py). "all"
# guards every one at once (any-buffer OOB -> fault); --localize re-runs each
# in isolation to name the culprit.
_BUFFERS = [
    "q",
    "qrope",
    "kv_buffer",
    "kvrope",
    "qo_indptr",
    "kv_indptr",
    "kv_page_indices",
    "kv_last_page_lens",
    "split_indptr",
    "sink",
    "logits",
    "attn_lse",
    "output",
]


# ---------------------------------------------------------------------------
# Guard-page allocator.
# ---------------------------------------------------------------------------
def _tail_guarded(t):
    """Return a copy of `t` (same shape/dtype/contents) whose storage ends
    exactly on a 2 MiB boundary, so any read/write past its logical end hits
    the unmapped next page and faults the GPU.

    Relies on PYTORCH_NO_HIP_MEMORY_CACHING=1 so the padded buffer is its own
    tight hipMalloc (page-aligned base AND size); the tail slice then abuts the
    allocation end.
    """
    t = t.contiguous()
    n = t.numel()
    if n == 0:
        return t
    s = t.element_size()
    nbytes = n * s
    pad_bytes = ((nbytes + _PAGE - 1) // _PAGE) * _PAGE
    pad_elems = pad_bytes // s
    full = torch.empty(pad_elems, dtype=t.dtype, device=t.device)
    flat = full[pad_elems - n :]  # [n], contiguous, ends at the page boundary
    flat.copy_(t.reshape(-1))
    return flat.view(t.shape)  # shares `full`'s storage -> kept alive by view


def _maybe_guard(t, name, guard_set):
    return _tail_guarded(t) if name in guard_set else t.contiguous()


# ---------------------------------------------------------------------------
# Subprocess worker body: build one config, guard the requested buffers, run.
# ---------------------------------------------------------------------------
def _oob_worker_main(gqa, q_seq_logical, batch, kv_seq_lens, num_kv_splits,
                     out_16_nosplit, guard_set, inject="none"):
    device = "cuda"
    sm_scale = 1.0 / (T._QUANT_D**0.5)  # kernel ignores it; ABI needs a value

    inp = T._build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv_seq_lens,
        q_seq_logical=q_seq_logical,
        seed=T._SEED,
        gqa_ratio=gqa,
        attn_sink=True,
    )
    q_packed, q_rope = T._native_to_2buff_for_asm(inp["q_bf16"])
    kv_packed, kv_rope = T._native_to_2buff_for_asm(inp["kv_bf16"])

    # Self-test hook: deliberately drop the LAST kv page so the kernel's
    # legitimate read of the highest page index lands exactly on the guarded
    # tail's page boundary -> must fault. Proves the guard page is live (not a
    # silent always-pass). kv_page_indices still references the dropped page.
    if inject == "kv_short":
        kv_packed = kv_packed[:-1].contiguous()
        kv_rope = kv_rope[:-1].contiguous()

    total_q = inp["q_bf16"].size(0)
    num_seqs = inp["qo_indptr"].size(0) - 1
    num_heads = T.NUM_KV_HEADS * gqa

    output_buf = torch.empty(
        (total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=device
    )
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=device,
    )
    logits_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, T.V_HEAD_DIM),
        dtype=dtypes.fp32,
        device=device,
    )
    lse_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, 1), dtype=dtypes.fp32, device=device
    )

    kwargs = dict(
        q=_maybe_guard(q_packed, "q", guard_set),
        qrope=_maybe_guard(q_rope, "qrope", guard_set),
        kv_buffer=_maybe_guard(kv_packed, "kv_buffer", guard_set),
        kvrope=_maybe_guard(kv_rope, "kvrope", guard_set),
        output=_maybe_guard(output_buf, "output", guard_set),
        qo_indptr=_maybe_guard(inp["qo_indptr"], "qo_indptr", guard_set),
        kv_indptr=_maybe_guard(inp["kv_indptr"], "kv_indptr", guard_set),
        kv_page_indices=_maybe_guard(
            inp["kv_page_indices"], "kv_page_indices", guard_set
        ),
        kv_last_page_lens=_maybe_guard(
            inp["kv_last_page_lens"], "kv_last_page_lens", guard_set
        ),
        split_indptr=_maybe_guard(split_indptr, "split_indptr", guard_set),
        max_seqlen_q=inp["max_seqlen_q"],
        sink=_maybe_guard(inp["sink"], "sink", guard_set),
        sm_scale=sm_scale,
        out_16_nosplit=int(out_16_nosplit),
        num_kv_splits=num_kv_splits,
        logits=_maybe_guard(logits_buf, "logits", guard_set),
        attn_lse=_maybe_guard(lse_buf, "attn_lse", guard_set),
    )

    aiter.mla.mla_decode_fwd_v4_nm(**kwargs)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Config enumeration.
# ---------------------------------------------------------------------------
def _valid_split(kv_seq_lens, num_kv_splits):
    """Mirror test_mla_v4_kargpreld's guard: the .co inner KV loop processes
    pass_size=16 tokens/iter, so the smallest split must be >= 16 (page_size=1),
    else that split tail-drops (a correctness, not OOB, concern -> skip)."""
    if num_kv_splits <= 1:
        return True
    return (kv_seq_lens // num_kv_splits) >= 16


def _enumerate_configs(variants, batches, ctx_lens, splits):
    """Yield (gqa, qlen, batch, kv, split, o16) for every shipped, valid combo.
    o16=1 is added only where split==1 (bf16-direct is single-pass only)."""
    for name in variants:
        v = T._gfx1250_VARIANT_BY_KEY_NAME[name]
        gqa, qlen = v.nhead, v.decode_qlen
        if (gqa, qlen) not in T._SHIPPED_TILE_VARIANTS:
            continue
        for batch, kv, split in itertools.product(batches, ctx_lens, splits):
            if not _valid_split(kv, split):
                continue
            yield (gqa, qlen, batch, kv, split, 0)
            if split == 1:
                yield (gqa, qlen, batch, kv, split, 1)


# ---------------------------------------------------------------------------
# Parent: launch one guard-page worker subprocess per (config, guard_set).
# ---------------------------------------------------------------------------
def _run_worker(cfg, guard_spec, timeout=300, inject="none"):
    """Run a single guard-page probe in a subprocess. Returns (ok, combined_out).
    `guard_spec` is "all", "none", or a single buffer name."""
    gqa, qlen, batch, kv, split, o16 = cfg
    env = dict(os.environ)
    env["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"  # each buffer -> own tight hipMalloc
    env["HSA_XNACK"] = "0"  # page fault instead of demand-paging the OOB read
    env["AMD_LOG_LEVEL"] = "0"  # suppress noisy fault dumps
    _op_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_op_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [_repo_root, _op_dir, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    def _no_core_dump():
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    argv = [
        sys.executable, __file__, "--oob-worker",
        str(gqa), str(qlen), str(batch), str(kv), str(split), str(o16),
        guard_spec, inject,
    ]
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_no_core_dump,
        )
    except subprocess.TimeoutExpired as e:
        # A GPU memory-access fault can wedge the HSA queue so the worker hangs
        # at synchronize() instead of aborting. subprocess.run has already
        # killed the child; treat the hang as a (probable) OOB fault.
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return False, out + f"\n[timeout/hang after {timeout}s -> treated as FAULT]"

    combined = proc.stdout + proc.stderr
    faulted = (
        "Memory access fault" in combined
        or "HSA_STATUS_ERROR" in combined
        or "page fault" in combined.lower()
        or proc.returncode != 0
    )
    ok = (not faulted) and ("COMPLETED no fault" in combined)
    return ok, combined


def _cfg_label(cfg):
    gqa, qlen, batch, kv, split, o16 = cfg
    return (
        f"gqa={gqa:<3} qlen={qlen} batch={batch:<4} kv={kv:<6} "
        f"split={split} o16={o16}"
    )


def run_sweep(variants, batches, ctx_lens, splits, localize=False, timeout=300):
    configs = list(_enumerate_configs(variants, batches, ctx_lens, splits))
    print(
        f"[oob] gfx={get_gfx()}  configs={len(configs)}  "
        f"buffers_guarded=all({len(_BUFFERS)})  localize={localize}"
    )
    print("-" * 88)

    faults = []
    for cfg in configs:
        ok, out = _run_worker(cfg, "all", timeout=timeout)
        status = "OK  " if ok else "FAULT"
        print(f"[{status}] {_cfg_label(cfg)}")
        if not ok:
            tail = "\n    ".join(out.strip().splitlines()[-6:])
            print(f"    --- worker tail ---\n    {tail}")
            culprit = None
            if localize:
                culprit = _localize(cfg, timeout=timeout)
            faults.append((cfg, culprit))

    print("-" * 88)
    if not faults:
        print(f"[oob] PASS: no OOB across {len(configs)} configs (all buffers).")
        return True

    print(f"[oob] FAIL: {len(faults)} of {len(configs)} configs faulted:")
    for cfg, culprit in faults:
        extra = f"  -> buffer(s): {culprit}" if culprit else ""
        print(f"    {_cfg_label(cfg)}{extra}")
    return False


def _localize(cfg, timeout=300):
    """A config faulted with all buffers guarded; re-run guarding ONE buffer at
    a time to name which buffer(s) the kernel over-runs."""
    culprits = []
    for name in _BUFFERS:
        ok, _ = _run_worker(cfg, name, timeout=timeout)
        if not ok:
            culprits.append(name)
    return ",".join(culprits) if culprits else "unknown(none-in-isolation)"


def _run_selftest(timeout=300):
    """Prove the guard page actually faults on a real OOB (not a silent
    always-pass). Uses gqa=64 as a representative shipped variant."""
    cfg = (64, 1, 64, 131, 1, 0)  # gqa, qlen, batch, kv, split, o16
    print(f"[oob][selftest] cfg: {_cfg_label(cfg)}")

    ok_short, out_short = _run_worker(cfg, "all", timeout=timeout, inject="kv_short")
    print(f"[oob][selftest] kv_buffer short-by-one-page -> "
          f"{'FAULT (good)' if not ok_short else 'NO FAULT (BAD: guard is dead)'}")

    ok_full, _ = _run_worker(cfg, "all", timeout=timeout, inject="none")
    print(f"[oob][selftest] same config, unmodified       -> "
          f"{'OK (good)' if ok_full else 'FAULT (BAD: false positive)'}")

    passed = (not ok_short) and ok_full
    if ok_short:  # the short run should have faulted but didn't -> show why
        tail = "\n    ".join(out_short.strip().splitlines()[-6:])
        print(f"    --- short-run worker tail ---\n    {tail}")
    print(f"[oob][selftest] {'PASS: guard page is live.' if passed else 'FAIL.'}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# main.
# ---------------------------------------------------------------------------
def main():
    if get_gfx() not in SUPPORTED_GFX:
        print(f"[oob] skip: v4 nm shipped only for {SUPPORTED_GFX}; "
              f"current device is {get_gfx()}. Exiting 0.")
        return 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Guard-page OOB detector for the v4 nm asm kernel: all "
        "variants x all read/write buffers.",
    )
    parser.add_argument(
        "--variant",
        nargs="*",
        choices=[v.name for v in T._gfx1250_KERNEL_VARIANTS],
        default=[v.name for v in T._gfx1250_KERNEL_VARIANTS],
        help="Kernel variant name(s) to probe (default: all shipped).",
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[256],
        help="Batch size(s). Large batch => big buffers => OOB reaches farther.",
    )
    parser.add_argument(
        "-c", "--kv-seq-lens", type=int, nargs="*",
        default=[61, 131, 323, 1024],
        help="KV context length(s). Odd/boundary values stress tail handling.",
    )
    parser.add_argument(
        "--split-kv", type=int, nargs="*", default=[1, 2, 4],
        help="num_kv_splits value(s) (invalid small splits auto-skipped).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full test_mla_v4_kargpreld.py grid (slow: 1 subprocess/combo).",
    )
    parser.add_argument(
        "--localize", action="store_true",
        help="On fault, re-run per-buffer to name the over-run buffer(s).",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Validate the guard page is live: a kv_buffer short by one page "
        "MUST fault, the same config unmodified MUST pass. Then exit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    T._SEED = args.seed

    if args.selftest:
        return _run_selftest(timeout=args.timeout)

    if args.full:
        batches = T._gfx1250_BATCH_SIZES
        ctx_lens = T._gfx1250_CTX_LENS
        splits = T._gfx1250_SPLIT_PER_BATCH
    else:
        batches, ctx_lens, splits = args.batch, args.kv_seq_lens, args.split_kv

    ok = run_sweep(
        args.variant, batches, ctx_lens, splits,
        localize=args.localize, timeout=args.timeout,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    # Subprocess worker entry: `--oob-worker gqa qlen batch kv split o16 guard`.
    if len(sys.argv) >= 2 and sys.argv[1] == "--oob-worker":
        _gqa, _qlen, _batch, _kv, _split, _o16 = (int(x) for x in sys.argv[2:8])
        _guard_spec = sys.argv[8] if len(sys.argv) >= 9 else "all"
        _inject = sys.argv[9] if len(sys.argv) >= 10 else "none"
        if _guard_spec == "all":
            _guard_set = set(_BUFFERS)
        elif _guard_spec == "none":
            _guard_set = set()
        else:
            _guard_set = set(_guard_spec.split(","))
        _oob_worker_main(
            gqa=_gqa, q_seq_logical=_qlen, batch=_batch, kv_seq_lens=_kv,
            num_kv_splits=_split, out_16_nosplit=_o16, guard_set=_guard_set,
            inject=_inject,
        )
        print("COMPLETED no fault")
        sys.exit(0)

    sys.exit(main())
