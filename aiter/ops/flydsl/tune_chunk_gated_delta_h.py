# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tuner for FlyDSL chunk_gated_delta_h (K5) hidden-state recurrence.

Sweeps ``BV`` candidates for each (shape, dtype, arch) and writes the best
choice -- with the measured per-launch device time -- to
``aiter/ops/flydsl/chunk_gdn_h_tuned.jsonl``. The runtime path
(``linear_attention_prefill_kernels.chunk_gated_delta_rule_fwd_h_flydsl``)
loads this file via ``_lookup_tuned_bv`` and skips the per-call sweep.

Mirrors the ``gdr_decode_tuned.jsonl`` lookup-table pattern used by
``flydsl_gdr_decode``.

Usage:
    python -m aiter.ops.flydsl.tune_chunk_gated_delta_h
    python -m aiter.ops.flydsl.tune_chunk_gated_delta_h --append
    python -m aiter.ops.flydsl.tune_chunk_gated_delta_h \\
        --models Qwen3.5-tp8-8k Qwen3.5-tp4-1k

Each output line is a JSON object with keys::

    arch, dtype, K, V, BT, H, Hg, T_flat, N,
    use_g, use_gk, use_h0, store_fs, save_vn, is_varlen, wu_contig,
    config: {BV: int}, duration: float (us)

The lookup table is keyed on every field except ``duration``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton

from flydsl.runtime.device import get_rocm_arch

from .linear_attention_prefill_kernels import (
    _BV_CANDIDATES,
    _DEFAULT_BV,
    _TUNED_FILE,
    _get_or_compile,
    _launch_kernel,
)

_TUNED_PATH = Path(__file__).resolve().parent / _TUNED_FILE


# -- Shape definitions (mirrors test_flydsl_linear_attention_prefill.py) ---


@dataclass
class TuneShape:
    """Subset of PrefillArgs used by the tuner."""

    K: int
    V: int
    Hk: int
    Hv: int
    tp: int
    full_prompt_len: int
    model_name: str = ""
    BT: int = 64
    max_num_batched_tokens: int = 32768
    dtype: torch.dtype = torch.bfloat16
    is_varlen: bool = True
    output_final_state: bool = True
    save_new_value: bool = True
    use_g: bool = True
    use_gk: bool = False
    use_h0: bool = True
    wu_contiguous: bool = True

    @property
    def Hg(self):
        return self.Hk // self.tp

    @property
    def H(self):
        return self.Hv // self.tp

    def tag(self):
        t = self.model_name + "_" if self.model_name else ""
        t += f"K{self.K}_V{self.V}_Hk{self.Hk}_Hv{self.Hv}"
        t += f"_TP{self.tp}_T{self.full_prompt_len}"
        if not self.is_varlen:
            t += "_novarlen"
        if not self.output_final_state:
            t += "_nofs"
        return t


# Default sweep set. Includes:
#   (1) PREFILL_PARAMS in test_flydsl_linear_attention_prefill.py
#       (GQA: Hk=16, Hv in {32, 64})
#   (2) gdr_prefill_kernel_bench.py shapes (no GQA: Hk = Hv)
# Add new shapes here when the runtime starts emitting "no tuned BV" warnings.
DEFAULT_SHAPES = [
    # -- (1) test PREFILL_PARAMS -----------------------------------------
    # non-varlen + no final state
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B-gqa",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    # varlen + final_state (default path)
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=1024,
        model_name="Qwen3.5-tp4-1k",
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=8192,
        model_name="Qwen3.5-tp4-8k",
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=1024,
        model_name="Qwen3.5-tp8-1k",
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=8192,
        model_name="Qwen3.5-tp8-8k",
    ),
    # -- (2) gdr_prefill_kernel_bench.py shapes (Hk = Hv, no GQA) --------
    # Qwen3.5-35B-A3B: num_v_heads=32
    TuneShape(
        K=128,
        V=128,
        Hk=32,
        Hv=32,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B-A3B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=32,
        Hv=32,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B-A3B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=32,
        Hv=32,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B-A3B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=32,
        Hv=32,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B-A3B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    # Qwen3.5-397B-A17B: num_v_heads=64
    TuneShape(
        K=128,
        V=128,
        Hk=64,
        Hv=64,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B-A17B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=64,
        Hv=64,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B-A17B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=64,
        Hv=64,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B-A17B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    TuneShape(
        K=128,
        V=128,
        Hk=64,
        Hv=64,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B-A17B-bench",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
]


# -- Input helpers (mirror the test file) ---------------------------------


def _build_context_lens(full_prompt_len, max_tokens=32768):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


def _make_inputs(shape: TuneShape):
    context_lens = _build_context_lens(
        shape.full_prompt_len, shape.max_num_batched_tokens
    )

    if shape.is_varlen:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(context_lens), 0).tolist()),
            dtype=torch.int32,
            device="cuda",
        )
        T_total = int(cu_seqlens[-1].item())
        N = len(context_lens)
        B = 1
    else:
        T_total = sum(context_lens)
        B = 1
        N = B
        cu_seqlens = None

    Hg = shape.Hg
    H = shape.H

    k = torch.randn(B, T_total, Hg, shape.K, dtype=shape.dtype, device="cuda") * 0.1
    w = torch.randn(B, H, T_total, shape.K, dtype=shape.dtype, device="cuda") * 0.1
    u = torch.randn(B, H, T_total, shape.V, dtype=shape.dtype, device="cuda") * 0.1
    g = torch.randn(T_total, H, dtype=torch.float32, device="cuda").abs() * -0.5
    g = g.cumsum(dim=0)
    h0 = (
        torch.randn(N, H, shape.V, shape.K, dtype=torch.float32, device="cuda") * 0.01
        if shape.use_h0
        else None
    )
    return k, w, u, g, h0, cu_seqlens, T_total, N, H, Hg


# -- Bench loop -----------------------------------------------------------

_WARMUP = 5
_ITERS = 25


def _bench_bv(shape: TuneShape, BV: int, k, w, u, g, h0, cu, T_total, N, H, Hg):
    """Return per-launch us for a given BV using cuda.Event timing."""
    K = shape.K
    V = shape.V
    BT = shape.BT
    T_flat = T_total

    if cu is None:
        NT = triton.cdiv(T_flat, BT)
        chunk_offsets = None
    else:
        lens = cu[1:] - cu[:-1]
        NT = sum(triton.cdiv(int(seq_len), BT) for seq_len in lens.tolist())
        chunk_offsets = (
            torch.cat([cu.new_tensor([0]), triton.cdiv(lens, BT)])
            .cumsum(-1)
            .to(torch.int32)
        )

    h = k.new_empty(1, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32)
        if shape.output_final_state
        else None
    )
    v_new_buf = k.new_empty(1, H, T_flat, V, dtype=u.dtype)

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    gk_arg = dummy  # use_gk = False
    h0_arg = h0 if h0 is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    cu_arg = cu.to(torch.int32) if cu is not None else dummy.to(torch.int32)
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)
    stream = torch.cuda.current_stream()

    fn = _get_or_compile(
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        H=H,
        Hg=Hg,
        use_g=shape.use_g,
        use_gk=shape.use_gk,
        use_h0=shape.use_h0,
        store_fs=shape.output_final_state,
        save_vn=shape.save_new_value,
        is_varlen=shape.is_varlen,
        wu_contig=shape.wu_contiguous,
    )

    def call():
        _launch_kernel(
            fn,
            BV,
            V,
            N,
            H,
            k,
            u,
            w,
            v_new_buf,
            g_arg,
            gk_arg,
            h,
            h0_arg,
            ht_arg,
            cu_arg,
            co_arg,
            T_flat,
            T_flat,
            stream,
        )

    for _ in range(_WARMUP):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(_ITERS):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / _ITERS * 1000.0  # ms -> us


def _tune_one_shape(shape: TuneShape, arch: str) -> dict:
    """Sweep BV candidates and return the JSONL record for ``shape``."""
    k, w, u, g, h0, cu, T_total, N, H, Hg = _make_inputs(shape)

    candidates = [bv for bv in _BV_CANDIDATES if bv <= shape.V and shape.V % bv == 0]
    if not candidates:
        candidates = [_DEFAULT_BV]

    results = {}
    print(f"[tune] {shape.tag()}  candidates={candidates}")
    for bv in candidates:
        us = _bench_bv(shape, bv, k, w, u, g, h0, cu, T_total, N, H, Hg)
        results[bv] = us
        print(f"  BV={bv:3d}: {us:8.2f} us")

    best_bv = min(results, key=results.get)
    best_us = results[best_bv]
    print(f"  -> best BV={best_bv} ({best_us:.2f} us)")

    return {
        "arch": arch,
        "dtype": str(shape.dtype),
        "K": shape.K,
        "V": shape.V,
        "BT": shape.BT,
        "H": shape.H,
        "Hg": shape.Hg,
        "T_flat": T_total,
        "N": N,
        "use_g": shape.use_g,
        "use_gk": shape.use_gk,
        "use_h0": shape.use_h0,
        "store_fs": shape.output_final_state,
        "save_vn": shape.save_new_value,
        "is_varlen": shape.is_varlen,
        "wu_contig": shape.wu_contiguous,
        "config": {"BV": int(best_bv)},
        "duration": best_us,
    }


# -- IO -------------------------------------------------------------------


def _entry_key(obj: dict):
    """Lookup key used by both writer (dedup) and reader (_lookup_tuned_bv)."""
    return (
        obj["dtype"],
        obj["arch"],
        obj["K"],
        obj["V"],
        obj["BT"],
        obj["H"],
        obj["Hg"],
        obj["T_flat"],
        obj["N"],
        obj["use_g"],
        obj["use_gk"],
        obj["use_h0"],
        obj["store_fs"],
        obj["save_vn"],
        obj["is_varlen"],
        obj["wu_contig"],
    )


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 10:
                continue
            obj = json.loads(line)
            out[_entry_key(obj)] = obj
    return out


def _write(path: Path, entries: dict):
    """Write ``entries`` (key -> obj) sorted deterministically."""
    keys = sorted(entries.keys())
    with path.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(entries[k]) + "\n")


# -- CLI ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge with existing entries in chunk_gdn_h_tuned.jsonl. "
            "Default behavior overwrites any existing key with the freshly "
            "measured value (and keeps unrelated keys)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate the file before writing (drops all existing keys).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Restrict to shapes whose ``model_name`` is in this list. "
            "Use ``--models all`` to tune every shape in DEFAULT_SHAPES."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(_TUNED_PATH),
        help=f"Output jsonl path (default: {_TUNED_PATH})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ROCm/CUDA not available, aborting.", file=sys.stderr)
        sys.exit(1)
    torch.set_default_device("cuda")

    arch = get_rocm_arch()
    print(f"[tune] arch={arch}")

    out_path = Path(args.out)
    if args.overwrite or not out_path.exists():
        existing = {}
    else:
        existing = _load_existing(out_path)
        print(f"[tune] loaded {len(existing)} existing entries from {out_path}")

    shapes = DEFAULT_SHAPES
    if args.models is not None and "all" not in args.models:
        shapes = [s for s in DEFAULT_SHAPES if s.model_name in args.models]
        if not shapes:
            print(f"[tune] no shape matches --models {args.models}", file=sys.stderr)
            sys.exit(2)

    for shape in shapes:
        try:
            entry = _tune_one_shape(shape, arch=arch)
        except Exception as exc:
            print(f"[tune] FAILED for {shape.tag()}: {exc!r}", file=sys.stderr)
            continue
        existing[_entry_key(entry)] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write(out_path, existing)
    print(f"[tune] wrote {len(existing)} entries to {out_path}")


if __name__ == "__main__":
    main()
