# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-0 arm C: vllm#62589 at commit ``e7a2acb5e3``.

The arm is ``qsa_select_paged_tokens`` from
``vllm/models/qwen4_exp/amd/ops/qsa.py`` at that commit, called the way the
commit's ``Qwen4ExpQSAIndexer._select`` calls it in serving: one call per step
with the compressed key cache, the request block table, ``token_to_req``,
int64 ``logical_positions``, per-request ``seq_lens``, ``token_topk`` 2048,
compression ratio 4, and the ``[M, 2051]`` output. It chunks rows itself to a
128 MiB fp32 logits workspace (509 rows at 65,856 columns). Each chunk runs the
tiled Triton scorer ``_qsa_mqa_paged_kernel`` (``BLOCK_M`` 16, ``BLOCK_N`` 64),
the ROCm top-k ``torch.ops._C.top_k_per_row_decode`` and the Triton expand
``_expand_qsa_indices_kernel``.

The installed vLLM is main ``6ad2e06c4a``. Only the commit's ``qsa.py`` is
taken from ``e7a2acb5e3``: it is written to ``/tmp/qsa_arm_c`` from git,
checked against its blob id, and imported as a standalone module on top of
the installed vLLM. Its other dependencies (``vllm._custom_ops``,
``vllm.triton_utils``, ``vllm.platforms``) resolve to main, whose
``top_k_per_row_decode`` source matches the commit's on ROCm.

The commit's API has no decode-row or ``max_seq_len`` argument, so
``num_decode_rows`` is ignored.

GPU run (from the repo root, after the host compile in
``/tmp/qsa_arm_c/host_compile.py``)::

    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python -u \\
        op_tests/qsa_phase0_arm_vllm62589.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import torch

ARM_LABEL = "vllm#62589@e7a2acb5e3"

VLLM_REPO = "/workspaces/vllm"
COMMIT = "e7a2acb5e3a1be7158b0b6e9955c9dd2f82db00f"
QSA_PATH = "vllm/models/qwen4_exp/amd/ops/qsa.py"
QSA_BLOB = "194b13ee36bfe4ebec1299e50a76ccbded60a6fd"
CACHE_DIR = Path("/tmp/qsa_arm_c/vllm_e7a2acb5e3")
# Triton keys compiled kernels on the module name, so it must not change
# between the host compile and the GPU run.
MODULE_NAME = "qsa_arm_c_vllm62589_e7a2acb5e3"

COMPRESS = 4
TOKEN_TOPK = 2048
OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS - 1

_MODULE: ModuleType | None = None
_PROBE: dict | None = None
_CHUNKS = 0


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def materialize() -> Path:
    """Write the commit's ``qsa.py`` under ``CACHE_DIR`` and verify its blob id."""
    path = CACHE_DIR / "qsa.py"
    if not path.exists() or _blob_id(path.read_bytes()) != QSA_BLOB:
        data = subprocess.check_output(
            ["git", "-C", VLLM_REPO, "show", f"{COMMIT}:{QSA_PATH}"]
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    got = _blob_id(path.read_bytes())
    if got != QSA_BLOB:
        raise RuntimeError(f"{path} has blob {got}, expected {QSA_BLOB}")
    return path


def _bracket(stage: str, fn, *args, **kwargs):
    probe = _PROBE
    if probe is None:
        return fn(*args, **kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    probe[stage].append((start, end))
    return out


class _OpsProxy:
    """``vllm._custom_ops`` with ``top_k_per_row_decode`` bracketed as top-k."""

    def __init__(self, ops: ModuleType) -> None:
        self._ops = ops

    def top_k_per_row_decode(self, *args, **kwargs):
        return _bracket("topk", self._ops.top_k_per_row_decode, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._ops, name)


def load_module() -> ModuleType:
    """Import the commit's ``qsa.py`` once, with stage probes on its three stages.

    ``qsa_select_paged_tokens`` looks up ``qsa_mqa_paged``, ``ops`` and
    ``expand_qsa_block_indices_cuda`` in its module globals at call time, so
    replacing them there brackets every launch without changing its code.
    """
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = materialize()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)

    real_score = module.qsa_mqa_paged
    real_expand = module.expand_qsa_block_indices_cuda

    def score(*args, **kwargs):
        global _CHUNKS
        _CHUNKS += 1
        return _bracket("score", real_score, *args, **kwargs)

    def expand(*args, **kwargs):
        return _bracket("expand", real_expand, *args, **kwargs)

    module.qsa_mqa_paged = score
    module.expand_qsa_block_indices_cuda = expand
    module.ops = _OpsProxy(module.ops)
    _MODULE = module
    return module


def rows_per_chunk(columns: int) -> int:
    module = load_module()
    return max(1, module._LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))


class Arm:
    def __init__(self, index_k: torch.Tensor, max_model_len: int) -> None:
        if index_k.ndim != 4 or index_k.shape[2] != 1:
            raise ValueError("index_k must be [pages, page, 1, dim]")
        self.index_k = index_k
        self.max_model_len = max_model_len
        self.module = load_module()
        self.last_chunks = 0

    def step(
        self,
        q,
        table,
        token_to_req,
        positions,
        context,
        token_ids,
        num_decode_rows: int,
        probe: dict | None,
    ) -> None:
        """Run one full indexer select step and write token_ids [M, 2051] int32, -1 padded.
        Append (start_event, end_event) CUDA event pairs to probe['score'], probe['topk'], probe['expand'] around
        every launch of that stage (summed over the arm's own chunks). Do not synchronize inside.

        ``probe=None`` records no events. ``num_decode_rows`` is ignored: the
        commit's API takes no decode count.
        """
        global _PROBE, _CHUNKS
        del num_decode_rows
        if table.shape[1] * self.index_k.shape[1] * COMPRESS < self.max_model_len:
            raise ValueError("page table does not cover max_model_len")
        # Serving's logical_positions buffer is int64. It is metadata shared by
        # every QSA layer, so the cast is outside the stage events.
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        _PROBE = probe
        _CHUNKS = 0
        try:
            self.module.qsa_select_paged_tokens(
                q,
                self.index_k,
                table,
                token_to_req,
                positions,
                context,
                TOKEN_TOPK,
                COMPRESS,
                token_ids,
            )
        finally:
            _PROBE = None
            self.last_chunks = _CHUNKS


def _fill_live_pages(index_k: torch.Tensor, pages: int, seed: int = 0) -> None:
    gen = torch.Generator(device=index_k.device).manual_seed(seed)
    step = 512
    for start in range(0, pages, step):
        end = min(start + step, pages)
        index_k[start:end].normal_(generator=gen)


def _live_pages(case, page: int) -> int:
    return sum((req.context // COMPRESS + page - 1) // page for req in case.requests)


def _sorted_ids(ids: torch.Tensor) -> torch.Tensor:
    big = torch.iinfo(torch.int32).max
    return torch.where(ids < 0, big, ids).sort(dim=1).values


def _set_overlap(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: count of valid entries of ``a`` and how many are also in ``b``."""
    big = torch.iinfo(torch.int32).max
    sa = _sorted_ids(a)
    sb = _sorted_ids(b).contiguous()
    idx = torch.searchsorted(sb, sa.contiguous()).clamp_max(sb.shape[1] - 1)
    valid = sa != big
    hit = (sb.gather(1, idx) == sa) & valid
    return valid.sum(1), hit.sum(1)


def _blocks(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    tail = ((positions.long() + 1) // COMPRESS * COMPRESS)[:, None]
    keep = (ids >= 0) & (ids < tail) & (ids % COMPRESS == 0)
    return torch.where(keep, ids // COMPRESS, -1)


def _row_scores(q, index_k, table, req: int, pos: int, context: int) -> torch.Tensor:
    page = index_k.shape[1]
    visible = min((pos + 1) // COMPRESS, context // COMPRESS)
    pages = (visible + page - 1) // page
    keys = index_k[table[req, :pages].long()].reshape(-1, index_k.shape[-1])
    keys = keys[:visible].float()
    return torch.relu(q.float() @ keys.T).sum(0)


def _compare(case, built, ids_a, ids_c, index_k, max_tie_rows: int = 32) -> dict:
    q, _, table, token_to_req, positions, context, *_rest = built
    tok_n, tok_hit = _set_overlap(ids_c, ids_a)
    tok_n_a, _ = _set_overlap(ids_a, ids_c)
    rows_same = (tok_hit == tok_n) & (tok_n == tok_n_a)
    blk_c = _blocks(ids_c, positions)
    blk_a = _blocks(ids_a, positions)
    blk_n, blk_hit = _set_overlap(blk_c, blk_a)
    blk_n_a, _ = _set_overlap(blk_a, blk_c)
    diff_rows = torch.nonzero(~rows_same).flatten().tolist()
    worst_gap = 0.0
    for row in diff_rows[:max_tie_rows]:
        req = int(token_to_req[row])
        pos = int(positions[row])
        scores = _row_scores(q[row], index_k, table, req, pos, int(context[req]))
        kth = scores.topk(min(512, scores.numel())).values[-1]
        sa = {int(x) for x in blk_a[row].tolist() if x >= 0}
        sc = {int(x) for x in blk_c[row].tolist() if x >= 0}
        for blk in sa ^ sc:
            gap = abs(float(scores[blk] - kth)) / max(abs(float(kth)), 1e-6)
            worst_gap = max(worst_gap, gap)
    rows = q.shape[0]
    return {
        "rows": rows,
        "rows_identical": int(rows_same.sum()),
        "row_frac": float(rows_same.float().mean()),
        "block_agree": float(blk_hit.sum()) / max(float(blk_n.sum()), 1.0),
        "block_count_equal": bool(torch.equal(blk_n, blk_n_a)),
        "max_blocks_diff_row": int((blk_n - blk_hit).max()),
        "tie_rows_checked": min(len(diff_rows), max_tie_rows),
        "worst_rel_gap_to_512th": worst_gap,
    }


def _triton_entries() -> dict[str, int]:
    import os

    root = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton/cache"))
    counts = {"_qsa_mqa_paged_kernel": 0, "_expand_qsa_indices_kernel": 0}
    if not root.exists():
        return counts
    for entry in root.iterdir():
        for name in counts:
            if (entry / f"{name}.hsaco").exists():
                counts[name] += 1
    return counts


def _num_decode_rows(case) -> int:
    # vLLM counts a request as decode when its query length is at most
    # 1 + num_speculative_tokens, so the 4-row MTP request is decode.
    count = 0
    for req in case.requests:
        if req.rows > 4:
            break
        count += req.rows
    return count


def _sleep_cycles_per_ms() -> float:
    cycles = 10_000_000
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return cycles / start.elapsed_time(end)


def _time_step(
    arm: Arm, inputs: tuple, num_decode_rows: int, sleep_cycles: int = 0
) -> tuple:
    probe = {"score": [], "topk": [], "expand": []}
    if sleep_cycles:
        # Queued ahead of the step so the host issues every launch before the
        # GPU reaches it; the stage events then hold GPU time only.
        torch.cuda._sleep(sleep_cycles)
    arm.step(*inputs, num_decode_rows, probe)
    torch.cuda.synchronize()
    return tuple(
        sum(s.elapsed_time(e) for s, e in probe[k]) * 1000.0
        for k in ("score", "topk", "expand")
    )


def main() -> None:
    import argparse
    import datetime
    import statistics

    import bench_qsa_indexer_select as bench

    from aiter.ops.flydsl import qsa_paged as paged
    from aiter.ops.qsa import qsa_expand_block_indices

    parser = argparse.ArgumentParser(description=f"Arm C {ARM_LABEL}")
    parser.add_argument("--smoke-cases", default="p8k,p32k,p128k")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--host-ahead-ms",
        type=float,
        default=0.0,
        help="Also time each case with a GPU sleep of this length queued "
        "before every step, so the stage events exclude host launch gaps.",
    )
    args = parser.parse_args()

    print(f"DATE {datetime.datetime.now(datetime.UTC).isoformat()}", flush=True)
    bench._print_device()
    bench._assert_idle()
    if bench.get_gfx_runtime() not in bench.SUPPORTED_GFX:
        raise SystemExit(2)
    before = _triton_entries()
    print(f"TRITON_ENTRIES_BEFORE {before}", flush=True)

    cases = bench._cases()
    index_k = bench._wide_indexer_cache()
    need = max(_live_pages(case, bench.PAGE) for case in cases)
    _fill_live_pages(index_k, need)
    print(
        f"indexer pool pages={index_k.shape[0]} span_bytes={paged.span_bytes(index_k)} "
        f"filled_pages={need}",
        flush=True,
    )
    arm = Arm(index_k, bench.MAX_MODEL_LEN)
    print(f"ARM {ARM_LABEL} rows_per_chunk={rows_per_chunk(bench.COLUMNS)}", flush=True)

    smoke = [c for c in cases if c.name in args.smoke_cases.split(",")]
    for case in smoke:
        built = bench._build(case, index_k)
        q, _, table, token_to_req, positions, context, _, _, block_ids, ids_a = built
        paged.qsa_indexer_block_ids(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            out=block_ids,
            heads=(bench.HEADS,),
        )
        qsa_expand_block_indices(
            block_ids,
            positions,
            context,
            token_to_req,
            COMPRESS,
            TOKEN_TOPK,
            ids_a,
        )
        ids_c = torch.full_like(ids_a, -7)
        arm.step(q, table, token_to_req, positions, context, ids_c, 0, None)
        torch.cuda.synchronize()
        got = _compare(case, built, ids_a, ids_c, index_k)
        print(f"SMOKE {case.name} chunks={arm.last_chunks} {got}", flush=True)
        del built
        torch.cuda.empty_cache()

    if args.skip_timing:
        print(f"TRITON_ENTRIES_AFTER {_triton_entries()}", flush=True)
        return

    modes = [("synced", 0)]
    if args.host_ahead_ms > 0:
        per_ms = _sleep_cycles_per_ms()
        print(f"SLEEP cycles_per_ms={per_ms:.0f}", flush=True)
        modes.append(("host-ahead", int(per_ms * args.host_ahead_ms)))

    rows = []
    for case in cases:
        built = bench._build(case, index_k)
        q, _, table, token_to_req, positions, context, _, _, _, token_ids = built
        inputs = (q, table, token_to_req, positions, context, token_ids)
        n_dec = _num_decode_rows(case)
        for mode, cycles in modes:
            for _ in range(bench.WARMUP):
                _time_step(arm, inputs, n_dec, cycles)
            samples = [
                _time_step(arm, inputs, n_dec, cycles) for _ in range(bench.ITERS)
            ]
            score, topk, expand = (
                statistics.median(s[i] for s in samples) for i in range(3)
            )
            step = score + topk + expand
            row = (
                case.name,
                mode,
                arm.last_chunks,
                score,
                topk,
                expand,
                step,
                (topk + expand) / step,
            )
            rows.append(row)
            print(
                f"{case.name} {mode}: T_score {score:.1f} T_topk {topk:.1f} "
                f"T_expand {expand:.1f} T_step {step:.1f} share {row[-1]:.3f} "
                f"chunks {arm.last_chunks} num_decode_rows {n_dec}",
                flush=True,
            )
        del built
        torch.cuda.empty_cache()

    print(
        "case | arm | mode | chunks | T_score | T_topk | T_expand | T_step | "
        "top-k+expand share",
        flush=True,
    )
    for name, mode, chunks, score, topk, expand, step, share in rows:
        print(
            f"{name} | {ARM_LABEL} | {mode} | {chunks} | {score:.1f} | {topk:.1f} | "
            f"{expand:.1f} | {step:.1f} | {share:.3f}",
            flush=True,
        )
    after = _triton_entries()
    print(f"TRITON_ENTRIES_AFTER {after} new_compiles={after != before}", flush=True)


if __name__ == "__main__":
    main()
