# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-0 arm B: the vllm#65188 QSA indexer select at 3c5e29ffcd.

The arm is ``qsa_select_paged_tokens`` from
``vllm/models/qwen4_exp/amd/ops/qsa.py`` at vllm-project/vllm commit
``3c5e29ffcd`` ("[ROCm][Model] Qwen3.8 QSA: MFMA indexer scorer for prefill
chunks"), called the way ``QSAIndexer._select`` calls it there: with
``max_seq_len`` and ``num_decode_tokens`` from the indexer metadata, int64
``logical_positions`` and int32 ``token_to_req``, ``seq_lens`` and block table.

flydsl_venv has vLLM ``6ad2e06c4a`` installed editable. The two QSA op files and
the commit's ``jit_warmup_triton_helper.py`` (imported by the scorer for
``TritonWarmupTensor``) are materialized byte for byte from the commit under
``ARM_DIR`` and imported on top of that install. Their git blob ids are pinned
below. Every other dependency on the select path is the installed one. Its
diff between ``6ad2e06c4a`` and ``3c5e29ffcd`` does not change the path:

- ``torch.ops._C.top_k_per_row_decode`` (ROCm top-k, ``csrc/sampler.cu``):
  same kernel source under ``USE_ROCM``, same schema, same Python wrapper.
- ``vllm.utils.math_utils`` ``cdiv`` and ``round_up``: unchanged.
- ``vllm.triton_utils``: adds an autotuner override; these kernels are not
  autotuned.

Run modes (from the repo root, in flydsl_venv)::

    # Host compile, every GPU hidden. Compiles each Triton launch of the ten
    # bench cases without a device.
    HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \\
        GPU_ARCHS=gfx950 AITER_AOT_REPLAY=1 \\
        flydsl_venv/bin/python op_tests/qsa_phase0_arm_vllm65188.py --host-compile

    # Correctness against arm A, then one timing pass, through gpuq.
    gpuq --num-gpus=1 flydsl_venv/bin/python -u op_tests/qsa_phase0_arm_vllm65188.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import statistics
import subprocess
import sys
import types
from pathlib import Path

import torch

ARM_LABEL = "vllm#65188@3c5e29ffcd"
VLLM_COMMIT = "3c5e29ffcd0aad5b2d99c36ea765f9ccea479341"
VLLM_REPO = Path(os.environ.get("QSA_ARM_B_VLLM_REPO", "/workspaces/vllm"))
ARM_DIR = Path(os.environ.get("QSA_ARM_B_DIR", "/tmp/qsa_arm_b/pkg"))
PACKAGE = "qsa_arm_b_vllm"
HELPER_MODULE = "vllm.model_executor.warmup.jit_warmup_triton_helper"
TOKEN_TOPK = 2048
COMPRESS = 4

# file in PACKAGE -> (path at VLLM_COMMIT, git blob id at VLLM_COMMIT)
_SOURCES = {
    "qsa.py": (
        "vllm/models/qwen4_exp/amd/ops/qsa.py",
        "5f120b954aa20790f2f4ce394ca106bb952378c0",
    ),
    "qsa_mfma_scorer.py": (
        "vllm/models/qwen4_exp/amd/ops/qsa_mfma_scorer.py",
        "3b7013f015cd0a85a9b048b32ace4837b762c25b",
    ),
    "jit_warmup_triton_helper.py": (
        "vllm/model_executor/warmup/jit_warmup_triton_helper.py",
        "1fdaa20e870600fe3a4fdd0a605985e022f1bd56",
    ),
}

_STAGES = ("score", "topk", "expand")
_modules: tuple[types.ModuleType, types.ModuleType] | None = None
_probe: dict | None = None
_stats = {"chunks": 0, "mfma_chunks": 0}


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _materialize() -> Path:
    """Write the pinned files under ARM_DIR, reading them with ``git show``."""
    pkg = ARM_DIR / PACKAGE
    pkg.mkdir(parents=True, exist_ok=True)
    init = pkg / "__init__.py"
    if not init.exists():
        init.write_text(f"# Materialized from vllm-project/vllm {VLLM_COMMIT}.\n")
    for name, (path, blob) in _SOURCES.items():
        target = pkg / name
        if not target.exists() or _blob_id(target.read_bytes()) != blob:
            data = subprocess.check_output(
                ["git", "-C", str(VLLM_REPO), "show", f"{VLLM_COMMIT}:{path}"]
            )
            target.write_bytes(data)
        got = _blob_id(target.read_bytes())
        if got != blob:
            raise RuntimeError(f"{target} is blob {got}, expected {blob} at {path}")
    return pkg


def load() -> tuple[types.ModuleType, types.ModuleType]:
    """Import the arm's ``qsa`` and ``qsa_mfma_scorer`` and install stage probes."""
    global _modules
    if _modules is not None:
        return _modules
    pkg = _materialize()
    import vllm.model_executor.warmup  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        HELPER_MODULE, pkg / "jit_warmup_triton_helper.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[HELPER_MODULE] = helper
    spec.loader.exec_module(helper)
    if str(ARM_DIR) not in sys.path:
        sys.path.insert(0, str(ARM_DIR))
    qsa = importlib.import_module(f"{PACKAGE}.qsa")
    scorer = importlib.import_module(f"{PACKAGE}.qsa_mfma_scorer")
    _install_probes(qsa)
    _modules = (qsa, scorer)
    return _modules


def _bracket(stage: str, fn, *args, **kwargs):
    if _probe is None:
        return fn(*args, **kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    _probe.setdefault(stage, []).append((start, end))
    return out


def _install_probes(qsa: types.ModuleType) -> None:
    """Wrap the three stage calls ``qsa_select_paged_tokens`` resolves as
    module globals. The wrapped functions are the commit's, unchanged."""
    from vllm import _custom_ops

    real_score = qsa.qsa_mqa_paged
    real_expand = qsa.expand_qsa_block_indices_cuda
    applies = qsa.mfma_scorer_applies

    def score(q, k_cache, *args, allow_mfma=False, **kwargs):
        _stats["chunks"] += 1
        _stats["mfma_chunks"] += int(allow_mfma and applies(q, k_cache))
        return _bracket(
            "score", real_score, q, k_cache, *args, allow_mfma=allow_mfma, **kwargs
        )

    def topk(*args, **kwargs):
        return _bracket("topk", _custom_ops.top_k_per_row_decode, *args, **kwargs)

    def expand(*args, **kwargs):
        return _bracket("expand", real_expand, *args, **kwargs)

    qsa.qsa_mqa_paged = score
    qsa.expand_qsa_block_indices_cuda = expand
    qsa.ops = types.SimpleNamespace(top_k_per_row_decode=topk)


class Arm:
    """vllm#65188 at 3c5e29ffcd, one indexer select step per ``step``.

    ``max_seq_len`` is a host int in vLLM's metadata. ``step`` reads it from
    ``context`` once per context tensor (one ``.item()`` sync on the first step
    that sees it, normally a warmup step) unless ``set_max_seq_len`` gave it.
    """

    def __init__(self, index_k: torch.Tensor, max_model_len: int):
        self._qsa, self._scorer = load()
        if index_k.ndim != 4 or index_k.shape[2] != 1:
            raise ValueError("index_k must be [pages, page, 1, dim]")
        self.index_k = index_k
        self.max_model_len = max_model_len
        self.last_chunks = 0
        self.last_mfma_chunks = 0
        self._seq_len: dict[tuple[int, int, int], tuple[torch.Tensor, int]] = {}
        self._pos64: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _key(t: torch.Tensor) -> tuple[int, int, int]:
        return (id(t), t.data_ptr(), t._version)

    def set_max_seq_len(self, context: torch.Tensor, max_seq_len: int) -> None:
        self._seq_len[self._key(context)] = (context, int(max_seq_len))

    def _max_seq_len(self, context: torch.Tensor) -> int:
        key = self._key(context)
        if key not in self._seq_len:
            self._seq_len[key] = (context, int(context.max().item()))
        value = self._seq_len[key][1]
        if value > self.max_model_len:
            raise ValueError(f"max_seq_len {value} exceeds {self.max_model_len}")
        return value

    def _positions(self, positions: torch.Tensor) -> torch.Tensor:
        """vLLM's ``logical_positions`` are int64."""
        if positions.dtype == torch.int64:
            return positions
        key = self._key(positions)
        if key not in self._pos64:
            self._pos64[key] = (positions, positions.to(torch.int64))
        return self._pos64[key][1]

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
        """Run one full indexer select step and write token_ids [M, 2051] int32,
        -1 padded. Appends (start, end) CUDA events to probe['score'],
        probe['topk'] and probe['expand'] around every launch of that stage."""
        global _probe
        max_seq_len = self._max_seq_len(context)
        positions = self._positions(positions)
        before = dict(_stats)
        _probe = probe
        try:
            self._qsa.qsa_select_paged_tokens(
                q,
                self.index_k,
                table,
                token_to_req,
                positions,
                context,
                TOKEN_TOPK,
                COMPRESS,
                token_ids,
                max_seq_len=max_seq_len,
                num_decode_tokens=num_decode_rows,
            )
        finally:
            _probe = None
        self.last_chunks = _stats["chunks"] - before["chunks"]
        self.last_mfma_chunks = _stats["mfma_chunks"] - before["mfma_chunks"]


def num_decode_rows(case) -> int:
    """Leading requests vLLM's QSA metadata classes as decode.

    The builder sets ``reorder_batch_threshold`` to 1, raised to
    ``1 + num_speculative_tokens`` with speculative decoding
    (``supports_spec_as_decode=True``). The ``mtp`` case's 4-row request is a
    decode request under 3 speculative tokens. Any other request of more than
    one row is prefill.
    """
    if case.name == "mtp":
        return case.requests[0].rows
    rows = 0
    for req in case.requests:
        if req.rows != 1:
            break
        rows += 1
    return rows


def _bench():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_qsa_indexer_select as bench

    return bench


def host_compile() -> None:
    """Compile every Triton launch the ten bench cases reach, with no GPU.

    The arm's own ``qsa_select_paged_tokens`` runs under aiter's ``dry_run``
    (FakeTensors, Triton target hip/gfx950, warp 64). Each launch is replaced
    by ``kernel.warmup`` on the commit's ``TritonWarmupTensor`` stand-ins
    carrying the runtime tensor's dtype, shape, strides, 16-byte alignment and
    storage bytes (ROCm's ``tt.pointer_range`` key), and the launch's own
    scalars, so the compile keys are the runtime's. The top-k is a prebuilt
    C++ op and is skipped.
    """
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(var, None) != "":
            raise SystemExit(f"--host-compile needs {var}='' (every GPU hidden)")
    from unittest import mock

    import triton

    from aiter.aot.flydsl import dry_run as dry_run_module

    bench = _bench()
    qsa, scorer = load()
    from vllm import _custom_ops

    helper = sys.modules[HELPER_MODULE]
    warmed: list[str] = []

    def stand_in(arg):
        if not isinstance(arg, torch.Tensor):
            return arg
        offset = arg.storage_offset() * arg.element_size()
        return helper.TritonWarmupTensor(
            arg.dtype,
            aligned=offset % 16 == 0,
            shape=tuple(arg.shape),
            strides=tuple(arg.stride()),
            ptr_range_bytes=arg.untyped_storage().nbytes(),
        )

    class Launch:
        def __init__(self, kernel, grid):
            self.kernel = kernel
            self.grid = grid

        def __call__(self, *args, **kwargs):
            compiled = self.kernel.warmup(
                *(stand_in(a) for a in args), grid=self.grid, **kwargs
            )
            sig = ",".join(
                (
                    f"{a.dtype}{'' if a.aligned else '/u'}"
                    f"{'/S' if a.ptr_range_bytes <= 2**31 - 1 else ''}"
                    if isinstance(a, helper.TritonWarmupTensor)
                    else str(a)
                )
                for a in map(stand_in, args)
            )
            warmed.append(f"{self.kernel.fn.__name__} hash={compiled.hash} {sig}")

    class Kernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            return Launch(self.kernel, grid)

    patches = [
        (qsa, "_qsa_mqa_paged_kernel"),
        (qsa, "_expand_qsa_indices_kernel"),
        (scorer, "_qsa_mqa_paged_mfma_kernel"),
    ]
    saved = [(owner, name, getattr(owner, name)) for owner, name in patches]
    for owner, name, real in saved:
        setattr(owner, name, Kernel(real))
    real_topk = _custom_ops.top_k_per_row_decode
    _custom_ops.top_k_per_row_decode = lambda *a, **k: None
    print(f"triton cache dir {triton.knobs.cache.dir}", flush=True)
    # Triton's run() reads the stream before it sees warmup=True; it is unused.
    stream = mock.patch.object(
        dry_run_module._TritonDriver,
        "get_current_stream",
        lambda _self, _device=None: 0,
        create=True,
    )
    try:
        with dry_run_module.dry_run("gfx950", 256), stream:
            index_k = bench._wide_indexer_cache()
            arm = Arm(index_k, bench.MAX_MODEL_LEN)
            for case in bench._cases():
                built = bench._build(case, index_k)
                q, _, table, token_to_req, positions, context = built[:6]
                token_ids = built[-1]
                arm.set_max_seq_len(context, max(r.context for r in case.requests))
                start = len(warmed)
                arm.step(
                    q,
                    table,
                    token_to_req,
                    positions,
                    context,
                    token_ids,
                    num_decode_rows(case),
                    None,
                )
                print(
                    f"{case.name}: decode_rows={num_decode_rows(case)} "
                    f"chunks={arm.last_chunks} mfma_chunks={arm.last_mfma_chunks} "
                    f"launches={len(warmed) - start}",
                    flush=True,
                )
    finally:
        for owner, name, real in saved:
            setattr(owner, name, real)
        _custom_ops.top_k_per_row_decode = real_topk
    unique = sorted(set(warmed))
    print(f"TRITON_WARMUP_LAUNCHES {len(warmed)} unique_keys {len(unique)}")
    for line in unique:
        print(f"  {line}")
    print("HOST_COMPILE_DONE", flush=True)


def _triton_cache_entries() -> set[str]:
    import triton

    root = Path(triton.knobs.cache.dir)
    return {p.name for p in root.iterdir()} if root.exists() else set()


def _fill_live_pages(index_k: torch.Tensor, cases) -> None:
    pages = max(
        sum(
            (r.context // COMPRESS + index_k.shape[1] - 1) // index_k.shape[1]
            for r in case.requests
        )
        for case in cases
    )
    gen = torch.Generator(device="cuda").manual_seed(1481)
    index_k[:pages].copy_(
        torch.randn(index_k[:pages].shape, generator=gen, device="cuda").to(
            index_k.dtype
        )
    )


def _token_sets_agree(
    a: torch.Tensor, b: torch.Tensor, positions: torch.Tensor
) -> tuple[float, float]:
    """Fraction of rows with identical token sets, and block agreement
    |A ? B| / |A| over each row's selected complete blocks (the tail excluded)."""
    rows_equal = (a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1)
    tail = ((positions.to(torch.int64) + 1) // COMPRESS * COMPRESS)[:, None]
    row = torch.arange(a.shape[0], device=a.device, dtype=torch.int64)[:, None]

    def block_keys(t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.int64)
        keep = (t >= 0) & (t < tail) & (t % COMPRESS == 0)
        return (row * (1 << 20) + t // COMPRESS)[keep]

    ka = block_keys(a)
    kb = block_keys(b)
    block = torch.isin(ka, kb).sum().item() / max(ka.numel(), 1)
    if ka.numel() != kb.numel():
        block = min(block, torch.isin(kb, ka).sum().item() / max(kb.numel(), 1))
    return rows_equal.float().mean().item(), block


def _row_blocks(tokens: torch.Tensor, position: int) -> set[int]:
    tail = (position + 1) // COMPRESS * COMPRESS
    keep = (tokens >= 0) & (tokens < tail) & (tokens % COMPRESS == 0)
    return set((tokens[keep] // COMPRESS).tolist())


def _boundary(built, index_k, a, b, max_rows: int = 64) -> str:
    """fp64 rescore of up to ``max_rows`` rows whose sets differ.

    Each differing block's distance from the row's fp64 512th score, in units
    of vllm#65188's bound (head_dim + 8) * 2^-24 * sum_h sum_d |q k| / sqrt(d)
    taken at the larger of the block's and the 512th block's bound.
    """
    q, _, table, token_to_req, positions, context = built[:6]
    differ = (a.sort(dim=1).values != b.sort(dim=1).values).any(dim=1)
    rows = differ.nonzero().flatten()[:max_rows].tolist()
    page, dim = index_k.shape[1], q.shape[2]
    ratios = []
    picked_by_fp64 = {"a": 0, "b": 0}
    for r in rows:
        req = int(token_to_req[r])
        pos = int(positions[r])
        visible = min((pos + 1) // COMPRESS, int(context[req]) // COMPRESS)
        cols = torch.arange(visible, device=q.device)
        keys = index_k[table[req, cols // page].long(), cols % page, 0].double()
        qr = q[r].double()
        score = torch.relu(keys @ qr.T).sum(1) / dim**0.5
        bound = (dim + 8) * 2.0**-24 * (keys.abs() @ qr.abs().T).sum(1) / dim**0.5
        top = torch.topk(score, 512)
        s512 = top.values[-1]
        b512 = bound[top.indices[-1]]
        fp64 = set(top.indices.tolist())
        ba, bb = _row_blocks(a[r], pos), _row_blocks(b[r], pos)
        for blk in ba ^ bb:
            ratios.append(
                (abs(score[blk] - s512) / torch.maximum(bound[blk], b512)).item()
            )
        picked_by_fp64["a"] += len((ba - bb) & fp64)
        picked_by_fp64["b"] += len((bb - ba) & fp64)
    if not ratios:
        return "differing_rows=0"
    over = sum(x > 2.0 for x in ratios)
    return (
        f"rows_checked={len(rows)} differing_blocks={len(ratios)} "
        f"max_dist/bound={max(ratios):.4f} over_2x_bound={over} "
        f"fp64_top512_has: armA_only={picked_by_fp64['a']} "
        f"armB_only={picked_by_fp64['b']}"
    )


def _correctness(bench, arm: Arm, index_k: torch.Tensor, names) -> None:
    from aiter.ops.flydsl import qsa_paged as paged
    from aiter.ops.qsa import qsa_expand_block_indices

    for case in bench._cases():
        if case.name not in names:
            continue
        built = bench._build(case, index_k)
        q, _, table, token_to_req, positions, context = built[:6]
        block_ids, token_ids = built[-2], built[-1]
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
        ref = torch.empty_like(token_ids)
        qsa_expand_block_indices(
            block_ids, positions, context, token_to_req, COMPRESS, TOKEN_TOPK, ref
        )
        token_ids.fill_(-7)
        arm.step(
            q,
            table,
            token_to_req,
            positions,
            context,
            token_ids,
            num_decode_rows(case),
            None,
        )
        torch.cuda.synchronize()
        untouched = (token_ids == -7).sum().item()
        rows, blocks = _token_sets_agree(ref, token_ids, positions)
        valid_a = (ref >= 0).sum().item()
        valid_b = (token_ids >= 0).sum().item()
        print(
            f"CORRECT {case.name}: rows_identical={rows:.6f} block_agreement="
            f"{blocks:.7f} valid_tokens armA={valid_a} armB={valid_b} "
            f"untouched={untouched} chunks={arm.last_chunks} "
            f"mfma_chunks={arm.last_mfma_chunks}",
            flush=True,
        )
        print(
            f"BOUNDARY {case.name}: {_boundary(built, index_k, ref, token_ids)}",
            flush=True,
        )
        del built, ref


def _once(arm: Arm, args: tuple) -> tuple[float, ...]:
    probe = {stage: [] for stage in _STAGES}
    arm.step(*args, probe)
    torch.cuda.synchronize()
    return tuple(
        sum(s.elapsed_time(e) for s, e in probe[stage]) * 1000.0 for stage in _STAGES
    )


def _time(bench, arm: Arm, index_k: torch.Tensor, label: str) -> list[dict]:
    rows = []
    for case in bench._cases():
        built = bench._build(case, index_k)
        decode = num_decode_rows(case)
        args = (*built[:1], *built[2:6], built[-1], decode)
        for _ in range(bench.WARMUP):
            _once(arm, args)
        samples = [_once(arm, args) for _ in range(bench.ITERS)]
        score, topk, expand = (
            statistics.median(s[i] for s in samples) for i in range(3)
        )
        step = score + topk + expand
        got = {
            "case": case.name,
            "data": label,
            "chunks": arm.last_chunks,
            "mfma": arm.last_mfma_chunks,
            "decode": decode,
            "score": score,
            "topk": topk,
            "expand": expand,
            "step": step,
            "share": (topk + expand) / step,
        }
        print(
            f"TIME {label} {case.name}: T_score {score:.1f} T_topk {topk:.1f} "
            f"T_expand {expand:.1f} T_step {step:.1f} share {got['share']:.3f} "
            f"chunks {got['chunks']} mfma_chunks {got['mfma']} decode_rows {decode}",
            flush=True,
        )
        rows.append(got)
        del built
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=f"phase-0 arm {ARM_LABEL}")
    parser.add_argument("--host-compile", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--correct-cases",
        default="p8k,p32k,p128k,p256k,mix16,mtp",
        help="Comma-separated.",
    )
    args = parser.parse_args()
    if args.host_compile:
        host_compile()
        return

    bench = _bench()
    bench._print_device()
    bench._assert_idle()
    print(
        "GPUQ_ENV "
        + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if "GPUQ" in k),
        flush=True,
    )
    from vllm.platforms import current_platform

    cache_before = _triton_cache_entries()
    index_k = bench._wide_indexer_cache()
    arm = Arm(index_k, bench.MAX_MODEL_LEN)
    print(
        f"ARM {ARM_LABEL} platform={type(current_platform).__name__} "
        f"indexer pool pages={index_k.shape[0]} "
        f"span_bytes={index_k.untyped_storage().nbytes()} "
        f"triton_cache_entries={len(cache_before)}",
        flush=True,
    )
    results = []
    if not args.skip_timing:
        results += _time(bench, arm, index_k, "bench-K")
    _fill_live_pages(index_k, bench._cases())
    _correctness(bench, arm, index_k, set(args.correct_cases.split(",")))
    if not args.skip_timing:
        results += _time(bench, arm, index_k, "randn-K")
    new = _triton_cache_entries() - cache_before
    print(f"NEW_TRITON_CACHE_ENTRIES {len(new)} {sorted(new)}", flush=True)
    print(
        "case | arm | K data | chunks | mfma chunks | decode rows | T_score | "
        "T_topk | T_expand | T_step | top-k+expand share",
        flush=True,
    )
    for got in results:
        print(
            f"{got['case']} | {ARM_LABEL} | {got['data']} | {got['chunks']} | "
            f"{got['mfma']} | {got['decode']} | {got['score']:.1f} | "
            f"{got['topk']:.1f} | {got['expand']:.1f} | {got['step']:.1f} | "
            f"{got['share']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
