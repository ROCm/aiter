# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused DSpark Markov draft sampling (5 steps, vocab 129280, rank 256).

One persistent launch runs the whole sequential Markov chain of a DSpark
speculator (vLLM ``DSparkSpeculator._sample_markov_chain`` with
``draft_sample_method=probabilistic``) for every request of a batch: per step
the Markov bias GEMV of the previous token, the add to the base logits, the
draft-logits cache write and the Gumbel-max sample with vLLM's murmur3 draws
(``gumbel_sample(..., is_drafting=True)``, fp32), bit for bit the same noise.

Explicit specialization (callers fall back to their own loop otherwise, see
:func:`markov_chain_sample_unsupported_reason`): 5 draft steps, vocabulary
129280, Markov rank 256, bf16 ``markov_w1`` / ``markov_w2`` [129280, 256]
contiguous, bf16 base logits with unit column stride, a replicated bf16 draft
logits cache [slots, 5, 129280] (< 2 GiB), fp32 Gumbel draws (not fp64), no
confidence head, no reduced draft vocabulary, no draft watermarker, no
acceptance estimator, gfx950 with >= 253 CUs. The speculator features are
required arguments of :func:`markov_chain_sample_supported`, so a caller states
each one.

The bias sum has the MFMA's order, so a bias can differ from another order's
(hipBLASLt's) in its last bf16 bit; the rejection sampler verifies every draft
token against the target distribution, so this changes which tokens are
proposed, not the output distribution.

The 253 workgroups wait for each other (cross-workgroup exchange through
memory), so the launch must not share the GPU with a kernel that waits on it,
and calls of one device must be issued on one stream at a time (one workspace
per device and row-count variant).
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .kernels import markov_chain_gumbel as _kernel
from .kernels.tensor_shim import _run_compiled, ptr_arg

V, R, NSPEC = _kernel.V, _kernel.R, _kernel.NSPEC
MAX_ROWS = 160  # rows of one launch; more rows run as consecutive launches
# 16-row fragments of a launch's buffers; the row count rounds up to one.
NRF_VARIANTS = (1, 2, 3, 4, 6, 8, 10)
# The draft-noise salt (2^30) plus the draw position sample_pos - 1.
_POS_ADD = (1 << 30) - 1
SUPPORTED_GFX = ("gfx950",)


def nrf_for(n: int) -> int:
    """The row-fragment variant (of 16 rows) that serves ``n`` <= 160 rows."""
    return next(k for k in NRF_VARIANTS if 16 * k >= n)


@functools.cache
def _chain_exe(nrf: int):
    return _kernel.compile_markov_chain(nrf)


@functools.cache
def _workspace(device: torch.device, nrf: int):
    """Tile winners and the epoch of one variant on one device: every launch
    rewrites all of its winners and bumps its epoch, so a tag only ever meets
    the entries of the launch before (zeros once; no reset)."""
    winners = torch.zeros(
        NSPEC * _kernel.NT * 16 * nrf * 2, dtype=torch.int32, device=device
    )
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    return winners, epoch


def _tensor_reason(markov_w1, markov_w2, draft_logits_cache) -> str | None:
    """Arch and the load-time tensors (weights, cache)."""
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        return f"{gfx} is not supported (gfx950 only)"
    if get_cu_num() < _kernel.NT:
        return (
            f"needs {_kernel.NT} co-resident workgroups, the GPU has {get_cu_num()} CUs"
        )
    for name, w in (("markov_w1", markov_w1), ("markov_w2", markov_w2)):
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (V, R):
            return f"{name} must be bf16 [{V}, {R}], got {w.dtype} {tuple(w.shape)}"
        if not w.is_contiguous() or not w.is_cuda:
            return f"{name} must be a contiguous GPU tensor"
        if w.data_ptr() % 16:
            return f"{name} must be 16-byte aligned"
    if markov_w2.device != markov_w1.device:
        return "markov_w1 and markov_w2 must be on one device"
    c = draft_logits_cache
    if c.dtype != torch.bfloat16 or c.dim() != 3 or tuple(c.shape[1:]) != (NSPEC, V):
        return f"draft logits cache must be bf16 [slots, {NSPEC}, {V}]"
    if not c.is_contiguous() or c.numel() * 2 >= 1 << 31:
        return "draft logits cache must be contiguous and < 2 GiB"
    if c.device != markov_w1.device or c.data_ptr() % 8:
        return "draft logits cache must be 8-byte aligned, on the weights' device"
    return None


def markov_chain_sample_unsupported_reason(
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    draft_logits_cache: torch.Tensor,
    *,
    num_speculative_steps: int,
    use_fp64_gumbel: bool,
    draft_vocab_size: int,
    has_confidence_head: bool,
    has_draft_watermarker: bool,
    has_acceptance_estimator: bool,
) -> str | None:
    """Why :func:`flydsl_markov_chain_sample` cannot serve this speculator, or
    None. Everything here is fixed at load time; the per-call tensors (base
    logits, maps, positions) are checked by the op itself.

    The op computes exactly the probabilistic Markov chain: every speculator
    feature it does not compute is a required argument, so a caller cannot
    leave one unchecked. Greedy drafting (no draft logits cache) does not
    reach this helper; the caller keeps its own path for it.
    """
    if num_speculative_steps != NSPEC:
        return f"{num_speculative_steps} draft steps (only {NSPEC})"
    if use_fp64_gumbel:
        return "fp64 Gumbel draws are not supported"
    if draft_vocab_size != V:
        return f"a draft vocabulary of {draft_vocab_size} (only the full {V})"
    if has_confidence_head:
        return "a confidence head is not computed"
    if has_draft_watermarker:
        return "a draft watermarker is not applied"
    if has_acceptance_estimator:
        return "an acceptance estimator is not computed"
    return _tensor_reason(markov_w1, markov_w2, draft_logits_cache)


def markov_chain_sample_supported(
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    draft_logits_cache: torch.Tensor,
    *,
    num_speculative_steps: int,
    use_fp64_gumbel: bool,
    draft_vocab_size: int,
    has_confidence_head: bool,
    has_draft_watermarker: bool,
    has_acceptance_estimator: bool,
) -> bool:
    """Whether :func:`flydsl_markov_chain_sample` serves this speculator (see
    :func:`markov_chain_sample_unsupported_reason`)."""
    return (
        markov_chain_sample_unsupported_reason(
            markov_w1,
            markov_w2,
            draft_logits_cache,
            num_speculative_steps=num_speculative_steps,
            use_fp64_gumbel=use_fp64_gumbel,
            draft_vocab_size=draft_vocab_size,
            has_confidence_head=has_confidence_head,
            has_draft_watermarker=has_draft_watermarker,
            has_acceptance_estimator=has_acceptance_estimator,
        )
        is None
    )


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"[FlyDSL markov_chain_sample] {msg}")


def flydsl_markov_chain_sample(
    base_logits: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_idx: torch.Tensor,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    draft_logits_cache: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> None:
    """The 5 sequential Markov draft steps of every request, one launch per
    160 requests. Writes ``draft_tokens[:n]`` and the cache rows.

    - ``base_logits`` bf16 [n * 5, 129280]: row r * 5 + s is request r, step s
      (the draft lm_head output; unit column stride, any even row stride).
    - ``markov_w1`` / ``markov_w2`` bf16 [129280, 256] contiguous.
    - ``input_ids`` int32 / int64, ``anchor_idx`` int64 [n]: step 0's previous
      token is ``input_ids[anchor_idx[r]]``.
    - ``idx_map`` int32 / int64 [n, 5]: the request-state slot of (r, s); < 0
      marks an inert row (no cache write, sampled greedily).
    - ``sample_pos`` int64 [n, 5]: the predicted token's position (the draw is
      keyed by ``sample_pos - 1``).
    - ``temperature`` fp32 / ``seeds`` int64 [slots].
    - ``draft_logits_cache`` bf16 [slots, 5, 129280] contiguous: the
      pre-temperature logits ``bf16(base + bias)`` of each valid (r, s).
    - ``draft_tokens`` int64 [>= n, 5] (unit column stride).

    The speculator-level eligibility (draft steps, Gumbel precision, draft
    vocabulary, confidence head, watermarker, acceptance estimator) is the
    caller's to check with :func:`markov_chain_sample_supported`; this op
    checks the tensors it is given and raises ``ValueError`` on a mismatch.

    Every token is in [0, 129280): a row with no winner (all NaN / inf) gets
    token 0. Deterministic, graph safe, no host sync. Call
    :func:`warmup_markov_chain_sample` before graph capture.
    """
    n = anchor_idx.shape[0]
    base, cache, tokens = base_logits, draft_logits_cache, draft_tokens
    reason = _tensor_reason(markov_w1, markov_w2, cache)
    _check(reason is None, str(reason))
    _check(
        base.dtype == torch.bfloat16 and tuple(base.shape) == (n * NSPEC, V),
        f"base_logits must be bf16 [{n * NSPEC}, {V}]",
    )
    _check(
        base.stride(1) == 1 and base.stride(0) % 2 == 0 and base.data_ptr() % 8 == 0,
        "base_logits needs unit column stride, an even row stride, 8-byte alignment",
    )
    _check(base.shape[0] * base.stride(0) * 2 < 1 << 31, "base_logits spans >= 2 GiB")
    _check(
        input_ids.dtype in (torch.int32, torch.int64) and input_ids.is_contiguous(),
        "input_ids must be contiguous int32 / int64",
    )
    _check(
        anchor_idx.dtype == torch.int64 and anchor_idx.is_contiguous(),
        "anchor_idx must be contiguous int64",
    )
    _check(
        idx_map.dtype in (torch.int32, torch.int64)
        and tuple(idx_map.shape) == (n, NSPEC)
        and idx_map.stride(1) == 1,
        f"idx_map must be int32 / int64 [{n}, {NSPEC}] with unit column stride",
    )
    _check(
        sample_pos.dtype == torch.int64
        and tuple(sample_pos.shape) == (n, NSPEC)
        and sample_pos.stride(1) == 1,
        f"sample_pos must be int64 [{n}, {NSPEC}] with unit column stride",
    )
    _check(
        temperature.dtype == torch.float32
        and temperature.is_contiguous()
        and seeds.dtype == torch.int64
        and seeds.is_contiguous()
        and seeds.shape == temperature.shape,
        "temperature / seeds must be contiguous fp32 / int64 of one shape",
    )
    _check(
        tokens.dtype == torch.int64
        and tokens.dim() == 2
        and tokens.shape[0] >= n
        and tokens.shape[1] == NSPEC
        and tokens.stride(1) == 1,
        f"draft_tokens must be int64 [>= {n}, {NSPEC}] with unit column stride",
    )
    dev = base.device
    for name, t in (
        ("markov_w1", markov_w1),
        ("markov_w2", markov_w2),
        ("input_ids", input_ids),
        ("anchor_idx", anchor_idx),
        ("idx_map", idx_map),
        ("sample_pos", sample_pos),
        ("temperature", temperature),
        ("seeds", seeds),
        ("draft_logits_cache", cache),
        ("draft_tokens", tokens),
    ):
        _check(t.device == dev, f"{name} is on {t.device}, base_logits on {dev}")
    stream = torch.cuda.current_stream(dev)
    ide = input_ids.element_size() // 4
    idxe = idx_map.element_size() // 4
    for r0 in range(0, n, MAX_ROWS):
        m = min(MAX_ROWS, n - r0)
        nrf = nrf_for(m)
        winners, epoch = _workspace(dev, nrf)
        _run_compiled(
            _chain_exe(nrf),
            ptr_arg(markov_w2),
            ptr_arg(markov_w1),
            ptr_arg(input_ids),
            ide,
            ptr_arg(anchor_idx[r0:]),
            ptr_arg(tokens[r0:]),
            tokens.stride(0),
            ptr_arg(base[r0 * NSPEC :]),
            base.stride(0),
            ptr_arg(cache),
            ptr_arg(idx_map[r0:]),
            idx_map.stride(0) * idxe,
            idxe,
            ptr_arg(sample_pos[r0:]),
            sample_pos.stride(0),
            ptr_arg(temperature),
            ptr_arg(seeds),
            ptr_arg(winners),
            ptr_arg(epoch),
            m,
            _POS_ADD,
            stream,
            specialization_key=dev.index,
        )


def warmup_markov_chain_sample(
    device: torch.device,
    idx_dtype: torch.dtype = torch.int64,
    ids_dtype: torch.dtype = torch.int32,
) -> None:
    """Compile every row-count variant and allocate the workspaces, before
    graph capture (inert rows: no cache writes)."""
    w = torch.zeros(V, R, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(1, dtype=ids_dtype, device=device)
    cache = torch.zeros(1, NSPEC, V, dtype=torch.bfloat16, device=device)
    temp = torch.zeros(1, dtype=torch.float32, device=device)
    seeds = torch.zeros(1, dtype=torch.int64, device=device)
    for k in NRF_VARIANTS:
        n = 16 * k
        flydsl_markov_chain_sample(
            torch.zeros(n * NSPEC, V, dtype=torch.bfloat16, device=device),
            w,
            w,
            ids,
            torch.zeros(n, dtype=torch.int64, device=device),
            torch.full((n, NSPEC), -1, dtype=idx_dtype, device=device),
            torch.zeros(n, NSPEC, dtype=torch.int64, device=device),
            temp,
            seeds,
            cache,
            torch.zeros(n, NSPEC, dtype=torch.int64, device=device),
        )
    torch.cuda.synchronize(device)
