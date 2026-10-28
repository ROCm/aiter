# SPDX-License-Identifier: MIT
"""The decoder-loop hook vLLM's ``MiniMaxM3Model.forward`` calls (integration/patch_model.py).

``VLLM_M3_MONO=1``: a decode step of 1..MAX_TOKENS rows runs layers 0..2 on vLLM's
modules and 3..59 on the mono runner. The runner is created on the first forward
(vLLM's eager profiling run, before any graph capture) and compiles the kernels of
every graph width then. ``VLLM_M3_MONO_CHECK=1`` (eager only): the step also runs on
vLLM's modules, the difference is logged, vLLM's result is returned.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm.models.minimax_m3.amd.mono.config import MAX_TOKENS

logger = logging.getLogger(__name__)

N_DENSE = 3
ENABLED = os.environ.get("VLLM_M3_MONO", "0") == "1"
CHECK = os.environ.get("VLLM_M3_MONO_CHECK", "0")  # 1: whole step, layer: per layer
BACKEND = os.environ.get("VLLM_M3_MONO_BACKEND", "flydsl")


def _runner_cls():
    if BACKEND == "gluon":
        from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonM3Runner

        return GluonM3Runner
    from vllm.models.minimax_m3.amd.mono.vllm_runner import M3MonoRunner

    return M3MonoRunner


def get_runner(model, vllm_config):
    """The model's runner, created on the first call (collective); None when off
    or refused (logged once)."""
    if not ENABLED:
        return None
    r = getattr(model, "_m3_mono", None)
    if r is None:
        from vllm.models.minimax_m3.amd.mono.fw.runtime.consensus import MonoUnsupported

        try:
            r = _runner_cls()(model, vllm_config)
            sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
            mt = r.max_tokens() if hasattr(r, "max_tokens") else MAX_TOKENS
            q = getattr(r, "ql", 1)
            widths = sorted({w for w in sizes if w <= mt and w % q == 0} | {q})
            for w in widths:
                if not r.prepare(w):
                    raise MonoUnsupported(f"width {w} not prepared")
            logger.info("MiniMax-M3 mono (%s) on: widths %s ready", BACKEND, widths)
        except MonoUnsupported as e:
            logger.warning("MiniMax-M3 mono off: %s", e)
            r = False
        model._m3_mono = r
    return r or None


def supports(runner, hidden: torch.Tensor) -> bool:
    """A pure decode step the kernels serve (TP-uniform inputs only)."""
    n = hidden.shape[0]
    if not 1 <= n <= (runner.max_tokens() if hasattr(runner, "max_tokens") else MAX_TOKENS):
        return False
    md = runner.sparse_metadata()
    if md is None or getattr(md, "decode", None) is None:
        return False
    if getattr(md, "num_prefills", 0):
        return False
    q_len = int(md.decode.decode_query_len)
    if n % q_len or q_len != getattr(runner, "ql", q_len):
        return False
    if not runner.k4.prepare(n):
        return False
    if getattr(runner, "_caches_checked", False) is False:
        runner.check_caches()
        runner._caches_checked = True
    return True


def _nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def run_layers(model, positions, hidden, residual, vllm_config, aux=None):
    """The decoder loop with the mono hand-off; returns (hidden, residual). ``aux``:
    EAGLE3's aux hidden-state list (vLLM's ``_maybe_add_hidden_state``), appended to."""
    layers = model.layers[model.start_layer : model.end_layer]
    runner = get_runner(model, vllm_config) if model.start_layer == 0 else None
    use = runner is not None and supports(runner, hidden)
    for idx, layer in enumerate(layers):
        if use and idx == N_DENSE:
            break
        hidden, residual = layer(positions, hidden, residual)
        if aux is not None:
            model._maybe_add_hidden_state(aux, idx + 1, hidden, residual)
    if not use:
        return hidden, residual
    if CHECK == "layer" and not torch.cuda.is_current_stream_capturing():
        # each mono layer on the input vLLM's layer saw, then vLLM's layer
        n = hidden.shape[0]
        runner.begin(positions, n)
        worst = []
        for i, layer in enumerate(layers[N_DENSE:]):
            runner.ars[i % 2][:n].copy_(hidden)
            r_m = runner.layer(i, residual).clone()
            h_m = runner.ars[(i + 1) % 2][:n].clone()
            hidden, residual = layer(positions, hidden, residual)
            worst.append((_nrmse(h_m, hidden), _nrmse(r_m, residual), N_DENSE + i))
        if runner.debug:
            runner.raise_on_given_up_waits(n)
        hs = sorted(worst, reverse=True)
        logger.info("M3 mono layer check n=%d: hidden nRMSE median %.2e max %.2e (layer %d); residual max %.2e (layer %d)",
                    n, sorted(w[0] for w in worst)[len(worst) // 2], hs[0][0], hs[0][2],
                    max(w[1] for w in worst), max(worst, key=lambda w: w[1])[2])
        return hidden, residual
    h_mono, r_mono = runner.forward(positions, hidden, residual)
    if aux is not None and getattr(model, "aux_hidden_state_layers", None):
        aux.extend(runner.aux_states(hidden.shape[0]))
    if CHECK == "1" and not torch.cuda.is_current_stream_capturing():
        h_m, r_m = h_mono.clone(), r_mono.clone()
        for layer in layers[N_DENSE:]:
            hidden, residual = layer(positions, hidden, residual)
        logger.info("M3 mono check n=%d: hidden nRMSE %.3e residual nRMSE %.3e", hidden.shape[0],
                    _nrmse(h_m, hidden), _nrmse(r_m, residual))
        return hidden, residual
    return h_mono, r_mono
