# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_VMOE=1: a Qwen3.5/3.6 sparse MoE block (router, top-8, MXFP4 routed experts, shared expert, gated add) as one
gfx1250 FlyDSL dataflow kernel (ATOM_VMOE_DIR: the kernel sources, vmoe_atom.py / vmoe2.py).

TP1 / DP decode steps of at most 16 rows; other steps run ATOM's own chain. The block returns the MoE sum as a plain
tensor, so the next norm adds the residual on its unfused path. Weights come from ATOM_VMOE_CKPT (a checkpoint directory
with layers-N.safetensors): experts, router and shared gate are the served tensors; the shared expert must be MXFP8 1x32
there (the served bf16 shared expert is NOT what the kernel runs).
ATOM_VMOE_CHECK=1 (eager calls): ATOM's chain runs too and the relative difference is logged."""
from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger("atom")
ENABLED = os.environ.get("ATOM_VMOE", "0") == "1"
CHECK = os.environ.get("ATOM_VMOE_CHECK", "0") == "1"
# ATOM_VMOE_SHADOW=1: ATOM's chain serves; the kernel runs beside it on the same rows (also inside the decode graphs) and
# the squared difference is accumulated on the GPU per layer, dumped every 15 s to ATOM_VMOE_SHADOW_OUT.<rank>.json.
SHADOW = os.environ.get("ATOM_VMOE_SHADOW", "0") == "1"
MIX = os.environ.get("ATOM_VMOE_MIX", "")
XCLONE = os.environ.get("ATOM_VMOE_XCLONE", "0") == "1"
ACC = os.environ.get("ATOM_VMOE_ACC", "0") == "1"
PRESHADOW = os.environ.get("ATOM_VMOE_PRESHADOW", "0") == "1"
PADK = int(os.environ.get("ATOM_VMOE_PADK", "0"))
PREK = int(os.environ.get("ATOM_VMOE_PREK", "0"))
LITE = os.environ.get("ATOM_VMOE_LITE", "0") == "1"
_LITE = {}
_PAD_T = {}
LATE = os.environ.get("ATOM_VMOE_LATE", "1") == "1"   # with ACC: serve the late copy (1) or the immediate one (0)
OCLONE = os.environ.get("ATOM_VMOE_OCLONE", "0") == "1"
MAX_ROWS = 16
_ACC = None
_WORST = {}
_TILE = {}
_DUMP = [] if os.environ.get("ATOM_VMOE_DUMP") else None
_IMPL = None
_FAILED = False
_STATS = {"kernel": 0, "native_rows": 0, "native_capture": 0, "checks": 0}


def _impl():
    """The kernel runner with every layer's weights loaded, or None. Never built inside a graph capture."""
    global _IMPL, _FAILED
    if _IMPL is not None or _FAILED:
        return _IMPL
    if torch.cuda.is_current_stream_capturing():
        _STATS["native_capture"] += 1
        if _STATS["native_capture"] == 1:
            logger.warning("ATOM_VMOE: first call inside a graph capture; ATOM's MoE for this graph")
        return None
    try:
        sys.path.insert(0, os.environ["ATOM_VMOE_DIR"])
        import vmoe_atom

        dev = torch.device("cuda", torch.cuda.current_device())
        impl = vmoe_atom.VMoE(os.environ["ATOM_VMOE_CKPT"], dev)
        n = int(os.environ.get("ATOM_VMOE_LAYERS", "40"))
        for i in range(n):
            impl.load_layer(i)
        impl.run(0, torch.zeros(1, 2048, dtype=torch.bfloat16, device=dev))   # compile + first launch
        torch.cuda.synchronize(dev)
        _IMPL = impl
        _PAD_T[dev.index if dev.index is not None else 0] = torch.zeros(4096, device=dev)
        logger.info("ATOM_VMOE: MoE dataflow kernel ready on %s, %d layers, %.1f GB allocated (check=%s)",
                    dev, n, torch.cuda.memory_allocated(dev) / 2**30, CHECK)
    except Exception:  # noqa: BLE001 - the block falls back to ATOM's chain
        logger.exception("ATOM_VMOE: init failed, ATOM's MoE chain runs")
        _FAILED = True
    return _IMPL


def _materialize(ref):
    """ATOM's block output as one tensor (it may be left unsummed for the next norm)."""
    from atom.model_ops.moe_tail_norm import GatheredRouted, MoEGatedSum

    if isinstance(ref, MoEGatedSum):
        routed = ref.routed.materialize() if isinstance(ref.routed, GatheredRouted) else ref.routed
        routed = routed.reshape(-1, routed.shape[-1]).float()
        shared = ref.shared.reshape(-1, ref.shared.shape[-1]).float()
        if ref.gate_logits is None:
            return routed + shared
        return routed + torch.sigmoid(ref.gate_logits.float().reshape(-1, 1)) * shared
    if isinstance(ref, GatheredRouted):
        ref = ref.materialize()
    return ref.reshape(-1, ref.shape[-1]).float()


def _dump_loop(acc, path):
    import json
    import time

    while True:
        time.sleep(15)
        try:
            a = acc.detach().to("cpu", copy=True).tolist()
            with open(path, "w") as f:
                json.dump(a, f)
            if _TILE:
                with open(path.replace(".json", ".tile.json"), "w") as f:
                    json.dump({k: v.detach().to("cpu", copy=True).tolist() for k, v in _TILE.items()}, f)
            if _WORST:
                torch.save({k: v.detach().to("cpu", copy=True) for k, v in _WORST.items()}, path.replace(".json", ".worst.pt"))
        except Exception:  # noqa: BLE001
            logger.exception("ATOM_VMOE_SHADOW: dump failed")
            return


_PRE = [None]


def pre(block, x: torch.Tensor) -> None:
    """ATOM_VMOE_PRESHADOW=1 (with SHADOW): the kernel runs BEFORE ATOM's chain, where it runs when it serves (right
    after the kernel that wrote x); `shadow` then compares that output. The model still consumes ATOM's."""
    _PRE[0] = None
    T = x.shape[0]
    if "mtp" in block.prefix.lower() or T == 0 or T > MAX_ROWS or block.tp_size != 1 or x.dtype != torch.bfloat16:
        return
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    if context is None or context.is_prefill:
        return
    impl = _impl()
    if impl is None:
        return
    layer = getattr(block, "_vmoe_layer", None)
    if layer is None:
        from atom.models.utils import extract_layer_index

        layer = extract_layer_index(block.prefix)
        object.__setattr__(block, "_vmoe_layer", layer)
    out = impl.run(layer, x.contiguous())
    # the scratch the statistics read is overwritten by nothing until the next launch; keep copies of the parts anyway
    _PRE[0] = (out, impl.buf["slot"][:T].sum(1), impl.buf["sgate"][:T].unsqueeze(1).clone(), impl.buf["sdn"][:T].clone())


def shadow(block, x: torch.Tensor, native_out, kout=None) -> None:
    """Run the kernel beside ATOM's chain and accumulate |kernel - native|^2, |native|^2, calls and rows per layer."""
    global _ACC
    T = x.shape[0]
    if "mtp" in block.prefix.lower() or T == 0 or T > MAX_ROWS or block.tp_size != 1 or x.dtype != torch.bfloat16:
        return
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    if context is None or context.is_prefill:
        return
    impl = _impl()
    if impl is None:
        return
    if _ACC is None:
        if torch.cuda.is_current_stream_capturing():
            return
        import threading

        _ACC = torch.zeros(64, 32, dtype=torch.float64, device=x.device)
        z_ = lambda *sh: torch.zeros(*sh, dtype=torch.float32, device=x.device)
        zd = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=x.device)
        _TILE.update(n=zd(()), tile=zd(128), row=zd(MAX_ROWS), layer=zd(64), T=zd(MAX_ROWS + 1), rows_with=zd(()), calls_with=zd(()), maxel=z_(()), late_rows=zd(()), late_calls=zd(()), late_T=zd(MAX_ROWS + 1), late_row=zd(MAX_ROWS))
        _WORST.update(err=z_(()), meta=z_(2), x=z_(MAX_ROWS, 2048), kr=z_(MAX_ROWS, 2048), nr=z_(MAX_ROWS, 2048), ks=z_(MAX_ROWS, 2048),
                      ns=z_(MAX_ROWS, 2048), out=z_(MAX_ROWS, 2048), sel=z_(MAX_ROWS, 8), lg=z_(MAX_ROWS, 256))
        path = "%s.%d.json" % (os.environ.get("ATOM_VMOE_SHADOW_OUT", "/tmp/vmoe_shadow"), torch.cuda.current_device())
        threading.Thread(target=_dump_loop, args=(_ACC, path), daemon=True).start()
        logger.info("ATOM_VMOE_SHADOW: accumulating to %s", path)
    layer = getattr(block, "_vmoe_layer", None)
    if layer is None:
        from atom.models.utils import extract_layer_index

        layer = extract_layer_index(block.prefix)
        object.__setattr__(block, "_vmoe_layer", layer)
    ref = _materialize(native_out)
    prep = _PRE[0] if PRESHADOW else None
    _PRE[0] = None
    if PRESHADOW and prep is None:
        return
    if prep is not None:
        kout = prep[0]
    out = (impl.run(layer, x.contiguous()) if kout is None else kout).float()
    xf = torch.isfinite(x.float()).all(1)
    ok = xf & torch.isfinite(ref).all(1) & torch.isfinite(out).all(1)
    okf = ok.to(out.dtype).unsqueeze(1)
    d = torch.where(ok.unsqueeze(1), out - ref, torch.zeros_like(out))
    r = torch.where(ok.unsqueeze(1), ref, torch.zeros_like(ref))
    _ACC[layer, 0] += d.pow(2).sum()                  # |kernel - native|^2 over rows finite everywhere
    _ACC[layer, 1] += r.pow(2).sum()
    _ACC[layer, 2] += 1                               # calls
    _ACC[layer, 3] += T                               # rows
    _ACC[layer, 4] += (~xf).sum()                     # rows whose input is not finite (pad rows)
    _ACC[layer, 5] += (xf & ~torch.isfinite(out).all(1)).sum()    # finite input, non-finite kernel output
    _ACC[layer, 6] += (xf & ~torch.isfinite(ref).all(1)).sum()    # finite input, non-finite native output
    _ACC[layer, 7] += ok.sum()
    _ACC[layer, 8] += (d.pow(2).sum(1) > 0.25 * r.pow(2).sum(1)).sum()   # rows off by more than 50% in norm
    _ACC[40 + T, 0] += d.pow(2).sum()                 # the same by rows a call (rows 41..56 of the table)
    _ACC[40 + T, 1] += r.pow(2).sum()
    _ACC[40 + T, 2] += 1
    _ACC[40 + T, 3] += T
    _ACC[40 + T, 5] += (xf & ~torch.isfinite(out).all(1)).sum()
    _ACC[40 + T, 8] += (d.pow(2).sum(1) > 0.25 * r.pow(2).sum(1)).sum()
    _ACC[40 + T, 9] += (d.pow(2).sum(1) > 4.0 * r.pow(2).sum(1)).sum()      # rows off by more than 200%
    # per 16-column tile (one combine CTA's columns of a row): a wrong tile barely moves the row's norm
    dt = d.view(T, 128, 16).pow(2).sum(-1)
    rt = r.view(T, 128, 16).pow(2).sum(-1)
    floor = (r.pow(2).sum(1, keepdim=True) / 128.0) * 0.25
    badt = dt > torch.maximum(rt, floor)              # tile off by more than 100% (of itself, or of half a typical tile)
    _TILE["n"] += badt.sum()
    _TILE["tile"] += badt.sum(0)
    _TILE["row"][:T] += badt.sum(1)
    _TILE["layer"][layer] += badt.sum()
    _TILE["T"][T] += badt.sum()
    _TILE["rows_with"] += badt.any(1).sum()
    _TILE["calls_with"] += badt.any()
    _TILE["maxel"].copy_(torch.maximum(_TILE["maxel"], (d.abs().amax(1) / r.abs().amax(1).clamp_min(1e-20)).max()))
    from atom.model_ops.moe_tail_norm import GatheredRouted, MoEGatedSum

    if isinstance(native_out, MoEGatedSum) and native_out.gate_logits is not None:      # the parts, separately
        nr = native_out.routed.materialize() if isinstance(native_out.routed, GatheredRouted) else native_out.routed
        nr = nr.reshape(-1, nr.shape[-1]).float()
        ng = torch.sigmoid(native_out.gate_logits.float().reshape(-1, 1))
        ns = native_out.shared.reshape(-1, native_out.shared.shape[-1]).float()
        kr = impl.buf["slot"][:T].sum(1)
        kg = impl.buf["sgate"][:T].unsqueeze(1)
        ks = impl.buf["sdn"][:T]
        if prep is not None:
            kr, kg, ks = prep[1], prep[2], prep[3]
        _ACC[layer, 12] += (kr - nr).pow(2).sum()     # routed experts
        _ACC[layer, 13] += nr.pow(2).sum()
        _ACC[layer, 14] += (ks - ns).pow(2).sum()     # shared expert, before the gate
        _ACC[layer, 15] += ns.pow(2).sum()
        _ACC[layer, 16] += (kg - ng).pow(2).sum()     # the gate
        _ACC[layer, 17] += ng.pow(2).sum()
        _ACC[layer, 18] += (ng * ns).pow(2).sum()     # gated shared part of the native sum
        if _DUMP is not None and not torch.cuda.is_current_stream_capturing() and len(_DUMP) < 24 and layer in (0, 17, 32):
            # eager calls only (warm-up rows): inputs and both sides' parts, for offline analysis
            _DUMP.append(dict(layer=layer, x=x.detach().cpu().clone(), nr=nr.detach().cpu().clone(), ns=ns.detach().cpu().clone(),
                              ng=ng.detach().cpu().clone(), kr=kr.detach().cpu().clone(), ks=ks.detach().cpu().clone(),
                              sel=impl.buf["sel"][:T].detach().cpu().clone(), lg=impl.buf["lg"][:T].detach().cpu().clone()))
            if len(_DUMP) in (6, 12, 24):
                torch.save(_DUMP, "%s.%d.pt" % (os.environ["ATOM_VMOE_DUMP"], torch.cuda.current_device()))
        # the call with the worst routed row so far: its input and both sides' parts, kept on the GPU (graph-safe: no
        # Python branch) and dumped by the thread, to replay offline
        rre = torch.nan_to_num((kr - nr).pow(2).sum(1).sqrt() / nr.pow(2).sum(1).sqrt().clamp_min(1e-20), nan=1e9, posinf=1e9)
        tworst = torch.nan_to_num((dt / torch.maximum(rt, floor)).max(), nan=1e9, posinf=1e9)
        worse = tworst > _WORST["err"]
        pad = (0, 0, 0, MAX_ROWS - T)
        for key, val in (("x", x.float()), ("kr", kr), ("nr", nr), ("ks", ks), ("ns", ns), ("out", out), ("sel", impl.buf["sel"][:T].float()), ("lg", impl.buf["lg"][:T])):
            _WORST[key].copy_(torch.where(worse, torch.nn.functional.pad(val, pad), _WORST[key]))
        _WORST["meta"].copy_(torch.where(worse, torch.stack([torch.full((), float(layer), device=x.device), torch.full((), float(T), device=x.device)]), _WORST["meta"]))
        _WORST["err"].copy_(torch.where(worse, tworst, _WORST["err"]))
        _ACC[40 + T, 10] += (rre > 0.2).sum()         # routed rows off by more than 20% (A4: ATOM's own formats, ~3% expected)
        _ACC[40 + T, 11] += (rre > 1.0).sum()
        fin = torch.isfinite(kr).all(1) & torch.isfinite(nr).all(1)
        krf = torch.where(fin.unsqueeze(1), kr, torch.zeros_like(kr))
        nrf = torch.where(fin.unsqueeze(1), nr, torch.zeros_like(nr))
        rr = (krf - nrf).pow(2).sum(1).sqrt() / nrf.pow(2).sum(1).sqrt().clamp_min(1e-20)      # routed rel error of each row
        for i_, (lo, hi) in enumerate(((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e30))):
            _ACC[layer, 20 + i_] += (fin & (rr >= lo) & (rr < hi)).sum()
        _ACC[layer, 26] += (krf * nrf).sum()          # <kernel, native>, |kernel|^2, |native|^2 of the routed sum (finite rows)
        _ACC[layer, 27] += krf.pow(2).sum()
        _ACC[layer, 28] += nrf.pow(2).sum()
        _ACC[layer, 29] += (~fin).sum()
        _ACC[layer, 30] += (fin & (nrf.abs().amax(1) == 0)).sum()     # rows whose native routed sum is all zero
        _ACC[layer, 31] += (krf - nrf).pow(2).sum()


def vmoe_forward(block, x: torch.Tensor):
    """The block's output rows [T][hidden] bf16, or None when this step stays on ATOM's chain."""
    if SHADOW:
        return None
    T = x.shape[0]
    if "mtp" in block.prefix.lower():                 # the drafter's block: other weights (and its own kernel)
        return None
    if T == 0 or T > MAX_ROWS or block.tp_size != 1 or x.dtype != torch.bfloat16 or torch.compiler.is_compiling():
        if T > MAX_ROWS:
            _STATS["native_rows"] += 1
        return None
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    if context is None or context.is_prefill:
        return None
    impl = _impl()
    if impl is None:
        return None
    layer = getattr(block, "_vmoe_layer", None)
    if layer is None:
        from atom.models.utils import extract_layer_index

        layer = extract_layer_index(block.prefix)
        object.__setattr__(block, "_vmoe_layer", layer)
        logger.info("ATOM_VMOE: block %s -> layer %d", block.prefix, layer)
    if layer not in impl.layers:
        return None
    x = x.contiguous()
    if MIX:
        # diagnostic: the block's sum from chosen sides, returned as a plain tensor like the kernel's
        #   native: ATOM's routed + ATOM's shared (tests the plain-tensor return path alone)
        #   routed: the kernel's routed experts + ATOM's shared;  shared: ATOM's routed + the kernel's shared
        from atom.model_ops.moe_tail_norm import GatheredRouted, MoEGatedSum

        if MIX == "kpre":
            # the kernel FIRST, in its serving position, its immediate copy served; ATOM's chain after it (discarded, or
            # with ACC the reference of the shadow statistics on the trajectory the kernel serves)
            kout = impl.run(layer, x)
            first = kout.clone()
            nat = block.native_moe_forward(x)
            if LITE and "c" in _LITE:                 # a handful of kernels: rows of the served copy off ATOM's by > 20%
                ref = _materialize(nat)
                bad = (first.float() - ref).pow(2).sum(1) > 0.04 * ref.pow(2).sum(1)
                _LITE["c"] += torch.stack([bad.sum(), bad.any().sum(), torch.ones_like(bad[0]).sum(), (first.view(torch.int16) != kout.view(torch.int16)).any(1).sum()]).double()
            elif LITE and not torch.cuda.is_current_stream_capturing():
                import threading

                _LITE["c"] = torch.zeros(4, dtype=torch.float64, device=x.device)
                path = "%s.%d.lite.json" % (os.environ.get("ATOM_VMOE_SHADOW_OUT", "/tmp/vmoe_shadow"), torch.cuda.current_device())
                threading.Thread(target=_dump_loop, args=(_LITE["c"], path), daemon=True).start()
            if ACC:
                shadow(block, x, nat, kout)
                last = kout.clone()
                chg = (first.view(torch.int16) != last.view(torch.int16)).any(1)
                _TILE["late_rows"] += chg.sum()
                _TILE["late_calls"] += chg.any()
                _TILE["late_T"][T] += chg.sum()
                _TILE["late_row"][:T] += chg
            return first
        nat = block.native_moe_forward(x)
        if not (isinstance(nat, MoEGatedSum) and nat.gate_logits is not None):
            return None
        nr = nat.routed.materialize() if isinstance(nat.routed, GatheredRouted) else nat.routed
        nr = nr.reshape(-1, nr.shape[-1]).float()
        ng = torch.sigmoid(nat.gate_logits.float().reshape(-1, 1))
        ns = nat.shared.reshape(-1, nat.shared.shape[-1]).float()
        if MIX != "native":
            kout = impl.run(layer, x)
        if MIX == "both":                             # the kernel's parts, summed here (not by its combine)
            return (impl.buf["slot"][:T].sum(1) + impl.buf["sgate"][:T].unsqueeze(1) * impl.buf["sdn"][:T]).to(torch.bfloat16)
        if MIX == "kout":                             # the kernel's own sum (ATOM's chain also ran, unused)
            return kout
        if MIX == "kclone":
            if ACC:                                   # the shadow statistics on the trajectory the kernel itself serves
                first = kout.clone()                  # the kernel's output as the very next kernel on the stream reads it
                shadow(block, x, nat, kout)
                last = kout.clone()                   # ... and as it reads some sixty kernels later
                chg = (first.view(torch.int16) != last.view(torch.int16)).any(1)
                _TILE["late_rows"] += chg.sum()
                _TILE["late_calls"] += chg.any()
                _TILE["late_T"][T] += chg.sum()
                _TILE["late_row"][:T] += chg
                return last if LATE else first
            return kout.clone()
        if MIX == "routed":
            return (impl.buf["slot"][:T].sum(1) + ng * ns).to(torch.bfloat16)
        if MIX == "shared":
            return (nr + impl.buf["sgate"][:T].unsqueeze(1) * impl.buf["sdn"][:T]).to(torch.bfloat16)
        return (nr + ng * ns).to(torch.bfloat16)
    if XCLONE:                                        # diagnostic: the kernel reads a copy of x (one kernel between x's producer and it)
        x = x.clone()
    capturing = torch.cuda.is_current_stream_capturing()
    ref = None
    if CHECK and not capturing:
        ref = _materialize(block.native_moe_forward(x))
    for _ in range(PREK):                             # diagnostic: unrelated small kernels right before the launch
        _PAD_T[x.device.index if x.device.index is not None else 0].add_(1.0)
    out = impl.run(layer, x)
    for _ in range(PADK):                             # diagnostic: unrelated small kernels right after the launch
        _PAD_T[x.device.index if x.device.index is not None else 0].add_(1.0)
    if OCLONE:                                        # diagnostic: an exact-size copy of the output is returned
        out = out.clone()
    _STATS["kernel"] += 1
    if ref is not None:
        _STATS["checks"] += 1
        if _STATS["checks"] <= 120 or _STATS["checks"] % 2000 == 0:
            o = out.float()
            logger.info("ATOM_VMOE_CHECK layer %d T %d: rel %.3e (max abs %.3e of %.3e) call %d", layer, T,
                        ((o - ref).norm() / ref.norm().clamp_min(1e-30)).item(), (o - ref).abs().max().item(),
                        ref.abs().max().item(), _STATS["checks"])
    if _STATS["kernel"] in (1, 1000, 100000):
        logger.info("ATOM_VMOE: calls %s (python-side; graph replays are not counted)", _STATS)
    return out
