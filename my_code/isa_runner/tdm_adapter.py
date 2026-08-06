#!/usr/bin/env python3
"""Feed the real gemm1/gemm2 TDM inputs into a hand-edited ISA.

Building valid inputs by hand is not practical: A is preshuffled MXFP8, W is
preshuffled MXFP4, the scales are n32k4-folded, and m_tile_map is a per-expert
psum. Any of those constructed wrongly gives a plausible-looking wrong answer.

So instead of *constructing* inputs, this **captures** them. It monkey-patches
the FlyDSL launch, runs the production path once, and records the device
pointers and scalars the kernel was actually called with. Those exact arguments
are then replayed into a code object built by isa_runner from a .s file, so a
hand-edited ISA runs against production data.

    python tdm_adapter.py replay --which gemm1 --isa edited.s

The buffers live on the GPU only for the lifetime of the process, so production
capture and candidate replay always happen in the same invocation.

    python tdm_adapter.py run --isa edited.s --iters 100

Correctness is checked against the selected production kernel's output, cloned
immediately after that launch. A mismatch therefore means the candidate ISA
changed the selected GEMM's behaviour.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]          # <repo>/my_code/isa_runner -> <repo>
# The container also ships an older aiter at /app/aiter that lacks the TDM
# kernel; the repo checkout must win, which is why sweep_tdm.sh cds to the repo
# root. Put it ahead of everything before any aiter import happens.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "op_tests"))
sys.path.insert(0, str(_REPO))

from isa_runner import IsaModule, IsaRunnerError, KernelLaunchSpec, build  # noqa: E402

# The 15 kernargs, in order, as they are packed into the 104-byte kernarg
# segment. Verified against both the .amdgpu_metadata offsets in
# 21_final_isa.s and the s_load offsets the ISA actually reads: every pointer
# below is read by the kernel, and the three "*_desc" i32 slots are not
# (they are the tensor-descriptor words fx.Tensor lowers to).
KERNARG_LAYOUT = [
    ("arg_c", "ptr", 0),
    ("c_desc", "i32", 8),
    ("arg_a", "ptr", 16),
    ("arg_b", "ptr", 24),
    ("arg_scale_a", "ptr", 32),
    ("scale_a_desc", "i32", 40),
    ("arg_scale_b", "ptr", 48),
    ("scale_b_desc", "i32", 56),
    ("arg_m_tile_map", "ptr", 64),
    ("arg_bias", "ptr", 72),
    ("arg_quant_scale", "ptr", 80),
    ("quant_scale_desc", "i32", 88),
    ("i32_m", "i32", 92),
    ("i32_n", "i32", 96),
    ("f32_swiglu_limit", "f32", 100),
]
KERNARG_SIZE = 104

# Written before both production and candidate launches: a value the kernel
# would never produce, so "never written" differs from "wrote zero".
_POISON = -12345.0
_REL_L2_TOL = 1e-6


def arena_bytes(tile_m, tile_n, tile_k, num_buffers, m_warp=1, n_warp=4,
                a_is_fp4=0) -> int:
    """Dynamic LDS the kernel allocates, mirroring the frontend's arena math.

    The descriptor's group_segment_fixed_size is 0 because the arena is a
    dynamic SharedAllocator, so the launch must supply this. Validated against
    both the kernel's own ``ARENA=`` log line (158208 for 64x256x256_b3) and the
    dynamic_shared_memory_size in 19_gpu_module_to_binary.mlir (159744 -- the
    same arena rounded up by the tile_m<=64 zero-fill loop).
    """
    a_pack = 2 if a_is_fp4 else 1
    a_row_b = tile_k // a_pack
    stage_a = ((tile_m * (a_row_b + 16) + 15) // 16) * 16
    stage_b = (((tile_n // 16) * ((tile_k // 2) * 16) + 15) // 16) * 16
    wmma_m_rep = (tile_m // m_warp) // 16
    as_supers = tile_m // wmma_m_rep
    as_inner = (tile_k // 128) * wmma_m_rep
    stage_sa = ((as_supers * as_inner * 4 + 15) // 16) * 16
    stage_sb = (((tile_n // 32) * (tile_k // 4) * 4 + 15) // 16) * 16
    pitch = ((stage_a + stage_b + stage_sa + stage_sb + 511) // 512) * 512
    c_store = ((tile_m * tile_n * 2 + 127) // 128) * 128
    arena = max(num_buffers * pitch, c_store)
    if tile_m <= 64:  # zero-fill loop rounds the arena to 16 B * block
        zblk = 16 * (m_warp * n_warp * 32)
        arena = ((arena + zblk - 1) // zblk) * zblk
    return arena


@dataclass
class Capture:
    """One recorded kernel invocation."""
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: dict[str, int | float]
    out_ptr: int
    out_nbytes: int
    lds_bytes: int = 0
    tiles: dict[str, int] = field(default_factory=dict)
    reference: Any = field(default=None, repr=False)  # production tensor clone

    def to_json(self) -> dict:
        return {
            "kernel": self.kernel,
            "grid": list(self.grid),
            "block": list(self.block),
            "args": {k: (v if isinstance(v, float) else int(v))
                     for k, v in self.args.items()},
            "out_ptr": self.out_ptr,
            "out_nbytes": self.out_nbytes,
            "lds_bytes": self.lds_bytes,
            "tiles": self.tiles,
        }

    def pack_kernargs(self) -> list:
        """Build the ctypes arg list in kernarg order."""
        out = []
        for name, kind, _off in KERNARG_LAYOUT:
            v = self.args.get(name, 0)
            if kind == "ptr":
                out.append(ctypes.c_uint64(int(v)))
            elif kind == "i32":
                out.append(ctypes.c_int32(int(v)))
            else:
                out.append(ctypes.c_float(float(v)))
        return out


def _ptr_of(v) -> int:
    """Device pointer behind a flydsl jit arg / torch tensor / raw int."""
    if v is None:
        return 0
    if hasattr(v, "pointer"):  # PointerJitArg
        return int(v.pointer.value or 0)
    if hasattr(v, "data_ptr"):  # torch.Tensor
        return int(v.data_ptr())
    if isinstance(v, ctypes.c_void_p):
        return int(v.value or 0)
    if isinstance(v, int):
        return v
    # DLTensorJitArg / TorchTensorJitArg wrap a tensor
    for attr in ("tensor", "_tensor", "obj"):
        t = getattr(v, attr, None)
        if t is not None and hasattr(t, "data_ptr"):
            return int(t.data_ptr())
    raise IsaRunnerError(f"cannot extract device pointer from {type(v)}")


def _find_output_tensor(arg_c, keepalive):
    """Find the live torch output whose device pointer was passed as arg_c."""
    import torch

    target_ptr = _ptr_of(arg_c)
    seen: set[int] = set()

    def tensors_in(value):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from tensors_in(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from tensors_in(child)
            return
        # FlyDSL argument wrappers have used each of these names.
        for attr in ("tensor", "_tensor", "obj"):
            yield from tensors_in(getattr(value, attr, None))

    for root in (arg_c, *reversed(keepalive)):
        for tensor in tensors_in(root):
            if int(tensor.data_ptr()) == target_ptr:
                return tensor

    raise IsaRunnerError(
        "cannot find a live torch output tensor for "
        f"arg_c pointer 0x{target_ptr:x} in arg_c or the capture keepalive"
    )


def _poison_mask(tensor):
    """Return a poison mask without converting away the tensor's dtype."""
    # In bf16, -12345.0 is stored as -12352.0. Constructing the scalar in the
    # tensor's dtype is therefore required before comparing.
    return tensor.eq(tensor.new_full((), _POISON))


def _compare_outputs(reference, candidate) -> dict[str, Any]:
    """Compare production and candidate outputs with poison-aware padding."""
    import torch

    if reference.shape != candidate.shape:
        raise IsaRunnerError(
            f"output shape changed: production={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    if reference.dtype != candidate.dtype:
        raise IsaRunnerError(
            f"output dtype changed: production={reference.dtype}, "
            f"candidate={candidate.dtype}"
        )
    if reference.device != candidate.device:
        raise IsaRunnerError(
            f"output device changed: production={reference.device}, "
            f"candidate={candidate.device}"
        )

    production_poison = _poison_mask(reference)
    candidate_poison = _poison_mask(candidate)
    padding = production_poison & candidate_poison
    missing = (~production_poison) & candidate_poison
    unexpected = production_poison & (~candidate_poison)
    compared = ~padding

    # Determine poison status above in the original dtype. Float conversion is
    # safe only after the masks have captured the representable sentinel value.
    production_f = reference.float()
    candidate_f = candidate.float()
    zeros = torch.zeros_like(production_f)
    delta = torch.where(padding, zeros, candidate_f - production_f)
    base = torch.where(padding, zeros, production_f)
    denom = base.norm().item() or 1.0
    rel_l2 = delta.norm().item() / denom
    max_abs_diff = delta.abs().max().item() if delta.numel() else 0.0

    production_wrote = int((~production_poison).sum().item())
    candidate_wrote = int((~candidate_poison).sum().item())
    skipped_padding = int(padding.sum().item())
    compared_elems = int(compared.sum().item())
    missing_writes = int(missing.sum().item())
    unexpected_writes = int(unexpected.sum().item())
    return {
        "production_wrote_elems": production_wrote,
        "candidate_wrote_elems": candidate_wrote,
        "skipped_padding_elems": skipped_padding,
        "compared_elems": compared_elems,
        "rel_l2": rel_l2,
        "max_abs_diff": max_abs_diff,
        # Keep the old name in reports while making the failure mode explicit.
        "still_poisoned": missing_writes,
        "missing_writes": missing_writes,
        "unexpected_writes": unexpected_writes,
        "passed": bool(
            rel_l2 < _REL_L2_TOL
            and missing_writes == 0
            and unexpected_writes == 0
            and compared_elems > 0
        ),
    }


def capture_launches(which: str = "gemm1", *, tokens: int = 4096,
                     experts: int = 384, topk: int = 6,
                     model_dim: int = 7168, inter_dim: int = 768) -> Capture:
    """Run the production MoE path once and record the chosen kernel's args.

    Hooks flydsl's kernel launch rather than reimplementing the data prep, so
    the captured pointers are exactly what the real kernel receives.
    """
    import torch

    os.environ.setdefault("ENABLE_CK", "0")
    os.environ.setdefault("AITER_FORCE_GFX1250", "1")
    os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
    # Default to the tuned g2_m64_nb3 tiles so the captured kernel is the same
    # one 21_final_isa.s was dumped from; without this the CSV default
    # (16x256x256_b2) is captured and no dumped ISA matches it.
    for k, v in (("AITER_TDM_TILE_M", "64"), ("AITER_TDM_TILE_N", "256"),
                 ("AITER_TDM_TILE_K", "256"), ("AITER_TDM_NUM_BUFFERS", "3"),
                 ("AITER_TDM_TILE_M2", "64"), ("AITER_TDM_TILE_N2", "512"),
                 ("AITER_TDM_TILE_K2", "128"), ("AITER_TDM_NUM_BUFFERS2", "3")):
        os.environ.setdefault(k, v)

    records: list[Capture] = []

    # Everything the kernel reads must outlive run_moe: the kernel-level args
    # are raw pointers (ptr_arg), so without a Python reference torch's caching
    # allocator reuses those buffers and the replay reads freed memory --
    # which shows up as a NaN result, not as an error. Hold the tensors from
    # the wrapper, which still has them as tensors.
    keepalive: list[Any] = []
    from aiter.ops.flydsl import batched_gemm_mxfp4 as bgm

    real_grouped = bgm.flydsl_grouped_gemm_a8w4_masked

    def grouped_spy(out, a, w, a_scales, w_scales, m_tile_map, **kw):
        keepalive.extend([out, a, w, a_scales, w_scales, m_tile_map,
                          kw.get("bias"), kw.get("quant_scale")])
        return real_grouped(out, a, w, a_scales, w_scales, m_tile_map, **kw)

    # The launch goes through the kernel object's .launch(); wrap the module's
    # dispatch entry so we see the final grid/block and the raw arg tuple.
    from aiter.ops.flydsl.kernels import mxfp4_preshuffle_gfx1250_tdm as tdm_mod

    real_launch = tdm_mod.launch_gemm_a8w4_tdm

    def spy(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, stream, N, K,
            tile_m, tile_n, tile_k, m_warp, n_warp, out_is_f16, num_buffers,
            a_is_fp4, arg_m_tile_map, n_experts, stage1_act, has_bias, arg_bias,
            f32_swiglu_limit, stage1_quant_out=0, quant_wmma_rep=1,
            arg_quant_scale=None, **kw):
        act = {0: "noact", 1: "silu", 2: "swiglu"}.get(stage1_act, f"act{stage1_act}")
        name = (f"gemm_a8w4_tdm_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
                f"_b{num_buffers}_e{n_experts}"
                f"_a{'fp4' if a_is_fp4 else 'fp8'}"
                f"_out{'f16' if out_is_f16 else 'bf16'}"
                f"_{act}_bias{has_bias}"
                f"_qout{stage1_quant_out}_qrep{quant_wmma_rep}"
                f"_v{tdm_mod.TDM_DESCRIPTOR_VERSION}")
        block = m_warp * n_warp * 32
        m_tiles = (i32_m + tile_m - 1) // tile_m
        n_tiles = (N + tile_n - 1) // tile_n

        # gemm1 is the activated stage, gemm2 is noact.
        want_activated = which == "gemm1"
        if (stage1_act != 0) == want_activated and not records:
            out_tensor = _find_output_tensor(arg_c, keepalive)
            keepalive.append(out_tensor)

            # Establish known padding before the real production dispatch.
            # A device-wide sync also orders this torch fill before a FlyDSL
            # launch even if the caller supplied a different stream object.
            out_tensor.fill_(_POISON)
            torch.cuda.synchronize(out_tensor.device)

            record = Capture(
                kernel=name,
                grid=(m_tiles * n_tiles, 1, 1),
                block=(block, 1, 1),
                args={
                    "arg_c": _ptr_of(arg_c), "arg_a": _ptr_of(arg_a),
                    "arg_b": _ptr_of(arg_b),
                    "arg_scale_a": _ptr_of(arg_scale_a),
                    "arg_scale_b": _ptr_of(arg_scale_b),
                    "arg_m_tile_map": _ptr_of(arg_m_tile_map),
                    "arg_bias": _ptr_of(arg_bias),
                    "arg_quant_scale": _ptr_of(arg_quant_scale),
                    "i32_m": int(i32_m), "i32_n": int(N),
                    "f32_swiglu_limit": float(f32_swiglu_limit),
                },
                out_ptr=_ptr_of(arg_c),
                out_nbytes=0,
                lds_bytes=arena_bytes(tile_m, tile_n, tile_k, num_buffers,
                                      m_warp, n_warp, a_is_fp4),
                tiles={"tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
                       "num_buffers": num_buffers, "m_warp": m_warp,
                       "n_warp": n_warp},
            )
            record._out_tensor = out_tensor
            record.out_nbytes = out_tensor.numel() * out_tensor.element_size()
            records.append(record)
        else:
            record = None

        result = real_launch(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, stream, N, K,
            tile_m, tile_n, tile_k, m_warp, n_warp, out_is_f16, num_buffers,
            a_is_fp4, arg_m_tile_map, n_experts, stage1_act, has_bias, arg_bias,
            f32_swiglu_limit, stage1_quant_out, quant_wmma_rep, arg_quant_scale,
            **kw)
        if record is not None:
            # Capture the selected GEMM immediately. Waiting for run_moe to
            # return is too late because later fused stages may reuse or modify
            # this buffer.
            torch.cuda.synchronize(out_tensor.device)
            record.reference = out_tensor.detach().clone()
            torch.cuda.synchronize(out_tensor.device)
        return result

    tdm_mod.launch_gemm_a8w4_tdm = spy
    bgm.flydsl_grouped_gemm_a8w4_masked = grouped_spy
    # batched_gemm_mxfp4 imports launch_gemm_a8w4_tdm inside the function body,
    # so the module-level patch above is picked up on the next call.
    try:
        _run_production_moe(tokens, experts, topk, model_dim, inter_dim)
    finally:
        tdm_mod.launch_gemm_a8w4_tdm = real_launch
        bgm.flydsl_grouped_gemm_a8w4_masked = real_grouped

    if not records:
        raise IsaRunnerError(f"no {which} launch captured")

    cap = records[0]
    cap._keepalive = keepalive  # inputs must stay allocated for the replay
    t = getattr(cap, "_out_tensor", None)
    if t is None or not hasattr(t, "numel"):
        raise IsaRunnerError(
            f"captured {which} arg_c but could not retain its torch output tensor"
        )
    if cap.reference is None:
        raise IsaRunnerError(
            f"captured {which} launch but no immediate production output clone"
        )

    # Fail loudly if a captured pointer no longer belongs to a live tensor.
    live_ptrs = {int(x.data_ptr()) for x in keepalive
                 if x is not None and hasattr(x, "data_ptr")}
    missing = [n for n, v in cap.args.items()
               if n.startswith("arg_") and int(v) and int(v) not in live_ptrs]
    if missing:
        cap.args_not_pinned = missing
    return cap


def _run_production_moe(tokens, experts, topk, model_dim, inter_dim):
    """Drive one production MoE call with the standard bench shapes.

    Reuses the op_test's data prep (preshuffle, scales, routing) rather than
    reimplementing it -- that construction is the part most likely to be wrong.
    """
    import test_flydsl_grouped_gemm_gfx1250 as t

    t.set_data_format("a8w4")
    return t.run_moe(
        "a8w4",
        experts=experts, tokens=tokens, topk=topk, model_dim=model_dim,
        inter_dim=inter_dim, layout="gugu",
        activation=t.ActivationType.Silu,
        # eager path (bench=False) so each stage launches once, unwrapped by a
        # CUDA graph -- the spy has to see a real dispatch.
        bench=False, iters=1, warmup=0, raise_on_fail=False,
        check_aot_cache=False,
    )


def _launch_into(cap: Capture, isa_source, name_hint, spec, device, live):
    """Poison-fill the output, launch one ISA, return (module_kept, name)."""
    import torch
    res = build(isa_source)
    name = name_hint or (res.kernels[0] if len(res.kernels) == 1 else cap.kernel)
    if live is None:
        raise IsaRunnerError("no output tensor captured; rerun capture in-process")
    live.fill_(_POISON)
    torch.cuda.synchronize(live.device)
    mod = IsaModule(res.code_object, device=device, source=res.source)
    mod.function(name)
    mod.launch(name, cap.pack_kernargs(), spec)
    mod.synchronize()
    torch.cuda.synchronize(live.device)
    return mod, name, res


def _iqr_trimmed_median_us(latencies_us):
    """Match FlyDSL's benchmark_common IQR filter and upper median."""
    values = sorted(float(value) for value in latencies_us)
    if not values:
        raise IsaRunnerError("cannot summarize an empty benchmark")
    raw_count = len(values)
    if raw_count >= 8:
        q1, q3 = values[raw_count // 4], values[3 * raw_count // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [value for value in values if lo <= value <= hi]
        if filtered:
            values = filtered
    return values[len(values) // 2], raw_count, len(values)


def _benchmark_like_flydsl(
    mod,
    name,
    args,
    spec,
    live,
    *,
    iters,
    warmup,
    flush_l2,
):
    """Use the FlyDSL MoE test's per-iteration event/median methodology."""
    import torch

    if iters < 1:
        raise IsaRunnerError(f"benchmark iterations must be at least 1, got {iters}")
    if warmup < 0:
        raise IsaRunnerError(f"benchmark warmup must be non-negative, got {warmup}")

    stream = torch.cuda.current_stream(live.device)
    stream_ptr = int(stream.cuda_stream)
    if int(spec.stream) != stream_ptr:
        raise IsaRunnerError(
            f"benchmark stream mismatch: HIP=0x{int(spec.stream):x}, "
            f"torch=0x{stream_ptr:x}"
        )

    flush_buf = None
    flush_bytes = 0
    if flush_l2:
        props = torch.cuda.get_device_properties(live.device)
        l2_bytes = getattr(props, "L2_cache_size", 4 * 1024 * 1024)
        flush_bytes = max(int(l2_bytes) * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(
            flush_bytes, dtype=torch.uint8, device=live.device
        )

    def prepare():
        if flush_buf is not None:
            flush_buf.zero_()
        # Like the FlyDSL test's output zeroing, this is outside the timed
        # interval and gives every launch the same destination state.
        live.fill_(_POISON)

    for _ in range(warmup):
        prepare()
        mod.launch(name, args, spec)
    torch.cuda.synchronize(live.device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        prepare()
        starts[i].record(stream)
        mod.launch(name, args, spec)
        ends[i].record(stream)
    torch.cuda.synchronize(live.device)

    latencies = [
        starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters)
    ]
    median_us, raw_count, filtered_count = _iqr_trimmed_median_us(latencies)
    return {
        "kernel": name,
        "iters": iters,
        "warmup": warmup,
        "timer": "torch_cuda_events_per_iteration",
        "statistic": "iqr_trimmed_median",
        "flush_l2": bool(flush_l2),
        "l2_flush_bytes": flush_bytes,
        "sample_count": raw_count,
        "filtered_sample_count": filtered_count,
        "min_us": min(latencies),
        "max_us": max(latencies),
        "per_launch_us": median_us,
        "grid": list(spec.grid),
        "block": list(spec.block),
        "shared_mem_bytes": spec.shared_mem_bytes,
    }


def replay(cap: Capture, isa_source: str | Path, *, kernel: str | None = None,
           device: int = 0, lds_bytes: int | None = None,
           iters: int = 0, warmup: int = 20,
           flush_l2: bool = True) -> dict:
    """Launch *isa_source* with the captured arguments.

    The reference is the selected production output cloned immediately after
    its real launch. Both launches begin with the same poison fill. Positions
    left poison by both are padding; missing and unexpected writes fail.
    """
    import torch

    live = getattr(cap, "_out_tensor", None)
    if live is None:
        raise IsaRunnerError("no output tensor captured; rerun capture in-process")
    if cap.reference is None:
        raise IsaRunnerError(
            "no production output captured; rerun capture in-process"
        )

    torch_stream = torch.cuda.current_stream(live.device)
    spec = KernelLaunchSpec(
        grid=cap.grid, block=cap.block,
        shared_mem_bytes=(lds_bytes if lds_bytes is not None else cap.lds_bytes),
        stream=int(torch_stream.cuda_stream),
        device=device,
    )
    report: dict[str, Any] = {
        "grid": list(spec.grid), "block": list(spec.block),
        "shared_mem_bytes": spec.shared_mem_bytes,
        "capture": cap.to_json(),
        "reference": "production",
        "production_kernel": cap.kernel,
    }
    unpinned = getattr(cap, "args_not_pinned", None)
    if unpinned:
        # Reading recycled memory yields NaN rather than an error, so surface it.
        report["args_not_pinned"] = unpinned

    mod, name, res = _launch_into(cap, isa_source, kernel, spec, device, live)
    report["isa"], report["kernel"] = str(res.source), name
    try:
        got = live.detach().clone()
        torch.cuda.synchronize(live.device)
        report.update(_compare_outputs(cap.reference, got))

        if iters:
            report["benchmark"] = _benchmark_like_flydsl(
                mod,
                name,
                cap.pack_kernargs(),
                spec,
                live,
                iters=iters,
                warmup=warmup,
                flush_l2=flush_l2,
            )
    finally:
        mod.close()

    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["capture", "replay", "run"])
    p.add_argument("--isa", help="the .s to launch (replay/run)")
    p.add_argument("--kernel")
    p.add_argument("--which", default="gemm1", choices=["gemm1", "gemm2"])
    p.add_argument("--capture", help="capture JSON (metadata only; pointers are"
                                     " per-process and cannot be reused)")
    p.add_argument("--out", help="write the report JSON here")
    p.add_argument("--lds-bytes", type=int)
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--no-flush-l2",
        "--no_flush_l2",
        dest="no_flush_l2",
        action="store_true",
        help="disable the FlyDSL-style L2 flush before each timed launch",
    )
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--model-dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=768)
    args = p.parse_args(argv)

    cap = capture_launches(args.which, tokens=args.tokens, experts=args.experts,
                           topk=args.topk, model_dim=args.model_dim,
                           inter_dim=args.inter_dim)

    if args.command == "capture":
        report = cap.to_json()
    else:
        if not args.isa:
            print("--isa is required for replay/run", file=sys.stderr)
            return 1
        report = replay(cap, args.isa, kernel=args.kernel,
                        lds_bytes=args.lds_bytes, iters=args.iters,
                        warmup=args.warmup,
                        flush_l2=not args.no_flush_l2)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if report.get("passed", True) else 3


if __name__ == "__main__":
    sys.exit(main())
