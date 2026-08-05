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

    # capture once, then replay a hand-edited kernel against it
    python tdm_adapter.py capture --out capture.json
    python tdm_adapter.py replay --capture capture.json --isa edited.s --check

The buffers live on the GPU only for the lifetime of the process, so capture
and replay must happen in the same run; ``run`` does both.

    python tdm_adapter.py run --isa edited.s --iters 100

Correctness is checked against the *captured output buffer*: the reference is
produced by the unmodified kernel in the same process, on the same inputs, so a
mismatch means the ISA edit changed behaviour and nothing else.
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


@dataclass
class Capture:
    """One recorded kernel invocation."""
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: dict[str, int | float]
    out_ptr: int
    out_nbytes: int
    reference: Any = field(default=None, repr=False)  # torch tensor clone

    def to_json(self) -> dict:
        return {
            "kernel": self.kernel,
            "grid": list(self.grid),
            "block": list(self.block),
            "args": {k: (v if isinstance(v, float) else int(v))
                     for k, v in self.args.items()},
            "out_ptr": self.out_ptr,
            "out_nbytes": self.out_nbytes,
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

    import flydsl.compiler as flyc

    records: list[Capture] = []
    original = flyc.CompiledFunction.__call__ if hasattr(
        flyc, "CompiledFunction") else None

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
        want_silu = which == "gemm1"
        if (stage1_act != 0) == want_silu and not records:
            records.append(Capture(
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
            ))
            records[-1]._out_tensor = arg_c  # keep alive; sized after the run
        return real_launch(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, stream, N, K,
            tile_m, tile_n, tile_k, m_warp, n_warp, out_is_f16, num_buffers,
            a_is_fp4, arg_m_tile_map, n_experts, stage1_act, has_bias, arg_bias,
            f32_swiglu_limit, stage1_quant_out, quant_wmma_rep, arg_quant_scale,
            **kw)

    tdm_mod.launch_gemm_a8w4_tdm = spy
    # batched_gemm_mxfp4 imports the symbol inside the function body, so the
    # module-level patch above is picked up on the next call.
    try:
        _run_reference_moe(tokens, experts, topk, model_dim, inter_dim)
    finally:
        tdm_mod.launch_gemm_a8w4_tdm = real_launch

    if not records:
        raise IsaRunnerError(f"no {which} launch captured")

    cap = records[0]
    t = getattr(cap, "_out_tensor", None)
    if t is not None and hasattr(t, "numel"):
        cap.out_nbytes = t.numel() * t.element_size()
        cap.reference = t.detach().clone()
    return cap


def _run_reference_moe(tokens, experts, topk, model_dim, inter_dim):
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


def replay(cap: Capture, isa_source: str | Path, *, kernel: str | None = None,
           device: int = 0, lds_bytes: int | None = None,
           iters: int = 0, warmup: int = 20, check: bool = True) -> dict:
    """Launch *isa_source* with the captured arguments."""
    import torch

    res = build(isa_source)
    name = kernel or (res.kernels[0] if len(res.kernels) == 1 else cap.kernel)

    out_t = cap.reference
    if check and out_t is None:
        raise IsaRunnerError("no reference captured; rerun capture in-process")

    # Zero the output so a kernel that fails to write is not mistaken for a pass.
    live = getattr(cap, "_out_tensor", None)
    if check and live is not None:
        live.zero_()
        torch.cuda.synchronize()

    spec = KernelLaunchSpec(
        grid=cap.grid, block=cap.block,
        shared_mem_bytes=(lds_bytes if lds_bytes is not None
                          else _infer_lds(isa_source)),
        device=device,
    )
    report: dict[str, Any] = {
        "isa": str(res.source), "kernel": name,
        "grid": list(spec.grid), "block": list(spec.block),
        "shared_mem_bytes": spec.shared_mem_bytes,
        "capture": cap.to_json(),
    }

    with IsaModule(res.code_object, device=device, source=res.source) as mod:
        mod.function(name)
        args = cap.pack_kernargs()
        mod.launch(name, args, spec)
        mod.synchronize()

        if check and live is not None:
            torch.cuda.synchronize()
            got, ref = live.float(), out_t.float()
            denom = ref.norm().item() or 1.0
            rel_l2 = ((got - ref).norm().item()) / denom
            report["rel_l2"] = rel_l2
            report["max_abs_diff"] = (got - ref).abs().max().item()
            report["passed"] = rel_l2 < 1e-6
            report["all_zero"] = bool(got.abs().max().item() == 0.0)

        if iters:
            report["benchmark"] = mod.benchmark(
                name, args, spec, iters=iters, warmup=warmup)

    return report


def _infer_lds(isa_source: str | Path) -> int:
    """Dynamic LDS the kernel needs, from the ARENA the frontend would allocate.

    The descriptor's group_segment_fixed_size is 0 for these kernels because the
    arena is allocated dynamically, so the launch must supply it. Falls back to
    the recorded gemm1 profile.
    """
    from isa_runner import TDM_GEMM1_LDS_BYTES
    env = os.environ.get("AITER_ISA_LDS_BYTES")
    return int(env) if env else TDM_GEMM1_LDS_BYTES


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
    p.add_argument("--check", action="store_true", default=True)
    p.add_argument("--no-check", dest="check", action="store_false")
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
                        warmup=args.warmup, check=args.check)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if report.get("passed", True) else 3


if __name__ == "__main__":
    sys.exit(main())
