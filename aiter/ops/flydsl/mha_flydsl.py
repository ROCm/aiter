"""FlyDSL MHA kernel wrapper for gfx1250 varlen flash attention."""
import os
import sys
import importlib.util

import torch


_KERNEL_DIR = os.environ.get(
    "FLYDSL_MHA_KERNEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "triton-to-flydsl", "flydsl-kernels"),
)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128
BLOCK_THREADS = 256
KV_BLOCK = 32
MIN_SEQLEN_K = 3 * KV_BLOCK

_launch_fn = None


def _ensure_kernel():
    global _launch_fn
    if _launch_fn is not None:
        return

    flydsl_root = os.environ.get("FLYDSL_ROOT")
    if flydsl_root is None:
        raise RuntimeError(
            "FLYDSL_ROOT not set. Point it to the FlyDSL repo root "
            "(e.g. /data/zanzhang/FlyDSL)."
        )
    build_py = os.path.join(flydsl_root, "build-fly", "python_packages")
    if os.path.isdir(build_py):
        if build_py not in sys.path:
            sys.path.insert(0, build_py)

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.jit_function import CompilationContext

    kernel_file = os.path.join(_KERNEL_DIR, "mha_fa_kernel_8wave.py")
    if not os.path.isfile(kernel_file):
        raise FileNotFoundError(f"FlyDSL MHA kernel not found at {kernel_file}")

    spec = importlib.util.spec_from_file_location("mha_kernel_8wave", kernel_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kernel = mod.attn_fwd_pipelined_kernel

    @flyc.jit
    def _launch(q, k, v, o, cu_q, cu_k,
                sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_t, so_h,
                max_sk, grid_b, grid_h, grid_m,
                stream: fx.Stream = fx.Stream(None)):
        _allocs = [
            getattr(mod, n) for n in dir(mod)
            if n.startswith("_lds_alloc") or n == "_lds_allocator" or n == "_lds_allocator_b"
        ]
        _seen = set()
        _allocs_unique = []
        for a in _allocs:
            if id(a) not in _seen:
                _seen.add(id(a))
                _allocs_unique.append(a)
        for a in _allocs_unique:
            a.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            for a in _allocs_unique:
                a.finalize()
        L = kernel(q, k, v, o, cu_q, cu_k,
                   sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_t, so_h, max_sk)
        mod.set_kernel_attrs()
        L.launch(grid=[grid_b, grid_h, grid_m], block=[BLOCK_THREADS, 1, 1])

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    _launch.compile_hints["maxnreg"] = 512
    _launch_fn = _launch


def _run_compiled(exe, args):
    """First call compiles and executes; subsequent calls use cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        import flydsl.compiler as flyc
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


def flash_attn_varlen_flydsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
):
    """FlyDSL MHA forward for gfx1250, varlen layout.

    Args:
        q: (total_q, nheads, headdim_qk) bf16
        k: (total_k, nheads_k, headdim_qk) bf16
        v: (total_k, nheads_k, headdim_v) bf16
        cu_seqlens_q: (batch+1,) i32
        cu_seqlens_k: (batch+1,) i32

    Returns:
        out: (total_q, nheads, headdim_v) f32
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == HEAD_DIM_QK, f"Expected headdim_qk={HEAD_DIM_QK}, got {q.shape[-1]}"
    assert v.shape[-1] == HEAD_DIM_V, f"Expected headdim_v={HEAD_DIM_V}, got {v.shape[-1]}"
    assert max_seqlen_k >= MIN_SEQLEN_K, (
        f"max_seqlen_k={max_seqlen_k} < minimum {MIN_SEQLEN_K} "
        f"(pipeline requires >= 3*KV_BLOCK={MIN_SEQLEN_K})"
    )

    batch = cu_seqlens_q.shape[0] - 1
    nheads = q.shape[1]

    if out is None:
        out = torch.zeros(
            (q.shape[0], nheads, HEAD_DIM_V), dtype=torch.float32, device=q.device
        )

    _ensure_kernel()

    grid_m = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    stream = torch.cuda.current_stream()

    _run_compiled(
        _launch_fn,
        (
            q, k, v, out, cu_seqlens_q, cu_seqlens_k,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            out.stride(0), out.stride(1),
            max_seqlen_k,
            batch, nheads, grid_m,
            stream,
        ),
    )

    return out
