"""FlyDSL MHA kernel wrapper for gfx1250 flash attention."""
import os
import sys
import importlib.util

import torch


_KERNEL_DIR = os.environ.get(
    "FLYDSL_MHA_KERNEL_DIR",
    os.path.join(os.path.dirname(__file__), "kernels", "mha_1250"),
)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128
BLOCK_THREADS = 128  # WAVE_SIZE(32) * NUM_WAVES(4)
KV_TILE_N = 128
BPP = 2  # bytes per element (bf16)

_launch_fns = {}   # {is_causal: launch_fn}
_kernel_mod = None


def _ensure_kernel_mod():
    global _kernel_mod
    if _kernel_mod is not None:
        return
    flydsl_root = os.environ.get("FLYDSL_ROOT")
    if flydsl_root is None:
        raise RuntimeError("FLYDSL_ROOT not set.")
    build_py = os.path.join(flydsl_root, "build-fly", "python_packages")
    if os.path.isdir(build_py) and build_py not in sys.path:
        sys.path.insert(0, build_py)
    if _KERNEL_DIR not in sys.path:
        sys.path.insert(0, _KERNEL_DIR)
    kernel_file = os.path.join(_KERNEL_DIR, "fmha_kernel_gfx1250.py")
    spec = importlib.util.spec_from_file_location("fmha_kernel_gfx1250", kernel_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _kernel_mod = mod


def _patch_reusable_slot_specs():
    """Compat shim: older FlyDSL builds lack _reusable_slot_spec on Float32/Float64,
    which flyc.compile() requires for the AOT fast-dispatch path."""
    import ctypes
    from flydsl.expr.numeric import Float32, Float64

    if not hasattr(Float32, "_reusable_slot_spec"):
        @classmethod
        def _f32_slot_spec(cls, arg):
            return ctypes.c_float, lambda a: a.value if hasattr(a, "value") else a
        Float32._reusable_slot_spec = _f32_slot_spec
        Float32._reusable_ctype = ctypes.c_float

    if not hasattr(Float64, "_reusable_slot_spec"):
        @classmethod
        def _f64_slot_spec(cls, arg):
            return ctypes.c_double, lambda a: a.value if hasattr(a, "value") else a
        Float64._reusable_slot_spec = _f64_slot_spec
        Float64._reusable_ctype = ctypes.c_double


def _ensure_kernel(is_causal: bool):
    if is_causal in _launch_fns:
        return

    _ensure_kernel_mod()
    mod = _kernel_mod

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext

    _patch_reusable_slot_specs()

    kernel = mod.compile_fmha_fwd(is_causal=is_causal)
    _lds_alloc_k_a = mod._lds_alloc_k_a
    _lds_alloc_k_b = mod._lds_alloc_k_b
    _lds_alloc_v_a = mod._lds_alloc_v_a
    _lds_alloc_v_b = mod._lds_alloc_v_b
    _BLOCK_SIZE = mod.BLOCK_SIZE

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!enter")

    @flyc.jit
    def _launch(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_cu_seqlens_q: fx.Tensor,
        ptr_cu_seqlens_k: fx.Tensor,
        scalar_f: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch_size: fx.Int32,
    ):
        _lds_alloc_k_a.finalized = False
        _lds_alloc_k_b.finalized = False
        _lds_alloc_v_a.finalized = False
        _lds_alloc_v_b.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _lds_alloc_k_a.finalize()
            _lds_alloc_k_b.finalize()
            _lds_alloc_v_a.finalize()
            _lds_alloc_v_b.finalize()

        from flydsl.expr.arith import _to_raw

        num_tg = arith.index_cast(T.index, arith.ceildivui(
            _to_raw(max_seqlen_q), arith.constant(BLOCK_M, type=T.i32)))
        grid_x = arith.index_cast(T.index, batch_size)
        grid_z = arith.index_cast(T.index, num_heads)

        launcher = kernel(
            ptr_O, ptr_Q, ptr_K, ptr_V, ptr_LSE,
            ptr_cu_seqlens_q, ptr_cu_seqlens_k,
            scalar_f,
            stride_q_seq, stride_k_seq, stride_v_seq, stride_o_seq,
            gqa, max_seqlen_q, max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, num_tg, grid_z),
            block=(_BLOCK_SIZE, 1, 1),
        )

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    _launch_fns[is_causal] = _launch


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

    THD varlen layout — kernel operates directly on packed tensors.

    Args:
        q: (total_q, nheads, headdim_qk) bf16
        k: (total_k, nheads_k, headdim_qk) bf16
        v: (total_k, nheads_k, headdim_v) bf16
        cu_seqlens_q: (batch+1,) i32
        cu_seqlens_k: (batch+1,) i32

    Returns:
        out: (total_q, nheads, headdim_v) bf16
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == HEAD_DIM_QK, f"Expected headdim_qk={HEAD_DIM_QK}, got {q.shape[-1]}"
    assert v.shape[-1] == HEAD_DIM_V, f"Expected headdim_v={HEAD_DIM_V}, got {v.shape[-1]}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK ** 0.5)

    if out is None:
        out = torch.zeros(
            (total_q_tokens, nheads_q, HEAD_DIM_V),
            dtype=torch.bfloat16, device=q.device,
        )
    lse = torch.zeros(
        (batch, nheads_q, max_seqlen_q), dtype=torch.float32, device=q.device
    )

    # THD strides: per-token stride = nheads * dim * BPP bytes
    stride_q_seq = nheads_q * HEAD_DIM_QK * BPP
    stride_k_seq = nheads_k * HEAD_DIM_QK * BPP
    stride_v_seq = nheads_k * HEAD_DIM_V * BPP
    stride_o_seq = nheads_q * HEAD_DIM_V   # elements

    _ensure_kernel(bool(causal))

    _run_compiled(
        _launch_fns[bool(causal)],
        (
            out, q, k, v, lse,
            cu_seqlens_q, cu_seqlens_k,
            softmax_scale,
            stride_q_seq, stride_k_seq, stride_v_seq, stride_o_seq,
            gqa, max_seqlen_q, max_seqlen_k,
            nheads_q, batch,
        ),
    )

    return out
