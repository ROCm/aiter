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

    kernel_dir = _KERNEL_DIR
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext

    kernel_file = os.path.join(kernel_dir, "fmha_kernel_gfx1250.py")
    if not os.path.isfile(kernel_file):
        raise FileNotFoundError(f"FlyDSL MHA kernel not found at {kernel_file}")

    spec = importlib.util.spec_from_file_location(
        "fmha_kernel_gfx1250", kernel_file
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kernel = mod.fmha_fwd_kernel
    _lds_alloc_k_a = mod._lds_alloc_k_a
    _lds_alloc_k_b = mod._lds_alloc_k_b
    _lds_alloc_v_a = mod._lds_alloc_v_a
    _lds_alloc_v_b = mod._lds_alloc_v_b
    _BLOCK_SIZE = mod.BLOCK_SIZE

    @flyc.jit
    def _launch(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        scalar_f: fx.Float32,
        kv_seq_len: fx.Int32,
        stride_q_seq: fx.Int32,
        stride_q_tg: fx.Int32,
        stride_q_head: fx.Int32,
        stride_q_batch: fx.Int32,
        gqa: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_k_head: fx.Int32,
        stride_k_batch: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_v_head: fx.Int32,
        stride_v_batch: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_o_head: fx.Int32,
        stride_o_batch: fx.Int32,
        q_seq_len: fx.Int32,
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

        q_seq_raw = _to_raw(q_seq_len)
        num_tg = arith.index_cast(
            T.index,
            arith.divui(q_seq_raw, arith.constant(BLOCK_M, type=T.i32)),
        )
        grid_x = arith.index_cast(T.index, batch_size)
        grid_z = arith.index_cast(T.index, num_heads)

        launcher = kernel(
            ptr_O, ptr_Q, ptr_K, ptr_V, ptr_LSE,
            scalar_f, kv_seq_len,
            stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch,
            gqa,
            stride_k_seq, stride_k_head, stride_k_batch,
            stride_v_seq, stride_v_head, stride_v_batch,
            stride_o_seq, stride_o_head, stride_o_batch,
            q_seq_len,
        )
        launcher.launch(
            grid=(grid_x, num_tg, grid_z),
            block=(_BLOCK_SIZE, 1, 1),
        )

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

    Internally pads varlen tensors to BHSD for the FMHA kernel,
    then unpads the result back to varlen.

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

    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK ** 0.5)

    S_q = max_seqlen_q
    S_kv = max_seqlen_k

    # Pad varlen (total_tokens, nheads, hdim) → BHSD (B, H, S, D)
    q_bhsd = torch.zeros(
        (batch, nheads_q, S_q, HEAD_DIM_QK), dtype=torch.bfloat16, device=q.device
    )
    k_bhsd = torch.zeros(
        (batch, nheads_k, S_kv, HEAD_DIM_QK), dtype=torch.bfloat16, device=k.device
    )
    v_bhsd = torch.zeros(
        (batch, nheads_k, S_kv, HEAD_DIM_V), dtype=torch.bfloat16, device=v.device
    )

    cu_q = cu_seqlens_q.cpu()
    cu_k = cu_seqlens_k.cpu()
    for b in range(batch):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        # varlen: (total, H, D) → BHSD: (B, H, S, D)
        q_bhsd[b, :, :sq, :] = q[cu_q[b]:cu_q[b + 1]].transpose(0, 1)
        k_bhsd[b, :, :sk, :] = k[cu_k[b]:cu_k[b + 1]].transpose(0, 1)
        v_bhsd[b, :, :sk, :] = v[cu_k[b]:cu_k[b + 1]].transpose(0, 1)

    o_bhsd = torch.zeros(
        (batch, nheads_q, S_q, HEAD_DIM_V), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.zeros(
        (batch, nheads_q, S_q), dtype=torch.float32, device=q.device
    )

    # Byte strides for Q/K/V, element strides for O
    stride_q_seq = HEAD_DIM_QK * BPP
    stride_q_tg = BLOCK_M * HEAD_DIM_QK * BPP
    stride_q_head = S_q * HEAD_DIM_QK * BPP
    stride_q_batch = nheads_q * S_q * HEAD_DIM_QK * BPP

    stride_k_seq = HEAD_DIM_QK * BPP
    stride_k_head = S_kv * HEAD_DIM_QK * BPP
    stride_k_batch = nheads_k * S_kv * HEAD_DIM_QK * BPP

    stride_v_seq = HEAD_DIM_V * BPP
    stride_v_head = S_kv * HEAD_DIM_V * BPP
    stride_v_batch = nheads_k * S_kv * HEAD_DIM_V * BPP

    stride_o_seq = HEAD_DIM_V
    stride_o_head = S_q * HEAD_DIM_V
    stride_o_batch = nheads_q * S_q * HEAD_DIM_V

    _ensure_kernel()

    _run_compiled(
        _launch_fn,
        (
            o_bhsd, q_bhsd, k_bhsd, v_bhsd, lse,
            softmax_scale, S_kv,
            stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch,
            gqa,
            stride_k_seq, stride_k_head, stride_k_batch,
            stride_v_seq, stride_v_head, stride_v_batch,
            stride_o_seq, stride_o_head, stride_o_batch,
            S_q, nheads_q, batch,
        ),
    )

    # Unpad BHSD → varlen
    total_q = q.shape[0]
    if out is None:
        out = torch.zeros(
            (total_q, nheads_q, HEAD_DIM_V), dtype=torch.bfloat16, device=q.device
        )
    for b in range(batch):
        sq = cu_q[b + 1] - cu_q[b]
        out[cu_q[b]:cu_q[b + 1]] = o_bhsd[b, :, :sq, :].transpose(0, 1)

    return out
