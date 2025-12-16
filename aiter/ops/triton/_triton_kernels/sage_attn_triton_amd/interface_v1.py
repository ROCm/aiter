import os
import warnings
import torch
from typing import Optional, Union, Tuple
from .fwd_prefill import fav3_sage_triton_impl

from .utils import (
    DEBUG,
    USE_EXP2,
    BWD_MODE,
    PHILOX_SEED,
    PHILOX_OFFSET,
)


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    qv: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    seqlens_rotary: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    FP8_MAX: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    softcap: float,
    rotary_interleaved: bool,
    scheduler_metadata=None,
    num_splits: int = 1,
    pack_gqa=None,
    sm_margin: int = 0,
    return_lse: bool = True,
    layout: str = "bshd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sage Attention v1 forward pass compatible interface for AMD Triton implementation.

    This function maps v1 parameters to the existing AMD Triton implementation.
    """

    if DEBUG:
        print()
        print("interface_sage_v1.py::fwd inputs")
        print("q:", q.dtype if q is not None else None, q.shape)
        print("k:", k.dtype if k is not None else None, k.shape)
        print("v:", v.dtype if v is not None else None, v.shape)
        print(
            "k_new:",
            k_new.dtype if k_new is not None else None,
            k_new.shape if k_new is not None else None,
        )
        print(
            "v_new:",
            v_new.dtype if v_new is not None else None,
            v_new.shape if v_new is not None else None,
        )
        print(
            "qv:",
            qv.dtype if qv is not None else None,
            qv.shape if qv is not None else None,
        )
        print(
            "out:",
            out.dtype if out is not None else None,
            out.shape if out is not None else None,
        )
        print(
            "cu_seqlens_q:",
            cu_seqlens_q,
            cu_seqlens_q.shape if cu_seqlens_q is not None else None,
        )
        print(
            "cu_seqlens_k:",
            cu_seqlens_k,
            cu_seqlens_k.shape if cu_seqlens_k is not None else None,
        )
        print(
            "cu_seqlens_k_new:",
            cu_seqlens_k_new,
            cu_seqlens_k_new.shape if cu_seqlens_k_new is not None else None,
        )
        print(
            "seqused_q:", seqused_q, seqused_q.shape if seqused_q is not None else None
        )
        print(
            "seqused_k:", seqused_k, seqused_k.shape if seqused_k is not None else None
        )
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print(
            "page_table:",
            page_table,
            page_table.shape if page_table is not None else None,
        )
        print(
            "kv_batch_idx:",
            kv_batch_idx,
            kv_batch_idx.shape if kv_batch_idx is not None else None,
        )
        print(
            "leftpad_k:", leftpad_k, leftpad_k.shape if leftpad_k is not None else None
        )
        print(
            "rotary_cos:",
            rotary_cos,
            rotary_cos.shape if rotary_cos is not None else None,
        )
        print(
            "rotary_sin:",
            rotary_sin,
            rotary_sin.shape if rotary_sin is not None else None,
        )
        print(
            "seqlens_rotary:",
            seqlens_rotary,
            seqlens_rotary.shape if seqlens_rotary is not None else None,
        )
        print(
            "q_descale:",
            q_descale.dtype if q_descale is not None else None,
            q_descale.shape if q_descale is not None else None,
        )
        print(
            "k_descale:",
            k_descale.dtype if k_descale is not None else None,
            k_descale.shape if k_descale is not None else None,
        )
        print(
            "v_descale:",
            v_descale.dtype if v_descale is not None else None,
            v_descale.shape if v_descale is not None else None,
        )
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("attention_chunk:", attention_chunk)
        print("softcap:", softcap)
        print("rotary_interleaved:", rotary_interleaved)
        print("scheduler_metadata:", scheduler_metadata)
        print("num_splits:", num_splits)
        print("pack_gqa:", pack_gqa)
        print("sm_margin:", sm_margin)
        print("return_lse:", return_lse)
        print("layout:", layout)

    # Handle qv packed input
    if qv is not None:
        raise NotImplementedError(
            "QV packed input is not yet supported in the AMD Triton backend"
        )

    # Handle softcap
    if softcap != 0.0:
        raise NotImplementedError(
            f"Softcap is not yet supported in the AMD Triton backend (got softcap={softcap}, expected 0.0)"
        )

    # Handle attention_chunk
    if attention_chunk != 0 and attention_chunk != 1:
        raise NotImplementedError(
            f"attention_chunk is not yet supported in the AMD Triton backend (got attention_chunk={attention_chunk})"
        )

    # Handle scheduler metadata
    if scheduler_metadata is not None:
        raise NotImplementedError(
            "Scheduler metadata is not yet supported in the AMD Triton backend"
        )

    # Handle pack_gqa
    if pack_gqa is not None and pack_gqa is not False:
        raise NotImplementedError(
            f"pack_gqa is not yet supported in the AMD Triton backend (got pack_gqa={pack_gqa})"
        )

    # Handle num_splits
    if num_splits != 1:
        raise NotImplementedError(
            f"Split attention (num_splits > 1) is not yet supported in the AMD Triton backend (got num_splits={num_splits})"
        )

    # Handle sm_margin
    if sm_margin != 0:
        raise NotImplementedError(
            f"sm_margin is not yet supported in the AMD Triton backend (got sm_margin={sm_margin}, expected 0)"
        )

    # Handle leftpad_k
    if leftpad_k is not None:
        raise NotImplementedError(
            "Left padding (leftpad_k) is not yet supported in the AMD Triton backend"
        )

    # Handle cu_seqlens_k_new
    if cu_seqlens_k_new is not None:
        raise NotImplementedError(
            "cu_seqlens_k_new is not yet supported in the AMD Triton backend"
        )

    # establish layout / varlen & max seq lens
    if cu_seqlens_q is not None:
        if len(q.shape) != 3:
            raise ValueError(
                f"cu_seqlens_q provided but q has shape {q.shape}, expected 3D tensor for varlen"
            )
        layout = "thd"
        cu_seqlens_q_local = cu_seqlens_q
        max_seqlens_q_local = max_seqlen_q
        if cu_seqlens_k is not None:
            cu_seqlens_k_local = cu_seqlens_k
            max_seqlens_k_local = max_seqlen_k
        else:
            cu_seqlens_k_local = None
            max_seqlens_k_local = k.shape[1] if len(k.shape) == 4 else max_seqlen_k
    else:
        #layout is "bshd" or "bhsd"
        seq_dim = 1 if layout == "bshd" else 2
        cu_seqlens_q_local = None
        cu_seqlens_k_local = None
        max_seqlens_q_local = q.shape[seq_dim] if max_seqlen_q is None else max_seqlen_q
        max_seqlens_k_local = k.shape[seq_dim] if max_seqlen_k is None else max_seqlen_k

    if out is None:
        # NOTE: Using types that are lower precision than float32 such as bfloat16 for fp8 causes mismatches on a small set of tests.
        out_dtype = torch.float16
        if layout in ["bshd", "bhsd"]:
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                v.shape[-1],
                dtype=out_dtype,
                device=q.device,
            )
        elif layout == "thd":
            out = torch.zeros(
                q.shape[0], q.shape[1], v.shape[-1], dtype=out_dtype, device=q.device
            )
        else:
            raise ValueError(
                f"Unsupported layout: {layout}. Only 'bshd', 'bhsd' and 'thd' layouts are supported."
            )
    else:
        out = out.zero_()

    # Handle causal mask
    causal_flag = bool(causal)

    # Handle alibi slopes
    alibi_slopes = None

    # Handle dropout
    dropout_p = 0.0
    return_softmax = False
    philox_seed = PHILOX_SEED
    philox_offset = PHILOX_OFFSET

    # Call implementation
    if DEBUG:
        print("Using Prefill Triton implementation")

    # Create softmax_lse tensor - FA3 always uses exact shapes
    if layout == "thd":
        # varlen: (Hq, Total_Q)
        total_q, nheads_q, _ = q.shape
        softmax_lse = torch.zeros(
            (nheads_q, total_q), device=q.device, dtype=torch.float32
        ) if return_lse else None
    else:
        # bshd: (B, Hq, Sq)
        if layout == "bshd":
            batch, seqlen_q, nheads_q, _ = q.shape
        else:
            batch, nheads_q, seqlen_q, _ = q.shape
        softmax_lse = torch.zeros(
            (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
        ) if return_lse else None

    # sd_mask is not returned in v3 interface
    sd_mask = None

    fav3_sage_triton_impl(
        q,
        k,
        v,
        out,
        softmax_lse,
        sd_mask,
        softmax_scale,
        alibi_slopes,
        causal_flag,
        window_size_left,
        window_size_right,
        None,
        layout,
        cu_seqlens_q_local,
        cu_seqlens_k_local,
        max_seqlens_q_local,
        max_seqlens_k_local,
        dropout_p,
        philox_seed,
        philox_offset,
        return_softmax,
        USE_EXP2,
        q_descale,
        k_descale,
        v_descale,
        FP8_MAX,
        seqused_q,
        seqused_k,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=rotary_interleaved,
        seqlens_rotary=seqlens_rotary,
    )

    if DEBUG:
        print("interface_sage_v1.py::fwd outputs")
        print(
            "out:",
            out.dtype if out is not None else None,
            out.shape if out is not None else None,
        )
        print(
            "softmax_lse:",
            softmax_lse.dtype if softmax_lse is not None else None,
            softmax_lse.shape if softmax_lse is not None else None,
        )

    # --- Assertions (FA3 always expects exact shapes) ---
    # out: same shape as q except last dim is v's head_dim
    if layout == "thd":
        # varlen: (Total_Q, Hq, Dv)
        assert (
            out.shape[0] == q.shape[0]
        ), f"[fwd_v3] out.shape[0] {out.shape[0]} != q.shape[0] {q.shape[0]}"
        assert (
            out.shape[1] == q.shape[1]
        ), f"[fwd_v3] out.shape[1] {out.shape[1]} != q.shape[1] {q.shape[1]}"
        assert (
            out.shape[2] == v.shape[-1]
        ), f"[fwd_v3] out.shape[2] {out.shape[2]} != v.shape[-1] {v.shape[-1]}"
    else:
        # bshd: (B, Sq, Hq, Dv)
        assert (
            out.shape[0] == q.shape[0]
        ), f"[fwd_v3] out.shape[0] {out.shape[0]} != q.shape[0] {q.shape[0]}"
        assert (
            out.shape[1] == q.shape[1]
        ), f"[fwd_v3] out.shape[1] {out.shape[1]} != q.shape[1] {q.shape[1]}"
        assert (
            out.shape[2] == q.shape[2]
        ), f"[fwd_v3] out.shape[2] {out.shape[2]} != q.shape[2] {q.shape[2]}"
        assert (
            out.shape[3] == v.shape[-1]
        ), f"[fwd_v3] out.shape[3] {out.shape[3]} != v.shape[-1] {v.shape[-1]}"

    # softmax_lse dtype
    if softmax_lse is not None:
        assert (
            softmax_lse.dtype == torch.float32
        ), f"[fwd_v3] softmax_lse dtype {softmax_lse.dtype} != torch.float32"
        # softmax_lse shape depends on layout
        if layout == "thd":
            # varlen: (Hq, Total_Q)
            expected_lse_shape = (q.shape[1], q.shape[0])
        elif layout == "bshd":
            # bshd: (B, Sq, Hq)
            expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
        else:
            # bhsd: (B, Hq, Sq)
            expected_lse_shape = (q.shape[0], q.shape[1], q.shape[2])
        assert (
            softmax_lse.shape == expected_lse_shape
        ), f"[fwd_v3] softmax_lse shape {softmax_lse.shape} != {expected_lse_shape}"

    # Return format compatible with v3
    # V3 returns (out, softmax_lse, *rest) where rest can be empty or contain additional outputs
    return out, softmax_lse
