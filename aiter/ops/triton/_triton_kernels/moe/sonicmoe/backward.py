import torch
import triton
import triton.language as tl

from .activation_kernels import activation_bwd, activation_fwd
from .enums import LIBRARY_NAME
from .grouped_gemm_triton import grouped_gemm
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton


def _get_powers_of_2(start: int, end: int) -> list[int]:
    output = []
    n = start
    while n <= end:
        output.append(n)
        n = n << 1
    return output


def _get_autotune_configs_for_db2_and_ds() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in _get_powers_of_2(4, 32):
        configs.append(triton.Config({"BLOCK_TK": BLOCK_TK}, num_warps=8, num_stages=4))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_for_db2_and_ds(),
    key=["H", "E"],
)
@triton.jit
def db2_and_ds_kernel(
    dout_ptr,
    s_ptr,
    new_ds_partial_ptr,
    old_ds_partial_ptr,
    b2_ptr,
    db2_ptr,
    x_gather_idx_ptr,
    s_scatter_idx_ptr,
    expert_offset_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    OLD_DS_PARTIAL_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_OLD_DS_PARTIAL_N: tl.constexpr,
):
    Eidx = tl.program_id(0)
    Hidx = tl.program_id(1)
    NUM_H_BLOCKS: tl.constexpr = tl.num_programs(1)

    h_offsets = Hidx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    b2 = tl.load(b2_ptr + Eidx * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    db2_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for block_start in tl.range(0, n_tokens, BLOCK_TK):
        tk_offsets = block_start + tl.arange(0, BLOCK_TK)
        tk_mask = tk_offsets < n_tokens
        tk_grouped = E_count_start + tk_offsets

        token_indices = tl.load(
            x_gather_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        scatter_indices = tl.load(
            s_scatter_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        s = tl.load(s_ptr + scatter_indices, mask=tk_mask, other=0.0).to(tl.float32)

        dout_offsets = token_indices[:, None] * H + h_offsets[None, :]
        dout_mask = tk_mask[:, None] & h_mask[None, :]
        dout = tl.load(dout_ptr + dout_offsets, mask=dout_mask, other=0.0).to(
            tl.float32
        )

        db2_acc += tl.sum(dout * s[:, None], axis=0)

        ds_partial = tl.sum(dout * b2[None, :], axis=1)

        if Hidx == 0:
            n_offsets = tl.arange(0, BLOCK_OLD_DS_PARTIAL_N)
            old_ds_partial_offsets = (
                scatter_indices[:, None] * OLD_DS_PARTIAL_N + n_offsets[None, :]
            )
            old_ds_partial_mask = tk_mask[:, None] & (
                n_offsets[None, :] < OLD_DS_PARTIAL_N
            )
            old_ds_partial_vals = tl.load(
                old_ds_partial_ptr + old_ds_partial_offsets,
                mask=old_ds_partial_mask,
                other=0.0,
            ).to(tl.float32)
            ds_partial += tl.sum(old_ds_partial_vals, axis=1)

        tl.store(
            new_ds_partial_ptr + scatter_indices * NUM_H_BLOCKS + Hidx,
            ds_partial,
            mask=tk_mask,
        )

    tl.store(db2_ptr + Eidx * H + h_offsets, db2_acc, mask=h_mask)


def _get_autotune_configs_for_db1() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in _get_powers_of_2(4, 128):
        for BLOCK_I in _get_powers_of_2(64, 4096):
            if 4096 <= BLOCK_I * BLOCK_TK <= 16384:
                configs.append(
                    triton.Config(
                        {"BLOCK_I": BLOCK_I, "BLOCK_TK": BLOCK_TK},
                        num_warps=8,
                        num_stages=4,
                    )
                )
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        if c.kwargs["BLOCK_I"] <= triton.next_power_of_2(nargs["I"]):
            pruned_configs.append(c)
    return pruned_configs


@triton.autotune(
    configs=_get_autotune_configs_for_db1(),
    key=["I", "E"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def db1_kernel(
    dh_ptr,
    db1_ptr,
    expert_offset_ptr,
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr = False,
):
    Eidx = tl.program_id(0)

    E_count_start = tl.load(expert_offset_ptr + Eidx).to(tl.int64)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1).to(tl.int64)
    n_tokens = E_count_end - E_count_start

    NUM_I_BLOCKS: tl.constexpr = triton.cdiv(I, BLOCK_I)
    I_HALF: tl.constexpr = I // 2
    for Iidx in tl.static_range(0, NUM_I_BLOCKS, 1):
        i_offsets = Iidx * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < I

        db1_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        for block_start in tl.range(0, n_tokens, BLOCK_TK):
            tk_offsets = block_start + tl.arange(0, BLOCK_TK)
            tk_mask = tk_offsets < n_tokens
            tk_grouped = E_count_start + tk_offsets

            dz_offsets = tk_grouped[:, None] * I + i_offsets[None, :]
            dz_mask = tk_mask[:, None] & i_mask[None, :]
            dz = tl.load(dh_ptr + dz_offsets, mask=dz_mask, other=0.0).to(tl.float32)
            db1_acc += tl.sum(dz, axis=0)

        if CONCAT_LAYOUT:
            out_offsets = i_offsets // 2 + (i_offsets % 2) * I_HALF
        else:
            out_offsets = i_offsets
        db1_offsets = Eidx.to(tl.int64) * I + out_offsets
        tl.store(db1_ptr + db1_offsets, db1_acc, mask=i_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_backward_act_rocm",
    mutates_args={"dx_expanded", "db1"},
)
def _up_projection_backward_act(
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dh: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
) -> None:
    I_full, _H, E = w1.size()
    I = I_full // 2 if is_glu_activation else I_full

    grouped_gemm(
        dh,
        w1.permute(2, 0, 1),
        expert_frequency_offset,
        out=dx_expanded,
    )

    if db1 is not None:
        db1_kernel[(E,)](
            dh,
            db1,
            expert_frequency_offset,
            (2 * I if is_glu_activation else I),
            E,
            CONCAT_LAYOUT=concat_layout and is_glu_activation,
        )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_backward_act_rocm",
    mutates_args={"dh", "ds", "db2", "a_prime"},
)
def _down_projection_backward_act(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
) -> None:
    H, I, E = w2.size()
    TK = x_gather_idx.size(0)
    s = topk_scores[s_scatter_idx]

    # 1. Gather dout rows, scale by router scores
    dout_gathered = dout[x_gather_idx]  # (TK, H)
    dy = dout_gathered * s.unsqueeze(-1)  # (TK, H)

    # 2. Grouped GEMM: dh_raw = dy @ w2^T per expert -> (TK, I)
    # w2 is (H, I, E), we need (E, H, I) for B
    # But we need dy @ w2^T -> dy @ (I, H)^T doesn't work directly.
    # Actually w2 is (H, I, E), permuted to (E, H, I) for grouped gemm.
    # We want y = a @ w2_e where w2_e is (H, I, E) -> per expert (H, I)
    # So backward: da = dy @ w2_e^T = dy @ (I, H) where dy is (TK, H), result is (TK, I)
    # Grouped gemm: A=(TK, H), B=(E, H, I), C=(TK, I) — this is correct.
    # w2.permute(2, 0, 1) = (E, H, I) — K=H, N=I
    dh_raw = torch.empty(TK, I, dtype=dh.dtype, device=dh.device)
    grouped_gemm(dy, w2.permute(2, 0, 1), expert_frequency_offset, out=dh_raw)

    # 3. ds: dot(dout_gathered, y) where y = a @ w2 per expert
    # ds_scattered = sum_h(dout_gathered * (a @ w2_e)) but we already have dy = dout_gathered * s
    # Actually ds = sum_h(dout_gathered[i] * y[i]) for each token
    # We compute it as: ds = (dy / s) dot y, but simpler: recompute from dout_gathered and y.
    # From the original: ds = colvec_reduce of (dout_gathered^T @ a_e @ w2_e) but that's complex.
    # Simpler: compute a_prime = activation(h), then y = a_prime @ w2_e per expert
    # ds[i] = sum(dout_gathered[i] * y[i])
    a_prime_val = activation_fwd(h, I, activation_type)
    a_prime.copy_(a_prime_val)

    # y_recomputed = a_prime @ w2 per expert
    y_recomputed = torch.empty(TK, H, dtype=dy.dtype, device=dy.device)
    grouped_gemm(
        a_prime_val, w2.permute(2, 1, 0), expert_frequency_offset, out=y_recomputed
    )
    # w2 is (H, I, E), permute(2,1,0) = (E, I, H) — so A=(TK, I), B=(E, I, H) -> C=(TK, H)

    ds_scattered = (dout_gathered * y_recomputed).sum(dim=-1)

    # 4. dactivation: dh = dh_raw * d_activation(h)
    # For swiglu: dh includes both gate and up gradients
    dh_act = activation_bwd(h, dh_raw, I, activation_type)
    dh.copy_(dh_act)

    if db2 is None:
        ds[s_scatter_idx] = ds_scattered
    else:
        old_ds_partial = torch.empty(
            TK, 1, device=ds_scattered.device, dtype=ds_scattered.dtype
        )
        old_ds_partial[s_scatter_idx, 0] = ds_scattered

        BLOCK_H = min(triton.next_power_of_2(H), 2048)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)
        new_ds_partial = torch.empty(
            TK, NUM_H_BLOCKS, dtype=torch.float32, device=ds.device
        )

        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            old_ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            1,
            BLOCK_H=BLOCK_H,
            BLOCK_OLD_DS_PARTIAL_N=1,
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_token_broadcast_backward_rocm", mutates_args={"dx_reduced"}
)
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor | None,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    if num_activated_expert_per_token_offset is None:
        assert not is_varlen_K
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )
