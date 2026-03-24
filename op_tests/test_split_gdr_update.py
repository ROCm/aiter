import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.test_common import benchmark, perftest


def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """Convert [N, Hv, K, V] -> [N, Hv, K/4, V, 4]."""
    n, hv, k, v = state.shape
    assert k % 4 == 0, f"K ({k}) must be divisible by 4"
    return state.reshape(n, hv, k // 4, 4, v).permute(0, 1, 2, 4, 3).contiguous()


def from_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """Convert [N, Hv, K/4, V, 4] -> [N, Hv, K, V]."""
    n, hv, k4, v, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    return state.permute(0, 1, 2, 4, 3).reshape(n, hv, k4 * 4, v).contiguous()


def create_inputs(
    batch_size: int,
    seqlen: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    dtype: torch.dtype,
    extra_state_slots: int,
) -> dict[str, torch.Tensor | int]:
    """Create all inputs required by split_gdr update kernels."""
    device = "cuda"
    key_dim = num_heads_qk * head_dim
    value_dim = num_heads_v * head_dim
    dim = 2 * key_dim + value_dim
    mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
    a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
    b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
    ssm_state = torch.randn(
        batch_size + extra_state_slots,
        num_heads_v,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    return {
        "mixed_qkv": mixed_qkv,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "ssm_state": ssm_state,
        "ssm_state_indices": ssm_state_indices,
        "key_dim": key_dim,
        "value_dim": value_dim,
    }


def split_gdr_reference(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """CPU reference implementation for correctness check."""
    bsz, _, seqlen = mixed_qkv.shape
    h = num_heads_qk
    hv = num_heads_v
    kdim = head_dim
    vdim = head_dim
    group_size = hv // h
    if scale is None:
        scale = kdim**-0.5

    mixed_qkv_f = mixed_qkv.float().cpu()
    A_log_f = A_log.float().cpu()
    dt_bias_f = dt_bias.float().cpu()
    a_f = a.float().cpu().view(bsz, seqlen, hv)
    b_f = b.float().cpu().view(bsz, seqlen, hv)

    state_f = torch.zeros(bsz, hv, kdim, vdim, dtype=torch.float32)
    idx_cpu = initial_state_indices.cpu()
    for n in range(bsz):
        idx = idx_cpu[n].item()
        if idx >= 0:
            state_f[n] = initial_state_source[idx].float().cpu()

    q_all = mixed_qkv_f[:, :key_dim, :]
    k_all = mixed_qkv_f[:, key_dim : 2 * key_dim, :]
    v_all = mixed_qkv_f[:, 2 * key_dim : 2 * key_dim + value_dim, :]
    output = torch.zeros(bsz, seqlen, hv, vdim, dtype=torch.float32)

    for t in range(seqlen):
        for i_hv in range(hv):
            i_h = i_hv // group_size
            q_vec = q_all[:, i_h * kdim : (i_h + 1) * kdim, t]
            k_vec = k_all[:, i_h * kdim : (i_h + 1) * kdim, t]
            v_vec = v_all[:, i_hv * vdim : (i_hv + 1) * vdim, t]

            a_t = a_f[:, t, i_hv]
            b_t = b_f[:, t, i_hv]
            x = a_t + dt_bias_f[i_hv]
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x,
            )
            g = -torch.exp(A_log_f[i_hv]) * softplus_x
            beta = torch.sigmoid(b_t)

            if use_qk_l2norm_in_kernel:
                q_vec = q_vec / torch.sqrt(
                    torch.sum(q_vec * q_vec, dim=-1, keepdim=True) + 1e-6
                )
                k_vec = k_vec / torch.sqrt(
                    torch.sum(k_vec * k_vec, dim=-1, keepdim=True) + 1e-6
                )

            q_vec = q_vec * scale
            state_f[:, i_hv, :, :] *= torch.exp(g).unsqueeze(-1).unsqueeze(-1)
            v_vec = v_vec - torch.einsum("bkv,bk->bv", state_f[:, i_hv, :, :], k_vec)
            v_vec = v_vec * beta.unsqueeze(-1)
            state_f[:, i_hv, :, :] += torch.einsum("bk,bv->bkv", k_vec, v_vec)
            output[:, t, i_hv, :] = torch.einsum(
                "bkv,bk->bv", state_f[:, i_hv, :, :], q_vec
            )

    for n in range(bsz):
        idx = idx_cpu[n].item()
        if idx >= 0:
            initial_state_source[idx] = (
                state_f[n]
                .to(initial_state_source.dtype)
                .to(initial_state_source.device)
            )

    return output.to(mixed_qkv.dtype).to(mixed_qkv.device)


@perftest()
def run_fused_split_gdr_update_decode(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b_gate: torch.Tensor,
    initial_state_source_template: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """Run fused_split_gdr_update decode kernel for perf measurement."""
    # Clone per iteration to keep the same state-update semantics as the old event loop.
    st = initial_state_source_template.clone()
    return aiter.fused_split_gdr_update(
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_gate=b_gate,
        initial_state_source=st,
        initial_state_indices=initial_state_indices,
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@benchmark()
def test_split_gdr_update_decode(
    batch_size: int,
    seqlen: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    extra_state_slots: int = 10,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
) -> dict:
    """Check correctness and benchmark HIP split_gdr update decode kernel."""
    if not torch.cuda.is_available():
        return {
            "batch_size": batch_size,
            "seqlen": seqlen,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "all_close": False,
            "skip_reason": "CUDA/HIP is required",
        }

    seed_everything(42)
    inputs = create_inputs(
        batch_size=batch_size,
        seqlen=seqlen,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        dtype=dtype,
        extra_state_slots=extra_state_slots,
    )

    key_dim = int(inputs["key_dim"])
    value_dim = int(inputs["value_dim"])
    scale = head_dim**-0.5
    rtol, atol = (
        (1e-2, 5e-2) if dtype in (torch.bfloat16, torch.float16) else (3e-4, 1e-3)
    )

    ssm_state_ref = inputs["ssm_state"].clone()
    output_ref = split_gdr_reference(
        mixed_qkv=inputs["mixed_qkv"],
        A_log=inputs["A_log"],
        a=inputs["a"],
        dt_bias=inputs["dt_bias"],
        b=inputs["b"],
        initial_state_source=ssm_state_ref,
        initial_state_indices=inputs["ssm_state_indices"],
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    ssm_state_swizzled = to_swizzled_layout(inputs["ssm_state"].clone())
    output_hip = aiter.fused_split_gdr_update(
        mixed_qkv=inputs["mixed_qkv"],
        A_log=inputs["A_log"],
        a=inputs["a"],
        dt_bias=inputs["dt_bias"],
        b_gate=inputs["b"],
        initial_state_source=ssm_state_swizzled,
        initial_state_indices=inputs["ssm_state_indices"],
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

    output_diff = (output_ref - output_hip).abs().max().item()
    state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()
    all_close_out = torch.allclose(output_ref, output_hip, rtol=rtol, atol=atol)
    all_close_state = torch.allclose(
        ssm_state_ref, ssm_state_hip_final, rtol=rtol, atol=atol
    )
    all_close = all_close_out and all_close_state

    # Perf uses the common @perftest workflow, consistent with other op_tests.
    ssm_state_swizzled_template = to_swizzled_layout(inputs["ssm_state"].clone())
    _, hip_us = run_fused_split_gdr_update_decode(
        mixed_qkv=inputs["mixed_qkv"],
        A_log=inputs["A_log"],
        a=inputs["a"],
        dt_bias=inputs["dt_bias"],
        b_gate=inputs["b"],
        initial_state_source_template=ssm_state_swizzled_template,
        initial_state_indices=inputs["ssm_state_indices"],
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return {
        "batch_size": batch_size,
        "seqlen": seqlen,
        "num_heads_qk": num_heads_qk,
        "num_heads_v": num_heads_v,
        "head_dim": head_dim,
        "output_diff": output_diff,
        "state_diff": state_diff,
        "all_close": all_close,
        "hip_us": hip_us,
    }


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Benchmark HIP split_gdr_update decode kernel",
)
parser.add_argument(
    "-b",
    "--batch-size",
    "--batch",
    dest="batch_size",
    type=int,
    default=[64],
    nargs="+",
)
parser.add_argument("-s", "--seqlen", type=int, default=[1], nargs="+")
parser.add_argument(
    "--num-heads-qk",
    "--heads-qk",
    dest="num_heads_qk",
    type=int,
    default=[4],
    nargs="+",
)
parser.add_argument(
    "--num-heads-v", "--heads-v", dest="num_heads_v", type=int, default=[8], nargs="+"
)
parser.add_argument("--head-dim", type=int, default=[128], nargs="+")
parser.add_argument(
    "--dtype",
    "--itype",
    dest="dtype",
    type=str,
    default="bf16",
    choices=["bf16", "fp16", "fp32"],
    help="Input dtype",
)
parser.add_argument(
    "--extra-state-slots",
    type=int,
    default=10,
    help="Additional state rows beyond batch size",
)
parser.add_argument(
    "--use-qk-l2norm-in-kernel",
    nargs="*",
    default=["true"],
    choices=["true", "false"],
    help="Enable Q/K L2 norm inside kernel (default: true)",
)
args = parser.parse_args()

dtype = _DTYPE_MAP[args.dtype]


def _parse_bool_list(lst):
    return [v.lower() == "true" for v in (lst or ["true", "false"])]


l2norm_list = _parse_bool_list(args.use_qk_l2norm_in_kernel)

df = []
for (
    batch_size,
    seqlen,
    num_heads_qk,
    num_heads_v,
    head_dim,
    l2norm,
) in itertools.product(
    args.batch_size,
    args.seqlen,
    args.num_heads_qk,
    args.num_heads_v,
    args.head_dim,
    l2norm_list,
):
    ret = test_split_gdr_update_decode(
        batch_size=batch_size,
        seqlen=seqlen,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        dtype=dtype,
        extra_state_slots=args.extra_state_slots,
        use_qk_l2norm_in_kernel=l2norm,
    )
    df.append(ret)

df = pd.DataFrame(df)
dedup_cols = [
    "batch_size",
    "seqlen",
    "num_heads_qk",
    "num_heads_v",
    "head_dim",
    "dtype",
    "use_qk_l2norm_in_kernel",
]
try:
    df_md = df.to_markdown(index=False)
except ImportError:
    df_md = df.to_string(index=False)
aiter.logger.info("split_gdr_update summary:\n%s", df_md)
