import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

# The kernel uses 64-lane __shfl_xor reductions. These are the wave64 targets
# supported in-tree; gfx1250 is wave32 and cannot execute those reductions.
SUPPORTED_GFX = ("gfx942", "gfx950")
_MAX_PERF_ROTATIONS = 32
_PERF_ROTATION_BUDGET = 256 * 1024 * 1024


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


def _perf_rotation_count(state: torch.Tensor, output: torch.Tensor) -> int:
    bytes_per_call = max(1, state.nbytes + output.nbytes)
    return max(
        1,
        min(_MAX_PERF_ROTATIONS, _PERF_ROTATION_BUDGET // bytes_per_call),
    )


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


def run_fused_split_gdr_update_decode(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b_gate: torch.Tensor,
    initial_state_source: torch.Tensor,
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
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the kernel with its real swizzled, in-place state and output layout."""
    return aiter.fused_split_gdr_update(
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_gate=b_gate,
        initial_state_source=initial_state_source,
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
        output=output,
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

    # Keep an untouched swizzled state and output template. Perf buffers may be
    # mutated repeatedly, while correctness always starts from pristine clones.
    state_pristine = to_swizzled_layout(inputs["ssm_state"].clone())
    output_pristine = torch.zeros(
        (batch_size, seqlen, num_heads_v, head_dim),
        device=inputs["mixed_qkv"].device,
        dtype=dtype,
    )

    def hip(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return run_fused_split_gdr_update_decode(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=state,
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
            output=output,
        )

    candidates = {"hip": hip}

    # Dominant tensor arithmetic per token/head: state decay (K*V), state-k
    # projection (2*K*V), outer-product update (2*K*V), output projection
    # (2*K*V), q scaling (K), and v correction/gating (2*V). Q/K L2 norm adds
    # about 6*K FLOPs. Scalar exp/log/sqrt/sigmoid operations are not assigned
    # an arbitrary FLOP equivalent.
    flops_per_token_head = 7 * head_dim * head_dim + head_dim + 2 * head_dim
    if use_qk_l2norm_in_kernel:
        flops_per_token_head += 6 * head_dim
    flops = batch_size * seqlen * num_heads_v * flops_per_token_head

    # Logical op traffic: unique activation/gate inputs, output, index/A/dt
    # vectors, plus one read and one write of each indexed fp32 state element.
    output_bytes = output_pristine.numel() * output_pristine.element_size()
    state_bytes = (
        2
        * batch_size
        * num_heads_v
        * head_dim
        * head_dim
        * inputs["ssm_state"].element_size()
    )
    nbytes = (
        inputs["mixed_qkv"].numel() * inputs["mixed_qkv"].element_size()
        + inputs["a"].numel() * inputs["a"].element_size()
        + inputs["b"].numel() * inputs["b"].element_size()
        + inputs["dt_bias"].numel() * inputs["dt_bias"].element_size()
        + inputs["A_log"].numel() * inputs["A_log"].element_size()
        + inputs["ssm_state_indices"].numel()
        * inputs["ssm_state_indices"].element_size()
        + state_bytes
        + output_bytes
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        # Timing uses dedicated buffers because the state is intentionally
        # updated in-place; correctness below never observes these mutations.
        perf_state = state_pristine.clone()
        perf_output = output_pristine.clone()
        _, us = run_perftest(
            fn,
            perf_state,
            perf_output,
            num_rotate_args=_perf_rotation_count(perf_state, perf_output),
        )

        check_state = state_pristine.clone()
        check_output = output_pristine.clone()
        output = fn(check_state, check_output)
        state_final = from_swizzled_layout(check_state)
        output_err = checkAllclose(
            output_ref.to(dtypes.fp32),
            output.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: output",
        )
        state_err = checkAllclose(
            ssm_state_ref.to(dtypes.fp32),
            state_final.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: final state",
        )

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(output_err, state_err)
    return ret


test_split_gdr_update_decode.__test__ = False


def _str2bool(value: str) -> bool:
    value = value.lower()
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected 'true' or 'false'")
    return value == "true"


def main():
    if not torch.cuda.is_available():
        aiter.logger.warning("split_gdr_update requires ROCm; skipping")
        return
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "split_gdr_update unsupported on %s; supported targets: %s",
            get_gfx(),
            ", ".join(SUPPORTED_GFX),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark HIP split_gdr_update decode kernel",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        "--itype",
        dest="dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16],
        nargs="*",
        default=[dtypes.bf16],
        help="Input dtype list (the HIP kernel supports bf16).",
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
        "--num-heads-v",
        "--heads-v",
        dest="num_heads_v",
        type=int,
        default=[8],
        nargs="+",
    )
    parser.add_argument("--head-dim", type=int, default=[128], nargs="+")
    parser.add_argument(
        "--extra-state-slots",
        type=int,
        default=[10],
        nargs="+",
        help="Additional state rows beyond batch size.",
    )
    parser.add_argument(
        "--use-qk-l2norm-in-kernel",
        type=_str2bool,
        nargs="*",
        default=[True],
        choices=[True, False],
        help="Q/K L2-normalization modes to sweep: true false.",
    )
    args = parser.parse_args()

    rows = []
    for (
        dtype,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        extra_state_slots,
        l2norm,
    ) in itertools.product(
        args.dtype,
        args.batch_size,
        args.seqlen,
        args.num_heads_qk,
        args.num_heads_v,
        args.head_dim,
        args.extra_state_slots,
        args.use_qk_l2norm_in_kernel,
    ):
        rows.append(
            test_split_gdr_update_decode(
                batch_size=batch_size,
                seqlen=seqlen,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                dtype=dtype,
                extra_state_slots=extra_state_slots,
                use_qk_l2norm_in_kernel=l2norm,
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "split_gdr_update summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
