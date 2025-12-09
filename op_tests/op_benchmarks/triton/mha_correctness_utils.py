import torch

from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
    per_block_int8,
    _get_config,
)


def primary_output(result):
    """Return the main tensor output produced by a Triton kernel."""
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        return result[0]
    return result


def prepare_tensor_for_layout(tensor, layout):
    """Convert tensors between NHD and HND layouts as required by SageAttn."""
    if layout == "HND":
        return tensor.detach().clone().transpose(1, 2).contiguous()
    return tensor.detach().clone().contiguous()


def restore_tensor_layout(tensor, layout):
    """Bring SageAttn outputs back to the layout used by FlashAttention."""
    if layout == "HND":
        return tensor.detach().clone().transpose(1, 2).contiguous()
    return tensor.detach().clone().contiguous()


def print_output_comparison_stats(current, reference):
    """Print quick statistics comparing FP8 and SageAttn tensors."""
    current_f = current.float()
    reference_f = reference.float()
    abs_diff = torch.abs(reference_f - current_f)

    print("Output Tensor Stats:")
    print(
        f"  Reference ({tuple(reference_f.shape)}): min={reference_f.min().item():.6f}, max={reference_f.max().item():.6f}, "
        f"mean={reference_f.mean().item():.6f}, std={reference_f.std().item():.6f}"
    )
    print(
        f"  Test      ({tuple(current_f.shape)}): min={current_f.min().item():.6f}, max={current_f.max().item():.6f}, "
        f"mean={current_f.mean().item():.6f}, std={current_f.std().item():.6f}"
    )
    
    print("Correctness Comparison:")
    print(f"  Mean Absolute Error: {abs_diff.mean().item():.6e}")
    ref_flat = reference_f.reshape(-1)
    test_flat = current_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), test_flat.unsqueeze(0))
    print(f"  Cosine Similarity: {cos_sim.item():.8f}")


def run_sage_reference(q, k, v, args, d_head, d_head_v):
    """Execute the SageAttn reference kernel for correctness checking."""
    assert args.layout == "bshd", "Reference run only supports dense layout (bshd)."

    tensor_layout = args.qk_int8_layout
    config = _get_config()

    q_ref = prepare_tensor_for_layout(q, tensor_layout)
    k_ref = prepare_tensor_for_layout(k, tensor_layout)
    if tensor_layout == "HND":
        v_ref = v.detach().clone().transpose(1, 2).contiguous().to(torch.float16)
    else:
        v_ref = v.detach().clone().contiguous().to(torch.float16)

    sm_scale = d_head ** -0.5
    k_mean = None
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q_ref,
        k_ref,
        km=k_mean,
        BLKQ=config["BLOCK_SIZE_M"],
        BLKK=config["BLOCK_SIZE_N"],
        sm_scale=sm_scale,
        tensor_layout=tensor_layout,
    )
    q_scale = q_scale.to(torch.float32).unsqueeze(-1).contiguous()
    k_scale = k_scale.to(torch.float32).unsqueeze(-1).contiguous()

    return attn_qk_int8_per_block(
        q_int8,
        k_int8,
        v_ref,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=torch.float16,
        config=config,
    )


__all__ = [
    "primary_output",
    "prepare_tensor_for_layout",
    "restore_tensor_layout",
    "print_output_comparison_stats",
    "run_sage_reference",
]
