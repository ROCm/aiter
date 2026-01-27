import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def unpack_fp4_to_fp32(uint8_tensor):
    """
    Unpack uint8 tensor containing packed e2m1 fp4 values into fp32.
    Each uint8 contains two 4-bit e2m1 values (2-bit exponent, 1-bit mantissa).
    
    Args:
        uint8_tensor: torch.Tensor with dtype uint8, shape [..., D]
    
    Returns:
        torch.Tensor with dtype float32, shape [..., 2*D]
    """
    # Move to CPU for processing
    uint8_np = uint8_tensor.detach().cpu().numpy()
    original_shape = uint8_np.shape
    
    # Flatten to process
    uint8_flat = uint8_np.flatten()
    
    # Extract two 4-bit values from each uint8
    # Lower 4 bits
    low_nibble = uint8_flat & 0x0F
    # Upper 4 bits
    high_nibble = (uint8_flat >> 4) & 0x0F
    
    # Interleave them to maintain order
    fp4_values = np.empty(len(uint8_flat) * 2, dtype=np.uint8)
    fp4_values[0::2] = low_nibble
    fp4_values[1::2] = high_nibble
    
    # Convert e2m1 fp4 to fp32
    # e2m1 format: [sign:1][exp:2][mantissa:1]
    sign = ((fp4_values >> 3) & 0x1).astype(np.float32)
    exp = ((fp4_values >> 1) & 0x3).astype(np.int32)
    mantissa = (fp4_values & 0x1).astype(np.float32)
    
    # Convert to float
    # For e2m1: value = (-1)^sign * 2^(exp-1) * (1 + mantissa * 0.5)
    # The mantissa bit represents 0.5, so mantissa=0 → 1.0, mantissa=1 → 1.5
    # Special cases: exp=0 means subnormal or zero
    fp32_values = np.zeros_like(sign, dtype=np.float32)
    
    # Normal numbers (exp != 0)
    normal_mask = exp != 0
    fp32_values[normal_mask] = (1 - 2 * sign[normal_mask]) * np.power(2.0, exp[normal_mask] - 1) * (1 + mantissa[normal_mask] * 0.5)
    
    # Subnormal numbers (exp == 0, mantissa != 0)
    subnormal_mask = (exp == 0) & (mantissa != 0)
    fp32_values[subnormal_mask] = (1 - 2 * sign[subnormal_mask]) * np.power(2.0, -1) * (mantissa[subnormal_mask] * 0.5)
    
    # Reshape to [..., 2*D]
    new_shape = list(original_shape)
    new_shape[-1] = new_shape[-1] * 2
    fp32_values = fp32_values.reshape(new_shape)
    
    return torch.from_numpy(fp32_values)

def visualize_distribution(data, title, ax, bins=100, is_fp4_packed=False):
    """
    Visualize the distribution of a tensor.
    
    Args:
        data: torch.Tensor - the data to visualize
        title: str - title for the subplot
        ax: matplotlib axis - the axis to plot on
        bins: int - number of bins for histogram
        is_fp4_packed: bool - whether data is packed fp4 in uint8 format
    """
    # Unpack fp4 if needed
    if is_fp4_packed:
        data = unpack_fp4_to_fp32(data)
    
    # Convert to numpy and flatten
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().float().numpy().flatten()
    else:
        data_np = np.array(data).flatten()
    
    # Plot histogram
    ax.hist(data_np, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {data_np.mean():.4f}\nStd: {data_np.std():.4f}\nMin: {data_np.min():.4f}\nMax: {data_np.max():.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)


def visualize_sage_quant_distributions(q, k, q_fp4, q_scale, q_scale_pre, 
                                       k_fp4, k_scale, k_scale_pre, 
                                       output_path='sage_quant_distributions.png',
                                       figsize=(20, 12)):
    """
    Visualize distributions of original and quantized Q/K tensors.
    
    Args:
        q: Original Q tensor
        k: Original K tensor
        q_fp4: Quantized Q tensor (FP4)
        q_scale: Q scaling factors
        q_scale_pre: Q pre-scaling factors
        k_fp4: Quantized K tensor (FP4)
        k_scale: K scaling factors
        k_scale_pre: K pre-scaling factors
        output_path: Path to save the concatenated figure
        figsize: Figure size (width, height)
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)
    
    # Define all variables to plot (with flag for fp4 packed data)
    variables = [
        (q, 'Q (Original) Distribution', 0, 0, False),
        (k, 'K (Original) Distribution', 0, 1, False),
        (q_fp4, 'Q FP4 Distribution', 1, 0, True),
        (k_fp4, 'K FP4 Distribution', 1, 1, True),
        (q_scale, 'Q Scale Distribution', 2, 0, False),
        (k_scale, 'K Scale Distribution', 2, 1, False),
        (q_scale_pre, 'Q Pre-Scale Distribution', 3, 0, False),
        (k_scale_pre, 'K Pre-Scale Distribution', 3, 1, False),
    ]
    
    # Plot each variable
    for data, title, row, col, is_fp4 in variables:
        ax = fig.add_subplot(gs[row, col])
        visualize_distribution(data, title, ax, is_fp4_packed=is_fp4)
    
    # Add overall title
    fig.suptitle('SAGE Quantization Output Distributions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    return fig


# Example usage:
if __name__ == "__main__":
    # Assuming you have the outputs from sage_quant_v2:
    # q_fp4, q_scale, q_scale_pre, k_fp4, k_scale, k_scale_pre, v_fp8, v_scale = sage_quant_v2(...)
    
    # Example with dummy data (replace with your actual outputs)
    from aiter.ops.triton._triton_kernels.sage_attn_triton_amd.fwd_prefill_v2 import sage_quant_v2
    
    # Create sample tensors
    batch_size = 2
    seq_len_q = 512
    seq_len_kv = 512
    num_heads = 32
    head_dim = 128
    
    q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    
    # Run quantization
    import aiter
    q_fp4, q_scale, q_scale_pre, k_fp4, k_scale, k_scale_pre, v_fp8, v_scale = sage_quant_v2(
        q, k, v,
        FP8_TYPE=aiter.dtypes.fp8,
        FP8_MAX=448.0,
        layout='bshd'
    )
    
    # Visualize distributions
    visualize_sage_quant_distributions(
        q, k,
        q_fp4, q_scale, q_scale_pre,
        k_fp4, k_scale, k_scale_pre,
        output_path='sage_quant_distributions.png'
    )
