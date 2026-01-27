import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def visualize_distribution(data, title, ax, bins=100):
    """
    Visualize the distribution of a tensor.
    
    Args:
        data: torch.Tensor - the data to visualize
        title: str - title for the subplot
        ax: matplotlib axis - the axis to plot on
        bins: int - number of bins for histogram
    """
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
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define all variables to plot
    variables = [
        (q, 'Q (Original) Distribution', 0, 0),
        (k, 'K (Original) Distribution', 0, 1),
        (q_fp4, 'Q FP4 Distribution', 1, 0),
        (k_fp4, 'K FP4 Distribution', 1, 1),
        (q_scale, 'Q Scale Distribution', 2, 0),
        (k_scale, 'K Scale Distribution', 2, 1),
        (q_scale_pre, 'Q Pre-Scale Distribution', 3, 0),
        (k_scale_pre, 'K Pre-Scale Distribution', 3, 1),
    ]
    
    # Plot each variable
    for data, title, row, col in variables:
        ax = fig.add_subplot(gs[row, col])
        visualize_distribution(data, title, ax)
    
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
