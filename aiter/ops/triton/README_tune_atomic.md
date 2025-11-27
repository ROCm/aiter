# GEMM A16W16 Atomic Kernel Tuning

This document explains how to use the tuning script for the atomic GEMM kernel.

## Overview

The atomic GEMM kernel (`gemm_a16w16_atomic`) is a specialized kernel that uses atomic operations for split-K reduction. This allows for better parallelization in certain scenarios, especially for larger matrix dimensions.

## Tuning Script

The tuning script is located at `aiter/ops/triton/tune_a16w16_atomic.py`. It follows a similar pattern to the regular GEMM tuning script but with specific adaptations for the atomic kernel.

### Key Differences from Regular GEMM Tuning

1. **NUM_KSPLIT Parameter**: The atomic kernel includes a `NUM_KSPLIT` parameter that controls how the K dimension is split across multiple thread blocks for parallel reduction.

2. **Configuration Categories**: The atomic kernel uses different configuration categories based on the M dimension:
   - `small`: M < 32
   - `medium_M32`: M ≤ 128 with BLOCK_SIZE_M = 32
   - `medium_M64`: M ≤ 128 with BLOCK_SIZE_M = 64
   - `medium_M128`: M ≤ 128 with BLOCK_SIZE_M = 128
   - `large`: M ≤ 256
   - `xlarge`: M > 256

3. **No Bias Parameter**: The atomic kernel doesn't support a bias parameter.

### Running the Tuning Script

To run the tuning script:

```bash
cd aiter/ops/triton
python tune_a16w16_atomic.py
```

You can also specify specific parameters:

```bash
python tune_a16w16_atomic.py --batch-size 512 --input-type bfloat16 --out-dtype bfloat16
```

### Output

The tuning script will generate configuration files in the `aiter/ops/triton/configs/gemm/` directory with names like:
- `R9700-GEMM-A16W16-ATOMIC-N={N}-K={K}.json` for specific N,K dimensions
- `R9700-GEMM-A16W16-ATOMIC.json` - A default config file (without N,K parameters) that contains the most common optimal configurations across all tested shapes

The default config file is created by analyzing all the specific N,K configurations and selecting the most common optimal configuration for each category (small, medium_M32, etc.). This provides a good general-purpose configuration when a specific N,K configuration is not available.

### Configuration Parameters

The tuning script searches through these parameters:
- `BLOCK_SIZE_M`: [16, 32, 64, 128]
- `BLOCK_SIZE_N`: [32, 64, 128, 256]
- `BLOCK_SIZE_K`: [64, 128]
- `GROUP_SIZE_M`: [1, 8, 16]
- `NUM_KSPLIT`: [1, 2, 4, 8] (atomic kernel specific)
- `num_warps`: [4, 8]
- `num_stages`: [2]
- `waves_per_eu`: [3]

### Batch Sizes Tested

The script tests these batch sizes (M dimensions):
- 16 (small)
- 32 (medium_M32)
- 64 (medium_M64)
- 128 (medium_M128)
- 256 (large)
- 512 (large)
- 2048 (xlarge)
- 4096 (xlarge)

### Weight Shapes Tested

The script tunes for these weight shapes (N, K):
- (1024, 1024)
- (4096, 1024)
- (1024, 2048)
- (6144, 1024)
- (1024, 3072)

## Usage in Code

Once the tuning is complete, the atomic kernel can be used in your code:

```python
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic

# Basic usage
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

# The kernel will automatically use the tuned configurations
y = gemm_a16w16_atomic(x, w, dtype=torch.bfloat16)
```

## Notes

1. The atomic kernel is particularly useful for larger matrices where split-K parallelization provides benefits.
2. For smaller matrices, the regular GEMM kernel might be more efficient.
3. The tuning process can take considerable time as it tests many configurations.
4. Make sure you have sufficient GPU memory for the tuning process.