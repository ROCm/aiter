# Triton GEMM A8W8 Blockscale Tune

This script tunes the Triton-based GEMM A8W8 blockscale kernel for optimal performance across different matrix dimensions and batch sizes.

## Overview

The `tune_a8w8_blockscale.py` script performs automatic kernel tuning by:
- Exploring a comprehensive search space of kernel configurations
- Benchmarking each configuration for performance and correctness
- Selecting the best configuration for each batch size category (small, medium, large)
- Saving results to JSON configuration files

## Usage

### 1. Install aiter:
```bash
cd $aiter_path
python3 setup.py develop
```

### 2. Run tuning:
Execute the tuning script directly. The script will automatically test various matrix dimensions and batch sizes:

```bash
python3 aiter/ops/triton/tune/tune_a8w8_blockscale.py
```

The script will:
- Detect your GPU architecture
- Generate a search space of kernel configurations
- Test configurations for multiple (M, N, K) dimension combinations:
  - `(256, 2048, 4096) × (1024, 1024)` - Standard attention head dims
  - `(256, 2048, 4096) × (4096, 1024)` - FFN intermediate
  - `(256, 2048, 4096) × (1024, 2048)` - Wider input
  - `(256, 2048, 4096) × (6144, 1024)` - Larger FFN
  - `(256, 2048, 4096) × (1024, 3072)` - Deeper input
- Save best configurations to JSON files

### 3. Check results:
Results are saved to the `aiter/aiter/ops/triton/configs/gemm/` directory with filenames in the format:
```
{device_name}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json
```


**Batch size categories:**
- `small_M`: M ≤ 256
- `medium_M`: 256 < M ≤ 2048
- `large_M`: M > 2048

## Customizing Tuning

### Modifying Test Configurations

Edit the `test_configs` list in the [`main()`](tune_a8w8_blockscale.py:287) function to add or modify dimension combinations:

```python
test_configs = [
    # (batch_sizes, N, K)
    ([256, 2048, 4096], 1024, 1024),  # Your custom config
    ([128, 512, 1024], 2048, 512),    # Another custom config
]
```

### Modifying Search Space

The search space is defined in [`get_configs_compute_bound()`](tune_a8w8_blockscale.py:189). You can adjust the parameter ranges:

```python
def get_configs_compute_bound() -> list:
    configs = []
    block_k = 128  # Fixed for (128, 128) quantization blocks

    for num_stages in [1, 2, 3, 4]:           # Pipeline stages
        for block_m in [32, 64, 128]:        # Block size for M dimension
        for block_n in [32, 64, 128]:        # Block size for N dimension
        for group_size_m in [1, 8, 16]:      # Group size for M dimension
        for num_warps in [2, 4, 8]:          # Number of warps per block
        for num_ksplit in [1, 2, 4]:         # K dimension split factor
        for waves_per_eu in [2, 4, 8]:      # Waves per execution unit
        for kpack in [2]:                    # K packing factor
        for cache_modifier in ["", ".cg"]:   # Cache modifier
            configs.append({...})
    return configs
```

## Kernel Parameters

### Block Sizes
- **BLOCK_SIZE_M**: Number of M elements processed per thread block
- **BLOCK_SIZE_N**: Number of N elements processed per thread block
- **BLOCK_SIZE_K**: Number of K elements processed per thread block (fixed at 128 for blockscale)

### Grouping
- **GROUP_SIZE_M**: Group size for M dimension reduction
- **NUM_KSPLIT**: Split factor for K dimension parallelization

### Execution Configuration
- **num_warps**: Number of warps per thread block (typically 2, 4, or 8)
- **num_stages**: Number of pipeline stages for software pipelining (1-4)
- **waves_per_eu**: Number of wavefronts per execution unit (2, 4, or 8)

### Optimization
- **kpack**: K packing factor for vectorized loads (fixed at 2)
- **matrix_instr_nonkdim**: Matrix instruction dimension (fixed at 16)
- **cache_modifier**: Cache modifier for loads ("" or ".cg" for cache global)

## Notes

### Correctness Validation
Each configuration is validated against a PyTorch reference implementation ([`run_torch_reference()`](tune_a8w8_blockscale.py:21)) to ensure numerical correctness. Configurations that fail validation are skipped.


### Data Types
- Input tensors: FP8 (E4M3 format)
- Output tensors: bfloat16 (configurable)
- Scale tensors: float32

### Quantization Blocks
The kernel uses fixed (128, 128) quantization blocks for both N and K dimensions, which constrains `BLOCK_SIZE_K` to 128.

## Example: Tuning for Specific Dimensions

To tune for a specific use case, modify the script:

```python
def main():
    dev = arch_info.get_arch()
    torch.cuda.init()

    # Custom test configuration
    test_configs = [
        ([512, 1024], 2048, 768),  # Your specific dimensions
    ]

    search_space = get_configs_compute_bound()
    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    for batch_sizes, N, K in test_configs:
        tag = f"A8W8_BLOCKSCALE-N={N}-K={K}"
        tune_and_save_configs(
            batch_sizes=batch_sizes,
            N=N,
            K=K,
            dtype=torch.bfloat16,
            search_space=search_space,
            save_path=save_path,
            device_name=dev,
            tag=tag,
        )
```

## Related Files

- [`gemm_a8w8_blockscale.py`](../gemm/basic/gemm_a8w8_blockscale.py): The kernel implementation being tuned
- [`utils.py`](utils.py): Tuning utilities and benchmarking functions
- Configuration files: `aiter/aiter/ops/triton/configs/gemm/*.json`
