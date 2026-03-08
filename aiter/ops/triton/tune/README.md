# Triton Kernel Tuning Guide

This guide explains how to tune Triton kernels for optimal performance on your GPU architecture.

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Configure GEMM shapes to tune:

Edit the tuning script to specify which dimensions you want to tune. For example, for `gemm_a8w8_blockscale` tuning, edit `aiter/ops/triton/tune/tune_a8w8_blockscale.py` and modify the `test_configs` list in the `main()` function (around line 410):

```python
test_configs = [
    # (batch_sizes, N, K)
    (
        [16, 32, 64, 128, 256, 512, 2048, 4096],  # M dimensions to test
        1024,  # N dimension
        1024,  # K dimension
    ),  # Standard attention head dims
    (
        [16, 32, 64, 128, 256, 512, 2048, 4096],
        4096,  # N dimension
        1024,  # K dimension
    ),  # FFN intermediate
    # Add more (batch_sizes, N, K) tuples as needed
]
```

Each configuration tuple specifies:
- `batch_sizes`: List of M (batch/sequence) dimensions to test
- `N`: Output feature dimension
- `K`: Input feature dimension

3. Start tuning:

Run the following command to start tuning. Please wait a few minutes as it will build kernels via JIT compilation:

`python3 aiter/ops/triton/tune/tune_a8w8_blockscale.py`

The tuning process will:
- Detect your GPU architecture automatically (e.g., `gfx942`, `gfx942`, `gfx1201`)
- Test each configuration in the search space against a PyTorch reference implementation
- Measure performance and select the best configuration for each category
- Save results to JSON files in `aiter/configs/triton/gemm/`

You will see output like:
```
Architecture: gfx1201
Search space size: 3456 configurations
Total test configurations: 5

Running A8W8_BLOCKSCALE-N=1024-K=1024...
Tuning A8W8_BLOCKSCALE-N=1024-K=1024: 100%|██████████| 8/8 [02:15<00:00, 16.92s/it]
Tuning for A8W8_BLOCKSCALE-N=1024-K=1024 took 135.42 seconds
```

Results are saved with the naming pattern: `{device_name}-GEMM-{tag}.json`

For example: `gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=1024.json`

The JSON files contain tuned configurations categorized by M dimension size:
- `small`: M < 32
- `medium_M32`, `medium_M64`, `medium_M128`: M <= 128 (specific power of 2)
- `large`: M <= 256
- `xlarge`: M > 256
- `any`: Catch-all for the largest tested size

Example output format:
```json
{
    "small": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
        "NUM_KSPLIT": 1,
        "waves_per_eu": 2,
        "kpack": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg"
    },
    "medium_M64": { ... },
    "large": { ... }
}
```

A default config file `{device_name}-GEMM-A8W8_BLOCKSCALE.json` is also created by selecting the most common configuration across all tested (N, K) shapes.

4. Build tuned kernels and test:

The tuned configurations are automatically picked up by the kernel when you run it. To test the performance:

Modify the test instance in `op_tests/test_gemm_a8w8_blockscale.py` (or the relevant test file) and run:

`python3 op_tests/test_gemm_a8w8_blockscale.py`

The kernel will automatically load the tuned configuration from `aiter/configs/triton/gemm/{device_name}-GEMM-A8W8_BLOCKSCALE.json` and apply the appropriate configuration based on the input dimensions.

If you have previously built Triton kernels and want to force a reload of the new tuned configurations, you may need to restart your Python process to clear the kernel cache.

## More Options

### Customizing Search Space

The search space defines which kernel configurations will be tested during tuning. To customize it for your specific kernel needs, edit the `get_configs_compute_bound()` function in the tuning script.

For `gemm_a8w8_blockscale`, the default search space explores:

```python
for num_stages in [1, 2, 3, 4]:
    for block_m in [32, 64, 128]:
        for block_n in [32, 64, 128]:
            for group_size_m in [1, 8, 16]:
                for num_warps in [2, 4, 8]:
                    for num_ksplit in [1, 2, 4]:
                        for waves_per_eu in [2, 4, 8]:
                            for kpack in [2]:
                                for cache_modifier in ["", ".cg"]:
                                    # Test this configuration
```

**Common Parameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `BLOCK_SIZE_M` | Tile size for M dimension | 32, 64, 128 |
| `BLOCK_SIZE_N` | Tile size for N dimension | 32, 64, 128 |
| `BLOCK_SIZE_K` | Tile size for K dimension | 128 (fixed for blockscale) |
| `GROUP_SIZE_M` | Group size for M dimension | 1, 8, 16 |
| `num_warps` | Number of warps per thread block | 2, 4, 8 |
| `num_stages` | Pipeline stages for software pipelining | 1, 2, 3, 4 |
| `NUM_KSPLIT` | K dimension splitting factor | 1, 2, 4 |
| `waves_per_eu` | Waves per execution unit | 2, 4, 8 |
| `kpack` | K packing factor | 2 |
| `cache_modifier` | Cache modifier hint | `""`, `".cg"` |
| `matrix_instr_nonkdim` | Matrix instruction dimension | 16 |

### Tuning Parameters

You can adjust tuning behavior by modifying the `tune_kernel()` call in the tuning function:

#### `num_iters`
- **Type**: Integer
- **Default**: `10`
- **Description**: Number of benchmark iterations for each configuration. Higher values give more stable timing results but take longer.

To modify, edit the `tune_gemm_a8w8_blockscale()` function and add:
```python
best_config = tune_kernel(
    search_space=search_space,
    make_run_and_gt_fn=make_run_and_gt,
    num_iters=20,  # Increase for more stable results
)
```

#### `atol` and `rtol`
- **Type**: Float
- **Default**: `atol=1e-2`, `rtol=1e-2`
- **Description**: Absolute and relative tolerances for output correctness validation. Configurations that produce results outside these tolerances compared to the PyTorch reference are rejected.

To modify:
```python
best_config = tune_kernel(
    search_space=search_space,
    make_run_and_gt_fn=make_run_and_gt,
    atol=1e-1,  # Relax absolute tolerance
    rtol=1e-1,  # Relax relative tolerance
)
```

### Output Data Type

The tuning script defaults to `torch.bfloat16` output. To change the output dtype, modify the `dtype` parameter in the `tune_and_save_configs()` call:

```python
best_configs = tune_and_save_configs(
    batch_sizes=batch_sizes,
    N=N,
    K=K,
    dtype=torch.float16,  # Use float16 instead of bfloat16
    search_space=search_space,
    save_path=save_path,
    device_name=dev,
    tag=tag,
)
```

## Notes

- **GPU Architecture Detection**: The tuning script automatically detects your GPU architecture using `arch_info.get_arch()`. Make sure you tune on the same GPU architecture where you plan to run the kernels.

- **Search Space Size**: The default search space for `gemm_a8w8_blockscale` contains 3456 configurations. With 8 batch sizes per (N, K) tuple and 5 different (N, K) configurations, full tuning can take several hours. Consider reducing the search space or number of test configurations for faster iteration during development.

- **Correctness Validation**: Each configuration is validated against a PyTorch reference implementation to ensure correctness. Configurations that fail validation are automatically skipped.

- **Resource Limits**: Some configurations may fail with `OutOfResources` errors on smaller GPUs. These are automatically skipped during tuning.

- **Triton Kernel Cache**: Triton caches compiled kernels. If you modify kernel source code or want to force recompilation with new tuned configs, restart your Python process to clear the cache.

- **Multiple GPU Support**: The current tuning script runs on a single GPU. For multi-GPU tuning, you would need to modify the script to use `torch.cuda.set_device()` and run separate tuning processes for each GPU.

- **Performance Metrics**: The tuning process optimizes for average latency (microseconds). Lower latency configurations are preferred. TFLOPS and bandwidth metrics are not currently computed but can be added to the benchmarking function if needed.

- **Extending to Other Kernels**: To tune other Triton kernels, create a new tuning script following the pattern in `tune_a8w8_blockscale.py`, adapting the input generation, reference implementation, and kernel invocation for your specific kernel.
