# Triton Kernel Tuning Guide

This guide explains how to write tuning scripts for Triton kernels using the utility functions provided in [`aiter/aiter/ops/triton/tune/utils.py`](aiter/aiter/ops/triton/tune/utils.py:1).

## Overview

The tuning utilities provide a framework for:
- Defining a search space of kernel configurations
- Benchmarking each configuration against a ground truth reference
- Selecting the best-performing configuration
- Saving the results to JSON files

## Core Utility Functions

### `tune_kernel()`

The main tuning function that iterates through configurations and finds the best performer.

```python
def tune_kernel(
    search_space: List[Dict[str, Any]],
    make_run_and_gt_fn: Callable[[Dict[str, Any]], Tuple[Callable[[], Any], Any]],
    config_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    num_iters: int = 10,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Dict[str, Any]
```

**Parameters:**
- `search_space`: List of configuration dictionaries to test
- `make_run_and_gt_fn`: Callable that takes a config and returns `(run_fn, ground_truth)`
- `config_callback`: Optional function to modify configs before use
- `num_iters`: Number of iterations for benchmarking (default: 10)
- `atol`: Absolute tolerance for output comparison (default: 1e-2)
- `rtol`: Relative tolerance for output comparison (default: 1e-2)

**Returns:**
- The best configuration dictionary (lowest average latency)

### `benchmark_config()`

Benchmarks a single kernel configuration.

```python
def benchmark_config(
    run: Callable[[], Any],
    ground_truth: Any,
    num_iters: int = 10,
    atol: float = 1e-1,
    rtol: float = 1e-1,
) -> float
```

**Parameters:**
- `run`: Callable that executes the kernel (no arguments)
- `ground_truth`: Expected output tensor for correctness verification
- `num_iters`: Number of benchmark iterations (default: 10)
- `atol`: Absolute tolerance for comparison (default: 1e-1)
- `rtol`: Relative tolerance for comparison (default: 1e-1)

**Returns:**
- Average latency in microseconds

**Behavior:**
- Performs 5 warmup iterations
- Records CUDA events for timing
- Validates output against ground truth after each iteration
- Handles `OutOfResources` and `AssertionError` exceptions gracefully

### `get_search_space()`

Generates a default search space for tuning.

```python
def get_search_space(small: bool = False) -> List[Dict[str, Any]]
```

**Parameters:**
- `small`: If `True`, returns a reduced search space for testing

**Returns:**
- List of configuration dictionaries with parameters:
  - `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`: Tile dimensions
  - `GROUP_SIZE_M`: Group size for M dimension
  - `num_warps`: Number of warps per block
  - `num_stages`: Pipeline stages
  - `waves_per_eu`: Waves per execution unit
  - `matrix_instr_nonkdim`: Matrix instruction dimension

### `save_configs_to_json()`

Saves configuration dictionaries to a JSON file.

```python
def save_configs_to_json(
    json_file_name: str,
    save_path: str,
    configs: Dict[str, Any],
) -> None
```

**Parameters:**
- `json_file_name`: Name of the JSON file
- `save_path`: Directory path to save the file
- `configs`: Dictionary of configurations to save

## Writing a Tuning Script

### Step 1: Define Input Generation

Create a helper function to generate test inputs for your kernel. This function should accept all necessary parameters including the configuration dictionary.

```python
def input_helper(
    *dimensions,
    *kernel_specific_params,
    config: dict,
):
    """
    Generate input tensors for the kernel.
    
    Args:
        *dimensions: Problem dimensions (e.g., M, N, K, etc.)
        *kernel_specific_params: Kernel-specific parameters (e.g., quantization flags)
        config: Configuration dictionary containing tuning parameters
    
    Returns:
        Tuple of all input tensors and config
    """
    # Generate input tensors on CUDA
    input_tensor = torch.randn(dims, dtype=dtype, device="cuda")
    
    # Generate output tensor(s)
    output_tensor = torch.empty(output_dims, dtype=dtype, device="cuda")
    
    # Generate any additional tensors needed by the kernel
    # This may include scales, routing information, etc.
    
    return (input_tensor, output_tensor, ..., config)
```

### Step 2: Create Reference Implementation

Implement a reference (ground truth) version using PyTorch or another trusted implementation.

```python
# Reference implementation can be:
# 1. Imported from existing test utilities
from op_tests.triton_tests.your_kernel import your_kernel_ref

# 2. Implemented inline using PyTorch operations
def run_reference(*inputs, **kwargs):
    # Compute the correct output using PyTorch
    return torch_output
```

### Step 3: Create the `make_run_and_gt_fn` Factory

Define a factory function that creates the `make_run_and_gt_fn` callable. This is the core pattern used in tuning scripts.

```python
def make_run_and_gt_fn_factory(
    *dimensions,
    *kernel_specific_params,
):
    """
    Factory function to create run and ground truth functions.
    
    Args:
        *dimensions: Problem dimensions
        *kernel_specific_params: Kernel-specific parameters
    
    Returns:
        Function that takes a config and returns (run_fn, ground_truth)
    """
    def make_run_and_gt(config):
        # Generate fresh inputs for each configuration
        inputs = input_helper(
            *dimensions,
            *kernel_specific_params,
            config=config,
        )
        
        def run():
            torch.cuda.empty_cache()
            # Call your Triton kernel here
            result = your_kernel(*inputs, config)
            return result
        
        # Generate ground truth using reference implementation
        ground_truth = your_kernel_ref(*inputs)
        
        return run, ground_truth
    
    return make_run_and_gt
```

### Step 4: Implement the Tuning Function

Create a function that orchestrates the tuning process for a specific set of parameters.

```python
def tune_your_kernel(
    *dimensions,
    *kernel_specific_params,
    search_space: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tune the kernel for specific dimensions.
    
    Args:
        *dimensions: Problem dimensions
        *kernel_specific_params: Kernel-specific parameters
        search_space: List of configuration dictionaries to test
    
    Returns:
        Best configuration dictionary
    """
    make_run_and_gt = make_run_and_gt_fn_factory(
        *dimensions,
        *kernel_specific_params,
    )
    
    best_config = tune_kernel(
        search_space=search_space,
        make_run_and_gt_fn=make_run_and_gt,
    )
    return best_config
```

### Step 5: Create Batch Tuning Function

For tuning multiple configurations and categorizing results by problem size ranges:

```python
def tune_and_save_configs(
    dimension_values: List[int],
    *other_dimensions,
    *kernel_specific_params,
    search_space: List[Dict[str, Any]],
    save_path: str,
    device_name: str,
):
    """
    Tune configurations for multiple dimension values and save results.
    
    Args:
        dimension_values: List of values for the primary dimension to test
        *other_dimensions: Fixed values for other dimensions
        *kernel_specific_params: Kernel-specific parameters
        search_space: List of configuration dictionaries
        save_path: Path to save configuration files
        device_name: Device/architecture name for file naming
    """
    start = time.time()
    
    benchmark_results = [
        tune_your_kernel(
            dim_value,
            *other_dimensions,
            *kernel_specific_params,
            search_space=search_space,
        )
        for dim_value in tqdm(dimension_values)
    ]
    
    # Categorize configs by dimension ranges
    best_configs = {
        (
            "small" if dim_value <= 256
            else "medium" if dim_value <= 2048
            else "large"
        ): config
        for dim_value, config in zip(dimension_values, benchmark_results)
    }
    
    json_file_name = f"{device_name}-KERNEL.json"
    save_configs_to_json(json_file_name, save_path, best_configs)
    
    end = time.time()
    print(f"Tuning took {end - start:.2f} seconds")
```

### Step 6: Define Custom Search Space (Optional)

If the default search space from `get_search_space()` doesn't meet your needs, create a custom one:

```python
def get_custom_search_space() -> List[Dict[str, Any]]:
    """
    Generate configuration space for your kernel.
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Define parameter ranges based on your kernel requirements
    for num_stages in [1, 2, 3, 4]:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for block_k in [64, 128]:
                    for num_warps in [2, 4, 8]:
                        configs.append({
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            # Add other kernel-specific parameters
                        })
    
    return configs
```

## Configuration Parameters

Common configuration parameters for Triton kernels:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `BLOCK_SIZE_M` | Tile size for M dimension | 16, 32, 64, 128, 256 |
| `BLOCK_SIZE_N` | Tile size for N dimension | 16, 32, 64, 128, 256 |
| `BLOCK_SIZE_K` | Tile size for K dimension | 32, 64, 128 |
| `GROUP_SIZE_M` | Group size for M dimension | 1, 8, 16, 32, 64 |
| `num_warps` | Number of warps per thread block | 2, 4, 8 |
| `num_stages` | Pipeline stages for software pipelining | 1, 2, 3, 4, 5 |
| `waves_per_eu` | Waves per execution unit | 1, 2, 4, 6, 8 |
| `matrix_instr_nonkdim` | Matrix instruction dimension | 16 |
