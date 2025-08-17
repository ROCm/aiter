# Module Test Suite

This directory contains **module_test** implementations for various fusion kernels.

## Current Implementation: Fused MoE 2-Stage

Test suite for Fused MoE 2-Stage with **real-world model configurations**.

**Note**: Fused MoE is currently the only implemented test case. The `TestConfig` format can be extended for other fusion kernels.

## Files
- `test_fused_moe.py` - Main test runner
- `README.md` - This documentation

## Key Features
- **Real production model shapes** from DeepSeek, Qwen3, etc.
- **Automated testing** with timeout control and logging
- **Flexible configuration** via `TEST_CONFIGS` list
- **Extensible framework** for future fusion kernels
- **Module test pattern** that can be replicated for other kernels

## Usage
```bash
cd op_tests/module_test
python -u test_fused_moe.py
```

### Command Line Options

The test runner supports filtering tests by model and tensor parallel size:

```bash
# Run all tests (default behavior)
python test_fused_moe.py

# Run tests for a specific model (all tp_sizes)
python test_fused_moe.py --model "DeepSeek-R1"

# Run tests for a specific model and tensor parallel size
python test_fused_moe.py --model "Qwen3-235B-A22B" --tp_size 4

# List all available test configurations
python test_fused_moe.py --list-configs

# Get help and see all options
python test_fused_moe.py --help
```

### Usage Constraints

- **`--tp_size` can only be used with `--model`**: This prevents ambiguity since the same tensor parallel size might exist across different models
- **Model names are case-sensitive**: Use exact names like "DeepSeek-R1", "Qwen3-235B-A22B", "Qwen3-30B-A3B"
- **Valid tp_sizes**: Only 4 and 8 are supported

### Examples by Use Case

**Development and Debugging:**
```bash
# Test a specific configuration quickly
python test_fused_moe.py --model "DeepSeek-R1" --tp_size 8

# Verify all configurations for a model
python test_fused_moe.py --model "Qwen3-30B-A3B"
```

**CI/CD and Batch Testing:**
```bash
# Run all tests (comprehensive validation)
python test_fused_moe.py

# Run tests for specific model family
python test_fused_moe.py --model "Qwen3-235B-A22B"
```

**Configuration Discovery:**
```bash
# See all available test configurations
python test_fused_moe.py --list-configs

# Get detailed help
python test_fused_moe.py --help
```

### Test Output and Logging

When running tests, the script provides detailed feedback:

**Console Output:**
- Progress indicators showing which test is currently running
- Real-time status updates (PASS/FAIL/TIMEOUT/ERROR)
- Summary of results at the end
- Clear error messages for invalid arguments

**Log Files:**
- Individual test logs in `test_logs/` directory
- Overall results summary in `test_results.txt`
- Detailed command execution logs for debugging

**Example Output:**
```
2024-01-15 10:30:15 - INFO - Running filtered tests:
  Model: DeepSeek-R1
  TP Size: All available sizes for DeepSeek-R1
  Total tests to run: 2
  Selected configurations:
    - DeepSeek-R1 (TP8)
    - DeepSeek-R1 (TP4)

2024-01-15 10:30:16 - INFO - Progress: 1/2
2024-01-15 10:30:16 - INFO - Running test: model=DeepSeek-R1.tp8.hidden_dim=7168.moe_intermediate_size=256.expert=256.topk=8
```

## Configuration

### TestConfig Parameters
```python
TestConfig(
    model_name,        # Model name (e.g., "DeepSeek-R1")
    tp_size,          # Tensor Parallel size (8 or 4)
    dim,              # "hidden_dim,moe_intermediate_dim"
    expert,           # Number of MoE experts
    topk              # Top-k experts to use
)
```

## Supported Models

**Real production model shapes** from industry-leading AI models:

- **DeepSeek-R1**: TP8/TP4 with 256 experts
- **Qwen3-235B-A22B**: TP8/TP4 with 128 experts  
- **Qwen3-30B-A3B**: TP8/TP4 with 128 experts

## Adding New Configurations

Add to `TEST_CONFIGS` list:
```python
TestConfig("NewModel-Name", 8, "8192,4096", 512, 16)
```

**Pro Tip**: Use real production model configurations for maximum relevance.

## Output Files

- **Logs**: `test_logs/` directory with detailed test output
- **Results**: `test_results.txt` with PASS/FAIL status
- **Format**: `model={name}.tp{size}.hidden_dim={dim}.moe_intermediate_size={size}.expert={num}.topk={num}.log`

## Troubleshooting

- Check logs in `test_logs/` directory
- Copy commands from logs to manually reproduce issues
- Ensure parameters are within reasonable ranges

## Module Test Pattern

The `TestConfig` format and testing structure established here serves as a **template for future module tests**. When implementing tests for new fusion kernels:

- **Follow the same structure**: Use similar `TestConfig` parameters
- **Maintain consistency**: Keep same logging and error handling patterns
- **Enable standardization**: Consistent testing across all fusion kernels
- **Simplify maintenance**: Common infrastructure reduces duplication
- **Replicate the pattern**: Copy this directory structure for new kernel implementations
