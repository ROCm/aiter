# Contributing to AITER

Thank you for your interest in contributing to AITER! We are building a high-performance inference runtime optimized for AMD GPUs with ROCm. Our community welcomes contributions of all kinds, whether you're fixing bugs, optimizing kernels, adding new operators, or improving documentation.

## Ways to Contribute

There are several ways you can contribute to AITER:

* **Report Issues**: Identify and report bugs, performance issues, or unexpected behavior
* **Add Operators**: Request or implement new operators for LLM inference
* **Optimize Kernels**: Improve existing HIP/CK/Triton kernel performance
* **Hardware Support**: Extend support for new AMD GPU architectures (MI200, MI300, etc.)
* **Documentation**: Improve docs, add tutorials, or write performance guides
* **Code Review**: Review pull requests and provide constructive feedback
* **Community Support**: Answer questions and help other users

We also encourage you to share your experiences with AITER in blog posts, social media, or conference talks. If AITER helps your project, please consider starring our repository!

---

## Getting Started

### Job Board

Not sure where to start? Check out these tasks:

* **Good First Issues**: Simple bugs or small enhancements
* **Help Wanted**: Features or optimizations that need community help
* **Kernel Optimization**: Performance improvement opportunities for existing operators
* **New Operator Requests**: Missing operators needed for new models

### Prerequisites

Before contributing, ensure you have:

* **AMD GPU**: MI200, MI300 series, or compatible ROCm hardware
* **ROCm**: Version 5.7+ installed and configured
* **Python**: 3.9, 3.10, 3.11, or 3.12
* **PyTorch**: ROCm-enabled PyTorch 2.0+
* **Git**: For version control

---

## Development Setup

### 1. Fork and Clone

Fork the AITER repository to your GitHub account, then clone it:

```bash
git clone https://github.com/<<your_username>>/aiter.git
cd aiter
git remote add upstream https://github.com/ROCm/aiter.git  # Add upstream remote
```

### 2. Set Up Python Environment

We recommend using a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install AITER from Source

For Python-only development:

```bash
# Install with precompiled kernels
PREBUILD_KERNELS=1 GPU_ARCHS="gfx942;gfx950" python setup.py develop
```

For full development (Python + HIP/CK kernels):

```bash
# Build all kernels from source
pip install -e .
```

### 4. Install Development Dependencies

```bash
pip install -r requirements/dev.txt
```

---

## Code Quality

### Pre-commit Hooks

AITER uses `pre-commit` to maintain code quality. Install it before making changes:

```bash
pip install black==25.1.0 ruff==0.11.11
apt install clang-format  # For C++/HIP code formatting

# Install pre-commit hooks
bash ./.githooks/install
```

This will automatically:
* Format Python code with `black`
* Lint Python code with `ruff`
* Format C++/HIP code with `clang-format`
* Check for common issues

### Manual Linting

To manually run linters:

```bash
# Python formatting
black aiter/ tests/

# Python linting
ruff check aiter/ tests/

# C++/HIP formatting
find csrc/ -name "*.cu" -o -name "*.h" -o -name "*.cpp" | xargs clang-format -i
```

### Code Style Guidelines

**Python Code:**
* Follow [PEP 8](https://pep8.org/)
* Use type hints for function signatures
* Maximum line length: 88 characters (black default)
* Use descriptive variable names

**C++/HIP Code:**
* Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
* Use `snake_case` for functions and variables
* Use `PascalCase` for classes
* Comment complex kernel logic and optimizations
* Always document performance-critical sections

**Kernel Development:**
* Optimize for memory bandwidth (most operators are memory-bound)
* Use vectorized loads/stores when possible (vec4, vec8, vec16)
* Minimize global memory accesses
* Document arithmetic intensity and roofline analysis
* Include performance benchmarks in PR description
* **Minimize external dependencies** - avoid adding new third-party libraries

---

## Testing

### Running Tests

AITER tests are standalone Python scripts in the `op_tests/` directory:

```bash
# Run all tests using the CI script
bash .github/scripts/aiter_test.sh

# Run a specific test file directly
python op_tests/test_rmsnorm2d.py

# Run with specific parameters
python op_tests/test_rmsnorm2d.py --dtype bf16 --m 1024 --n 4096

# Run Triton-specific tests
python op_tests/triton_tests/normalization/test_rmsnorm.py

# Run multi-GPU tests
MULTIGPU=TRUE bash .github/scripts/aiter_test.sh
```

### Adding Tests

When adding new features or fixing bugs, include tests. AITER tests are standalone Python scripts:

```python
# op_tests/test_new_operator.py
import torch
import argparse
from aiter.ops import new_operator
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes

@perftest()
def run_reference(input, param):
    """Reference implementation."""
    # Your reference implementation
    return expected_output

@perftest()
def run_aiter(input, param):
    """AITER optimized implementation."""
    return new_operator(input, param)

def test_new_operator(dtype, m, n):
    """Test operator correctness and performance."""
    input_tensor = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Run both implementations
    expected = run_reference(input_tensor, param)
    output = run_aiter(input_tensor, param)
    
    # Check correctness
    checkAllclose(output, expected, rtol=1e-3, atol=1e-3)
    print(f"✓ Test passed for dtype={dtype}, m={m}, n={n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dtype", type=str, default=None,
                       help="Data type (e.g., bf16, fp16)")
    parser.add_argument("-m", type=int, default=None, help="M dimension")
    parser.add_argument("-n", type=int, default=None, help="N dimension")
    
    args = parser.parse_args()
    
    # Define test configurations
    l_dtype = [torch.float16, torch.bfloat16]
    l_m = [1024, 2048, 4096]
    l_n = [4096, 8192]
    
    if args.dtype:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.m:
        l_m = [args.m]
    if args.n:
        l_n = [args.n]
    
    # Run tests
    for dtype in l_dtype:
        for m in l_m:
            for n in l_n:
                test_new_operator(dtype, m, n)
```

**Key Features:**
- Use `@perftest()` decorator for performance timing
- Use `checkAllclose()` for numerical comparison
- Support command-line arguments for flexible testing
- Print clear pass/fail messages

### Testing on Different Hardware

If you don't have access to specific AMD GPU models, mention this in your PR. Our CI system will run tests on:
* MI250X (gfx90a)
* MI300X (gfx942)

---

## Kernel Development

### General Principles

When developing new operators:

* **Minimize External Dependencies**: Avoid introducing new third-party libraries whenever possible
  - Prefer using existing dependencies: PyTorch, ROCm/HIP, Composable Kernel (CK), Triton
  - If a new dependency is absolutely necessary, discuss it in the PR and provide strong justification
  - Consider implementing functionality from scratch if the dependency is small or simple
  - Avoid dependencies that are not well-maintained or AMD GPU-specific

### HIP Kernel Development

When developing or modifying HIP kernels:

1. **Use JIT Compilation System**:
   ```python
   from aiter.jit import compile_ops
   
   @compile_ops(
       srcs=["path/to/kernel.cu"],
       extra_hip_flags=["-O3", "-DCK_TILE_FMHA_FWD_FAST_EXP2=1"]
   )
   def my_operator(input: torch.Tensor) -> torch.Tensor:
       return torch_ops.my_operator_kernel(input)
   ```

2. **Profile Memory Access Patterns**:
   ```bash
   # Use rocprof to analyze kernel performance
   rocprof --stats python tests/profile_kernel.py
   ```

3. **Check Roofline Model**:
   - Document arithmetic intensity (FLOPs/Byte)
   - Compare against hardware peak (MI300X: ~380 TFLOPS FP16, 3.2 TB/s)
   - Explain if kernel is compute-bound or memory-bound

4. **Optimize for Coalesced Access**:
   ```cpp
   // Good: Coalesced access
   vec8_t<scalar_t> data = vectorized_ptr[blockIdx.x * vec_size + threadIdx.x];
   
   // Bad: Strided access
   scalar_t data = ptr[threadIdx.x * stride];  // Avoid when stride is large
   ```

### Composable Kernel (CK) Integration

When integrating CK tiles:

1. **Generate CK Instances**:
   ```bash
   cd 3rdparty/composable_kernel/example/ck_tile/01_fmha
   python generate.py -d fwd --receipt 200 --filter "*bf16*" --output_dir /tmp/ck_gen
   ```

2. **Add to JIT Pipeline**:
   ```python
   blob_gen_cmd = [
       f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --filter {filter_pattern} --output_dir {{}}"
   ]
   ```

3. **Register with PyTorch**:
   ```cpp
   TORCH_LIBRARY(aiter, m) {
       m.def("my_ck_op(Tensor input) -> Tensor");
   }
   ```

### Triton Kernel Development

For Triton kernels:

1. **Use Appropriate Block Sizes**:
   ```python
   @triton.jit
   def kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       # Use BLOCK_SIZE between 128-1024 for AMD GPUs
       pass
   ```

2. **Enable Software Pipelining**:
   ```python
   for blk_idx in tl.range(0, n_blocks, num_stages=2):  # Enable pipelining
       data = tl.load(ptr + blk_idx * BLOCK_SIZE)
   ```

3. **Use Cache Modifiers**:
   ```python
   # For data that will be reused
   x = tl.load(input_ptr, cache_modifier=".cg")  # Cache globally
   ```

---

## Performance Testing

### Benchmarking Operators

Always benchmark your changes:

```bash
# Benchmark single operator
python benchmarks/bench_rmsnorm.py --dtype fp16 --hidden-size 4096

# Compare before/after
python benchmarks/compare.py --baseline main --pr my-branch --op rmsnorm

# Full model benchmark
python benchmarks/bench_llama.py --model llama-7b --batch-size 32
```

### Performance Requirements

For kernel PRs, include in description:
* **Hardware**: GPU model (e.g., MI300X)
* **Baseline**: Performance before changes
* **Optimized**: Performance after changes
* **Improvement**: Percentage gain
* **Bandwidth Utilization**: % of peak memory bandwidth
* **Roofline Analysis**: Where operator sits on roofline model

Example:
```
## Performance Results (MI300X)

| Config | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| FP16, BS=1024, HS=4096 | 180 μs | 150 μs | 16.7% |
| BF16, BS=2048, HS=8192 | 720 μs | 600 μs | 16.7% |

Bandwidth Utilization: 78% of peak (2.5 TB/s / 3.2 TB/s)
Arithmetic Intensity: 0.83 FLOPs/Byte (memory-bound as expected)
```

---

## Documentation

### Building Documentation

AITER uses Sphinx for documentation:

```bash
# Install documentation dependencies
pip install -r requirements/docs.txt

# Build docs
cd docs/
make html

# View locally
python -m http.server 8000 -d _build/html
# Open http://localhost:8000
```

### Writing Documentation

When adding features, update docs:

1. **API Documentation**: Add docstrings to public functions
   ```python
   def new_operator(input: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
       """
       Apply new operator to input tensor.
       
       Args:
           input: Input tensor of shape [batch, hidden_size]
           epsilon: Small constant for numerical stability
           
       Returns:
           Output tensor of same shape as input
           
       Example:
           >>> x = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
           >>> y = new_operator(x)
       """
       pass
   ```

2. **Operator Guides**: Add operator-specific documentation
   - Algorithm description
   - Performance characteristics
   - Memory usage
   - Supported data types and shapes

3. **Kernel Optimization Guides**: Document optimization techniques
   - Memory access patterns
   - Roofline analysis
   - Tuning parameters

---

## Pull Requests

### Before Submitting

Ensure your PR:
* [ ] Passes all pre-commit hooks
* [ ] Includes relevant tests
* [ ] Updates documentation if needed
* [ ] Includes performance benchmarks (for kernel changes)
* [ ] Has a clear, descriptive title

### PR Title Format

Use one of these prefixes:

* `[Bugfix]` - Bug fixes
* `[Feature]` - New features or operators
* `[Kernel]` - Kernel optimizations or new kernels
* `[HIP]` - HIP-specific changes
* `[CK]` - Composable Kernel integration
* `[Triton]` - Triton kernel changes
* `[JIT]` - JIT compilation system changes
* `[Perf]` - Performance optimizations
* `[Doc]` - Documentation improvements
* `[Test]` - Test additions or fixes
* `[CI]` - CI/CD improvements
* `[Hardware]` - Hardware-specific changes (e.g., `[Hardware][MI300X]`)
* `[Misc]` - Miscellaneous changes

Examples:
* `[Kernel][Perf] Optimize RMSNorm for MI300X using vec16 loads`
* `[Feature] Add PagedAttention operator with CK backend`
* `[Bugfix] Fix numerical instability in FP16 softmax`

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Detailed list of changes
- Impact on existing code

## Performance (if applicable)
| Configuration | Before | After | Improvement |
|---------------|--------|-------|-------------|
| ...           | ...    | ...   | ...         |

## Testing
- [ ] Unit tests added/updated
- [ ] Performance benchmarks run
- [ ] Tested on MI250X
- [ ] Tested on MI300X

## Documentation
- [ ] Docstrings updated
- [ ] User guide updated
- [ ] Performance guide updated

## Dependencies
- [ ] No new third-party dependencies added
- [ ] If new dependencies added, justification provided

## Breaking Changes
List any breaking changes and migration guide
```

### Code Review Process

1. **Initial Review**: A maintainer will review within 3-5 business days
2. **Feedback**: Address comments and push updates
3. **Approval**: After approval, CI will run full test suite
4. **Merge**: Once CI passes, a maintainer will merge

If your PR is urgent or hasn't been reviewed, ping maintainers on the issue or PR.

---

## Specific Contribution Areas

### Adding New Models

When adding support for a new model:

1. Identify required operators
2. Check if operators exist in AITER
3. Implement missing operators
4. Add model configuration
5. Add accuracy tests
6. Benchmark performance

### Optimizing Existing Kernels

For kernel optimizations:

1. Profile baseline performance with `rocprof`
2. Identify bottleneck (memory-bound vs compute-bound)
3. Apply optimizations:
   - Increase vectorization width
   - Improve memory coalescing
   - Use shared memory effectively
   - Enable software pipelining
4. Verify correctness with tests
5. Document performance improvement

### Hardware-Specific Optimizations

For new AMD GPU architectures:

1. Check architecture features (cache sizes, VGPR count, LDS size)
2. Tune kernel parameters for new hardware
3. Update `get_gfx()` detection
4. Add hardware-specific code paths if needed
5. Update CI to test on new hardware

---

## Community

### Getting Help

* **Issues**: [GitHub Issues](https://github.com/ROCm/aiter/issues)
* **Discussions**: [GitHub Discussions](https://github.com/ROCm/aiter/discussions)
* **Slack**: [Join our Slack workspace](#)

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and constructive in all interactions.

---

## Developer Certificate of Origin (DCO)

By contributing to AITER, you certify that your contribution was created in whole or in part by you and you have the right to submit it under the Apache 2.0 license.

All commits must be signed off:

```bash
git commit -s -m "Your commit message"
```

This adds a `Signed-off-by` line to your commit message.

**Tip**: Enable automatic sign-off in your Git config:
```bash
git config --global format.signoff true
```

---

## Advanced Topics

### JIT Compilation System

AITER uses a custom JIT system for compiling kernels. When modifying:

1. **Adding New Modules**: Update `optCompilerConfig.json`
2. **Dynamic Configuration**: Use `gen_func` in `@compile_ops` decorator
3. **Blob Generation**: Add code generation commands to `blob_gen_cmd`

See [JIT_COMPILE_FLOW_ANALYSIS.md](JIT_COMPILE_FLOW_ANALYSIS.md) for details.

### Debugging Kernels

```bash
# Enable debug symbols
export AITER_DEBUG=1

# Keep intermediate files
export AITER_SAVE_TEMPS=1

# Verbose compilation
export AITER_VERBOSE=1

# Run with kernel debugging
python tests/test_kernel.py
```

### Profiling Tools

**rocprof**: Profile GPU kernels
```bash
rocprof --stats --basenames on python benchmark.py
```

**roctracer**: Trace HIP API calls
```bash
roctracer -o trace.json python benchmark.py
```

**Omniperf**: Detailed GPU profiling
```bash
omniperf profile -n my_profile -- python benchmark.py
omniperf analyze -p workloads/my_profile/
```

---

## FAQ

**Q: My kernel compiles but is slower than expected. What should I check?**

A: 
1. Check memory access patterns (use `rocprof`)
2. Verify coalescing (check `MemUnitBusy`)
3. Measure bandwidth utilization
4. Compare against roofline model
5. Profile with `omniperf` for detailed metrics

**Q: How do I test on hardware I don't have access to?**

A: Submit your PR and mention hardware limitations. Our CI will test on MI250X and MI300X.

**Q: When should I use HIP vs CK vs Triton?**

A:
* **HIP**: Maximum control, hardware-specific optimizations
* **CK**: High-performance tile-based GEMM-like operations
* **Triton**: Rapid prototyping, easier to write

**Q: My PR conflicts with main branch. How do I resolve?**

A:
```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts
git push --force-with-lease origin my-branch
```

**Q: Can I add a new third-party library dependency?**

A:
We strongly prefer to avoid new dependencies. If you believe a new library is necessary:
1. Check if the functionality can be implemented using existing dependencies (PyTorch, HIP, CK, Triton)
2. Consider implementing the feature from scratch if it's relatively simple
3. If the dependency is essential, provide strong justification in your PR:
   - Why existing solutions don't work
   - Performance benefits or features it provides
   - Library maintenance status and AMD GPU support
   - Impact on build time and binary size

---

## Thank You!

Thank you for contributing to AITER! Your contributions help make AITER the best inference runtime for AMD GPUs. Whether you're optimizing a single kernel or adding a major feature, we appreciate your effort and dedication to the project.

Happy coding! 🚀
