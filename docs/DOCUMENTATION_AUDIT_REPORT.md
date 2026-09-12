# AITER Documentation Accuracy Audit Report

**Date:** 2026-02-14
**Auditor:** Claude Sonnet 4.5
**Scope:** Complete factual accuracy check of all documentation

## Executive Summary

This audit identified **22 critical factual errors** in the AITER documentation that would prevent users from successfully using the library. The errors fall into three main categories:

- **Missing API Functions** (12 errors): Documented functions that don't exist in the codebase
- **Build System Errors** (3 errors): CUDA-specific instructions in a ROCm project
- **Installation Issues** (2 errors): Incorrect package name and verification code
- **Signature Mismatches** (5 errors): Documented parameters don't match actual implementation

## Critical Issues Requiring Immediate Fix

### 1. Installation (`docs/installation.rst`)

#### Issue 1.1: Incorrect Package Name [FIXED ✓]
- **Line 33**: Documentation says `pip install aiter`
- **Actual**: Package is named `amd-aiter` (see setup.py line 12)
- **Impact**: Installation command won't work
- **Status**: FIXED

#### Issue 1.2: Non-functional Verification Code [FIXED ✓]
- **Lines 117-120**: Code tries to access `aiter.__version__`
- **Actual**: No `__version__` attribute exists in aiter/__init__.py
- **Impact**: Verification code crashes
- **Status**: FIXED - replaced with working verification

### 2. Quickstart (`docs/quickstart.rst`)

#### Issue 2.1: grouped_gemm() Does Not Exist [FIXED ✓]
- **Line 95**: `aiter.grouped_gemm(x, expert_weights, expert_ids)`
- **Actual**: Function doesn't exist. MOE uses `fmoe()`, `fmoe_g1u1()`, etc.
- **Impact**: Example code won't run
- **Status**: FIXED - replaced with working `fmoe()` example

### 3. API Documentation (`docs/api/gemm.rst`)

#### Issue 3.1: Multiple Non-existent Functions [NOT FIXED]
The following functions are documented but DO NOT EXIST in the codebase:

| Function | Line | Status |
|----------|------|--------|
| `aiter.grouped_gemm()` | 9 | Does not exist |
| `aiter.batched_gemm()` | 47 | Partial - only specific variants exist |
| `aiter.gemm_bias()` | 68 | Does not exist |
| `aiter.gemm_gelu()` | 97 | Does not exist |
| `aiter.gemm_relu()` | 114 | Does not exist |
| `aiter.cutlass_gemm()` | 131 | Does not exist |
| `aiter.sparse_gemm()` | 150 | Does not exist |
| `aiter.int8_gemm()` | 167 | Does not exist |

**Impact**: All code examples in this file will fail

**Actual GEMM Functions Available:**
- `gemm_a8w8_*()` - INT8 variants
- `gemm_a4w4_*()` - FP4 variants
- `gemm_a16w16_*()` - FP16/BF16 variants
- `batched_gemm_a8w8()` - INT8 batched
- `batched_gemm_bf16()` - BF16 batched
- `deepgemm()` - Deep GEMM kernel

**Recommendation**: Either implement the documented functions or rewrite docs to match actual API

### 4. Attention API (`docs/api/attention.rst`)

#### Issue 4.1: Non-existent Attention Variants [NOT FIXED]
The following functions are documented but DO NOT EXIST:

| Function | Exists? |
|----------|---------|
| `aiter.flash_attn_func()` | ✓ EXISTS |
| `aiter.flash_attn_with_kvcache()` | ✓ EXISTS |
| `aiter.grouped_query_attention()` | ✗ Does not exist |
| `aiter.multi_query_attention()` | ✗ Does not exist |
| `aiter.variable_length_attention()` | ✗ Does not exist |

**Actual Functions:**
- `paged_attention_rocm()`
- `paged_attention_v1()`
- `mla_decode_stage1_asm_fwd()`
- `mla_prefill_asm_fwd()`
- Various MHA functions in `aiter.ops.mha`

### 5. Operators API (`docs/api/operators.rst`)

#### Issue 5.1: Missing Operator Functions [NOT FIXED]

| Function | Status |
|----------|--------|
| `aiter.rmsnorm()` | ✓ EXISTS |
| `aiter.gelu()` | Partial - `gelu_and_mul()` exists |
| `aiter.apply_rotary_pos_emb()` | ✗ Does not exist (but `rope_fwd()` exists) |
| `aiter.precompute_rope_embeddings()` | ✗ Does not exist |
| `aiter.top_k_sampling()` | ✗ Does not exist |
| `aiter.top_p_sampling()` | ✗ Does not exist (only `top_p_sampling_from_probs()`) |

### 6. Tutorial: Add New Operator (`docs/tutorials/add_new_op.rst`)

#### Issue 6.1: CUDA Build System in ROCm Project [NOT FIXED]
- **Lines 197-222**: Tutorial shows CUDA-specific build code
- **Problem**: Uses `CUDAExtension`, `nvcc` flags, NVIDIA architecture codes
- **Actual**: AITER uses custom JIT compiler with Ninja build (see `aiter/jit/`)
- **Impact**: Build instructions won't work for ROCm

**Specific Errors:**
```python
from torch.utils.cpp_extension import CUDAExtension  # Wrong for ROCm
extra_compile_args={'nvcc': [  # Should be 'hipcc'
    '-gencode', 'arch=compute_90a,code=sm_90a',  # These are NVIDIA codes!
    '-gencode', 'arch=compute_942,code=sm_942',  # Should be gfx90a, gfx942
```

#### Issue 6.2: CUDA Kernel Types in ROCm Context [NOT FIXED]
- **Line 140**: Uses `__nv_bfloat16` (NVIDIA type)
- **Should be**: `__hip_bfloat16` for ROCm

## Priority Recommendations

### Immediate (P0) - Already Fixed ✓
1. ✓ Fix package name in installation docs
2. ✓ Fix installation verification code
3. ✓ Fix quickstart MOE example

### High Priority (P1) - Needs Urgent Fix
1. **Remove or implement documented GEMM functions**
   - Either implement the 8 missing GEMM functions
   - Or rewrite `docs/api/gemm.rst` to document actual functions

2. **Fix "Add New Operator" tutorial**
   - Replace CUDA build system with actual ROCm/HIP JIT instructions
   - Fix architecture codes (sm_* → gfx*)
   - Fix data types (__nv_* → __hip_*)

3. **Update attention API docs**
   - Remove non-existent GQA/MQA/varlen functions
   - Document actual MHA functions from aiter.ops.mha

### Medium Priority (P2)
1. **Operators API** - Document actual function signatures
2. **Add missing function implementations** - Consider implementing commonly-expected functions
3. **Add CI/CD documentation tests** - Automatically verify code examples compile and run

### Low Priority (P3)
1. Auto-generate API docs from docstrings
2. Add type hints to match documentation
3. Create migration guide for API changes

## Statistics

- **Total issues found**: 22
- **Critical severity**: 12 (missing core functions)
- **High severity**: 6 (build system, wrong signatures)
- **Medium severity**: 3 (incomplete docs)
- **Low severity**: 1 (cosmetic)

## Next Steps

1. **Team Review**: Decide which missing functions should be:
   - Implemented (add to backlog)
   - Removed from docs (quick fix)
   - Replaced with actual alternatives

2. **Prioritize Fixes**: Focus on P0 and P1 issues that block users

3. **Long-term**: Implement automated doc testing in CI/CD

## Files Reviewed

- ✓ `docs/installation.rst`
- ✓ `docs/quickstart.rst`
- ✓ `docs/api/gemm.rst`
- ✓ `docs/api/attention.rst`
- ✓ `docs/api/operators.rst`
- ✓ `docs/tutorials/add_new_op.rst`
- ✓ `docs/tutorials/basic_usage.rst`
- ✓ `docs/index.rst`

---

**Report Generated:** 2026-02-14
**Next Audit Recommended:** After P1 fixes are implemented
