# AITER GEMM Tuning & Gradlib Guide

This guide documents the GEMM auto-tuning system, including the tuned GEMM dispatch layer, CSV-based configuration, gradlib tuning framework, and how to tune for new model shapes.

---

## Quick Reference

| Task | How |
|------|-----|
| **Use tuned GEMM** | `aiter.tuned_gemm.gemm_a16w16(x, weight)` — auto-dispatches |
| **Capture untuned shapes** | `AITER_TUNE_GEMM=1 python your_workload.py` |
| **Run BF16 tuner** | `python3 gradlib/gradlib/gemm_tuner.py --input_file ... --tuned_file ...` |
| **Override config** | `AITER_CONFIG_GEMM_BF16="/path/to/custom.csv"` |
| **Merge configs** | `AITER_CONFIG_GEMM_BF16="/path1.csv:/path2.csv"` |
| **Log config selection** | `AITER_LOG_TUNED_CONFIG=1` |

---

## 1. How the Tuned GEMM System Works

### Architecture

```
User code: aiter.tuned_gemm.gemm_a16w16(x, weight)
    │
    ▼
get_GEMM_A16W16_config(M, N, K, ...)  ← LRU-cached dispatch
    │
    ├── Lookup (cu_num, M, N, K, dtype...) in tuned CSV
    ├── Try padded M values if exact match not found
    └── Fallback heuristics if no config exists
    │
    ▼
Dispatch to best backend:
    ├── "asm"       → Assembly GEMM kernels
    ├── "hipblaslt" → hipBLASLt library
    ├── "skinny"    → Custom skinny-M kernels
    ├── "triton"    → Triton GEMM kernel
    └── "torch"     → PyTorch F.linear (fallback)
```

### Dispatch Flow

1. Load the tuned CSV (once, cached).
2. Index by `(cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`.
3. Look up exact M first, then try padded M values at two granularity levels.
4. If a match is found, dispatch to the stored `libtype` with its `solidx` (solution index).
5. If no match, apply fallback heuristics (hipBLASLt for preshuffle, skinny for small M, torch otherwise).

---

## 2. CSV Configuration Format

### Tuned CSV Columns

```
cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle, libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw
```

| Column | Description |
|--------|-------------|
| `cu_num` | GPU compute units (e.g., 304 for MI300X, 256 for MI350X) |
| `M, N, K` | GEMM dimensions |
| `bias` | Whether bias is added |
| `dtype` | Input data type |
| `outdtype` | Output data type |
| `scaleAB` | Per-tensor FP8 scaling |
| `bpreshuffle` | Weight pre-shuffled layout |
| `libtype` | Backend: `hipblaslt`, `asm`, `triton`, `skinny`, `torch`, `ck` |
| `solidx` | Solution/kernel index within the library |
| `splitK` | Split-K factor for assembly kernels |
| `us` | Execution time in microseconds |
| `kernelName` | Full kernel name string |
| `tflops` | Achieved TFLOPS |

### Available Config Files

| File | Format | Description |
|------|--------|-------------|
| `bf16_tuned_gemm.csv` | BF16 | BF16/FP16 GEMM configs |
| `a8w8_tuned_gemm.csv` | A8W8 | FP8/INT8 per-tensor GEMM |
| `a8w8_blockscale_tuned_gemm.csv` | A8W8 blockscale | FP8 block-scale GEMM |
| `a8w8_bpreshuffle_tuned_gemm.csv` | A8W8 preshuffle | A8W8 with weight pre-shuffling |
| `a4w4_blockscale_tuned_gemm.csv` | A4W4 | MXFP4 block-scale GEMM |
| `bf16_tuned_batched_gemm.csv` | Batched BF16 | Batched BF16 GEMM |
| `a8w8_tuned_batched_gemm.csv` | Batched A8W8 | Batched FP8/INT8 |
| `tuned_fmoe.csv` | Fused MoE | Fused MoE kernel configs |

### Model-Specific Configs

Pre-tuned configs in `aiter/configs/model_configs/`:

| File | Model |
|------|-------|
| `dsv3_bf16_tuned_gemm.csv` | DeepSeek-V3 |
| `llama70B_untuned_gemm.csv` | Llama 3.3-70B |
| `llama405B_untuned_gemm.csv` | Llama 3.1-405B |
| `qwen32B_untuned_gemm.csv` | Qwen3-32B |

---

## 3. Running the Tuner

### Step 1: Capture GEMM Shapes

Run your workload with shape capture enabled:

```bash
AITER_TUNE_GEMM=1 python your_inference_script.py
```

This writes encountered shapes to `aiter/configs/bf16_untuned_gemm.csv`.

### Step 2: Run the BF16 GEMM Tuner

```bash
python3 gradlib/gradlib/gemm_tuner.py \
    --tuned_file aiter/configs/bf16_tuned_gemm.csv \
    --input_file aiter/configs/bf16_untuned_gemm.csv
```

### Tuner Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mp N` | all GPUs | Parallel tuning across N GPUs |
| `--errRatio 0.05` | 5% | Max tolerable error vs reference |
| `--libtype all` | `all` | Backends to test: `all`, `asm`, `hipblaslt`, `triton` |
| `--all` | — | Retune all shapes (even previously tuned) |
| `--batch 100` | — | Process in batches of this size |
| `--profile_file path.csv` | — | Save all candidate results |
| `--verbose` | — | Detailed logging |

### Model-Based Tuning

```bash
python3 gradlib/gradlib/gemm_tuner.py \
    --model_dir /path/to/model \
    --tp 8 \
    --nsets 1,512,1024,2048,4096
```

Auto-generates shapes from the model's `config.json` and tunes for specified batch sizes.

### For Quantized GEMM Tuning

Each quantization format has its own tuner:

| Format | Tuner Location |
|--------|---------------|
| A8W8 per-tensor | `csrc/ck_gemm_a8w8/` |
| A8W8 blockscale | `csrc/ck_gemm_a8w8_blockscale/` |
| A8W8 bpreshuffle | `csrc/ck_gemm_a8w8_bpreshuffle/` |
| A4W4 blockscale | `csrc/ck_gemm_a4w4_blockscale/` |
| Batched A8W8 | `csrc/ck_batched_gemm_a8w8/` |
| Fused MoE | `csrc/ck_gemm_moe_2stages_codegen/` |

---

## 4. Tuning Process (What the Tuner Does)

### Phase 1: Fast Screening (hipBLASLt only)

1. `hipb_findallsols()` enumerates all hipBLASLt algorithm candidates.
2. Benchmarks with reduced warmup (5 iters, 2 cold).
3. Selects top 20 fastest.

### Phase 2: Accurate Benchmarking (all backends)

1. Re-benchmarks top 20 hipBLASLt solutions with full warmup + correctness check.
2. Benchmarks ASM kernels (all tile sizes × split-K factors).
3. Benchmarks Triton GEMM kernel.
4. Filters by error ratio, selects fastest valid solution.

hipBLASLt is preferred unless another backend is ≥0.5% faster (`hipb_prefer_ratio = 0.995`).

---

## 5. Backend Details

### hipBLASLt (Primary)

AMD's high-performance GEMM library. Provides hundreds of algorithm candidates per shape.

```python
aiter.hipb_create_extension()  # Initialize (once)
aiter.hipb_mm(A, B, solution_index, bias=None, out_dtype=None, scaleA=None, scaleB=None)
```

### ASM (Assembly)

Hand-tuned assembly kernels for specific tile sizes. Supports split-K for small-M shapes.

### Skinny GEMM

Custom kernels optimized for small M (batch size) with large N and K.

### Triton

Triton-based GEMM with auto-config selection. Portable but may not match hipBLASLt/ASM peak performance.

---

## 6. Environment Variables

### Config Overrides

| Variable | Default CSV |
|----------|------------|
| `AITER_CONFIG_GEMM_BF16` | `aiter/configs/bf16_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8` | `aiter/configs/a8w8_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` | `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | `aiter/configs/a8w8_blockscale_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A4W4` | `aiter/configs/a4w4_blockscale_tuned_gemm.csv` |
| `AITER_CONFIG_A8W8_BATCHED_GEMM` | `aiter/configs/a8w8_tuned_batched_gemm.csv` |
| `AITER_CONFIG_BF16_BATCHED_GEMM` | `aiter/configs/bf16_tuned_batched_gemm.csv` |
| `AITER_CONFIG_FMOE` | `aiter/configs/tuned_fmoe.csv` |

All support colon-separated multi-path values for merging: `"/path1.csv:/path2.csv"`.

### Runtime

| Variable | Description |
|----------|-------------|
| `AITER_TUNE_GEMM` | `1` = capture untuned shapes during runtime |
| `AITER_LOG_TUNED_CONFIG` | `1` = log which config was selected |
| `HIP_ONLINE_TUNING` | `1` = hipBLASLt online tuning at runtime |

---

## 7. Adding Custom Configs

### Method 1: Environment Variable

```bash
export AITER_CONFIG_GEMM_BF16="/path/to/my_tuned.csv"
# Or merge with default:
export AITER_CONFIG_GEMM_BF16="/path/to/my_tuned.csv:aiter/configs/bf16_tuned_gemm.csv"
```

### Method 2: Model Config Template

1. Create shapes in `aiter/configs/model_configs/my_model_untuned_gemm.csv`.
2. Copy to `aiter/configs/bf16_untuned_gemm.csv`.
3. Run the tuner.

---

## 8. Source Files

| Component | Path |
|---|---|
| Tuned GEMM dispatch | `aiter/tuned_gemm.py` |
| BF16 GEMM tuner | `gradlib/gradlib/gemm_tuner.py` |
| Tuner base class | `aiter/utility/base_tuner.py` |
| hipBLASLt bindings | `aiter/ops/gradlib.py` |
| hipBLASLt C++ | `gradlib/csrc/hipbsolgemm.cu` |
| rocBLAS C++ | `gradlib/csrc/rocsolgemm.cu` |
| Config CSV files | `aiter/configs/*.csv` |
| Model-specific configs | `aiter/configs/model_configs/` |

---

## 9. Test Files

| Test | Path |
|------|------|
| Tuned GEMM benchmarks | `op_tests/op_benchmarks/test_tuned_gemm.py` |
| GEMM A16W16 | `op_tests/triton_tests/gemm/test_gemm_a16w16.py` |
| GEMM A8W8 | `op_tests/test_gemm_a8w8.py` |
