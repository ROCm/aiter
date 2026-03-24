```
                      _ _ _ _
   __ _ _ __ __ _  __| | (_) |__
  / _` | '__/ _` |/ _` | | | '_ \
 | (_| | | | (_| | (_| | | | |_) |
  \__, |_|  \__,_|\__,_|_|_|_.__/
  |___/
```

## What Is gradlib
`gradlib` is a vLLM-derived tuning toolkit for GEMM kernels. It helps you find the best kernel parameters for your current hardware to improve model inference performance.

## Quick Start

### 1) Capture Untuned GEMM Shapes
Replace `F.linear` with `tgemm.mm` in `aiter/tuned_gemm.py`, then run your workload:

```bash
AITER_TUNE_GEMM=1 python {workload_tests}
```

Captured shapes are written to `aiter/configs/bf16_untuned_gemm.csv`.

### 2) Tune GEMMs
Run the tuner:

```bash
python3 gradlib/gradlib/gemm_tuner.py \
  --tuned_file aiter/configs/bf16_tuned_gemm.csv \
  --input_file aiter/configs/bf16_untuned_gemm.csv
```

Tuned results are saved to `aiter/configs/bf16_tuned_gemm.csv`.

Example columns:

|**cu_num**|**M**|**N**|**K**|**bias**|**dtype**|**outdtype**|**scaleAB**|**bpreshuffle**|**libtype**|**solidx**|**splitK**|**soltimes**|**kernelName**|**tflops**|**bw**|
|----------|-----|-----|-----|--------|---------|-----------|-----------|---------------|-----------|----------|----------|------------|--------------|----------|------|
|80|128|1536|7168|False|torch.bfloat16|torch.float32|False|False|hipblaslt|667788|0|10.6|xxxxxxx|xx|xx|

Notes:
- `cu_num`: compute units for current GPU.
- `bpreshuffle`: whether weight is shuffled.
- `dtype`: input dtype (`hipblaslt` supports fp8/bf16/fp16; asm/triton supports bf16/fp16).
- `libtype`: kernel backend (`hipblaslt` / `rocblas` / `asm` / `triton`).
- `splitK`: valid when `libtype == asm`.
- `tflops`: throughput in TFLOPS.
- `bw`: bandwidth in GB/s.

### 3) Run Your Workload Normally
After tuning, run your model/tests as usual.

## More Features

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Required**: No
- **Description**: Optional output file storing **all** tuning candidates (not only the best).

**Example**:
```bash
--profile_file /path/to/all_results.csv
```

#### `--mp`
- **Type**: Integer
- **Default**: `torch.cuda.device_count()`
- **Description**: Number of parallel processes / GPUs used for tuning.

**Example**:
```bash
--mp 1
```

### Tuning Configuration

#### `--errRatio`
- **Type**: Float
- **Default**: `0.05` (5%)
- **Description**: Max tolerable error ratio for valid kernels.

**Example**:
```bash
--errRatio 0.01
--errRatio 0.10
```

#### `--sort`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Sort output by key columns.

**Example**:
```bash
--sort
```

#### `--all`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Retune all shapes based on file relationship.
  - If `tune_file == untune_file`: retune all shapes in tune file.
  - If `tune_file != untune_file`: retune shapes that exist in untuned file.

**Example**:
```bash
--all
```

#### `--run_config [TUNED_CSV]`
- **Type**: Optional argument
- **Default**: disabled
- **Description**: Run production benchmark only and exit (no tuning).
  - `--run_config /path/to/tuned.csv`: read shapes from that tuned CSV and run tuned kernels from that file.
  - `--run_config` (no path): read shapes from `--input_file` (or auto-generated shapes when `--input_file` is omitted) and run default kernels.

**Examples**:
```bash
# benchmark tuned kernels from specified tuned config
python3 gradlib/gradlib/gemm_tuner.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --run_config aiter/configs/bf16_tuned_gemm.csv

# benchmark default kernels using shapes from --input_file
python3 gradlib/gradlib/gemm_tuner.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --run_config
```

#### `--compare`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Run pre-tune and post-tune production benchmark and report speedup.
  - Pre-tune reads shapes from `--input_file` (or auto-generated shapes).
  - Post-tune uses configs written to `--tuned_file` (alias of `-o/--tune_file`).

**Example**:
```bash
--compare
```

### Debugging and Verbose Output

#### `-v, --verbose`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable verbose logs.

**Example**:
```bash
--verbose
-v
```

## hipBLASLt Online Tuning

Enable hipBLASLt online tuning:

```bash
export HIP_ONLINE_TUNING=1
```

The one-time overhead can take several minutes. Results are saved to `hip_online_tuning_res.csv`.

