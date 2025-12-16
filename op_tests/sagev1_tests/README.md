# CogVideoX Attention Comparison Tools

This directory contains tools for rendering CogVideoX videos with different attention implementations and creating side-by-side comparison videos.

## Overview

- **`sageattn_cogvideo.py`** - Renders a CogVideoX video using a specified attention implementation
- **`render_all_attention.sh`** - Batch renders videos for all attention types
- **`stitch_comparison.py`** - Stitches multiple videos into a grid layout with labels

## Prerequisites

### For Video Rendering

```bash
> pip install diffusers transformers opencv-python
> pip install sentencepiece protobuf
> pip install accelerate==1.12.0 --no-deps
```

The CogVideoX model will be automatically downloaded from Hugging Face on first run.

### For Video Stitching

```bash
pip install moviepy
```

MoviePy also requires ImageMagick for text rendering:

```bash
# Ubuntu/Debian
sudo apt-get install imagemagick
```

## Available Attention Types

| Type | Description |
|------|-------------|
| `sdpa` | PyTorch native scaled dot product attention |
| `sagev1` | SageAttention v1 (INT8 quantized Q/K) |
| `fav2` | Flash Attention v2 (Triton) |
| `fav3` | Flash Attention v3 (Triton) |
| `fav3_fp8` | Flash Attention v3 with FP8 quantization |
| `fav3_sage` | SageAttention v1 fused on FA3 backend |

## Usage

### Option 1: Render All + Stitch (Recommended)

Render videos for all attention types and automatically create a comparison grid:

```bash
./op_tests/sagev1_tests/render_all_attention.sh --stitch
```

### Option 2: Step-by-Step

#### Step 1: Render Individual Videos

```bash
# Render a single attention type
python op_tests/sagev1_tests/sageattn_cogvideo.py --attention_type sagev1

# Render with a local model path
python op_tests/sagev1_tests/sageattn_cogvideo.py \
    --attention_type fav3_fp8 \
    --model_path /path/to/local/model
```

#### Step 2: Render All Attention Types

```bash
# Render all types (saves to ./rendered_videos by default)
./op_tests/sagev1_tests/render_all_attention.sh

# Render specific types only
./op_tests/sagev1_tests/render_all_attention.sh --types sagev1,fav3_fp8,fav3_sage

# Custom output directory
./op_tests/sagev1_tests/render_all_attention.sh --output-dir ./my_videos
```

#### Step 3: Stitch Videos into Comparison Grid

```bash
# Basic usage
python op_tests/sagev1_tests/stitch_comparison.py \
    --input_dir ./op_tests/sagev1_tests/rendered_videos

# Custom output filename
python op_tests/sagev1_tests/stitch_comparison.py \
    --input_dir ./rendered_videos \
    --output my_comparison.mp4

# Larger labels for presentations
python op_tests/sagev1_tests/stitch_comparison.py \
    --input_dir ./rendered_videos \
    --font_size 48

# Custom FPS
python op_tests/sagev1_tests/stitch_comparison.py \
    --input_dir ./rendered_videos \
    --fps 24
```
## Saving Inputs for Benchmarking

The `sageattn_cogvideo.py` script can capture attention inputs (Q, K, V tensors) during video generation for later benchmarking with different attention kernels.

### Capture Inputs During Rendering

```bash
# Save first 10 attention inputs (default)
python op_tests/sagev1_tests/sageattn_cogvideo.py \
    --attention_type sagev1 \
    --save_inputs \
    --input_dir ./captured_inputs

# Save more inputs for comprehensive benchmarking
python op_tests/sagev1_tests/sageattn_cogvideo.py \
    --attention_type sagev1 \
    --save_inputs \
    --input_dir ./captured_inputs \
    --max_captures 100

# Save all inputs (use with caution - can be large)
python op_tests/sagev1_tests/sageattn_cogvideo.py \
    --attention_type sagev1 \
    --save_inputs \
    --input_dir ./captured_inputs \
    --max_captures 0
```

### Captured Input Format

Each captured input is saved as a `.pt` file containing:
- `q`, `k`, `v` - The input tensors (CPU, cloned)
- `q_shape`, `k_shape`, `v_shape` - Tensor shapes
- `dtype` - Data type string
- `call_idx` - Index of this attention call in the pipeline
- `kwargs` - Non-tensor keyword arguments (e.g., `is_causal`, `softmax_scale`)

A metadata file `{kernel_name}_metadata.pt` is also saved with:
- Total number of attention calls
- Number of inputs saved
- Unique shape configurations encountered

### Benchmark with Captured Inputs

Use the benchmark script to evaluate different attention kernels on the captured inputs:

```bash
# Benchmark sagev1 kernel
python op_tests/op_benchmarks/triton/bench_cogvideo.py \
    --input_dir ./captured_inputs \
    --kernel sagev1

# Benchmark FA3 FP8 kernel
python op_tests/op_benchmarks/triton/bench_cogvideo.py \
    --input_dir ./captured_inputs \
    --kernel fav3_fp8

# Compare throughput across kernels
python op_tests/op_benchmarks/triton/bench_cogvideo.py \
    --input_dir ./captured_inputs \
    --kernel sdpa \
    -metric throughput

# Available kernels: sagev1, fav3_sage, sdpa, fav2, fav3, fav3_fp8
```

## Script Reference

### `render_all_attention.sh`

```
Usage: render_all_attention.sh [OPTIONS]

Options:
  --output-dir DIR    Directory to save rendered videos (default: ./rendered_videos)
  --model-path PATH   Path to CogVideoX model (default: THUDM/CogVideoX-2b)
  --stitch            Automatically stitch videos after rendering
  --types TYPES       Comma-separated list of attention types to render
  -h, --help          Show help message
```

### `stitch_comparison.py`

```
Usage: stitch_comparison.py [OPTIONS]

Options:
  --input_dir DIR     Directory containing rendered videos (default: ./rendered_videos)
  --output FILE       Output video filename (default: comparison_grid.mp4)
  --pattern PATTERN   Glob pattern to match video files (default: cogvideox-*.mp4)
  --font_size SIZE    Font size for attention type labels (default: 36)
  --fps FPS           Output FPS (default: same as source)
```

### `sageattn_cogvideo.py`

```
Usage: sageattn_cogvideo.py [OPTIONS]

Options:
  --model_path PATH       Path to CogVideoX model (default: THUDM/CogVideoX-2b)
  --attention_type TYPE   Attention implementation to use
  --compile               Enable torch.compile (NVIDIA only)
  --save_inputs           Save attention inputs for benchmarking
  --input_dir DIR         Directory for captured inputs
  --max_captures N        Maximum inputs to save (0 = unlimited)
```
