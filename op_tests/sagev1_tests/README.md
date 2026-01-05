# Video Generation Attention Comparison Tools

This directory contains tools for rendering videos with different attention implementations using CogVideoX and WAN models.

## Overview

### Common tools
- **`stitch_comparison.py`** - Stitches multiple videos into a grid layout with labels

### CogVideoX Tools
- **`sageattn_cogvideo.py`** - Renders a CogVideoX video using a specified attention implementation
- **`render_all_attention.sh`** - Batch renders videos for all attention types

### WAN Tools
- **`run_wan.py`** - Renders WAN 2.1/2.2 videos with configurable attention implementations (supports distributed inference via xfuser)
- **`render_all_WAN.sh`** - Batch renders WAN videos for all attention types and stitches into comparison grid
- **`wan_utils.py`** - Utility functions for WAN video generation

## Prerequisites

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

---

## WAN Video Generation

The `run_wan.py` script enables WAN 2.2 video generation with different attention implementations using distributed inference via xfuser.

### Available WAN Attention Types

| Type | Description |
|------|-------------|
| `default` | Internal xfuser attention (not working with source builds) |
| `sagev1` | SageAttention v1 (INT8 quantized Q/K) |
| `fav3_sage` | SageAttention v1 fused on FA3 backend |
| `fav3_fp8` | Flash Attention v3 with FP8 quantization |

### Batch Render All Attention Types

Use `render_all_WAN.sh` to render videos for all attention types and create a comparison grid:

```bash
# Render all attention types (sagev1, fav3_sage, fav3_fp8)
./op_tests/sagev1_tests/render_all_WAN.sh

# Render and stitch into comparison grid
./op_tests/sagev1_tests/render_all_WAN.sh --stitch

# Reuse existing videos (skip rendering if video exists)
./op_tests/sagev1_tests/render_all_WAN.sh --reuse --stitch

# Render specific types only
./op_tests/sagev1_tests/render_all_WAN.sh --types sagev1,fav3_fp8

# Custom output directory
./op_tests/sagev1_tests/render_all_WAN.sh --output-dir ./my_wan_videos --stitch
```

**Note:** The `default` attention type is NOT rendered by this script, but if a pre-existing video exists, it will be included in the stitched comparison.

### Basic WAN Usage

```bash
# Run from workspace root
cd /workspace/aiter

torchrun --nproc_per_node=8 op_tests/sagev1_tests/run_wan.py \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory results_Wan22 \
    --attention_type fav3_sage \
    --save_inputs \
    --max_captures 10 \
    2>&1 | tee results_Wan22/logs.txt
```

### `run_wan.py` Reference

```
Usage: torchrun --nproc_per_node=N run_wan.py [OPTIONS]

Required Options:
  --model PATH              Path or HuggingFace model ID for WAN model
  --prompt TEXT             Text prompt for video generation

Task Options:
  --task TYPE               Task type: i2v (image-to-video), t2v (text-to-video), ti2v
  --img_file_path PATH      Input image path (required for i2v task)
  --height H                Output video height (default: 480)
  --width W                 Output video width (default: 720)
  --num_frames N            Number of frames to generate (default: 49)
  --force_output_size       Force exact output dimensions by resizing/cropping input

Attention Options:
  --attention_type TYPE     Attention implementation: default, sagev1, fav3_sage, fav3_fp8
  --save_inputs             Save attention inputs for benchmarking
  --max_captures N          Maximum inputs to save (0 = unlimited, default: 10)

Distributed Options:
  --ulysses_degree N        Ulysses parallelism degree
  --ring_degree N           Ring parallelism degree

Performance Options:
  --use_torch_compile       Enable torch.compile for faster inference
  --use_bf16_te_gemms       Use bfloat16 for time embedding GEMMs
  --use_fp8_gemms           Use FP8 for linear layer GEMMs

Benchmark Options:
  --num_repetitions N       Number of benchmark repetitions (default: 5)
  --benchmark_output_directory DIR  Output directory for results and captures

Profiling Options:
  --profile_output NAME     Enable PyTorch profiler with given output name
  --profile_wait N          Profiler wait steps (default: 2)
  --profile_warmup N        Profiler warmup steps (default: 2)
  --profile_active N        Profiler active steps (default: 1)
```

### `render_all_WAN.sh` Reference

```
Usage: render_all_WAN.sh [OPTIONS]

Render WAN videos for all attention types and stitch into a comparison grid.
Attention types rendered: sagev1, fav3_sage, fav3_fp8
Note: 'default' is NOT rendered but will be included in stitching if pre-existing.

Options:
  --output-dir DIR      Output directory (default: ./rendered_video_wan22)
  --stitch              Automatically stitch videos after rendering
  --reuse               Reuse existing videos if available (skip rendering)
  --types TYPES         Comma-separated list of attention types to render
                        (default: sagev1,fav3_sage,fav3_fp8)
  --img-file-path PATH  Input image path (default: /app/Wan/i2v_input.JPG)
  --model-path PATH     Model path (default: Wan-AI/Wan2.2-I2V-A14B-Diffusers)
  --nproc N             Number of GPUs (default: 8)
  --font-size SIZE      Font size for labels in comparison video (default: 36)
  --fps FPS             Output FPS for comparison video (default: inherit)
  -h, --help            Show this help message
```

### Output Files

When running `run_wan.py`, the following files are created in `benchmark_output_directory`:

```
benchmark_output_directory/
├── results/
│   ├── wan_result_*.mp4           # Generated video
│   └── timing.json                # Benchmark timing data
├── traces/                        # (if --profile_output used)
│   └── {profile_output}/
│       └── wan_traces_rank*.json.gz
└── {attention_type}_input_*.pt    # (if --save_inputs used)
```

When running `render_all_WAN.sh`, the following structure is created:

```
rendered_video_wan22/
├── results/
│   ├── wan_result_i2v_*_attn_sagev1_720x1280.mp4
│   ├── wan_result_i2v_*_attn_fav3_sage_720x1280.mp4
│   ├── wan_result_i2v_*_attn_fav3_fp8_720x1280.mp4
│   └── wan_result_i2v_*_attn_default_720x1280.mp4  # Only if pre-existing
└── comparison_grid.mp4            # Created with --stitch flag
```
