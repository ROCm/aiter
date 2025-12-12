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
| `fa2` | Flash Attention v2 (Triton) |
| `fa3` | Flash Attention v3 (Triton) |
| `fa3_fp8` | Flash Attention v3 with FP8 quantization |
| `sagev1_fa3` | SageAttention v1 fused on FA3 backend |

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
    --attention_type fa3_fp8 \
    --model_path /path/to/local/model
```

#### Step 2: Render All Attention Types

```bash
# Render all types (saves to ./rendered_videos by default)
./op_tests/sagev1_tests/render_all_attention.sh

# Render specific types only
./op_tests/sagev1_tests/render_all_attention.sh --types sagev1,fa3_fp8,sagev1_fa3

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
