#!/bin/bash
# Render WAN videos for all attention types and stitch into a comparison grid.
# Supported attention types: default (reuse only), sagev1, fav3_sage, fav3_fp8

set -o errexit
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STITCH_SCRIPT="${SCRIPT_DIR}/stitch_comparison.py"
RUN_WAN_SCRIPT="${SCRIPT_DIR}/run_wan.py"

# Default configuration
OUTPUT_DIR="${SCRIPT_DIR}/rendered_video_wan22"
REUSE_EXISTING=false
AUTO_STITCH=false
FONT_SIZE=36
FPS=""
NPROC=8

# Attention types to render (default is excluded - only reused if available)
RENDER_TYPES=("sagev1" "fav3_sage" "fav3_fp8")
# All types to include in stitching (if videos exist)
ALL_TYPES=("default" "sagev1" "fav3_sage" "fav3_fp8")

# WAN pipeline configuration
MODEL_PATH="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
IMG_FILE_PATH="/app/Wan/i2v_input.JPG"
TASK="i2v"
HEIGHT=720
WIDTH=1280
ULYSSES_DEGREE=8
SEED=42
NUM_FRAMES=81
NUM_REPETITIONS=1
NUM_INFERENCE_STEPS=40
PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

# Track results
declare -a SUCCEEDED=()
declare -a FAILED=()
declare -a REUSED=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat <<'EOF'
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

Example:
  ./render_all_WAN.sh --output-dir ./[SCRIPT_DIR]/rendered_video_wan22 --stitch
  ./render_all_WAN.sh --types sagev1,fav3_fp8 --nproc 8 --reuse
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stitch)
            AUTO_STITCH=true
            shift
            ;;
        --reuse)
            REUSE_EXISTING=true
            shift
            ;;
        --types)
            IFS=',' read -ra RENDER_TYPES <<< "$2"
            shift 2
            ;;
        --img-file-path)
            IMG_FILE_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --font-size)
            FONT_SIZE="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            print_usage
            exit 1
            ;;
    esac
done

# Resolve output directory to absolute path
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  WAN Attention Type Renderer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Output directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Model path: ${YELLOW}${MODEL_PATH}${NC}"
echo -e "Image path: ${YELLOW}${IMG_FILE_PATH}${NC}"
echo -e "Attention types to render: ${YELLOW}${RENDER_TYPES[*]}${NC}"
echo -e "Reuse existing: ${YELLOW}${REUSE_EXISTING}${NC}"
echo -e "Auto stitch: ${YELLOW}${AUTO_STITCH}${NC}"
echo -e "GPUs: ${YELLOW}${NPROC}${NC}"
echo ""

# Check if run_wan.py script exists
if [[ ! -f "$RUN_WAN_SCRIPT" ]]; then
    echo -e "${RED}Error: run_wan.py not found at ${RUN_WAN_SCRIPT}${NC}" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to find video for an attention type
find_video_for_type() {
    local attn_type="$1"
    local pattern="$OUTPUT_DIR/results/*_attn_${attn_type}_*.mp4"
    local video
    video=$(ls $pattern 2>/dev/null | head -1)
    if [[ -f "$video" ]]; then
        echo "$video"
    fi
}

# Render each attention type
for attn_type in "${RENDER_TYPES[@]}"; do
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Processing attention type: ${YELLOW}${attn_type}${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    
    # Check if video already exists
    existing_video=$(find_video_for_type "$attn_type")
    
    if [[ "$REUSE_EXISTING" == true ]] && [[ -n "$existing_video" ]]; then
        echo -e "${GREEN}✓ Reusing existing video: ${existing_video}${NC}"
        REUSED+=("$attn_type")
        continue
    fi
    
    echo -e "Rendering with attention type: ${YELLOW}${attn_type}${NC}"
    
    # Run the WAN pipeline
    if torchrun --nproc_per_node="$NPROC" "$RUN_WAN_SCRIPT" \
        --task "$TASK" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --model "$MODEL_PATH" \
        --img_file_path "$IMG_FILE_PATH" \
        --ulysses_degree "$ULYSSES_DEGREE" \
        --seed "$SEED" \
        --num_frames "$NUM_FRAMES" \
        --prompt "$PROMPT" \
        --num_repetitions "$NUM_REPETITIONS" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --use_torch_compile \
        --benchmark_output_directory "$OUTPUT_DIR" \
        --save_inputs \
        --attention_type "$attn_type"; then
        
        # Check if video was created
        new_video=$(find_video_for_type "$attn_type")
        if [[ -n "$new_video" ]]; then
            echo -e "${GREEN}✓ Successfully rendered: ${new_video}${NC}"
            SUCCEEDED+=("$attn_type")
        else
            echo -e "${YELLOW}⚠ Render completed but video not found${NC}"
            FAILED+=("$attn_type")
        fi
    else
        echo -e "${RED}✗ Failed to render with attention type: ${attn_type}${NC}"
        FAILED+=("$attn_type")
    fi
    echo ""
done

# Print rendering summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Rendering Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo -e "${GREEN}Rendered (${#SUCCEEDED[@]}):${NC}"
    for attn_type in "${SUCCEEDED[@]}"; do
        echo -e "  ${GREEN}✓${NC} $attn_type"
    done
    echo ""
fi

if [[ ${#REUSED[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Reused (${#REUSED[@]}):${NC}"
    for attn_type in "${REUSED[@]}"; do
        echo -e "  ${YELLOW}↻${NC} $attn_type"
    done
    echo ""
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}Failed (${#FAILED[@]}):${NC}"
    for attn_type in "${FAILED[@]}"; do
        echo -e "  ${RED}✗${NC} $attn_type"
    done
    echo ""
fi

# Collect all available videos for stitching (including default if exists)
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Checking available videos for stitching...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"

declare -a VIDEOS_FOR_STITCH=()
declare -a LABELS_FOR_STITCH=()

for attn_type in "${ALL_TYPES[@]}"; do
    video=$(find_video_for_type "$attn_type")
    if [[ -n "$video" ]]; then
        echo -e "${GREEN}✓${NC} Found: $attn_type -> $(basename "$video")"
        VIDEOS_FOR_STITCH+=("$video")
        LABELS_FOR_STITCH+=("$attn_type")
    else
        echo -e "${YELLOW}○${NC} Not found: $attn_type"
    fi
done
echo ""

# Stitch videos if requested and enough videos are available
if [[ "$AUTO_STITCH" == true ]]; then
    if [[ ${#VIDEOS_FOR_STITCH[@]} -lt 2 ]]; then
        echo -e "${YELLOW}⚠ Need at least 2 videos to create comparison grid. Skipping stitch.${NC}"
    else
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}Stitching ${#VIDEOS_FOR_STITCH[@]} videos into comparison grid...${NC}"
        echo -e "${BLUE}----------------------------------------${NC}"
        
        if [[ ! -f "$STITCH_SCRIPT" ]]; then
            echo -e "${RED}Error: Stitch script not found at ${STITCH_SCRIPT}${NC}" >&2
            exit 1
        fi
        
        # Build stitch command
        STITCH_CMD=("python" "$STITCH_SCRIPT" 
            "--input_dir" "$OUTPUT_DIR/results" 
            "--output" "$OUTPUT_DIR/comparison_grid.mp4" 
            "--pattern" "wan_result_*.mp4"
            "--font_size" "$FONT_SIZE")
        
        if [[ -n "$FPS" ]]; then
            STITCH_CMD+=("--fps" "$FPS")
        fi
        
        echo -e "${BLUE}Running:${NC} ${STITCH_CMD[*]}"
        
        if "${STITCH_CMD[@]}"; then
            echo -e "${GREEN}✓ Comparison video created: ${OUTPUT_DIR}/comparison_grid.mp4${NC}"
        else
            echo -e "${RED}✗ Failed to create comparison video${NC}"
        fi
    fi
fi

echo ""
echo -e "${BLUE}Done!${NC}"

# Exit with error if all renders failed
if [[ ${#SUCCEEDED[@]} -eq 0 ]] && [[ ${#REUSED[@]} -eq 0 ]] && [[ ${#RENDER_TYPES[@]} -gt 0 ]]; then
    exit 1
fi

exit 0
