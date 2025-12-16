#!/bin/bash
# Render CogVideoX videos for all attention types
# Supported types: sdpa, sagev1, fav2, fav3, fav3_fp8, fav3_sage

set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COGVIDEO_SCRIPT="${SCRIPT_DIR}/sageattn_cogvideo.py"
OUTPUT_DIR="${SCRIPT_DIR}/rendered_videos"
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
AUTO_STITCH=false

# All attention types to try
ATTENTION_TYPES=("sdpa" "sagev1" "fav2" "fav3" "fav3_fp8" "fav3_sage")

# Track results
declare -a SUCCEEDED=()
declare -a FAILED=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Render CogVideoX videos for all available attention types."
    echo ""
    echo "Options:"
    echo "  --output-dir DIR    Directory to save rendered videos (default: ./rendered_videos)"
    echo "  --model-path PATH   Path to CogVideoX model (default: THUDM/CogVideoX-2b)"
    echo "  --stitch            Automatically stitch videos after rendering"
    echo "  --types TYPES       Comma-separated list of attention types to render"
    echo "                      (default: sdpa,sagev1,fav2,fav3,fav3_fp8,fav3_sage)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --output-dir ./my_videos --stitch"
    echo "  $0 --types sagev1,fav3_fp8,fav3_sage"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --stitch)
            AUTO_STITCH=true
            shift
            ;;
        --types)
            IFS=',' read -ra ATTENTION_TYPES <<< "$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CogVideoX Attention Type Renderer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Output directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Model path: ${YELLOW}${MODEL_PATH}${NC}"
echo -e "Attention types: ${YELLOW}${ATTENTION_TYPES[*]}${NC}"
echo ""

# Check if cogvideo script exists
if [[ ! -f "$COGVIDEO_SCRIPT" ]]; then
    echo -e "${RED}Error: CogVideo script not found at ${COGVIDEO_SCRIPT}${NC}"
    exit 1
fi

# Render each attention type
for attn_type in "${ATTENTION_TYPES[@]}"; do
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Rendering with attention type: ${YELLOW}${attn_type}${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    
    OUTPUT_FILE="${OUTPUT_DIR}/cogvideox-2b_${attn_type}.mp4"
    
    # Run the cogvideo script
    # We cd to the script directory so relative imports work correctly
    if (cd "$SCRIPT_DIR" && python sageattn_cogvideo.py \
        --model_path "$MODEL_PATH" \
        --attention_type "$attn_type" 2>&1); then
        
        # Move the output file to our output directory if it was created in a different location
        DEFAULT_OUTPUT="cogvideox-2b_${attn_type}.mp4"
        if [[ -f "${SCRIPT_DIR}/${DEFAULT_OUTPUT}" ]] && [[ "$SCRIPT_DIR" != "$OUTPUT_DIR" ]]; then
            mv "${SCRIPT_DIR}/${DEFAULT_OUTPUT}" "$OUTPUT_FILE"
        elif [[ -f "./${DEFAULT_OUTPUT}" ]] && [[ "$(pwd)" != "$OUTPUT_DIR" ]]; then
            mv "./${DEFAULT_OUTPUT}" "$OUTPUT_FILE"
        fi
        
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo -e "${GREEN}✓ Successfully rendered: ${OUTPUT_FILE}${NC}"
            SUCCEEDED+=("$attn_type")
        else
            echo -e "${YELLOW}⚠ Script completed but output file not found${NC}"
            FAILED+=("$attn_type")
        fi
    else
        echo -e "${RED}✗ Failed to render with attention type: ${attn_type}${NC}"
        FAILED+=("$attn_type")
    fi
    echo ""
done

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Rendering Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo -e "${GREEN}Succeeded (${#SUCCEEDED[@]}):${NC}"
    for attn_type in "${SUCCEEDED[@]}"; do
        echo -e "  ${GREEN}✓${NC} $attn_type"
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

# Auto-stitch if requested and we have successful renders
if [[ "$AUTO_STITCH" == true ]]; then
    if [[ ${#SUCCEEDED[@]} -lt 2 ]]; then
        echo -e "${YELLOW}⚠ Need at least 2 successful renders to stitch. Skipping.${NC}"
    else
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}Stitching videos into comparison grid...${NC}"
        echo -e "${BLUE}----------------------------------------${NC}"
        
        STITCH_SCRIPT="${SCRIPT_DIR}/stitch_comparison.py"
        if [[ -f "$STITCH_SCRIPT" ]]; then
            python "$STITCH_SCRIPT" \
                --input_dir "$OUTPUT_DIR" \
                --output "${OUTPUT_DIR}/comparison_grid.mp4"
            
            if [[ $? -eq 0 ]]; then
                echo -e "${GREEN}✓ Comparison video created: ${OUTPUT_DIR}/comparison_grid.mp4${NC}"
            else
                echo -e "${RED}✗ Failed to create comparison video${NC}"
            fi
        else
            echo -e "${RED}Error: Stitch script not found at ${STITCH_SCRIPT}${NC}"
            echo -e "${YELLOW}Run: python ${STITCH_SCRIPT} --input_dir ${OUTPUT_DIR} --output comparison_grid.mp4${NC}"
        fi
    fi
fi

echo ""
echo -e "${BLUE}Done!${NC}"

# Exit with error if all renders failed
if [[ ${#SUCCEEDED[@]} -eq 0 ]]; then
    exit 1
fi

exit 0

