#!/bin/bash
# Stitch a custom list of Wan benchmark videos into a single comparison grid.
# Edit the VIDEO_SOURCES (and optional VIDEO_LABELS) arrays below to decide
# which runs are included. The number of videos (N) is determined solely by the
# number of entries you add to VIDEO_SOURCES.

set -o errexit
set -o pipefail
set -o nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STITCH_SCRIPT="${SCRIPT_DIR}/stitch_comparison.py"
DEFAULT_OUTPUT="${SCRIPT_DIR}/comparison_grid.mp4"
DEFAULT_FONT_SIZE=36
DEFAULT_FPS=""
KEEP_TEMP=false

# --------------------------------------------------------------------------------------
# EDIT THESE ARRAYS TO MATCH YOUR WAN RUNS
# Each entry in VIDEO_SOURCES must point to an existing .mp4 file.
# Optionally provide a matching label in VIDEO_LABELS; otherwise the filename is used.
# --------------------------------------------------------------------------------------
# VIDEO_SOURCES=(
#     "results_Wan21_default/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
#     "results_Wan21_hybrid_fp8_attn/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
#     "results_Wan21_fp8_attn/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
# )

# VIDEO_LABELS=(
#     "Wan2.1_Default"
#     "Wan2.1_hybrid_fp8_attn"
#     "Wan2.1_fp8_attn"
# )

# VIDEO_SOURCES=(
#     "results_Wan22_default/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
#     "results_Wan22_hybrid_fp8_attn/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
#     "results_Wan22_fp8_attn/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
# )

# VIDEO_LABELS=(
#     "Wan2.2_default"
#     "Wan2.2_hybrid_fp8_attn"
#     "Wan2.2_fp8_attn"
# )

VIDEO_SOURCES=(
    "results_Wan22_default/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
    "results_Wan22_fp8_attn/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
    "results_Wan22_fp8_attn_v1/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
    "results_Wan22_fp8_attn_v2/results/wan_result_i2v_ulysses8_ringNone_True_720x1280.mp4"
)

VIDEO_LABELS=(
    "Wan2.2_default"
    "Wan2.2_fp8_attn"
    "Wan2.2_fp8_attn_v1"
    "Wan2.2_fp8_attn_v2"
)
# --------------------------------------------------------------------------------------

print_usage() {
    cat <<'EOF'
Usage: render_all_WAN.sh [OPTIONS]

Stitch a hand-picked set of Wan videos into a single comparison clip.
Edit the VIDEO_SOURCES array inside this script to control which files are used.

Options:
  --output PATH       Output video path (default: ./wan_comparison_grid.mp4)
  --font-size SIZE    Font size for labels (default: 36)
  --fps FPS           Output frames per second (default: inherit first clip)
  --keep-temp         Keep the staging directory for debugging
  -h, --help          Show this help message

Example:
  ./render_all_WAN.sh --output ~/videos/wan_compare.mp4 --font-size 48
EOF
}

OUTPUT_VIDEO="$DEFAULT_OUTPUT"
FONT_SIZE=$DEFAULT_FONT_SIZE
FPS=$DEFAULT_FPS

resolve_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        return 1
    fi
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$(pwd)" "$path"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_VIDEO="$2"
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
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

OUTPUT_VIDEO="$(resolve_path "$OUTPUT_VIDEO")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [[ ! -f "$STITCH_SCRIPT" ]]; then
    echo -e "${RED}Error:${NC} stitch script not found at $STITCH_SCRIPT" >&2
    exit 1
fi

if [[ ${#VIDEO_SOURCES[@]} -lt 2 ]]; then
    echo -e "${RED}Error:${NC} at least two video paths are required. Edit VIDEO_SOURCES." >&2
    exit 1
fi

if [[ ${#VIDEO_LABELS[@]} -gt 0 ]] && [[ ${#VIDEO_LABELS[@]} -ne ${#VIDEO_SOURCES[@]} ]]; then
    echo -e "${RED}Error:${NC} VIDEO_LABELS count must match VIDEO_SOURCES (or be empty)." >&2
    exit 1
fi
for idx in "${!VIDEO_SOURCES[@]}"; do
    video="${VIDEO_SOURCES[$idx]}"
    if [[ "$video" == REPLACE_WITH_PATH_* ]]; then
        echo -e "${RED}Error:${NC} Replace placeholder paths in VIDEO_SOURCES with real files." >&2
        exit 1
    fi
    if [[ ! -f "$video" ]]; then
        echo -e "${RED}Error:${NC} Video not found: $video" >&2
        exit 1
    fi
    if [[ ! -r "$video" ]]; then
        echo -e "${RED}Error:${NC} Cannot read $video" >&2
        exit 1
    fi
    if [[ ! -s "$video" ]]; then
        echo -e "${RED}Error:${NC} $video is empty." >&2
        exit 1
    fi
    if [[ "${video##*.}" != "mp4" ]]; then
        echo -e "${YELLOW}Warning:${NC} $video does not use the .mp4 extension." >&2
    fi

    # Store absolute paths so symlinks stay valid outside the workspace root
    VIDEO_SOURCES[$idx]="$(realpath "$video")"
done

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wan_stitch.XXXXXX")"
cleanup() {
    if [[ "$KEEP_TEMP" == false ]]; then
        rm -rf "$STAGING_DIR"
    else
        echo -e "${YELLOW}Keeping staging dir:${NC} $STAGING_DIR"
    fi
}
trap cleanup EXIT

sanitize_label() {
    local label="$1"
    label="${label// /_}"
    label="${label//[^A-Za-z0-9_\-]/}"
    if [[ -z "$label" ]]; then
        label="wan_clip"
    fi
    echo "$label"
}

printf -v PADDED_COUNT "%02d" ${#VIDEO_SOURCES[@]}
echo -e "${BLUE}Preparing ${PADDED_COUNT} Wan videos for stitching...${NC}"

for idx in "${!VIDEO_SOURCES[@]}"; do
    src="${VIDEO_SOURCES[$idx]}"
    label="${VIDEO_LABELS[$idx]:-$(basename "${src%.*}")}"
    safe_label="$(sanitize_label "$label")"
    printf -v name "cogvideox-wan_%02d_%s.mp4" "$((idx + 1))" "$safe_label"
    ln -sf "$src" "${STAGING_DIR}/${name}"
    echo -e "${GREEN}✓${NC} Added ${label} -> ${name}"
done

mkdir -p "$(dirname "$OUTPUT_VIDEO")"

CMD=("python" "$STITCH_SCRIPT" "--input_dir" "$STAGING_DIR" "--output" "$OUTPUT_VIDEO" "--pattern" "*.mp4" "--font_size" "$FONT_SIZE")
if [[ -n "$FPS" ]]; then
    CMD+=("--fps" "$FPS")
fi

echo -e "${BLUE}Running:${NC} ${CMD[*]}"
"${CMD[@]}"

echo -e "${GREEN}Done:${NC} ${OUTPUT_VIDEO}"
