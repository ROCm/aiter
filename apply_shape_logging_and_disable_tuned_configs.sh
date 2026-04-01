#!/usr/bin/env bash
# =============================================================================
# Script: apply_shape_logging_and_disable_tuned_configs.sh
#
# TWO GOALS:
#   1) Add "always-on" shape logging to fused_moe.py and all gemm files so
#      that every shape seen at runtime is logged (not only when a default/
#      fallback config is selected). We reuse the same logger.info style,
#      but with a message prefix "[shape_collected]" so it is easy to grep.
#
#   2) Truncate every *tuned* CSV config file to header-only (keeping the
#      header row so pandas can still read them without crashing) so that
#      no tuned config is ever found. This forces every path through the
#      default/heuristic code.
#
# Run from the aiter repo root:
#   cd /workspaces/WS/aiter && bash apply_shape_logging_and_disable_tuned_configs.sh
# =============================================================================

set -euo pipefail

AITER_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "==> AITER_ROOT = $AITER_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# PART 1:  Add always-on shape logging
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=== PART 1: Adding [shape_collected] logger.info lines ==="

# -------------- aiter/fused_moe.py  (get_2stage_cfgs) ----------------------
# The function already logs at line ~916 whether it is using a tuned or default
# config.  We add an unconditional shape log RIGHT BEFORE the tuned-config
# lookup (just after `keys = (...)` is built).
#
# We anchor on the unique line:
#   cfg = cfg_2stages.get(keys, None) if cfg_2stages and use_cfg() else None
#   ... (first occurrence, the one NOT after mp_lock)
#
# We insert a logger.info BEFORE that line.

FILE="$AITER_ROOT/aiter/fused_moe.py"
echo "  -> $FILE"

# Use python to do a precise, safe insertion.
python3 - "$FILE" <<'PYEOF'
import sys, re

fpath = sys.argv[1]
with open(fpath, "r") as f:
    lines = f.readlines()

# Find the FIRST occurrence of the cfg_2stages.get(keys, ...) line that is
# NOT inside the mp_lock block.  We look for:
#   cfg = cfg_2stages.get(keys, None) if cfg_2stages and use_cfg() else None
# The first occurrence is the one before the online-tune block.
target = "cfg = cfg_2stages.get(keys, None) if cfg_2stages and use_cfg() else None"
found = False
for i, line in enumerate(lines):
    if target in line and not found:
        indent = line[: len(line) - len(line.lstrip())]
        insert = (
            f'{indent}logger.info(\n'
            f'{indent}    f"[shape_collected][fused_moe] token={{token}} model_dim={{model_dim}} inter_dim={{inter_dim}} expert={{expert}} topk={{topk}} '
            f'dtype={{dtype}} q_dtype_a={{q_dtype_a}} q_dtype_w={{q_dtype_w}} q_type={{q_type}} use_g1u1={{use_g1u1}} '
            f'activation={{activation}} doweight_stage1={{doweight_stage1}} hidden_pad={{hidden_pad}} intermediate_pad={{intermediate_pad}}"\n'
            f'{indent})\n'
        )
        lines.insert(i, insert)
        found = True
        break

if not found:
    print(f"  WARNING: could not find target line in {fpath}", file=sys.stderr)
    sys.exit(1)

with open(fpath, "w") as f:
    f.writelines(lines)
print(f"  inserted [shape_collected] log in get_2stage_cfgs()")
PYEOF


# -------------- aiter/tuned_gemm.py  (get_GEMM_A16W16_config) ---------------
# Currently logs only when a tuned config IS found (behind AITER_LOG_TUNED_CONFIG)
# or when it is NOT found.  We add an unconditional log at function entry.
FILE="$AITER_ROOT/aiter/tuned_gemm.py"
echo "  -> $FILE"

python3 - "$FILE" <<'PYEOF'
import sys

fpath = sys.argv[1]
with open(fpath, "r") as f:
    lines = f.readlines()

# Find the line: def get_GEMM_A16W16_config(
# Then find the first line after the docstring / first executable line inside
# (the line "cfg = get_GEMM_A16W16_config_()")
target = "cfg = get_GEMM_A16W16_config_()"
found = False
for i, line in enumerate(lines):
    if target in line and not found:
        indent = line[: len(line) - len(line.lstrip())]
        insert = (
            f'{indent}logger.info(\n'
            f'{indent}    f"[shape_collected][bf16_gemm] M:{{M}} N:{{N}} K:{{K}} {{dtype=}} {{otype=}} {{bias=}} {{scaleAB=}} {{bpreshuffle=}}"\n'
            f'{indent})\n'
        )
        lines.insert(i, insert)
        found = True
        break

if not found:
    print(f"  WARNING: could not find target in {fpath}", file=sys.stderr)
    sys.exit(1)

with open(fpath, "w") as f:
    f.writelines(lines)
print(f"  inserted [shape_collected] log in get_GEMM_A16W16_config()")
PYEOF


# -------------- aiter/ops/gemm_op_a8w8.py ----------------------------------
# Two config-lookup functions: get_CKGEMM_config and get_GEMM_config_with_quant_type
# Add unconditional shape log at their entry (before the cache-load logic runs,
# but the shape params are available).
FILE="$AITER_ROOT/aiter/ops/gemm_op_a8w8.py"
echo "  -> $FILE"

python3 - "$FILE" <<'PYEOF'
import sys

fpath = sys.argv[1]
with open(fpath, "r") as f:
    content = f.read()

# 1) get_CKGEMM_config  -- insert after "def get_CKGEMM_config(..." line
# We anchor on the unique line right after the def:
#   if tuned_file is None:
anchor1 = '    if tuned_file is None:\n        tuned_file = "a8w8_tuned_gemm.csv"'
insert1 = (
    '    logger.info(\n'
    '        f"[shape_collected][a8w8_gemm] M:{M} N:{N} K:{K} tuned_file={tuned_file}"\n'
    '    )\n'
)
if anchor1 in content:
    content = content.replace(anchor1, insert1 + anchor1, 1)
    print("  inserted [shape_collected] log in get_CKGEMM_config()")
else:
    print("  WARNING: could not find get_CKGEMM_config anchor", file=sys.stderr)

# 2) get_GEMM_config_with_quant_type -- insert after def line
# Anchor on the unique line:
#   if not hasattr(get_GEMM_config_with_quant_type, "file_cache"):
anchor2 = '    if not hasattr(get_GEMM_config_with_quant_type, "file_cache"):'
insert2 = (
    '    logger.info(\n'
    '        f"[shape_collected][a8w8_gemm_qt] M:{M} N:{N} K:{K} q_dtype_w:{q_dtype_w} tuned_file={tuned_file}"\n'
    '    )\n'
)
if anchor2 in content:
    content = content.replace(anchor2, insert2 + anchor2, 1)
    print("  inserted [shape_collected] log in get_GEMM_config_with_quant_type()")
else:
    print("  WARNING: could not find get_GEMM_config_with_quant_type anchor", file=sys.stderr)

with open(fpath, "w") as f:
    f.write(content)
PYEOF


# -------------- aiter/ops/gemm_op_a4w4.py ----------------------------------
FILE="$AITER_ROOT/aiter/ops/gemm_op_a4w4.py"
echo "  -> $FILE"

python3 - "$FILE" <<'PYEOF'
import sys

fpath = sys.argv[1]
with open(fpath, "r") as f:
    content = f.read()

# Anchor: right inside get_GEMM_config, before the cache-load check
anchor = '    if not hasattr(get_GEMM_config, "gemm_dict"):'
insert = (
    '    logger.info(\n'
    '        f"[shape_collected][a4w4_gemm] M:{M} N:{N} K:{K}"\n'
    '    )\n'
)
if anchor in content:
    content = content.replace(anchor, insert + anchor, 1)
    print("  inserted [shape_collected] log in get_GEMM_config() [a4w4]")
else:
    print("  WARNING: could not find anchor in a4w4", file=sys.stderr)

with open(fpath, "w") as f:
    f.write(content)
PYEOF


# -------------- aiter/ops/batched_gemm_op_a8w8.py --------------------------
FILE="$AITER_ROOT/aiter/ops/batched_gemm_op_a8w8.py"
echo "  -> $FILE"

python3 - "$FILE" <<'PYEOF'
import sys

fpath = sys.argv[1]
with open(fpath, "r") as f:
    content = f.read()

anchor = '    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):\n        print(\n            "Loading CKBatchedGEMM config from:",'
insert = (
    '    logger.info(\n'
    '        f"[shape_collected][a8w8_batched_gemm] B:{B} M:{M} N:{N} K:{K}"\n'
    '    )\n'
)
if anchor in content:
    content = content.replace(anchor, insert + anchor, 1)
    print("  inserted [shape_collected] log in get_CKBatchedGEMM_config() [a8w8_batched]")
else:
    print("  WARNING: could not find anchor in batched_gemm_op_a8w8.py", file=sys.stderr)

with open(fpath, "w") as f:
    f.write(content)
PYEOF


# -------------- aiter/ops/batched_gemm_op_bf16.py --------------------------
FILE="$AITER_ROOT/aiter/ops/batched_gemm_op_bf16.py"
echo "  -> $FILE"

python3 - "$FILE" <<'PYEOF'
import sys

fpath = sys.argv[1]
with open(fpath, "r") as f:
    content = f.read()

# This file has its own get_CKBatchedGEMM_config function for bf16
anchor = '    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):\n        ck_batched_gemm_dict = pd.read_csv('
insert = (
    '    logger.info(\n'
    '        f"[shape_collected][bf16_batched_gemm] B:{B} M:{M} N:{N} K:{K}"\n'
    '    )\n'
)
if anchor in content:
    content = content.replace(anchor, insert + anchor, 1)
    print("  inserted [shape_collected] log in get_CKBatchedGEMM_config() [bf16_batched]")
else:
    print("  WARNING: could not find anchor in batched_gemm_op_bf16.py", file=sys.stderr)

with open(fpath, "w") as f:
    f.write(content)
PYEOF


# ─────────────────────────────────────────────────────────────────────────────
# PART 2:  Truncate all tuned CSV files to header-only
#          (This ensures pandas reads them as empty DataFrames, so no tuned
#           config will ever be found — all paths fall through to defaults.)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=== PART 2: Truncating tuned CSV files to header-only ==="

CONFIGS_DIR="$AITER_ROOT/aiter/configs"

# List of tuned CSV files (NOT untuned, NOT profile)
TUNED_CSVS=(
    "$CONFIGS_DIR/tuned_fmoe.csv"
    "$CONFIGS_DIR/a8w8_tuned_gemm.csv"
    "$CONFIGS_DIR/a8w8_bpreshuffle_tuned_gemm.csv"
    "$CONFIGS_DIR/a8w8_blockscale_tuned_gemm.csv"
    "$CONFIGS_DIR/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    "$CONFIGS_DIR/a8w8_tuned_batched_gemm.csv"
    "$CONFIGS_DIR/a4w4_blockscale_tuned_gemm.csv"
    "$CONFIGS_DIR/bf16_tuned_gemm.csv"
    "$CONFIGS_DIR/bf16_tuned_batched_gemm.csv"
)

for csv in "${TUNED_CSVS[@]}"; do
    if [[ -f "$csv" ]]; then
        header=$(head -1 "$csv")
        echo "$header" > "$csv"
        echo "  truncated: $csv"
    else
        echo "  skipped (not found): $csv"
    fi
done

# Also truncate any tuned CSVs under model_configs/
echo ""
echo "  Truncating model_configs/ tuned CSVs..."
find "$CONFIGS_DIR/model_configs/" -name "*tuned*" -not -name "*untuned*" -name "*.csv" -type f 2>/dev/null | while read -r csv; do
    header=$(head -1 "$csv")
    echo "$header" > "$csv"
    echo "  truncated: $csv"
done

echo ""
echo "=== Done ==="
echo ""
echo "To collect shapes at runtime, run your workload and then grep the logs:"
echo "  grep '\\[shape_collected\\]' your_log_file.log"
echo ""
echo "To restore tuned configs, use: git checkout -- aiter/configs/"
