#!/bin/bash
# Run operator performance benchmarks and output CSV results.
# Called by the perf-bench CI workflow.
#
# Usage:
#   op_perf_bench.sh [bench_filter] [output_dir]
#
# Examples:
#   op_perf_bench.sh ""                    # Run all benchmarks
#   op_perf_bench.sh "gemm"               # Only GEMM benchmarks
#   op_perf_bench.sh "attention,moe"      # Attention and MoE benchmarks
#   op_perf_bench.sh "auto"               # Auto-detect from changed files

set -euo pipefail

BENCH_FILTER="${1:-}"
OUTPUT_DIR="${2:-perf_results}"
BENCH_DIR="op_tests/op_benchmarks/triton"
MODEL_CONFIGS="op_tests/op_benchmarks/triton/utils/model_configs.json"

mkdir -p "$OUTPUT_DIR"

# Collect system info (supports both ROCm gcnArchName and CUDA sm_XX)
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
GPU_ARCH=$(python3 - <<'PY' 2>/dev/null || echo "unknown")
import torch
try:
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    if arch:
        print(str(arch).split(":", 1)[0])
    else:
        major, minor = torch.cuda.get_device_capability(0)
        print(f"sm_{major}{minor}")
except Exception:
    print("unknown")
PY
COMMIT_SHA="${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
COMMIT_DATE=$(git show -s --format=%ci HEAD 2>/dev/null || date -Iseconds)
BRANCH="${GITHUB_HEAD_REF:-$(git branch --show-current 2>/dev/null || echo unknown)}"

echo "=== Operator Performance Benchmark ==="
echo "GPU: $GPU_NAME ($GPU_ARCH)"
echo "Commit: $COMMIT_SHA"
echo "Branch: $BRANCH"
echo "Filter: ${BENCH_FILTER:-all}"
echo "Output: $OUTPUT_DIR"
echo ""

# Write metadata
cat > "$OUTPUT_DIR/metadata.json" <<METADATA
{
  "gpu": "$GPU_NAME",
  "gpu_arch": "$GPU_ARCH",
  "commit": "$COMMIT_SHA",
  "commit_date": "$COMMIT_DATE",
  "branch": "$BRANCH",
  "timestamp": "$(date -Iseconds)"
}
METADATA

# ── Normalize filter: handle "all", aliases ──
normalize_filter() {
    local f="$1"
    f=$(echo "$f" | tr '[:upper:]' '[:lower:]')
    # "all" or empty = run everything
    if [ -z "$f" ] || [ "$f" = "all" ]; then echo ""; return; fi
    # Alias: mha -> attention
    f=$(echo "$f" | sed 's/\bmha\b/attention/g')
    echo "$f"
}

BENCH_FILTER=$(normalize_filter "$BENCH_FILTER")

# ── Auto-detect changed operators from PR diff ──
if [ "$BENCH_FILTER" = "auto" ]; then
    BENCH_FILTER=""
    CHANGED=$(git diff --name-only HEAD~1 2>/dev/null || echo "")

    if echo "$CHANGED" | grep -qiE "gemm|a8w8|a16w16|afp4|a4w4"; then
        BENCH_FILTER="${BENCH_FILTER:+$BENCH_FILTER,}gemm"
    fi
    if echo "$CHANGED" | grep -qiE "mha|attention|flash|mla|pa_"; then
        BENCH_FILTER="${BENCH_FILTER:+$BENCH_FILTER,}attention"
    fi
    if echo "$CHANGED" | grep -qiE "moe|fused_moe|expert"; then
        BENCH_FILTER="${BENCH_FILTER:+$BENCH_FILTER,}moe"
    fi
    if echo "$CHANGED" | grep -qiE "rmsnorm|norm"; then
        BENCH_FILTER="${BENCH_FILTER:+$BENCH_FILTER,}rmsnorm"
    fi
    if echo "$CHANGED" | grep -qiE "rope|positional"; then
        BENCH_FILTER="${BENCH_FILTER:+$BENCH_FILTER,}rope"
    fi

    if [ -z "$BENCH_FILTER" ]; then
        echo "No kernel changes detected, running core benchmarks only."
        BENCH_FILTER="gemm,attention"
    fi
    echo "Auto-detected benchmarks: $BENCH_FILTER"
fi

# ── Define benchmark jobs ──
# Format: name:script:args
# Each benchmark outputs CSV via -o flag
declare -a BENCH_JOBS=()

should_run() {
    local name="$1"
    # Empty filter = run all
    if [ -z "$BENCH_FILTER" ]; then return 0; fi
    echo ",$BENCH_FILTER," | grep -qi ",$name," && return 0
    return 1
}

# GEMM benchmarks — model-based shapes (most relevant)
if should_run "gemm"; then
    BENCH_JOBS+=(
        "gemm_a8w8_m1:bench_gemm_a8w8.py:--model deepseek -M 1 -tp 1 --metric throughput -o"
        "gemm_a8w8_m256:bench_gemm_a8w8.py:--model deepseek -M 256 -tp 1 --metric throughput -o"
        "gemm_a8w8_m4096:bench_gemm_a8w8.py:--model deepseek -M 4096 -tp 1 --metric throughput -o"
        "gemm_a16w16_m256:bench_gemm_a16w16.py:--model deepseek -M 256 -tp 1 --metric throughput -o"
        "gemm_a8w8_bs:bench_gemm_a8w8_blockscale.py:--shape 256 7168 18432 --metric throughput -o"
        "gemm_afp4:bench_gemm_afp4wfp4.py:--shape 256 7168 18432 --metric throughput -o"
    )
fi

# Attention benchmarks — use custom shapes to avoid model config version issues
if should_run "attention"; then
    BENCH_JOBS+=(
        "mha_ds_s1k:bench_mha.py:-b 1 -hq 128 -hk 128 -sq 1024 -d 128 -dv 128 -mode fwd -causal true --metric throughput -o"
        "mha_ds_s4k:bench_mha.py:-b 1 -hq 128 -hk 128 -sq 4096 -d 128 -dv 128 -mode fwd -causal true --metric throughput -o"
        "mha_llama70_s4k:bench_mha.py:-b 1 -hq 64 -hk 8 -sq 4096 -d 128 -dv 128 -mode fwd -causal true --metric throughput -o"
    )
fi

# MoE benchmarks
if should_run "moe"; then
    BENCH_JOBS+=(
        "moe_fp16:bench_moe.py:--model deepseek -M 256 -o"
    )
fi

# RMSNorm
if should_run "rmsnorm"; then
    BENCH_JOBS+=(
        "rmsnorm:bench_rmsnorm.py:--model deepseek --metric bandwidth -o"
    )
fi

# ── Run benchmarks ──
FAILED=()
PASSED=()

for job in "${BENCH_JOBS[@]}"; do
    IFS=':' read -r name script args <<< "$job"
    echo ""
    echo "────────────────────────────────────────────────"
    echo "Running: $name"
    echo "  Script: $BENCH_DIR/$script"
    echo "  Args: $args"
    echo "────────────────────────────────────────────────"

    # Clean up any stale CSVs before running
    rm -f "$BENCH_DIR"/*.csv 2>/dev/null

    # Run benchmark, capture output
    if (cd "$BENCH_DIR" && python3 "$script" $args) 2>&1 | tee "$OUTPUT_DIR/${name}.log"; then
        echo "✅ $name PASSED"
        PASSED+=("$name")

        # Find generated CSV (bench scripts output to current dir with -o)
        for csv in "$BENCH_DIR"/*.csv; do
            if [ -f "$csv" ]; then
                base=$(basename "$csv")
                cp "$csv" "$OUTPUT_DIR/${name}_${base}"
                echo "  → Saved: $OUTPUT_DIR/${name}_${base}"
                rm -f "$csv"
            fi
        done
    else
        echo "❌ $name FAILED (non-fatal, continuing)"
        FAILED+=("$name")
    fi
done

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════"
echo "BENCHMARK SUMMARY"
echo "═══════════════════════════════════════════════"
echo "Passed: ${#PASSED[@]} (${PASSED[*]:-none})"
echo "Failed: ${#FAILED[@]} (${FAILED[*]:-none})"
echo "Results: $OUTPUT_DIR/"
echo ""

# List all output files
ls -la "$OUTPUT_DIR/"

# Generate combined summary CSV
python3 -c "
import csv, json, os, glob, sys

output_dir = '$OUTPUT_DIR'
meta_path = os.path.join(output_dir, 'metadata.json')

with open(meta_path) as f:
    meta = json.load(f)

# Collect all CSV results
all_rows = []
for csv_file in sorted(glob.glob(os.path.join(output_dir, '*.csv'))):
    bench_name = os.path.basename(csv_file).split('_')[0]
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['_benchmark'] = bench_name
            row['_gpu'] = meta['gpu']
            row['_commit'] = meta['commit']
            row['_branch'] = meta['branch']
            row['_timestamp'] = meta['timestamp']
            all_rows.append(row)

if all_rows:
    # Write combined CSV
    combined = os.path.join(output_dir, 'combined_results.csv')
    all_keys = []
    for row in all_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(combined, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Combined {len(all_rows)} results into {combined}')
else:
    print('No CSV results to combine.')
" 2>/dev/null || echo "Warning: could not generate combined CSV"

# Always exit 0 — failures are tracked in summary and artifacts, not exit code
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} benchmark(s) failed. See logs and artifacts for details."
fi

exit 0
