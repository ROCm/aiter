#!/bin/bash
# Usage (inside gcy_atom): bash dump_router.sh [mode] [model]
# Applies $DUMP_PATCH (default atom_router_dump.patch; atom_hash_compare.patch
# records hash-layer routing on the Triton decode path, run it with mode
# triton_herd) to ATOM moe.py on top of atom_flydsl_herd.patch, serves
# eagerly, drives 64 concurrent ISL=1024 /
# OSL=$DUMP_OSL requests so decode runs at bs=64, dumps each routed layer's
# router logits every ATOM_DUMP_ROUTER_EVERY decode steps, up to
# ATOM_DUMP_ROUTER_STEPS files per layer, then restores moe.py from a backup.
# OUT is wiped first, so do not point it at a dump you want to keep.
set -uo pipefail
MODE="${1:-flydsl_herd}"
MODEL_PATH="${2:-/data/DeepSeek-V4-Flash}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/router_dump}"
ATOM_DIR=/app/ATOM
MOE="$ATOM_DIR/atom/model_ops/moe.py"
BACKUP="$(mktemp)"
rm -rf "$OUT" && mkdir -p "$OUT"

cp "$MOE" "$BACKUP"
restore() {
  cp "$BACKUP" "$MOE"
  cmp -s "$BACKUP" "$MOE" && echo "[dump] moe.py restored" && rm -f "$BACKUP"
}
PATCH="${DUMP_PATCH:-$HERE/atom_router_dump.patch}"
git -C "$ATOM_DIR" apply "$PATCH" || {
  echo "[dump] $PATCH does not apply (is atom_flydsl_herd.patch applied?)"
  exit 1
}
trap restore EXIT

cd /app/scripts/dsv4
ATOM_DUMP_ROUTER_DIR="$OUT" SERVE_EXTRA_ARGS="--enforce-eager" MOE_MODE="$MODE" \
  setsid bash "$HERE/serve_dsv4_tp1_herd.sh" "$MODEL_PATH" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
stop_server() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null
  restore
}
trap stop_server EXIT

until curl -sf http://localhost:8000/health > /dev/null; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[dump] server died"; exit 1; }
  sleep 10
done
echo "[dump] $(date) server healthy"
python -m atom.benchmarks.benchmark_serving --model="$MODEL_PATH/" --backend=vllm \
  --base-url=http://localhost:8000 --dataset-name=random --random-input-len=1024 \
  --random-output-len="${DUMP_OSL:-64}" --random-range-ratio 1 --num-prompts=64 --max-concurrency=64 \
  --request-rate=inf --ignore-eos --num-warmups 0 > "$OUT/load.log" 2>&1
echo "[dump] $(date) load done: $(ls "$OUT"/*.pt 2>/dev/null | wc -l) files"
