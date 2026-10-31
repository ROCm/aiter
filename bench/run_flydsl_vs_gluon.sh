#!/bin/bash
# One click, HERD mode: DeepSeek-V4-Flash TP1 e2e, FlyDSL+HERD vs Triton(gluon)+HERD, plus a FlyDSL no-HERD baseline.
#
#   1 back up the python env (aiter editable install, flydsl, pip freeze), unless a backup exists
#   2 check that aiter#6527 is in AITER_DIR
#   3 build the ATOM copy carrying ATOM#2713 (patch_atom_herd.sh); /app/ATOM is not modified
#   4 pip install AITER_DIR (flydsl follows its pin, system triton kept)
#   5 run each config with run_config.sh: server -> cold run1 -> prewarm -> run2..run4 -> stop
#     FlyDSL+HERD aborts unless the server log shows "FlyDSL HERD route"
#   6 check the HERD route kernel was built
#   7 write the report; the python env is restored on exit (also on failure)
#
# Usage: bash run_flydsl_vs_gluon.sh [--tag TAG] [--out FILE] [--no-baseline] [--unit-test] [--keep-env] [--dry-run]
#   --no-baseline  skip the FlyDSL no-HERD config (saves ~7 min)
#   --unit-test    also run aiter op_tests/test_flydsl_herd.py (~4 min)
#   --keep-env     leave AITER_DIR installed afterwards
#   --dry-run      check prerequisites and print the plan; touches neither the GPU nor the python env
# Env overrides: AITER_DIR ATOM_SRC ATOM_DIR MODEL BACKUP_DIR RESULTS_ROOT
# Takes ~25 min and needs the GPU to itself. If interrupted mid-run, stop the server
# (pkill -INT -f "^python -m atom.entrypoints.openai_server") and run: bash env.sh restore <BACKUP_DIR>

set -euo pipefail
BENCH=$(cd "$(dirname "$0")" && pwd)
AITER_DIR=${AITER_DIR:-/data/binding/moe_1031/opt/aiter}
ATOM_SRC=${ATOM_SRC:-/app/ATOM}
ATOM_DIR=${ATOM_DIR:-/data/binding/moe_1031/atom_pr2713}
MODEL=${MODEL:-/data/models/DeepSeek-V4-Flash}
# Per container: a backup restores this container's own site-packages, not another image's.
BACKUP_DIR=${BACKUP_DIR:-$BENCH/env_backup/$(hostname)}
RESULTS_ROOT=${RESULTS_ROOT:-$BENCH/results}
AITER_PR6527_MERGE=b5078fc49a131b39178558d9160cb08ccf1a817f

TAG=$(date +%Y%m%d_%H%M%S)
OUT=""
BASELINE=1
UNIT_TEST=0
KEEP_ENV=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --no-baseline) BASELINE=0; shift ;;
    --unit-test) UNIT_TEST=1; shift ;;
    --keep-env) KEEP_ENV=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
RESULTS=$RESULTS_ROOT/$TAG
OUT=${OUT:-$RESULTS/flydsl_vs_gluon.txt}
step() { echo; echo "[$(date +%T)] ==== $*"; }

if [ $DRY_RUN -eq 0 ]; then
  if [ -e "$RESULTS" ]; then echo "$RESULTS already exists; pass another --tag"; exit 1; fi
  mkdir -p "$RESULTS"
  exec > >(tee -a "$RESULTS/run_flydsl_vs_gluon.log") 2>&1
fi

step "0/7 prerequisites"
[ -f "$MODEL/config.json" ] || { echo "model not found: $MODEL"; exit 1; }
if curl -sf http://localhost:8000/health > /dev/null; then echo "port 8000 is serving; stop that server first"; exit 1; fi
if pgrep -f "^python -m atom.entrypoints.openai_server" > /dev/null; then echo "an ATOM server is running; stop it first"; exit 1; fi
echo "ok: model $MODEL, port 8000 free"

step "1/7 python env backup ($BACKUP_DIR)"
if [ -f "$BACKUP_DIR/pip_freeze_before.txt" ]; then
  echo "reusing the existing backup"
elif [ $DRY_RUN -eq 1 ]; then
  echo "would back up the current env to $BACKUP_DIR"
else
  bash "$BENCH/env.sh" backup "$BACKUP_DIR"
fi

step "2/7 aiter#6527 in $AITER_DIR"
if ! git -C "$AITER_DIR" merge-base --is-ancestor $AITER_PR6527_MERGE HEAD 2>/dev/null; then
  echo "aiter#6527 (merge $AITER_PR6527_MERGE) is not in $AITER_DIR; merge it first"
  exit 1
fi
echo "ok: $(git -C "$AITER_DIR" log --oneline -1)"

step "3/7 ATOM#2713 copy ($ATOM_DIR)"
CHECK_ONLY=$DRY_RUN bash "$BENCH/patch_atom_herd.sh" "$ATOM_SRC" "$ATOM_DIR"

CONFIGS=("flydsl_herd serve_dsv4_tp1_flydsl_herd.sh REQUIRE_LOG=FlyDSL HERD route")
if [ $BASELINE -eq 1 ]; then CONFIGS+=("flydsl_noherd serve_dsv4_tp1.sh FORBID_LOG=FlyDSL HERD route"); fi
CONFIGS+=("triton_herd serve_dsv4_tp1_triton_herd.sh REQUIRE_LOG=Triton MoE experts")

if [ $DRY_RUN -eq 1 ]; then
  step "plan (dry run, nothing executed)"
  echo "pip install $AITER_DIR; results -> $RESULTS; report -> $OUT"
  for c in "${CONFIGS[@]}"; do
    read -r name serve check <<< "$c"
    echo "  ${check%%=*}='${check#*=}' AITER_DIR=$AITER_DIR ATOM_DIR=$ATOM_DIR bash $BENCH/run_config.sh $RESULTS/$name $serve"
  done
  echo "env restore from $BACKUP_DIR $([ $KEEP_ENV -eq 1 ] && echo '(skipped: --keep-env)')"
  exit 0
fi

restore() {
  if [ $KEEP_ENV -eq 0 ]; then
    step "restore python env"
    bash "$BENCH/env.sh" restore "$BACKUP_DIR" || true
  fi
  chown -R --reference="$(dirname "$BENCH")" "$RESULTS" 2>/dev/null || true
}
trap restore EXIT

step "4/7 pip install $AITER_DIR"
bash "$BENCH/env.sh" install "$AITER_DIR"
if [ $UNIT_TEST -eq 1 ]; then
  step "aiter HERD unit tests"
  (cd "$AITER_DIR" && PYTHONPATH=$AITER_DIR python -m pytest -q -p no:cacheprovider op_tests/test_flydsl_herd.py | tail -n 3)
fi

step "5/7 e2e runs -> $RESULTS"
export AITER_DIR ATOM_DIR MODEL
for c in "${CONFIGS[@]}"; do
  read -r name serve check <<< "$c"
  env "${check%%=*}=${check#*=}" bash "$BENCH/run_config.sh" "$RESULTS/$name" "$serve"
done

step "6/7 HERD check"
if ! compgen -G "$RESULTS/flydsl_herd/flydsl_cache/launch_route_psum_herd_*" > /dev/null; then
  echo "launch_route_psum_herd was not built: FlyDSL HERD did not run"
  exit 1
fi
echo "ok: $(grep -c 'FlyDSL HERD route' "$RESULTS/flydsl_herd/server.log") 'FlyDSL HERD route' log lines, HERD route kernel built"

step "7/7 report -> $OUT"
REPORT_ARGS=(--results "$RESULTS" --out "$OUT")
[ $BASELINE -eq 1 ] || REPORT_ARGS+=(--flydsl-noherd "")
python3 "$BENCH/report_flydsl_vs_gluon.py" "${REPORT_ARGS[@]}"
grep -A 3 "^结论" "$OUT" || true
chown --reference="$(dirname "$BENCH")" "$OUT" 2>/dev/null || true
echo "report: $OUT"
