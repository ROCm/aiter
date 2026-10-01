#!/usr/bin/env bash
# Turn-key driver for tuning the FlyDSL all-reduce dispatch tables.
# See README.md in this directory for the full walkthrough.
#
#   tune.sh [options] <command> [args]
#
# Commands (run in this order; `all` = check, sweep, fit, fused-sweep, fused-fit):
#   setup        pip-install the Python deps the benchmark needs (flydsl pin, pyzmq, tabulate)
#   check        verify GPUs, deps, fabric type and that the benchmark can be imported
#   sweep        plain all-reduce sweep  (fast accuracy; fitting data)      -> $OUT/sweep
#   fit          fit family policy + ladders from the sweep                  -> $OUT/fit/fit.txt
#   fused-sweep  fused all-reduce+RMSNorm one-shot sweep (split-H)           -> $OUT/fused/sweep
#   fused-fit    fit FUSED_ONESHOT_LADDER + FUSED_FAMILY_POLICY              -> $OUT/fused/fit.txt
#   ---- paste the fitted values into the source tables (README step 5), then: ----
#   test         run the table-invariant unit tests (after pasting, before validate)
#   validate     re-sweep the shipped dispatcher (fly_auto) vs pinned rows   -> $OUT/validate
#   audit        grade fly_auto against the best pinned row (PASS/FAIL)      -> $OUT/audit
#   report       reference benchmark tables: fly_auto vs cdr/qr_int4/rccl    -> $OUT/report
#
# Options (before the command):
#   --out DIR       output root (default: <this dir>/out/<host>; "out/" is git-ignored)
#   --tp "8 4 2"    world sizes (default "8 4 2"; sizes above the GPU count are skipped)
#   --link L        pcie|xgmi|auto (default auto: read from the KFD topology)
#   --widths "..."  hidden sizes for the fused sweep (default "3072 4096 7168 8192")
#   --smoke         plumbing check: 3 shapes, a handful of candidates, 10 iters
#   --force         redo chunks whose CSV already exists (default: skip = resumable)
#   -h, --help      this text
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
BENCH="$(dirname "$HERE")/bench_comm_allreduce.py"
PY=${PYTHON:-python}

OUT=""; TPS="8 4 2"; LINK="auto"; WIDTHS="3072 4096 7168 8192"; SMOKE=0; FORCE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --tp) TPS=$2; shift 2 ;;
    --link) LINK=$2; shift 2 ;;
    --widths) WIDTHS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) break ;;
  esac
done
CMD=${1:-}; [[ $# -gt 0 ]] && shift
[[ -z $CMD ]] && { echo "no command given (see --help)" >&2; exit 2; }

[[ -z $OUT ]] && OUT="$HERE/out/$(hostname -s)$([[ $SMOKE == 1 ]] && echo -smoke)"
mkdir -p "$OUT"

# --- environment ---------------------------------------------------------------------
# AITER_META_DIR/PYTHONPATH: aiter is usually not pip-installed, and an "install"-mode
#   aiter would otherwise pick up a stale aiter_meta/ build artifact (ImportError on
#   get_kernel_instance in aiter/ops/opus/dispatch.py).
# AITER_FLY_AR: opt the FlyDSL all-reduce path in.
# FLYDSL_EXTRA_SOURCE_DIRS: the FlyDSL disk cache does NOT key on the shared kernel
#   modules the jit entry points import. Without this, editing a ladder table runs
#   STALE compiled kernels and the validation falsely passes.
export AITER_META_DIR=$REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export AITER_FLY_AR=1
export FLYDSL_EXTRA_SOURCE_DIRS=$REPO/aiter/ops/flydsl/kernels${FLYDSL_EXTRA_SOURCE_DIRS:+:$FLYDSL_EXTRA_SOURCE_DIRS}

# --- candidates ----------------------------------------------------------------------
MESH_PINNED=""
for b in 64 128 256 512; do for st in 1 8; do for ss in "" _ss; do
  MESH_PINNED+=" fly_int4_b${b}_st${st}_g128${ss}"
done; done; done
RING_PINNED=""
for b in 64 128 256 512; do for st in 8 16 32; do
  RING_PINNED+=" fly_int4_ring_b${b}_st${st}_g128"
done; done
ONESHOT_PINNED="fly_1stage_b256_a1_g64 fly_1stage_b256_a1_g128 fly_1stage_b128_a1_g128 \
fly_1stage_b64_a1_g64 fly_1stage_b64_a1_g256 fly_1stage_b512_a1_g64 \
fly_1stage_b256_a2_g64 fly_1stage_b256_a4_g64 fly_1stage_b512_a4_g64"

FAST_C="cdr cdr_naive qr_int4 fly_auto fly_1stage $ONESHOT_PINNED $MESH_PINNED $RING_PINNED"
# Exact mode only ever reaches the one-shot (quantized rows are n/a).
EXACT_C="cdr cdr_naive fly_auto fly_1stage $ONESHOT_PINNED"
# Validation: shipped dispatcher vs every pinned row it could have chosen (no ring: never dispatched on xGMI).
VAL_FAST_C="cdr cdr_naive fly_auto fly_1stage $ONESHOT_PINNED $MESH_PINNED"
VAL_EXACT_C=$EXACT_C
REPORT_C="cdr cdr_naive qr_int4 rccl fly_auto"

WARMUP=5; ITERS=50
SHAPES=$HERE/shapes; FUSED_SHAPES=$HERE/fused/shapes
if [[ $SMOKE == 1 ]]; then
  FAST_C="cdr fly_auto fly_1stage fly_1stage_b256_a1_g128 fly_int4_b256_st1_g128_ss"
  EXACT_C="cdr fly_auto fly_1stage fly_1stage_b256_a1_g128"
  VAL_FAST_C=$FAST_C; VAL_EXACT_C=$EXACT_C
  REPORT_C="cdr fly_auto"
  WARMUP=2; ITERS=10; TPS=${TPS_SMOKE:-2}
  WIDTHS="4096"
  SHAPES=$OUT/smoke_shapes; FUSED_SHAPES=$OUT/smoke_shapes_fused
  mkdir -p "$SHAPES" "$FUSED_SHAPES"
  for f in "$HERE"/shapes/*.csv; do head -4 "$f" > "$SHAPES/$(basename "$f")"; done
  for f in "$HERE"/fused/shapes/*.csv; do head -4 "$f" > "$FUSED_SHAPES/$(basename "$f")"; done
fi
FAILED=()

# --- helpers -------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "error: $*" >&2; exit 1; }

ngpu() { $PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0; }

resolve_link() {
  if [[ $LINK == auto ]]; then
    LINK=$($PY -c "
from aiter.ops.flydsl.allreduce_shared import has_xgmi_peer_links
print('xgmi' if has_xgmi_peer_links() else 'pcie')" 2>/dev/null | tail -1)
    [[ $LINK == xgmi || $LINK == pcie ]] || die "could not detect fabric; pass --link pcie|xgmi"
  fi
}

write_provenance() {
  {
    echo "date:   $(date -Is)"
    echo "host:   $(hostname)"
    echo "repo:   $REPO @ $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" branch --show-current))"
    echo "dirty:  $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) modified tracked file(s)"
    echo "link:   $LINK"
    echo "gpus:   $(ngpu)"
    echo "flydsl: $($PY -c 'import importlib.metadata as m; print(m.version("flydsl"))' 2>/dev/null)"
    echo "cmd:    $CMD (tp=$TPS widths=$WIDTHS smoke=$SMOKE)"
  } >> "$OUT/provenance.txt"
}

# bench_chunk <tag> <outdir> <bench args...>: one resumable benchmark invocation.
bench_chunk() {
  local tag=$1 dir=$2; shift 2
  mkdir -p "$dir"
  if [[ $FORCE == 0 && -s $dir/$tag.csv ]]; then log "skip $tag (done)"; return 0; fi
  log "run  $tag"
  if $PY "$BENCH" "$@" --timing graph --warmup $WARMUP --iters $ITERS \
       -o "$dir/$tag.md" --output-csv "$dir/$tag.csv" >"$dir/$tag.log" 2>&1 && [[ -s $dir/$tag.csv ]]; then
    log "done $tag"
  else
    rm -f "$dir/$tag.csv"
    log "FAIL $tag (see $dir/$tag.log)"; FAILED+=("$tag")
  fi
}

tp_ok() { [[ $1 -le $NGPU ]] || { log "skip tp$1: only $NGPU GPU(s) visible"; return 1; }; }

finish() {
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED chunks: ${FAILED[*]}  (re-run the same command to retry them)" >&2; exit 1
  fi
}

# --- commands ------------------------------------------------------------------------
cmd_setup() {
  local pin; pin=$(grep -E '^flydsl==' "$REPO/requirements.txt") || die "no flydsl pin in requirements.txt"
  log "installing $pin pyzmq tabulate"
  $PY -m pip install "$pin" pyzmq tabulate
}

cmd_check() {
  local ok=1
  NGPU=$(ngpu); log "GPUs visible: $NGPU"; [[ $NGPU -ge 2 ]] || { log "need >= 2 GPUs"; ok=0; }
  for mod in torch pandas tabulate zmq flydsl; do
    if $PY -c "import $mod" 2>/dev/null; then log "ok   import $mod"; else log "MISS import $mod  (run: tune.sh setup)"; ok=0; fi
  done
  local want have
  want=$(grep -E '^flydsl==' "$REPO/requirements.txt" | cut -d= -f3)
  have=$($PY -c 'import importlib.metadata as m; print(m.version("flydsl"))' 2>/dev/null)
  [[ $want == "$have" ]] && log "ok   flydsl $have" || { log "flydsl is '$have', repo pins '$want'  (run: tune.sh setup)"; ok=0; }
  if $PY -c "import aiter" 2>"$OUT/check_import.err"; then log "ok   import aiter"; else
    log "FAIL import aiter: $(tail -1 "$OUT/check_import.err")"; ok=0; fi
  resolve_link && log "fabric: $LINK"
  if $PY "$BENCH" --list-candidates >/dev/null 2>"$OUT/check_bench.err"; then log "ok   benchmark loads"; else
    log "FAIL benchmark: $(tail -1 "$OUT/check_bench.err")"; ok=0; fi
  [[ $ok == 1 ]] || die "environment check failed"
  log "environment OK"
}

cmd_sweep() {
  NGPU=$(ngpu); resolve_link; write_provenance
  for tp in $TPS; do tp_ok "$tp" || continue
    for shape in a_small a_large b_ksens; do
      bench_chunk "tp${tp}_fast_${shape}" "$OUT/sweep" -tp "$tp" \
        --shape-csv "$SHAPES/ar_sweep_${shape}.csv" -c $FAST_C --fly-accuracy fast
    done
  done
  finish
  log "sweep done -> $OUT/sweep   next: tune.sh fit"
}

fit_py() { $PY "$HERE/fit_allreduce_policy.py" "$@"; }

cmd_fit() {
  resolve_link; mkdir -p "$OUT/fit"
  local fit_in=() hold=()
  while IFS= read -r f; do fit_in+=("$f"); done < <(ls "$OUT"/sweep/tp*_fast_a_*.csv 2>/dev/null)
  while IFS= read -r f; do hold+=("$f"); done < <(ls "$OUT"/sweep/tp*_fast_b_ksens.csv 2>/dev/null)
  [[ ${#fit_in[@]} -gt 0 ]] || die "no sweep CSVs under $OUT/sweep (run: tune.sh sweep)"
  local args=("${fit_in[@]}" --link "$LINK")
  [[ ${#hold[@]} -gt 0 ]] && args+=(--holdout "${hold[@]}")
  log "fitting (link=$LINK, ${#fit_in[@]} sweep file(s), ${#hold[@]} holdout file(s))"
  fit_py "${args[@]}" 2>&1 | tee "$OUT/fit/fit.txt"
  local rc=${PIPESTATUS[0]}
  [[ $rc == 0 ]] || die "fit failed (rc=$rc)"
  log "fit output -> $OUT/fit/fit.txt  (see the '## paste-ready' block; README step 5)"
}

cmd_test() {
  (cd "$REPO" && $PY -m pytest -q op_tests/flydsl_tests/test_flydsl_allreduce_policy.py)
}

cmd_validate() {
  NGPU=$(ngpu); resolve_link; write_provenance
  for tp in $TPS; do tp_ok "$tp" || continue
    for shape in a_small a_large b_ksens; do
      bench_chunk "val_tp${tp}_fast_${shape}" "$OUT/validate" -tp "$tp" \
        --shape-csv "$SHAPES/ar_sweep_${shape}.csv" -c $VAL_FAST_C --fly-accuracy fast
      bench_chunk "val_tp${tp}_exact_${shape}" "$OUT/validate" -tp "$tp" \
        --shape-csv "$SHAPES/ar_sweep_${shape}.csv" -c $VAL_EXACT_C --fly-accuracy exact
    done
  done
  finish
  log "validate done -> $OUT/validate   next: tune.sh audit"
}

cmd_audit() {
  mkdir -p "$OUT/audit"; local rc=0 f
  for acc in fast exact; do
    f=(); while IFS= read -r x; do f+=("$x"); done < <(ls "$OUT"/validate/val_tp*_"$acc"_*.csv 2>/dev/null)
    [[ ${#f[@]} -gt 0 ]] || { log "no $acc validation CSVs (run: tune.sh validate)"; rc=1; continue; }
    log "auditing $acc"
    fit_py "${f[@]}" --audit-auto 2>&1 | tee "$OUT/audit/audit_$acc.txt"
    [[ ${PIPESTATUS[0]} == 0 ]] || rc=1
  done
  [[ $rc == 0 ]] && log "AUDIT PASS" || { log "AUDIT FAIL (see $OUT/audit)"; exit 1; }
}

cmd_report() {
  NGPU=$(ngpu); resolve_link; write_provenance
  for tp in $TPS; do tp_ok "$tp" || continue
    bench_chunk "report_tp${tp}_fast"  "$OUT/report" -tp "$tp" -c $REPORT_C --fly-accuracy fast
    bench_chunk "report_tp${tp}_exact" "$OUT/report" -tp "$tp" -c $REPORT_C --fly-accuracy exact
  done
  finish
  log "report tables -> $OUT/report/*.md"
}

cmd_fused_sweep() {
  NGPU=$(ngpu); resolve_link; write_provenance
  local fly1s cands
  fly1s=$($PY - "$BENCH" <<'EOF' 2>/dev/null | tail -1
import importlib.util, sys
s = importlib.util.spec_from_file_location("b", sys.argv[1])
b = importlib.util.module_from_spec(s); sys.modules["b"] = b; s.loader.exec_module(b)
print(" ".join(c.key for c in b.CANDIDATES if c.family == "fused_fly1s"))
EOF
)
  [[ -n $fly1s ]] || die "failed to enumerate fused_fly1s candidates"
  # Every fused one-shot row (ladder default, pinned unsplit grid, split-H grid) plus
  # exact baselines and the quantized mesh for the fast-mode boundary.
  cands="fused_cdr_1stage fused_cdr_2stage separate_cdr separate_rccl fused_fly_auto fused_fly_mesh fused_fly_mesh_ss $fly1s"
  if [[ $SMOKE == 1 ]]; then cands="fused_cdr_1stage fused_fly_auto $(cut -d' ' -f1-3 <<<"$fly1s")"; fi
  log "fused sweep: $(wc -w <<<"$cands") candidates, tp=[$TPS] widths=[$WIDTHS]"
  for tp in $TPS; do tp_ok "$tp" || continue
    for h in $WIDTHS; do
      bench_chunk "tp${tp}_w${h}" "$OUT/fused/sweep" --fusion ar_rmsnorm -tp "$tp" \
        --shape-csv "$FUSED_SHAPES/fused_w${h}.csv" -c $cands
    done
  done
  finish
  log "fused sweep done -> $OUT/fused/sweep   next: tune.sh fused-fit"
}

cmd_fused_fit() {
  resolve_link; local d=$OUT/fused/sweep
  ls "$d"/tp*_w*.csv >/dev/null 2>&1 || die "no fused sweep CSVs under $d (run: tune.sh fused-sweep)"
  $PY "$HERE/fused/analyze_fused.py" "$d" --summary-csv "$OUT/fused/summary.csv" > "$OUT/fused/summary.txt" \
    || die "analyze_fused failed"
  $PY "$HERE/fused/fit_fused.py" "$d" --link "$LINK" 2>&1 | tee "$OUT/fused/fit.txt"
  [[ ${PIPESTATUS[0]} == 0 ]] || die "fused fit failed"
  log "fused fit -> $OUT/fused/fit.txt (per-shape summary: $OUT/fused/summary.txt); README step 5"
}

case $CMD in
  setup) cmd_setup ;;
  check) cmd_check ;;
  sweep) cmd_sweep ;;
  fit) cmd_fit ;;
  test) cmd_test ;;
  validate) cmd_validate ;;
  audit) cmd_audit ;;
  report) cmd_report ;;
  fused-sweep) cmd_fused_sweep ;;
  fused-fit) cmd_fused_fit ;;
  all) cmd_check && cmd_sweep && cmd_fit && cmd_fused_sweep && cmd_fused_fit
       log "all fits done. Now paste the fitted values into the source tables (README step 5), then: tune.sh validate && tune.sh audit" ;;
  *) die "unknown command '$CMD' (see --help)" ;;
esac
