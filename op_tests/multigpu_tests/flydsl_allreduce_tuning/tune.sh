#!/usr/bin/env bash
# Turn-key driver for tuning the FlyDSL all-reduce dispatch tables.
# See README.md in this directory for the full walkthrough.
#
#   tune.sh [options] <command> [args]
#
# Every measurement is the bench scenario of the two multi-GPU tests
# (test_flydsl_one_shot_allreduce.py, test_flydsl_quick_allreduce.py), with the
# pinned configurations of candidates.csv timed next to the shipping engines.
#
# Commands (run in this order; `all` = check, sweep, fit, fused-sweep, fused-fit):
#   setup        pip-install the Python deps the benchmark needs (flydsl pin, pyzmq, tabulate)
#   check        verify GPUs, deps, fabric type, the test scripts and candidates.csv
#   sweep        plain all-reduce sweep (fitting data)                       -> $OUT/sweep
#   fit          fit FAMILY_POLICIES rows + ladders from the sweep             -> $OUT/fit/fit.txt
#   fused-sweep  fused all-reduce+RMSNorm sweep: one-shot (split-H) and
#                quantized mesh/ring (atoms per row x super-tile)            -> $OUT/fused/sweep
#   fused-fit    fit the fused ladders (default + per width) and
#                FUSED_FAMILY_POLICIES                                       -> $OUT/fused/fit.txt
#                                         + the ladder file $OUT/fused/fused_allreduce_ladders.json
#   ---- paste the fitted values into allreduce_policy.py and copy the ladder file
#        over aiter/ops/flydsl/fused_allreduce_ladders.json (README step 5), then: ----
#   test         run the table-invariant unit tests (after pasting, before validate)
#   validate     re-sweep the shipping engines vs the pinned rows            -> $OUT/validate
#   audit        grade the shipped dispatcher against the best pinned row    -> $OUT/audit
#   report       reference bench tables: shipping engines vs stock kernels   -> $OUT/report
#   low-latency  the low-latency fused mesh build vs the shipping fused mesh,
#                at each (TP, width) that has the build; sweep, then
#                compare (not part of `all`)                    -> $OUT/low_latency/compare.txt
#
# Options (before the command):
#   --out DIR       output root (default: <this dir>/out/<host>; "out/" is git-ignored)
#   --tp "8 4 2"    world sizes (default "8 4 2"; sizes above the GPU count are skipped)
#   --link L        pcie|xgmi|auto (default auto: read from the KFD topology)
#   --widths "..."  hidden sizes for the fused sweep (default "3072 4096 7168 8192")
#                   and for low-latency (default "2048 4096 8192", its only widths)
#   --width-gain G  speedup over the default fused ladder, minus 1, that a width needs
#                   to keep a ladder of its own (default 0.02)
#   --smoke         plumbing check: 3 shapes, the `smoke` candidates, few samples
#   --force         redo chunks whose CSV already exists (default: skip = resumable)
#   --dry-run       GPU-free rehearsal: runs the checks (GPU count not required), lists every
#                   chunk with its bench command and verifies its inputs exist, and only
#                   loads (--help) the fit scripts. Benchmarks and fits never run.
#   -h, --help      this text
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
ONESHOT="$REPO/op_tests/multigpu_tests/test_flydsl_one_shot_allreduce.py"
QUICK="$REPO/op_tests/multigpu_tests/test_flydsl_quick_allreduce.py"
CANDS="$HERE/candidates.csv"
PY=${PYTHON:-python}

OUT=""; TPS="8 4 2"; LINK="auto"; WIDTHS="3072 4096 7168 8192"; LL_WIDTHS="2048 4096 8192"
WIDTH_GAIN=0.02; SMOKE=0; FORCE=0; DRY=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --tp) TPS=$2; shift 2 ;;
    --link) LINK=$2; shift 2 ;;
    --widths) WIDTHS=$2; LL_WIDTHS=$2; shift 2 ;;
    --width-gain) WIDTH_GAIN=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) break ;;
  esac
done
CMD=${1:-}; [[ $# -gt 0 ]] && shift
[[ -z $CMD ]] && { echo "no command given (see --help)" >&2; exit 2; }

[[ -z $OUT ]] && OUT="$HERE/out/$(hostname -s)$([[ $SMOKE == 1 ]] && echo -smoke)$([[ $DRY == 1 ]] && echo -dryrun)"
mkdir -p "$OUT"

# --- environment ---------------------------------------------------------------------
# AITER_META_DIR/PYTHONPATH: aiter is usually not pip-installed, and an "install"-mode
#   aiter would otherwise pick up a stale aiter_meta/ build artifact (ImportError on
#   get_kernel_instance in aiter/ops/opus/dispatch.py).
# FLYDSL_EXTRA_SOURCE_DIRS: the FlyDSL disk cache does NOT key on the shared kernel
#   modules the jit entry points import. Without this, editing a ladder table runs
#   STALE compiled kernels and the validation falsely passes.
export AITER_META_DIR=$REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
KERNELS=$REPO/aiter/ops/flydsl/kernels
export FLYDSL_EXTRA_SOURCE_DIRS=$KERNELS:$KERNELS/comm${FLYDSL_EXTRA_SOURCE_DIRS:+:$FLYDSL_EXTRA_SOURCE_DIRS}
# FLYDSL_QR_TIMEOUT: seconds a bench spawn may take. A fused chunk times every
#   candidate at every shape of a width in one spawn, which on a cold JIT cache
#   outlasts the tests' 600 s verify default.
export FLYDSL_QR_TIMEOUT=${FLYDSL_QR_TIMEOUT:-3600}
# TRITON_LIBHIP_PATH: the rocm/pytorch rocm10.0 image points it at a second copy of
#   libamdhip64 (_rocm_sdk_devel) next to torch's (_rocm_sdk_core). Triton then maps
#   both, CustomAllreduce's IPC check binds to the one that does not know torch's
#   allocations, and every fused bench fails on its cdr baseline. Unset, Triton
#   uses the runtime torch already loaded.
unset TRITON_LIBHIP_PATH

# --- candidates and sampling -----------------------------------------------------------
# Sets of candidates.csv (see gen_candidates.py): `sweep` for the plain fit,
# `validate` for the shipped dispatcher's choices, `fused` and `fused_qr` for the
# fused fit, `fused_ll` for the low-latency comparison.
SWEEP_SET=sweep; VAL_SET=validate; FUSED_SET=fused; FUSED_QR_SET=fused_qr; LL_SET=fused_ll
REPS=10; ROUNDS=12
SHAPES=$HERE/shapes; FUSED_SHAPES=$HERE/fused/shapes
if [[ $SMOKE == 1 ]]; then
  SWEEP_SET=smoke; VAL_SET=smoke; FUSED_SET=smoke; FUSED_QR_SET=smoke; LL_SET=smoke
  REPS=2; ROUNDS=3; TPS=${TPS_SMOKE:-2}
  WIDTHS="4096"; LL_WIDTHS="4096"
  SHAPES=$OUT/smoke_shapes; FUSED_SHAPES=$OUT/smoke_shapes_fused
  mkdir -p "$SHAPES" "$FUSED_SHAPES"
  for f in "$HERE"/shapes/*.csv; do head -4 "$f" > "$SHAPES/$(basename "$f")"; done
  for f in "$HERE"/fused/shapes/*.csv; do head -4 "$f" > "$FUSED_SHAPES/$(basename "$f")"; done
fi
FAILED=()

# --- helpers -------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "error: $*" >&2; exit 1; }

# The tests' count (flydsl_allreduce_bench.gpu_count): torch's own reads 0 where amdsmi
# sees no GPUs, as in the rocm/pytorch rocm10.0 image.
ngpu_real() {
  PYTHONPATH=$(dirname "$ONESHOT"):$PYTHONPATH \
    $PY -c "import flydsl_allreduce_bench as fab; print(fab.gpu_count())" 2>/dev/null | tail -1 | grep -E '^[0-9]+$' || echo 0
}
# A dry run has no GPUs; it pretends the node is full so every chunk is enumerated.
ngpu() { if [[ $DRY == 1 ]]; then echo 8; else ngpu_real; fi; }

resolve_link() {
  if [[ $LINK == auto ]]; then
    LINK=$($PY -c "
from aiter.ops.flydsl.allreduce_policy import detect_link
print(detect_link())" 2>/dev/null | tail -1)
    if [[ $LINK != xgmi && $LINK != pcie ]]; then
      [[ $DRY == 1 ]] || die "could not detect fabric; pass --link pcie|xgmi"
      LINK=xgmi; log "dry-run: no KFD topology, assuming link=xgmi"
    fi
  fi
}

write_provenance() {
  [[ $DRY == 1 ]] && return 0
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

# Progress: a command calls `plan <n>` with its chunk count; every bench_chunk then logs
# [done/total], elapsed time and an ETA (mean time of the chunks actually run x chunks left).
CHUNK_TOTAL=0; CHUNK_DONE=0; CHUNK_RAN=0; CHUNK_SECS=0; PLAN_T0=$SECONDS

plan() { CHUNK_TOTAL=$1; CHUNK_DONE=0; CHUNK_RAN=0; CHUNK_SECS=0; PLAN_T0=$SECONDS; log "plan: $1 chunk(s)"; }

fmt_secs() { printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

progress() {
  CHUNK_DONE=$((CHUNK_DONE + 1))
  local msg="progress $CHUNK_DONE/$CHUNK_TOTAL ($((100 * CHUNK_DONE / CHUNK_TOTAL))%), elapsed $(fmt_secs $((SECONDS - PLAN_T0)))"
  if [[ $CHUNK_RAN -gt 0 && $CHUNK_DONE -lt $CHUNK_TOTAL ]]; then
    msg+=", ETA ~$(fmt_secs $((CHUNK_SECS / CHUNK_RAN * (CHUNK_TOTAL - CHUNK_DONE))))"
  fi
  log "$msg"
}

# count_tps: how many of $TPS fit the visible GPUs (silent tp_ok).
count_tps() { local n=0 tp; for tp in $TPS; do [[ $tp -le $NGPU ]] && n=$((n + 1)); done; echo $n; }

# bench_chunk <tag> <outdir> <test script> <bench args...>: one resumable bench run.
bench_chunk() {
  local tag=$1 dir=$2 script=$3 prev=""; shift 3
  [[ $DRY == 1 ]] || mkdir -p "$dir"
  if [[ $DRY == 0 && $FORCE == 0 && -s $dir/$tag.csv ]]; then log "skip $tag (done)"; progress; return 0; fi
  if [[ $DRY == 1 ]]; then
    local a
    for a in "$@"; do
      if [[ $prev == --shape-csv && ! -s $a ]]; then log "FAIL $tag: missing shape file $a"; FAILED+=("$tag"); fi
      prev=$a
    done
    log "dry  $tag: $(basename "$script") --scenario bench $* --reps $REPS --rounds $ROUNDS"
    progress; return 0
  fi
  log "run  $tag"
  local t0=$SECONDS
  if $PY "$script" --scenario bench "$@" --reps $REPS --rounds $ROUNDS \
       --report "$dir/$tag.md" --csv "$dir/$tag.csv" >"$dir/$tag.log" 2>&1 && [[ -s $dir/$tag.csv ]]; then
    log "done $tag"
  else
    rm -f "$dir/$tag.csv"
    log "FAIL $tag (see $dir/$tag.log)"; FAILED+=("$tag")
  fi
  CHUNK_RAN=$((CHUNK_RAN + 1)); CHUNK_SECS=$((CHUNK_SECS + SECONDS - t0)); progress
}

tp_ok() { [[ $1 -le $NGPU ]] || { log "skip tp$1: only $NGPU GPU(s) visible"; return 1; }; }

finish() {
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED chunks: ${FAILED[*]}  (re-run the same command to retry them)" >&2; exit 1
  fi
}

# plain_chunks <outdir> <prefix> <candidate set>: both tests over every shape set.
plain_chunks() {
  local dir=$1 prefix=$2 set=$3 tp shape
  plan $(($(count_tps) * 3 * 2))
  for tp in $TPS; do tp_ok "$tp" || continue
    for shape in a_small a_large b_ksens; do
      bench_chunk "${prefix}oneshot_tp${tp}_${shape}" "$dir" "$ONESHOT" --tp "$tp" \
        --shape-csv "$SHAPES/ar_sweep_${shape}.csv" --candidates "$CANDS" --candidate-set "$set"
      bench_chunk "${prefix}quick_tp${tp}_${shape}" "$dir" "$QUICK" --tp "$tp" \
        --shape-csv "$SHAPES/ar_sweep_${shape}.csv" --candidates "$CANDS" --candidate-set "$set"
    done
  done
}

# --- commands ------------------------------------------------------------------------
cmd_setup() {
  local pin; pin=$(grep -E '^flydsl==' "$REPO/requirements.txt") || die "no flydsl pin in requirements.txt"
  log "installing $pin pyzmq tabulate"
  $PY -m pip install "$pin" pyzmq tabulate
}

cmd_check() {
  local ok=1
  NGPU=$(ngpu); log "GPUs visible: $(ngpu_real)"
  if [[ $DRY == 1 ]]; then log "dry-run: GPU count not required"
  else [[ $NGPU -ge 2 ]] || { log "need >= 2 GPUs"; ok=0; }; fi
  for mod in torch pandas tabulate zmq flydsl; do
    if $PY -c "import $mod" 2>/dev/null; then log "ok   import $mod"; else log "MISS import $mod  (run: tune.sh setup)"; ok=0; fi
  done
  local want have
  # The pin may carry an environment marker: "flydsl==X; sys_platform != ...".
  want=$(grep -E '^flydsl==' "$REPO/requirements.txt" | sed -E 's/^flydsl==([^;[:space:]]+).*/\1/')
  have=$($PY -c 'import importlib.metadata as m; print(m.version("flydsl"))' 2>/dev/null)
  [[ $want == "$have" ]] && log "ok   flydsl $have" || { log "flydsl is '$have', repo pins '$want'  (run: tune.sh setup)"; ok=0; }
  if $PY -c "import aiter" 2>"$OUT/check_import.err"; then log "ok   import aiter"; else
    log "FAIL import aiter: $(tail -1 "$OUT/check_import.err")"; ok=0; fi
  resolve_link && log "fabric: $LINK"
  local t
  for t in "$ONESHOT" "$QUICK"; do
    if $PY "$t" --help >/dev/null 2>"$OUT/check_bench.err"; then log "ok   $(basename "$t") loads"; else
      log "FAIL $(basename "$t"): $(tail -1 "$OUT/check_bench.err")"; ok=0; fi
  done
  if $PY "$HERE/gen_candidates.py" --check >/dev/null 2>"$OUT/check_cands.err"; then log "ok   candidates.csv is current"; else
    log "FAIL candidates.csv: $(tail -1 "$OUT/check_cands.err")"; ok=0; fi
  [[ $ok == 1 ]] || die "environment check failed"
  log "environment OK"
}

cmd_sweep() {
  NGPU=$(ngpu); resolve_link; write_provenance
  plain_chunks "$OUT/sweep" "" "$SWEEP_SET"
  finish
  log "sweep done -> $OUT/sweep   next: tune.sh fit"
}

fit_py() { $PY "$HERE/fit_allreduce_policy.py" "$@"; }

# dry_fit_scripts: a dry run has no sweep data, so only prove the fit scripts import.
dry_fit_scripts() {
  local f
  for f in "$@"; do
    if $PY "$HERE/$f" --help >/dev/null 2>"$OUT/check_fit.err"; then log "ok   $f loads"; else
      log "FAIL $f: $(tail -1 "$OUT/check_fit.err")"; FAILED+=("$f"); fi
  done
  finish
}

cmd_fit() {
  if [[ $DRY == 1 ]]; then dry_fit_scripts fit_allreduce_policy.py; return; fi
  resolve_link; mkdir -p "$OUT/fit"
  local fit_in=() hold=()
  while IFS= read -r f; do fit_in+=("$f"); done < <(ls "$OUT"/sweep/*_tp*_a_*.csv 2>/dev/null)
  while IFS= read -r f; do hold+=("$f"); done < <(ls "$OUT"/sweep/*_tp*_b_ksens.csv 2>/dev/null)
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
  [[ $DRY == 1 ]] && { log "dry-run: skipping the unit tests"; return 0; }
  (cd "$REPO" && $PY -m pytest -q op_tests/flydsl_tests/test_flydsl_allreduce_policy.py)
}

cmd_validate() {
  NGPU=$(ngpu); resolve_link; write_provenance
  plain_chunks "$OUT/validate" "val_" "$VAL_SET"
  finish
  log "validate done -> $OUT/validate   next: tune.sh audit"
}

cmd_audit() {
  if [[ $DRY == 1 ]]; then dry_fit_scripts fit_allreduce_policy.py; return; fi
  resolve_link; mkdir -p "$OUT/audit"; local rc=0 acc f=()
  while IFS= read -r x; do f+=("$x"); done < <(ls "$OUT"/validate/val_*_tp*.csv 2>/dev/null)
  [[ ${#f[@]} -gt 0 ]] || die "no validation CSVs (run: tune.sh validate)"
  for acc in fast exact; do
    log "auditing $acc"
    fit_py "${f[@]}" --link "$LINK" --audit-auto --accuracy "$acc" 2>&1 | tee "$OUT/audit/audit_$acc.txt"
    [[ ${PIPESTATUS[0]} == 0 ]] || rc=1
  done
  [[ $rc == 0 ]] && log "AUDIT PASS" || { log "AUDIT FAIL (see $OUT/audit)"; exit 1; }
}

cmd_report() {
  NGPU=$(ngpu); resolve_link; write_provenance
  local tp op
  plan $(($(count_tps) * 2 * 2))
  for tp in $TPS; do tp_ok "$tp" || continue
    for op in ar ar_rmsnorm; do
      bench_chunk "report_oneshot_tp${tp}_${op}" "$OUT/report" "$ONESHOT" --tp "$tp" --operation "$op"
      bench_chunk "report_quick_tp${tp}_${op}" "$OUT/report" "$QUICK" --tp "$tp" --operation "$op"
    done
  done
  finish
  log "report tables -> $OUT/report/*.md"
}

cmd_fused_sweep() {
  NGPU=$(ngpu); resolve_link; write_provenance
  log "fused sweep: candidate sets $FUSED_SET (one-shot) and $FUSED_QR_SET (mesh/ring), tp=[$TPS] widths=[$WIDTHS]"
  local tp h
  plan $(($(count_tps) * $(wc -w <<<"$WIDTHS") * 2))
  for tp in $TPS; do tp_ok "$tp" || continue
    for h in $WIDTHS; do
      # One chunk per test: each pinned row is an engine with its own inbox, so
      # this keeps the per-run inbox total down, and an older sweep directory
      # picks up the quantized rows without re-running the one-shot.
      bench_chunk "tp${tp}_w${h}" "$OUT/fused/sweep" "$ONESHOT" --operation ar_rmsnorm \
        --tp "$tp" --shape-csv "$FUSED_SHAPES/fused_w${h}.csv" \
        --candidates "$CANDS" --candidate-set "$FUSED_SET"
      bench_chunk "qr_tp${tp}_w${h}" "$OUT/fused/sweep" "$QUICK" --operation ar_rmsnorm \
        --tp "$tp" --shape-csv "$FUSED_SHAPES/fused_w${h}.csv" \
        --candidates "$CANDS" --candidate-set "$FUSED_QR_SET"
    done
  done
  finish
  log "fused sweep done -> $OUT/fused/sweep   next: tune.sh fused-fit"
}

cmd_fused_fit() {
  if [[ $DRY == 1 ]]; then dry_fit_scripts fused/analyze_fused.py fused/fit_fused.py; return; fi
  resolve_link; local d=$OUT/fused/sweep
  ls "$d"/tp*_w*.csv >/dev/null 2>&1 || die "no fused sweep CSVs under $d (run: tune.sh fused-sweep)"
  $PY "$HERE/fused/analyze_fused.py" "$d" --summary-csv "$OUT/fused/summary.csv" > "$OUT/fused/summary.txt" \
    || die "analyze_fused failed"
  $PY "$HERE/fused/fit_fused.py" "$d" --link "$LINK" --min-width-gain "$WIDTH_GAIN" \
    --out-json "$OUT/fused/fused_allreduce_ladders.json" 2>&1 | tee "$OUT/fused/fit.txt"
  [[ ${PIPESTATUS[0]} == 0 ]] || die "fused fit failed"
  log "fused fit -> $OUT/fused/fit.txt, ladder file $OUT/fused/fused_allreduce_ladders.json" \
    "(per-shape summary: $OUT/fused/summary.txt); README step 5"
}

# ll_ok <tp> <hidden>: whether the low-latency mesh build exists there. It is a
# 256-thread build, so a row is hidden/2048 atoms, which must divide a rank's
# 8/tp atoms of a tile (_low_latency_hidden_ok in the quick test).
ll_ok() {
  local apr=$(($2 / 2048))
  [[ $(($2 % 2048)) == 0 && $apr -ge 1 && $((8 / $1 % apr)) == 0 ]]
}

cmd_low_latency() {
  NGPU=$(ngpu); resolve_link; write_provenance
  local tp h p pairs=()
  for tp in $TPS; do tp_ok "$tp" || continue
    for h in $LL_WIDTHS; do
      if ll_ok "$tp" "$h"; then pairs+=("$tp:$h"); else log "skip tp$tp w$h: no low-latency build"; fi
    done
  done
  [[ ${#pairs[@]} -gt 0 ]] || die "no low-latency build at tp=[$TPS] widths=[$LL_WIDTHS]"
  log "low-latency sweep: candidate set $LL_SET, (tp:width) = ${pairs[*]}"
  plan ${#pairs[@]}
  # Only the low-latency rows and the shipping mesh (-a mesh): the ring and the
  # rest of the mesh grid are the fused sweep's business.
  for p in "${pairs[@]}"; do
    tp=${p%:*}; h=${p#*:}
    bench_chunk "ll_tp${tp}_w${h}" "$OUT/low_latency/sweep" "$QUICK" --operation ar_rmsnorm -a mesh \
      --tp "$tp" --shape-csv "$FUSED_SHAPES/fused_w${h}.csv" \
      --candidates "$CANDS" --candidate-set "$LL_SET"
  done
  finish
  if [[ $DRY == 1 ]]; then dry_fit_scripts fused/compare_low_latency.py; return; fi
  $PY "$HERE/fused/compare_low_latency.py" "$OUT/low_latency/sweep" --link "$LINK" \
    --csv "$OUT/low_latency/compare.csv" 2>&1 | tee "$OUT/low_latency/compare.txt"
  [[ ${PIPESTATUS[0]} == 0 ]] || die "compare_low_latency failed"
  log "low-latency comparison -> $OUT/low_latency/compare.txt (per shape: compare.csv)"
}

case $CMD in
  setup) cmd_setup ;;
  low-latency) cmd_low_latency ;;
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
       [[ $DRY == 1 ]] && { log "dry-run OK: nothing was benchmarked or fitted"; exit 0; }
       log "all fits done. Now paste the fitted values into allreduce_policy.py (README step 5), then: tune.sh validate && tune.sh audit" ;;
  *) die "unknown command '$CMD' (see --help)" ;;
esac
