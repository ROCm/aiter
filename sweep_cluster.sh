#!/bin/bash
# Sweep AITER_A4W4_QUAD_CLUSTER_M x _N and collect the gemm1 (K7168) mode
# distribution over many process launches. The mode is fixed per process, so
# relaunching is the only way to see modality. gemm2 (K3072) is the control.
#
# Illegal cluster shapes are rejected BEFORE launching, from the three
# conditions that make a cluster unformable (a cluster that cannot form hangs
# rather than failing):
#   1. co-residency -- a cluster lives in one shader engine, and this kernel
#      takes a whole CU's LDS, so size <= (enabled CUs per SE) = 16 here.
#      Asked of the topology, not hardcoded; see max_cluster_workgroups().
#   2. grid.y % cluster_n == 0 -- else the last cluster column is partial. The
#      driver silently falls back to the non-quad kernel instead, which would
#      quietly contaminate the sweep, so it is checked here too. gemm1 has 24
#      N-tiles and gemm2 28, so cluster_n must divide gcd(24,28)=4.
#   3. grid.x % cluster_m == 0 -- structural now (the m_run split works in
#      whole clusters), so it is asserted rather than filtered.
# The timeout stays as a backstop for anything these three do not cover.
#
# Usage: bash sweep_cluster.sh "M N M N ..." [launches]
set -u
LAUNCHES=${2:-6}
CONFIGS="${1:-4 4}"
TLIMIT=${TLIMIT:-180}
ITERS=${ITERS:-128}

legal() {  # legal M N -> prints "ok" or the reason it cannot form
  python3 - "$1" "$2" <<'PY'
import sys
m, n = int(sys.argv[1]), int(sys.argv[2])
from aiter.ops.flydsl.kernels.gemm_a4w4_moe_gfx1250 import max_cluster_workgroups
cap = max_cluster_workgroups(327680)
if m < 1 or n < 1:
    print(f"non-positive {m}x{n}")
elif m * n > cap:
    print(f"size {m*n} > {cap} co-resident workgroups per shader engine")
elif 24 % n or 28 % n:
    bad = [t for t in (24, 28) if t % n]
    print(f"cluster_n={n} does not divide N-tiles {bad} (would fall back to non-quad)")
else:
    print("ok")
PY
}

run_one() {
  timeout -k 15 "$TLIMIT" env \
  AITER_A4W4_QUAD_CLUSTER_M=$1 AITER_A4W4_QUAD_CLUSTER_N=$2 \
  AITER_A4W4_LOG_GRID=1 \
  AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 FLYDSL_DUMP_IR=0 \
  AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario kernel --data-format a4w4 --experts 96 --tokens 16384 --topk 6 \
    --model-dim 7168 --inter-dim 3072 --act silu --no-bias --iters "$ITERS" \
    --no-check-aot-cache --const-init 0 >/tmp/sweep_out.log 2>&1
  rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then printf "__HANG__"; return; fi
  # A fallback to the non-quad kernel is a silent confound, so name the kernel
  # that actually ran rather than assuming it was the quad one.
  if ! grep -q "a4w4_quad.*K7168" /tmp/sweep_out.log; then printf "__FALLBACK__"; return; fi
  grep -E "^0 +a4w4_quad" /tmp/sweep_out.log | tr -s ' ' \
  | awk '{if ($2 ~ /K7168/) g1=$6; if ($2 ~ /K3072/) g2=$6} END{printf "%s %s", g1, g2}'
}

set -- $CONFIGS
while [ $# -ge 2 ]; do
  M=$1; N=$2; shift 2
  why=$(legal "$M" "$N")
  if [ "$why" != "ok" ]; then
    printf "=== cluster %sx%s (size %s) === SKIPPED: %s\n" "$M" "$N" "$((M*N))" "$why"
    continue
  fi
  printf "=== cluster %sx%s (size %s) ===\n" "$M" "$N" "$((M*N))"
  g1list=""; g2list=""
  for i in $(seq 1 "$LAUNCHES"); do
    out=$(run_one "$M" "$N")
    case "$out" in
      __HANG__)     printf "  launch %2d: HANG (killed after %ss)\n" "$i" "$TLIMIT"; continue ;;
      __FALLBACK__) printf "  launch %2d: fell back off the quad kernel\n" "$i"; continue ;;
    esac
    g1=$(echo "$out" | awk '{print $1}'); g2=$(echo "$out" | awk '{print $2}')
    if [ -z "$g1" ]; then
      printf "  launch %2d: FAILED (%s)\n" "$i" "$(grep -iE 'error|assert|exception' /tmp/sweep_out.log | tail -1 | cut -c1-100)"
    else
      [ "$i" = 1 ] && printf "  %s\n" "$(grep -m1 'a4w4-grid.*K7168' /tmp/sweep_out.log)"
      printf "  launch %2d: gemm1=%-8s gemm2=%s\n" "$i" "$g1" "$g2"
      g1list="$g1list $g1"; g2list="$g2list $g2"
    fi
  done
  for lbl in gemm1 gemm2; do
    [ "$lbl" = gemm1 ] && lst="$g1list" || lst="$g2list"
    [ -n "$lst" ] && echo "$lst" | tr ' ' '\n' | grep . | tr -d ',' | sort -n \
      | awk -v l="$lbl" '{v[NR]=$1; s+=$1} END{if(NR){printf "  %s: n=%d min=%.1f max=%.1f mean=%.1f  sorted:", l, NR, v[1], v[NR], s/NR; for(i=1;i<=NR;i++)printf " %.0f", v[i]; print ""}}'
  done
done
