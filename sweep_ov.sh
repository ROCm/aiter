#!/usr/bin/env bash
# Overlap tuning sweep: per_layer wall + fused GEMM1 per_call, one line per config.
cd /app/aiter || exit 1

BS=${BS:-64}; HD=${HD:-7168}; ID=${ID:-3072}; E=${E:-384}; K=${K:-6}; L=${L:-8}

# GPU0 on this box throttles to ~1440 MHz under sustained load while the other
# three stay near 2350, and a 4-rank collective runs at the speed of the slowest
# rank, so an unpaced sweep measures the cooling and not the change. Wait for the
# slowest card to come back before every run.
cool() {
  while true; do
    c=$(rocm-smi --showclocks 2>/dev/null \
        | grep -oP 'sclk clock level: \d+: \(\K[0-9]+' | sort -n | head -1)
    [ "${c:-0}" -ge 2200 ] && break
    sleep 10
  done
}

run() {
  tag="$1"; shift
  cool
  # cool() only proves the cards were fast at the start line. GPU0 is power
  # limited on this box and drops mid-run, and a 4-rank collective runs at the
  # slowest rank, so sample through the run and report the floor: a config is
  # only comparable against another that saw the same floor.
  clkfile=$(mktemp)
  ( while :; do
      rocm-smi --showclocks 2>/dev/null \
        | grep -oP 'sclk clock level: \d+: \(\K[0-9]+' | sort -n | head -1
      sleep 1
    done ) > "$clkfile" &
  sampler=$!
  out=$(env "$@" timeout 240 torchrun --standalone --nproc_per_node=4 \
      op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
      -bs $BS -hd $HD -id $ID -e $E -k $K --layers $L \
      --combine scatter_fused --push_group 1 --acc_verify 1 --profile_table 1 2>&1)
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  # Drop samples taken in the gaps between launches -- the cards idle at 500 MHz
  # there, and that floor says nothing about the clock the kernels actually saw.
  clk=$(awk '$1 > 1000' "$clkfile" | sort -n | head -1); rm -f "$clkfile"
  per=$(echo "$out"  | grep -oP 'per_layer=\K[0-9.]+' | head -1)
  # $3 is the mean over all 24 calls and $5 the fastest of them. A run that hits
  # the clock dip drags every mean in the table but leaves the best call alone,
  # so the min is the one column that compares a change instead of the cooling.
  g1=$(echo "$out"   | grep -P '^a8w4_tdm_fp8_\S+K'$HD  | awk '{print $5}')
  g1m=$(echo "$out"  | grep -P '^a8w4_tdm_fp8_\S+K'$HD  | awk '{print $3}')
  g2=$(echo "$out"   | grep -P '^a8w4_tdm_fp8_\S+K'$ID  | awk '{print $5}')
  dsp=$(echo "$out"  | grep -P '^ep_dispatch_0'         | awk '{print $5}')
  cmb=$(echo "$out"  | grep -P '^ep_combine_fused_0'    | awk '{print $5}')
  chk=$(echo "$out"  | grep -oP 'MEGA-CHECK[^(]*: \K\w+' | head -1)
  [ -z "$per" ] && per="FAIL/HANG"
  printf "%-28s clk=%-6s g1min=%-8s g12=%-8s g1avg=%-9s gemm2=%-8s comb=%-7s per_layer=%-9s %s\n" \
      "$tag" "${clk:--}" "${g1:--}" \
      "$(awk -v a="${g1:-0}" -v b="${g2:-0}" 'BEGIN{printf "%.1f", a+b}')" \
      "${g1m:--}" "${g2:--}" "${cmb:--}" "$per" "${chk:-?}"
}
