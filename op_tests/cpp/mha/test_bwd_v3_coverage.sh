#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# bwd_v3 coverage test driver. Run individual suites or `all`.
# See `--help` for the suite list.

set -u
# Note: deliberately NOT `set -e` — we want to count failing cases, not abort.

EXE="${EXE:-$(find . -name bwd.exe -type f | head -n 1)}"
COMMON_ARGS='-v=1 -kname=1'
CASE_TIMEOUT_S=120

# Per-suite counters; reset by reset_counters().
TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAIL_LOG=""

# Aggregate, set by main().
ANY_FAILED=0
ARCH=""

usage() {
    cat <<EOF
Usage: $0 [suite ...]
       $0 -l           list suites with arch + description
       $0 all          run every arch-compatible suite in declared order
       $0 --selftest   run internal unit tests of helpers (no GPU needed)

Examples:
  $0 batch group swa
  $0 bias dropout
  $0 all
EOF
}

reset_counters() {
    TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
    FAIL_LOG="bwd_v3_fail_${1}_$(date +%Y%m%d_%H%M%S)_$$.log"
}

detect_arch() {
    if [ -n "${ARCH_OVERRIDE:-}" ]; then
        ARCH="$ARCH_OVERRIDE"
        return
    fi
    ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || echo "unknown")
}

# usage: should_run_on_arch <space-separated globs> <arch>
# echoes "yes" or "no". Empty arch list => yes.
should_run_on_arch() {
    local globs="$1" arch="$2" g
    [ -z "$globs" ] && { echo yes; return; }
    for g in $globs; do
        case "$arch" in $g) echo yes; return ;; esac
    done
    echo no
}

# usage: skip_v3_constraint prec hdim sq sk mask v3_atomic_fp32 v3_bf16_cvt
# Returns 0 (true) if the case should be skipped due to known kernel restrictions.
# Mirrors the four guard clauses in the original smoke_test_bwd_v3.sh::run_batch_mode_tests.
skip_v3_constraint() {
    local prec=$1 hdim=$2 sq=$3 sk=$4 mask=$5 atom=$6 cvt=$7
    if [ "$atom" -eq 0 ] && { [ "$sq" -ne "$sk" ] || [ $((sk % 64)) -ne 0 ]; }; then
        return 0
    fi
    if [ "$hdim" -gt 128 ] && [ "$atom" -eq 0 ]; then
        return 0
    fi
    if [ "$prec" = "fp16" ] && [ "$cvt" -gt 0 ]; then
        return 0
    fi
    case "$mask" in
        b|2|b:*) [ "$atom" -eq 0 ] && return 0 ;;
    esac
    return 1
}

# usage: run_case <suite> <bwd.exe args...>
# Classifies based on stdout sentinels:
#   "valid:y"           -> pass
#   "valid:n"           -> fail (validator mismatch)
#   "not supported yet" -> skip
#   rc != 0             -> fail (crash)
#   rc == 124           -> fail (timeout)
run_case() {
    local suite="$1"; shift
    local cmd="$EXE $* $COMMON_ARGS"
    local out rc
    out=$(timeout "${CASE_TIMEOUT_S}" "$EXE" "$@" $COMMON_ARGS 2>&1)
    rc=$?
    TOTAL=$((TOTAL+1))
    if [ "$rc" -eq 124 ]; then
        FAILED=$((FAILED+1))
        printf '[TIMEOUT] %s\n%s\n----\n' "$cmd" "$out" >> "$FAIL_LOG"
    elif [ "$rc" -ne 0 ]; then
        FAILED=$((FAILED+1))
        printf '[CRASH rc=%d] %s\n%s\n----\n' "$rc" "$cmd" "$out" >> "$FAIL_LOG"
    elif printf '%s' "$out" | grep -q 'valid:n'; then
        FAILED=$((FAILED+1))
        printf '[MISMATCH] %s\n%s\n----\n' "$cmd" "$out" >> "$FAIL_LOG"
    elif printf '%s' "$out" | grep -q 'not supported yet'; then
        SKIPPED=$((SKIPPED+1))
    elif printf '%s' "$out" | grep -q 'valid:y'; then
        PASSED=$((PASSED+1))
    else
        # Unrecognized output — treat as fail so we don't silently miss bugs.
        FAILED=$((FAILED+1))
        printf '[UNKNOWN] %s\n%s\n----\n' "$cmd" "$out" >> "$FAIL_LOG"
    fi
    if [ $((TOTAL % 50)) -eq 0 ]; then
        printf '[%s] %d run, %d pass, %d fail, %d skip\n' \
            "$suite" "$TOTAL" "$PASSED" "$FAILED" "$SKIPPED"
    fi
}

# Single source of truth for suite ordering. Each entry MUST have a
# matching `suite_<name>` function and the SUITE_<name>_ARCHS / _DESC
# variables defined alongside it.
ALL_SUITES=(
    batch group swa
    gfx950_batch gfx950_group hd192_128
    bias dbias dropout deterministic layout gqa
    kernel_flags shape_edges hdim_v combined
)

list_suites() {
    local s archs desc
    printf '%-18s %-14s %s\n' SUITE ARCHS DESCRIPTION
    for s in "${ALL_SUITES[@]}"; do
        eval "archs=\${SUITE_${s}_ARCHS:-}"
        eval "desc=\${SUITE_${s}_DESC:-}"
        printf '%-18s %-14s %s\n' "$s" "${archs:-any}" "$desc"
    done
}

run_one_suite() {
    local s="$1" archs
    eval "archs=\${SUITE_${s}_ARCHS:-}"
    if [ "$(should_run_on_arch "$archs" "$ARCH")" = no ]; then
        printf '[skip] suite=%s (requires %s, have %s)\n' "$s" "$archs" "$ARCH"
        return 0
    fi
    if ! declare -F "suite_${s}" >/dev/null; then
        printf '[error] suite=%s has no suite_%s function\n' "$s" "$s" >&2
        return 1
    fi
    reset_counters "$s"
    printf '===== suite=%s arch=%s =====\n' "$s" "$ARCH"
    "suite_${s}"
    printf '[%s] DONE: %d run, %d pass, %d fail, %d skip\n' \
        "$s" "$TOTAL" "$PASSED" "$FAILED" "$SKIPPED"
    if [ "$FAILED" -gt 0 ]; then
        ANY_FAILED=1
        printf '[%s] fail log: %s\n' "$s" "$FAIL_LOG"
    else
        # Empty fail log; remove it so logs/ stays tidy.
        rm -f "$FAIL_LOG"
    fi
}

on_interrupt() {
    printf '\n[interrupted] partial summary follows\n'
    printf '[%s] %d run, %d pass, %d fail, %d skip; fail log=%s\n' \
        "${CURRENT_SUITE:-?}" "$TOTAL" "$PASSED" "$FAILED" "$SKIPPED" "$FAIL_LOG"
    exit 130
}
trap on_interrupt INT TERM

# --- Suite implementations ---

SUITE_batch_ARCHS=''
SUITE_batch_DESC='Batch-mode cross product (port of run_batch_mode_tests).'
suite_batch() {
    local prec perm hdim sq sk atom cvt mask
    for prec in fp16 bf16; do
    for perm in 0 1; do
    for hdim in 64 72 96 128 144 176 192; do
    for sq in 64 192 200; do
    for sk in 33 64 192; do
    for atom in 0 1; do
    for cvt in 0 1 2; do
    for mask in 0 t b; do
        if skip_v3_constraint "$prec" "$hdim" "$sq" "$sk" "$mask" "$atom" "$cvt"; then
            continue
        fi
        run_case batch -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=$sq -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -v3_bf16_cvt=$cvt -mode=0
    done; done; done; done; done; done; done; done
}

SUITE_group_ARCHS=''
SUITE_group_DESC='Group-mode cross product (port of run_group_mode_tests).'
suite_group() {
    local sk prec perm hdim mask cvt
    for sk in 63 127 200; do
    for prec in bf16 fp16; do
    for perm in 0 1; do
    for hdim in 64 80 96 120 128 144 160 192; do
    for mask in 0 t b; do
    for cvt in 0 1 2; do
        if [ "$prec" = "fp16" ] && [ "$cvt" -gt 0 ]; then continue; fi
        run_case group -prec=$prec -b=2 -h=3 -d=$hdim -s=65 -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_bf16_cvt=$cvt -v3_atomic_fp32=1 -mode=1
        run_case group -prec=$prec -b=1 -h=4 -h_k=1 -d=$hdim -s=129 -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_bf16_cvt=$cvt -v3_atomic_fp32=1 -mode=1
    done; done; done; done; done; done
}

SUITE_swa_ARCHS=''
SUITE_swa_DESC='Sliding-window attention masks (port of run_swa_tests).'
suite_swa() {
    local prec perm sq sk hdim mask
    for prec in bf16 fp16; do
    for perm in 0 1; do
    for sq in 192 301 512 700; do
    for sk in 192 301 512 700; do
    for hdim in 72 96 128; do
    for mask in 't:-1,10' 't:15,-1' 't:15,15' 't:190,187' \
                'b:-1,10' 'b:15,-1' 'b:15,15' 'b:190,187'; do
        run_case swa -prec=$prec -b=2 -h=3 -h_k=1 -d=$hdim -s=$sq -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -mode=0
        run_case swa -prec=$prec -b=2 -h=2       -d=$hdim -s=$sq -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -mode=0
    done; done; done; done; done; done
}

SUITE_gfx950_batch_ARCHS='gfx95?'
SUITE_gfx950_batch_DESC='gfx950 batch-mode (port of run_gfx950_bwd_v3).'
suite_gfx950_batch() {
    local prec mask atom hdim batch head sq sk perm hdim_v
    for prec in bf16 fp16; do
    for mask in 0 1 2; do
    for atom in 1 0; do
    for hdim in 72 112 128 192; do
    for batch in 3; do
    for head in 2 4; do
    for sq in 62 174; do
    for sk in 65 174 299 577; do
    for perm in 0 1; do
        hdim_v=$hdim
        if [ "$hdim" -eq 192 ]; then
            hdim_v=128
            if [ "$mask" -eq 2 ]; then continue; fi
        fi
        run_case gfx950_batch -prec=$prec -b=$batch -h=$head -h_k=2 \
            -d=$hdim -d_v=$hdim_v -s=$sq -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -mode=0
    done; done; done; done; done; done; done; done; done
}

SUITE_gfx950_group_ARCHS='gfx95?'
SUITE_gfx950_group_DESC='gfx950 group-mode (port of run_gfx950_group_bwd_v3).'
suite_gfx950_group() {
    local prec mask atom seqlen hdim perm
    for prec in bf16 fp16; do
    for mask in 0 1 2; do
    for atom in 0 1; do
    for seqlen in 65 174 299 577; do
    for hdim in 80 120 128; do
    for perm in 0 1; do
        run_case gfx950_group -prec=$prec -b=2 -h=3 \
            -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -mode=1
        run_case gfx950_group -prec=$prec -b=3 -h=4 -h_k=1 \
            -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -mode=1
    done; done; done; done; done; done
}

SUITE_hd192_128_ARCHS='gfx95?'
SUITE_hd192_128_DESC='hdim=192, hdim_v=128 cas_kb kernel coverage.'
suite_hd192_128() {
    local prec atom mask perm batch head sq sk
    local hdim=192 hdim_v=128
    for prec in bf16 fp16; do
    for atom in 0 1; do
    for mask in 0 1 2; do
    for perm in 0 1; do
    for batch in 1 2 3; do
    for head in 1 2 4; do
    for sq in 62 174 299 577; do
    for sk in 65 174 299 577; do
        run_case hd192_128 -prec=$prec -b=$batch -h=$head \
            -d=$hdim -d_v=$hdim_v -s=$sq -s_k=$sk \
            -iperm=$perm -operm=$perm -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -mode=0
    done; done; done; done; done; done; done; done
}

# --- Canonical shape table ---
# Three canonical shapes used by axis suites (b h h_k sq sk hdim).
# tile-aligned, off-by-one, prime-ish/MQA.
CANONICAL_SHAPES=(
    "2 4 2 128 128 64"
    "2 4 2 129 127 80"
    "1 3 1 173 211 96"
)

# --- New axis suites ---

SUITE_bias_ARCHS=''
SUITE_bias_DESC='Bias variants (none, elementwise 1/h/bh, alibi).'
suite_bias() {
    local shape b h h_k sq sk hdim prec mask bias atom
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 h=$2 h_k=$3 sq=$4 sk=$5 hdim=$6
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for bias in n e e:1 e:2 a a:1; do
        for atom in 0 1; do
            if skip_v3_constraint "$prec" "$hdim" "$sq" "$sk" "$mask" "$atom" 0; then
                continue
            fi
            run_case bias -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask -bias=$bias \
                -bwd_v3=1 -v3_atomic_fp32=$atom -mode=0 \
                -init=1 -seed=11939
        done; done; done; done
    done
}

SUITE_dbias_ARCHS=''
SUITE_dbias_DESC='Bias-gradient output (-dbias=1) for elementwise bias variants.'
suite_dbias() {
    local shape b h h_k sq sk hdim prec mask bias
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 h=$2 h_k=$3 sq=$4 sk=$5 hdim=$6
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for bias in e e:1 e:2; do
            run_case dbias -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask -bias=$bias -dbias=1 \
                -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
                -init=1 -seed=11939
        done; done; done
    done
}

SUITE_dropout_ARCHS=''
SUITE_dropout_DESC='Dropout (p_drop > 0) across drop_prefs, prec, mask.'
suite_dropout() {
    local shape b h h_k sq sk hdim prec mask p prefs
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 h=$2 h_k=$3 sq=$4 sk=$5 hdim=$6
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for p in 0.1 0.5 0.9; do
        for prefs in 0 1; do
            run_case dropout -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask \
                -p_drop=$p -drop_seed=1 -drop_offset=0 -drop_prefs=$prefs \
                -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
                -init=1 -seed=11939
        done; done; done; done
    done
}

SUITE_deterministic_ARCHS=''
SUITE_deterministic_DESC='Deterministic (multi-buffer) reduction strategy.'
suite_deterministic() {
    local shape b h h_k sq sk prec mask hdim atom
    # Canonical hdim ($6) is intentionally ignored — this suite sweeps hdim explicitly.
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 h=$2 h_k=$3 sq=$4 sk=$5
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for hdim in 64 128 192; do
        for atom in 0 1; do
            if skip_v3_constraint "$prec" "$hdim" "$sq" "$sk" "$mask" "$atom" 0; then
                continue
            fi
            run_case deterministic -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask -deterministic=1 \
                -bwd_v3=1 -v3_atomic_fp32=$atom -mode=0 \
                -init=1 -seed=11939
        done; done; done; done
    done
}

SUITE_layout_ARCHS=''
SUITE_layout_DESC='ilayout x olayout x prec x mask, including sbhd (=2).'
suite_layout() {
    local shape b h h_k sq sk hdim prec mask il ol
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 h=$2 h_k=$3 sq=$4 sk=$5 hdim=$6
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for il in 0 1 2; do
        for ol in 0 1 2; do
            run_case layout -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask \
                -ilayout=$il -olayout=$ol \
                -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
                -init=1 -seed=11939
        done; done; done; done
    done
}

SUITE_gqa_ARCHS=''
SUITE_gqa_DESC='Many h/h_k ratios incl. MHA, MQA, irregular GQA.'
suite_gqa() {
    local shape b sq sk hdim prec mask pair h h_k
    # Reuse only sq/sk/hdim from canonical shapes; override h/h_k from the GQA pair list.
    for shape in "${CANONICAL_SHAPES[@]}"; do
        set -- $shape; b=$1 sq=$4 sk=$5 hdim=$6
        for prec in fp16 bf16; do
        for mask in 0 t b; do
        for pair in "1 1" "2 1" "4 1" "4 2" "8 1" "8 2" "8 4" "12 3"; do
            set -- $pair; h=$1 h_k=$2
            run_case gqa -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
                -s=$sq -s_k=$sk -mask=$mask \
                -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
                -init=1 -seed=11939
        done; done; done
    done
}

SUITE_kernel_flags_ARCHS=''
SUITE_kernel_flags_DESC='v3_atomic_fp32 x v3_bf16_cvt, single shape (validation-pass only).'
suite_kernel_flags() {
    local prec mask atom cvt
    # One fixed shape: tile-aligned canonical.
    local b=2 h=4 h_k=2 sq=128 sk=128 hdim=64
    # Note: v3_api_check is a probe knob (returns 1.0f without launching the
    # kernel); validating its zeroed output is meaningless by design, so the
    # apick axis is intentionally pinned to 0 here. Probe semantics belong in
    # a dedicated smoke test, not in the validation-pass coverage suite.
    for prec in fp16 bf16; do
    for mask in 0 t b; do
    for atom in 0 1; do
    for cvt in 0 1 2; do
        if skip_v3_constraint "$prec" "$hdim" "$sq" "$sk" "$mask" "$atom" "$cvt"; then
            continue
        fi
        run_case kernel_flags -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=$atom -v3_bf16_cvt=$cvt \
            -v3_api_check=0 -mode=0 \
            -init=1 -seed=11939
    done; done; done; done
}

SUITE_shape_edges_ARCHS=''
SUITE_shape_edges_DESC='Tile-boundary seqlens, tiny/extreme shapes.'
suite_shape_edges() {
    local prec mask hdim sq sk
    # Tile-boundary seqlens (off-by-one around 64/128/192/256) plus tiny and primes.
    local seqlens=(1 2 63 64 65 127 128 129 191 192 193 257)
    for prec in fp16 bf16; do
    for mask in 0 t b; do
    for hdim in 64 128; do
    for sq in "${seqlens[@]}"; do
    for sk in "${seqlens[@]}"; do
        # Use atomic_fp32=1 so we don't need sq==sk and sk%64==0.
        run_case shape_edges -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
            -init=1 -seed=11939
    done; done; done; done; done

    # b=1 h=1 corner.
    for prec in fp16 bf16; do
    for mask in 0 t b; do
        run_case shape_edges -prec=$prec -b=1 -h=1 -h_k=1 -d=64 \
            -s=128 -s_k=128 -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
            -init=1 -seed=11939
    done; done
}

SUITE_hdim_v_ARCHS=''
SUITE_hdim_v_DESC='hdim != hdim_v configurations beyond 192/128.'
suite_hdim_v() {
    local prec mask pair d dv
    for prec in fp16 bf16; do
    for mask in 0 t b; do
    for pair in "64 64" "128 64" "96 72" "192 128" "176 128"; do
        set -- $pair; d=$1 dv=$2
        run_case hdim_v -prec=$prec -b=2 -h=4 -h_k=2 \
            -d=$d -d_v=$dv -s=128 -s_k=128 -mask=$mask \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 \
            -init=1 -seed=11939
    done; done; done
}

SUITE_combined_ARCHS=''
SUITE_combined_DESC='Pairwise interactions: bias x dropout, swa x deterministic, sbhd x GQA, etc.'
suite_combined() {
    local prec mask
    local b=2 h=4 h_k=2 sq=128 sk=128 hdim=64

    # Pair 1: bias x dropout
    for prec in fp16 bf16; do for mask in 0 t b; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -bias=e -p_drop=0.5 \
            -drop_seed=1 -drop_offset=0 -drop_prefs=0 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 2: bias x swa
    for prec in fp16 bf16; do for mask in 't:15,15' 'b:15,15'; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -bias=e \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 3: dropout x swa
    for prec in fp16 bf16; do for mask in 't:15,15' 'b:15,15'; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -p_drop=0.3 \
            -drop_seed=1 -drop_offset=0 -drop_prefs=0 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 4: swa x deterministic
    for prec in fp16 bf16; do for mask in 't:15,15' 'b:15,15'; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -deterministic=1 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 5: sbhd x GQA (irregular ratio)
    for prec in fp16 bf16; do for mask in 0 t b; do
        run_case combined -prec=$prec -b=$b -h=8 -h_k=2 -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -ilayout=2 -olayout=2 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 6: dbias x deterministic
    for prec in fp16 bf16; do for mask in 0 t b; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -bias=e -dbias=1 -deterministic=1 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done

    # Pair 7: dropout x deterministic
    for prec in fp16 bf16; do for mask in 0 t b; do
        run_case combined -prec=$prec -b=$b -h=$h -h_k=$h_k -d=$hdim \
            -s=$sq -s_k=$sk -mask=$mask -p_drop=0.3 \
            -drop_seed=1 -drop_offset=0 -drop_prefs=0 -deterministic=1 \
            -bwd_v3=1 -v3_atomic_fp32=1 -mode=0 -init=1 -seed=11939
    done; done
}

# --- Suite stubs; replaced incrementally by later tasks ---
for _stub in "${ALL_SUITES[@]}"; do
    declare -F "suite_${_stub}" >/dev/null && continue
    eval "suite_${_stub}() { echo '[${_stub}] not yet implemented'; }"
    eval "SUITE_${_stub}_ARCHS=''"
    eval "SUITE_${_stub}_DESC='(stub)'"
done
unset _stub

selftest_run_case() {
    # Build a stub EXE that emits a chosen output and exit code.
    local tmpdir; tmpdir=$(mktemp -d)
    trap "rm -rf '$tmpdir'" RETURN

    _selftest_make_stub() {
        # $1 = stdout text, $2 = exit code, $3 = optional sleep seconds
        local _text="$1" _rc="$2" _sleep="${3:-}"
        cat > "$tmpdir/stub" <<STUB
#!/bin/sh
echo "${_text}"
[ -n "${_sleep}" ] && sleep ${_sleep}
exit ${_rc}
STUB
        chmod +x "$tmpdir/stub"
        EXE="$tmpdir/stub"
    }

    local rc=0

    # Case 1: pass
    reset_counters "selftest_pass"
    _selftest_make_stub ", 0.029 ms, 2.89 TFlops, valid:y" 0
    run_case "selftest_pass" -prec=fp16
    [ "$PASSED" -eq 1 ] && [ "$FAILED" -eq 0 ] && [ "$SKIPPED" -eq 0 ] || \
        { echo "FAIL: pass classification ($PASSED/$FAILED/$SKIPPED)"; rc=1; }

    # Case 2: mismatch
    reset_counters "selftest_mismatch"
    _selftest_make_stub ", 0.029 ms, 2.89 TFlops, valid:n" 0
    run_case "selftest_mismatch" -prec=fp16
    [ "$FAILED" -eq 1 ] && [ "$PASSED" -eq 0 ] || \
        { echo "FAIL: mismatch classification"; rc=1; }
    grep -q '\[MISMATCH\]' "$FAIL_LOG" || { echo "FAIL: mismatch tag missing"; rc=1; }

    # Case 3: not supported -> skip
    reset_counters "selftest_skip"
    _selftest_make_stub "not supported yet" 0
    run_case "selftest_skip" -prec=fp16
    [ "$SKIPPED" -eq 1 ] && [ "$FAILED" -eq 0 ] || \
        { echo "FAIL: skip classification"; rc=1; }

    # Case 4: crash
    reset_counters "selftest_crash"
    _selftest_make_stub "segfault-ish" 139
    run_case "selftest_crash" -prec=fp16
    [ "$FAILED" -eq 1 ] || { echo "FAIL: crash classification"; rc=1; }
    grep -q '\[CRASH rc=139\]' "$FAIL_LOG" || { echo "FAIL: crash tag missing"; rc=1; }

    # Case 5: unknown output
    reset_counters "selftest_unknown"
    _selftest_make_stub "weird unrecognized output" 0
    run_case "selftest_unknown" -prec=fp16
    [ "$FAILED" -eq 1 ] || { echo "FAIL: unknown classification"; rc=1; }
    grep -q '\[UNKNOWN\]' "$FAIL_LOG" || { echo "FAIL: unknown tag missing"; rc=1; }

    # Cleanup fail logs created by selftest
    rm -f bwd_v3_fail_selftest_*.log

    unset -f _selftest_make_stub
    return $rc
}

selftest_arch() {
    local rc=0
    [ "$(should_run_on_arch ''            'gfx942')" = yes ] || { echo "FAIL: empty arch list should match"; rc=1; }
    [ "$(should_run_on_arch 'gfx95?'      'gfx950')" = yes ] || { echo "FAIL: gfx95? vs gfx950"; rc=1; }
    [ "$(should_run_on_arch 'gfx95?'      'gfx942')" = no  ] || { echo "FAIL: gfx95? vs gfx942"; rc=1; }
    [ "$(should_run_on_arch 'gfx94? gfx95?' 'gfx942')" = yes ] || { echo "FAIL: multi glob"; rc=1; }
    return $rc
}

selftest_constraints() {
    local rc=0
    # atomic16 + sq!=sk -> skip
    skip_v3_constraint fp16 64 128 192 0 0 0 || { echo "FAIL: atomic16 sq!=sk should skip"; rc=1; }
    # atom=0 (no-atomic-fp32) but sq==sk and sk%64==0 -> should run (not skip)
    skip_v3_constraint fp16 64 128 128 0 0 0 && { echo "FAIL: atom=0 aligned should run"; rc=1; }
    # atomic16 + hdim>128 -> skip
    skip_v3_constraint fp16 192 128 128 0 0 0 || { echo "FAIL: atomic16 hdim>128 should skip"; rc=1; }
    # fp16 + cvt>0 -> skip
    skip_v3_constraint fp16 64 128 128 0 1 1 || { echo "FAIL: fp16+cvt should skip"; rc=1; }
    # bf16 + cvt>0 -> run
    skip_v3_constraint bf16 64 128 128 0 1 1 && { echo "FAIL: bf16+cvt should run"; rc=1; }
    # atomic16 + bottom-r mask -> skip
    skip_v3_constraint fp16 64 128 128 b 0 0 || { echo "FAIL: atomic16 bottom-r should skip"; rc=1; }
    # atomic_fp32=1 + bottom-r -> run
    skip_v3_constraint fp16 64 128 128 b 1 0 && { echo "FAIL: atomic_fp32 bottom-r should run"; rc=1; }
    return $rc
}

run_selftest() {
    local rc=0
    selftest_run_case    || rc=1
    selftest_arch        || rc=1
    selftest_constraints || rc=1
    if [ "$rc" -eq 0 ]; then echo "selftest: OK"; else echo "selftest: FAIL"; fi
    return $rc
}

main() {
    if [ "$#" -eq 0 ]; then usage; exit 0; fi
    case "$1" in
        --help|-h) usage; exit 0 ;;
        -l|--list)
            detect_arch
            list_suites
            exit 0 ;;
        --selftest)
            run_selftest; exit $? ;;
    esac

    detect_arch
    if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then
        printf 'error: bwd.exe not found (set EXE= or run from a build tree)\n' >&2
        exit 2
    fi

    local requested=("$@")
    if [ "${requested[0]}" = "all" ]; then
        requested=("${ALL_SUITES[@]}")
    fi

    for s in "${requested[@]}"; do
        # Validate name
        local found=0 known
        for known in "${ALL_SUITES[@]}"; do
            [ "$s" = "$known" ] && { found=1; break; }
        done
        if [ "$found" -eq 0 ]; then
            printf 'error: unknown suite "%s" (try -l)\n' "$s" >&2
            exit 2
        fi
        CURRENT_SUITE="$s"
        run_one_suite "$s"
    done
    exit "$ANY_FAILED"
}

main "$@"
