#!/usr/bin/env bash
# Build the torch-free standalone UA trace driver (ua_trace_main.cpp) into an
# executable whose ONLY device code object is the one instance we trace -- every
# other UA instance is compiled as a -DUA_STUB_INSTANCE host stub (symbol exists
# so the runtime dispatch switch links, but no device kernel is emitted). The
# result: rocprofv3 ATT disassembles ~1 kernel instead of PyTorch's 26MB HIP lib
# (+610 UA kernels), so trace collection drops from ~2.5min to seconds.
#
# Env:
#   ARCH=gfx950   target arch (must match the GPU under test)
#   DTYPE=fp8     fp8|bf16 -- selects the real (non-stub) instance
#   D=128         head dim (d64|d128 nopage instances exist)
#   MASK=0        0=non-causal (nmask) | 2=causal (mask)
#   JOBS=32       parallel compile jobs
#   OUT=...       build dir (default: standalone/build)
#
# Output: $OUT/ua_trace  (run it, or trace it via standalone/trace.sh)
set -euo pipefail

ARCH="${ARCH:-gfx950}"
DTYPE="${DTYPE:-fp8}"; D="${D:-128}"; MASK="${MASK:-0}"
JOBS="${JOBS:-32}"

MASKTAG=$([[ "$MASK" == "0" ]] && echo "nmask" || echo "mask")
TARGET_INSTANCE="${TARGET_INSTANCE:-unified_attention_d${D}_${DTYPE}_${MASKTAG}_nopage}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(dirname "$HERE")"
AITER_ROOT="$(dirname "$SCRIPTS")"
CK="$AITER_ROOT/3rdparty/composable_kernel"
UADIR="$CK/example/ck_tile/42_unified_attention"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT/obj"

[[ -f "$UADIR/instances/${TARGET_INSTANCE}.cpp" ]] || {
    echo "target instance not found: $UADIR/instances/${TARGET_INSTANCE}.cpp" >&2
    echo "available nopage instances:" >&2
    ls "$UADIR/instances" | grep nopage >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Freshness guard: this is what makes the standalone a trustworthy perf path
# (no "is it stale?" ambiguity like the JIT module). A rebuild happens iff:
#   * REBUILD=1, or
#   * the exe is missing / built for a different (arch,dtype,d,mask), or
#   * ANY kernel source or header (ck_tile tree, the UA example dir, or the
#     standalone driver/build script) is newer than the exe.
# Otherwise we no-op in <1s, so runners can call build.sh unconditionally.
# ---------------------------------------------------------------------------
EXE="$OUT/ua_trace"
WANT_STAMP="$ARCH $DTYPE $D $MASK ${XCFLAGS:-}"
if [[ "${REBUILD:-0}" != "1" && -x "$EXE" && "$(cat "$OUT/.built" 2>/dev/null || true)" == "$WANT_STAMP" ]]; then
    newer="$(find "$CK/include/ck_tile" "$UADIR" -type f \
                  \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -newer "$EXE" -print -quit 2>/dev/null || true)"
    [[ -z "$newer" ]] && newer="$(find "$HERE" -maxdepth 1 -type f \
                  \( -name '*.cpp' -o -name '*.sh' \) -newer "$EXE" -print -quit 2>/dev/null || true)"
    if [[ -z "$newer" ]]; then
        echo "[build] up to date ($WANT_STAMP) -- $EXE"
        exit 0
    fi
    echo "[build] source newer than exe ($(basename "$newer")) -> rebuilding"
fi

# Codegen-relevant flags copied from the JIT build.ninja for
# module_unified_attention (see isa_source_map.sh), MINUS all torch includes
# (the example dispatcher + instances are torch-free) so the binary never loads
# libtorch. The -mllvm flags are kept so the ISA matches the deployed kernel.
# -gline-tables-only embeds DWARF line tables without changing -O3 codegen.
cat > "$OUT/flags.rsp" <<EOF
-DWITH_HIP
-I$CK/../ck_helper
-I$CK/include
-I$CK/library/include
-I$UADIR
-std=c++20
-O3
-Wno-unknown-warning-option
-DENABLE_CK=1
-DCK_TILE_USE_OCP_FP8=1
-D__HIP_PLATFORM_AMD__=1
-DUSE_ROCM=1
-D__HIP_PLATFORM_HCC__=1
-mcmodel=large
-fno-unique-section-names
-ffunction-sections
-fdata-sections
--offload-arch=$ARCH
-Wno-macro-redefined
-Wno-missing-template-arg-list-after-template-kw
-Wno-switch-bool
-Wno-undefined-func-template
-Wno-unused-result
-Wno-vla-cxx-extension
-fgpu-flush-denormals-to-zero
-fno-offload-uniform-block
-mllvm
--amdgpu-kernarg-preload-count=16
-mllvm
--lsr-drop-solution=1
-mllvm
-amdgpu-early-inline-all=true
-mllvm
-amdgpu-function-calls=false
-mllvm
-enable-post-misched=0
-fno-gpu-rdc
-gline-tables-only
EOF

# Optional extra -D toggles (e.g. XCFLAGS="-DUA_FP8_WIDE_MMA=0") for A/B probes.
# Newline-split into the response file so each token is its own arg. Folded into
# the build stamp so toggling XCFLAGS forces a rebuild.
if [[ -n "${XCFLAGS:-}" ]]; then
    for tok in $XCFLAGS; do printf '%s\n' "$tok" >> "$OUT/flags.rsp"; done
fi

# Build the source list: driver + dispatcher + every instance.
SRCS=( "$HERE/ua_trace_main.cpp" "$UADIR/unified_attention.cpp" )
while IFS= read -r f; do SRCS+=( "$f" ); done < <(ls "$UADIR"/instances/*.cpp)

echo "============================================================"
echo " standalone UA trace build"
echo "   arch=$ARCH  real instance=$TARGET_INSTANCE  (others stubbed)"
echo "   TUs=${#SRCS[@]}  jobs=$JOBS  out=$OUT/ua_trace"
echo "============================================================"

# Per-TU compile. Instances other than the target get -DUA_STUB_INSTANCE.
compile_one() {
    local src="$1" out_dir="$2" rsp="$3" target="$4"
    local base; base="$(basename "$src" .cpp)"
    local obj="$out_dir/obj/${base}.o"
    local stub=()
    if [[ "$src" == */instances/* && "$base" != "$target" ]]; then
        stub=(-DUA_STUB_INSTANCE)
    fi
    /opt/rocm/bin/hipcc "@${rsp}" "${stub[@]}" -c "$src" -o "$obj" 2>"$out_dir/obj/${base}.log" \
        || { echo "FAILED: $base (see $out_dir/obj/${base}.log)"; return 1; }
}
export -f compile_one

printf '%s\n' "${SRCS[@]}" | \
    xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} "$OUT" "$OUT/flags.rsp" "$TARGET_INSTANCE"

echo "[link] ua_trace ..."
/opt/rocm/bin/hipcc "@${OUT}/flags.rsp" "$OUT"/obj/*.o -o "$OUT/ua_trace"
# Stamp which instance this exe was built for, so run.sh/check.sh can detect a
# stale exe (every non-target instance is a stub -> wrong mask/dtype = "no
# matching kernel" at runtime).
echo "$WANT_STAMP" > "$OUT/.built"
echo "[ok] $OUT/ua_trace"
ls -lah "$OUT/ua_trace" | awk '{print "     size:", $5}'
