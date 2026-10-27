#!/bin/bash
# ISA round-trip: .co -> standalone .s -> reassemble -> verify against the original
#
# Demonstrates the workflow from docs/isa_kernel_optimization.md:
#   1. Show what is in the kernel object (kernels, labels, descriptor, metadata)
#   2. Rebuild a self-contained .s (text + kernel descriptor + metadata)
#   3. Assemble and link it into a new .co with clang
#   4. Verify .text, kernel descriptor (.rodata), metadata and ELF flags are
#      identical to the original
#
# Usage:
#   bash roundtrip.sh [kernel.co] [--mcpu gfx942] [--keep DIR]
#
# With no kernel given, the first paged-attention kernel of the installed AITER
# package is used.  Set ROCM_PATH if ROCm is not in /opt/rocm.

set -euo pipefail

LLVM_BIN="${ROCM_PATH:-/opt/rocm}/llvm/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CO_FILE=""
MCPU=""
WORKDIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mcpu) MCPU="$2"; shift 2 ;;
        --keep) WORKDIR="$2"; shift 2 ;;
        --help|-h) sed -n '2,15p' "$0"; exit 0 ;;
        *) CO_FILE="$1"; shift ;;
    esac
done

for t in llvm-objdump llvm-objcopy llvm-readelf clang; do
    [[ -x "$LLVM_BIN/$t" ]] || { echo "error: $LLVM_BIN/$t not found (set ROCM_PATH)"; exit 1; }
done

if [[ -z "$CO_FILE" ]]; then
    ASM_DIR=$(python3 -c "import aiter, os; print(os.path.join(aiter.__path__[0], '..', 'hsa'))" 2>/dev/null || true)
    ASM_DIR="${AITER_ASM_DIR:-$ASM_DIR}"
    CO_FILE=$(find "$ASM_DIR" -path "*/${MCPU:-gfx942}/pa/*.co" 2>/dev/null | sort | head -1 || true)
    if [[ -z "$CO_FILE" ]]; then
        echo "error: no .co given and no AITER PA kernel found (looked in '$ASM_DIR')"
        echo "usage: $0 <kernel.co> [--mcpu gfx942]"
        exit 1
    fi
    echo "Using AITER kernel: $CO_FILE"
fi
[[ -f "$CO_FILE" ]] || { echo "error: $CO_FILE not found"; exit 1; }

if [[ -z "$MCPU" ]]; then
    MCPU=$(python3 "$SCRIPT_DIR/extract_asm.py" "$CO_FILE" --llvm-bin "$LLVM_BIN" --list | awk '/^arch:/{print $2}')
fi
[[ -n "$WORKDIR" ]] && mkdir -p "$WORKDIR" || WORKDIR=$(mktemp -d -t isa_roundtrip.XXXXXX)

echo "Kernel object : $CO_FILE"
echo "Architecture  : $MCPU"
echo "LLVM tools    : $LLVM_BIN ($("$LLVM_BIN/llvm-objdump" --version | grep -m1 -i 'version' | sed 's/^ *//'))"
echo "Work directory: $WORKDIR"
echo

# ---------- Step 1: what is inside ----------
echo "=== Step 1: Inspect ==="
"$LLVM_BIN/llvm-objdump" -d --mcpu="$MCPU" "$CO_FILE" > "$WORKDIR/kernel.isa"
python3 "$SCRIPT_DIR/extract_asm.py" "$CO_FILE" --mcpu "$MCPU" --llvm-bin "$LLVM_BIN" --list | sed 's/^/  /'
# AMDGPU objdump lines look like "<TAB>instr operands   // ADDR: ENCODING"
count() { grep -cP "^\t$1.*//" "$WORKDIR/kernel.isa" || true; }
echo "  instructions : $(count '\S')"
echo "    MFMA       : $(count 'v_(mfma|smfma)')"
echo "    VMEM buffer: $(count 'buffer_')   global/flat: $(count '(global|flat|scratch)_')"
echo "    LDS (ds_*) : $(count 'ds_')"
echo "    SMEM       : $(count 's_(buffer_)?load')"
echo "    branches   : $(count 's_(cbranch|branch|setpc|swappc)')"
echo "    s_waitcnt  : $(count 's_waitcnt')   s_nop: $(count 's_nop')   s_barrier: $(count 's_barrier')"
echo "  kernel descriptor / metadata:"
"$LLVM_BIN/llvm-objdump" -D -j .rodata --mcpu="$MCPU" "$CO_FILE" \
    | grep -E 'amdhsa_(group_segment_fixed_size|next_free_vgpr|next_free_sgpr|accum_offset|ieee_mode|dx10_clamp|float_denorm)' \
    | sed 's/^[[:space:]]*/    /'
"$LLVM_BIN/llvm-readelf" --notes "$CO_FILE" \
    | grep -E '^\s+\.(vgpr_count|sgpr_count|kernarg_segment_size|wavefront_size|max_flat_workgroup_size):' \
    | sed 's/^[[:space:]]*/    /'
echo

# ---------- Step 2: extract ----------
echo "=== Step 2: Extract standalone .s (text + kernel descriptor + metadata) ==="
S_FILE="$WORKDIR/kernel.s"
python3 "$SCRIPT_DIR/extract_asm.py" "$CO_FILE" --mcpu "$MCPU" --llvm-bin "$LLVM_BIN" -o "$S_FILE"
echo

# ---------- Step 3: reassemble + link ----------
echo "=== Step 3: Reassemble ==="
NEW_CO="$WORKDIR/kernel_rebuilt.co"
"$LLVM_BIN/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu="$MCPU" -o "$NEW_CO" "$S_FILE"
echo "  $NEW_CO ($(wc -c < "$NEW_CO") bytes; original $(wc -c < "$CO_FILE") bytes)"
echo

# ---------- Step 4: verify ----------
echo "=== Step 4: Verify against the original ==="
status=0
for sec in .text .rodata; do
    "$LLVM_BIN/llvm-objcopy" -O binary -j "$sec" "$CO_FILE" "$WORKDIR/orig$sec.bin"
    "$LLVM_BIN/llvm-objcopy" -O binary -j "$sec" "$NEW_CO"  "$WORKDIR/new$sec.bin"
    label="$sec"; [[ "$sec" == ".rodata" ]] && label=".rodata (kernel descriptor)"
    if cmp -s "$WORKDIR/orig$sec.bin" "$WORKDIR/new$sec.bin"; then
        printf '  %-28s identical (%s bytes)\n' "$label" "$(wc -c < "$WORKDIR/orig$sec.bin")"
    else
        printf '  %-28s DIFFERS\n' "$label"
        status=1
        if [[ "$sec" == ".text" ]]; then
            # Map the first differing byte back to an instruction
            off=$( (cmp -l "$WORKDIR/orig.text.bin" "$WORKDIR/new.text.bin" || true) | awk 'NR==1{print $1-1}')
            base=$(grep -m1 -oP '^[0-9a-f]+(?= <)' "$WORKDIR/kernel.isa")
            addr=$(printf '%012X' $(( (0x$base + off) & ~3 )))
            echo "    first difference at .text+$off (address 0x$addr):"
            echo "    original : $(grep -m1 "// $addr" "$WORKDIR/kernel.isa" | sed 's/^[[:space:]]*//')"
            echo "    rebuilt  : $("$LLVM_BIN/llvm-objdump" -d --mcpu="$MCPU" "$NEW_CO" | grep -m1 "// $addr" | sed 's/^[[:space:]]*//')"
            echo "    ($( (cmp -l "$WORKDIR/orig.text.bin" "$WORKDIR/new.text.bin" || true) | wc -l) differing bytes in total; see the guide's"
            echo "     'Known re-encoding differences' section before assuming the rebuild is wrong)"
        else
            diff <(xxd "$WORKDIR/orig$sec.bin") <(xxd "$WORKDIR/new$sec.bin") | sed 's/^/    /' || true
        fi
    fi
done
if diff <("$LLVM_BIN/llvm-readelf" --notes "$CO_FILE" | tail -n +4) \
        <("$LLVM_BIN/llvm-readelf" --notes "$NEW_CO"  | tail -n +4) > "$WORKDIR/metadata.diff"; then
    printf '  %-28s identical\n' "metadata (.note)"
else
    printf '  %-28s DIFFERS (see %s)\n' "metadata (.note)" "$WORKDIR/metadata.diff"; status=1
fi
of=$("$LLVM_BIN/llvm-readelf" -h "$CO_FILE" | awk '/Flags:/{print $2}')
nf=$("$LLVM_BIN/llvm-readelf" -h "$NEW_CO"  | awk '/Flags:/{print $2}')
if [[ "$of" == "$nf" ]]; then
    printf '  %-28s identical (%s)\n' "ELF e_flags (target id)" "${of%,}"
else
    printf '  %-28s DIFFERS (%s vs %s)\n' "ELF e_flags (target id)" "${of%,}" "${nf%,}"; status=1
fi
echo

if [[ $status -eq 0 ]]; then
    echo "PASS: $NEW_CO is equivalent to the original and can replace it as-is."
else
    echo "CHECK: the rebuilt kernel is not byte-identical; inspect the differences above."
fi
cat <<EOF

Next steps:
  1. Edit $S_FILE
  2. Rebuild:  $LLVM_BIN/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=$MCPU -o my_kernel.co $S_FILE
  3. Test:     replace the original file with my_kernel.co (in a develop install that is the
               checkout's hsa/ tree -- git restores it) and run the family's op test, e.g.
               python3 op_tests/test_pa.py
               (the "[aiter] LoadKernel: ... hsaco: <path>" log line confirms which file was loaded)
  4. Profile:  rocprofv3 --kernel-trace --stats --kernel-include-regex '<kernel name>' -- python3 ...
EOF
exit $status
