#!/bin/bash
# ISA Round-Trip: disassemble -> extract .s -> recompile -> verify
#
# Demonstrates the full workflow from the isa_kernel_optimization.md guide.
# Uses a PA kernel from AITER as a concrete example.
#
# Usage:
#   ./roundtrip.sh <kernel.co> [--mcpu gfx942]
#
# The script will:
#   1. Disassemble the .co to ISA text
#   2. List available kernel symbols
#   3. Extract reassemblable .s for each symbol
#   4. Recompile to a new .co
#   5. Verify .text section is binary-identical
#   6. Produce a loadable .co via llvm-objcopy --update-section

set -euo pipefail

LLVM_BIN="${ROCM_PATH:-/opt/rocm}/llvm/bin"
OBJDUMP="$LLVM_BIN/llvm-objdump"
OBJCOPY="$LLVM_BIN/llvm-objcopy"
CLANGXX="$LLVM_BIN/clang++"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Parse args ----------
CO_FILE=""
MCPU="gfx942"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mcpu) MCPU="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 <kernel.co> [--mcpu gfx942]"
            exit 0 ;;
        *) CO_FILE="$1"; shift ;;
    esac
done

if [[ -z "$CO_FILE" ]]; then
    # Default: find a PA kernel from AITER
    AITER_PATH=$(python3 -c "import aiter; print(aiter.__path__[0])" 2>/dev/null || true)
    if [[ -n "$AITER_PATH" ]]; then
        CO_FILE=$(find "$AITER_PATH" -path "*/hsa/${MCPU}/pa/*.co" | head -1)
    fi
    if [[ -z "$CO_FILE" ]]; then
        echo "Error: no .co file specified and no AITER PA kernel found for $MCPU"
        echo "Usage: $0 <kernel.co> [--mcpu gfx942]"
        exit 1
    fi
    echo "Using AITER kernel: $CO_FILE"
fi

if [[ ! -f "$CO_FILE" ]]; then
    echo "Error: $CO_FILE not found"
    exit 1
fi

WORKDIR=$(mktemp -d -t isa_roundtrip.XXXXXX)
echo "Working directory: $WORKDIR"
echo "Target architecture: $MCPU"
echo

# ---------- Step 1: Disassemble ----------
echo "=== Step 1: Disassemble ==="
ISA_FILE="$WORKDIR/kernel.isa"
"$OBJDUMP" -d --mcpu="$MCPU" "$CO_FILE" > "$ISA_FILE"
TOTAL_INSTR=$(grep -cE '^\s+[0-9a-f]+:' "$ISA_FILE" || true)
echo "Total instructions: $TOTAL_INSTR"

# Quick instruction stats
echo "  MFMA ops:     $(grep -c 'v_mfma_' "$ISA_FILE" || true)"
echo "  Buffer loads: $(grep -c 'buffer_load' "$ISA_FILE" || true)"
echo "  LDS ops:      $(grep -c 'ds_' "$ISA_FILE" || true)"
echo "  DPP ops:      $(grep -c '_dpp' "$ISA_FILE" || true)"
echo

# ---------- Step 2: List symbols ----------
echo "=== Step 2: Kernel symbols ==="
SYMBOLS=$(grep -oP '(?<=<).+?(?=>:)' "$ISA_FILE" || true)
FIRST_SYMBOL=$(echo "$SYMBOLS" | head -1)
echo "$SYMBOLS" | while read -r sym; do
    echo "  $sym"
done
echo

if [[ -z "$FIRST_SYMBOL" ]]; then
    echo "Error: no kernel symbols found"
    exit 1
fi

# ---------- Step 3: Extract .s ----------
echo "=== Step 3: Extract reassemblable .s ==="
TARGET="amdgcn-amd-amdhsa--${MCPU}"
S_FILE="$WORKDIR/kernel.s"
python3 "$SCRIPT_DIR/extract_asm.py" "$ISA_FILE" "$FIRST_SYMBOL" \
    --target "$TARGET" -o "$S_FILE"
echo

# ---------- Step 4: Recompile ----------
echo "=== Step 4: Recompile ==="
RECOMP_CO="$WORKDIR/kernel_recompiled.co"
"$CLANGXX" -x assembler -target amdgcn-amd-amdhsa \
    -mcpu="$MCPU" -o "$RECOMP_CO" "$S_FILE"
echo "Recompiled: $RECOMP_CO"
echo

# ---------- Step 5: Verify .text ----------
echo "=== Step 5: Verify .text section ==="
ORIG_TEXT="$WORKDIR/original_text.bin"
RECOMP_TEXT="$WORKDIR/recompiled_text.bin"
"$OBJCOPY" -O binary -j .text "$CO_FILE" "$ORIG_TEXT"
"$OBJCOPY" -O binary -j .text "$RECOMP_CO" "$RECOMP_TEXT"

ORIG_MD5=$(md5sum "$ORIG_TEXT" | cut -d' ' -f1)
RECOMP_MD5=$(md5sum "$RECOMP_TEXT" | cut -d' ' -f1)
echo "Original .text:   $ORIG_MD5 ($(wc -c < "$ORIG_TEXT") bytes)"
echo "Recompiled .text: $RECOMP_MD5 ($(wc -c < "$RECOMP_TEXT") bytes)"

if [[ "$ORIG_MD5" == "$RECOMP_MD5" ]]; then
    echo "PASS: .text sections are binary-identical"
else
    echo "FAIL: .text sections differ"
    echo "  Run: diff <(xxd $ORIG_TEXT) <(xxd $RECOMP_TEXT) | head -20"
    exit 1
fi
echo

# ---------- Step 6: Produce loadable .co ----------
echo "=== Step 6: Produce loadable .co ==="
LOADABLE_CO="$WORKDIR/kernel_modified.co"
cp "$CO_FILE" "$LOADABLE_CO"
"$OBJCOPY" --update-section .text="$RECOMP_TEXT" "$LOADABLE_CO"
echo "Loadable kernel: $LOADABLE_CO"
echo "  Original size:  $(wc -c < "$CO_FILE") bytes"
echo "  Modified size:  $(wc -c < "$LOADABLE_CO") bytes"
echo

echo "=== Round-trip complete ==="
echo
echo "Files in $WORKDIR:"
ls -la "$WORKDIR"
echo
echo "Next steps:"
echo "  1. Edit $S_FILE to modify ISA instructions"
echo "  2. Recompile: $CLANGXX -x assembler -target amdgcn-amd-amdhsa -mcpu=$MCPU -o new.co $S_FILE"
echo "  3. Inject: cp $CO_FILE modified.co && $OBJCOPY --update-section .text=new_text.bin modified.co"
echo "  4. Benchmark the modified kernel against the original"
