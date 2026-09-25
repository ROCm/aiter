#!/bin/bash
# Does the dense path still emit the same machine code as upstream?
#
# This is the claim the whole PR rests on: with block skipping off, nothing
# changes for anyone. "The conditional folds away at compile time" is an
# assertion about the compiler, so assert it against the compiler -- compile
# both trees and diff the AMDGCN.
#
# METHOD. Two COPIES of the aiter tree under /tmp: one as-is, one with the two
# touched files restored from upstream/main. Copies, because /workspace/aiter is
# a live bind mount of the host repo and patching it in place would edit the
# host. Each is run with PYTHONPATH pointing at it and TRITON_KERNEL_DUMP on,
# then the emitted assembly is normalised and compared.
#
# NORMALISATION. Our change adds ENABLE_BLOCK_SKIP and PRELOAD_V to the kernel's
# config-aware repr, so the kernel SYMBOL NAME legitimately differs. That is
# evidence the flags reached codegen, not a regression, so the name is
# normalised out before diffing. Debug line-table directives go too: they encode
# source line numbers, which of course moved.
set -u

SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-$(cd "$(dirname "$0")" && pwd)/results_codegen}
KERN=aiter/ops/triton/_triton_kernels/attention/unified_attention.py
WRAP=aiter/ops/triton/attention/unified_attention.py

mkdir -p "$OUT"
echo "node $(hostname)  $(date -Is)"
python3 -c 'import torch,triton;p=torch.cuda.get_device_properties(0);print("gpu",p.name,p.gcnArchName,"| triton",triton.__version__)'
echo "chain-dot: ${TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF:-<unset>}"
echo

rm -rf /tmp/tree_new /tmp/tree_old /tmp/dump_new /tmp/dump_old
cp -r "$SRC" /tmp/tree_new
cp -r "$SRC" /tmp/tree_old

# Restore the two touched files in the "old" tree from the upstream reference.
# Extracted on the HOST, not with `git show` here: deploy.sh's rsync excludes
# .git, so /workspace/aiter is a source tree with no repository attached.
REF=${REF:-$(cd "$(dirname "$0")" && pwd)/upstream_ref}
[ -f "$REF/kernel.py" ] && [ -f "$REF/wrapper.py" ] || {
  echo "FATAL: no upstream reference at $REF. Create it with:"
  echo "  mkdir -p $REF"
  echo "  git show <base-rev>:$KERN > $REF/kernel.py"
  echo "  git show <base-rev>:$WRAP > $REF/wrapper.py"
  exit 1
}
# Guard against a stale or mis-extracted reference silently making the two
# trees equal, which would turn this whole check into a tautology.
if grep -q "ENABLE_BLOCK_SKIP" "$REF/kernel.py"; then
  echo "FATAL: the upstream reference already contains ENABLE_BLOCK_SKIP."
  echo "       It was extracted from the wrong revision; re-extract it."
  exit 1
fi
cp "$REF/kernel.py" /tmp/tree_old/"$KERN"
cp "$REF/wrapper.py" /tmp/tree_old/"$WRAP"
echo "old tree: upstream reference $(cat "$REF/REV" 2>/dev/null | cut -c1-9)"
echo "new tree: the working branch as synced"
# Prove the reverse-patch actually did something -- a silent no-op here would
# make the whole comparison vacuous.
if diff -q /tmp/tree_old/"$KERN" /tmp/tree_new/"$KERN" >/dev/null; then
  echo "FATAL: old and new kernel sources are identical; nothing to compare."
  exit 1
fi
echo "sources differ as expected ($(diff /tmp/tree_old/"$KERN" /tmp/tree_new/"$KERN" | grep -c '^[<>]') changed lines)"
echo

# Same dense workload against each tree.
cat > /tmp/dense_once.py <<'PYEOF'
import os, sys, torch
sys.path.insert(0, os.environ["TREE"])
from aiter.ops.triton.attention.unified_attention import unified_attention
torch.manual_seed(0)
S, NQ, NKV, D, BS = 8192, 32, 8, 128, 16
nb = (S + BS - 1) // BS
dev = "cuda"
q = torch.randn(S, NQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(nb, BS, NKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(nb, BS, NKV, D, dtype=torch.bfloat16, device=dev)
unified_attention(
    q=q, k=k, v=v, out=torch.empty_like(q),
    cu_seqlens_q=torch.tensor([0, S], dtype=torch.int32, device=dev),
    max_seqlen_q=S,
    seqused_k=torch.tensor([S], dtype=torch.int32, device=dev),
    max_seqlen_k=S, softmax_scale=D**-0.5, causal=True, window_size=(-1, -1),
    block_table=torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0),
    softcap=0, q_descale=None, k_descale=None, v_descale=None, backend="triton",
)
torch.cuda.synchronize()
print("  dense launch ok")
PYEOF

for side in new old; do
  echo "--- compiling $side tree ---"
  TREE=/tmp/tree_$side \
  TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump_$side \
  TRITON_CACHE_DIR=/tmp/tcache_$side \
    python3 /tmp/dense_once.py || { echo "FATAL: $side tree failed to run"; exit 1; }
done
echo

python3 - "$OUT" <<'PYEOF'
import collections
import glob
import os
import re
import sys

out_dir = sys.argv[1]


def find_asm(dump):
    hits = []
    for f in glob.glob(os.path.join(dump, "**", "*.amdgcn"), recursive=True):
        if "unified_attention_2d" in open(f, errors="ignore").read(200000):
            hits.append(f)
    return hits


def normalise(text):
    """Drop everything expected to differ for reasons that are not the code.

    Three classes, each of which fooled an earlier version of this script:

    1. DEBUG INFO -- not just `.loc`/`.file`. The DWARF tree is emitted as
       `.byte` lines carrying DW_TAG / DW_CHILDREN / abbrev offsets, and those
       shift whenever source lines move. Filtering only `.loc` leaves them in
       and reports pure line-numbering churn as a codegen difference.
    2. BUILD PATHS -- the harness compiles from /tmp/tree_old and /tmp/tree_new
       and those strings are baked into the debug info, so the script would
       flag a difference it created itself.
    3. THE KERNEL SYMBOL -- its config-aware repr now includes
       ENABLE_BLOCK_SKIP and PRELOAD_V by design, so the name differs on
       purpose.
    """
    out, in_debug = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(".section"):
            in_debug = ".debug_" in s
        if in_debug:
            continue
        if s.startswith((".loc", ".file", ".Ltmp", ".cfi", ".size", ".type")):
            continue
        if "DW_" in s or ".debug_" in s:
            continue
        ln = re.sub(r"/tmp/tree_(old|new)", "/tmp/tree", ln)
        ln = re.sub(r"kernel_unified_attention_2d[0-9A-Za-z_]*", "KERNEL", ln)
        out.append(ln.rstrip())
    return out


# Adding a RUNTIME (non-constexpr) parameter necessarily grows the kernarg
# segment and shifts the offsets of every argument after it. That is a
# signature change, not generated code, so it is scored separately -- otherwise
# a known and quantified 8 bytes reads as though the kernel body had moved.
SIG = (".amdhsa_kernarg_size", ".kernarg_segment_size", ".offset:",
       ".value_kind:", ".size:")


def code_only(lines):
    return [ln for ln in lines
            if ln.strip()
            and not ln.strip().startswith((";", "//"))
            and not any(k in ln for k in SIG)]


def instr_hist(lines):
    h = collections.Counter()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith((".", ";", "/", "KERNEL")):
            continue
        h[s.split()[0]] += 1
    return h


new_files, old_files = find_asm("/tmp/dump_new"), find_asm("/tmp/dump_old")
print(f"asm files found: new={len(new_files)} old={len(old_files)}")
if not new_files or not old_files:
    print("FAIL: could not locate the 2D kernel assembly in one of the dumps")
    sys.exit(1)

n_all = normalise(open(max(new_files, key=os.path.getsize), errors="ignore").read())
o_all = normalise(open(max(old_files, key=os.path.getsize), errors="ignore").read())
open(os.path.join(out_dir, "new.norm.s"), "w").write("\n".join(n_all))
open(os.path.join(out_dir, "old.norm.s"), "w").write("\n".join(o_all))

n, o = code_only(n_all), code_only(o_all)
hn, ho = instr_hist(n_all), instr_hist(o_all)
print(f"code lines:   new={len(n)} old={len(o)}")
print(f"instructions: new={sum(hn.values())} old={sum(ho.values())}")

sig = sorted(x for x in (set(n_all) ^ set(o_all)) if any(k in x for k in SIG))
print(f"signature-directive differences: {len(sig)}")

print()
if n == o:
    print("RESULT: emitted CODE is IDENTICAL.")
    print("  Block skipping costs nothing when it is off: the dense path")
    print("  executes exactly the instructions upstream emits.")
    if sig:
        print()
        print("  The only differences are in the kernel SIGNATURE, because")
        print("  log2_threshold and num_q_blocks are runtime scalars rather")
        print("  than constexprs -- 8 bytes more kernarg segment and the")
        print("  offsets after them shift. No instruction loads either with")
        print("  skipping off, which is why the body and the register counts")
        print("  are unchanged:")
        for x in sig:
            print(f"    {x.strip()}")
    sys.exit(0)

print("RESULT: emitted CODE DIFFERS -- a real regression, not a signature change.")
diff_keys = {k for k in set(hn) | set(ho) if hn.get(k, 0) != ho.get(k, 0)}
if diff_keys:
    print("  instruction-count differences:")
    for k in sorted(diff_keys, key=lambda k: -abs(hn.get(k, 0) - ho.get(k, 0)))[:15]:
        print(f"    {k:<24} new={hn.get(k, 0):<6} old={ho.get(k, 0):<6} "
              f"delta={hn.get(k, 0) - ho.get(k, 0):+d}")
else:
    print("  instruction MIX is identical; ordering or operands differ.")
print(f"  normalised assembly written to {out_dir}/{{new,old}}.norm.s")
sys.exit(1)
PYEOF
rc=$?
echo "### codegen diff rc=$rc"
echo "CODEGEN_DIFF_DONE"
exit $rc
