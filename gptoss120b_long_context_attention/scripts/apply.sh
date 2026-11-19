#!/bin/bash
# Install (or restore) the gpt-oss prefill-attention changes into the server's aiter tree.
# The change is 4 tuning-table entries in the gfx1250 gluon unified_attention DEFAULT.json, used only for a
# long prefill chunk of >= 9216 tokens (head dim <= 64; 8K/1K prompts are <= 8192, so 1K/1K and 8K/1K keep the original configs):
#   D_LEQ_64.Q_GEQ_9216.DT_fp8_fp8     / .SW.DT_fp8_fp8     <- fp8 KV + fp8 query (--kv-cache-dtype fp8_e4m3,
#                                                               SGLANG_6003_PREFILL_Q=static: the TP1 launch)
#   D_LEQ_64.Q_GEQ_9216.DT_bf16_bf16   / .SW.DT_bf16_bf16   <- bf16 KV
# The fp8 full-attention entry selects LOOP_VARIANT 3 + NUM_BUFFERS 3 (LOOP_VARIANT 3 is new in unified_attention_2d.py: variant 2 with the fp8
# P scale folded into the exponent; ~17% faster per call than variant 2, identical error); the others keep variants 2 / 0.
# Plus two launcher guards.  No env variable, no setup.py / JIT rebuild; Triton compiles the Gluon kernel on
# first use.  Restart the SGLang server afterwards.
# unified_attention_utils.py is installed as the ORIGINAL file (it undoes an earlier env-variable version).
# SGLang: currently NO change (the MIXED-batch decode split, debug/aiter_backend_mixed_split.py.txt, was dropped:
# ~0.5% end to end).  apply.sh still installs $BASE/sglang/aiter_backend.py into the sglang package python imports,
# only if that file is still the original this change was made from (else nothing is touched).
# Files are read from $BASE/aiter/ and $BASE/sglang/ (each changed file next to its *_origin copy).
#   bash apply.sh            # install the changed files
#   bash apply.sh --restore  # put the original files back
set -euo pipefail

AITER_REPO="${AITER_REPO:-$HOME/draftbase_1111/aiter}"
BASE="${BASE:-/dockerx/gpt_oss/opt/attention}"
# the package sits one level down in the repo (aiter/aiter/ops/...); accept either layout
if [ -d "$AITER_REPO/aiter/ops" ]; then PKG="$AITER_REPO/aiter"; else PKG="$AITER_REPO"; fi

if [ "${1:-}" = "--restore" ]; then SRC="$BASE/aiter"; SUF="_origin"; MODE=restore
else SRC="$BASE/aiter"; SUF=""; MODE=install; fi

# <file in SRC (without suffix)>  <destination inside the aiter package>
FILES=(
  "unified_attention.py                         ops/triton/attention/unified_attention.py"
  "unified_attention_utils.py                   ops/triton/utils/unified_attention_utils.py"
  "unified_attention_gfx1250_gluon_DEFAULT.json ops/triton/configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json"
  "unified_attention_2d_gfx1250.py              ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py"
)
NEW_KEYS="D_LEQ_64.Q_GEQ_9216.DT_fp8_fp8,D_LEQ_64.Q_GEQ_9216.SW.DT_fp8_fp8,D_LEQ_64.Q_GEQ_9216.DT_bf16_bf16,D_LEQ_64.Q_GEQ_9216.SW.DT_bf16_bf16"

# 1) check aiter the way the launch scripts load it: they run
#      export PYTHONPATH="$HOME/draftbase_1111/aiter${PYTHONPATH:+:$PYTHONPATH}"
#    before "python3 -m sglang.launch_server", so mimic that (from /root, outside any repo folder that
#    could shadow the package).  If an SGLang server is running, also report the aiter its PYTHONPATH selects.
CHECK_DIR=/root; [ -d "$CHECK_DIR" ] || CHECK_DIR=/tmp
export PYTHONPATH="$AITER_REPO${PYTHONPATH:+:$PYTHONPATH}"
IMPORTED=$(cd "$CHECK_DIR" && python3 -c "import os, aiter; print(os.path.dirname(os.path.abspath(aiter.__file__)))" 2>/dev/null || true)
if [ -z "$IMPORTED" ]; then
  echo "ERROR: 'import aiter' failed with PYTHONPATH=$AITER_REPO; nothing copied"
  echo "       run: cd $CHECK_DIR && PYTHONPATH=$AITER_REPO python3 -c 'import aiter'   to see why"
  exit 1
elif [ "$(realpath "$IMPORTED")" != "$(realpath "$PKG")" ]; then
  echo "ERROR: with PYTHONPATH=$AITER_REPO first, python still imports aiter from $IMPORTED, not $PKG; nothing copied"
  exit 1
fi
echo "with PYTHONPATH=$AITER_REPO (as in the launch script), python imports aiter from $IMPORTED  (ok)"
SERVER_PID=$(pgrep -f "sglang.launch_server" 2>/dev/null | head -1 || true)
if [ -n "$SERVER_PID" ]; then
  SPP=$(tr '\0' '\n' < "/proc/$SERVER_PID/environ" 2>/dev/null | sed -n 's/^PYTHONPATH=//p')
  case ":$SPP:" in
    *":$AITER_REPO:"*|*":$AITER_REPO/:"*) echo "running server (pid $SERVER_PID) has $AITER_REPO on its PYTHONPATH (ok); restart it to pick up the change" ;;
    *) echo "WARNING: running server (pid $SERVER_PID) PYTHONPATH='$SPP' does not include $AITER_REPO" ;;
  esac
fi

# 1b) the sglang package python imports with the same PYTHONPATH, and its aiter_backend.py must be the
#     original this change was made from (or already the changed one); check BEFORE copying anything
SGL_PKG=$(cd "$CHECK_DIR" && python3 -c "import os, sglang; print(os.path.dirname(os.path.abspath(sglang.__file__)))" 2>/dev/null | tail -1 || true)
[ -n "$SGL_PKG" ] || { echo "ERROR: 'import sglang' failed with PYTHONPATH=$PYTHONPATH; nothing copied"; exit 1; }
SGL_DEST="$SGL_PKG/srt/layers/attention/aiter_backend.py"
SGL_ORIG="$BASE/sglang/aiter_backend_origin.py"
SGL_NEW="$BASE/sglang/aiter_backend.py"
for f in "$SGL_ORIG" "$SGL_NEW" "$SGL_DEST"; do [ -f "$f" ] || { echo "MISSING $f; nothing copied"; exit 1; }; done
# accepted: the original, the changed file, or a file carrying our (dropped) MIXED-batch split
if ! cmp -s "$SGL_DEST" "$SGL_ORIG" && ! cmp -s "$SGL_DEST" "$SGL_NEW" \
   && ! grep -q "def _mixed_decode_split" "$SGL_DEST"; then
  echo "ERROR: $SGL_DEST is neither the original ($SGL_ORIG) nor the changed file: a different sglang"
  echo "       version is installed there, so the change would not apply cleanly; nothing copied"
  exit 1
fi
echo "python imports sglang from $SGL_PKG  (aiter_backend.py matches the expected version: ok)"

# 2) copy
#    sglang: when $SGL_NEW equals the original there is no sglang change (the MIXED-batch decode split was
#    dropped: ~0.5% end to end); the original is (re)installed, which also removes an installed split.
if [ "$MODE" = restore ] || cmp -s "$SGL_NEW" "$SGL_ORIG"; then SGL_SRC="$SGL_ORIG"; else SGL_SRC="$SGL_NEW"; fi
if cmp -s "$SGL_SRC" "$SGL_DEST"; then
  echo "sglang   $SGL_DEST already up to date ($(basename "$SGL_SRC"))"
else
  cp "$SGL_SRC" "$SGL_DEST"; echo "$MODE  $SGL_SRC -> $SGL_DEST"
  python3 -m py_compile "$SGL_DEST" || { echo "ERROR: $SGL_DEST does not compile"; exit 1; }
fi
HAS_SPLIT=$(grep -c "def _mixed_decode_split" "$SGL_DEST" || true)
echo "  sglang MIXED-batch decode split: $([ "$HAS_SPLIT" -gt 0 ] && echo present || echo "absent (original aiter_backend.py)")"
for entry in "${FILES[@]}"; do
  read -r name dest <<<"$entry"
  src="$SRC/${name%.*}$SUF.${name##*.}"          # foo.py -> foo_origin.py, foo.json -> foo_origin.json
  [ -f "$src" ] || { echo "MISSING $src"; exit 1; }
  cp "$src" "$PKG/$dest"
  echo "$MODE  $src -> $PKG/$dest"
done

# 3) smoke test: aiter imports, the config table it actually loads has (or lacks) the new entries, what
#    the gpt-oss prefill will resolve to, and no env-variable override is left behind
cd "$CHECK_DIR" && python3 - "$MODE" "$NEW_KEYS" <<'PY'
import sys
import torch
from aiter.ops.triton.utils import unified_attention_utils as u
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.attention import unified_attention as ua  # noqa: F401
mode, keys = sys.argv[1], sys.argv[2].split(",")
table, axes, cfg_dir = u._load("attn_2d", "gluon", "gfx1250")
print(f"import ok; config table {cfg_dir}/DEFAULT.json")
present = [k in table for k in keys]
for k, p in zip(keys, present):
    print(f"  {k:40s} {'present' if p else 'absent'}")
# what gpt-oss TP1 prefill attention resolves to (head 64, page 64, 4 seqs, ~62k KV)
for label, dt in (("fp8 KV + fp8 q (TP1 launch)", e4m3_dtype), ("bf16 KV", torch.bfloat16)):
    for q, what in ((16384, "70k 16k chunk"), (16381, "70k MIXED chunk"), (12000, "70k split chunk"), (1024, "1K/1K prefill"), (8192, "8K/1K prefill")):
        for sw in (0, 127):
            key, cfg = u._lookup(table, axes, u._axis_values(64, q, 65536, sw, False, 64, dt, dt, 4))
            print(f"  {label:28s} {what:16s} {'sw  ' if sw else 'full'} -> {key:38s} "
                  f"BLOCK_M={cfg['BLOCK_M']} LOOP_VARIANT={cfg['LOOP_VARIANT']}")
if hasattr(u, "ua2d_prefill_override"):
    print("ERROR: the old env-variable version of unified_attention_utils.py is still installed"); sys.exit(1)
import aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d as k2d
has_lv3 = hasattr(k2d, "attention_loop_fp8_folded")
print(f"  kernel {k2d.__file__}: LOOP_VARIANT 3 (attention_loop_fp8_folded) {'present' if has_lv3 else 'absent'}")
want = mode == "install"
ok = all(p == want for p in present) and has_lv3 == want
if want and table.get(keys[0], {}).get("LOOP_VARIANT") == 3 and not has_lv3:
    print("ERROR: the table selects LOOP_VARIANT 3 but the kernel file lacks it")
sys.exit(0 if ok else 1)
PY

echo "done ($MODE). Restart the SGLang server (no env variable needed)."
