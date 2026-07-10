"""Final decode sweep with seq-fastest grid (matched to Triton 3d)."""
import sys
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B

rows = []
for sh in B.DECODE_SHAPES:
    r = B.run_decode(sh)
    rows.append(r)
    print(f"{r['shape']:18s} S{r['S']:<3d} triton {r['t_tot']:7.1f}us {r['t_gb']:5.0f}GB | "
          f"gluon {r['g_tot']:7.1f}us {r['g_gb']:5.0f}GB | {r['speedup']:.2f}x  xc {r['xcheck']:.0e}")

L = ["# Decode: Triton 3d vs gfx950 Gluon (seq-fastest grid, matched)\n",
     "- bf16, HEAD_SIZE=128, causal; split-KV + Triton reduce_segments (identical reduce).",
     "- gluon grid order now matches Triton 3d (seq/q-block fastest). 512MB flush, torch.profiler.",
     "- gluon MFMA 32x32 (BLOCK_M=32/nw1); Triton BLOCK_M=16/nw2 (MFMA-forced diff).",
     "- speedup = triton_total / gluon_total (>1 = gluon faster).\n",
     "| shape | S | Triton us | Triton GB/s | Gluon us | Gluon GB/s | speedup | xcheck |",
     "|---|--:|--:|--:|--:|--:|--:|--:|"]
for r in rows:
    L.append(f"| {r['shape']} | {r['S']} | {r['t_tot']:.1f} | {r['t_gb']:.0f} | "
             f"{r['g_tot']:.1f} | {r['g_gb']:.0f} | **{r['speedup']:.2f}x** | {r['xcheck']:.0e} |")
open("/app/aiter/bench_gluon_ua/decode_final.md", "w").write("\n".join(L) + "\n")
print("wrote decode_final.md")
