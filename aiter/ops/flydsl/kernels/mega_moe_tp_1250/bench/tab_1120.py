# tab_1120.py <model> <mode> <base.log> <mega.log> [out.csv]: speedup table (+ csv rows appended to out.csv)
import os, re, sys
model, mode, fb, fg = sys.argv[1:5]
rd = lambda f: open(f).read() if os.path.exists(f) else ""
tb, tg = rd(fb), rd(fg)
b = {}
for m in re.finditer(r"T=\s*(\d+) moe=\s*([\d.]+)(.*?)(?:moe\+ar|ag\+moe\+rs)=\s*([\d.]+) \(([^)]*)\)", tb):
    r = re.search(r"rccl_tot=([\d.]+)", m.group(3))
    b[int(m.group(1))] = (float(m.group(4)), float(m.group(2)), float(r.group(1)) if r else float("nan"), m.group(5))
g = {int(m.group(1)): (float(m.group(4)), m.group(2), m.group(3)) for m in
     re.finditer(r"T=\s*(\d+) err=(\d+).*?logits_diff=(\S+) us=([\d.]+)", tg)}
print("tokens | baseline | (moe only) | (rccl-only) | mega | speedup | err | logits_diff | comm")
rows = []
for t in sorted(set(b) | set(g)):
    bb, gg = b.get(t), g.get(t)
    sp = bb[0] / gg[0] if bb and gg else float("nan")
    wrong = "  <<< WRONG RESULT" if gg and float(gg[2]) > 1e-3 else ""
    print(f"{t:6d} | {bb[0] if bb else float('nan'):8.1f} | {bb[1] if bb else float('nan'):8.1f} | "
          f"{bb[2] if bb else float('nan'):8.1f} | {gg[0] if gg else float('nan'):7.1f} | {sp:5.2f} | "
          f"{gg[1] if gg else '-'} | {gg[2] if gg else '-'} | {bb[3] if bb else '-'}{wrong}")
    rows.append(f"{model},{mode},{t},{bb[0] if bb else ''},{bb[1] if bb else ''},{bb[2] if bb else ''},"
                f"{gg[0] if gg else ''},{sp:.2f},{gg[1] if gg else ''},{gg[2] if gg else ''},{bb[3] if bb else ''}")
if len(sys.argv) > 5:
    new = not os.path.exists(sys.argv[5])
    with open(sys.argv[5], "a") as f:
        if new:
            f.write("model,mode,tokens,baseline_us,baseline_moe_us,baseline_rccl_only_us,mega_us,speedup,err,logits_diff,baseline_comm\n")
        f.write("\n".join(rows) + "\n")
