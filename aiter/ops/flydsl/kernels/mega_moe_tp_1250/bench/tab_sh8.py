# tab_sh8.py <mode ar|ag_rs> <base_sep.log> <base_fuse.log> <mega.log> [out.csv]
# M3.1 A8W8-shared table: baseline B1 (sep) / B2 (fuse) vs mega (sh8); appends csv rows to out.csv
import os, re, sys
mode, fs, ff, fg = sys.argv[1:5]
rd = lambda f: open(f, errors="ignore").read() if os.path.exists(f) else ""
key = r"moe\+ar" if mode == "ar" else r"ag\+moe\+rs"
def base(txt):
    return {int(m.group(1)): (float(m.group(3)), float(m.group(2)))
            for m in re.finditer(r"T=\s*(\d+) moe=\s*([\d.]+).*?%s=\s*([\d.]+)" % key, txt)}
b1, b2 = base(rd(fs)), base(rd(ff))
g = {int(m.group(1)): (float(m.group(4)), m.group(2), m.group(3)) for m in
     re.finditer(r"T=\s*(\d+) err=(\d+).*?logits_diff=(\S+) us=([\d.]+)", rd(fg))}
print(f"mode={mode}")
print("tokens | B1 sep (MoE) | B2 fuse (MoE) | mega sh8 | vs B1 | vs B2 | err | logits_diff")
rows = []
for t in sorted(set(b1) | set(b2) | set(g)):
    s, f, m = b1.get(t), b2.get(t), g.get(t)
    x1 = f"{s[0] / m[0]:.2f}" if s and m else ""
    x2 = f"{f[0] / m[0]:.2f}" if f and m else ""
    bad = "  <<< WRONG RESULT" if m and float(m[2]) > 1e-3 else ""
    print(f"{t:6d} | {s[0] if s else float('nan'):7.1f} ({s[1] if s else float('nan'):6.1f}) | "
          f"{f[0] if f else float('nan'):7.1f} ({f[1] if f else float('nan'):6.1f}) | "
          f"{m[0] if m else float('nan'):8.1f} | {x1:>5} | {x2:>5} | {m[1] if m else '-'} | {m[2] if m else '-'}{bad}")
    rows.append(f"{mode},{t},{s[0] if s else ''},{s[1] if s else ''},{f[0] if f else ''},{f[1] if f else ''},"
                f"{m[0] if m else ''},{x1},{x2},{m[1] if m else ''},{m[2] if m else ''}")
if len(sys.argv) > 5:
    new = not os.path.exists(sys.argv[5])
    with open(sys.argv[5], "a") as o:
        if new:
            o.write("mode,tokens,b1_sep_us,b1_sep_moe_us,b2_fuse_us,b2_fuse_moe_us,mega_sh8_us,"
                    "speedup_vs_b1,speedup_vs_b2,err,logits_diff\n")
        o.write("\n".join(rows) + "\n")
