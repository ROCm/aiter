# gen_fixed_csv.py <out.csv>: the merged tuned grouped-fmoe table of the working tree (= upstream main),
# like aiter.jit.core's merge (default + model_configs/*tuned_grouped_fmoe*.csv, duplicates by the untuned
# key columns -> lowest us), without rewriting any source file.
import glob, sys
import pandas as pd
R = "/home/binding2/moe_1026/aiter/aiter/configs"
files = [f"{R}/tuned_grouped_fmoe.csv"] + [p for p in sorted(glob.glob(f"{R}/model_configs/*tuned_grouped_fmoe*.csv")) if "untuned" not in p]
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in files]
m = pd.concat(dfs, ignore_index=True).fillna("")
keys = pd.read_csv(f"{R}/untuned_grouped_fmoe.csv").columns.to_list()
if "cu_num" not in keys: keys.append("cu_num")
if "gfx" in m.columns and "gfx" not in keys: keys.append("gfx")
keys = [k for k in keys if k in m.columns]
def num(c):
    v = pd.to_numeric(c, errors="coerce"); w = c.astype(str).str.strip() != ""
    return v if v[w].notna().all() else c
comp = m.apply(num)
dup = comp.duplicated(subset=keys, keep=False)
ranked = comp.assign(us=pd.to_numeric(m["us"], errors="coerce")).sort_values("us", kind="stable")
best = set(ranked.drop_duplicates(subset=keys, keep="first").index)
out = m[m.index.isin(best)]
out.to_csv(sys.argv[1], index=False)
print(f"{len(files)} files, {len(m)} rows, {int(dup.sum())} in duplicate groups -> {len(out)} rows; keys={keys}")
