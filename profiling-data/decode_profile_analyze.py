"""Untracked: summarize decode_prof rocprofv3 outputs into per-case tables."""
import subprocess, json, glob, sys
R = '/tmp/ua_repro/decode_prof'
KV = lambda b, ctx=32768: b * ctx * 16 * 256 * 2   # K+V fp8 bytes (window 1024 not applied to reads? report full)
def kern(name):
    if name == 'decode_0': return 'fly_split'
    if name == 'combine_0': return 'fly_combine'
    if name.startswith('kernel_unified_attention'): return 'triton'
    return None
def q(sql):
    o = subprocess.run(['duckdb','-json','-c',sql],capture_output=True,text=True).stdout
    return json.loads(o) if o.strip() else []
for case in ['B64_p32', 'B64_p64', 'B16_p64', 'B16_p32']:
    for be in ['flydsl', 'triton']:
        tag = f'{be}_{case}'
        print(f'=== {tag}')
        t = [tuple(d.values()) for d in q(f"select Kernel_Name,Dispatch_Id,End_Timestamp-Start_Timestamp ns,Grid_Size_X gx,Grid_Size_Y gy,Grid_Size_Z gz,Workgroup_Size_X wx,VGPR_Count v,Accum_VGPR_Count a,SGPR_Count s,LDS_Block_Size lds,Scratch_Size scr from read_csv_auto('{R}/{tag}/trace/kt_kernel_trace.csv') order by Dispatch_Id")]
        by = {}
        for r in t:
            k = kern(r[0])
            if k: by.setdefault(k, []).append(r)
        for k, rs in by.items():
            rs = rs[-10:]
            ns = sorted(x[2] for x in rs)
            print(f'{k}: n={len(rs)} med_ns={ns[len(ns)//2]} mean_ns={sum(ns)/len(ns):.0f} min={ns[0]} grid={rs[0][3:6]} wg={rs[0][6]} vgpr={rs[0][7]} agpr={rs[0][8]} sgpr={rs[0][9]} lds={rs[0][10]} scratch={rs[0][11]}')
            if k == 'triton': print('  name:', rs[0][0][:400])
        # counters
        vals = {}
        for f in sorted(glob.glob(f'{R}/{tag}/g*/pmc_counter_collection.csv')):
            rows = [tuple(d.values()) for d in q(f"select Kernel_Name,Dispatch_Id,Counter_Name,sum(Counter_Value) from read_csv_auto('{f}') where Kernel_Name in ('decode_0','combine_0') or Kernel_Name like 'kernel_unified_attention%' group by all")]
            per = {}
            for kn, d, c, v in rows:
                k = kern(kn)
                if k: per.setdefault((k, c), []).append((d, v))
            for key, l in per.items():
                l.sort(); l = l[-10:]
                vals[key] = sum(v for _, v in l) / len(l)
        ks = sorted({k for k, _ in vals})
        cs = sorted({c for _, c in vals})
        for c in cs:
            print(f'  {c}: ' + ', '.join(f'{k}={vals.get((k,c), float("nan")):.6g}' for k in ks))
