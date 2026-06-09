"""Combine two split PMC dirs (L2 hit/miss; DRAM reqs) for the jdbba kernel.
Usage: parse_pmc_split.py <tag> <dir_a> <dir_b>"""
import sys, sqlite3, glob, os

tag, da, dbdir = sys.argv[1], sys.argv[2], sys.argv[3]


def sums(outdir):
    dbs = glob.glob(os.path.join(outdir, "**", "*.db"), recursive=True)
    if not dbs:
        return {}
    c = sqlite3.connect(dbs[0])
    names = {n for (n,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def T(b):
        return next((n for n in names if n.startswith(b)), None)

    pmc, info, disp, sym = T("rocpd_pmc_event"), T("rocpd_info_pmc"), T("rocpd_kernel_dispatch"), T("rocpd_info_kernel_symbol")
    q = f"""SELECT i.name, SUM(p.value) FROM {pmc} p
            JOIN {info} i ON p.pmc_id=i.id
            JOIN {disp} d ON p.event_id=d.event_id
            JOIN {sym} s ON d.kernel_id=s.id
            WHERE s.kernel_name LIKE '%jdbba%' GROUP BY i.name"""
    return {nm: v for nm, v in c.execute(q)}


acc = {}
acc.update(sums(da))
acc.update(sums(dbdir))
hit, miss = acc.get("TCC_HIT"), acc.get("TCC_MISS")
dram, dram32 = acc.get("TCC_EA0_RDREQ_DRAM"), acc.get("TCC_EA0_RDREQ_DRAM_32B")
l2 = 100.0 * hit / (hit + miss) if hit and miss else float("nan")
dram_gb = (dram32 * 32 + (dram - dram32) * 64) / 1e9 if dram and dram32 is not None else float("nan")
print(f"{tag}\tL2={l2:.1f}%\tDRAM_rd={dram_gb:.1f}GB")
