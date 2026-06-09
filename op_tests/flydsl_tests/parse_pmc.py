"""Parse rocprofv3 (rocpd schema) --pmc output for the jdbba kernel: L2 hit and
LLC-hit (DRAM-avoided) fraction, summed over all XCD instances and all jdbba
dispatches. Usage: parse_pmc.py <pmc_outdir> <p10_us> <tag>"""
import sys, sqlite3, glob, os

outdir, us, tag = sys.argv[1], sys.argv[2], sys.argv[3]
dbs = glob.glob(os.path.join(outdir, "**", "*.db"), recursive=True)
if not dbs:
    print(f"{tag}\t{us}\tNO_PMC_DB")
    sys.exit(0)
c = sqlite3.connect(dbs[0])
names = {n for (n,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def T(base):
    for n in names:
        if n.startswith(base):
            return n


pmc = T("rocpd_pmc_event")
info = T("rocpd_info_pmc")
disp = T("rocpd_kernel_dispatch")
sym = T("rocpd_info_kernel_symbol")

# Sum each counter over the jdbba dispatches only (all XCD instances included).
q = f"""
SELECT i.name, SUM(p.value)
FROM {pmc} p
JOIN {info} i ON p.pmc_id = i.id
JOIN {disp} d ON p.event_id = d.event_id
JOIN {sym} s ON d.kernel_id = s.id
WHERE s.kernel_name LIKE '%jdbba%'
GROUP BY i.name
"""
acc = {nm: v for nm, v in c.execute(q)}


def g(name):
    return acc.get(name)


hit = g("TCC_HIT")
miss = g("TCC_MISS")
dram = g("TCC_EA0_RDREQ_DRAM")
dram32 = g("TCC_EA0_RDREQ_DRAM_32B")
l2 = (100.0 * hit / (hit + miss)) if (hit and miss is not None and (hit + miss) > 0) else float("nan")
# Actual HBM read volume: DRAM reqs are 32B or 64B. bytes = n32*32 + (n-n32)*64.
dram_gb = float("nan")
if dram is not None and dram32 is not None:
    dram_gb = (dram32 * 32 + (dram - dram32) * 64) / 1e9
print(f"{tag}\t{us}\tL2={l2:.1f}%\tDRAM_rd={dram_gb:.1f}GB\t(hit={hit} miss={miss} dram_req={dram})")
