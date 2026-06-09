import sys, sqlite3, statistics, glob, os
db=glob.glob(os.path.join(sys.argv[1],"**","*.db"),recursive=True)
sub=sys.argv[2]
if not db: print("nan"); sys.exit(0)
c=sqlite3.connect(db[0])
tabs=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type=\x27table\x27")]
disp=[t for t in tabs if "kernel_dispatch" in t]; sym=[t for t in tabs if "kernel_symbol" in t]
if not disp or not sym: print("nan"); sys.exit(0)
rows=c.execute(f"SELECT s.kernel_name,d.end-d.start FROM {disp[0]} d JOIN {sym[0]} s ON d.kernel_id=s.id").fetchall()
d=sorted(v for nm,v in rows if v and v>0 and sub in (nm or ""))
print(f"{statistics.median(d)/1000:.2f}" if d else "nan")
