import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lib

if len(sys.argv) < 2:
    sys.exit("usage: gates.sh <WORK_DIR> [PROJECT_ROOT]")
W = sys.argv[1]
PROJ = (
    sys.argv[2]
    if len(sys.argv) > 2
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print(f"======== SEVEN GATES ({W}) ========")
res = _lib.run_gates(W, PROJ)
npass = 0
for name, rc, first, full in res:
    if rc == 0:
        npass += 1
        print(f"  ✅ {name:<12} {first}")
    else:
        print(f"  ❌ {name:<12} exit={rc}")
        for l in full.split("\n"):
            print(f"       {l}")
nfail = len(res) - npass
print(f"======== {npass} green / {nfail} red ========")
sys.exit(0 if nfail == 0 else 1)
