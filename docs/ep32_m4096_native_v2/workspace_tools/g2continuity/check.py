"""Reuse the actual-input oracle, adding repeated default dispatch checks."""
import sys
from pathlib import Path


output = Path(sys.argv[1]).resolve()
source = (Path(__file__).resolve().parent.parent / "g2var/check80.py").read_text()
source = source.replace("Path('g2var/check80_summary.json')", f"Path({str(output / 'check_summary.json')!r})")
source = source.replace("'g2var/check80_bench'", repr(str(output / "input")))
source = source.replace(
    "    return original(*a,**kw)\n",
    "    for replay in range(3):\n"
    "        out.fill_(float('nan'))\n"
    "        original(*a,**kw)\n"
    "        torch.cuda.synchronize()\n"
    "        assert torch.equal(out[small],ref[small]),f'dispatch {replay}'\n"
    "        assert torch.isfinite(out).all(),f'unwritten dispatch {replay}'\n"
    "    print('repeated default dispatch: bit-exact <=2 routes, all output finite',flush=True)\n"
    "    return out\n",
)
exec(compile(source, "g2var/check80.py", "exec"))
