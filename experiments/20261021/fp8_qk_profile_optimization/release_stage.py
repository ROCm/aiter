"""Preserve phase 1 and prepare an independent final validation directory."""

import shutil
import sys
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021")
out = work / "release"
out.mkdir()
tag = sys.argv[1]
shutil.copytree(work / tag, out / tag)
for name in [
    "prepare_final.py",
    "full_benchmark.py",
    "shared.py",
    "audit.py",
    "workload.py",
    "collect.py",
    "final_profiles.py",
    "summarize.py",
    "att_compare.py",
    "analyze_final.py",
    "finish_runs.py",
]:
    source = (work / name).read_text()
    source = source.replace(
        "'fp8-qk-profile-opt-20261021", "'fp8-qk-profile-opt-20261021/release"
    )
    source = source.replace(
        "/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021",
        "/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release",
    )
    if name == "prepare_final.py":
        source = source.replace(
            "subprocess.run([formatter,'-i',str(repo/header)],check=True)",
            "subprocess.run([formatter,'-i',str(repo/header),str(repo/main)],check=True)",
        )
    compile(source, name, "exec")
    (out / name).write_text(source)
for p in work.iterdir():
    if p.is_dir() and (
        p.name.startswith(("baseline_row_", "baseline_pr_")) or p.name == "format_tool"
    ):
        (out / p.name).symlink_to(p, target_is_directory=True)
print("RELEASE_STAGED", tag)
