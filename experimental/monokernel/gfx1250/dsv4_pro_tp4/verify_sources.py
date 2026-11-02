"""CPU-only integrity and syntax check; does not import GPU modules."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for entry in manifest:
    path = root / entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
python_files = list(root.rglob("*.py"))
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
shell_files = list(root.rglob("*.sh"))
for path in shell_files:
    subprocess.run(["bash", "-n", str(path)], check=True)
print(json.dumps({"hashes_verified": len(manifest), "python_parsed": len(python_files), "shell_parsed": len(shell_files), "gpu_tests_run": False}))
