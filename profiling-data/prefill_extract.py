"""Untracked: extract each prefill experiment's code object and descriptor."""
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
objdump = '/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/llvm/bin/llvm-objdump'
for variant in ('base', 'single', 'mask', 'final', 'final64'):
    directory = root / f'prefill-{variant}-dump'
    text = (directory / 'attention_0/20_gpu_module_to_binary.mlir').read_text()
    escaped = re.search(r'bin = "((?:\\.|[^"\\])*)"', text).group(1)
    parts = re.findall(r'\\([0-9A-Fa-f]{2})|\\(.)|([^\\])', escaped)
    binary = bytes(int(hexbyte, 16) if hexbyte else ord(special or char)
                   for hexbyte, special, char in parts)
    co = directory / 'attention.co'
    co.write_bytes(binary)
    with (directory / 'attention.disassembly').open('w') as output:
        subprocess.run([objdump, '--disassemble-all', str(co)], stdout=output, check=True)
