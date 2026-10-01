"""Untracked: extract MLIR-escaped code-object bytes for descriptor disassembly."""
import re
from pathlib import Path
root = Path('/home/jograner/projects/aiter/unified-attention-gemma4/profiling-data/decode16-final-dump')
for name in ('decode_0', 'combine_0'):
    text = (root / name / '20_gpu_module_to_binary.mlir').read_text()
    escaped = re.search(r'bin = "((?:\\.|[^"\\])*)"', text).group(1)
    parts = re.findall(r'\\([0-9A-Fa-f]{2})|\\(.)|([^\\])', escaped)
    binary = bytes(int(hexbyte, 16) if hexbyte else ord(special or char)
                   for hexbyte, special, char in parts)
    (root / f'{name}.co').write_bytes(binary)
