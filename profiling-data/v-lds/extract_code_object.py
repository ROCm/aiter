"""Extract the local FlyDSL compiled artifact for offline ISA inspection."""
import pathlib
import pickle
import re
import sys

artifact = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes())
blob = re.search(r'bin = "((?:\\.|[^"\\])*)"', artifact.ir).group(1)
data = bytearray()
i = 0
while i < len(blob):
    if blob[i] == "\\":
        if blob[i + 1] in ('\\', '"'):
            data.append(ord(blob[i + 1]))
            i += 2
        else:
            data.append(int(blob[i + 1:i + 3], 16))
            i += 3
    else:
        data.append(ord(blob[i]))
        i += 1
pathlib.Path(sys.argv[2]).write_bytes(data)
