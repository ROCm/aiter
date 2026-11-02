import json
from pathlib import Path
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if (len(args)>2 and args[1:3]==['-m','atom.entrypoints.openai_server']) or (args and args[0].strip().endswith('::api_server')):
   stat=(p/'stat').read_text().rsplit(')',1)[1].split()
   matches.append(dict(pid=int(p.name),start_ticks=int(stat[19]),args=[x for x in args if x]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(matches)==1,matches
print(json.dumps(matches[0],indent=2))
