import json
from pathlib import Path
root=Path(__file__).parent
report={'valid_reference':'dual-native-level0-responses.json','notes':['Earlier early-binding runs cached MoE views before child weight preparation: exclude from native-path acceptance.','Compressor invalid-harness-strides run is invalid.','Compressor late run failed with illegal access using prepared weight layout; raw-checkpoint rerun is separate.'],'regions':{},'responses':{}}
for mode in ['qkv','compressor','mhc','attention_output_ffn']:
 path=root/f'dual-{mode}.jsonl'
 if not path.exists():continue
 rows=[json.loads(x) for x in path.read_text().splitlines()]
 for group in sorted({r['group'] for r in rows}):
  rr=[r for r in rows if r['group']==group];out={}
  for key in sorted({k for r in rr for k,v in r.items() if isinstance(v,dict) and 'rms' in v}):
   cases=[r for r in rr if key in r];w=max(cases,key=lambda r:r[key]['rms'])
   out[key]={'cases':len(cases),'exact':sum(r[key]['neq']==0 for r in cases),'worst':w}
  report['regions'][group]=out
ref=json.loads((root/'dual-native-level0-responses.json').read_text())
for mode in ['qkv-late','compressor-packed','mhc-final','attention_output_ffn-final']:
 p=root/f'dual-{mode}-responses.json'
 if p.exists():
  data=json.loads(p.read_text());report['responses'][mode]={'completed':len(data),'matches':[a['choices']==b['choices'] for a,b in zip(data,ref)]}
(root/'dual-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['responses']))
