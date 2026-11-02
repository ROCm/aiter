from pathlib import Path
import json,shutil,datetime,hashlib
r=Path('/home/vigokhal/dsv4-monokernel');out=r/'latency-overlap'
result={}
for tag in ['native','previous','selected']:
 for rnd in [1,2]:
  p=r/'concurrency-rollout'/f'latency-overlap-{tag}-c8-round{rnd}.json'
  data=json.loads(p.read_text());assert data['completed']==64
  shutil.copy2(p,out/p.name)
  log=p.with_suffix('.log')
  if log.exists():shutil.copy2(log,out/log.name)
  if rnd==2:result[tag]={k:data[k] for k in ['completed','duration','median_ttft_ms','median_tpot_ms','total_token_throughput','output_throughput'] if k in data}
smoke=json.loads((out/'serving-text-check.json').read_text())
checks={name:[a['text']==b['text'] for a,b in zip(smoke['previous'],data)] for name,data in smoke.items()}
source=json.loads((out/'serving-source-sha256.json').read_text())
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in source.items()), 'Serving sources changed during A/B'
record=dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workload='Flash TP1, all43decode layers, C8,1024input/1024output,64requests',method='One server start per configuration, client twice, second client only. Configurations assigned separate GPUs.',selected_env=dict(ATOM_DSV4_EDGE_MASK=0,ATOM_DSV4_TILE_MASK=192,ATOM_DSV4_OVERLAP=2),e2e=result,matched_text_to_previous=checks,accuracy_note='Three greedy128-token prompt smoke checks only; no new GSM8K run.',new_defaults=False)
(out/'RESULTS.json').write_text(json.dumps(record,indent=2))
lines=['Whole-monokernel optimization pass — 2026-10-21','',record['workload'],record['method'],'','Configuration\tMedian_TTFT_ms\tMedian_TPOT_ms\tTotal_tokens_per_second']
for name,d in result.items():lines.append(f"{name}\t{d['median_ttft_ms']:.3f}\t{d['median_tpot_ms']:.3f}\t{d['total_token_throughput']:.3f}")
lines+=['','Selected: mHC pre reads shuffled input directly; final mHC post writes shuffled output directly; early OA/OB TDM. Earlier local input quant and bounded QB L2/TDM retained.','Every eligible decode layer remains a single attention+shared/routed-MoE kernel. Native prefill dispatch retained.','Both previous/selected A/B include corrected output TDM prologue ordering.','No automatic serving-default change. Use run-selected.sh.','',f'Matched text checks versus previous: {checks}',record['accuracy_note'],'','Finer producer/consumer and tile/expert readiness was implemented throughout the layer, not only QB. Pure-load combined replay has unresolved health failures; atomic-read controls passed SWA/HCA but regressed. These candidates remain unselected. Extra broad L2-prefetch variants stalled and remain off.','See DEPENDENCIES.tsv for all32 chronological handoffs and their status. See MICROBENCH.txt for C1/2/4/8 SWA/CSA/HCA full-layer A/B. Sources, failed logs, pre-edit backups, restore notes, ATOM overlay and image identifiers are retained.','No new image or Git push performed.']
(out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
