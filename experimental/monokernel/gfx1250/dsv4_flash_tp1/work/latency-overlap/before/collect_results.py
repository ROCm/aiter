from pathlib import Path
import json,re,hashlib,subprocess
r=Path('/home/vigokhal/dsv4-monokernel');o=r/'integrated-c8'
results={}
for name in ['native','previous','optimized']:
 p=r/'concurrency-rollout'/f'integrated-{name}-c8-round2.json'
 if p.exists():
  d=json.loads(p.read_text());results[name]={k:d[k] for k in ['completed','median_ttft_ms','median_tpot_ms','total_token_throughput','output_throughput']}
  results[name]['path']=str(p)
accuracy={}
for name in ['native','previous','optimized']:
 for p in (r/'batch-full'/f'gsm8k-integrated-{name}-c8').rglob('results_*.json'):
  d=json.loads(p.read_text());accuracy[name]={'metrics':d.get('results',{}),'path':str(p)}
log=(o/'optimized-server.log').read_text() if (o/'optimized-server.log').exists() else ''
compiled={}
for m in [1,2,4,8]:
 matches=re.findall(r'OPTIMIZED FULL MONOKERNEL layer=(\d+) batch='+str(m)+r' ratio=(\d+) VGPR=(\d+) LDS=(\d+) spills=(\d+)',log)
 compiled[m]=dict(layers=sorted(set(int(t[0]) for t in matches)),ratios=sorted(set(int(t[1]) for t in matches)),vgpr=sorted(set(int(t[2]) for t in matches)),lds=sorted(set(int(t[3]) for t in matches)),spills=sorted(set(int(t[4]) for t in matches)))
report=dict(model='DeepSeek-V4-Flash',tp=1,ep=False,concurrency=8,input_len=1024,output_len=1024,num_prompts=64,method='One server start per configuration; two clients; report second client only. Median TTFT/TPOT. Total throughput includes input+output tokens. Native GPU0, previous GPU1, optimized GPU3.',results=results,accuracy=accuracy,compiled=compiled,integrated=['local QKV input quant and early bounded QB prefetch; split16 retains B3','direct paged SWA/HCA/CSA attention; row-interleaved8splits; no compact-cache staging','native-derived fused attention mHCpost+FFNpre/RMSNorm with packed weights, explicit FMAs, dedicated64producer barrier, warp specialization, adapted coefficient/residual layouts','parallel routing metadata','routedMoE BN256 BK512 two buffers, active-expert scheduling and coherent polling','HCA four-way pooling plus compressor subgroup barrier for ratio4'],integration_adjustments=['HCA compressor subgroup barriers timed out in full-kernel integration; use all256 barriers before/after parallel pooling. Preserve ratio4 subgroup and mHC64producer barrier.','Preserve negative BatchIds on padding, destination bounds, 8-field production plans, FP32 cross-layer residual statistic inputs.'],limitations=['Attention split reduction and HCA pooling order differ from previous/native; not bitwise equivalent overall.','Synthetic full-layer tests can amplify small attention differences through quantization/routing; saved errors retained, no blanket exactness claim.','GSM8K is a32-example3shot smoke test, max512 output tokens, not full accuracy qualification.','No ProTP4 run or image/branch push in this integration turn.'])
(o/'RESULTS.json').write_text(json.dumps(report,indent=2))
lines=['# Integrated C8 full-layer monokernel — 2026-10-21','','Flash single GPU (TP1, no EP). One fused attention+MoE kernel per eligible decode layer; all43layers. Prefill stays native.','','## End-to-end performance','','1k input/1k output, concurrency8,64requests. Server started once per configuration; client run twice; only second result below. Median TTFT/TPOT. Total tok/s includes input+output.','','|Configuration|TTFT ms|TPOT ms|Total tok/s|Output tok/s|','|---|---:|---:|---:|---:|']
for n,d in results.items():lines.append(f"|{n}|{d['median_ttft_ms']:.3f}|{d['median_tpot_ms']:.3f}|{d['total_token_throughput']:.3f}|{d['output_throughput']:.3f}|")
lines+=['','## Integrated changes','']+['- '+x for x in report['integrated']]+['','## Integration adjustments','']+['- '+x for x in report['integration_adjustments']]
lines+=['','## Validation','','Resources: every reported serving layer/bucket has zero spills; inspect compiled table in RESULTS.json. Synthetic stage checks, padded requests, and320 full-layer graph replays each for HCA/CSA are saved alongside. Recurrent state exact in tested padded cases. HCA compressed cache can differ by up to0.00048828125 (relative RMS1.93e-5) due to pooling reduction order.','','GSM8K32 examples,3-shot,max512 output:']
for n,d in accuracy.items():lines.append('- '+n+': '+json.dumps(d['metrics']))
lines+=['','## Limitations','']+['- '+x for x in report['limitations']]+['','## Reproduce','','`install.py` updates only the existing ATOM adapter directory selection; default remains /work/batch-full. `run.sh fused` selects /work/integrated-c8 through ATOM_DSV4_BATCH_FULL_DIR. `run.sh baseline` disables fusion via the existing baseline script. Set HIP_VISIBLE_DEVICES, HTTP_PORT, PORT, CONCURRENCY=8, SERVER_LOG explicitly. Model /data/DeepSeek-V4-Flash; host directory mounted /work. Run `/work/concurrency-rollout/clients.sh TAG HTTP_PORT 8` after health ready; it runs both clients and saves both.','', 'Do not rerun integrate.py or prepare_tests.py blindly: they are initial generation records and precede the HCA full-grid synchronization fix. Final executable source is model_layer_batch.py, runtime_batch.py, mhc_integrated.py, and imported modules. Optional ATOM_DSV4_OPT_MHC/OPT_COMP switches retain diagnostic comparisons. All enabled for the optimized result.','', 'The archive contains selected source dependencies, fixture, code patches, logs, compiled ISA, raw client outputs, and accuracy records. Matching ROCm/Triton/AITER image and the model under /data are still required.']
(o/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(o/'SOURCE_SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in o.glob('*.py')},indent=2))
print(json.dumps(results,indent=2));print('accuracy',list(accuracy))
