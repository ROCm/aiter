import urllib.request,time,subprocess,os
for _ in range(600):
 try:urllib.request.urlopen('http://127.0.0.1:10100/health',timeout=2);break
 except Exception:time.sleep(2)
else:raise RuntimeError('graph server not ready')
subprocess.run(['docker','exec','-e','FUSED_PORT=10100','-e','QUALITY_TAG=-bounded-graph','vigokhal_dsv4_tp4','python','/work/modelwide/quality_smoke.py'],check=True,stdout=open('/home/vigokhal/dsv4-monokernel/modelwide/quality-smoke-bounded-graph.log','w'))
subprocess.run(['docker','exec','vigokhal_dsv4_tp4','bash','/work/modelwide/clients.sh','corrected-c1','10100'],check=True)
