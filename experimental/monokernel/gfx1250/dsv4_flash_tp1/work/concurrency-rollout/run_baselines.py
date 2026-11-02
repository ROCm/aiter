import urllib.request,time,subprocess
for _ in range(600):
 try:
  urllib.request.urlopen('http://127.0.0.1:10200/health',timeout=2);break
 except Exception:time.sleep(2)
else:raise RuntimeError('baseline server did not become healthy')
for c in [2,4,8]:
 subprocess.run(['docker','exec','vigokhal_dsv4_tp4','bash','/work/concurrency-rollout/clients.sh','baseline','10200',str(c)],check=True)
 print('baseline client pair complete C'+str(c),flush=True)
