import urllib.request,time,subprocess,sys
port=int(sys.argv[1]);tag=sys.argv[2];con=sys.argv[3]
for _ in range(600):
 try:
  with urllib.request.urlopen(f'http://localhost:{port}/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 time.sleep(1)
else:raise RuntimeError('Server readiness timeout')
subprocess.run(['bash','/work/concurrency-rollout/clients.sh',tag,str(port),con],check=True)
