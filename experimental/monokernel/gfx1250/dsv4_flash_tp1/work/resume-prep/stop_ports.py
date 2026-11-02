import subprocess,sys,os,signal
ports=[int(p) for p in sys.argv[1:]]
assert ports and all(10000<=p<12000 for p in ports)
rows=[l.strip().split(None,2) for l in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
ids={int(p) for p,pp,a in rows if a.startswith('python ') and any(f'--server-port {port} ' in a+' ' or f'--base-url=http://localhost:{port} ' in a+' ' for port in ports)}
for _ in range(8):ids|={int(p) for p,pp,a in rows if int(pp) in ids}
for p in sorted(ids,reverse=True):
 try:os.kill(p,signal.SIGKILL)
 except ProcessLookupError:pass
print('Stopped selected experiment processes',sorted(ids))
