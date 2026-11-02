"""Install only the opt-in native-led diagnostic hook; run inside serving image."""
from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py')
s=p.read_text()
if 'from dual import attach' not in s:
 anchor='    def hc_pre(\n'
 assert s.count(anchor)==1
 hook='''        if os.environ.get("ATOM_DSV4_DUAL_REGION"):
            import sys
            sys.path.insert(0, "/work/modelwide")
            from dual import attach
            attach(self)

'''
 s=s.replace(anchor,hook+anchor)
 compile(s,str(p),'exec')
 p.write_text(s)
print('Diagnostic hook installed. Run dual-server.sh with explicit experimental flags disabled.')
