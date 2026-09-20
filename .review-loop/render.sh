#!/usr/bin/env bash
# Substitute the placeholders in prompts/<template>.md with this run's actual values, to stdout.
#   render.sh worker <WORK>
#   render.sh refuter <WORK> <findings-file>
# PR number and title are read from <WORK>/pr_meta.json, not typed by hand (hand entry errs).
set -euo pipefail
T="${1:?usage: render.sh <template> <WORK_DIR> [findings-file]}"
WORK="${2:?work dir}"
FIND="${3:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
TPL="$HERE/prompts/$T.md"
[ -f "$TPL" ] || { echo "no such template: $TPL" >&2; ls "$HERE/prompts"/*.md >&2; exit 1; }
[ -s "$WORK/pr_meta.json" ] || { echo "missing $WORK/pr_meta.json -- run fetch.sh first" >&2; exit 1; }
if [ -n "$FIND" ] && [ ! -s "$FIND" ]; then echo "findings file missing or empty: $FIND" >&2; exit 1; fi

# Do all substitution in python: shell word-splitting shreds a title with spaces (hit that).
TPL="$TPL" WORK="$WORK" PROJ="$PROJ" FIND="$FIND" python3 <<'PY'
import json,os,re,sys
W=os.environ['WORK']; P=os.environ['PROJ']
d=json.load(open(os.path.join(W,'pr_meta.json'),encoding='utf-8'))
def dig(*ks):
    for k in ks:
        cur=d
        for part in k.split('.'):
            cur = cur.get(part) if isinstance(cur,dict) else None
            if cur is None: break
        if cur: return cur
    return ''
vals={
 'PR':    str(dig('number','pr') or ''),
 'TITLE': str(dig('title') or '').replace('\n',' ').strip(),
 'REPO':  str(dig('repo','base.repo.full_name') or 'ROCm/aiter'),
 'WORK':  W,
 'PROJ':  P,
 'SKILL': os.path.join(P,'.claude','skills','review-pr'),
 'FINDINGS': open(os.environ['FIND'],encoding='utf-8').read().rstrip() if os.environ.get('FIND') else '',
}
if not vals['PR']:    sys.exit('no PR number in pr_meta.json')
if not vals['TITLE']: sys.exit('no title in pr_meta.json')
t=open(os.environ['TPL'],encoding='utf-8').read()
for k,v in vals.items(): t=t.replace('{{%s}}'%k, v)
un=sorted(set(re.findall(r'\{\{[A-Z]+\}\}', t)))
if un: print('warning: unsubstituted placeholders: '+' '.join(un), file=sys.stderr)
sys.stdout.write(t)
PY
