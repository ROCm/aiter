// Dry-run harness for loop/run.workflow.js.
//
// Loads the workflow script body, stubs the four hooks the workflow tool provides
// (agent / parallel / phase / log) and exercises the control flow WITHOUT calling any model.
// Its job is to prove that the guards fire: a happy path completes, and every failure the
// post-round-3 review identified is refused rather than carried.
//
//   node loop/dryrun.mjs
//
// Exit code 0 = every scenario behaved as expected.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const body = readFileSync(join(here, '..', 'workflows', 'run.workflow.js'), 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const make = () => new AsyncFunction('args', 'agent', 'parallel', 'pipeline', 'phase', 'log', body);

const ARGS = {
  pr: 4961,
  repo: 'ROCm/aiter',
  headSha: 'HEADSHA',
  baseSha: 'BASESHA',
  stale: true,
  localWork: 'W:\\work',
  localReports: 'W:\\reports',
  skillRoot: 'S:\\skills\\review-pr',
  projectRoot: 'P:\\aiter',
  python: 'python3',
  hostUser: 'u',
  uidGid: '1000:1000',
  validateSkillRoot: 'S:\\\\skills\\\\validate-kernel-pr',
  local: { ssh: 'ssh', scp: 'scp', identityFile: '/k/id', toolsRoot: 'T:' },
  files: ['a/one.py', 'a/two.py', 'a/three.py'],
  ruleGroups: [{ id: 'A_B', ids: 'A1 B1' }, { id: 'C_D', ids: 'C1 D1' }],
  targets: [{
    t: 'op_tests/test_x.py', host: 'h1.example', user: 'u', container: 'c1',
    worktree: '/work/pr', patch: '/work/pr.patch', scratch: '/work/_tmp/run', hostRoot: '/home/u/pr', refresh: false,
  }],
  validationRequired: true,
  providers: {
    worker: { provider: 'w', model: 'w' },
    refuter: { provider: 'r', model: 'r' },
    formatter: { provider: 'f', model: 'f' },
    remote: { provider: 'm', model: 'm' },
  },
  maxGateRounds: 2,
  sinceHead: 'PREVHEAD',
  resume: false,
};

// ---------------------------------------------------------------- the stub

function makeAgent(scenario, calls) {
  return async function agent(prompt, opts = {}) {
    const label = String(opts.label || '');
    calls.push(label);
    const o = scenario.overrides || {};
    if (Object.prototype.hasOwnProperty.call(o, label)) return o[label];
    for (const [pattern, value] of Object.entries(scenario.patterns || {})) {
      if (label.startsWith(pattern)) return typeof value === 'function' ? value(label) : value;
    }

    if (label.startsWith('S4 ')) {
      const f = label.slice(3);
      return `Looked at the hunks.\n${f} TIER2 COVERED -- the change to do_thing() is exercised by the added test`;
    }
    if (label.startsWith('S5 rules')) return 'wrote the verdict file; 2 rules adjudicated';
    if (label === 'S2 semantic') return 'Q1..Q5 and BLIND written';
    if (label === 'S6 AI diagnostic') return 'six checks written';
    if (label === 'write core_files.txt') return `core_files.txt has ${ARGS.files.length} lines`;
    if (label === 'count FIRE verdicts') return `FIRE=${scenario.fire ?? 3}`;
    if (label === 'collect findings') {
      const n = scenario.collected ?? 4;
      const lines = Array.from({ length: n }, (_, i) => `RED|finding number ${i + 1} in a/one.py:${i + 10}`);
      return `BEGIN_FINDINGS\n${lines.join('\n')}\nEND_FINDINGS`;
    }
    if (label === 'check resumable refutations') return scenario.existing || 'EXISTING=NONE';
    if (label.startsWith('refute F')) return `SURVIVED -- opened a/one.py and could not kill it`;
    if (label === 'assemble ledger') return 'refutations.txt and independent.txt written';
    if (label === 'ledger provenance') {
      const n = scenario.ledger ?? (scenario.collected ?? 4);
      return `FILES=${n}\nREFUT=${n}\nINDEP=${n}\nBAD=${scenario.bad ?? 0}`;
    }
    if (label.startsWith('card + gates')) return 'card written, gates run';
    if (label.startsWith('gate check')) {
      const n = scenario.collected ?? 4;
      return scenario.gatesGreen === false
        ? `card gate failed: UNANCHORED finding\nrefutations.txt lines: ${n}`
        : `all seven exit 0\nrefutations.txt lines: ${n}\nALLGREEN`;
    }
    if (label.startsWith('validate ')) return 'verdict INCONCLUSIVE, all stages pass except the shape grid';
    if (label.startsWith('refute executor')) return 'the shape-grid note is a property of the invocation - KILLED';
    if (label === 'fold validation into card') return 'validation folded in, gates re-run';
    if (label === 'final independent gate check') {
      const n = scenario.collected ?? 4;
      return scenario.finalGreen === false
        ? `card gate failed after refold\nrefutations.txt lines: ${n}`
        : `all seven exit 0\nrefutations.txt lines: ${n}\nALLGREEN`;
    }
    if (label.startsWith('cleanup ')) return 'scratch deleted, worktree clean, 0 root-owned';
    return `stub response for ${label}`;
  };
}

async function run(scenario) {
  const calls = [];
  const logs = [];
  const agent = makeAgent(scenario, calls);
  const parallel = async (thunks) => {
    const out = [];
    for (const t of thunks) {
      try { out.push(await t()); } catch { out.push(null); }
    }
    return out;
  };
  const fn = make();
  try {
    const value = await fn(ARGS, agent, parallel, async () => {}, () => {}, (m) => logs.push(String(m)));
    return { ok: true, value, calls, logs };
  } catch (err) {
    return { ok: false, error: String(err && err.message || err), calls, logs };
  }
}

// ---------------------------------------------------------------- scenarios

const scenarios = [
  {
    name: 'happy path completes and returns a card',
    scenario: { fire: 3, collected: 4 },
    expect: (r) => r.ok && r.value.findings === 4 && r.value.gatesGreen === true
      && r.value.validated.length === 1 && r.value.card.endsWith('card.md'),
  },
  {
    name: 'FIX 4 — a dead S2 analysis agent aborts the run',
    scenario: { fire: 3, collected: 4, patterns: { 'S5 rules C_D': null } },
    expect: (r) => !r.ok && /analysis agents produced nothing/.test(r.error)
      && /rule ledger and the core-file/.test(r.error),
  },
  {
    name: 'FIX 8 — a collection short of the FIRE count aborts the run',
    scenario: { fire: 7, collected: 5 },
    expect: (r) => !r.ok && /5 findings but the verdict files contain 7 FIRE/.test(r.error),
  },
  {
    name: 'FIX 2 — an empty collection aborts instead of certifying a clean card',
    scenario: { fire: 0, collected: 0 },
    expect: (r) => !r.ok && /0 findings/.test(r.error),
  },
  {
    name: 'FIX 2 — a truncated ledger (the r2 failure) aborts on provenance',
    scenario: { fire: 3, collected: 17, ledger: 2 },
    expect: (r) => !r.ok && /provenance failed: 2 refuter files/.test(r.error),
  },
  {
    name: 'provenance also catches a malformed ledger line',
    scenario: { fire: 3, collected: 4, bad: 1 },
    expect: (r) => !r.ok && /1 malformed/.test(r.error),
  },
  {
    name: 'FIX 3 — gates that never go green abort before validation',
    scenario: { fire: 3, collected: 4, gatesGreen: false },
    expect: (r) => !r.ok && /did not go green within 2 rounds/.test(r.error)
      && !r.calls.some(c => c.startsWith('validate ')),
  },
  {
    name: 'FIX 9 — a card that fails the gates after the refold aborts',
    scenario: { fire: 3, collected: 4, finalGreen: false },
    expect: (r) => !r.ok && /do not pass after the validation refold/.test(r.error),
  },
  {
    name: 'FIX 7 — resume skips refuters whose file already exists',
    scenario: { fire: 3, collected: 4, existing: 'EXISTING=F01.md,F02.md' },
    args: { resume: true },
    expect: (r) => r.ok && r.calls.filter(c => c.startsWith('refute F')).length === 2
      && r.logs.some(l => /resuming - 2 refuter files/.test(l)),
  },
  {
    name: 'gate check that loses the ledger line count is not accepted as green',
    scenario: { fire: 3, collected: 4, overrides: { 'gate check (round 1)': 'all seven exit 0\nALLGREEN', 'gate check (round 2)': 'all seven exit 0\nALLGREEN' } },
    expect: (r) => !r.ok && /did not go green/.test(r.error),
  },
];

let failed = 0;
for (const s of scenarios) {
  if (s.args) Object.assign(ARGS, s.args);
  const r = await run(s.scenario);
  if (s.args) Object.assign(ARGS, { resume: false });
  let pass = false;
  try { pass = !!s.expect(r); } catch { pass = false; }
  if (!pass) failed++;
  const detail = r.ok ? `returned ${JSON.stringify(r.value && r.value.findings)}` : `threw: ${r.error.split('\n')[0].slice(0, 110)}`;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${s.name}\n        ${detail}`);
}

console.log(`\n${scenarios.length - failed}/${scenarios.length} scenarios behaved as specified`);
process.exit(failed ? 1 : 0);
