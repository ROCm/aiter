#!/usr/bin/env node
// The watchdog is the only part of this bot that reports the failure the bot cannot report about
// itself: a review job that never gets a runner stays `queued`, so run_one.sh never starts and
// _notify.py never runs. Nothing else is listening. That makes its two silences as load-bearing
// as its alarm -- a watchdog that cries wolf while the single runner is merely busy would cost
// exactly what the mis-triaged GLM pages cost.
//
// The job only ever runs from the default branch (issue_comment workflows always do), so it
// cannot be exercised on a PR branch. Run it here instead: the real script, pulled out of the
// shipping workflow, against a faked GitHub API and a faked clock.
//
// Usage: node .claude/skills/review-pr/watchdog_test.js
'use strict';
const fs = require('fs');
const path = require('path');

const YML = path.resolve(__dirname, '../../../.github/workflows/aiter-review-bot.yml');

// Pull the script out of the workflow rather than keeping a copy here. A copy would pass forever
// after the shipping script changed under it.
function extractWatchdogScript() {
  const lines = fs.readFileSync(YML, 'utf8').split('\n');
  const job = lines.findIndex(l => /^  watchdog:\s*$/.test(l));
  if (job < 0) throw new Error(`no watchdog job in ${YML}`);
  const next = lines.findIndex((l, i) => i > job && /^  \S/.test(l));
  const end = next < 0 ? lines.length : next;
  const head = lines.findIndex((l, i) => i > job && i < end && /^(\s*)script: \|\s*$/.test(l));
  if (head < 0) throw new Error('the watchdog job has no `script: |` block');
  const indent = lines[head].match(/^(\s*)/)[1].length;
  const body = [];
  for (let i = head + 1; i < end; i++) {
    const l = lines[i];
    if (l.trim() === '') { body.push(''); continue; }
    if (l.match(/^(\s*)/)[1].length <= indent) break;
    body.push(l.slice(indent + 2));
  }
  const src = body.join('\n').trimEnd();
  if (!src) throw new Error('the watchdog script block is empty');
  return src;
}

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const SELF_RUN = 111;

// Drive the script with injected globals, so nothing real is touched: no network, no wall clock.
async function drive(src, scn) {
  const rec = { posts: [], notices: [], warnings: [], failed: [], selfPolls: 0 };
  let clock = 1_700_000_000_000;

  const Fake = {
    now: () => clock,
  };
  const setTimeout_ = (fn, ms) => { clock += ms; return global.setTimeout(fn, 0); };

  const jobsFor = async ({ run_id }) => {
    if (run_id === SELF_RUN) {
      rec.selfPolls++;
      // A watchdog that never returns would hang this test instead of failing it.
      if (rec.selfPolls > 60) throw new Error('the watchdog polled past any sane deadline');
      const st = scn.selfStatus(rec.selfPolls);
      return { data: { jobs: st === null ? [] : [{ name: 'review', status: st }] } };
    }
    return { data: { jobs: (scn.otherJobs || {})[run_id] || [] } };
  };

  // Model what the real endpoint does, not what is convenient: runs come back newest-first and
  // capped at per_page. A fake that always returns everything would hide a paging bug entirely.
  const listRuns = async ({ per_page = 30, page = 1 }) => {
    const src = typeof scn.liveRuns === 'function' ? scn.liveRuns(rec.selfPolls)
                                                  : (scn.liveRuns || []);
    const all = [...src].sort((a, b) => b.id - a.id);
    return { data: { workflow_runs: all.slice((page - 1) * per_page, page * per_page) } };
  };
  const github = { rest: { actions: {
    listJobsForWorkflowRun: jobsFor,
    listWorkflowRuns: listRuns,
  } },
    paginate: async (fn, params) => {
      const out = []; const per = params.per_page || 30;
      for (let page = 1; page <= 50; page++) {
        const r = await fn({ ...params, page });
        out.push(...r.data.workflow_runs);
        if (r.data.workflow_runs.length < per) break;
      }
      return out;
    } };

  const core = {
    notice: m => rec.notices.push(String(m)),
    warning: m => rec.warnings.push(String(m)),
    setFailed: m => rec.failed.push(String(m)),
  };

  const fetch_ = async (url, opts) => {
    rec.posts.push({ url, method: opts.method, headers: opts.headers,
                     body: JSON.parse(opts.body).body });
    return scn.postOk === false ? { ok: false, status: scn.postStatus || 403 }
                                : { ok: true, status: 201 };
  };

  const proc = { env: Object.assign(
    { STUCK_MINUTES: '20', PR: '42', AITER_BOT_TOKEN: 'tok-abc', OWNER_OVERRIDE: '' },
    scn.env || {}) };
  const context = { repo: { owner: 'ROCm', repo: 'aiter' }, runId: SELF_RUN };

  const fn = new AsyncFunction(
    'github', 'context', 'core', 'fetch', 'process', 'setTimeout', 'Date', src);
  await fn(github, context, core, fetch_, proc, setTimeout_, Fake);
  return rec;
}

// ---------------------------------------------------------------- the behaviours worth guarding

const QUEUED_FOREVER = () => 'queued';
const BUSY_ELSEWHERE = {
  liveRuns: [{ id: 222 }],
  otherJobs: { 222: [{ name: 'review', status: 'in_progress' }] },
};

const CASES = [
  { name: 'alarms when no runner ever claims the review',
    scn: { selfStatus: QUEUED_FOREVER },
    check: r => [
      [r.posts.length === 1, `posted ${r.posts.length} notices, want 1`],
      [r.posts.length === 1 && r.posts[0].url ===
        'https://api.github.com/repos/ROCm/aiter/issues/42/comments', 'posted to the wrong URL'],
      [r.posts.length === 1 && /no self-hosted runner claimed this review in 20 min/
        .test(r.posts[0].body), 'the notice does not say what happened'],
      [r.posts.length === 1 && /token tok-abc/
        .test(r.posts[0].headers.authorization), 'did not post as aiter-bot'],
      [r.failed.length === 1, 'the job did not fail, so the alarm is only a comment'],
    ] },

  { name: 'stays silent once our own review starts',
    scn: { selfStatus: n => (n === 1 ? 'queued' : 'in_progress') },
    check: r => [
      [r.posts.length === 0, 'paged someone about a review that had already started'],
      [r.failed.length === 0, 'failed the job for a review that had already started'],
      [r.selfPolls === 2, `kept polling after ours started (${r.selfPolls} polls)`],
    ] },

  { name: 'stays silent while the box is busy with another review',
    scn: Object.assign({ selfStatus: QUEUED_FOREVER }, BUSY_ELSEWHERE),
    check: r => [
      [r.posts.length === 0, 'paged someone for normal queueing behind the single runner'],
      [r.failed.length === 0, 'failed the job for normal queueing behind the single runner'],
      [r.notices.some(m => /busy/.test(m)), 'left no trace of why it went quiet'],
      [r.selfPolls > 1, 'stopped watching the moment it saw the box busy'],
    ] },

  // The single runner serialises reviews, so a batch of them piles up as a long list of
  // in_progress runs -- and the one actually running a review is the OLDEST, which the API
  // returns last. Looking at only the first page finds every run except the one that matters.
  { name: 'finds the busy runner under a large backlog',
    scn: { selfStatus: QUEUED_FOREVER,
           liveRuns: Array.from({ length: 120 }, (_, i) => ({ id: 1000 + i })),
           otherJobs: { 1000: [{ name: 'review', status: 'in_progress' }] } },
    check: r => [
      [r.posts.length === 0, 'paged someone although a review was running the whole time'],
      [r.notices.some(m => /busy/.test(m)), 'did not recognise the box as busy'],
    ] },

  // Seeing another review run is not a reason to stop watching: the runner can die one minute
  // later, and then this PR's job sits `queued` with nobody left alive to notice -- the exact
  // days-long silence this job exists to break.
  { name: 'alarms when the busy runner dies mid-wait',
    scn: { selfStatus: QUEUED_FOREVER,
           liveRuns: n => (n <= 2 ? [{ id: 222 }] : []),
           otherJobs: { 222: [{ name: 'review', status: 'in_progress' }] } },
    check: r => [
      [r.posts.length === 1, 'went quiet for a runner that died while it was watching'],
      [r.failed.length === 1, 'did not fail the run after the runner disappeared'],
    ] },

  // The watchdog finds its subject by job name. If the name no longer matches, `!review` reads
  // as "ours started" and it returns success forever -- a monitor that reports green precisely
  // because it can no longer see anything.
  { name: 'fails loudly when its own run has no review job',
    scn: { selfStatus: () => null },
    check: r => [
      [r.posts.length === 0, 'paged the runner owner about a workflow-structure problem'],
      [r.failed.length === 1, 'reported success while watching nothing at all'],
      [r.failed.length === 1 && /review/.test(r.failed[0]),
        `did not say what it could not find: ${r.failed[0]}`],
    ] },

  { name: 'pages the override owner when the repo sets one',
    scn: { selfStatus: QUEUED_FOREVER, env: { OWNER_OVERRIDE: '  gyohuangxin  ' } },
    check: r => [
      [r.posts.length === 1 && /@gyohuangxin\b/.test(r.posts[0].body), 'ignored the override'],
      [r.posts.length === 1 && !/@zufayu\b/.test(r.posts[0].body), 'paged the default owner too'],
    ] },

  // A misconfigured STUCK_MINUTES makes `deadline` NaN, and `Date.now() < NaN` is false, so the
  // loop never runs and the alarm fires on the spot. That is the false page this job exists to
  // avoid, triggered by a typo in the workflow rather than by anything on the box.
  { name: 'does not page anyone when STUCK_MINUTES is unusable',
    scn: { selfStatus: QUEUED_FOREVER, env: { STUCK_MINUTES: '' } },
    check: r => [
      [r.posts.length === 0, 'paged someone because of a workflow typo, not a dead runner'],
      [r.failed.length === 1, 'a watchdog that cannot run must not pass for a healthy one'],
      [r.failed.length === 1 && /STUCK_MINUTES/.test(r.failed[0]),
        `blamed the runner for a workflow typo: ${r.failed[0]}`],
    ] },

  { name: 'still fails the job when the notice cannot be posted',
    scn: { selfStatus: QUEUED_FOREVER, postOk: false, postStatus: 403 },
    check: r => [
      [r.warnings.some(m => /403/.test(m)), 'swallowed the failed post'],
      [r.failed.length === 1, 'a token problem would have hidden a dead runner entirely'],
    ] },
];

// Every one of the above passes against code that does nothing at all in the paths it claims to
// guard -- unless each is shown to go red when its path is broken. Break them on purpose.
const MUTANTS = [
  { why: 'the early return for a review that started',
    find: 'return;            // ours started', with: ';',
    breaks: 'stays silent once our own review starts' },
  { why: 'the guard on an unusable STUCK_MINUTES',
    find: 'if (!Number.isFinite(mins) || mins <= 0) {', with: 'if (false) {',
    breaks: 'does not page anyone when STUCK_MINUTES is unusable' },
  { why: 'the guard on a run with no review job',
    find: 'if (!sawReview) {', with: 'if (false) {',
    breaks: 'fails loudly when its own run has no review job' },
  // Proves the final busy check reflects the box as it is at the end, not as it was mid-wait.
  { why: 'the box-busy answer at the end of the window',
    find: 'if (await boxBusy()) {', with: 'if (true) {',
    breaks: 'alarms when the busy runner dies mid-wait' },
  { why: 'paging through every in_progress run',
    find: 'await github.paginate(github.rest.actions.listWorkflowRuns, {',
    with: 'await ((f, p) => f(p).then(r => r.data.workflow_runs))(github.rest.actions.listWorkflowRuns, {',
    breaks: 'finds the busy runner under a large backlog' },
  { why: 'the busy-runner check',
    find: 'if (await boxBusy()) {', with: 'if (false) {',
    breaks: 'stays silent while the box is busy with another review' },
  { why: 'failing the job on alarm',
    find: 'core.setFailed(`no runner', with: 'core.notice(`no runner',
    breaks: 'alarms when no runner ever claims the review' },
  { why: 'the warning when the post is rejected',
    find: 'core.warning(`could not post', with: 'String(`could not post',
    breaks: 'still fails the job when the notice cannot be posted' },
  { why: 'the owner override',
    find: "(process.env.OWNER_OVERRIDE || '').trim() || 'zufayu'", with: "'zufayu'",
    breaks: 'pages the override owner when the repo sets one' },
];

// ------------------------------------------------------------------------------------- the run

async function runSuite(src) {
  const out = new Map();
  for (const c of CASES) {
    try {
      const rec = await drive(src, c.scn);
      const bad = c.check(rec).filter(([ok]) => !ok).map(([, msg]) => msg);
      out.set(c.name, bad.length ? bad.join('; ') : null);
    } catch (e) {
      out.set(c.name, `threw: ${e.message}`);
    }
  }
  return out;
}

(async () => {
  const src = extractWatchdogScript();
  let ok = 0, bad = 0;
  const t = (name, err) => {
    if (!err) { console.log(`  ✅ ${name}`); ok++; }
    else { console.log(`  ❌ ${name} — ${err}`); bad++; }
  };

  console.log('[watchdog behaviour]');
  for (const [name, err] of await runSuite(src)) t(name, err);

  console.log('[watchdog guards bite]');
  for (const m of MUTANTS) {
    if (!src.includes(m.find)) {           // the script drifted; the mutation tests nothing
      t(`breaking ${m.why} is caught`, `the script no longer contains \`${m.find}\``);
      continue;
    }
    const res = await runSuite(src.replace(m.find, m.with));
    const err = res.get(m.breaks);
    t(`breaking ${m.why} turns "${m.breaks}" red`,
      err ? null : 'it stayed green, so that check proves nothing');
  }

  console.log(`=== ${ok} green / ${bad} red ===`);
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
