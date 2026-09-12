// .agent_loop — PHASE R (run to report). No human break inside this phase.
//
// Shape: rlar. The worker holds the analysis, a FRESH refuter that has not seen the worker's
// reasoning attacks every finding, and the skill's own seven gates are the `done` signal —
// a failing gate's complaint is fed back as the next round's notes, verbatim.
//
// This file IS the artifact that ran on ROCm/aiter#4961 round 3 (44 agents, gates green,
// ledger provenance 15/15/15), plus the fixes from the post-round-3 review. Do not let an
// inline variant diverge from it again: the deliverable and the tested thing must be one file.
//
// Loaded by the main agent and passed to the `workflow` tool as the `script` parameter.
//
// args = {
//   pr, repo, headSha, baseSha, stale,
//   localWork,        // ...\tmp\pr-review\<pr>\<round>\work\review-pr-XXXXXX   ($WORK)
//   localReports,     // ...\tmp\pr-review\<pr>\<round>\reports
//   skillRoot,        // ...\aiter\.claude\skills\review-pr
//   projectRoot,      // ...\aiter          (the corefiles gate's PROJECT_ROOT)
//   python,           // "C:\\Python314\\python.exe"
//   files: [...],                                   // files to assess in Step 4
//   ruleGroups: [ { id, ids }, ... ],               // Step 5 partition of rules.txt
//   targets: [ { t, host, container, worktree, patch, scratch, hostRoot, refresh } ],
//   validationRequired: true|false,
//   providers: { worker, refuter, formatter, remote },   // each { provider, model }
//   maxGateRounds: 3,
//   sinceHead: "<previous reviewed head sha>" | null,     // for the cross-round delta line
//   resume: true|false          // reuse refutation_lines/F*.md that already exist
// }

const A = args;
const W = A.localWork;
const R = A.localReports;
const L = `${W}\\refutation_lines`;
const SK = A.skillRoot;
const VSK = A.validateSkillRoot;
const PROJ = A.projectRoot;
const PY = A.python;

// Connection details are configuration, never literals: see config/loop.example.json.
const LOCAL = A.local || {};
const SSHX = LOCAL.ssh || 'ssh';
const IDENT = LOCAL.identityFile ? `-i ${LOCAL.identityFile} -o IdentitiesOnly=yes ` : '';
const SSH_TO = (t) => `${SSHX} ${IDENT}-o BatchMode=yes -o StrictHostKeyChecking=no ` +
  `-o ConnectTimeout=25 -o ServerAliveInterval=30 ${t.user || A.hostUser}@${t.host}`;

const WORKER = A.providers.worker;
const REFUTER = A.providers.refuter;
const FORMATTER = A.providers.formatter;
const REMOTE = A.providers.remote;
const MAX_ROUNDS = A.maxGateRounds || 3;

const GATES = [
  `& ${PY} ${SK}\\triage.py answers ${W}\\answers.txt`,
  `& ${PY} ${SK}\\triage.py diagnostic ${W}\\ai_diagnostic.txt`,
  `& ${PY} ${SK}\\triage.py corefiles ${W}\\core_files.txt ${W}\\pr.diff ${PROJ}`,
  `& ${PY} ${SK}\\triage.py ledger ${W}\\rules.txt ${W}\\verdicts.txt ${W}\\pr.diff`,
  `& ${PY} ${SK}\\triage.py card ${R}\\card.md ${W}\\verdicts.txt ${W}\\ai_diagnostic.txt ${W}\\answers.txt ${W}\\pr.diff ${W}\\late_findings.txt`,
  `& ${PY} ${SK}\\triage.py refutations ${W}\\refutations.txt ${W}\\pr.diff ${R}\\card.md`,
  `& ${PY} ${SK}\\triage.py independent ${W}\\independent.txt ${R}\\card.md`,
].join('\n  ');

const CTX = `
REVIEW TARGET: ${A.repo} pull request #${A.pr}. Base ${A.baseSha}, head ${A.headSha}.${A.stale ? ' The PR is STALE against the live main tip - any CI result on it describes a tree that has moved.' : ''}

Step 1 of the review-pr skill has ALREADY run at this head. Its artifacts are at ${W}\\ - read them, never re-derive:
  pr.diff (the full base..head diff - ALWAYS Select-String it, never read it whole), pr_meta.json, rules.txt,
  rules_expanded.txt (the full text of exactly the rules this diff derives - your rule book), applies.txt,
  merge_target.txt, guards.txt (deleted asserts: moved / returned changed / gone), siblings.txt, symbols.txt
  (first-party imports that do not resolve against the merge target - a REBASE signal, not invented code), twins.txt,
  test_quality.txt, kernel_tests.txt, ci_coverage.txt, perf_claims.txt, struct_abi.txt, comment_only.txt,
  evidence.txt (cross-file handling of every removed guard and changed signature - READ IT before writing any finding
  about a removed guard), validation_requirement.json,
  merge-target\\ (the BASE worktree - read base files HERE), head\\ (the PR-head post-images).

Use your file and shell tools. Run any python with UTF-8 forced (PYTHONUTF8=1) - the skill's
tooling mis-parses its own rules file otherwise.
Never invent file contents, line numbers or symbols. Answer in PROSE - do not attempt to emit JSON.
`;

const FIND = `
FINDING DISCIPLINE: 🔴 high risk / ⚠️ should fix / 📝 note, advisory only. Before firing a 🔴 you must name the
concrete triggering input (shape / dtype / arch / value / scale); if you cannot, downgrade. Each finding has three
parts: Problem (file:line) + Impact at runtime + Action ending in "**Author must** ..." or
"**Reviewer should ask** ...". Tag [verified] or [inferred].
`;

// ============================================================ S2 analysis
phase('S2 analysis');

const analysisTasks = [
  () => agent(`${CTX}
TASK - Step 2: semantic understanding plus the Step 7.5 blind-spot check. Answer from the DIFF, not the description:
Q1 what specifically changed computationally? Q2 hardware scope (arch, precision, execution phase)? Q3 does this
change any public aiter API? Q4 the performance claim's MECHANISM? Q5 does the description explain WHY or only WHAT?
BLIND: any correctness risk, resource hazard or behavioural edge case steps 1-7 would not catch - a bare "no" is
rejected; say what you looked for and did not find.
Read pr_meta.json for the existing reviewer threads and summarise what reviewers already objected to.
WRITE ${W}\\answers.txt with exactly six lines "Q1: " .. "Q5: " then "BLIND: ", each naming concrete files/symbols.
The BLIND line is a FINDING SOURCE, not commentary: if it names a concrete risk, state it as a finding would be
stated (file, symbol, trigger), because a later step collects findings from this file.`,
    { label: 'S2 semantic', phase: 'S2 analysis', ...WORKER }),

  () => agent(`${CTX}${FIND}
TASK - Step 6: the six structural AI-code checks, one recorded line each ("clean" alone is not an answer).
1 Unresolved imports / hallucinated APIs - symbols.txt holds the static sweep; what is left for you is new kwargs on
  existing calls, new attributes, new enum members - grep those against merge-target\\.
2 Twin divergence - twins.txt and siblings.txt; arch codegen twins and gemm-vs-bmm twins, field by field.
3 Claim vs code and number provenance - does the code enforce what the description asserts? Trace the biggest number
  in perf_claims.txt to a committed script or log; untraceable => [unverified].
4 Safety theater - each new if/try/assert: reachable? will it fire? does an except swallow a real error?
5 Test calibrated to pass rather than to falsify - test_quality.txt; mirrored reference, loosened tolerance,
  self-comparison. Also check for DELETED tests and say what coverage went with them.
6 Magic constants - new tile sizes, thresholds, kid bands, offsets: is a derivation or tuning basis stated?
Count how many cheap signs fire: description explains only WHAT; suspiciously clean perf numbers; screenshot-only
perf; tests only M=1/M=16; gated-off params silently ignored; module-level sys.path/os.environ mutation; unrelated
files; new default path not revertible by an env var; empty Test Plan; AI attribution footer.
WRITE ${W}\\ai_diagnostic.txt with exactly six lines "1: " .. "6: ". Then report each finding.`,
    { label: 'S6 AI diagnostic', phase: 'S2 analysis', ...WORKER }),
];

for (const f of A.files) {
  analysisTasks.push(() => agent(`${CTX}
TASK - Step 4 for exactly ONE file: ${f}
1 Find its hunks: Select-String -Path ${W}\\pr.diff -Pattern '${f}' -Context 0,60 | Select-Object -First 40
2 Decide the tier. TIER1 = if it breaks, "import aiter" fails (only aiter/jit/core.py and aiter/__init__.py qualify).
  TIER2 = holds the Python dispatch selecting which kernel runs for an op class used by more than one production
  model family, OR is the public aiter API for an op, OR is a header included by 10+ translation units.
  TIER3 = an individual op wrapper or kernel.
3 Decide COVERED / GAP / N/A against this PR's own tests (${W}\\test_quality.txt, kernel_tests.txt, ci_coverage.txt).
Return ONE line, exactly, and nothing else:
  ${f} TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason over 30 chars naming a symbol THIS PR changes>`,
    { label: `S4 ${f}`, phase: 'S2 analysis', ...WORKER }));
}

for (const g of A.ruleGroups) {
  analysisTasks.push(() => agent(`${CTX}${FIND}
TASK - Step 5 for THIS SUBSET of the derived rules: ${g.ids}
Read ${W}\\rules_expanded.txt for each id; that text is the rule and the only definition you may use. Adjudicate
EVERY id in your subset. CLEAR is a claim, not a default - it means you looked and it does not apply TO THIS DIFF,
and the reason is what makes it checkable ("ok"/"n/a"/"fine" are rejected by the gate). Use guards.txt, evidence.txt,
siblings.txt, twins.txt, symbols.txt, struct_abi.txt, ci_coverage.txt, test_quality.txt, kernel_tests.txt and
perf_claims.txt rather than re-deriving them. A FIRE must cite at least one file THIS PR changes.
WRITE ${W}\\verdicts_${g.id}.txt, one line per rule id, exactly:
  <RULE-ID> FIRE|CLEAR|N/A — <specific reason naming file:line, symbol, or the condition>`,
    { label: `S5 rules ${g.id}`, phase: 'S2 analysis', ...WORKER }));
}

log(`S2: ${analysisTasks.length} worker agents (2 + ${A.files.length} files + ${A.ruleGroups.length} rule groups)`);
const analysis = await parallel(analysisTasks);

// FAIL CLOSED (review fix #4). A silently failed analysis agent means a missing verdict file or a
// missing core-file line, which the gates then blame on the card - and the card's author is the
// only one who can "fix" it, by writing the missing adjudication itself. Stop instead.
const dead = analysis.map((r, i) => (r ? null : i)).filter(i => i !== null);
if (dead.length) {
  const names = dead.map(i => analysisTasks[i].label || `#${i + 1}`);
  throw new Error(`S2: ${dead.length} of ${analysisTasks.length} analysis agents produced nothing ` +
    `(indices ${dead.join(', ')}). Their artifacts are missing, so the rule ledger and the core-file ` +
    `ledger are incomplete. Refusing to continue: re-run those agents rather than letting the card ` +
    `author invent the missing adjudications.`);
}

const analysisText = analysis.map(String).join('\n\n---\n\n');
const s4 = analysis.slice(2, 2 + A.files.length).map(String);

await agent(`Mechanical extraction and file writing only - do not analyse, do not invent a line for a file that has none.
Each report below covers ONE file and contains one ledger line shaped
  <path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason>
Extract exactly that line from each (ignore surrounding prose) and WRITE them one per line to ${W}\\core_files.txt,
reasons verbatim. Read the file back and report its line count, which must be ${A.files.length}.

REPORTS:
${s4.map((t, i) => `### ${A.files[i]}\n${t.slice(0, 3000)}`).join('\n\n')}`,
  { label: 'write core_files.txt', phase: 'S2 analysis', ...FORMATTER });

// ============================================================ S3 refutation
phase('S3 refutation');

// The minimum finding count is DERIVED, not guessed (review fix #8): every FIRE is a finding the
// collector must have carried, so a collection that returns fewer has lost some.
const fireCount = await agent(`Counting only - change no files, analyse nothing. Report exactly one line:
  FIRE=<the number of lines across ${W}\\verdicts_*.txt that contain the word FIRE>
For example (PowerShell): (Select-String -Path ${W}\\verdicts_*.txt -Pattern 'FIRE').Count`,
  { label: 'count FIRE verdicts', phase: 'S3 refutation', ...FORMATTER });
const minFindings = Number((String(fireCount || '').match(/FIRE=(\d+)/) || [])[1] || 0);
log(`S3: ${minFindings} FIRE verdicts - the collector must return at least that many findings`);

const collected = await agent(`Mechanical extraction only - do not analyse, do not add findings of your own.
Read and extract every distinct FINDING asserted by:
  ${W}\\verdicts_A_B.txt and every other ${W}\\verdicts_*.txt   (every line marked FIRE)
  ${W}\\ai_diagnostic.txt   (any defect the six structural checks name)
  ${W}\\core_files.txt      (any line marked GAP)
  ${W}\\answers.txt         (the BLIND: line - if it names a concrete risk, that is a finding too)
Each entry: <SEVERITY>|<one self-contained sentence naming the file:line and what is wrong>, where SEVERITY is
RED, WARN or NOTE. It must be understandable by someone who has read none of those files. Merge duplicates that
describe the same defect. Exclude CLEAR and N/A verdicts.

Write the list, and ONLY the list, between markers - nothing before the first, nothing after the last:
BEGIN_FINDINGS
<SEVERITY>|<sentence>
END_FINDINGS
Also WRITE the same list to ${W}\\findings_raw.txt so it survives if this answer is lost.`,
  { label: 'collect findings', phase: 'S3 refutation', ...FORMATTER });

const between = String(collected || '').split('BEGIN_FINDINGS')[1];
const findings = String(between || '').split('END_FINDINGS')[0]
  .split('\n').map(s => s.trim()).filter(s => /^(RED|WARN|NOTE)\s*\|/i.test(s));

// FAIL CLOSED (review fix #2/#8). An empty or short collection is a lost collection, not a clean PR.
if (findings.length < Math.max(1, minFindings)) {
  throw new Error(`collect findings returned ${findings.length} findings but the verdict files contain ` +
    `${minFindings} FIRE lines. Refusing to continue: an incomplete refutation stage cannot certify a card.`);
}
log(`S3: ${findings.length} findings, one refuter each (~${findings.length * 15}k input tokens estimated)`);

// Resume support (review fix #7): a refuter that already wrote its own file is not re-run.
let already = [];
if (A.resume) {
  const existing = await agent(`Listing only - change no files. Report exactly one line:
  EXISTING=<comma-separated basenames of the *.md files in ${L}, or the word NONE>`,
    { label: 'check resumable refutations', phase: 'S3 refutation', ...FORMATTER });
  already = (String(existing || '').match(/F\d+\.md/g) || []).map(s => s.replace(/\D/g, ''));
  if (already.length) log(`S3: resuming - ${already.length} refuter files already present, skipping those`);
}

const refuteTasks = findings.map((f, i) => {
  const nn = String(i + 1).padStart(2, '0');
  if (already.includes(nn)) return () => Promise.resolve(`SKIPPED (F${nn}.md already present)`);
  return () => agent(`${CTX}

YOU ARE AN ADVERSARIAL REFUTER and you have NOT seen the reasoning behind this finding. Treat it as FALSE UNTIL
DEFENDED and try to KILL it with evidence. Do not improve it, do not agree politely - attack it.

FINDING F${i + 1}: ${f}

Kills to try: the symbol or guard MOVED rather than being deleted (grep the whole family, headers and callers
included); it still exists or is re-exported elsewhere; an earlier validation makes the claimed trigger unreachable;
the behaviour is pre-existing in merge-target\\ and this PR did not change it; the cited line or symbol does not
exist; the asymmetry is a legitimate hardware-contract difference; the tool or scan that produced it matched
positionally or by a regex that discards meaning. It SURVIVES only if you opened the evidence and could not kill it.
A RED survives only if you can verify a CONCRETE triggering input; otherwise say DOWNGRADE.

SEVERITY BAR (fixed, so that rounds are comparable): a 🔴 requires a triggering point verifiable IN THIS TREE. A
protocol documented outside the tree (a README telling downstream integrations to call something) is real evidence
but caps the finding at ⚠️ with an action asking the author to confirm the downstream.

Use a handful of targeted Select-String / read calls; keep every command's output small.

THEN - mandatory, and this file is YOURS; nobody else may write it - use the write tool to create
  ${L}\\F${nn}.md
whose FIRST LINE is exactly one line in this format and nothing else:
  ${f.split('|')[0].trim()} <SURVIVED|KILLED> -- <what you opened and what it said, one line, no newlines>
and whose remaining lines are your full evidence: the finding restated, the files and line numbers you opened, the
kill attempts you made, and your keep / downgrade / drop recommendation.
Finally report the same verdict in your answer.`,
    { label: `refute F${i + 1}`, phase: 'S3 refutation', ...REFUTER });
});

const refuted = await parallel(refuteTasks);
log(`S3: ${refuted.filter(Boolean).length}/${findings.length} refuters returned`);

await agent(`Mechanical assembly only - do not analyse, do not add, drop or reword any entry.
The directory ${L}\\ contains one file per finding, F01.md .. F${String(findings.length).padStart(2, '0')}.md, each
whose FIRST LINE is that finding's ledger line.
1. List the directory; report how many files are present and name any missing one - do NOT invent a line for it.
2. WRITE ${W}\\refutations.txt: the first line of each file, in numeric file order, one per line, verbatim.
3. WRITE ${W}\\independent.txt: the same lines with the leading severity word removed.
4. WRITE ${R}\\findings.md: readable markdown concatenating every file's full content under a heading naming its
   finding number and verdict. This is the companion to the five-finding card; nothing is dropped here.
5. Create ${W}\\late_findings.txt if absent - keep any existing content. Report the final line counts.`,
  { label: 'assemble ledger', phase: 'S3 refutation', ...FORMATTER });

const verify = await agent(`Counting only - change no files. Report exactly these four lines:
  FILES=<number of *.md files in ${L}>
  REFUT=<(Get-Content ${W}\\refutations.txt | Measure-Object -Line).Lines>
  INDEP=<(Get-Content ${W}\\independent.txt | Measure-Object -Line).Lines>
  BAD=<lines in refutations.txt NOT matching '^(RED|WARN|NOTE) (SURVIVED|KILLED) -- '>`,
  { label: 'ledger provenance', phase: 'S3 refutation', ...FORMATTER });

const vm = String(verify || '');
const nfiles = Number((vm.match(/FILES=(\d+)/) || [])[1] || 0);
const nrefut = Number((vm.match(/REFUT=(\d+)/) || [])[1] || 0);
const nindep = Number((vm.match(/INDEP=(\d+)/) || [])[1] || 0);
const nbad = Number((vm.match(/BAD=(\d+)/) || [])[1] || 999);
if (nfiles !== findings.length || nrefut !== findings.length || nindep !== findings.length || nbad !== 0) {
  throw new Error(`ledger provenance failed: ${nfiles} refuter files, ${nrefut} refutation lines, ` +
    `${nindep} independent lines, ${nbad} malformed, for ${findings.length} findings. ` +
    `Refusing to certify a card on a ledger whose provenance cannot be established.`);
}
log(`S3: provenance ok - ${nfiles} refuter-written files, ${nrefut} ledger lines, 0 malformed`);

// ============================================================ S4/S5 card and gates
phase('S4/S5 card and gates');

const SEPARATION = `
SEPARATION OF DUTIES - ABSOLUTE. You may NOT create, edit, append to or delete any of:
  ${W}\\refutations.txt , ${W}\\independent.txt , ${R}\\findings.md , anything under ${L}\\ ,
  and you may NOT add a verdict line to ${W}\\verdicts.txt or any ${W}\\verdicts_*.txt for a rule that has none.
Those files are other parties' records: the refuter's, and the rule adjudicators'. A party that writes its own
review record has destroyed the only guarantee this loop offers. If a gate complains about one of them, do NOT fix
the file: change the CARD so it only claims what the existing records support (drop the finding, or add a
"-- not reported:" line), and say in your report that the record was left untouched. If a rule genuinely has no
verdict, that is a failed analysis agent - report it as a blocker instead of adjudicating it yourself.
`;

let notes = '';
let gatesGreen = false;

for (let round = 1; round <= MAX_ROUNDS; round++) {
  await agent(`${CTX}${FIND}

TASK - Step 8: write the verdict card, then make the gates green.
${round === 1 ? '' : `\nTHIS IS ROUND ${round}. The gates rejected the previous attempt. Their complaints, verbatim - fix exactly what they name and nothing else:\n${notes}\n`}

THE LEDGER IS AUTHORITATIVE: ${W}\\refutations.txt has one line per finding, written by an independent refuter that
never saw the analysis. Read it first. A KILLED finding must NOT reach the card; a line advising DOWNGRADE must be
honoured. Full evidence per finding is in ${R}\\findings.md.

Preparation: concatenate every ${W}\\verdicts_*.txt into ${W}\\verdicts.txt, preserving every line.

Write ${R}\\card.md (read pr_meta.json for the title), header exactly:

## [aiter] PR #${A.pr} — <title>

**<one sentence: what this PR does, for a reviewer who has not read the diff>**

Review (advisory): <✅ NO FINDINGS | ⚠️ NEEDS WORK | 🔴 HIGH RISK>${A.stale ? ' — the PR is STALE against the live main tip, so any CI result on it describes a tree that has moved' : ''}
Validation (deterministic): PENDING
Perf (advisory): PENDING
${A.sinceHead ? `Since last review (${A.sinceHead}): <what the author changed since the previously reviewed head, from the delta the fetch phase reported - name any test or guard that was added or removed; this is invisible in the diff once the base moves>\n` : ''}
then at most FIVE findings (🔴/⚠️/📝, most severe first), then "-- not reported: <reason>" accounting lines for
every FIRE rule and every surviving finding below the cut, plus any "-- late finding:" lines already recorded.
${SEPARATION}
THEN run the gates with UTF-8 forced (PYTHONUTF8=1), one at a time, reporting output and $LASTEXITCODE verbatim:
  ${GATES}
Do not fabricate a pass.`, { label: `card + gates (round ${round})`, phase: 'S4/S5 card and gates', ...WORKER });

  const check = await agent(`Run these seven commands with UTF-8 forced (PYTHONUTF8=1), one at a time, and report each
one's output and $LASTEXITCODE. Change no files.
  ${GATES}
Also report (Get-Content ${W}\\refutations.txt | Measure-Object -Line).Lines so we can confirm the ledger was not
rewritten - it must still be ${findings.length}.
End with the single word ALLGREEN if and only if all seven exit codes were 0.`,
    { label: `gate check (round ${round})`, phase: 'S4/S5 card and gates', ...FORMATTER });

  notes = String(check || '').slice(0, 6000);
  const ledgerStillIntact = new RegExp(`\\b${findings.length}\\b`).test(notes);
  if (/ALLGREEN/.test(notes) && ledgerStillIntact) {
    gatesGreen = true;
    log(`gates green at round ${round}`);
    break;
  }
  log(`gates not green at round ${round} - feeding the complaint back`);
}

// FAIL CLOSED (review fix #3). An ungated card is not a deliverable, and running validation on top
// of one wastes GPU time on a review nobody can trust.
if (!gatesGreen) {
  throw new Error(`the seven gates did not go green within ${MAX_ROUNDS} rounds. Last complaint:\n${notes}\n` +
    `Refusing to continue to validation: an ungated card must not be delivered.`);
}

// ============================================================ S6 validation
phase('S6 validation');

const results = [];
if (A.validationRequired && A.targets && A.targets.length) {
  for (const tg of A.targets) {
    const v = await agent(`
Use your shell tool locally and ssh for the remote side:
  ${SSH_TO(tg)} 'bash <script>'
Write remote scripts locally and scp them into the scratch dir; never inline multi-line remote commands; the
Conductor banner makes ssh exit codes unreliable, so judge by stdout. Keep outputs small.
IF ssh is refused by the local harness with a sandbox-backend error, SAY SO IMMEDIATELY AND STOP - do not silently
skip the task and do not report success.

TASK: run validate-kernel-pr for target ${tg.t} on host ${tg.host}, inside container ${tg.container}.
PR: ${A.repo}#${A.pr}, head ${A.headSha}, base ${A.baseSha}.
Container-side paths: worktree ${tg.worktree}, patch ${tg.patch}, scratch ${tg.scratch}. Host PR body: ${tg.hostRoot}.

${tg.refresh ? `THIS HOST HOLDS AN OLDER ROUND'S STATE and must be refreshed first: fetch the new head
(\`git fetch origin pull/${A.pr}/head:pr${A.pr} --force\`) and origin/main; remove the old worktree
(\`git worktree remove --force\`; if permissions deny it, clean from inside the container - in-container runs write as
root - and chown back to ${A.uidGid}, never sudo); verify merge-base(origin/main, pr${A.pr}) == ${A.baseSha};
regenerate the patch; create a fresh detached worktree at ${A.baseSha}; verify both \`git status --porcelain\` and
\`--porcelain --ignored\` are empty and \`git apply --check\` passes.
` : `Verify the worktree is at base ${A.baseSha}, both \`git status --porcelain\` and \`--porcelain --ignored\` are
empty, and \`git apply --check\` passes.
`}ALWAYS verify the 3rdparty/composable_kernel submodule is initialised at the repo-pinned sha; skipping it produces a
false BLOCK ("fatal error: 'ck_tile/core.hpp' file not found") that looks exactly like a PR defect.

HARD GATE BEFORE LAUNCH: \`/opt/rocm/bin/rocm-smi --showpids\` must report no KFD processes (rocm-smi is not always on
PATH; use the full path). Quote it verbatim. If anything holds the GPU, do NOT launch - report the occupancy as an
environment gap and stop. Also report \`--showuse --showmemuse\`.

Then launch in the BACKGROUND (nohup setsid):
  docker exec -w ${tg.worktree} -e PYTHONDONTWRITEBYTECODE=1 ${tg.container} \\
    bash .claude/skills/validate-kernel-pr/validate_pr.sh \\
      --repo ${tg.worktree} --patch ${tg.patch} \\
      --head-sha ${A.headSha} --target ${tg.t} \\
      --label pr${A.pr} \\
      --out ${tg.scratch}/validation_${tg.t.replace(/[^A-Za-z0-9]/g, '_')}.json
PYTHONDONTWRITEBYTECODE=1 is REQUIRED: the validator's own GPU-picker import otherwise writes __pycache__ into the
worktree, and its own isolation check then reads the tree as dirty and skips every correctness and perf stage.
Choose --expected-route by reading the target's HEAD source (a function whose own frame binds the shape variables)
and justify it. Add --shape-argnames / --shape-env ONLY if you can point at the consuming code. Use --no-perf only if
the target has no benchmark entry point at all. Note the validator runs pytest with -x, so a failure hides whatever
would have run after it - say so explicitly if a failure occurs.

Wait with ONE blocking remote poll (pgrep every 30s, up to ~30 minutes), then report: .verdict, every stage name +
status + note, .findings verbatim, .test_selection, .execution_receipt, .stages.correctness_repo_tests head and base
counts, .stages.perf, .arch_coverage, and the wall time. Copy the report back to ${R}\\ and give its local path.
If no report exists, say NO REPORT and why - never infer a verdict from another target.
Treat the executor's own findings as CLAIMS, not facts.`,
      { label: `validate ${tg.t} on ${String(tg.host).split('.')[0]}`, phase: 'S6 validation', ...REMOTE });
    results.push({ target: tg.t, host: tg.host, report: String(v || 'NO REPORT - the validation agent produced nothing') });
  }
} else {
  log('S6 skipped: validation not required, or no target was chosen');
}

// ============================================================ S6b executor refutation
if (results.length) {
  phase('S6b executor refutation');
  const execRefuted = await parallel(results.map((r, i) => () => agent(`
You are an ADVERSARIAL REFUTER. Below is a validate-kernel-pr report for ${A.repo}#${A.pr} (head ${A.headSha}),
target ${r.target}, host ${r.host}.

Every finding the EXECUTOR raised is a CLAIM, not a fact: the tool can be wrong about the PR. Treat each as FALSE
UNTIL DEFENDED and attack it. Consider specifically whether the TOOL manufactured it: does its extractor match by
position rather than by name, does a regex capture only the first literal of a conditional, does a cleanliness check
count artifacts the tool itself created, is a "skip" a property of the invocation rather than of the PR?
The executor's source is at ${VSK} and the PR artifacts (pr.diff, head\\, merge-target\\)
are at ${W}\\ - open them and compare like with like. Re-derive rather than assume.

REPORT:
${r.report.slice(0, 12000)}

For EACH executor finding report SURVIVED or KILLED, the deciding file:line you opened, and keep/downgrade/drop.
If there are no findings, say what the run positively establishes and what it does not.`,
    { label: `refute executor: ${r.target}`, phase: 'S6b executor refutation', ...REFUTER })));

  // ========================================================== S7 refold
  phase('S7 refold');

  await agent(`${CTX}${FIND}

TASK: fold the deterministic validation evidence into ${R}\\card.md, then re-gate.

VALIDATION REPORTS:
${results.map(r => `TARGET ${r.target} (host ${r.host}):\n${r.report.slice(0, 5000)}`).join('\n\n')}

INDEPENDENT REFUTATION OF THE EXECUTOR'S OWN FINDINGS:
${execRefuted.map((r, i) => `${results[i].target}:\n${String(r || 'NO VERDICT').slice(0, 2000)}`).join('\n\n')}

Rules:
- Replace "Validation (deterministic): PENDING" with the real state: each target's verdict, the host architecture,
  which stages passed and which skipped with their real reasons. If a stage failed but the refutation KILLED it as a
  tool artifact, say so on that line - a NEEDS_WORK caused by the validator's own extractor is NOT a PR defect. A
  target with no report is recorded as NO REPORT with the reason; never infer it from another target.
- Replace "Perf (advisory): PENDING": a measured ratio only if stages.perf carries median_ratio, else
  "NOT RUN — <the real reason>". Never label a hand-run number deterministic.
- A KILLED executor finding must not become a card finding: record it as "-- late finding: <withdrawal + evidence>"
  and append the same line to ${W}\\late_findings.txt.
- A SURVIVED executor finding may take a card slot on severity; keep the cap at five and account for what it displaces.
${SEPARATION}
Then re-run all seven gates and report each exit code verbatim:
  ${GATES}`,
    { label: 'fold validation into card', phase: 'S7 refold', ...WORKER });

  // Independent re-verification (review fix #9): the refold agent must not be the only witness
  // that the card still passes after it edited the card.
  const finalCheck = await agent(`Run these seven commands with UTF-8 forced (PYTHONUTF8=1), one at a time, and report
each output and $LASTEXITCODE. Change no files.
  ${GATES}
Also report (Get-Content ${W}\\refutations.txt | Measure-Object -Line).Lines - it must still be ${findings.length}.
End with the single word ALLGREEN if and only if all seven exit codes were 0.`,
    { label: 'final independent gate check', phase: 'S7 refold', ...FORMATTER });

  if (!/ALLGREEN/.test(String(finalCheck || ''))) {
    throw new Error(`the gates do not pass after the validation refold:\n${String(finalCheck || '').slice(0, 4000)}\n` +
      `The card was edited into a state it cannot defend; fix the card before delivering.`);
  }
  log('S7: gates green after refold, independently verified');
}

// ============================================================ S8 cleanup
phase('S8 cleanup');

const hostKeys = [...new Set((A.targets || []).map(t => `${t.host}|${t.container}|${t.scratch}|${t.worktree}`))];
const cleaned = await parallel(hostKeys.map(h => {
  const [host, container, scratch, worktree] = h.split('|');
  return () => agent(`
Use your shell tool locally and ssh for the remote side (see below).
Judge success by stdout. IF ssh is refused by the local harness with a sandbox-backend error, SAY SO AND STOP -
a silently skipped cleanup is a failed round, not a clean one.

TASK: leave ${host} clean after this round.
1. Copy every validation_*.json and validator log out of ${scratch} (container ${container}) to ${R}\\ - report what.
2. Delete the scratch: \`docker exec ${container} rm -rf ${scratch}\`.
3. Clean the worktree of root-owned in-container artifacts:
   \`docker exec ${container} bash -lc 'cd ${worktree} && git clean -xdf && git status --porcelain --ignored'\`
   - the last command must print nothing - and chown anything still root-owned back to ${A.uidGid}. Report the
   root-owned count before and after; it must end at 0.
4. Remove any script you scp'd to the host, naming each. Nothing of ours may remain in the host /tmp.
5. Leave the PR body (clone, worktree, patch) and the container itself in place; confirm you touched no other user's
   container or files.
6. Report the final \`/opt/rocm/bin/rocm-smi --showpids\` reading.`,
    { label: `cleanup ${String(host).split('.')[0]}`, phase: 'S8 cleanup', ...REMOTE });
}));

const cleanupFailed = cleaned.filter(c => !c).length;
if (cleanupFailed) log(`S8: WARNING - ${cleanupFailed} cleanup agent(s) produced nothing; the remote side may be dirty`);

return {
  findings: findings.length,
  fireVerdicts: minFindings,
  refuterFiles: nfiles,
  ledgerLines: nrefut,
  gatesGreen,
  validated: results.map(r => `${r.target} @ ${String(r.host).split('.')[0]}`),
  cleanupFailed,
  card: `${R}\\card.md`,
  findingsFile: `${R}\\findings.md`,
};
