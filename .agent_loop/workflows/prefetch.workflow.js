// .agent_loop — PHASE P (prefetch).
//
// The cheap stage that must run BEFORE the single human break: the questions the loop owes a
// human (which validation target? what will refutation cost?) can only be asked concretely once
// review-pr's Step 1 has produced validation_requirement.json.
//
// This is a DSH `workflow` script body: the harness supplies `args`, `agent`, `parallel`,
// `phase` and `log`. It has no filesystem of its own - every path arrives through args, and the
// agents do the file work. Nothing in this file is installation-specific; see config/loop.example.json.
//
// args = {
//   pr, repo,                        // 4961, "ROCm/aiter"
//   host: { name, sshHost, user, uidGid, container, image, hostRoot, containerMount, scratch },
//   local: { ssh, scp, python, identityFile, toolsRoot },
//   localWork,                       // where the $WORK scratch is synced to on this machine
//   runId,
//   provider, model,                 // the remote-exec role
//   envKnown,                        // true when a confirmed env record already exists
//   sinceHead                        // previously reviewed head sha, or null on a first round
// }

const A = args;
const H = A.host;
const LOCAL = A.local || {};
const REMOTE = { provider: A.provider, model: A.model };

const SSHX = LOCAL.ssh || 'ssh';
const SCPX = LOCAL.scp || 'scp';
const IDENT = LOCAL.identityFile ? `-i ${LOCAL.identityFile} -o IdentitiesOnly=yes ` : '';
const SSH_TO = `${SSHX} ${IDENT}-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30 ${H.user}@${H.sshHost}`;
const SCRATCH = `${H.scratch}`;

const PRIMER = `
You drive a remote Linux GPU host from this machine. Use your shell tool for local commands and ssh for remote ones:
  ${SSH_TO} 'bash <script>'
  ${SCPX} ${IDENT}<local> ${H.user}@${H.sshHost}:<remote>

RULES OF THE ROAD (each one exists because it has already gone wrong):
  - Never inline a multi-line or quote-heavy remote command; write the script to a local file, copy it over, and run
    \`bash <path>\`. Quoting is mangled otherwise.
  - Some hosts print a long login banner on every connection, which makes ssh exit codes unreliable. Judge success by
    stdout content, not by the exit code.
  - IF ssh is refused by your own harness (a sandbox or permission error on the LOCAL side), SAY SO IMMEDIATELY AND
    STOP. Do not silently skip the task and do not report success: a skipped step reported as done is worse than a
    failed step reported as failed.
  - Keep every command's output small.

PLACEMENT RULES - hard constraints:
  - Every temporary script and every test artifact goes in the container scratch ${SCRATCH}. Writing scratch to the
    remote host's /tmp is forbidden (beyond a single script you delete afterwards).
  - The PR body (clone, base worktree, patch) lives on the host at ${H.hostRoot}, bind-mounted into container
    ${H.container} at ${H.containerMount}.
  - All PR testing runs inside container ${H.container}. The host does git and docker only.
`;

phase('env');

const env = await agent(`${PRIMER}

TASK: ${A.envKnown ? 'VERIFY the environment contract still holds (it was established and confirmed earlier) and report any drift.' : 'Establish the environment contract and gather the facts a human needs in order to confirm it.'}

1. ssh reachability, hostname, GPU architecture (\`/opt/rocm/bin/rocminfo | grep -o 'gfx[0-9a-z]*' | sort -u\`).
2. Per-GPU busy % AND VRAM used (\`rocm-smi --showuse --showmemuse\`) plus \`rocm-smi --showpids\`. Name the indices
   that are genuinely idle. rocm-smi is not always on PATH - prefer /opt/rocm/bin/rocm-smi. A GPU at 0% busy but high
   VRAM is NOT usable for validation; say so rather than calling it idle.
3. Container ${H.container}: does it exist, and is it RUNNING? Verify inside it that torch, aiter and pytest import,
   and report versions and the aiter path. Do NOT create, start, stop or remove any container - if it is stopped,
   report that and stop; restarting someone's container is the human's decision, not yours.
4. Host tooling: git, python3, docker, flock, jq, gh (and whether gh is authenticated - it usually is not). Free disk
   on ${H.hostRoot}'s filesystem. Whether the host reaches GitHub.
5. What is already under ${H.hostRoot} from an earlier round: the clone (origin, branch), whether a worktree exists
   and is clean, and whether root-owned artifacts or old scratch directories remain. Report, do not clean yet.

Report a prose summary and, explicitly, ANY blocker that would stop the run phase.`, {
  label: 'env contract', phase: 'env', ...REMOTE
});

log(`env -> ${env ? 'reported' : 'FAILED'}`);
if (!env) throw new Error('the environment phase produced nothing; refusing to fetch against an unverified host');

phase('fetch');

const fetched = await agent(`${PRIMER}

Environment facts just established:
${String(env).slice(0, 4000)}

TASK: run Step 1 of the review-pr skill for ${A.repo}#${A.pr} at its CURRENT head, and bring the artifacts back.
${A.sinceHead ? `This PR has been reviewed before at head ${A.sinceHead}; it has since moved. Nothing may be reused from the previous fetch.\n` : ''}
Preconditions - do them, do not assume:
1. Under ${H.hostRoot}: reuse the existing clone of https://github.com/${A.repo}.git at ${H.hostRoot}/aiter (clone it
   if absent). \`git fetch origin pull/${A.pr}/head:pr${A.pr} --force\` and \`git fetch origin main\`.
2. Remove any PREVIOUS round's worktree at ${H.hostRoot}/pr-${A.pr} (\`git worktree remove --force\`). In-container
   runs write as root, so if removal is denied by permissions, clean it from inside the container
   (\`docker exec ${H.container} bash -lc 'cd <container-side worktree> && git clean -xdf'\`) and chown back to
   ${H.uidGid} - never use sudo.
3. BASE = \`git merge-base origin/main pr${A.pr}\`. Use the MERGE-BASE, not today's main tip: a stale PR does not apply
   to the tip, and the merge-base is the commit its diff actually describes. Record BASE and the head sha.
4. The patch: \`git diff --no-color <BASE> pr${A.pr} > ${H.hostRoot}/pr-${A.pr}.patch\`. Record line count and sha256.
5. A fresh detached BASE worktree at ${H.hostRoot}/pr-${A.pr}, verified clean by BOTH \`git status --porcelain\` and
   \`--porcelain --ignored\`, with \`git apply --check\` proving the patch applies. Do NOT leave the patch applied.
6. INITIALISE the \`3rdparty/composable_kernel\` submodule at the repo-pinned sha in that worktree. Skipping it makes
   the JIT build fail with "fatal error: 'ck_tile/core.hpp' file not found", which looks exactly like a PR defect and
   has already produced one false BLOCK.

Then run the skill's Step 1:
7. If gh is missing or unauthenticated - and note that a large PR's diff exceeds GitHub's 20000-line API cap, which
   makes \`gh pr diff\` fail with HTTP 406 even when gh works - install the drop-in replacement shipped with this loop:
   copy ${LOCAL.toolsRoot}/gh_shim.py into ${SCRATCH}/bin/gh_shim.py, create ${SCRATCH}/bin/gh as a two-line sh
   wrapper (\`#!/bin/sh\` then \`exec python3 .../gh_shim.py "$@"\`), chmod +x, prepend that dir to PATH, and set
   GH_SHIM_REPO_DIR to the clone. It answers pr view / pr diff / api / issue view from the unauthenticated REST API
   and produces the diff locally with git.
8. Run INSIDE container ${H.container}, cwd at the clone, with \`PYTHONUTF8=1 PYTHONIOENCODING=utf-8\` (without them
   triage.py mis-parses rules.md and reports MISSING-RULE-TEXT) and \`PYTHONDONTWRITEBYTECODE=1\` (the skill's own
   tooling otherwise drops __pycache__ into the tree):
     bash .claude/skills/review-pr/fetch.sh ${A.pr} ${A.repo}
   Capture output to ${SCRATCH}/fetch.log and take the scratch dir it prints.
   Consider setting REVIEW_AUTO_VALIDATE=0: fetch.sh will otherwise auto-run the GPU validator in its own worktree,
   which has no initialised submodule and pre-empts the human's target choice. If you set it, say so and why.
9. VERIFY before believing it: pr.diff non-empty, rules.txt non-empty, rules_expanded.txt present with NO
   "MISSING-RULE-TEXT". Fix and re-run if not.
10. Bring the scratch dir back to ${A.localWork} (tar in the container, copy out, extract locally) and verify the
    local file sizes match the in-container ones.

REPORT precisely - the human's next decision depends on it:
  - BASE sha, head sha, patch line count and sha256, whether apply --check passed;
${A.sinceHead ? `  - THE CROSS-ROUND DELTA, which the diff cannot show: \`git log --oneline ${A.sinceHead}..pr${A.pr}\` and
    \`git diff --stat ${A.sinceHead} pr${A.pr}\`. Separate the author's own commits from commits that arrived via a
    merge of main, and NAME EVERY TEST FILE OR GUARD ADDED OR REMOVED. Once the base moves, a file the previous head
    added and this head removed is invisible in the diff - it reads as if it never existed. State plainly whether the
    author fixed what the last review raised or removed the thing that exposed it;
  - whether the old head is an ancestor of the new head (a normal update) or the branch was force-pushed;
` : ''}  - whether applies.txt says the PR is STALE against the merge target, quoted;
  - the FULL contents of validation_requirement.json: required, runtime_path_count, candidates, blocking_reason,
    perf_required, perf_command;
  - the rule ids in rules.txt and how many were derived out of how many;
  - the changed-file list, and which of them are backbone files (aiter/jit/core.py, aiter/__init__.py,
    aiter/tuned_gemm.py, aiter/ops/gemm_op_a8w8.py, aiter/ops/batched_gemm_op_a8w8.py, aiter/ops/quant.py,
    aiter/fused_moe.py, aiter/ops/mha.py, aiter/ops/attention.py, aiter/mla.py, aiter/ops/moe_op.py,
    csrc/include/rocm_ops.hpp);
  - anything that failed.`, {
  label: 'skill Step 1 (fetch)', phase: 'fetch', ...REMOTE
});

log(`fetch -> ${fetched ? 'reported' : 'FAILED'}`);
if (!fetched) throw new Error('the fetch phase produced nothing; there is nothing for the human break to decide on');

// The caller turns these two prose reports into the single human question batch.
return { env, fetched };
