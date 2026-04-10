# AITER Release Process

## Release Cycle

AITER follows a time-based release cycle. Each release goes through:

```
main (development) → release branch → RC tags → final tag → GitHub Release
```

### Timeline

| Phase | Duration | Activities |
|-------|----------|-----------|
| Development | Ongoing | PRs merge to `main` |
| Release prep | 1-2 days | Create release branch, identify blockers |
| RC validation | 2-3 days | Downstream testing, fix blockers |
| Release | 1 day | Tag, build, publish |

## Step-by-Step Release Procedure

### 1. Create Release Branch

Pick a stable commit on `main` (ideally one that passed nightly CI):

```bash
git checkout <commit-hash>
git checkout -b release/vX.Y.Z
git push origin release/vX.Y.Z
```

CI workflows automatically trigger on `release/**` branches.

### 2. Identify and Fix Blockers

- Check open issues labeled `blocker`
- Review accuracy test results (SGLang, ATOM)
- Cherry-pick fixes from `main` to release branch if needed
- Do NOT merge new features into the release branch

### 3. Release Candidate

Tag a release candidate for downstream testing:

```bash
git tag vX.Y.Z-rc1 release/vX.Y.Z
git push origin vX.Y.Z-rc1
```

Notify downstream teams to test against the RC:
- SGLang team: accuracy tests (DSR1, GPT-OSS)
- vLLM team: benchmark regression check
- ATOM team: E2E model tests

If issues found, fix on release branch and tag `rc2`, `rc3`, etc.

### 4. Final Release

When RC is validated:

```bash
git tag vX.Y.Z release/vX.Y.Z
git push origin vX.Y.Z
```

This automatically triggers `aiter-release.yaml` which:
1. Builds .whl for Python 3.10 and 3.12 (gfx942 + gfx950)
2. Runs smoke test (install + import validation)
3. Generates changelog
4. Creates GitHub Release with .whl assets attached

### 5. Post-Release

- Notify downstream teams of the release
- Update pinned AITER versions in downstream projects
- Close related GitHub issues

## Release Notes

### Auto-Generated

Run the changelog script to generate categorized release notes:

```bash
./scripts/generate_changelog.sh <previous_tag> <new_tag> RELEASE_NOTES_vX.Y.Z.md
```

Categories: Features, Performance, Bug Fixes, Refactoring, CI/Infrastructure.

### Hand-Written Highlights

Edit the generated `RELEASE_NOTES_vX.Y.Z.md` to add a Highlights section at the top. The release workflow will prefer a hand-written file if it exists.

## Hotfix / .postX Releases

### When to Use

`.postX` releases are for **critical hotfixes only**:
- Accuracy regressions caught after release
- Build/install breakage
- Security issues

Do NOT use `.postX` for new features or non-critical improvements.

### Procedure

```bash
# Work on the existing release branch
git checkout release/vX.Y.Z
git cherry-pick <fix-commit-from-main>
git tag vX.Y.Z.postN
git push origin vX.Y.Z.postN
```

### Rules

- Maximum 3 `.postX` per release. If more needed, bump to `vX.Y.(Z+1)`.
- Every `.postX` MUST have a GitHub Release with description of what changed.
- `.postX` MUST only cherry-pick from `main` — no direct commits to release branch without corresponding main PR.

## Version Numbering

AITER uses `setuptools_scm` — version is derived from git tags automatically.

| Format | Meaning | Example |
|--------|---------|---------|
| `vX.Y.Z` | Regular release | `v0.1.12` |
| `vX.Y.Z.postN` | Hotfix release | `v0.1.12.post1` |
| `vX.Y.Z-rcN` | Release candidate | `v0.1.12-rc1` |
| `vX.Y.Z.devN+gHASH` | Development build | Auto-generated between tags |

## Release Checklist

See `scripts/release_checklist.md` for the full pre/post-release validation checklist.

## Downstream Test Coverage

| Project | CI Workflow | Release Gate? |
|---------|-----------|--------------|
| SGLang | `sglang_downstream.yaml` | Yes — accuracy must pass |
| vLLM | `vllm_benchmark.yaml` | Yes — no regression |
| ATOM | `atom-test.yaml` | Yes — E2E pass |
| TE | TBD | Not yet |
| Megatron | TBD | Not yet |
| hipblaslt | TBD | Not yet |
| Flashinfer | TBD | Not yet |
| Primus | TBD | Not yet |

## CI Resource Notes

- Build: ~1h (prebuild kernels)
- Standard tests: ~40min
- Total per PR: ~1h40m minimum
- Release branch CI runs on push and on PRs targeting `release/**` branches
- Runner fleet analytics: tracked via CI monitor workflow (#2606)
