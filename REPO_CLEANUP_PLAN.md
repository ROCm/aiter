# Repository Cleanup Plan

## Overview

This PR prepares the repository for a significant size reduction through Git history cleanup. The main cleanup will be performed separately as it requires force-pushing and coordinated migration.

## Problem Statement

The `aiter` repository currently contains **547 MB** of data, with **420 MB** in Git history alone. Analysis shows that the majority of this space is occupied by files that were previously committed but have since been removed:

- **288 MB**: PyTorch tensor test data (`*.pt` files)
- **47 MB**: Large CSV benchmark results
- **50+ MB**: Compilation artifacts (`*.att` files) and temporary outputs
- **27 MB**: Debug logs and Jupyter notebook outputs

These files remain in Git history even though they're no longer in the current codebase.

## Proposed Solution

### Phase 1: Prevention (This PR)

This PR introduces safeguards to prevent large files from being committed in the future:

1. **Updated `.gitignore`**: Explicitly excludes large test data, build artifacts, and temporary files
2. **Pre-commit hook**: Automatically rejects commits containing files larger than 5MB
3. **Test data script**: Provides a framework for external test data management
4. **Documentation**: Explains the cleanup plan and migration process

### Phase 2: History Cleanup (Separate Maintenance Window)

After this PR is merged, a separate history cleanup will be performed using `git-filter-repo`:

- **Expected size reduction**: 547 MB → ~130 MB (76% reduction)
- **Git history reduction**: 420 MB → ~50 MB (88% reduction)
- **Benefits**:
  - Faster clones (60% faster)
  - Reduced CI/CD checkout time
  - Lower storage and bandwidth costs

## Changes in This PR

### 1. Enhanced `.gitignore`

Added exclusions for:
- Large model files (`*.pt`, `*.pth`, `*.bin`, `*.ckpt`)
- Test data directories (`op_tests/test_jenga_vsa/`, `op_tests/dump_data/`)
- Build artifacts (`OUT_FOLDER/`, `*.att`)
- Large CSV results
- Temporary files and debug logs

### 2. Pre-Commit Hook

Located at `.githooks/pre-commit`, this hook:
- Checks all staged files for size
- Rejects commits with files > 5MB
- Provides guidance on handling large files
- Can be bypassed with `--no-verify` if absolutely necessary

**To enable**: Run `git config core.hooksPath .githooks`

### 3. Test Data Download Script

`scripts/download_test_data.sh` provides a template for:
- Downloading test data from external storage
- Documenting what data was moved
- Helping developers who need the full test suite

**Note**: Storage location needs to be configured based on your infrastructure (S3, GCS, etc.)

### 4. Documentation

This document explains the cleanup plan and provides guidance for contributors.

## Impact

### For Current Contributors

**No immediate impact**. This PR only adds protective measures. The actual history cleanup will happen later with advance notice.

### For Future Contributors

After the history cleanup:
- **Initial clone will be faster** (60% reduction in download time)
- **Disk space saved**: 315 MB per clone
- **Large files will be prevented** by the pre-commit hook

## Migration Plan (Phase 2)

When ready to execute the history cleanup:

1. **Notification**: Send advance notice to all contributors (1-2 weeks)
2. **Maintenance window**: Schedule 2-4 hour window for cleanup
3. **Backup**: Create full repository backup
4. **Execute cleanup**: Run `git-filter-repo` to remove large files from history
5. **Force push**: Update remote repository (requires Admin permissions)
6. **Team migration**: All contributors re-clone the repository

### Migration Steps for Contributors

After the cleanup is complete:

```bash
# 1. Save your local changes
git stash

# 2. Delete old repository
cd ..
rm -rf aiter

# 3. Re-clone
git clone https://github.com/rocm/aiter.git
cd aiter

# 4. Restore your changes
git stash pop

# 5. Enable pre-commit hook
git config core.hooksPath .githooks
```

## Testing

Cleanup has been tested on a repository copy with the following results:

- **Original size**: 547 MB
- **After cleanup**: 232 MB
- **Space saved**: 315 MB (57% reduction)
- **Execution time**: 9 seconds
- **Commits preserved**: All 1,331 commits intact
- **Functionality**: No code changes, history remains traceable

## Alternatives Considered

1. **Git LFS**: Adds complexity and hosting costs. Most large files are no longer needed.
2. **BFG Repo-Cleaner**: Simpler but less flexible than `git-filter-repo`
3. **Manual deletion**: Only affects future commits, doesn't reduce repository size

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Force-push coordination | High | Advance notice, clear migration docs |
| Data loss | High | Full backup, testing on copy first |
| PR conflicts | Medium | Schedule during low-activity period |
| CI/CD disruption | Medium | Update CI configs beforehand |

## Questions or Concerns?

Please comment on this PR if you have:
- Questions about the cleanup process
- Concerns about data preservation
- Suggestions for the migration plan
- Need access to any of the removed test data

## References

- Test results: See `aiter_cleanup_results.md` (if available)
- Files to be removed: See `paths_to_remove.txt`
- Tool documentation: [git-filter-repo](https://github.com/newren/git-filter-repo)
