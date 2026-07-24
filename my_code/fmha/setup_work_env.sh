#!/bin/bash
# One-click helper for preparing the ROCm performance-test work environment.
# It supports:
#   1. Cloning the work repository collection on the host.
#   2. Creating a new Docker container and running the performance-test entry.
#   3. Running the performance-test entry in an existing Docker container.
#
# Examples:
#   ./setup_work_env.sh clone /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1
#   ./setup_work_env.sh update /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1
#   ./setup_work_env.sh setup-new --target-dir ./wk_sp1 --container-name wk_sp1_rocm
#   ./setup_work_env.sh new-container --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1
#   ./setup_work_env.sh existing-container --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1
#   HIP_VISIBLE_DEVICES=1 FLYDSL_BUILD_JOBS=64 ./setup_work_env.sh existing-container --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1
#   PERF_TEST_CMD='cd FlyDSL && python tests/kernels/test_flash_attn_fwd.py --iters 100 --compare' ./setup_work_env.sh existing-container --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1

set -x
set -euo pipefail

# export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -i ~/.ssh/id_rsa.hyg -o IdentitiesOnly=yes}"
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes}"
# export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0}"
# export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/fw-bringup:gfx1250-atom-dev-team-20260626}"
# export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/fw-bringup:satya_rocprofv3_2026_06_01}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/fw-bringup:gfx1250-atom-dev-20260713-ep4_flydsl}"
export PERF_TEST_CMD="${PERF_TEST_CMD:-}"
export FLYDSL_BUILD_JOBS="${FLYDSL_BUILD_JOBS:-128}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export FLY_BUILD_DIR="${FLY_BUILD_DIR:-}"
export PERF_LOG_FILE="${PERF_LOG_FILE:-}"
export FORCE_REBUILD_LLVM="${FORCE_REBUILD_LLVM:-0}"
# Build the cloned llvm-project (with lld + zstd) as a standalone step, decoupled
# from the FlyDSL install flow. The resulting buildmlir satisfies both FlyDSL and
# the comgr build needed by rocprofv3 --att (see comgr build_comgr_gfx1250.sh).
export BUILD_LLVM="${BUILD_LLVM:-0}"
# Build the full rocprofv3 --att gfx1250 stack (clones yanguahe/rocm-systems fork,
# builds rocprofv3 + comgr, installs the 0.1.5 trace-decoder, writes rocprof_env.sh).
# Implies an LLVM build (reuses buildmlir if present). Decoupled from FlyDSL/perf.
export BUILD_ATT="${BUILD_ATT:-0}"
export ROCM_SYSTEMS_URL="${ROCM_SYSTEMS_URL:-https://github.com/yanguahe/rocm-systems.git}"
export ROCM_SYSTEMS_BRANCH="${ROCM_SYSTEMS_BRANCH:-rocprofv3-gfx1250-att-debug}"
export CONTAINER_LANG="${CONTAINER_LANG:-C.UTF-8}"
export CONTAINER_LC_ALL="${CONTAINER_LC_ALL:-C.UTF-8}"
export CONTAINER_TERM="${CONTAINER_TERM:-xterm-256color}"

build_docker_input_env() {
    local env_string=""

    printf -v env_string '%sGIT_SSH_COMMAND=%q ' "$env_string" "$GIT_SSH_COMMAND"
    printf -v env_string '%sDOCKER_IMAGE=%q ' "$env_string" "$DOCKER_IMAGE"
    printf -v env_string '%sPERF_TEST_CMD=%q ' "$env_string" "$PERF_TEST_CMD"
    printf -v env_string '%sFLYDSL_BUILD_JOBS=%q ' "$env_string" "$FLYDSL_BUILD_JOBS"
    printf -v env_string '%sHIP_VISIBLE_DEVICES=%q ' "$env_string" "$HIP_VISIBLE_DEVICES"
    printf -v env_string '%sFLY_BUILD_DIR=%q ' "$env_string" "$FLY_BUILD_DIR"
    printf -v env_string '%sPERF_LOG_FILE=%q ' "$env_string" "$PERF_LOG_FILE"
    printf -v env_string '%sFORCE_REBUILD_LLVM=%q ' "$env_string" "$FORCE_REBUILD_LLVM"
    printf -v env_string '%sBUILD_LLVM=%q ' "$env_string" "$BUILD_LLVM"
    printf -v env_string '%sBUILD_ATT=%q ' "$env_string" "$BUILD_ATT"
    printf -v env_string '%sROCM_SYSTEMS_URL=%q ' "$env_string" "$ROCM_SYSTEMS_URL"
    printf -v env_string '%sROCM_SYSTEMS_BRANCH=%q ' "$env_string" "$ROCM_SYSTEMS_BRANCH"
    printf -v env_string '%sLANG=%q ' "$env_string" "$CONTAINER_LANG"
    printf -v env_string '%sLC_ALL=%q ' "$env_string" "$CONTAINER_LC_ALL"
    printf -v env_string '%sTERM=%q ' "$env_string" "$CONTAINER_TERM"

    echo "${env_string% }"
}

DOCKER_INPUT_ENV="$(build_docker_input_env)"

# Repository collection cloned by do_clone(). Keep these lists aligned by index.
CLONE_REPO_PATHS=(
    "."
    "llvm-project"
    "poc_kl"
    "FlyDSL"
    "triton"
    "aiter"
    "cursor_rules"
)

CLONE_REPO_URLS=(
    "git@github.com:yanguahe/my_asm_code.git"

    "git@github.com:yanguahe/llvm-project.git"
    # "git@github.com:ROCm/llvm-project.git"

    "git@github.com:yanguahe/poc_kl.git"

    # "git@github.com:yanguahe/FlyDSL.git"
    "git@github.com:ROCm/FlyDSL.git"

    "git@github.com:yanguahe/triton.git"

    # "git@github.com:yanguahe/aiter.git"
    "git@github.com:ROCm/aiter.git"

    "git@github.com:yanguahe/cursor_rules.git"
)

CLONE_REPO_BRANCHES=(
    # my_asm_code
    "asm_opt"

    # llvm-project
    # "fyd_main1"
    "fix/gfx1250-tensor-disasm-dont-care-bits"
    # ""

    # poc_kl
    "asm_opt"

    # FlyDSL
    ""
    # "fyd_main1"
    # "opus_align"

    # triton
    "gluon_opt"

    # aiter
    # ""
    # "opt_mtp_tt"
    # "jli/gfx1250/mha_batchmode_sugar"
    "hyg_gfx1250_gemm"

    # cursor_rules
    "main"
)

# Optional pinned commit/tag for each repository. Leave empty to use the branch head.
CLONE_REPO_REFS=(
    # my_asm_code
    ""

    # llvm-project
    ""
    # "c8cf6da4367c"
    # "7f77ca0dbda4abbf9af06537b2c475f20ccd6007"

    # poc_kl
    ""

    # FlyDSL
    ""
    # "v0.1.8"

    # triton
    ""

    # aiter
    ""
    # "45c428e54ac15b9b49d66018c8a1108b20c8336a"
    # "e5d13e1b70ba3f11cc53f49f0c6f61fff4440da7"

    # cursor_rules
    ""
)

CLONE_REPO_OPTIONS=(
    # my_asm_code
    ""

    # llvm-project
    ""

    # poc_kl
    ""

    # FlyDSL
    ""

    # triton
    ""

    # aiter
    "--recursive"

    # cursor_rules
    ""
)

CLONE_REPO_NOTES=(
    # my_asm_code
    ""

    # llvm-project
    "large repo ~3.2GB"

    # poc_kl
    ""

    # FlyDSL
    ""

    # triton
    ""

    # aiter
    "recursive"

    # cursor_rules
    ""
)

TARGET_DIR=""
PARENT_DIR=""
CLONE_NAME=""
CONTAINER_NAME=""
CODE_DIR=""
WORK_DIR=""
WORK_DIR_INPUT=""
REPOS=()

usage() {
    local exit_code="${1:-1}"
    cat <<EOF
Usage:
  $0 clone <target_directory>
      Clone the work repository collection on the host.
      <target_directory> is the destination directory for the root repo
      git@github.com:yanguahe/my_asm_code.git. Its parent directory must exist.

  $0 update <target_directory>
      Update the cloned repositories with git pull --ff-only.
      <target_directory> must point to an existing cloned root work directory.

  $0 setup-new --target-dir <target_directory> --container-name <name> [--workdir <dir>] [--force-rebuild-llvm]
      Clone repositories, create a new container, then run the performance-test entry.
      If --workdir is omitted, <target_directory> is used.
      --target-dir is the host clone destination for the root work directory.
      --container-name is the Docker container name to create.
      --workdir is only used when creating a new container. It is mounted into
        the container at the same path and used as docker -w.
      --force-rebuild-llvm removes the existing LLVM build/install directories
        before building FlyDSL.

  $0 new-container [--target-dir <target_directory>] --container-name <name> --workdir <dir> [--force-rebuild-llvm]
      Create a new container and run the performance-test entry.
      --target-dir is optional here. When provided, logs are written there and
        the directory is mounted into the new container if it differs from --workdir.
      --container-name is the Docker container name to create.
      --workdir is required for new-container. It is the host directory mounted
        into the new container and the value passed to docker -w.
      --force-rebuild-llvm removes the existing LLVM build/install directories
        before building FlyDSL.

  $0 existing-container --target-dir <target_directory> --container-name <name> --workdir <dir> [--force-rebuild-llvm] [--build-llvm] [--build-att]
      Run the performance-test entry in an existing container.
      --target-dir is the cloned code directory and preferred log directory.
      --container-name is the Docker container name to reuse. If it is stopped,
        the script starts it before docker exec.
      --workdir is the directory where the generated container script is written.
      --force-rebuild-llvm removes the existing LLVM build/install directories
        before building FlyDSL.
      --build-llvm only builds the cloned llvm-project (with lld + zstd) and
        skips aiter/perf. Produces a comgr-capable LLVM for rocprofv3 --att.
      --build-att builds the full rocprofv3 --att gfx1250 stack (clones the
        rocm-systems fork, builds rocprofv3 + comgr, installs the 0.1.5 decoder,
        writes rocprof_env.sh). Reuses buildmlir if present, else builds LLVM
        first. Skips aiter/perf.

Compatibility:
  $0 <target_directory>
      Same as: $0 clone <target_directory>

  $0 --update <target_directory>
      Same as: $0 update <target_directory>

Arguments:
  <target_directory>
      Host path for the root work directory. Relative paths are normalized to
      absolute paths. The parent directory must already exist.

  --target-dir <target_directory>
      Clone destination used by setup-new before creating the container.
      For existing-container, it is the code directory and preferred log
      directory.

  --container-name <name>, --name <name>
      Docker container name. For new-container/setup-new it must not already
      exist. For existing-container it must already exist.

  --workdir <dir>, --pwd <dir>
      Required for new-container and existing-container. For new-container, it
      controls the directory mounted into the container and the docker run -w
      value. For existing-container, it controls where the generated container
      script is written. Relative paths are normalized on the host.

Notes:
  - Default FlyDSL test:
    cd FlyDSL && python tests/kernels/test_flash_attn_fwd.py --iters 100 --compare
  - Optional environment variables:
    GIT_SSH_COMMAND, DOCKER_IMAGE, PERF_TEST_CMD, FLYDSL_BUILD_JOBS,
    HIP_VISIBLE_DEVICES, FLY_BUILD_DIR, PERF_LOG_FILE, FORCE_REBUILD_LLVM,
    BUILD_LLVM, BUILD_ATT, ROCM_SYSTEMS_URL, ROCM_SYSTEMS_BRANCH,
    CONTAINER_LANG, CONTAINER_LC_ALL, CONTAINER_TERM.
  - Standalone LLVM build (decoupled from FlyDSL):
    Pass --build-llvm (or BUILD_LLVM=1) to new-container/existing-container to
    only build the cloned llvm-project (with lld + zstd) and skip aiter/perf.
    The resulting llvm-project/buildmlir satisfies both FlyDSL and the comgr
    build required by rocprofv3 --att. Combine with --force-rebuild-llvm to
    rebuild from scratch. Example:
      BUILD_LLVM=1 ./setup_work_env.sh existing-container \
        --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1
  - Full rocprofv3 --att gfx1250 stack:
    Pass --build-att (or BUILD_ATT=1) to clone git@/https yanguahe/rocm-systems
    (branch rocprofv3-gfx1250-att-debug), build rocprofv3 + comgr from it, install
    the bundled 0.1.5 trace-decoder, and write <workdir>/rocprof_env.sh. It reuses
    buildmlir if present (else builds an lld+zstd LLVM first) and skips aiter/perf.
    Override the source with ROCM_SYSTEMS_URL / ROCM_SYSTEMS_BRANCH. Example:
      BUILD_ATT=1 ./setup_work_env.sh existing-container \
        --target-dir ./wk_sp1 --container-name wk_sp1_rocm --workdir ./wk_sp1
    Then, before rocprofv3 --att: source <workdir>/rocprof_env.sh
  - Docker performance-test output is written to PERF_LOG_FILE. If it is not
    set, the script creates a timestamped log under --target-dir, or --workdir
    for new-container when --target-dir is not provided.
  - Set PERF_TEST_CMD='your command' to temporarily run a custom performance command
    after the repository install steps.
EOF
    exit "$exit_code"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

need_value() {
    local option="$1"
    local value="${2:-}"
    if [ -z "$value" ]; then
        die "$option requires a value."
    fi
}

resolve_path_with_existing_parent() {
    local input_path="$1"
    local parent_dir
    local base_name

    parent_dir=$(dirname "$input_path")
    base_name=$(basename "$input_path")

    if [ ! -d "$parent_dir" ]; then
        die "Parent directory '$parent_dir' does not exist."
    fi

    parent_dir=$(cd "$parent_dir" && pwd)
    echo "$parent_dir/$base_name"
}

resolve_existing_dir() {
    local input_path="$1"
    local resolved_path

    resolved_path=$(resolve_path_with_existing_parent "$input_path")
    if [ ! -d "$resolved_path" ]; then
        die "Directory '$resolved_path' does not exist."
    fi

    echo "$resolved_path"
}

set_clone_target() {
    local repo_path

    TARGET_DIR=$(resolve_path_with_existing_parent "$1")
    PARENT_DIR=$(dirname "$TARGET_DIR")
    CLONE_NAME=$(basename "$TARGET_DIR")

    REPOS=()
    for repo_path in "${CLONE_REPO_PATHS[@]}"; do
        if [ "$repo_path" = "." ]; then
            REPOS+=("$TARGET_DIR")
        else
            REPOS+=("$TARGET_DIR/$repo_path")
        fi
    done
}

set_work_dir() {
    WORK_DIR=$(resolve_existing_dir "$1")
}

set_code_dir_from_target() {
    if [ -z "$TARGET_DIR" ]; then
        die "--target-dir is required."
    fi
    if [ ! -d "$TARGET_DIR" ]; then
        die "--target-dir '$TARGET_DIR' must exist."
    fi

    CODE_DIR="$TARGET_DIR"
}

validate_clone_repo_config() {
    local repo_count="${#CLONE_REPO_PATHS[@]}"

    if [ "${#CLONE_REPO_URLS[@]}" -ne "$repo_count" ] ||
        [ "${#CLONE_REPO_BRANCHES[@]}" -ne "$repo_count" ] ||
        [ "${#CLONE_REPO_REFS[@]}" -ne "$repo_count" ] ||
        [ "${#CLONE_REPO_OPTIONS[@]}" -ne "$repo_count" ] ||
        [ "${#CLONE_REPO_NOTES[@]}" -ne "$repo_count" ]; then
        die "Clone repository config lists must have the same length."
    fi
}

ensure_repo_origin() {
    local repo_url="$1"
    local current_origin

    if [ -z "$repo_url" ]; then
        return
    fi

    current_origin=$(git remote get-url origin 2>/dev/null || true)
    if [ -z "$current_origin" ]; then
        echo "  -> origin is not set, adding origin: $repo_url"
        git remote add origin "$repo_url"
    elif [ "$current_origin" != "$repo_url" ]; then
        echo "  -> origin differs, updating origin:"
        echo "     current: $current_origin"
        echo "     target:  $repo_url"
        git remote set-url origin "$repo_url"
    fi
}

fetch_repo_update_target() {
    local repo_branch="$1"
    local repo_ref="$2"

    if [ -n "$repo_branch" ]; then
        echo "  -> Fetching requested branch only: $repo_branch"
        git fetch --no-recurse-submodules origin "refs/heads/$repo_branch:refs/remotes/origin/$repo_branch"
    elif [ -n "$repo_ref" ]; then
        echo "  -> Skipping full origin fetch; requested ref will be fetched directly if needed: $repo_ref"
    else
        git fetch --no-recurse-submodules origin
    fi
}

checkout_repo_branch() {
    local repo_branch="$1"
    local current_branch

    if [ -z "$repo_branch" ]; then
        return
    fi

    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" = "$repo_branch" ]; then
        return
    fi

    echo "  -> Current branch is $current_branch, switching to requested branch $repo_branch..."
    if git show-ref --verify --quiet "refs/heads/$repo_branch"; then
        git checkout "$repo_branch"
    elif git show-ref --verify --quiet "refs/remotes/origin/$repo_branch"; then
        git checkout -t "origin/$repo_branch"
    else
        die "Requested branch '$repo_branch' not found locally or on origin in $(pwd)."
    fi
}

checkout_repo_ref() {
    local repo_ref="$1"

    if [ -z "$repo_ref" ]; then
        return
    fi

    if ! git cat-file -e "${repo_ref}^{commit}" 2>/dev/null; then
        echo "  -> Ref $repo_ref not found locally, fetching it from origin..."
        git fetch --no-recurse-submodules origin "$repo_ref" || true
    fi

    if ! git cat-file -e "${repo_ref}^{commit}" 2>/dev/null; then
        echo "  -> Ref $repo_ref still not found locally, trying it as a tag..."
        git fetch --no-recurse-submodules origin tag "$repo_ref" || true
    fi

    if ! git cat-file -e "${repo_ref}^{commit}" 2>/dev/null; then
        die "Cannot resolve commit/tag '$repo_ref' in $(pwd)."
    fi

    git checkout "$repo_ref"
    echo "  -> Checked out ref $repo_ref."
}

setup_cursor_rules_symlink() {
    local repo_count="${#CLONE_REPO_URLS[@]}"
    local repo_index
    local repo_path
    local link_path
    local link_target
    local target_abs
    local current_target

    for ((repo_index = 0; repo_index < repo_count; repo_index++)); do
        if [ "${CLONE_REPO_URLS[$repo_index]}" != "git@github.com:yanguahe/cursor_rules.git" ]; then
            continue
        fi

        repo_path="${CLONE_REPO_PATHS[$repo_index]}"
        if [ "$repo_path" = "." ]; then
            die "cursor_rules repository cannot use '.' as CLONE_REPO_PATHS entry."
        fi

        link_path="$TARGET_DIR/.cursor"
        link_target="$repo_path/fmha_flydsl_new_api_opt/.cursor"
        target_abs="$TARGET_DIR/$link_target"

        echo "=== Setting Cursor rules symlink ==="
        if [ ! -d "$target_abs" ]; then
            die "Cursor rules directory '$target_abs' does not exist."
        fi

        if [ -L "$link_path" ]; then
            current_target=$(readlink "$link_path")
            if [ "$current_target" = "$link_target" ]; then
                echo "  -> .cursor already points to $link_target."
                echo ""
                return
            fi

            echo "  -> Replacing existing .cursor symlink: $current_target"
            rm -f "$link_path"
        elif [ -e "$link_path" ]; then
            die "$link_path exists and is not a symlink. Please move it before running this script."
        fi

        cd "$TARGET_DIR"
        ln -s "$link_target" ".cursor"
        echo "  -> .cursor -> $link_target"
        echo ""
        return
    done
}

do_clone() {
    local repo_count
    local repo_index
    local repo_path
    local repo_url
    local repo_branch
    local repo_ref
    local repo_options
    local repo_note
    local repo_name
    local repo_target
    local repo_parent
    local repo_dest_name
    local -a git_clone_options
    local -a git_branch_options

    if [ -z "$TARGET_DIR" ]; then
        die "TARGET_DIR is not set."
    fi

    validate_clone_repo_config

    echo "=== Cloning work repositories to: $TARGET_DIR ==="
    echo "PARENT_DIR: $PARENT_DIR"
    echo "CLONE_NAME: $CLONE_NAME"
    echo ""

    repo_count="${#CLONE_REPO_PATHS[@]}"
    for ((repo_index = 0; repo_index < repo_count; repo_index++)); do
        repo_path="${CLONE_REPO_PATHS[$repo_index]}"
        repo_url="${CLONE_REPO_URLS[$repo_index]}"
        repo_branch="${CLONE_REPO_BRANCHES[$repo_index]}"
        repo_ref="${CLONE_REPO_REFS[$repo_index]}"
        repo_options="${CLONE_REPO_OPTIONS[$repo_index]}"
        repo_note="${CLONE_REPO_NOTES[$repo_index]}"

        if [ "$repo_path" = "." ]; then
            repo_name="$CLONE_NAME"
            repo_target="$TARGET_DIR"
            repo_parent="$PARENT_DIR"
            repo_dest_name="$CLONE_NAME"
        else
            repo_name=$(basename "$repo_path")
            repo_target="$TARGET_DIR/$repo_path"
            repo_parent=$(dirname "$repo_target")
            repo_dest_name=$(basename "$repo_target")
        fi

        if [ -n "$repo_note" ]; then
            echo "[$((repo_index + 1))/$repo_count] Cloning $repo_name (branch: ${repo_branch:-default}, ref: ${repo_ref:-branch HEAD}) -> $repo_target - $repo_note..."
        else
            echo "[$((repo_index + 1))/$repo_count] Cloning $repo_name (branch: ${repo_branch:-default}, ref: ${repo_ref:-branch HEAD}) -> $repo_target ..."
        fi

        if [ -d "$repo_target/.git" ]; then
            echo "  -> $repo_name already exists, skipping."
            if [ -n "$repo_ref" ]; then
                cd "$repo_target"
                checkout_repo_ref "$repo_ref"
            fi
            echo ""
            continue
        elif [ -e "$repo_target" ]; then
            die "$repo_target exists but is not a git repository."
        fi

        if [ ! -d "$repo_parent" ]; then
            die "Parent directory '$repo_parent' does not exist for $repo_name."
        fi

        git_clone_options=()
        if [ -n "$repo_options" ]; then
            read -r -a git_clone_options <<< "$repo_options"
        fi

        git_branch_options=()
        if [ -n "$repo_branch" ]; then
            git_branch_options=(-b "$repo_branch")
        fi

        cd "$repo_parent"
        git clone "${git_branch_options[@]}" "${git_clone_options[@]}" "$repo_url" "$repo_dest_name"
        if [ -n "$repo_ref" ]; then
            cd "$repo_dest_name"
            checkout_repo_ref "$repo_ref"
        fi
        echo "  -> Done."
        echo ""
    done

    setup_cursor_rules_symlink

    echo "=== All repositories cloned successfully! ==="
    echo ""
}

do_update() {
    local repo_count
    local repo_index
    local repo_path
    local repo
    local repo_name
    local repo_url
    local repo_branch
    local repo_ref
    local branch

    if [ -z "$TARGET_DIR" ]; then
        die "TARGET_DIR is not set."
    fi

    validate_clone_repo_config

    echo "=== Updating repositories to latest remote state ==="
    echo ""

    repo_count="${#CLONE_REPO_PATHS[@]}"
    for ((repo_index = 0; repo_index < repo_count; repo_index++)); do
        repo_path="${CLONE_REPO_PATHS[$repo_index]}"
        repo="${REPOS[$repo_index]}"
        repo_name=$(basename "$repo")
        repo_url="${CLONE_REPO_URLS[$repo_index]}"
        repo_branch="${CLONE_REPO_BRANCHES[$repo_index]}"
        repo_ref="${CLONE_REPO_REFS[$repo_index]}"

        if [ ! -d "$repo/.git" ]; then
            echo "[$repo_name] Not a git repo, skipping."
            echo ""
            continue
        fi

        cd "$repo"
        branch=$(git rev-parse --abbrev-ref HEAD)
        echo "[$repo_name] $repo (branch: $branch, requested: ${repo_branch:-current}, ref: ${repo_ref:-branch HEAD}), fetching..."
        ensure_repo_origin "$repo_url"
        fetch_repo_update_target "$repo_branch" "$repo_ref"

        checkout_repo_branch "$repo_branch"

        if [ -n "$repo_branch" ]; then
            echo "  -> Fast-forwarding from origin/$repo_branch..."
            if git merge --ff-only "origin/$repo_branch"; then
                echo "  -> Updated branch $repo_branch successfully."
            else
                echo "  -> WARNING: Cannot fast-forward $repo_branch. You may have local commits or diverged history."
                echo "     Please resolve manually in: $repo"
            fi
        elif [ -z "$repo_ref" ]; then
            echo "  -> Pulling current branch with --ff-only..."
            if git pull --ff-only --no-recurse-submodules; then
                echo "  -> Updated successfully."
            else
                echo "  -> WARNING: Cannot fast-forward. You may have local commits or diverged history."
                echo "     Please resolve manually in: $repo"
            fi
        fi

        if [ -n "$repo_ref" ]; then
            checkout_repo_ref "$repo_ref"
            echo "  -> Updated successfully."
        fi

        if [ "$repo_path" = "aiter" ]; then
            echo "  -> Updating aiter submodules..."
            if git submodule update --init --recursive; then
                echo "  -> aiter submodules updated successfully."
            else
                echo "  -> WARNING: Failed to update aiter submodules."
                echo "     Please resolve manually in: $repo"
            fi
        fi
        echo ""
    done

    echo "=== Update complete! ==="
    echo ""
}

do_perf_test_script() {
    cat <<'CONTAINER_SCRIPT'
set -euo pipefail
set -x

cd "$(dirname "$0")"

apply_docker_input_env() {
    if [ -n "${DOCKER_INPUT_ENV:-}" ]; then
        eval "export $DOCKER_INPUT_ENV"
    fi
}

handle_interrupt() {
    echo ""
    echo "Interrupted. Exiting container workload..."
    exit 130
}

setup_perf_logging() {
    if [ -z "${PERF_LOG_FILE:-}" ]; then
        return
    fi

    mkdir -p "$(dirname "$PERF_LOG_FILE")"
    touch "$PERF_LOG_FILE"
    exec > >(tee -a "$PERF_LOG_FILE") 2>&1
    echo "=== Logging to: $PERF_LOG_FILE ==="
}

configure_git_safe_directories() {
    local repo_dir

    if ! command -v git >/dev/null 2>&1; then
        return
    fi

    git config --global --add safe.directory "$(pwd)" || true
    for repo_dir in ./*; do
        if [ -d "$repo_dir/.git" ]; then
            git config --global --add safe.directory "$(cd "$repo_dir" && pwd)" || true
        fi
    done
}

apply_docker_input_env
trap handle_interrupt INT TERM
setup_perf_logging

install_aiter() {
    echo "=== Installing aiter ==="
    if [ ! -d "aiter" ]; then
        echo "aiter directory not found under $(pwd). Skipping aiter install."
        echo ""
        return
    fi

    python3 -m pip install vcs_versioning
    # python3 -m pip install triton==3.6.0

    cd aiter
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found under $(pwd). Skipping aiter requirements install."
    fi

    echo "Cleaning aiter JIT build artifacts..."
    rm -f aiter/jit/*.so
    rm -rf aiter/jit/build
    python3 setup.py develop
    cd -
    echo "=== aiter install complete ==="
    echo ""
}

find_flydsl_dir() {
    if [ -d "FlyDSL" ]; then
        echo "$(pwd)/FlyDSL"
        return
    fi

    if [ -f "scripts/build.sh" ] && [ -d "tests/kernels" ]; then
        pwd
        return
    fi

    return 1
}

find_llvm_project_dir() {
    if [ -d "llvm-project/llvm" ]; then
        cd llvm-project
        pwd
        cd - >/dev/null
        return
    fi

    if [ -d "../llvm-project/llvm" ]; then
        cd ../llvm-project
        pwd
        cd - >/dev/null
        return
    fi

    return 1
}

install_flydsl_prereqs() {
    echo "=== Installing FlyDSL build prerequisites ==="

    python3 -m pip install nanobind numpy pybind11

    if ! command -v cmake >/dev/null 2>&1 ||
        ! command -v ninja >/dev/null 2>&1 ||
        ! command -v patchelf >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build patchelf
    fi

    python3 -c "import nanobind, numpy, pybind11"
    cmake --version
    ninja --version
    patchelf --version
    echo ""
}

build_cloned_llvm_project() {
    local jobs="$1"
    local force_rebuild="${2:-0}"
    local original_dir

    echo "=== Building cloned llvm-project ==="
    if ! LLVM_PROJECT_DIR=$(find_llvm_project_dir); then
        echo "llvm-project directory not found under $(pwd)."
        return 1
    fi

    original_dir=$(pwd)
    cd "$LLVM_PROJECT_DIR"

    if [ "$force_rebuild" = "1" ] || [ "$force_rebuild" = "true" ] || [ "$force_rebuild" = "yes" ]; then
        echo "Force rebuilding LLVM: removing $LLVM_PROJECT_DIR/buildmlir and $LLVM_PROJECT_DIR/mlir_install"
        rm -rf buildmlir mlir_install
    fi

    if [ ! -d "buildmlir" ]; then
        rm -rf buildmlir
        mkdir -p buildmlir
        cd buildmlir
        # lld + zstd are required so this buildmlir can also drive the comgr build
        # for rocprofv3 --att: comgr's LLVM export references the lld cmake package
        # and the zstd::libzstd_shared imported target; without zstd the resulting
        # comgr aborts at runtime with "zstd::decompress is unavailable".
        local -a llvm_zstd_args
        llvm_zstd_args=()
        local sysdeps_dir="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps"
        if [ -f "$sysdeps_dir/lib/libzstd.so" ] && [ -f "$sysdeps_dir/include/zstd.h" ]; then
            echo "Enabling LLVM zstd via rocm_sysdeps: $sysdeps_dir"
            llvm_zstd_args=(
                -DLLVM_ENABLE_ZSTD=ON
                -Dzstd_INCLUDE_DIR="$sysdeps_dir/include"
                -Dzstd_LIBRARY="$sysdeps_dir/lib/libzstd.so"
            )
        else
            echo "WARNING: rocm_sysdeps zstd not found; enabling FORCE_ON to use system zstd."
            llvm_zstd_args=(-DLLVM_ENABLE_ZSTD=FORCE_ON)
        fi
        cmake -G Ninja \
            -S ../llvm \
            -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
            -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;AMDGPU' \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_STANDARD=17 \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DLLVM_INSTALL_UTILS=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DPython3_EXECUTABLE="$(which python3)" \
            -Dnanobind_DIR="$(python3 -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')")" \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLVM_BUILD_LLVM_DYLIB=OFF \
            -DLLVM_LINK_LLVM_DYLIB=OFF \
            "${llvm_zstd_args[@]}"
        cd ..
    else
        echo "Reusing existing LLVM build directory: $LLVM_PROJECT_DIR/buildmlir"
    fi

    if [ ! -d "buildmlir" ]; then
        die "LLVM build directory '$LLVM_PROJECT_DIR/buildmlir' does not exist."
    fi

    if [ ! -f "buildmlir/build.ninja" ]; then
        die "LLVM build directory '$LLVM_PROJECT_DIR/buildmlir' exists but build.ninja is missing."
    fi

    if [ ! -d "mlir_install" ]; then
        echo "Building LLVM/MLIR with $jobs jobs..."
        cmake --build buildmlir -j "$jobs"
        cmake --install buildmlir --prefix mlir_install
    else
        echo "Reusing existing LLVM install directory: $LLVM_PROJECT_DIR/mlir_install"
    fi
    cd "$original_dir"

    echo "=== cloned llvm-project build complete ==="
    echo ""
}

install_flydsl() {
    local flydsl_dir

    echo "=== Installing FlyDSL ==="
    if ! flydsl_dir=$(find_flydsl_dir); then
        echo "FlyDSL directory not found under $(pwd)."
        return 1
    fi

    cd "$flydsl_dir"
    rm -rf build-fly
    export MLIR_PATH="${LLVM_PROJECT_DIR}/mlir_install"

    # zstd_ROOT: the LLVM23 build enables zstd, so mlir_install's LLVMConfig.cmake
    # runs find_package(zstd) and LLVMExports.cmake links LLVMSupport against
    # zstd::libzstd_shared. FlyDSL's cmake does not set a zstd search path, so
    # point find_package(zstd) at the rocm_sysdeps root (which has the standard
    # lib/libzstd.so + include/zstd.h that LLVM's module-mode Findzstd.cmake needs).
    local zstd_root="${ZSTD_ROOT:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps}"
    if [ ! -f "${zstd_root}/lib/libzstd.so" ] || [ ! -f "${zstd_root}/include/zstd.h" ]; then
        echo "Error: zstd_ROOT '${zstd_root}' must contain lib/libzstd.so and include/zstd.h." >&2
        echo "       Set ZSTD_ROOT to the directory holding zstd (rocm_sysdeps root)." >&2
        return 1
    fi
    export zstd_ROOT="$zstd_root"
    echo "Using zstd_ROOT=$zstd_ROOT"

    bash scripts/build.sh -j"$FLYDSL_BUILD_JOBS"
    python3 -m pip install -e .

    if [ -f "opus_attn/install_python.sh" ]; then
        export OPUS_INCLUDE_DIR="${OPUS_INCLUDE_DIR:-$(dirname "$flydsl_dir")/aiter/csrc/include}"
        echo "Installing FlyDSL opus_attn with OPUS_INCLUDE_DIR=$OPUS_INCLUDE_DIR"
        (
            cd opus_attn
            ./install_python.sh
        )
    else
        echo "opus_attn/install_python.sh not found under $flydsl_dir. Skipping opus_attn install."
    fi

    if [ -f "exp_isa/build.sh" ]; then
        echo "Building FlyDSL exp_isa..."
        ./exp_isa/build.sh
    else
        echo "exp_isa/build.sh not found under $flydsl_dir. Skipping exp_isa build."
    fi
    cd -

    echo "=== FlyDSL install complete ==="
    echo ""
}

set_flydsl_test_library_path() {
    local repo_root="$1"
    local mlir_libs_dir

    mlir_libs_dir="${FLY_BUILD_DIR:-${repo_root}/build-fly}/python_packages/flydsl/_mlir/_mlir_libs"
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":${mlir_libs_dir}:"* ]]; then
        export LD_LIBRARY_PATH="${mlir_libs_dir}:${LD_LIBRARY_PATH:-}"
    fi
}

run_flydsl_perf_test() {
    local flydsl_dir

    echo "=== Running FlyDSL performance test ==="
    if ! flydsl_dir=$(find_flydsl_dir); then
        echo "FlyDSL directory not found under $(pwd)."
        return 1
    fi

    cd "$flydsl_dir"
    set_flydsl_test_library_path "$flydsl_dir"
    python tests/kernels/test_flash_attn_fwd.py --iters 100 --compare
    cd -
}

do_build_llvm_only() {
    echo "=== Building llvm-project only (BUILD_LLVM=1) ==="
    echo "=== llvm-project build step done ==="
    echo ""
}

ROCM_SYSTEMS_URL="${ROCM_SYSTEMS_URL:-https://github.com/yanguahe/rocm-systems.git}"
ROCM_SYSTEMS_BRANCH="${ROCM_SYSTEMS_BRANCH:-rocprofv3-gfx1250-att-debug}"

# Full rocprofv3 --att gfx1250 stack build. Assumes build_cloned_llvm_project has
# already run so $LLVM_PROJECT_DIR/buildmlir (with lld + zstd) exists. Clones the
# rocm-systems fork, builds rocprofv3 + comgr from it, installs the 0.1.5 decoder,
# and writes rocprof_env.sh so rocprofv3 --att can collect + decode on gfx1250.
build_rocprofv3_att_stack() {
    local work_root
    local rocm_sys_dir
    local llvm_build_dir
    local comgr_new_dir
    local decoder_new_dir
    local rocprof_install_dir
    local decoder_deb
    local dec_tmp
    local dec_so
    local comgr_so

    work_root="$(pwd)"
    rocm_sys_dir="$work_root/rocm-systems"
    llvm_build_dir="${LLVM_PROJECT_DIR:-$work_root/llvm-project}/buildmlir"
    comgr_new_dir="$work_root/comgr_new"
    decoder_new_dir="$work_root/decoder_new"
    rocprof_install_dir="$work_root/rocprof-install"

    echo "=== Building rocprofv3 --att gfx1250 stack ==="
    echo "work_root      = $work_root"
    echo "rocm-systems   = $rocm_sys_dir (branch $ROCM_SYSTEMS_BRANCH)"
    echo "LLVM buildmlir = $llvm_build_dir"
    echo ""

    if [ ! -d "$llvm_build_dir/lib/cmake/lld" ]; then
        die "LLVM build '$llvm_build_dir' missing lib/cmake/lld; comgr needs an lld+zstd LLVM build."
    fi

    # 1. Reuse the rocm-systems checkout prepared by the host script.
    if [ -d "$rocm_sys_dir/.git" ]; then
        echo "  -> reusing host-prepared rocm-systems checkout: $rocm_sys_dir"
    else
        die "rocm-systems checkout missing at $rocm_sys_dir; host-side setup should clone it before container execution."
    fi

    # 2. Build + install rocprofv3 into $work_root/rocprof-install.
    echo "=== [att 1/4] Building rocprofv3 ==="
    ( cd "$rocm_sys_dir" && INSTALL_PREFIX="$rocprof_install_dir" \
        bash build_rocprofv3_gfx1250.sh )

    # 3. Build comgr (LLVM23 + zstd) and deploy to comgr_new.
    echo "=== [att 2/4] Building comgr (LLVM23) ==="
    ( cd "$rocm_sys_dir" && LLVM_BUILD_DIR="$llvm_build_dir" \
        bash build_comgr_gfx1250.sh )
    comgr_so="$rocm_sys_dir/comgr/build/libamd_comgr.so.3.0.0"
    [ -f "$comgr_so" ] || die "comgr build did not produce $comgr_so"
    mkdir -p "$comgr_new_dir"
    cp -af "$comgr_so" "$comgr_new_dir/"
    ( cd "$comgr_new_dir" \
        && ln -sf libamd_comgr.so.3.0.0 libamd_comgr.so.3 \
        && ln -sf libamd_comgr.so.3 libamd_comgr.so )
    echo "  -> comgr deployed to $comgr_new_dir"

    # 4. Install the 0.1.5 trace-decoder (supports gfx1250 navi) to decoder_new.
    echo "=== [att 3/4] Installing 0.1.5 trace-decoder ==="
    decoder_deb="$rocm_sys_dir/rocprof-trace-decoder-ubuntu-24.04-0.1.5-Linux-runtime.deb"
    [ -f "$decoder_deb" ] || die "decoder deb not found: $decoder_deb"
    dec_tmp="$(mktemp -d)"
    dpkg-deb -x "$decoder_deb" "$dec_tmp"
    dec_so="$(find "$dec_tmp" -name 'librocprof-trace-decoder.so*' | head -1)"
    [ -n "$dec_so" ] || die "no librocprof-trace-decoder.so inside $decoder_deb"
    mkdir -p "$decoder_new_dir"
    cp -af "$dec_so" "$decoder_new_dir/librocprof-trace-decoder.so"
    rm -rf "$dec_tmp"
    echo "  -> decoder installed to $decoder_new_dir"

    # 5. Write rocprof_env.sh (source before rocprofv3 --att). Pins comgr_new first
    #    on LD_LIBRARY_PATH (LLVM23 comgr disassembles gfx1250 new insts) and adds
    #    /opt/rocm/lib (HSA bare-name dlopen of libhsa-amd-aqlprofile64.so).
    echo "=== [att 4/4] Writing rocprof_env.sh ==="
    cat > "$work_root/rocprof_env.sh" <<ENVEOF
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\$PATH
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
CORE=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
SYSDEPS=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps
# COMGR_NEW: self-built comgr (LLVM23) that disassembles gfx1250 new insts such as
# v_writelane_b32 VOP3(0xd7610000). Must precede \$CORE so rocprofv3 loads it over
# the container's LLVM22 comgr (which emits illegal .long -> truncated code.json).
COMGR_NEW=$comgr_new_dir
# /opt/rocm/lib is required: HSA runtime bare-name dlopen("libhsa-amd-aqlprofile64.so")
# resolves only there; missing it aborts ATT collection with
# "aqlprofile API table load failed".
export LD_LIBRARY_PATH=\$COMGR_NEW:\$CORE:\$SYSDEPS/lib:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
export PKG_CONFIG_PATH=\$SYSDEPS/lib/pkgconfig:\${PKG_CONFIG_PATH:-}
# ROCPROF_ATT_LIBRARY_PATH: pin the 0.1.5 decoder that supports gfx1250 navi.
export ROCPROF_ATT_LIBRARY_PATH=$decoder_new_dir
# ROCPROF: self-built rocprofv3 (rocprofiler-sdk fork) with gfx1250 --att fixes.
export ROCPROF=$rocprof_install_dir/bin/rocprofv3
export PATH=$rocprof_install_dir/bin:\$PATH
ENVEOF
    echo "  -> wrote $work_root/rocprof_env.sh"

    echo ""
    echo "=== rocprofv3 --att gfx1250 stack ready ==="
    echo "  rocprofv3 : $rocprof_install_dir/bin/rocprofv3"
    echo "  comgr     : $comgr_new_dir"
    echo "  decoder   : $decoder_new_dir"
    echo "  env       : $work_root/rocprof_env.sh"
    "$rocprof_install_dir/bin/rocprofv3" --version 2>/dev/null | head -3 || true
    echo ""
}

do_build_att_only() {
    echo "=== Building rocprofv3 --att gfx1250 stack (BUILD_ATT=1) ==="
    build_rocprofv3_att_stack
    echo "=== rocprofv3 --att stack build step done ==="
    echo ""
}

do_perf_test() {
    local jobs
    local llvm_built=0

    echo "=== Preparing performance-test environment ==="
    echo "Container hostname: $(hostname)"
    echo "Working directory: $(pwd)"
    echo ""

    configure_git_safe_directories
    jobs="$FLYDSL_BUILD_JOBS"

    # Full rocprofv3 --att gfx1250 stack (clone rocm-systems fork, build rocprofv3
    # + comgr, install decoder, write rocprof_env.sh). Implies an LLVM build.
    if [ "${BUILD_ATT:-0}" = "1" ] || [ "${BUILD_ATT:-0}" = "true" ] || [ "${BUILD_ATT:-0}" = "yes" ]; then
        install_flydsl_prereqs
        # Reuses buildmlir if present; builds it (lld+zstd) otherwise.
        build_cloned_llvm_project "$jobs" "${FORCE_REBUILD_LLVM:-0}"
        llvm_built=1
        do_build_att_only
    fi

    # Standalone LLVM build, decoupled from the FlyDSL/aiter/perf flow. When
    # BUILD_LLVM=1 the script only builds llvm-project (with lld + zstd) and exits,
    # so it can be used purely to produce a comgr-capable LLVM for rocprofv3 --att.
    if [ "${BUILD_LLVM:-0}" = "1" ] || [ "${BUILD_LLVM:-0}" = "true" ] || [ "${BUILD_LLVM:-0}" = "yes" ]; then
        if [ "$llvm_built" != "1" ]; then
            install_flydsl_prereqs
            build_cloned_llvm_project "$jobs" "${FORCE_REBUILD_LLVM:-0}"
            llvm_built=1
        fi
        do_build_llvm_only
        return
    fi

    install_aiter
    if [ "$llvm_built" != "1" ]; then
        install_flydsl_prereqs
        build_cloned_llvm_project "$jobs" "${FORCE_REBUILD_LLVM:-0}"
        llvm_built=1
    fi
    install_flydsl

    if [ -n "$PERF_TEST_CMD" ]; then
        echo "Running PERF_TEST_CMD:"
        echo "  $PERF_TEST_CMD"
        bash -c "$PERF_TEST_CMD"
        return
    fi

    # run_flydsl_perf_test
}

do_perf_test
set +x
CONTAINER_SCRIPT
}

docker_tty_args() {
    if [ -t 0 ] && [ -t 1 ]; then
        echo "-it"
    else
        echo "-i"
    fi
}

docker_init_args() {
    case "${DOCKER_HOST:-}" in
        *podman.sock*|*/podman/*)
            return
            ;;
    esac

    echo "--init"
}

ensure_container_args() {
    if [ -z "$CONTAINER_NAME" ]; then
        die "Container name is required."
    fi
}

ensure_work_dir() {
    if [ -z "$WORK_DIR" ]; then
        die "Work directory is required."
    fi
}

build_att_enabled() {
    [ "${BUILD_ATT:-0}" = "1" ] || [ "${BUILD_ATT:-0}" = "true" ] || [ "${BUILD_ATT:-0}" = "yes" ]
}

prepare_rocm_systems_on_host() {
    local rocm_sys_dir

    if ! build_att_enabled; then
        return
    fi

    if [ -z "$CODE_DIR" ]; then
        die "CODE_DIR is not set."
    fi

    rocm_sys_dir="$CODE_DIR/rocm-systems"

    echo "=== Preparing rocm-systems on host ==="
    if [ -d "$rocm_sys_dir/.git" ]; then
        echo "  -> rocm-systems already cloned, reusing $rocm_sys_dir"
    elif [ -e "$rocm_sys_dir" ]; then
        die "$rocm_sys_dir exists but is not a git repository."
    else
        git clone --branch "$ROCM_SYSTEMS_BRANCH" --single-branch \
            "$ROCM_SYSTEMS_URL" "$rocm_sys_dir"
    fi
    git config --global --add safe.directory "$rocm_sys_dir" || true
}

prepare_perf_log_file() {
    local log_base_dir
    local log_timestamp

    if [ -n "$PERF_LOG_FILE" ]; then
        mkdir -p "$(dirname "$PERF_LOG_FILE")"
        DOCKER_INPUT_ENV="$(build_docker_input_env)"
        echo "Log file: $PERF_LOG_FILE"
        return
    fi

    if [ -n "$TARGET_DIR" ]; then
        log_base_dir="$TARGET_DIR"
    else
        log_base_dir="$WORK_DIR"
    fi

    if [ ! -d "$log_base_dir" ]; then
        die "Log directory '$log_base_dir' does not exist."
    fi

    log_timestamp=$(date +%Y%m%d_%H%M%S)
    PERF_LOG_FILE="$log_base_dir/setup_work_env_${CONTAINER_NAME}_${log_timestamp}.log"
    export PERF_LOG_FILE

    DOCKER_INPUT_ENV="$(build_docker_input_env)"
    echo "Log file: $PERF_LOG_FILE"
}

write_container_script_file() {
    local script_path="$CODE_DIR/.setup_work_env_${CONTAINER_NAME}.sh"

    do_perf_test_script > "$script_path"
    chmod +x "$script_path"
    echo "$script_path"
}

container_idle_script() {
    cat <<'CONTAINER_IDLE_SCRIPT'
trap 'exit 0' INT TERM
while true; do
    sleep 3600 &
    wait $!
done
CONTAINER_IDLE_SCRIPT
}

run_new_container() {
    local -a docker_volume_args

    ensure_container_args
    ensure_work_dir

    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        die "Container '$CONTAINER_NAME' already exists. Use existing-container or choose another name."
    fi

    prepare_perf_log_file
    set_code_dir_from_target
    prepare_rocm_systems_on_host

    docker_volume_args=(-v "$WORK_DIR:$WORK_DIR")
    if [ -n "$TARGET_DIR" ] && [ "$TARGET_DIR" != "$WORK_DIR" ]; then
        docker_volume_args+=(-v "$TARGET_DIR:$TARGET_DIR")
    fi

    echo "=== Creating container '$CONTAINER_NAME' ==="
    echo "Docker image: $DOCKER_IMAGE"
    echo "Code directory: $CODE_DIR"
    echo "Work directory: $WORK_DIR"
    echo ""

    # This matches the required ROCm privileged container setup.
    docker run -d \
        --name "$CONTAINER_NAME" \
        $(docker_init_args) \
        --sig-proxy=true \
        --network=host \
        --privileged \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --group-add=video \
        --ipc=host \
        --cap-add=CAP_SYS_ADMIN \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/mem \
        -e LANG="$CONTAINER_LANG" \
        -e LC_ALL="$CONTAINER_LC_ALL" \
        -e TERM="$CONTAINER_TERM" \
        -e DOCKER_INPUT_ENV="$DOCKER_INPUT_ENV" \
        "${docker_volume_args[@]}" \
        -w "$WORK_DIR" \
        --entrypoint /usr/bin/bash \
        "$DOCKER_IMAGE" \
        -c "$(container_idle_script)" >/dev/null

    run_existing_container
}

run_existing_container() {
    local status
    local container_script_path

    ensure_container_args
    ensure_work_dir

    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        die "Container '$CONTAINER_NAME' does not exist."
    fi

    status=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
    if [ "$status" != "running" ]; then
        echo "Container '$CONTAINER_NAME' is $status, starting it..."
        docker start "$CONTAINER_NAME" >/dev/null
    fi

    prepare_perf_log_file
    set_code_dir_from_target
    prepare_rocm_systems_on_host
    container_script_path=$(write_container_script_file)

    echo "=== Running performance test in existing container '$CONTAINER_NAME' ==="
    echo "Code directory: $CODE_DIR"
    echo "Work directory: $WORK_DIR"
    echo ""

    docker exec $(docker_tty_args) \
        -e LANG="$CONTAINER_LANG" \
        -e LC_ALL="$CONTAINER_LC_ALL" \
        -e TERM="$CONTAINER_TERM" \
        -e DOCKER_INPUT_ENV="$DOCKER_INPUT_ENV" \
        "$CONTAINER_NAME" \
        bash "$container_script_path"
}

print_clone_location() {
    if [ -n "$TARGET_DIR" ]; then
        echo "Location: $TARGET_DIR/"
        ls -1d "$TARGET_DIR"/*/ 2>/dev/null || true
    fi
}

parse_container_options() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --container-name|--name)
                need_value "$1" "${2:-}"
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --workdir|--pwd)
                need_value "$1" "${2:-}"
                WORK_DIR_INPUT="$2"
                shift 2
                ;;
            --target-dir)
                need_value "$1" "${2:-}"
                set_clone_target "$2"
                shift 2
                ;;
            --force-rebuild-llvm)
                FORCE_REBUILD_LLVM=1
                export FORCE_REBUILD_LLVM
                shift
                ;;
            --build-llvm)
                BUILD_LLVM=1
                export BUILD_LLVM
                shift
                ;;
            --build-att)
                BUILD_ATT=1
                export BUILD_ATT
                shift
                ;;
            -h|--help)
                usage 0
                ;;
            *)
                die "Unknown option '$1'."
                ;;
        esac
    done
}

main() {
    if [ $# -lt 1 ]; then
        usage
    fi

    case "$1" in
        -h|--help)
            usage 0
            ;;
        clone)
            shift
            if [ $# -ne 1 ]; then
                usage
            fi
            set_clone_target "$1"
            do_clone
            print_clone_location
            ;;
        update|--update)
            shift
            if [ $# -ne 1 ]; then
                usage
            fi
            set_clone_target "$1"
            do_update
            print_clone_location
            ;;
        setup-new)
            shift
            parse_container_options "$@"
            if [ -z "$TARGET_DIR" ]; then
                die "--target-dir is required for setup-new."
            fi
            do_clone
            if [ -n "$WORK_DIR_INPUT" ]; then
                set_work_dir "$WORK_DIR_INPUT"
            else
                WORK_DIR="$TARGET_DIR"
            fi
            run_new_container
            print_clone_location
            ;;
        new-container)
            shift
            parse_container_options "$@"
            if [ -z "$WORK_DIR_INPUT" ]; then
                die "--workdir is required for new-container."
            fi
            set_work_dir "$WORK_DIR_INPUT"
            run_new_container
            ;;
        existing-container)
            shift
            parse_container_options "$@"
            if [ -z "$TARGET_DIR" ]; then
                die "--target-dir is required for existing-container."
            fi
            if [ ! -d "$TARGET_DIR" ]; then
                die "--target-dir '$TARGET_DIR' must exist for existing-container."
            fi
            if [ -z "$WORK_DIR_INPUT" ]; then
                die "--workdir is required for existing-container."
            fi
            set_work_dir "$WORK_DIR_INPUT"
            run_existing_container
            ;;
        --all)
            shift
            if [ $# -ne 1 ]; then
                usage
            fi
            set_clone_target "$1"
            do_clone
            do_update
            print_clone_location
            ;;
        --*)
            die "Unknown option '$1'."
            ;;
        *)
            if [ $# -ne 1 ]; then
                usage
            fi
            set_clone_target "$1"
            do_clone
            print_clone_location
            ;;
    esac
}

main "$@"

set +x
