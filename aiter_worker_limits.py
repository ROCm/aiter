"""Worker limits shared by setup and AITER runtime code."""

import logging
import os
import posixpath
import warnings

CPU_CORE_COUNT_UTILIZATION = 0.80
# Approximate peak RSS observed per AOT worker, rounded up to 1.5 GB.
EST_WORKER_RSS_BYTES = 1_500_000_000
_WORKER_ENV = "AITER_MAX_JOBS"
_LEGACY_WORKER_ENV = "MAX_JOBS"
_PROC_SELF_CGROUP_PATH = "/proc/self/cgroup"
_PROC_SELF_MOUNTINFO_PATH = "/proc/self/mountinfo"
_logger = logging.getLogger(__name__)
_cgroup_memory_diagnostic_emitted = False
_last_cgroup_memory_observation: dict[str, object] | None = None


def _process_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def get_cpu_worker_budget(cpu_count: int | None = None) -> int:
    """Return at most 80% of logical CPUs, with one worker as the floor."""
    logical_cpus = (_process_cpu_count() if cpu_count is None else cpu_count) or 1
    return max(1, int(logical_cpus * CPU_CORE_COUNT_UTILIZATION))


def _host_available_memory_bytes() -> int:
    """Return currently available host memory as reported by psutil."""
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except Exception:  # noqa: BLE001
        return EST_WORKER_RSS_BYTES


def _decode_mountinfo_path(path: str) -> str:
    """Decode the escapes used for mount paths in /proc/self/mountinfo."""
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        path = path.replace(encoded, decoded)
    return path


def _resolve_cgroup_directory(
    mount_root: str, mount_point: str, membership_path: str
) -> str | None:
    """Map a cgroup membership path into its mounted filesystem path."""
    mount_root = posixpath.normpath(mount_root)
    membership_path = posixpath.normpath(membership_path)
    if membership_path == mount_root:
        relative = ""
    elif mount_root == "/":
        relative = membership_path.lstrip("/")
    elif membership_path.startswith(f"{mount_root.rstrip('/')}/"):
        relative = membership_path[len(mount_root) :].lstrip("/")
    else:
        return None
    return os.path.join(mount_point, *relative.split("/")) if relative else mount_point


def _cgroup_memory_directories() -> list[tuple[str, str]]:
    """Return current-to-root memory cgroup directories for v2 or v1."""
    try:
        with open(_PROC_SELF_CGROUP_PATH) as cgroup_file:
            cgroup_lines = cgroup_file.readlines()
        with open(_PROC_SELF_MOUNTINFO_PATH) as mountinfo_file:
            mountinfo_lines = mountinfo_file.readlines()
    except (FileNotFoundError, OSError):
        return []

    unified_path = None
    memory_path = None
    for line in cgroup_lines:
        try:
            hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
        except ValueError:
            continue
        if hierarchy == "0" and not controllers:
            unified_path = path
        if "memory" in controllers.split(","):
            memory_path = path

    directories = []
    seen_directories = set()
    for line in mountinfo_lines:
        try:
            mount_fields, filesystem_fields = line.rstrip("\n").split(" - ", 1)
            mount_fields = mount_fields.split()
            filesystem_fields = filesystem_fields.split()
            mount_root = _decode_mountinfo_path(mount_fields[3])
            mount_point = _decode_mountinfo_path(mount_fields[4])
            filesystem_type = filesystem_fields[0]
            super_options = filesystem_fields[2].split(",")
        except (IndexError, ValueError):
            continue

        version = None
        membership_path = None
        if filesystem_type == "cgroup2" and unified_path is not None:
            version = "v2"
            membership_path = unified_path
        elif (
            filesystem_type == "cgroup"
            and memory_path is not None
            and "memory" in super_options
        ):
            version = "v1"
            membership_path = memory_path
        if version is None:
            continue

        current = _resolve_cgroup_directory(mount_root, mount_point, membership_path)
        if current is None:
            continue

        while True:
            directory = (version, current)
            if directory not in seen_directories:
                directories.append(directory)
                seen_directories.add(directory)
            if current == mount_point:
                break
            parent = os.path.dirname(current)
            if (
                parent == current
                or os.path.commonpath((mount_point, parent)) != mount_point
            ):
                break
            current = parent
    return directories


def _cgroup_memory_remaining_bytes() -> int | None:
    """Return the tightest finite remaining-memory bound across cgroup ancestors."""
    global _last_cgroup_memory_observation

    remaining = None
    best_observation = None
    for version, directory in _cgroup_memory_directories():
        if version == "v2":
            limit_path = os.path.join(directory, "memory.max")
            usage_path = os.path.join(directory, "memory.current")
        else:
            limit_path = os.path.join(directory, "memory.limit_in_bytes")
            usage_path = os.path.join(directory, "memory.usage_in_bytes")

        try:
            with open(limit_path) as limit_file:
                raw_limit = limit_file.read().strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
        except (FileNotFoundError, OSError, ValueError):
            continue

        usage_readable = True
        try:
            with open(usage_path) as usage_file:
                usage = int(usage_file.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            # A finite limit with unknown usage must never be treated as an
            # empty cgroup: that would fail open and overstate safe headroom.
            usage = None
            usage_readable = False

        candidate = 0 if usage is None else max(0, limit - usage)
        if remaining is None or candidate < remaining:
            remaining = candidate
            best_observation = {
                "version": version,
                "directory": directory,
                "limit": limit,
                "usage": usage,
                "usage_readable": usage_readable,
            }
    _last_cgroup_memory_observation = best_observation
    return remaining


def _read_cgroup_memory_stat(observation: dict[str, object]) -> dict[str, int]:
    """Read diagnostic memory.stat fields for the selected cgroup bound."""
    directory = str(observation["directory"])
    try:
        with open(os.path.join(directory, "memory.stat")) as stat_file:
            values: dict[str, int] = {}
            for line in stat_file:
                key, _, raw_value = line.partition(" ")
                if key in {
                    "anon",
                    "file",
                    "active_file",
                    "inactive_file",
                    "slab_reclaimable",
                }:
                    try:
                        values[key] = int(raw_value.strip())
                    except ValueError:
                        continue
            return values
    except (FileNotFoundError, OSError):
        return {}


def _maybe_log_cgroup_memory_diagnostic(memory_budget: int) -> None:
    """Emit one diagnostic when conservative cgroup accounting is restrictive."""
    global _cgroup_memory_diagnostic_emitted

    if _cgroup_memory_diagnostic_emitted or memory_budget > 3:
        return
    observation = _last_cgroup_memory_observation
    if observation is None:
        return

    stats = _read_cgroup_memory_stat(observation)
    limit_gib = int(observation["limit"]) / 1024**3
    usage = observation["usage"]
    remaining = (
        int(observation["limit"]) - int(usage) if usage is not None else 0
    )
    remaining_gib = max(0, remaining) / 1024**3
    usage_text = (
        f"{int(observation['usage']) / 1024**3:.2f} GiB"
        if observation["usage"] is not None
        else "unavailable"
    )
    stat_text = ", ".join(
        f"{key}={value / 1024**3:.2f} GiB" for key, value in stats.items()
    ) or "memory.stat unavailable"
    unreadable_text = (
        "; usage was unreadable, so remaining memory was conservatively treated as 0"
        if not observation["usage_readable"]
        else ""
    )
    log = _logger.warning if memory_budget == 1 else _logger.info
    log(
        "AITER worker budget limited to %d by cgroup memory: limit=%.2f GiB, "
        "current=%s, remaining=%.2f GiB, worker_estimate=%.2f GiB; "
        "cgroup current usage conservatively includes page cache (%s)%s",
        memory_budget,
        limit_gib,
        usage_text,
        remaining_gib,
        EST_WORKER_RSS_BYTES / 1024**3,
        stat_text,
        unreadable_text,
    )
    _cgroup_memory_diagnostic_emitted = True


def _available_memory_bounds() -> tuple[int, int | None]:
    """Return host available memory and the optional cgroup bound."""
    host_available = _host_available_memory_bytes()
    cgroup_remaining = _cgroup_memory_remaining_bytes()
    return host_available, cgroup_remaining


def _available_memory_bytes() -> int:
    """Return memory available under both host and cgroup constraints."""
    host_available, cgroup_remaining = _available_memory_bounds()
    return (
        host_available
        if cgroup_remaining is None
        else min(host_available, cgroup_remaining)
    )


def get_automatic_worker_budgets() -> tuple[int, int]:
    """Return the CPU and memory worker budgets."""
    cpu_budget = get_cpu_worker_budget()
    host_available, cgroup_remaining = _available_memory_bounds()
    available_memory = (
        host_available
        if cgroup_remaining is None
        else min(host_available, cgroup_remaining)
    )
    memory_budget = max(1, available_memory // EST_WORKER_RSS_BYTES)
    if cgroup_remaining is not None and cgroup_remaining <= host_available:
        _maybe_log_cgroup_memory_diagnostic(memory_budget)
    return cpu_budget, memory_budget


def adopt_legacy_max_jobs() -> None:
    """Adopt a valid legacy ``MAX_JOBS`` value at AITER-owned entrypoints.

    This compatibility bridge must not be called from imports or runtime JIT
    helpers: parent frameworks own their generic ``MAX_JOBS`` setting. AITER's
    standalone build entrypoints call it explicitly before selecting workers.
    """
    if _WORKER_ENV in os.environ:
        return

    jobs = _get_legacy_worker_limit()
    if jobs is None:
        return

    os.environ[_WORKER_ENV] = str(jobs)
    warnings.warn(
        "MAX_JOBS controlling standalone AITER builds is deprecated; "
        "use AITER_MAX_JOBS instead.",
        FutureWarning,
        stacklevel=2,
    )


def _get_legacy_worker_limit() -> int | None:
    """Return a positive legacy MAX_JOBS ceiling, if one is configured."""
    raw = os.environ.get(_LEGACY_WORKER_ENV)
    if raw is None:
        return None
    try:
        jobs = int(raw)
    except ValueError:
        return None
    return jobs if jobs > 0 else None


def _get_worker_count(*, honor_legacy_max_jobs: bool) -> int:
    """Apply AITER's explicit ceiling to the current automatic worker budget."""
    automatic_workers = max(1, min(get_automatic_worker_budgets()))
    raw = os.environ.get(_WORKER_ENV)
    if raw is None and honor_legacy_max_jobs:
        legacy_limit = _get_legacy_worker_limit()
        if legacy_limit is None:
            return automatic_workers
        return min(automatic_workers, legacy_limit)
    if raw is None:
        return automatic_workers
    try:
        configured_limit = max(1, int(raw))
    except ValueError:
        return automatic_workers
    return min(automatic_workers, configured_limit)


def get_worker_count() -> int:
    """Return the current AITER worker budget using only AITER_MAX_JOBS."""
    return _get_worker_count(honor_legacy_max_jobs=False)


def get_compile_worker_count() -> int:
    """Return the worker budget for AITER-owned runtime compilation.

    Runtime JIT consults ``MAX_JOBS`` as a non-mutating legacy ceiling only
    when ``AITER_MAX_JOBS`` is unset. Both ceilings remain bounded by the live
    CPU and memory limits; importing AITER does not perform this lookup.
    """
    return _get_worker_count(honor_legacy_max_jobs=True)


def get_worker_count_for(work_count: int) -> int:
    """Cap the global worker budget to available work, with a floor of one."""
    return min(get_worker_count(), max(1, int(work_count)))


def configure_worker_subprocesses() -> None:
    """Force compiler descendants of a process-pool worker to one job."""
    os.environ[_WORKER_ENV] = "1"
    os.environ.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "MAKEFLAGS": "-j1",
            "NINJAFLAGS": "-j1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
