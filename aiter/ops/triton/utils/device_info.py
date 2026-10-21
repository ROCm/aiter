import functools
import os
import re

# hipDeviceAttributeNumberOfXccs is not a stable integer. New enumerators
# inserted ahead of it in hip_runtime_api.h shift the value, and
# hipDeviceAttributeClockInstructionRate shares the value of
# hipDeviceAttributeAmdSpecificBegin, so a naive count is off by one.
_HIP_NUMBER_OF_XCCS_ATTRIBUTE = "hipDeviceAttributeNumberOfXccs"


@functools.lru_cache(maxsize=1)
def get_num_sms():
    """Return the compute-unit count used for grid sizing.

    Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    so grid sizing stays consistent with the selected tuned configs. Fall back
    to torch's multi_processor_count when get_cu_num() is unavailable, for
    example when rocminfo is missing or unparseable.
    """
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:  # noqa: BLE001
        import torch

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


# Kernels use the XCD count to keep workgroups that share data on one L2.
# ROCm 7+ reports the live count through hipDeviceAttributeNumberOfXccs,
# including partition mode. Older runtimes do not, so those releases keep
# this table. Every other arch and every unlisted CU count returns 1.
# 0 is not a usable count: callers divide and modulo by it.
_XCDS_BY_GFX_AND_CU_NUM = {
    ("gfx942", 304): 8,  # MI300X, MI325X
    ("gfx942", 228): 6,  # MI300A
    ("gfx950", 256): 8,  # MI350X, MI355X
    ("gfx950", 128): 4,  # MI350P
    ("gfx1250", 256): 8,  # MI455X
}


def _xcds_for_part(gfx: str, cu_num: int) -> int:
    """Return the XCD count for this arch and CU count, or 1 if it is not listed."""
    return _XCDS_BY_GFX_AND_CU_NUM.get((gfx, cu_num), 1)


def _cu_num_override() -> int | None:
    """Return a positive CU_NUM override, or None when it is unset.

    get_cu_num() treats 0 as "detect the device", so only a positive value
    selects a simulated part.
    """
    raw = os.environ.get("CU_NUM")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _xcds_from_known_part(device_id=None) -> int:
    """Look up XCDs from the architecture and CU count.

    A CU_NUM override uses that count, so the result matches get_num_sms().
    Otherwise the CU count is read from ``device_id`` when that device can be
    queried. The architecture comes from process-wide detection. Returns 1
    when the arch or CU count cannot be read.
    """
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        gfx = get_gfx_runtime()
        override = _cu_num_override()
        if device_id is None and override is None:
            try:
                device_id = _current_device_id()
            except Exception:  # noqa: BLE001
                device_id = None
        cu_num = None
        if device_id is not None and override is None:
            cu_num = _cu_num_for_device(int(device_id), gfx)
        if not cu_num:
            cu_num = get_cu_num()
        return _xcds_for_part(gfx, cu_num)
    except Exception:  # noqa: BLE001
        return 1


@functools.cache
def _rocm_major():
    """Return the live HIP runtime's ROCm major version, or None on failure."""
    try:
        import ctypes

        libhip = _hip_runtime()
        libhip.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        libhip.hipRuntimeGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        if libhip.hipRuntimeGetVersion(ctypes.byref(version)) != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    return version.value // 10_000_000


def _current_device_id() -> int:
    """Return the torch index of the HIP device currently in use."""
    import torch

    return int(torch.cuda.current_device())


def _strip_c_comments(text: str) -> str:
    """Remove C block and line comments so enumerator names can be scanned."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", text, flags=re.MULTILINE)


def _assigned_attribute_value(rhs: str, known: dict[str, int]) -> int:
    """Evaluate one enumerator initializer.

    A bare integer is that value. A previous enumerator keeps its value, and
    ``previous + N`` or ``previous - N`` adjusts it. Anything else cannot be
    evaluated from the text alone.
    """
    rhs = rhs.strip()
    number = re.fullmatch(r"[+-]?\d+|0[xX][0-9A-Fa-f]+", rhs)
    if number:
        return int(number.group(0), 0)
    alias = re.fullmatch(r"hipDeviceAttribute\w+", rhs)
    if alias and alias.group(0) in known:
        return known[alias.group(0)]
    adjusted = re.fullmatch(
        r"(hipDeviceAttribute\w+)\s*([+-])\s*(\d+|0[xX][0-9A-Fa-f]+)",
        rhs,
    )
    if adjusted and adjusted.group(1) in known:
        delta = int(adjusted.group(3), 0)
        base = known[adjusted.group(1)]
        return base + delta if adjusted.group(2) == "+" else base - delta
    raise ValueError(f"cannot evaluate enumerator initializer {rhs!r}")


def _hip_device_attribute_value(
    header_text: str,
    target: str,
    origin: str = "hipDeviceAttributeAmdSpecificBegin",
) -> int:
    """Return the integer value of one hipDeviceAttribute_t enumerator.

    The walk starts at ``origin``. An enumerator assigned to an earlier name
    keeps that name's value, including a constant added to or subtracted from
    it. A preprocessor directive in that span is rejected: the active branch
    is not known from the header text.
    """
    text = _strip_c_comments(header_text)
    origin_at = re.search(rf"\b{re.escape(origin)}\b", text)
    if origin_at is None:
        raise ValueError(f"{origin} is missing")
    begin = origin_at.start()
    match = re.search(rf"\b{re.escape(target)}\b", text[begin:])
    if match is None:
        raise ValueError(f"{target} is missing")
    target_at = begin + match.start()
    rest = text[target_at + len(target) :]
    comma = rest.find(",")
    end = len(text) if comma < 0 else target_at + len(target) + comma
    chunk = text[begin:end]
    if re.search(r"^\s*#", chunk, flags=re.MULTILINE):
        raise ValueError(f"{target} depends on a preprocessor conditional")
    value = None
    known = {}
    for raw in chunk.split(","):
        raw = raw.strip()
        if not raw:
            continue
        left, eq, rhs = raw.partition("=")
        name = left.strip().split()[-1]
        if not name.startswith("hipDeviceAttribute"):
            continue
        if eq:
            value = _assigned_attribute_value(rhs, known)
        else:
            value = 0 if value is None else value + 1
        known[name] = value
        if name == target:
            return value
    raise ValueError(f"{target} is missing")


def _header_beside_library(library_path: str) -> str | None:
    """Return hip_runtime_api.h beside this libamdhip64, if that tree has it."""
    directory = os.path.dirname(os.path.realpath(library_path))
    for _ in range(5):
        header = os.path.join(directory, "include", "hip", "hip_runtime_api.h")
        if os.path.isfile(header):
            return header
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return None


def _hip_runtime():
    """Return the shared, cached HIP runtime handle."""
    from aiter.jit.utils.chip_info import _load_hip_runtime

    return _load_hip_runtime()


def _loaded_amdhip64_path() -> str | None:
    """Return the path of the shared HIP runtime handle."""
    libhip = _hip_runtime()
    name = getattr(libhip, "_name", "")
    if name and os.path.isfile(name):
        return os.path.realpath(name)
    maps = "/proc/self/maps"
    if not os.path.isfile(maps):
        return None
    with open(maps) as loaded:
        for line in loaded:
            if "amdhip64" not in line:
                continue
            path = line.rsplit(" ", 1)[-1].strip()
            if path and os.path.isfile(path):
                return os.path.realpath(path)
    return None


def _hip_runtime_api_header() -> str:
    """Return the hip_runtime_api.h that matches the loaded libamdhip64.

    A header from any other installation is unsafe because HIP attribute
    integers can change between releases.
    """
    library = _loaded_amdhip64_path()
    if library is None:
        raise FileNotFoundError("loaded libamdhip64 path")
    header = _header_beside_library(library)
    if header is None:
        raise FileNotFoundError(f"hip_runtime_api.h beside {library}")
    return header


@functools.cache
def _number_of_xccs_attribute() -> int:
    """Read hipDeviceAttributeNumberOfXccs from the loaded library's header."""
    with open(_hip_runtime_api_header()) as header:
        return _hip_device_attribute_value(header.read(), _HIP_NUMBER_OF_XCCS_ATTRIBUTE)


def _hip_device_get_attribute(device_id: int, attribute: int) -> int:
    """Call hipDeviceGetAttribute for one device and attribute id.

    hipError_t hipDeviceGetAttribute(
        int* pi, hipDeviceAttribute_t attr, int deviceId)
    """
    import ctypes

    libhip = _hip_runtime()
    libhip.hipDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    libhip.hipDeviceGetAttribute.restype = ctypes.c_int
    value = ctypes.c_int()
    err = libhip.hipDeviceGetAttribute(
        ctypes.byref(value),
        int(attribute),
        int(device_id),
    )
    if err != 0:
        raise RuntimeError(f"hipDeviceGetAttribute failed with error {err}")
    return int(value.value)


def _hip_num_xccs(device_id: int) -> int:
    """Query hipDeviceAttributeNumberOfXccs.

    Raises when the runtime rejects the attribute, the HIP library cannot
    be loaded, or the loaded library's header does not define the attribute.
    """
    return _hip_device_get_attribute(device_id, _number_of_xccs_attribute())


_HIP_MULTIPROCESSOR_COUNT_ATTRIBUTE = "hipDeviceAttributeMultiprocessorCount"


@functools.cache
def _multiprocessor_count_attribute() -> int:
    """Read hipDeviceAttributeMultiprocessorCount from the loaded HIP header."""
    with open(_hip_runtime_api_header()) as header:
        return _hip_device_attribute_value(
            header.read(),
            _HIP_MULTIPROCESSOR_COUNT_ATTRIBUTE,
            origin="hipDeviceAttributeCudaCompatibleBegin",
        )


@functools.cache
def _cu_num_for_device(device_id: int, gfx: str) -> int:
    """Return this device's CU count, or 0 when it cannot be queried.

    hipDeviceAttributeMultiprocessorCount is one CU in CU mode and one WGP
    in WGP mode. The cached 0 keeps a rejected query off the launch path.
    """
    try:
        from aiter.jit.utils.build_targets import torch_processor_count_to_cu

        processors = _hip_device_get_attribute(
            device_id, _multiprocessor_count_attribute()
        )
    except Exception:  # noqa: BLE001
        return 0
    return max(0, torch_processor_count_to_cu(gfx, processors))


@functools.cache
def _query_num_xccs(device_id: int) -> int:
    """Return the XCC count for device_id, or 0 when the query fails.

    Both results are cached. functools.cache does not store exceptions, and
    this runs on kernel launch paths. A runtime that rejects the attribute
    keeps rejecting it, so a failed driver call is not repeated. The part-table
    lookup that follows a 0 is already cached.
    """
    try:
        count = _hip_num_xccs(int(device_id))
    except Exception:  # noqa: BLE001
        return 0
    return max(0, count)


def get_num_xcds(device_id=None) -> int:
    """Return the number of XCDs visible on ``device_id``.

    On ROCm 7 and newer this is ``hipDeviceAttributeNumberOfXccs``. Earlier
    releases use the architecture and CU count. A positive CU_NUM override
    uses that table as well, so a simulated part stays consistent with
    get_num_sms(). Only the CDNA3, CDNA4, and CDNA5 parts in the table have
    a known XCD count. Every other case returns 1, because callers divide by
    the count and one L2 domain is an identity remap. An undetected ROCm
    version tries the query first.
    """
    if _cu_num_override() is not None:
        return _xcds_from_known_part(None)
    major = _rocm_major()
    if major is None or major >= 7:
        try:
            if device_id is None:
                device_id = _current_device_id()
            count = _query_num_xccs(int(device_id))
        except Exception:  # noqa: BLE001
            count = 0
        if count > 0:
            return count
    return _xcds_from_known_part(device_id)
