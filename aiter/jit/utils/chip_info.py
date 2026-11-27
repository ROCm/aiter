# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging
import os
import re
import subprocess
import sys
from typing import NamedTuple

from build_targets import (
    GFX_MAP,
    _parse_gpu_archs_env,
    filter_tune_df,
    get_build_targets_env,
    torch_processor_count_to_cu,
)
from cpp_extension import executable_path
from hip_runtime import (
    get_current_hip_device,
    get_hip_runtime_version,
    load_hip_runtime,
)
from torch_guard import torch_compile_guard

try:
    from torch.compiler import assume_constant_result
except ImportError:  # no torch, as in some build environments

    def assume_constant_result(fn):
        return fn


logger = logging.getLogger("aiter")

IS_WINDOWS = sys.platform == "win32"
# Windows ROCm ships `hipinfo` instead of `rocminfo`; the output formats differ,
# so each has its own compute-unit parser below.
_GPU_INFO_TOOL = "hipinfo" if IS_WINDOWS else "rocminfo"
_GFX_ARCH_PATTERN = r"gfx[0-9a-z]+(?:-[0-9a-z]+)*"


@functools.lru_cache(maxsize=1)
def _gpu_info_output() -> str:
    """rocminfo / hipinfo stdout, run once and shared by every parser below.

    check=False on purpose: rocminfo can exit non-zero while still printing
    usable agent blocks. Each parser decides whether what it needs is missing.
    """
    tool = executable_path(_GPU_INFO_TOOL)
    result = subprocess.run([tool], capture_output=True, text=True, check=False)
    return result.stdout


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        # rocminfo: "Name: gfx942"; hipinfo: "gcnArchName: gfx1200"
        for line in _gpu_info_output().splitlines():
            # Preserve the exact ROCr-reported identity, including the
            # gfx1250-strict (MI450 A0) variant. strict is mapped to the gfx1250
            # kernel family only at explicitly validated dispatch sites, via
            # base_gfx_arch()/is_gfx1250_family(); build artifacts, caches, tuned
            # configs and prebuilt binaries stay keyed by the exact target.
            # Folding here would make _needs_arch_rebuild() treat a strict .so as
            # not covering a strict device and rebuild in a loop once ROCr
            # reports gfx1250-strict.
            match = re.search(rf"\b({_GFX_ARCH_PATTERN})\b", line, re.IGNORECASE)
            if match:
                return [match.group(1).lower()]
    except Exception as e:
        raise RuntimeError(f"Get GPU arch from {_GPU_INFO_TOOL} failed: {e}") from e
    raise RuntimeError(f"No gfx arch found in {_GPU_INFO_TOOL} output.")


@torch_compile_guard()
def get_gfx_custom_op() -> int:
    return get_gfx_custom_op_core()


@functools.lru_cache(maxsize=10)
def get_gfx_custom_op_core() -> int:
    gfx = os.getenv("GPU_ARCHS", "native")
    gfx_mapping = {v: k for k, v in GFX_MAP.items()}
    if gfx == "native":
        gfx = _detect_native()[0]
    elif ";" in gfx:
        # TODO: multi-arch GPU_ARCHS (e.g. "gfx942;gfx950") -- picking the
        # last entry is a known limitation for build-time codegen callers.
        # For runtime dispatch, prefer get_gfx_runtime().
        gfx = gfx.split(";")[-1]
    try:
        return gfx_mapping[gfx]
    except KeyError:
        raise KeyError(
            f"Unknown GPU architecture: {gfx}. "
            f"Supported architectures: {list(gfx_mapping.keys())}"
        )


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx_num = get_gfx_custom_op()
    return GFX_MAP.get(gfx_num, "unknown")


if __package__:
    from . import lds_capacity as _lds_capacity
else:
    import lds_capacity as _lds_capacity

_LDS_CAPACITY_BYTES = _lds_capacity._LDS_CAPACITY_BYTES


def get_lds_capacity_bytes(gfx: str | None = None) -> int:
    """Return the architectural LDS capacity for one workgroup."""
    return _lds_capacity.get_lds_capacity_bytes(gfx or get_gfx())


@functools.lru_cache(maxsize=1)
def get_gfx_runtime() -> str:
    """Return the live GPU arch, using Torch if Windows hipinfo is unavailable.

    Unlike get_gfx(), ignores GPU_ARCHS -- always detects the actual running
    GPU.  Use for runtime dispatch decisions (selecting tuned kernels, picking
    code paths).  Use get_gfx() for build-time codegen paths (gen_instances,
    csrc module-level arch selection) where no GPU may be available.

    The exception is an AOT replay process (AITER_AOT_REPLAY=1, started by
    aiter.aot.flydsl.common.replay_env): it runs the runtime without a GPU for
    the one arch GPU_ARCHS names, which stands in for the live GPU there.
    """
    if os.getenv("AITER_AOT_REPLAY") == "1":
        targets = _parse_gpu_archs_env(os.getenv("GPU_ARCHS", ""))
        if len(targets) != 1:
            raise RuntimeError(
                "AITER_AOT_REPLAY=1 needs GPU_ARCHS to name the one arch the "
                f"process compiles for, got {os.getenv('GPU_ARCHS')!r}"
            )
        gfx_arch = targets[0]
    else:
        try:
            gfx_arch = _detect_native()[0]
        except RuntimeError:
            if not IS_WINDOWS:
                raise
            import torch

            if torch.version.hip is None or not torch.cuda.is_available():
                raise
            gfx_arch = get_gfx_runtime_for_device(0)
    supported = set(GFX_MAP.values())
    if gfx_arch not in supported:
        raise KeyError(
            f"Unknown GPU architecture: {gfx_arch}. "
            f"Supported architectures: {sorted(supported)}"
        )
    return gfx_arch


def get_gfx_runtime_for_device(device=None) -> str:
    """Return the selected Torch device's exact runtime architecture.

    None (or an unindexed CUDA device) follows the current device each call,
    so this query must not use the node-wide get_gfx_runtime cache. Device
    errors propagate. Build-time GPU_ARCHS does not affect this query.
    """
    import torch

    return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0].lower()


def _parse_asic_revision(value: str) -> int:
    """Parse rocminfo/hipinfo decimal or hex ASIC Revision, not chip revision."""
    token = value.split("(", 1)[0].strip()
    if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", token):
        raise ValueError(f"invalid ASIC Revision {value!r}")
    return int(token, 16 if token.lower().startswith("0x") else 10)


@functools.lru_cache(maxsize=1)
def _rocminfo_gpu_agents() -> list[tuple[str, int]]:
    """(arch, asicRevision) of every GPU agent rocminfo reports.

    rocminfo's `ASIC Revision:` is HSA_AMD_AGENT_INFO_ASIC_REVISION, the same
    value hipDeviceGetAttribute(hipDeviceAttributeAsicRevision) returns.
    """
    agents: list[tuple[str, int]] = []
    name = kind = None
    rev = None
    for line in _gpu_info_output().splitlines():
        if re.match(r"^\s*Agent\s+\d+\s*$", line):
            name = kind = rev = None
            continue
        # 2-space indent = agent header; ISA/pool sub-blocks nest deeper.
        m = re.match(r"^  (Name|Device Type|ASIC Revision):\s+(.*?)\s*$", line)
        if m is None:
            continue
        key, value = m.group(1), m.group(2)
        if key == "Name" and name is None:
            # Agent names can carry target-id features ("gfx942:sramecc+").
            m2 = re.search(rf"\b({_GFX_ARCH_PATTERN})\b", value, re.IGNORECASE)
            name = m2.group(1) if m2 else value
        elif key == "Device Type":
            kind = value
        elif key == "ASIC Revision":
            rev = _parse_asic_revision(value)
        if kind == "GPU" and name is not None and rev is not None:
            agents.append((name.lower(), rev))
            name = kind = rev = None
    return agents


@functools.lru_cache(maxsize=1)
def _hipinfo_gpu_agents() -> list[tuple[str, int]]:
    """(arch, asicRevision) of every device hipinfo reports."""
    agents: list[tuple[str, int]] = []
    for block in re.split(r"^device#", _gpu_info_output(), flags=re.MULTILINE)[1:]:
        # gcnArchName can carry target-id features ("gfx942:sramecc+").
        name = re.search(
            rf"^gcnArchName:\s*({_GFX_ARCH_PATTERN})",
            block,
            re.MULTILINE | re.IGNORECASE,
        )
        rev = re.search(
            r"^asicRevision:\s*(0[xX][0-9a-fA-F]+|\d+)", block, re.MULTILINE
        )
        if name and rev:
            agents.append((name.group(1).lower(), _parse_asic_revision(rev.group(1))))
    return agents


@functools.lru_cache(maxsize=1)
def get_asic_revision() -> int:
    """Silicon stepping of this node's GPUs: 0=A0, 1=B0, 2=C0, ...

    Node-wide, not per-device: rocminfo enumerates HSA agents, which
    HIP_VISIBLE_DEVICES does not filter, so agent index and HIP device index
    can disagree. The lowest stepping is reported, so the asm gate on a
    mixed-stepping node fails closed. Raises when nothing can be read.
    hipinfo on Windows lists only the devices HIP_VISIBLE_DEVICES leaves.
    """
    agents = _hipinfo_gpu_agents() if IS_WINDOWS else _rocminfo_gpu_agents()
    arch = get_gfx_runtime()
    revs = [rev for name, rev in agents if name == arch]
    if not revs:
        raise RuntimeError(
            f"{_GPU_INFO_TOOL} reported no ASIC Revision for a {arch} agent "
            f"(GPU agents seen: {agents})"
        )
    rev = min(revs)
    # A bogus parse must not read as A0 and disable asm on good silicon.
    if not 0 <= rev <= 15:
        raise RuntimeError(
            f"implausible ASIC Revision {rev} parsed from {_GPU_INFO_TOOL}"
        )
    return rev


# Backfill map for legacy tuned configs that predate the `gfx` column.
# These cu_num values were only ever tuned on a single arch historically:
#   256 -> gfx950, 80/228/304 -> gfx942.
# Newer archs that happen to share a cu_num (e.g. gfx1250 also reports 256)
# are always written with their real arch by the tuner, so they never rely on
# this backfill.
_LEGACY_CU_NUM_TO_GFX = {
    256: "gfx950",
    80: "gfx942",
    228: "gfx942",
    304: "gfx942",
}


def gfx_from_cu_num(cu_num) -> str:
    """Infer the gfx arch for a legacy config row that has no `gfx` column.

    Used to migrate old tuned CSVs (keyed on cu_num only) to the new
    (gfx, cu_num, ...) schema. Unknown cu_num falls back to the live GPU arch.
    """
    try:
        cu_num = int(cu_num)
    except (TypeError, ValueError):
        return get_gfx_runtime()
    gfx = _LEGACY_CU_NUM_TO_GFX.get(cu_num)
    if gfx is not None:
        return gfx
    try:
        return get_gfx_runtime()
    except Exception:  # noqa: BLE001
        return "gfx942"


@functools.lru_cache(maxsize=1)
def get_gfx_list() -> list[str]:

    gfx_env = os.getenv("GPU_ARCHS", "native")
    if gfx_env == "native":
        try:
            gfxs = _detect_native()
        except RuntimeError:
            gfxs = ["cpu"]
    else:
        gfxs = _parse_gpu_archs_env(gfx_env)
    os.environ["AITER_GPU_ARCHS"] = ";".join(gfxs)

    return gfxs


class PartitionInfo(NamedTuple):
    """Compute partition and CU counts of one logical GPU.

    The partition fields are None when the source cannot report them.
    partition_inferred is True when amdsmi reports no partition and the mode
    and number of partitions come from the CU count of the full chip.
    """

    gfx: str
    cu_num: int
    full_chip_cu_num: int | None
    compute_partition: str | None
    memory_partition: str | None
    num_partitions: int | None
    partition_inferred: bool = False


def _amdsmi_value(value):
    """amdsmi returns "N/A" for a value that the device does not report."""
    return None if value == "N/A" else value


def _amdsmi_call(amdsmi, name: str, *args):
    """Result of an amdsmi function, or None.

    The result is None when the device does not report the value, or when
    the installed amdsmi is too old to have the function.
    """
    func = getattr(amdsmi, name, None)
    if func is None:
        return None
    try:
        return func(*args)
    except amdsmi.AmdSmiException:
        return None


def _amdsmi_partition(amdsmi, handle) -> tuple:
    """(mode, memory mode, number of partitions) that one handle reports."""
    profile = _amdsmi_call(
        amdsmi, "amdsmi_get_gpu_accelerator_partition_profile", handle
    )
    if profile is not None:
        profile = profile["partition_profile"]
        mode = _amdsmi_value(profile["profile_type"])
        num_partitions = _amdsmi_value(profile["num_partitions"])
    else:
        mode = _amdsmi_call(amdsmi, "amdsmi_get_gpu_compute_partition", handle)
        num_partitions = None
    nps = _amdsmi_call(amdsmi, "amdsmi_get_gpu_memory_partition", handle)
    return _amdsmi_value(mode) or None, _amdsmi_value(nps) or None, num_partitions


# CU count of the full chip, by (PCI device ID, revision ID). The IDs and names
# are from data/amdgpu.ids of libdrm. The CU counts are from the ROCm GPU
# specifications, and 80 for MI308X.
_FULL_CHIP_CU_NUM = {
    (0x74A0, 0x00): 228,  # MI300A
    (0x74A1, 0x00): 304,  # MI300X
    (0x74A2, 0x00): 80,  # MI308X
    (0x74A2, 0xA1): 304,  # MI300X
    (0x74A5, 0x00): 304,  # MI325X
    (0x74B5, 0x00): 304,  # MI300X VF
    (0x74B6, 0x00): 80,  # MI308X
    (0x74B6, 0xA1): 304,  # MI300X
    (0x74B9, 0x00): 304,  # MI325X VF
    (0x75A0, 0x00): 256,  # MI350X
    (0x75A3, 0x00): 256,  # MI355X
    (0x75A8, 0x00): 128,  # MI350P
    (0x75B0, 0x00): 256,  # MI350X VF
    (0x75B3, 0x00): 256,  # MI355X VF
    (0x75B8, 0x00): 128,  # MI350P VF
}
_PARTITION_MODES = {1: "SPX", 2: "DPX", 4: "QPX", 8: "CPX"}


def _pci_id(value) -> int | None:
    try:
        return int(value, 16) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def _inferred_partition(asic: dict, cu_num: int) -> tuple[str, int] | None:
    """(mode, number of partitions) of a GPU that does not report its partition.

    An SR-IOV VF guest, or a container that sees only a secondary partition,
    gets N/A for the mode. A compute partition has an equal share of the CUs
    of its chip, so a GPU with 1/2, 1/4 or 1/8 of the CUs of its full chip is
    a partition, and a GPU with all of them is in SPX mode. A GPU of an
    unknown device ID gets no mode.
    """
    full_cu_num = _FULL_CHIP_CU_NUM.get(
        (_pci_id(asic.get("device_id")), _pci_id(asic.get("rev_id")))
    )
    if not full_cu_num or not cu_num or full_cu_num % cu_num:
        return None
    mode = _PARTITION_MODES.get(full_cu_num // cu_num)
    return (mode, full_cu_num // cu_num) if mode else None


_SYSFS_DRM = "/sys/class/drm"


def _render_device(render_minor) -> str | None:
    """Name of the device that owns a render node: a PCI address or amdgpu_xcp_<n>."""
    if render_minor is None:
        return None
    path = os.path.join(_SYSFS_DRM, f"renderD{render_minor}", "device")
    return os.path.basename(os.path.realpath(path)) if os.path.exists(path) else None


def _amdsmi_gpus() -> list[tuple[str, PartitionInfo]]:
    """(UUID, PartitionInfo) of every GPU amdsmi reports, in HIP order.

    One amdsmi handle is one KFD node, that is one compute partition. Its CU
    count is the CU count of the partition, the value HIP reports. KFD gives
    each partition an equal share of the CUs of the chip. Partition 0 owns the
    render node of the PCI device. A secondary partition owns the render node
    of an amdgpu_xcp platform device, and KFD gives it the address of its GPU
    with the partition index as the PCI function. Only such a partition takes
    a field that it does not report from partition 0 of its GPU. Separate PCI
    functions, for example SR-IOV VFs, never share fields. A GPU that still
    has no mode and no number of partitions gets them from the CU count of
    its full chip (see _inferred_partition).
    """
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        gpus = []
        for index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
            enum = _amdsmi_call(amdsmi, "amdsmi_get_gpu_enumeration_info", handle) or {}
            gpus.append(
                (
                    enum.get("hip_id", index),
                    enum.get("hip_uuid", ""),
                    amdsmi.amdsmi_get_gpu_device_bdf(handle).lower(),
                    _render_device(enum.get("drm_render")),
                    amdsmi.amdsmi_get_gpu_asic_info(handle),
                    _amdsmi_partition(amdsmi, handle),
                )
            )
    finally:
        amdsmi.amdsmi_shut_down()
    if not gpus:
        raise RuntimeError("amdsmi reports no GPU")
    partition0 = {bdf: fields for _, _, bdf, dev, _, fields in gpus if dev == bdf}
    result = []
    for _, uuid, bdf, dev, asic, fields in sorted(gpus, key=lambda gpu: gpu[0]):
        if dev is not None and dev.startswith("amdgpu_xcp"):
            gpu_fields = partition0.get(bdf[:-1] + "0", (None, None, None))
            fields = tuple(own or gpu for own, gpu in zip(fields, gpu_fields))
        mode, nps, num_partitions = fields
        cu_num = int(asic["num_compute_units"])
        inferred = None
        if mode is None and num_partitions is None:
            inferred = _inferred_partition(asic, cu_num)
        if inferred:
            mode, num_partitions = inferred
            logger.info(
                "GPU %s (%s, device ID %s) reports no compute partition; "
                "it has %d of the %d CUs of the full chip, so it is in %s mode",
                bdf,
                asic.get("market_name"),
                asic.get("device_id"),
                cu_num,
                cu_num * num_partitions,
                mode,
            )
        info = PartitionInfo(
            gfx=str(asic["target_graphics_version"]).lower(),
            cu_num=cu_num,
            full_chip_cu_num=cu_num * num_partitions if num_partitions else None,
            compute_partition=mode,
            memory_partition=nps,
            num_partitions=num_partitions,
            partition_inferred=inferred is not None,
        )
        result.append((uuid, info))
    return result


def _rocminfo_gpus() -> list[tuple[str, PartitionInfo]]:
    """(UUID, PartitionInfo) of every GPU agent rocminfo reports.

    rocminfo applies ROCR_VISIBLE_DEVICES itself and cannot report partitions.
    """
    gpus = []
    for block in re.split(
        r"^\s*Agent\s+\d+\s*$", _gpu_info_output(), flags=re.MULTILINE
    ):
        fields = dict(
            re.findall(
                r"^  (Name|Uuid|Device Type|Compute Unit):\s+(\S+)", block, re.MULTILINE
            )
        )
        if fields.get("Device Type") == "GPU" and "Compute Unit" in fields:
            info = PartitionInfo(
                fields["Name"].lower(),
                int(fields["Compute Unit"]),
                None,
                None,
                None,
                None,
            )
            gpus.append((fields.get("Uuid", ""), info))
    return gpus


@functools.lru_cache(maxsize=1)
def _query_node_gpus() -> tuple[list[tuple[str, PartitionInfo]], bool, str | None]:
    try:
        return _amdsmi_gpus(), True, None
    except Exception as e:  # noqa: BLE001  amdsmi can be missing or unsupported
        logger.debug("amdsmi partition query failed (%s), using %s", e, _GPU_INFO_TOOL)
    try:
        return _rocminfo_gpus(), False, None
    except Exception as e:  # noqa: BLE001  blanket catch is intentional here
        return [], False, f"Get GPU Compute Unit from {_GPU_INFO_TOOL} failed {e!s}"


def _node_gpus() -> tuple[list[tuple[str, PartitionInfo]], bool]:
    """The GPUs of the node and whether amdsmi reported them.

    The amdsmi session is the expensive part, so the query runs once per
    process, also when it fails. Without amdsmi, rocminfo gives the GPUs.
    """
    gpus, from_amdsmi, error = _query_node_gpus()
    if error:
        raise RuntimeError(error)
    return gpus, from_amdsmi


def _rocr_visible(uuids: list[str]) -> list[int]:
    """KFD GPU indices left by ROCR_VISIBLE_DEVICES, as ROCr parses it."""
    env = os.environ.get("ROCR_VISIBLE_DEVICES")
    if env is None:
        return list(range(len(uuids)))
    visible: list[int] = []
    for token in env.split(",")[: len(uuids)]:
        token = token.strip().upper()
        if token.startswith("G"):
            hits = [i for i, uuid in enumerate(uuids) if uuid.upper().startswith(token)]
            if not 5 <= len(token) <= 20 or len(hits) != 1:
                break
            index = hits[0]
        else:
            try:
                index = int(token, 0)
            except ValueError:
                break
        if not 0 <= index < len(uuids) or index in visible:
            break
        visible.append(index)
    return visible


def _hip_visible(visible: list[int], uuids: list[str]) -> list[int]:
    """Narrow ROCr-visible indices by HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not env:
        return visible
    selected: list[int] = []
    for token in env.split(","):
        if "GPU-" in token:
            token = next(
                (str(i) for i, index in enumerate(visible) if token in uuids[index]),
                token,
            )
        if (
            not token.isdigit()
            or str(int(token)) != token
            or int(token) >= len(visible)
        ):
            break
        if visible[int(token)] not in selected:
            selected.append(visible[int(token)])
    return selected


def _current_device() -> int:
    """HIP ordinal of the current device, without initializing HIP."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return torch.cuda.current_device()
    return 0


def _visible_gpus() -> list[PartitionInfo]:
    """PartitionInfo of every GPU visible to HIP, in HIP order.

    The GPUs are selected by ROCR_VISIBLE_DEVICES and then HIP_VISIBLE_DEVICES
    or CUDA_VISIBLE_DEVICES, as ROCr and HIP do, so HIP is not initialized.
    """
    gpus, from_amdsmi = _node_gpus()
    uuids = [uuid for uuid, _ in gpus]
    visible = _rocr_visible(uuids) if from_amdsmi else list(range(len(gpus)))
    return [gpus[i][1] for i in _hip_visible(visible, uuids)]


def get_partition_info(device: int | None = None) -> PartitionInfo:
    """Compute partition and CU counts of a logical GPU, from amdsmi.

    device is a HIP ordinal and defaults to the current device (see
    get_cu_num). Without amdsmi, rocminfo gives the CU count and the
    partition fields are None. CU_NUM overrides cu_num.
    """
    if device is None:
        device = _current_device()
    gpus = _visible_gpus()
    if not 0 <= device < len(gpus):
        raise RuntimeError(
            f"GPU {device} is not visible: {len(gpus)} of "
            f"{len(_node_gpus()[0])} GPUs are visible."
        )
    info = gpus[device]
    cu_override = int(os.getenv("CU_NUM", "0"))
    return info._replace(cu_num=cu_override) if cu_override else info


@torch_compile_guard()
def get_cu_num_custom_op(device: int | None = None) -> int:
    cu_num = int(os.getenv("CU_NUM", "0"))
    if cu_num == 0:
        if device is None:
            device = _current_device()
        if IS_WINDOWS:
            import torch

            props = torch.cuda.get_device_properties(device)
            gfx = props.gcnArchName.split(":", 1)[0]
            return torch_processor_count_to_cu(gfx, props.multi_processor_count)
        cu_num = get_partition_info(device).cu_num
    return cu_num


@functools.cache
def _cu_num(device: int) -> int:
    return get_cu_num_custom_op(device)


@functools.lru_cache(maxsize=1)
@assume_constant_result
def _shared_cu_num() -> int | None:
    """The CU count of every visible GPU, or None if they differ."""
    if IS_WINDOWS:
        return None
    cu_override = int(os.getenv("CU_NUM", "0"))
    cu_nums = {cu_override or info.cu_num for info in _visible_gpus()}
    return cu_nums.pop() if len(cu_nums) == 1 else None


def get_cu_num() -> int:
    """CU count of the current device, cached per device.

    When all visible GPUs have the same CU count, that count is returned
    without a device query. Otherwise the current device is the current
    device of torch once torch has initialized CUDA, and device 0 before
    that. HIP is not initialized here. A process that calls this before it
    selects its device gets the CU count of device 0.
    """
    cu_num = _shared_cu_num()
    if cu_num is None:
        cu_num = _cu_num(_current_device())
    return cu_num


def _clear_cu_num_cache() -> None:
    _shared_cu_num.cache_clear()
    _cu_num.cache_clear()
    _tuned_cu_nums_of_device.cache_clear()
    _shared_tuned_cu_nums.cache_clear()


get_cu_num.cache_clear = _clear_cu_num_cache


@functools.cache
def _tuned_cu_nums_of_device(device: int) -> tuple[int, ...]:
    """get_tuned_cu_nums() of one HIP device; empty without partition info."""
    try:
        info = get_partition_info(device)
    except Exception as e:  # noqa: BLE001  no partition info: exact rows only
        logger.debug("no partition info for GPU %d (%s)", device, e)
        return ()
    if (info.num_partitions or 0) > 1 and info.full_chip_cu_num not in (
        None,
        info.cu_num,
    ):
        return (info.cu_num, info.full_chip_cu_num)
    return (info.cu_num,)


@functools.lru_cache(maxsize=1)
@assume_constant_result
def _shared_tuned_cu_nums() -> tuple[int, ...] | None:
    """The tuned CU counts of every visible GPU, or None if they differ."""
    try:
        num_gpus = len(_visible_gpus())
    except Exception:  # noqa: BLE001  no GPU info: the current device decides
        return None
    cu_nums = {_tuned_cu_nums_of_device(device) for device in range(num_gpus)}
    return cu_nums.pop() if len(cu_nums) == 1 else None


def get_tuned_cu_nums(
    cu_num: int | None = None, device: int | None = None
) -> tuple[int, ...]:
    """CU counts whose tuned rows serve a lookup for cu_num, in lookup order.

    A compute partition (DPX, QPX, CPX) has an equal share of the CUs of its
    chip. Its lookups use its own CU count, then the CU count of the full
    chip, which has the same gfx. Other GPUs, such as MI308X or MI300A, use
    only their own CU count. A cu_num other than the CU count of the device
    also uses only itself.

    device is a HIP ordinal and defaults to the current device, as for
    get_cu_num(). When all visible GPUs have the same CU counts, a lookup
    does not query the device.

    The CU counts are the inner loop of a lookup. For each shape that a table
    tries, in its own order (exact shape, then padded or nearest shapes), the
    lookup tries these CU counts in turn. Thus an exact-shape row of the full
    chip comes before a padded-shape row of the partition. Tables whose rows
    set CU or CTA counts of cross-GPU kernels (MegaMoE, comm_fused_moe) keep
    the exact CU count.
    """
    if device is not None:
        cu_nums = _tuned_cu_nums_of_device(int(device))
    else:
        cu_nums = _shared_tuned_cu_nums()
        if cu_nums is None:
            cu_nums = _tuned_cu_nums_of_device(_current_device())
    if cu_num is None:
        return cu_nums or (get_cu_num(),)
    if cu_nums and int(cu_num) == cu_nums[0]:
        return cu_nums
    return (int(cu_num),)


def find_tuned_cu_row(
    tuned, key, gfx: str, cu_num: int, source: str, device: int | None = None
):
    """tuned[key(cu)] for the first CU count of get_tuned_cu_nums() that has one.

    For tables with one shape key. source names the table in the log line of
    a fallback. None when no CU count has a row.
    """
    for tuned_cu in get_tuned_cu_nums(cu_num, device):
        row = tuned.get(key(tuned_cu))
        if row is not None:
            if tuned_cu != cu_num:
                log_tuned_cu_fallback(source, gfx, cu_num, tuned_cu)
            return row
    return None


@functools.cache
def log_tuned_cu_fallback(source: str, gfx: str, cu_num: int, tuned_cu: int):
    """Log once that source serves a gfx partition of cu_num CUs with tuned_cu rows."""
    logger.info(
        f"{source}: no row for some shapes on the {cu_num}-CU {gfx} partition; "
        f"using the rows tuned for the full chip ({tuned_cu} CUs)"
    )


_CDNA4_CUS_PER_XCD = 32
_CDNA4_MAX_XCDS = 8


def get_cdna4_num_xcds(cu_num: int | None = None) -> int:
    """Return the number of CDNA4 XCDs represented by ``cu_num``.

    gfx950 exposes 32 CUs per XCD. ``cu_num`` can come from a live device, the
    ``CU_NUM`` override, or offline AOT metadata, so reject partial-XCD and
    out-of-range values here rather than silently truncating them.
    """
    cu_num = int(get_cu_num() if cu_num is None else cu_num)
    if cu_num <= 0 or cu_num % _CDNA4_CUS_PER_XCD != 0:
        raise ValueError(
            "CDNA4 CU count must be a positive multiple of "
            f"{_CDNA4_CUS_PER_XCD}, got {cu_num}"
        )
    num_xcds = cu_num // _CDNA4_CUS_PER_XCD
    if num_xcds > _CDNA4_MAX_XCDS:
        raise ValueError(
            f"CDNA4 XCD count must be no larger than {_CDNA4_MAX_XCDS}, "
            f"got {num_xcds} from {cu_num} CUs"
        )
    return num_xcds


@functools.cache
def get_cu_num_device(device_index: int) -> int:
    """CU count of one GPU, which `CU_NUM` may lower but never raise past the card.

    Unlike `get_cu_num()`, it does not assert when the node's GPUs differ, e.g. an
    iGPU beside the card.
    """
    import torch

    props = torch.cuda.get_device_properties(device_index)
    physical = torch_processor_count_to_cu(
        props.gcnArchName, props.multi_processor_count
    )
    cu_num = int(os.getenv("CU_NUM", "0"))
    return min(cu_num, physical) if cu_num > 0 else physical


def get_build_targets() -> list[tuple[str, int]]:
    """Return (gfx, cu_num) pairs to compile kernels for.

    Used by gen_instances.py in all CK GEMM modules to filter the tuning CSV
    to exactly the right set of kernels for the target GPU(s).

    Priority:
      1. GPU_ARCHS set to an explicit non-empty target list -> delegate to
         get_build_targets_env() (no GPU needed), which pairs each arch with
         every CU count in GFX_CU_NUMS_MAP unless CU_NUM is set.
      2. GPU_ARCHS unset, empty/whitespace, or "native" -> call get_gfx()
         (GPU_ARCHS-aware; falls back to rocminfo/hipinfo when GPU_ARCHS is
         unset) and get_tuned_cu_nums(), which correctly reflect partition
         mode and binned variants. A compute partition also gets the full
         chip, whose rows its lookups fall back to.
      3. Neither -> raise RuntimeError with a clear message.
    """
    gpu_archs = os.getenv("GPU_ARCHS")
    gpu_archs_normalized = gpu_archs.strip() if gpu_archs is not None else ""
    if gpu_archs_normalized and gpu_archs_normalized.lower() != "native":
        return get_build_targets_env()

    try:
        # get_gfx() is intentional here -- this is a build-time path; get_gfx_runtime()
        # would fail in CI environments without a live GPU.
        gfx = get_gfx()
        return [(gfx, cu_num) for cu_num in get_tuned_cu_nums(get_cu_num())]
    except Exception as e:
        raise RuntimeError(
            "No GPU detected and GPU_ARCHS is not set to an explicit target. "
            "Set GPU_ARCHS=gfx942 (or similar) to build without a GPU."
        ) from e


def build_tune_dict(
    tune_df, default_dict, kernels_list, libtype=None, kernels_by_name=None
):
    """Filter tune_df to rows matching the current build targets and return a
    (gfx, cu_num, M, N, K)-keyed dispatch dict, starting from a copy of default_dict.

    Replaces the duplicated get_tune_dict filtering loop in each gen_instances.py.
    Modules keep their own default_dict and kernels_list; only the CSV filtering
    and key construction are shared here.

    Args:
        tune_df:          pandas DataFrame already loaded from the tuning CSV.
        default_dict:     module-level fallback dict (negative-int keys) to start from.
        kernels_list:     module-level dict mapping kernelId -> kernelInstance.
        libtype:          Optional string to filter the "libtype" column (e.g. "ck").
                          Required for CSVs that mix multiple library types (e.g.
                          a8w8_bpreshuffle_tuned_gemm.csv mixes "ck" and "cktile").
                          If None, no libtype filtering is applied.
        kernels_by_name:  Optional dict mapping kernelName string -> kernelInstance.
                          When provided and the CSV has a "kernelName" column, kernel
                          lookup uses the name instead of kernelId. Falls back to
                          kernelId if the kernelName column is absent from the CSV.

    Strict on stale tuned-CSV rows: any row whose kernelName (or kernelId, in the
    fallback path) is not present in the registry will raise RuntimeError listing
    every offending row. A row that codegen silently drops would otherwise compile
    into a .so guaranteed to TORCH_CHECK(false, ...) at runtime for that shape.

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (gfx, cu_num, M, N, K) 5-tuples (from the filtered CSV rows).
    """
    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    use_name = kernels_by_name is not None and "kernelName" in tune_df.columns
    if kernels_by_name is not None and not use_name:
        logger.warning(
            "kernels_by_name provided but CSV has no kernelName column, falling back to kernelId."
        )
    bad_rows: list[str] = []
    for _, row in filtered.iterrows():
        key = (
            str(row["gfx"]),
            int(row["cu_num"]),
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
        )
        if use_name:
            kname = str(row["kernelName"])
            kernel = kernels_by_name.get(kname)
            if kernel is not None:
                tune_dict[key] = kernel
            else:
                bad_rows.append(
                    f"  kernelName={kname!r} not in kernels_by_name "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]})"
                )
        else:
            kid = int(row["kernelId"])
            kernel = kernels_list.get(kid)
            if kernel is not None:
                tune_dict[key] = kernel
            else:
                bad_rows.append(
                    f"  kernelId={kid} not in kernels_list "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]}, "
                    f"kernels_list size={len(kernels_list)})"
                )
    if bad_rows:
        raise RuntimeError(
            "build_tune_dict: tuned CSV references kernels not in the build registry. "
            "Either re-tune the CSV against the current kernel list or restore the "
            "missing kernel definition; the build refuses to produce a .so that would "
            "TORCH_CHECK(false, ...) at runtime for these shapes:\n"
            + "\n".join(bad_rows)
        )
    return tune_dict


def build_tune_dict_batched(tune_df, default_dict, kernels_list, libtype=None):
    """Like build_tune_dict, but for batched GEMM modules whose dispatch key
    includes the batch dimension B.

    Builds a (gfx, cu_num, B, M, N, K) 6-tuple keyed dict suitable for use with
    BatchedGemmDispatchMap in the C++ dispatch layer.

    Args:
        tune_df:      pandas DataFrame loaded from the batched tuning CSV.
        default_dict: module-level fallback dict (negative-int keys) to start from.
        kernels_list: module-level dict mapping kernelId -> kernelInstance.
        libtype:      Optional string to filter the "libtype" column (same semantics
                      as build_tune_dict).

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (gfx, cu_num, B, M, N, K) 6-tuples (from the filtered CSV rows).
    """
    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    bad_rows: list[str] = []
    for _, row in filtered.iterrows():
        key = (
            str(row["gfx"]),
            int(row["cu_num"]),
            int(row["B"]),
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
        )
        kid = int(row["kernelId"])
        kernel = kernels_list.get(kid)
        if kernel is not None:
            tune_dict[key] = kernel
        else:
            bad_rows.append(
                f"  kernelId={kid} not in kernels_list "
                f"(gfx={key[0]}, cu_num={key[1]}, B={key[2]}, M={key[3]}, N={key[4]}, K={key[5]}, "
                f"kernels_list size={len(kernels_list)})"
            )
    if bad_rows:
        raise RuntimeError(
            "build_tune_dict_batched: tuned CSV references kernels not in the build "
            "registry. Either re-tune the CSV against the current kernel list or "
            "restore the missing kernel definition; the build refuses to produce a "
            ".so that would TORCH_CHECK(false, ...) at runtime for these shapes:\n"
            + "\n".join(bad_rows)
        )
    return tune_dict


def write_name_keyed_lookup_header(
    output_path, kernels_dict, lookup_head, lookup_template, lookup_end
):
    """Write a name-keyed C++ GEMM dispatch lookup header from a kernels_dict.

    Sister of write_lookup_header(), but emits {"<kernel_name>", &kernel<...>}
    entries instead of (gfx,cu_num,M,N,K) tuple keys.  Used by the blockscale
    GEMM modules whose runtime dispatch is now driven by Python-resolved
    kernel name strings (read from the tuned CSV) rather than a build-time
    tuple-keyed lookup.  The kernels_dict may contain duplicate entries for
    the same kernel (multiple shapes mapping to the same kernel.name); we
    dedupe by name so each kernel is registered exactly once.

    Skips negative-int default_dict keys (heuristic fallbacks the dispatch
    layer references directly by symbol).

    Args:
        output_path:     Full path of the .h file to write.
        kernels_dict:    Dict returned by build_tune_dict.
        lookup_head:     String written before the loop (defines the macro header).
        lookup_template: String with {kernel_name} placeholder (used twice:
                          once for the C++ string key, once for the symbol).
        lookup_end:      String written after the loop (closes the macro / #endif).
    """
    seen = set()
    with open(output_path, "w") as f:
        f.write(lookup_head)
        for key, k in kernels_dict.items():
            if isinstance(key, int) and key < 0:
                # default_dict heuristic-fallback entries; the dispatch layer
                # references the heuristic kernel by symbol, not via the table.
                continue
            if k.name in seen:
                continue
            seen.add(k.name)
            f.write(lookup_template.format(kernel_name=k.name))
        f.write(lookup_end)


def write_lookup_header(
    output_path,
    kernels_dict,
    lookup_head,
    lookup_template,
    lookup_end,
    istune=False,
    extra_format_args=None,
):
    """Write a C++ GEMM dispatch lookup header from a kernels_dict.

    Replaces the duplicated gen_lookup_dict loop in each gen_instances.py codegen
    class.  Each module still defines its own lookup_head / lookup_template /
    lookup_end strings (they embed the module-specific GENERATE_LOOKUP_TABLE macro
    type parameters), but the iteration and key-formatting logic is shared here.

    Key layout in kernels_dict:
      - Negative ints          (default_dict entries) -> skipped in non-tune mode.
      - (gfx,cu_num,M,N,K) 5-tuples (tuned entries)  -> written as {"gfx",cu_num,M,N,K} C++ key.
      - (gfx,cu_num,B,M,N,K) 6-tuples (batched)      -> written as {"gfx",cu_num,B,M,N,K} C++ key.
      - Non-negative ints (tune mode only)            -> written as plain integer kernel ID.

    Args:
        output_path:     Full path of the .h file to write.
        kernels_dict:    Dict returned by build_tune_dict (or get_tune_dict).
        lookup_head:     String written before the loop (defines the macro header).
        lookup_template: String with {MNK} and {kernel_name} placeholders.
        lookup_end:      String written after the loop (closes the macro / #endif).
        istune:          True when generating the tune-mode lookup (int kernelId keys).
        extra_format_args: Optional callable returning additional format arguments
                           for a kernel instance.
    """

    def format_entry(key_value, kernel):
        format_args = {"MNK": key_value, "kernel_name": kernel.name}
        if extra_format_args is not None:
            format_args.update(extra_format_args(kernel))
        return lookup_template.format(**format_args)

    with open(output_path, "w") as f:
        f.write(lookup_head)
        for key, k in kernels_dict.items():
            if not istune and (isinstance(key, tuple) and isinstance(key[0], str)):
                # 5-tuple key: (gfx, cu_num, M, N, K)
                # 6-tuple key: (gfx, cu_num, B, M, N, K)
                # key[0] is the gfx arch string; the remaining elements are ints.
                cpp_key = (
                    '{"' + key[0] + '", ' + ", ".join(str(x) for x in key[1:]) + "}"
                )
                f.write(format_entry(cpp_key, k))
            elif istune and isinstance(key, int) and key >= 0:
                f.write(format_entry(key, k))
        f.write(lookup_end)


def _get_pci_chip_id(device_id=None):
    import ctypes

    if device_id is None:
        device_id = get_current_hip_device()
        if device_id is None:
            raise RuntimeError(
                "hipGetDevice failed, so there is no current device to read "
                "PciChipId from."
            )

    # ROCm 7.1 inserted MaxAvailableVgprsPerThread ahead of PciChipId, shifting it.
    version = get_hip_runtime_version()
    if version is None:
        raise RuntimeError(
            "Could not read the HIP runtime version, so the PciChipId attribute "
            f"of device {device_id} cannot be located."
        )
    if version[:2] < (7, 1):
        hipDeviceAttributePciChipId = 10019
    else:
        hipDeviceAttributePciChipId = 10020

    libhip = load_hip_runtime()
    chip_id = ctypes.c_int(0)
    err = libhip.hipDeviceGetAttribute(
        ctypes.byref(chip_id),
        hipDeviceAttributePciChipId,
        device_id,
    )
    if err != 0:
        raise RuntimeError(f"hipDeviceGetAttribute(PciChipId) failed with error {err}")
    return chip_id.value


MI308_CHIP_IDS = {0x74A2, 0x74A8, 0x74B6, 0x74BC}
# MI300A is the gfx942 APU: 228 CUs and 6 XCCs, against MI300X's 304 and 8.
MI300A_CHIP_IDS = {0x74A0}


def get_device_name():
    """Name of the device this thread is bound to.

    Needs a live GPU: the arch, PCI id and XCD count all come from that device.
    """
    device_id = get_current_hip_device()
    if device_id is None:
        raise RuntimeError(
            "hipGetDevice failed, so there is no current device to name."
        )
    return _get_device_name(device_id)


def _get_device_name(device_id: int) -> str:
    import torch

    gfx = torch.cuda.get_device_properties(device_id).gcnArchName
    gfx = gfx.split(":", 1)[0].lower()

    if gfx == "gfx942":
        chip_id = _get_pci_chip_id(device_id)
        if chip_id in MI308_CHIP_IDS:
            return "MI308"
        if chip_id in MI300A_CHIP_IDS:
            return "MI300A"
        return "MI300"
    elif gfx == "gfx950":
        num_xcds = _num_xcds_for_device(device_id)
        if num_xcds is None:
            raise RuntimeError(
                "HIP runtime predates hipDeviceAttributeNumberOfXccs, so the XCD "
                f"count of device {device_id} cannot be determined."
            )
        # MI350P is a half-size gfx950: 4 XCDs against 8 on MI350X and MI355X.
        if num_xcds == 4:
            return "MI350P"
        return "MI350"
    elif gfx in ("gfx1250", "gfx1250-strict"):
        return "MI400"
    else:
        raise RuntimeError("Unsupported gfx")


def _query_num_xccs(device_id: int) -> int:
    """Ask HIP how many XCCs the given device has."""
    import ctypes

    hipDeviceAttributeNumberOfXccs = 10018

    libhip = load_hip_runtime()
    val = ctypes.c_int(0)
    err = libhip.hipDeviceGetAttribute(
        ctypes.byref(val),
        hipDeviceAttributeNumberOfXccs,
        device_id,
    )
    if err != 0:
        raise RuntimeError(
            f"hipDeviceGetAttribute(NumberOfXccs) failed with error {err} "
            f"for device {device_id}"
        )

    # A shifted ordinal resolves to an unrelated attribute, whose value lands well
    # outside this range. Loose because MI300A has 6 dies and partitions vary.
    n = val.value
    if not 1 <= n <= 64:
        raise RuntimeError(
            f"hipDeviceGetAttribute(NumberOfXccs) returned {n} for device "
            f"{device_id}, which is not a usable XCD count"
        )
    return n


@functools.cache
def _num_xcds_for_device(device_id: int) -> int | None:
    """XCD count of a device, or None for a HIP runtime that lacks the attribute."""
    version = get_hip_runtime_version()
    if version is None:
        raise RuntimeError(
            f"Could not read the HIP runtime version for device {device_id}, "
            "so the XCD count cannot be determined."
        )

    # Below ROCm 7.0 the ordinal names a different attribute and returns a
    # plausible integer, so this gate has to precede the query.
    if version[:2] < (7, 0):
        logger.warning(
            "HIP %d.%d predates hipDeviceAttributeNumberOfXccs.", *version[:2]
        )
        return None

    return _query_num_xccs(device_id)


@functools.cache
def _warn_xcd_fallback(reason: str, default: int) -> None:
    logger.warning(
        "%s; assuming %d XCDs until a later query succeeds.", reason, default
    )


def get_num_xcds(default: int | None = 8) -> int:
    """XCD (accelerator die) count of the device this thread is bound to.

    Only the first successful lookup per device id is cached; a failed one is
    retried on the next call. On failure, the default is returned and the
    failure logged once; pass default=None to raise instead, as any caller whose
    correctness or identity depends on the count must. A HIP runtime that lacks
    the attribute (before 7.0) also returns the default, or raises without one.
    """
    try:
        device_id = get_current_hip_device()
        if device_id is None:
            raise RuntimeError(
                "hipGetDevice failed, so there is no current device to read the "
                "XCD count from."
            )
        xcds = _num_xcds_for_device(device_id)
    except (RuntimeError, OSError) as e:
        if default is None:
            raise
        _warn_xcd_fallback(str(e), default)
        return default
    if xcds is None:
        if default is None:
            raise RuntimeError(
                "HIP runtime predates hipDeviceAttributeNumberOfXccs, so the XCD "
                f"count of device {device_id} cannot be determined."
            )
        return default
    return xcds
