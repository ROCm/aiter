# SPDX-License-Identifier: MIT
"""Offline FlyDSL tuner for fused heterogeneous MoE (FHMoE).

The tuner measures the complete two-stage FHMoE call, including sorting,
stage-1 activation/quantization and stage-2 accumulation.  It updates only rows
matching the requested contract and preserves every other row in the CSV. Each
selected full-M result must also match an independent FP32 reference on sampled
token rows before its kernels can be written.

Candidate ranking remains a full-M comparison against the current kernel. A
full FP32 oracle for every token at large M would make tuning prohibitively
slow, so only the final winner receives the expensive independent check. The
tuner retains raw weights for experts reached by representative token rows,
computes the complete routed-plus-shared FP32 result for those rows, and compares
it with the corresponding rows from the winner's full-M output.

Example (HY4 MXFP8 routed / FP8 shared on gfx950):

    python op_tests/tuners/tune_fhmoe.py \
      -i aiter/configs/tuned_fhmoe.csv \
      -o aiter/configs/tuned_fhmoe.csv \
      --tokens 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import aiter
from aiter import dtypes
from aiter.fhmoe import _use_fhmoe_wrappers
from aiter.fhmoe_contract import (
    HY4_FHMOE_MAX_TOKENS,
    _is_hy4_mxfp8_fhmoe_contract,
)
from aiter.fused_moe import (
    _fused_moe_impl,
    cfg_2stages_by_file,
    get_2stage_cfgs,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_kernel_params,
    get_flydsl_stage1_kernels,
    get_flydsl_stage2_kernels,
    runtime_swiglu_limit,
)
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
)
from aiter.utility import fp4_utils
from aiter.utility.mx_types import MxDtypeInt


@dataclass(frozen=True)
class Contract:
    gfx: str = "gfx950"
    cu_num: int = 256
    model_dim: int = 6144
    inter_dim: int = 256
    routed_experts: int = 256
    routed_topk: int = 8
    shared_expert_id: int = 256
    act_type: str = "ActivationType.Silu"
    dtype: str = "torch.bfloat16"
    q_dtype_a: str = "torch.float8_e4m3fn"
    q_dtype_w: str = "torch.float8_e4m3fn"
    q_type: str = "QuantType.per_1x32"
    use_g1u1: int = 1
    doweight_stage1: int = 0
    hidden_pad: int = 0
    intermediate_pad: int = 0
    gate_mode: str = "GateMode.INTERLEAVE"
    ksplit: int = 0

    @property
    def experts(self) -> int:
        return self.routed_experts + 1

    @property
    def topk(self) -> int:
        return self.routed_topk + 1


@dataclass
class Tensors:
    hidden: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    shared_w1: torch.Tensor
    shared_w2: torch.Tensor
    shared_w1_scale: torch.Tensor
    shared_w2_scale: torch.Tensor
    output: torch.Tensor
    # Minimal raw state retained for the final sampled FP32 acceptance check.
    # Candidate timing still uses the full tensors above.
    sample_rows: tuple[int, ...]
    sample_expert_ids: tuple[int, ...]
    sample_w1: torch.Tensor
    sample_w2: torch.Tensor
    sample_w1_scale: torch.Tensor
    sample_w2_scale: torch.Tensor
    sample_shared_w1: torch.Tensor
    sample_shared_w2: torch.Tensor
    sample_shared_w1_scale: torch.Tensor
    sample_shared_w2_scale: torch.Tensor


CSV_FIELDS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "shared_expert_id",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
)
PROFILE_FIELDS = ("token", "runtime_token", "stage", "kernel", "ms", "error")
_PAIR_BEAM_WIDTH = 4


def _parse_tokens(value: str) -> list[int]:
    tokens = sorted({int(token) for token in value.split(",") if token.strip()})
    if not tokens or any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must be positive comma-separated ints")
    return tokens


def _parse_nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _parse_positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _validate_benchmark_token(benchmark_token: int | None, tokens: list[int]) -> None:
    if benchmark_token is None:
        oversized = [token for token in tokens if token > HY4_FHMOE_MAX_TOKENS]
        if oversized:
            raise ValueError(
                "HY4 tuning tokens must not exceed "
                f"{HY4_FHMOE_MAX_TOKENS}, got {oversized}"
            )
        return
    from aiter.fused_moe import get_padded_M

    if benchmark_token <= 0:
        raise ValueError("--benchmark-token must be positive")
    if benchmark_token > HY4_FHMOE_MAX_TOKENS:
        raise ValueError(f"--benchmark-token must not exceed {HY4_FHMOE_MAX_TOKENS}")
    if len(tokens) != 1:
        raise ValueError("--benchmark-token requires exactly one --tokens bucket")
    if get_padded_M(benchmark_token) != tokens[0]:
        raise ValueError(
            f"benchmark M={benchmark_token} resolves to bucket "
            f"{get_padded_M(benchmark_token)}, not {tokens[0]}"
        )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _write_rows(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    durable: bool = False,
) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = path.stat().st_mode if path.exists() else 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            os.fchmod(stream.fileno(), previous_mode & 0o777)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _paths_alias(first: Path, second: Path) -> bool:
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            pass
    return first.resolve(strict=False) == second.resolve(strict=False)


def _validate_output_paths(
    input_path: Path, output_path: Path, profile_path: Path | None
) -> None:
    if output_path.is_symlink():
        raise ValueError(f"FHMoE tuner output must not be a symlink: {output_path}")
    if profile_path is None:
        return
    if profile_path.is_symlink():
        raise ValueError(f"FHMoE tuner profile must not be a symlink: {profile_path}")
    if _paths_alias(profile_path, input_path) or _paths_alias(
        profile_path, output_path
    ):
        raise ValueError("FHMoE tuner profile must differ from input and output")


def _lock_directory() -> Path:
    directory = Path("/tmp") / f"aiter-fhmoe-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"unsafe FHMoE lock directory: {directory}")
    os.chmod(directory, 0o700)
    return directory


@contextmanager
def _output_lock(path: Path):
    lock_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    lock_path = _lock_directory() / f"{lock_id}.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f"unsafe FHMoE lock file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_profile_rows(
    profile_path: Path,
    rows: list[dict[str, str | float]],
) -> None:
    """Durably replace the measured token/runtime pair under a lock."""
    normalized = [
        {field: row.get(field, "") for field in PROFILE_FIELDS} for row in rows
    ]
    keys = {(str(row["token"]), str(row["runtime_token"])) for row in normalized}
    with _output_lock(profile_path):
        existing: list[dict[str, str | float]] = []
        if profile_path.exists():
            fields, existing_rows = _read_rows(profile_path)
            legacy_fields = ["token", "stage", "kernel", "ms", "error"]
            if fields == legacy_fields:
                for row in existing_rows:
                    row["runtime_token"] = row["token"]
            elif fields != list(PROFILE_FIELDS):
                raise ValueError(
                    f"cannot merge FHMoE profile rows into {profile_path}: "
                    "CSV columns changed"
                )
            existing = [
                row
                for row in existing_rows
                if (row["token"], row["runtime_token"]) not in keys
            ]
        _write_rows(
            profile_path,
            list(PROFILE_FIELDS),
            [*existing, *normalized],
            durable=True,
        )


def _write_tuned_output(
    input_path: Path,
    output_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    selected: dict[int, int],
    baseline_rows: dict[int, dict[str, str]],
    contract: Contract,
) -> None:
    """Compare-and-swap selected rows under a lock."""
    if output_path.is_symlink():
        raise ValueError(f"FHMoE tuner output must not be a symlink: {output_path}")
    with _output_lock(output_path):
        source = output_path if output_path.exists() else input_path
        latest_fields, latest_rows = _read_rows(source)
        if latest_fields != fields:
            raise ValueError(
                f"cannot merge FHMoE rows into {source}: CSV columns changed"
            )
        for token, tuned_index in selected.items():
            matches = [
                index
                for index, row in enumerate(latest_rows)
                if _matches(row, contract, token)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one current HY4 FHMoE row for M={token}, "
                    f"found {len(matches)}"
                )
            latest_index = matches[0]
            if latest_rows[latest_index] != baseline_rows[token]:
                raise RuntimeError(
                    f"FHMoE row M={token} changed during tuning; refusing stale update"
                )
            updated = dict(latest_rows[latest_index])
            updated["kernelName1"] = rows[tuned_index]["kernelName1"]
            updated["kernelName2"] = rows[tuned_index]["kernelName2"]
            latest_rows[latest_index] = updated
        _write_rows(output_path, fields, latest_rows, durable=True)


def _validate_existing_output(
    output_path: Path,
    fields: list[str],
    baseline_rows: dict[int, dict[str, str]],
    contract: Contract,
) -> None:
    if not output_path.exists():
        return
    output_fields, output_rows = _read_rows(output_path)
    if output_fields != fields:
        raise ValueError(f"existing output has different CSV columns: {output_path}")
    for token, baseline in baseline_rows.items():
        matches = [row for row in output_rows if _matches(row, contract, token)]
        if len(matches) != 1 or matches[0] != baseline:
            raise RuntimeError(
                f"existing output row M={token} differs from input; use the "
                "output as --input to resume, or remove it"
            )


def _matches(row: dict[str, str], contract: Contract, token: int) -> bool:
    return (
        row["gfx"] == contract.gfx
        and int(row["cu_num"]) == contract.cu_num
        and int(row["token"]) == token
        and int(row["model_dim"]) == contract.model_dim
        and int(row["inter_dim"]) == contract.inter_dim
        and int(row["expert"]) == contract.experts
        and int(row["topk"]) == contract.topk
        and int(row["shared_expert_id"]) == contract.shared_expert_id
        and row["act_type"] == contract.act_type
        and row["dtype"] == contract.dtype
        and row["q_dtype_a"] == contract.q_dtype_a
        and row["q_dtype_w"] == contract.q_dtype_w
        and row["q_type"] == contract.q_type
        and int(row["use_g1u1"]) == contract.use_g1u1
        and int(row["doweight_stage1"]) == contract.doweight_stage1
        and int(row["hidden_pad"]) == contract.hidden_pad
        and int(row["intermediate_pad"]) == contract.intermediate_pad
        and row["gate_mode"] == contract.gate_mode
        and int(row["ksplit"]) == contract.ksplit
    )


def _select_rows(
    rows: list[dict[str, str]], contract: Contract, tokens: list[int]
) -> dict[int, int]:
    selected: dict[int, int] = {}
    requested = set(tokens)
    for index, row in enumerate(rows):
        token = int(row["token"])
        if token not in requested or not _matches(row, contract, token):
            continue
        if token in selected:
            raise ValueError(
                f"multiple HY4 FHMoE rows match the complete contract for M={token}"
            )
        selected[token] = index
    return selected


def _effective_sort_block(name: str) -> int:
    params = get_flydsl_kernel_params(name)
    if params is None:
        return -1
    return int(params.get("sort_block_m", 0) or params["tile_m"])


def stage1_candidates(block_m: int, quick: bool = False) -> list[str]:
    names: set[str] = set()
    for name, params in get_flydsl_stage1_kernels("fp8", "fp8", "bf16").items():
        if (
            int(params["tile_m"]) != block_m
            or int(params["tile_k"]) != 256
            or 256 % int(params["tile_n"]) != 0
            or int(params.get("k_batch", 1)) != 1
            or params.get("gate_mode") != "interleave"
        ):
            continue
        candidate = f"{name}_fp8"
        if get_flydsl_kernel_params(candidate) is None:
            continue
        if quick and (
            int(params["tile_n"]) not in (64, 128) or int(params.get("k_wave", 1)) != 1
        ):
            continue
        names.add(candidate)
    return sorted(names)


def stage2_candidates(block_m: int, quick: bool = False) -> list[str]:
    names: set[str] = set()
    for name, params in get_flydsl_stage2_kernels("fp8", "fp8", "bf16").items():
        tile_m = int(params["tile_m"])
        if (
            block_m % tile_m != 0
            or 256 % int(params["tile_k"]) != 0
            or int(params["tile_n"]) != 128
        ):
            continue
        candidate = name if tile_m == block_m else f"{name}_sbm{block_m}"
        parsed = get_flydsl_kernel_params(candidate)
        if (
            parsed is None
            or _effective_sort_block(candidate) != block_m
            or parsed.get("persist") is True
            or int(parsed.get("b_nt", 0)) != 0
        ):
            continue
        if quick and int(parsed.get("xcd_swizzle", 0)) != 0:
            continue
        names.add(candidate)
    return sorted(names)


def _random_fp8(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        .mul_(0.02)
        .to(torch.float8_e4m3fn)
    )


def _varying_e8m0_scales(
    shape: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """Generate deterministic 0.25/0.5/1/2 scales across E, N, and K groups."""
    experts, rows, groups = shape
    expert_index = torch.arange(experts, device=device).remainder_(4).to(torch.uint8)
    row_index = torch.arange(rows, device=device).remainder_(4).to(torch.uint8)
    group_index = torch.arange(groups, device=device).remainder_(4).to(torch.uint8)
    pattern = (
        expert_index[:, None, None]
        + row_index[None, :, None]
        + group_index[None, None, :]
    ).remainder_(4)
    return pattern.add_(0x7D)


def _sample_rows(token: int) -> tuple[int, ...]:
    """Choose cheap representative rows without weakening the full-M timing run.

    The first 32 rows collectively route through all 256 HY4 experts. The
    midpoint and final row additionally cover distant sorting positions.
    """
    candidates = (*range(min(token, 32)), token // 2, token - 1)
    return tuple(sorted({row for row in candidates if 0 <= row < token}))


def _sample_expert_ids(contract: Contract, rows: tuple[int, ...]) -> tuple[int, ...]:
    """Return only the routed weights needed by the sampled token rows."""
    return tuple(
        sorted(
            {
                (row * contract.routed_topk + slot) % contract.routed_experts
                for row in rows
                for slot in range(contract.routed_topk)
            }
        )
    )


def _make_tensors(
    contract: Contract, token: int, seed: int, device: torch.device
) -> Tensors:
    generator = torch.Generator(device=device).manual_seed(seed)
    e, h, i = contract.experts, contract.model_dim, contract.inter_dim

    raw_w1 = _random_fp8((e, 2 * i, h), generator, device)
    raw_w2 = _random_fp8((e, h, i), generator, device)
    raw_s1 = _varying_e8m0_scales((e, 2 * i, h // 32), device)
    raw_s2 = _varying_e8m0_scales((e, h, i // 32), device)
    sample_rows = _sample_rows(token)
    sample_expert_ids = _sample_expert_ids(contract, sample_rows)
    sample_indices = torch.tensor(sample_expert_ids, dtype=torch.long, device=device)
    sample_w1 = raw_w1.index_select(0, sample_indices).clone()
    sample_w2 = raw_w2.index_select(0, sample_indices).clone()
    sample_w1_scale = raw_s1.index_select(0, sample_indices).clone()
    sample_w2_scale = raw_s2.index_select(0, sample_indices).clone()
    w1 = shuffle_weight_a16w4(raw_w1, 16, True).contiguous()
    w2 = shuffle_weight_a16w4(raw_w2, 16, False).contiguous()
    w1_scale = shuffle_scale_a16w4(
        raw_s1.view(-1, raw_s1.shape[-1]).view(dtypes.fp8_e8m0), e, True
    ).contiguous()
    w2_scale = shuffle_scale(
        raw_s2.view(-1, raw_s2.shape[-1]).view(dtypes.fp8_e8m0)
    ).contiguous()
    del raw_w1, raw_w2, raw_s1, raw_s2

    shared_w1 = _random_fp8((1, 2 * i, h), generator, device)
    shared_w2 = _random_fp8((1, h, i), generator, device)
    shared_s1 = _varying_e8m0_scales((1, 2 * i, h // 32), device)
    shared_s2 = _varying_e8m0_scales((1, h, i // 32), device)
    sample_shared_w1 = shared_w1.clone()
    sample_shared_w2 = shared_w2.clone()
    sample_shared_w1_scale = shared_s1.clone()
    sample_shared_w2_scale = shared_s2.clone()
    shared_w1 = shuffle_weight_a16w4(shared_w1, 16, True).contiguous()
    shared_w2 = shuffle_weight_a16w4(shared_w2, 16, False).contiguous()
    shared_w1_scale = shuffle_scale_a16w4(
        shared_s1.view(-1, shared_s1.shape[-1]).view(dtypes.fp8_e8m0), 1, True
    ).contiguous()
    shared_w2_scale = shuffle_scale(
        shared_s2.view(-1, shared_s2.shape[-1]).view(dtypes.fp8_e8m0)
    ).contiguous()

    hidden = torch.randn(
        (token, h), dtype=torch.bfloat16, device=device, generator=generator
    )
    row = torch.arange(token, device=device, dtype=torch.int32)[:, None]
    slot = torch.arange(contract.routed_topk, device=device, dtype=torch.int32)[None, :]
    routed_ids = (row * contract.routed_topk + slot) % contract.routed_experts
    routed_weights = torch.arange(
        1,
        contract.routed_topk + 1,
        dtype=torch.float32,
        device=device,
    ).repeat(token, 1)
    routed_weights.mul_(2.5 / routed_weights.sum(dim=1, keepdim=True))
    shared_ids = torch.full(
        (token, 1), contract.shared_expert_id, dtype=torch.int32, device=device
    )
    shared_weights = torch.ones((token, 1), dtype=torch.float32, device=device)
    return Tensors(
        hidden=hidden,
        topk_weights=torch.cat((routed_weights, shared_weights), dim=1),
        topk_ids=torch.cat((routed_ids, shared_ids), dim=1),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w1_scale=shared_w1_scale,
        shared_w2_scale=shared_w2_scale,
        output=torch.empty((token, h), dtype=torch.bfloat16, device=device),
        sample_rows=sample_rows,
        sample_expert_ids=sample_expert_ids,
        sample_w1=sample_w1,
        sample_w2=sample_w2,
        sample_w1_scale=sample_w1_scale,
        sample_w2_scale=sample_w2_scale,
        sample_shared_w1=sample_shared_w1,
        sample_shared_w2=sample_shared_w2,
        sample_shared_w1_scale=sample_shared_w1_scale,
        sample_shared_w2_scale=sample_shared_w2_scale,
    )


def _candidate_row(base: dict[str, str], stage1: str, stage2: str) -> dict[str, str]:
    row = dict(base)
    row["kernelName1"] = stage1
    row["kernelName2"] = stage2
    return row


def _write_candidate(path: Path, row: dict[str, str]) -> None:
    _write_rows(path, list(CSV_FIELDS), [row])
    get_2stage_cfgs.cache_clear()
    cfg_2stages_by_file.clear()


def _runner(
    tensors: Tensors, contract: Contract, config_file: Path, swiglu_limit: float
):
    clamp_shared = not _is_hy4_mxfp8_fhmoe_contract(
        model_dim=contract.model_dim,
        inter_dim=contract.inter_dim,
        experts=contract.experts,
        topk=contract.topk,
        routed_mxfp8=contract.q_dtype_w == "torch.float8_e4m3fn",
        hidden_pad=contract.hidden_pad,
        intermediate_pad=contract.intermediate_pad,
        gate_interleaved=contract.gate_mode == "GateMode.INTERLEAVE",
        doweight_stage1=bool(contract.doweight_stage1),
        shared_expert_id=contract.shared_expert_id,
    )

    def run():
        return _fused_moe_impl(
            hidden_states=tensors.hidden,
            w1=tensors.w1,
            w2=tensors.w2,
            topk_weight=tensors.topk_weights,
            topk_ids=tensors.topk_ids,
            activation=aiter.ActivationType.Silu.value,
            quant_type=aiter.QuantType.per_1x32.value,
            w1_scale=tensors.w1_scale,
            w2_scale=tensors.w2_scale,
            dtype=torch.bfloat16,
            swiglu_limit=swiglu_limit,
            gate_mode=GateMode.INTERLEAVE.value,
            output=tensors.output,
            _q_dtype_a=dtypes.fp8,
            _metadata_transform=_use_fhmoe_wrappers,
            _metadata_config_file=str(config_file),
            _stage1_extra_args={
                "shared_w1": tensors.shared_w1,
                "shared_w1_scale": tensors.shared_w1_scale,
                "shared_expert_id": contract.shared_expert_id,
                "swiglu_limit": swiglu_limit,
                "clamp_shared": clamp_shared,
            },
            _stage2_extra_args={
                "shared_w2": tensors.shared_w2,
                "shared_w2_scale": tensors.shared_w2_scale,
                "shared_expert_id": contract.shared_expert_id,
            },
        )

    return run


def _time_candidate(
    run,
    warmup_ms: float,
    rep_ms: float,
    stability_tolerance: float = 8e-3,
) -> tuple[float, torch.Tensor]:
    result = run()
    torch.accelerator.synchronize()
    if not torch.isfinite(result).all():
        return float("inf"), result
    reference = result.detach().clone()
    timing = float(triton.testing.do_bench(run, warmup=warmup_ms, rep=rep_ms))
    for _ in range(3):
        verification = run()
        torch.accelerator.synchronize()
        if not torch.isfinite(verification).all():
            return float("inf"), verification
        if _relative_l2(verification, reference) > stability_tolerance:
            return float("inf"), verification
    return timing, reference


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(
        torch.linalg.vector_norm(actual.float() - expected.float()) / denominator
    )


def _fp8_group_quant_dequant(
    x: torch.Tensor, *, fused_intermediate: bool = False
) -> torch.Tensor:
    shape = x.shape
    blocks = x.float().view(-1, 32)
    amax = blocks.abs().amax(dim=1)
    scale = (
        fp4_utils.f32_to_fused_moe_mxfp8_scale(amax)
        if fused_intermediate
        else fp4_utils.f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale).view(-1, 1)
    quant = (blocks / scale_f32).to(torch.float8_e4m3fn)
    return (quant.float() * scale_f32).view(shape)


def _dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale_f32 = fp4_utils.e8m0_to_f32(scale.view(dtypes.fp8_e8m0))
    return weight.float() * scale_f32.repeat_interleave(32, dim=-1)


def _routed_silu(
    gate: torch.Tensor, up: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    activation_limit = runtime_swiglu_limit(swiglu_limit, "silu")
    gate = gate.clamp(max=activation_limit)
    up = up.clamp(min=-activation_limit, max=activation_limit)
    return F.silu(gate) * up


@torch.no_grad()
def _sampled_fp32_reference(
    tensors: Tensors,
    contract: Contract,
    swiglu_limit: float,
) -> torch.Tensor:
    """Compute an independent routed-plus-shared FP32 oracle for sampled rows.

    This reproduces MXFP8 activation/weight dequantization and the model's BF16
    rounding points, but performs the GEMMs and SiLU in FP32. Work scales with
    the number of sampled rows and their experts rather than full M.
    """
    rows = torch.tensor(
        tensors.sample_rows, dtype=torch.long, device=tensors.hidden.device
    )
    hidden = _fp8_group_quant_dequant(tensors.hidden.index_select(0, rows).float())
    routed_ids = tensors.topk_ids.index_select(0, rows)[:, : contract.routed_topk]
    routed_weights = tensors.topk_weights.index_select(0, rows)[
        :, : contract.routed_topk
    ]
    expanded_hidden = hidden[:, None, :].expand(-1, contract.routed_topk, -1)
    slot_output = torch.zeros(
        (*routed_ids.shape, contract.model_dim),
        dtype=torch.float32,
        device=hidden.device,
    )
    expert_storage = {
        expert_id: index for index, expert_id in enumerate(tensors.sample_expert_ids)
    }

    for expert_id in torch.unique(routed_ids).tolist():
        mask = routed_ids == expert_id
        storage = expert_storage[int(expert_id)]
        w1 = _dequant_fp8_weight(
            tensors.sample_w1[storage], tensors.sample_w1_scale[storage]
        )
        gate, up = F.linear(expanded_hidden[mask], w1).chunk(2, dim=-1)
        intermediate = _routed_silu(gate, up, swiglu_limit)
        intermediate = _fp8_group_quant_dequant(intermediate, fused_intermediate=True)
        w2 = _dequant_fp8_weight(
            tensors.sample_w2[storage], tensors.sample_w2_scale[storage]
        )
        slot_output[mask] = F.linear(intermediate, w2)

    routed_output = (slot_output * routed_weights[..., None]).sum(dim=1)
    shared_w1 = _dequant_fp8_weight(
        tensors.sample_shared_w1[0], tensors.sample_shared_w1_scale[0]
    )
    shared_gate_up = F.linear(hidden, shared_w1).to(torch.bfloat16).float()
    shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
    shared_intermediate = F.silu(shared_gate) * shared_up
    shared_intermediate = _fp8_group_quant_dequant(
        shared_intermediate.to(torch.bfloat16).float(),
        fused_intermediate=True,
    )
    shared_w2 = _dequant_fp8_weight(
        tensors.sample_shared_w2[0], tensors.sample_shared_w2_scale[0]
    )
    shared_output = F.linear(shared_intermediate, shared_w2)
    shared_weight = tensors.topk_weights.index_select(0, rows)[:, contract.routed_topk]
    return routed_output + shared_output * shared_weight[:, None]


def tune_row(
    row: dict[str, str],
    tensors: Tensors,
    contract: Contract,
    config_file: Path,
    *,
    quick: bool,
    warmup_ms: float,
    rep_ms: float,
    tolerance: float,
    swiglu_limit: float,
    min_improvement_pct: float,
    stability_tolerance: float = 8e-3,
    baseline_only: bool = False,
) -> tuple[dict[str, str], list[dict[str, str | float]]]:
    block_m = int(row["block_m"])
    current_s1, current_s2 = row["kernelName1"], row["kernelName2"]
    profile: list[dict[str, str | float]] = []
    current_s1_params = get_flydsl_kernel_params(current_s1)
    current_s2_params = get_flydsl_kernel_params(current_s2)
    if current_s1_params is not None and (
        int(current_s1_params["tile_m"]) != block_m
        or int(current_s1_params.get("k_batch", 1)) != 1
        or current_s1_params.get("gate_mode") != "interleave"
    ):
        raise ValueError(
            f"invalid baseline stage1 kernel for block_m={block_m}: {current_s1}"
        )
    if current_s2_params is not None and (
        _effective_sort_block(current_s2) != block_m
        or current_s2_params.get("persist") is True
        or int(current_s2_params.get("b_nt", 0)) != 0
    ):
        raise ValueError(
            f"invalid baseline stage2 kernel for block_m={block_m}: {current_s2}"
        )

    _write_candidate(config_file, row)
    current_run = _runner(tensors, contract, config_file, swiglu_limit)
    current_ms, reference = _time_candidate(
        current_run, warmup_ms, rep_ms, stability_tolerance
    )
    if not torch.isfinite(reference).all():
        raise RuntimeError(
            f"M={row['token']}: baseline FHMoE kernels produced non-finite output: "
            f"{current_s1}, {current_s2}"
        )
    if not math.isfinite(current_ms):
        raise RuntimeError(
            f"M={row['token']}: baseline FHMoE kernels produced an unstable "
            f"or non-finite timing: {current_s1}, {current_s2}"
        )
    profile.append(
        {"stage": "baseline", "kernel": f"{current_s1}|{current_s2}", "ms": current_ms}
    )
    if baseline_only:
        print(
            f"M={row['token']}: baseline {current_ms:.4f} ms "
            f"({current_s1}, {current_s2})",
            flush=True,
        )
        return row, profile

    def benchmark_candidate(
        stage: str,
        kernel: str,
        candidate_row: dict[str, str],
    ) -> tuple[float, float]:
        try:
            _write_candidate(config_file, candidate_row)
            run = _runner(tensors, contract, config_file, swiglu_limit)
            ms, output = _time_candidate(run, warmup_ms, rep_ms, stability_tolerance)
            error = _relative_l2(output, reference)
        except Exception as candidate_error:  # noqa: BLE001
            ms = float("inf")
            error = float("inf")
            print(
                f"M={row['token']}: rejected {stage} candidate {kernel!r}: "
                f"{candidate_error}",
                flush=True,
            )
        profile.append({"stage": stage, "kernel": kernel, "ms": ms, "error": error})
        return ms, error

    best_s2, best_ms = current_s2, current_ms
    stage2_scores: list[tuple[float, str]] = []
    for candidate in sorted(set(stage2_candidates(block_m, quick)) | {current_s2}):
        candidate_row = _candidate_row(row, current_s1, candidate)
        ms, error = benchmark_candidate(
            "stage2",
            candidate,
            candidate_row,
        )
        if error <= tolerance and math.isfinite(ms):
            stage2_scores.append((ms, candidate))
        if error <= tolerance and ms < best_ms:
            best_s2, best_ms = candidate, ms

    best_s1 = current_s1
    stage1_scores: list[tuple[float, str]] = []
    for candidate in sorted(set(stage1_candidates(block_m, quick)) | {current_s1}):
        candidate_row = _candidate_row(row, candidate, best_s2)
        ms, error = benchmark_candidate(
            "stage1",
            candidate,
            candidate_row,
        )
        if error <= tolerance and math.isfinite(ms):
            stage1_scores.append((ms, candidate))
        if error <= tolerance and ms < best_ms:
            best_s1, best_ms = candidate, ms

    measured_pairs = {(current_s1, candidate) for _, candidate in stage2_scores} | {
        (candidate, best_s2) for _, candidate in stage1_scores
    }
    pair_stage1 = sorted(stage1_scores)
    pair_stage2 = sorted(stage2_scores)
    if quick:
        pair_stage1 = pair_stage1[:_PAIR_BEAM_WIDTH]
        pair_stage2 = pair_stage2[:_PAIR_BEAM_WIDTH]
    for _, stage1 in pair_stage1:
        for _, stage2 in pair_stage2:
            if (stage1, stage2) in measured_pairs:
                continue
            candidate_row = _candidate_row(row, stage1, stage2)
            kernel = f"{stage1}|{stage2}"
            ms, error = benchmark_candidate(
                "pair",
                kernel,
                candidate_row,
            )
            if error <= tolerance and ms < best_ms:
                best_s1, best_s2, best_ms = stage1, stage2, ms

    improvement_pct = (
        100.0 * (current_ms - best_ms) / current_ms if current_ms > 0 else 0.0
    )
    if improvement_pct < min_improvement_pct:
        best_s1, best_s2, best_ms = current_s1, current_s2, current_ms
    tuned = _candidate_row(row, best_s1, best_s2)
    print(
        f"M={row['token']}: {current_ms:.4f} ms -> {best_ms:.4f} ms "
        f"({current_s1}, {current_s2}) -> ({best_s1}, {best_s2})",
        flush=True,
    )
    return tuned, profile


def _validate_sampled_reference(
    row: dict[str, str],
    tensors: Tensors,
    contract: Contract,
    config_file: Path,
    *,
    swiglu_limit: float,
    tolerance: float,
) -> float:
    """Accept the full-M winner only when sampled rows match the FP32 oracle."""
    _write_candidate(config_file, row)
    actual = _runner(tensors, contract, config_file, swiglu_limit)()
    torch.accelerator.synchronize()
    rows = torch.tensor(
        tensors.sample_rows, dtype=torch.long, device=tensors.hidden.device
    )
    actual_sample = actual.index_select(0, rows).float()
    expected = _sampled_fp32_reference(tensors, contract, swiglu_limit)
    if not torch.isfinite(actual_sample).all() or not torch.isfinite(expected).all():
        raise RuntimeError(
            f"M={row['token']}: sampled FP32 validation produced non-finite output"
        )
    error = _relative_l2(actual_sample, expected)
    if error > tolerance:
        raise RuntimeError(
            f"M={row['token']}: sampled FP32 validation failed with "
            f"relative L2 {error:.6f} > {tolerance:.6f}"
        )
    print(
        f"M={row['token']}: sampled FP32 validation passed for "
        f"{len(tensors.sample_rows)} rows (relative L2={error:.6f})",
        flush=True,
    )
    return error


def _validate_tuning_environment() -> None:
    for name in ("AITER_FLYDSL_FORCE_REDUCE", "AITER_FLYDSL_STAGE2_FP8"):
        if os.environ.get(name, "0") not in ("", "0"):
            raise RuntimeError(
                f"FHMoE canonical tuning requires {name}=0; environment "
                "overrides would benchmark a different implementation"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=_parse_tokens(
            "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768"
        ),
    )
    parser.add_argument(
        "--benchmark-token",
        type=int,
        help=(
            f"actual M (1..{HY4_FHMOE_MAX_TOKENS}) used to tune exactly one "
            "selected CSV bucket; "
            "it must resolve to that bucket, for example --tokens 32768 "
            f"--benchmark-token {HY4_FHMOE_MAX_TOKENS}"
        ),
    )
    parser.add_argument(
        "--warmup-ms", "--warmup", type=_parse_positive_float, default=30.0
    )
    parser.add_argument(
        "--rep-ms", "--iters", type=_parse_positive_float, default=120.0
    )
    parser.add_argument(
        "--mp",
        type=int,
        default=1,
        help="accepted for tuner-infrastructure compatibility; FHMoE tunes one GPU",
    )
    parser.add_argument(
        "--tolerance",
        type=_parse_nonnegative_float,
        default=8e-3,
        help="maximum baseline-relative candidate-output L2 error",
    )
    parser.add_argument(
        "--stability-tolerance",
        type=_parse_nonnegative_float,
        default=8e-3,
        help="maximum repeat-output relative L2 variation (not timing jitter)",
    )
    parser.add_argument(
        "--sampled-reference-tolerance",
        type=_parse_nonnegative_float,
        default=4e-2,
    )
    parser.add_argument(
        "--min-improvement-pct", type=_parse_nonnegative_float, default=1.0
    )
    parser.add_argument("--swiglu-limit", type=_parse_nonnegative_float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Benchmark current CSV rows without searching or replacing kernels.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help=(
            "incremental CSV profile with columns "
            "token,runtime_token,stage,kernel,ms,error"
        ),
    )
    args = parser.parse_args()

    _validate_output_paths(args.input, args.output, args.profile)
    _validate_tuning_environment()
    if args.mp != 1:
        raise ValueError("FHMoE tuning currently supports --mp 1 only")
    _validate_benchmark_token(args.benchmark_token, args.tokens)
    if get_gfx_runtime() != "gfx950":
        raise RuntimeError("FHMoE tuning currently requires gfx950")
    contract = Contract()
    cu_num = get_cu_num()
    if cu_num != contract.cu_num:
        raise RuntimeError(f"FHMoE tuning requires {contract.cu_num} CUs, got {cu_num}")
    fields, rows = _read_rows(args.input)
    if fields != list(CSV_FIELDS):
        raise ValueError(f"unexpected FHMoE columns: {fields}")

    selected = _select_rows(rows, contract, args.tokens)
    missing = sorted(set(args.tokens) - selected.keys())
    if missing:
        raise ValueError(f"missing HY4 FHMoE rows for tokens: {missing}")
    baseline_rows = {token: dict(rows[index]) for token, index in selected.items()}
    if not args.baseline_only and not _paths_alias(args.input, args.output):
        _validate_existing_output(args.output, fields, baseline_rows, contract)

    device = torch.device("cuda")
    with tempfile.TemporaryDirectory(prefix="fhmoe-tune-") as directory:
        candidate_file = Path(directory) / "candidate.csv"
        for token in args.tokens:
            index = selected[token]
            benchmark_token = (
                args.benchmark_token if args.benchmark_token is not None else token
            )
            if benchmark_token != token:
                print(
                    f"tuning CSV bucket M={token} with runtime M={benchmark_token}",
                    flush=True,
                )
            tensors = _make_tensors(
                contract,
                benchmark_token,
                args.seed + benchmark_token,
                device,
            )
            tuned, row_profile = tune_row(
                rows[index],
                tensors,
                contract,
                candidate_file,
                quick=args.quick,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                tolerance=args.tolerance,
                swiglu_limit=args.swiglu_limit,
                min_improvement_pct=args.min_improvement_pct,
                stability_tolerance=args.stability_tolerance,
                baseline_only=args.baseline_only,
            )
            # Candidate ranking above compares complete full-M outputs for speed.
            # Run the independent FP32 oracle only once, on the selected winner.
            _validate_sampled_reference(
                tuned,
                tensors,
                contract,
                candidate_file,
                swiglu_limit=args.swiglu_limit,
                tolerance=args.sampled_reference_tolerance,
            )
            rows[index] = tuned
            for profile_row in row_profile:
                profile_row["token"] = token
                profile_row["runtime_token"] = benchmark_token
            if not args.baseline_only:
                _write_tuned_output(
                    args.input,
                    args.output,
                    fields,
                    rows,
                    {token: index},
                    {token: baseline_rows[token]},
                    contract,
                )
            if args.profile:
                try:
                    _write_profile_rows(args.profile, row_profile)
                except Exception as error:
                    raise RuntimeError(
                        f"failed to commit FHMoE profile rows to {args.profile}"
                    ) from error
            del tensors
            torch.cuda.empty_cache()

    if args.baseline_only:
        _write_tuned_output(
            args.input,
            args.output,
            fields,
            rows,
            {},
            baseline_rows,
            contract,
        )
    print(
        f"processed {len(selected)} HY4 FHMoE rows and wrote {args.output} "
        f"on {get_gfx_runtime()} ({get_cu_num()} CUs)",
        flush=True,
    )


if __name__ == "__main__":
    main()
