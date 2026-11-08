# SPDX-License-Identifier: MIT
"""Offline candidate policy and exact-contract lookup for gfx1250 TDM GEMM."""

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "tdm_split_policy_v1"
KEYS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "schema",
    "dtype",
    "layout",
    "lda",
    "ldb",
    "ldc",
    "bias",
    "allow_atomic",
    "max_workspace_bytes",
    "relative_l2_limit",
]
# Small, medium and compute-bound families; do not enumerate thousands of tiles.
TILES = (
    (64, 64, 2, 2, False),
    (128, 128, 2, 2, True),
    (128, 256, 2, 2, True),
    (256, 128, 2, 2, True),
    (256, 256, 2, 2, True),
    (256, 256, 2, 4, True),
    (256, 256, 4, 4, False),
)


def precision_limit(dtype, allow_atomic, requested):
    """Independent of data: choose a finite per-dtype numerical acceptance cap."""
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("relative L2 bound must be finite and positive")
    if dtype not in ("torch.bfloat16", "torch.float16"):
        raise ValueError("unsupported tuning dtype")
    if allow_atomic:
        cap = 0.02 if dtype == "torch.bfloat16" else 0.003
    else:
        cap = 0.004 if dtype == "torch.bfloat16" else 0.0006
    return min(requested, cap)


@dataclass(frozen=True)
class TdmConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 128
    stages: int = 2
    m_waves: int = 2
    n_waves: int = 2
    split_k: int = 1
    split_k_policy: str = "workspace"
    group_m: int = 0
    interleave: bool = True

    def kwargs(self):
        return asdict(self)

    def encode(self):
        return json.dumps(self.kwargs(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def decode(cls, text):
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("invalid TDM config fields")
        obj = cls(**data)
        for key, value in obj.kwargs().items():
            if key == "split_k_policy":
                if value not in ("atomic", "workspace"):
                    raise ValueError("invalid reduction policy")
            elif key == "interleave":
                if type(value) is not bool:
                    raise ValueError("interleave must be boolean")
            elif type(value) is not int:
                raise ValueError(f"{key} must be integer")
        return obj

    def workspace_bytes(self, m, n):
        if self.split_k <= 1 or self.split_k_policy != "workspace":
            return 0
        tiles = ((m + self.block_m - 1) // self.block_m) * (
            (n + self.block_n - 1) // self.block_n
        )
        return tiles * self.split_k * self.block_m * self.block_n * 4

    def validate(self, m, n, k, layout, allow_atomic, max_workspace_bytes):
        from .kernels.gemm_a16w16_tdm_gfx1250 import validate_config

        # Also validates types for configurations constructed without decode().
        TdmConfig.decode(self.encode())
        if min(m, n, k) <= 0:
            raise ValueError("positive shape required")
        if type(max_workspace_bytes) is not int or max_workspace_bytes < 0:
            raise ValueError("workspace budget must be a nonnegative integer")
        if type(allow_atomic) is not bool:
            raise ValueError("allow_atomic must be boolean")
        validate_config(
            self.block_m,
            self.block_n,
            self.block_k,
            self.stages,
            self.m_waves,
            self.n_waves,
            self.split_k,
            self.group_m,
            layout,
            self.interleave,
            self.split_k_policy,
        )
        if (
            k % (self.split_k * self.block_k)
            or k // (self.split_k * self.block_k) < self.stages
        ):
            raise ValueError("K split/pipeline does not fit")
        if self.group_m * ((n + self.block_n - 1) // self.block_n) >= 2**31:
            raise ValueError("swizzle overflow")
        if (
            self.split_k > 1
            and self.split_k_policy == "atomic"
            and (not allow_atomic or n % 2)
        ):
            raise ValueError("atomic split-K not permitted or N is odd")
        if self.workspace_bytes(m, n) > max_workspace_bytes:
            raise ValueError("workspace budget exceeded")
        return self


def candidates(
    m,
    n,
    k,
    *,
    layout="nt",
    cu_num=256,
    allow_atomic=False,
    max_workspace_bytes=256 << 20,
    splits=(1, 2, 4, 8),
    exhaustive=False,
):
    """Finite joint tile/split/reduction search, with safe shape pruning."""
    if min(m, n, k, cu_num) <= 0 or max_workspace_bytes < 0:
        raise ValueError("invalid shape, CU count or workspace budget")
    if not splits or any(type(s) is not int or not 1 <= s <= 9 for s in splits):
        raise ValueError("splits must be nonempty and in 1..9")
    splits = tuple(dict.fromkeys((1, *splits)))
    result = []
    for tile, bk, stages, split in itertools.product(
        TILES, (64, 128, 256), (2, 3), splits
    ):
        bm, bn, mw, nw, interleave = tile
        if layout == "nn":
            interleave = False
        tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
        if not exhaustive:
            # Do not multiply a saturated output grid by K splits.
            if split > 1 and tiles * split > 2 * cu_num:
                continue
            if stages == 3 and bk == 64:
                continue
        policies = (
            ("workspace",)
            if split == 1 or not allow_atomic
            else ("workspace", "atomic")
        )
        for policy in policies:
            for group in (0, 4) if tiles >= cu_num else (0,):
                c = TdmConfig(
                    bm, bn, bk, stages, mw, nw, split, policy, group, interleave
                )
                try:
                    c.validate(m, n, k, layout, allow_atomic, max_workspace_bytes)
                except ValueError:
                    continue
                result.append(c)
    return list(dict.fromkeys(result))


def contract(
    a, b, out, bias, layout, allow_atomic, max_workspace_bytes, relative_l2_limit=0.004
):
    import torch

    props = torch.cuda.get_device_properties(a.device)
    m, k = a.shape
    n = b.shape[1]
    return dict(
        zip(
            KEYS,
            (
                props.gcnArchName.split(":")[0],
                props.multi_processor_count,
                m,
                n,
                k,
                SCHEMA,
                str(a.dtype),
                layout,
                a.stride(0),
                b.stride(1) if layout == "nt" else b.stride(0),
                out.stride(0) if out is not None else (n + 7) // 8 * 8,
                int(bias is not None),
                int(allow_atomic),
                max_workspace_bytes,
                format(
                    precision_limit(str(a.dtype), allow_atomic, relative_l2_limit),
                    ".12g",
                ),
            ),
        )
    )


class TdmTunedPolicy:
    """Load once before serving/graph capture; exact-match only, no hidden tuning.

    Missing contracts raise KeyError rather than guessing a split-K policy.
    Atomic entries require allow_atomic=True, even when a CSV has been edited.
    """

    def __init__(self, path):
        self.rows = {}
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            if not set(KEYS + ["kernelName", "us", "errRatio"]) <= set(
                reader.fieldnames or ()
            ):
                raise ValueError("missing TDM tuning columns")
            for row in reader:
                if row["schema"] != SCHEMA:
                    raise ValueError("unsupported TDM policy schema")
                config = TdmConfig.decode(row["kernelName"])
                us, err = float(row["us"]), float(row["errRatio"])
                if (
                    not math.isfinite(us)
                    or us <= 0
                    or not math.isfinite(err)
                    or err < 0
                ):
                    raise ValueError("invalid/unmeasured tuning row")
                if row["gfx"] != "gfx1250" or row["dtype"] not in (
                    "torch.bfloat16",
                    "torch.float16",
                ):
                    raise ValueError("unsupported tuning device/dtype")
                if row["allow_atomic"] not in ("0", "1") or row["bias"] not in (
                    "0",
                    "1",
                ):
                    raise ValueError("invalid tuning flags")
                m, n, k = (int(row[x]) for x in ("M", "N", "K"))
                config.validate(
                    m,
                    n,
                    k,
                    row["layout"],
                    row["allow_atomic"] == "1",
                    int(row["max_workspace_bytes"]),
                )
                requested = float(row["relative_l2_limit"])
                effective = precision_limit(
                    row["dtype"], row["allow_atomic"] == "1", requested
                )
                if effective != requested or err > effective:
                    raise ValueError("tuning error exceeds the precision contract")
                row["relative_l2_limit"] = format(effective, ".12g")
                if int(row["cu_num"]) < 1:
                    raise ValueError("invalid CU count")
                strides = {x: int(row[x]) for x in ("lda", "ldb", "ldc")}
                minima = {"lda": k, "ldb": k if row["layout"] == "nt" else n, "ldc": n}
                if any(strides[x] < minima[x] or strides[x] % 8 for x in strides):
                    raise ValueError("invalid tuning strides")
                if (
                    max(
                        m * strides["lda"],
                        (n if row["layout"] == "nt" else k) * strides["ldb"],
                        m * strides["ldc"],
                    )
                    * 2
                    >= 2**31
                ):
                    raise ValueError("tuning tensor footprint too large")
                key = tuple(str(row[k]) for k in KEYS)
                if key in self.rows:
                    raise ValueError("duplicate TDM tuning contract")
                self.rows[key] = config

    def select(
        self,
        a,
        b,
        out=None,
        bias=None,
        *,
        layout="nt",
        allow_atomic=False,
        max_workspace_bytes=256 << 20,
        relative_l2_limit=0.004,
    ):
        import torch

        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[0]
            or a.device.type != "cuda"
        ):
            raise ValueError("expected CUDA A[M,K], B[K,N]")
        if (
            b.device != a.device
            or b.dtype != a.dtype
            or a.dtype not in (torch.bfloat16, torch.float16)
        ):
            raise ValueError("inputs must have the same BF16/FP16 dtype and device")
        if a.stride(1) != 1 or b.stride(0 if layout == "nt" else 1) != 1:
            raise ValueError("minor strides do not match the tuning layout")
        key = contract(
            a,
            b,
            out,
            bias,
            layout,
            allow_atomic,
            max_workspace_bytes,
            relative_l2_limit,
        )
        config = self.rows[tuple(str(key[k]) for k in KEYS)]
        config.validate(
            a.shape[0],
            b.shape[1],
            a.shape[1],
            layout,
            allow_atomic,
            max_workspace_bytes,
        )
        if (
            torch.are_deterministic_algorithms_enabled()
            and config.split_k > 1
            and config.split_k_policy == "atomic"
        ):
            raise RuntimeError("selected atomic split-K is nondeterministic")
        return config

    def __call__(
        self,
        a,
        b,
        out=None,
        bias=None,
        *,
        layout="nt",
        allow_atomic=False,
        max_workspace_bytes=256 << 20,
        relative_l2_limit=0.004,
        stream=None,
    ):
        from .gemm_a16w16_tdm import gemm_a16w16_tdm

        config = self.select(
            a,
            b,
            out,
            bias,
            layout=layout,
            allow_atomic=allow_atomic,
            max_workspace_bytes=max_workspace_bytes,
            relative_l2_limit=relative_l2_limit,
        )
        return gemm_a16w16_tdm(
            a, b, out, bias, layout=layout, stream=stream, **config.kwargs()
        )
