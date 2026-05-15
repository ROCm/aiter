import re
from enum import IntEnum
from pathlib import Path

from ..jit.core import compile_ops

# Keep the public alias compatible with existing annotations.
Enum = int

__all__ = ["Enum", "ActivationType", "QuantType"]


@compile_ops("module_aiter_core", "ActivationType")
def _ActivationType(dummy): ...


@compile_ops("module_aiter_core", "QuantType")
def _QuantType(dummy): ...


def _find_aiter_enum_h() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "csrc" / "include" / "aiter_enum.h",
        root / "aiter_meta" / "csrc" / "include" / "aiter_enum.h",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"aiter_enum.h not found in {[str(p) for p in candidates]}")


def _parse_enum_values(header: Path, enum_name: str) -> dict[str, int]:
    text = header.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    match = re.search(rf"enum\s+class\s+{enum_name}\s*:\s*int\s*\{{([^}}]+)\}}", text)
    if match is None:
        raise ValueError(f"{enum_name} enum not found in {header}")

    values = {}
    next_value = 0
    for line in match.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip().rstrip(",")
        if not line:
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            name = name.strip()
            next_value = int(value.strip())
        else:
            name = line
        values[name] = next_value
        next_value += 1
    return values


def _fallback_enum(enum_name: str):
    return IntEnum(enum_name, _parse_enum_values(_find_aiter_enum_h(), enum_name))


try:
    ActivationType = type(_ActivationType(0))
    QuantType = type(_QuantType(0))
except (ImportError, RuntimeError, OSError, KeyError, ModuleNotFoundError):
    ActivationType = _fallback_enum("ActivationType")
    QuantType = _fallback_enum("QuantType")
