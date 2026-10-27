"""Build, merge and write the fused ladder file, ``fused_allreduce_ladders.json``.

``allreduce_policy.load_fused_ladders`` reads and checks the file; this
module only writes it. Every platform entry is::

    {"arch", "link", "world_size",
     "oneshot": {"default": {"rungs": [...]}, "hidden": {"<h>": {"rungs": [...]}}},
     "mesh":    {"default": {"atoms_per_row": a, "rungs": [...]}, "hidden": {...}},
     "ring":    {...}}

``write`` puts one rung per line, so a diff of the file reads rung by rung.
"""

import json

from aiter.ops.flydsl.allreduce_policy import FUSED_LADDERS_SCHEMA_VERSION

FAMILIES = ("oneshot", "mesh", "ring")


def byte_value(n: int):
    """*n* bytes as the file writes them: ``"48 KiB"``, ``"4 MiB"`` or ``0``."""
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if n and n % (1 << shift) == 0:
            return f"{n >> shift} {unit}"
    return int(n)


def oneshot_rung(min_bytes, atoms, grid_cap, split, fanout="peer") -> dict:
    return {
        "min_bytes": byte_value(min_bytes),
        "atoms": int(atoms),
        "grid_cap": int(grid_cap),
        "fanout": fanout,
        "split": int(split),
    }


def two_shot_rung(min_bytes, super_tile, grid_cap, low_latency=False) -> dict:
    """A mesh or ring rung; ``low_latency`` is written only when set."""
    rung = {
        "min_bytes": byte_value(min_bytes),
        "super_tile": int(super_tile),
        "grid_cap": int(grid_cap),
    }
    if low_latency:
        rung["low_latency"] = True
    return rung


def entry(rungs, **extra) -> dict:
    """A ladder entry: ``rungs``, plus ``atoms_per_row`` for the mesh and the
    ring when given."""
    return {**extra, "rungs": list(rungs)}


def empty() -> dict:
    return {"schema_version": FUSED_LADDERS_SCHEMA_VERSION, "platforms": []}


def read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def platform(doc: dict, arch: str, link: str, world_size: int) -> dict | None:
    """The platform entry of *doc* for *(arch, link, world_size)*, or None."""
    for plat in doc["platforms"]:
        if (plat["arch"], plat["link"], plat["world_size"]) == (
            arch,
            link,
            int(world_size),
        ):
            return plat
    return None


def set_family(
    doc: dict,
    arch: str,
    link: str,
    world_size: int,
    family: str,
    default: dict,
    hidden: dict,
    swept,
) -> tuple[int, ...]:
    """Replace *family*'s ladders on one platform of *doc* with a fit: the
    *default* entry, and for each width in *swept* its entry in *hidden*
    (``{width: entry}``), or none when *hidden* has none, so that the width runs
    the default. A width the fit did not sweep keeps its entry, as do the
    platform's other families. Returns the widths kept that way."""
    plat = platform(doc, arch, link, world_size)
    if plat is None:
        raise KeyError(
            f"{arch}/{link}/TP{world_size} is not in the base file: add the "
            "platform by hand first, with a ladder for every family"
        )
    old = plat.get(family, {}).get("hidden", {})
    swept = {int(h) for h in swept}
    kept = {k: v for k, v in old.items() if k.startswith("_") or int(k) not in swept}
    widths = {**kept, **{str(h): e for h, e in hidden.items()}}
    spec = {"default": default}
    if widths:
        # Comments first, then the widths in ascending order.
        def order(key):
            return (0, 0) if key.startswith("_") else (1, int(key))

        spec["hidden"] = {k: widths[k] for k in sorted(widths, key=order)}
    plat[family] = spec
    return tuple(sorted(int(k) for k in kept if not k.startswith("_")))


def _dumps(obj, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(obj, dict) and obj:
        if not any(isinstance(v, (dict, list)) for v in obj.values()):
            # A leaf object (a rung) stays on one line.
            return (
                "{"
                + ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in obj.items())
                + "}"
            )
        items = ",\n".join(
            f"{pad}  {json.dumps(k)}: {_dumps(v, depth + 1)}" for k, v in obj.items()
        )
        return "{\n" + items + "\n" + pad + "}"
    if isinstance(obj, list) and obj:
        items = ",\n".join(f"{pad}  {_dumps(v, depth + 1)}" for v in obj)
        return "[\n" + items + "\n" + pad + "]"
    return json.dumps(obj)


def dumps(doc: dict) -> str:
    return _dumps(doc) + "\n"


def write(doc: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write(dumps(doc))
