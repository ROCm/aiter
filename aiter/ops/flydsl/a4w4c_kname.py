# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel names of the FlyDSL A4W4 compact MoE (``aiter.ops.flydsl.moe_a4w4_compact``).

A tuned row names one kernel per stage::

    kernelName1 = flydsl_a4w4c_g1_<launch>[_qp][_ht]
    kernelName2 = flydsl_a4w4c_g2_<launch>[_ht][_fc_<launch>]
    <launch>    = <BM>x<BN>_<pipe>_d<D>_w<NW>[_wm<WM>][_persist][_fexp][_vacc][_<opt>...]

Launch-wide settings sit on the stage they change: ``qp`` (activation quant runs
inside the plan launch) on g1, ``ht`` (h stored K-step-major: g1 writes it, g2
reads it) on both, ``fc`` (fused combine; followed by the shared-expert tile
launch) on g2. BN follows from the wave layout and is only checked.

Tokens: ``persist`` persistent CTAs, ``fexp`` fast exp/rcp SwiGLU epilogue,
``vacc`` MFMA accumulators in VGPRs (no AGPRs); ``<opt>`` is a kernel option:
``bar<N>`` il4 barrier slot, ``s1tr`` transposed stage-1 epilogue, ``epg<N>``
row blocks per epilogue batch, ``nt`` non-temporal stage-2 stores, ``s2tl``
LDS-transposed stage-2 stores, ``s2db`` double-buffered s2tl, ``nos2w`` direct
stage-2 stores, ``wpe<N>`` waves per EU, ``agpr<N>`` AGPR budget, ``fcskip``
fused combine skips the shared expert's own slot.
"""

import re

IMPL_NAME = "flydsl_a4w4c"
G1_PREFIX = "flydsl_a4w4c_g1_"
G2_PREFIX = "flydsl_a4w4c_g2_"
_PIPES = ("async", "hybrid", "hybrid2", "il4")
_TILE_RE = re.compile(r"^(\d+)x(\d+)$")
_NUM_RE = re.compile(r"^(persist|wm|d|w)(\d*)$")
_NUM_KEYS = {"d": "D", "w": "NW", "wm": "WM", "persist": "PERS"}
_BOOL_KEYS = {"fexp": "EF", "vacc": "MV"}
# Name token -> kernel diag option; "#" is a trailing integer.
_OPTIONS = {
    "nt": "s2nt",
    "s1tr": "s1tr",
    "s2tl": "s2tl",
    "s2db": "s2db",
    "nos2w": "nos2w",
    "fcskip": "fcsk",
    "bar#": "bar#",
    "epg#": "epg#",
    "wpe#": "wpe#",
    "agpr#": "ag#",
}
# The fused-combine launch takes its AGPR budget from "agf#".
_FUSED_OPTIONS = {**_OPTIONS, "agpr#": "agf#"}
# Fused-launch settings inherited from stage 2 unless overridden (diag is not).
_FUSED_KEYS = ("BM", "NW", "WM", "D", "pipe")


def _split_num(token):
    stem = token.rstrip("0123456789")
    num = token[len(stem) :]
    return (stem + "#", num) if num else (token, "")


def _option_token(option, table):
    stem, num = _split_num(option)
    for token, diag in table.items():
        if diag == stem:
            return token.replace("#", num)
    raise ValueError(f"no kernel-name token for option {option!r}")


def _check_waves(cfg):
    nw, wm = cfg["NW"], cfg.get("WM", 1)
    if nw < 1 or wm < 1 or nw % wm:
        return f"NW={nw} must be a positive multiple of WM={wm}"
    return None


def _tile_bn(cfg):
    error = _check_waves(cfg)
    if error:
        raise ValueError(error)
    return 64 * (cfg["NW"] // cfg.get("WM", 1)) * (2 if cfg["pipe"] == "il4" else 1)


def _launch_tokens(cfg, table):
    tokens = [f"{cfg['BM']}x{_tile_bn(cfg)}", cfg["pipe"], f"d{cfg['D']}"]
    tokens.append(f"w{cfg['NW']}")
    if cfg.get("WM", 1) != 1:
        tokens.append(f"wm{cfg['WM']}")
    if cfg.get("PERS"):
        tokens.append("persist" if cfg["PERS"] == 1 else f"persist{cfg['PERS']}")
    for token, key in _BOOL_KEYS.items():
        if cfg.get(key):
            assert cfg[key] == 1, (key, cfg[key])
            tokens.append(token)
    if cfg.get("diag"):
        tokens += [_option_token(o, table) for o in cfg["diag"].split("+")]
    return tokens


def kernel_names(cell: dict) -> tuple[str, str]:
    """(kernelName1, kernelName2) for one tile-table cell."""
    glob = cell.get("global", {})
    g1 = _launch_tokens(cell["s1"], _OPTIONS)
    g2 = _launch_tokens(cell["s2"], _OPTIONS)
    if glob.get("QP"):
        g1.append("qp")
    if glob.get("HT"):
        g1.append("ht")
        g2.append("ht")
    if glob.get("FC"):
        fused = {k: glob.get(k + "F", cell["s2"].get(k)) for k in _FUSED_KEYS}
        fused = {k: v for k, v in fused.items() if v is not None}
        if glob.get("diagF"):
            fused["diag"] = glob["diagF"]
        g2 += ["fc", *_launch_tokens(fused, _FUSED_OPTIONS)]
    return G1_PREFIX + "_".join(g1), G2_PREFIX + "_".join(g2)


def _check_launch(cfg, stage, fused=False):
    """The build_gemm preconditions a name can violate, as a reason or None."""
    pipe, opts = cfg["pipe"], set(cfg.get("diag", "").split("+")) - {""}
    il4 = pipe == "il4"
    if cfg.get("WM", 1) > 1 and not il4:
        return "wm > 1 needs il4"
    if cfg.get("PERS") and not il4:
        return "persist needs il4"
    if il4 and (
        fused
        or cfg["BM"] != 256
        or cfg["NW"] != 4
        or cfg.get("WM", 1) != 2
        or not cfg.get("PERS")
    ):
        return "il4 needs 256x256, w4, wm2 and persist (not on the fused launch)"
    if il4 and stage == 2 and "s2tl" not in opts:
        return "il4 stage 2 needs s2tl"
    if "s1tr" in opts and not (stage == 1 and il4 and cfg.get("EF")):
        return "s1tr needs il4 stage 1 with fexp"
    if cfg["BM"] % (16 * cfg.get("WM", 1)):
        return "BM must be a multiple of 16 * WM"
    return None


def _parse_launch(name, tokens, table, flags):
    """One launch's settings, the trailing flags, and any tokens after them."""
    bad = f"bad A4W4 compact kernel name {name!r}"
    tile = _TILE_RE.match(tokens[0]) if tokens else None
    if not tile or len(tokens) < 4 or tokens[1] not in _PIPES:
        raise ValueError(bad)
    cfg = {"BM": int(tile.group(1)), "pipe": tokens[1]}
    options, rest = [], tokens[2:]
    while rest and rest[0] not in flags:
        token = rest.pop(0)
        num = _NUM_RE.match(token)
        stem, digits = _split_num(token)
        if token in _BOOL_KEYS:
            key, value = _BOOL_KEYS[token], 1
        elif num and (num.group(2) or num.group(1) == "persist"):
            key, value = _NUM_KEYS[num.group(1)], int(num.group(2) or 1)
        elif stem in table:
            option = table[stem].replace("#", digits)
            if option in options:
                raise ValueError(f"{bad}: repeated {token!r}")
            options.append(option)
            continue
        else:
            raise ValueError(f"{bad}: unknown token {token!r}")
        if key in cfg:
            raise ValueError(f"{bad}: repeated {token!r}")
        cfg[key] = value
    if "D" not in cfg or "NW" not in cfg:
        raise ValueError(f"{bad}: needs d<D> and w<NW>")
    if _check_waves(cfg):
        raise ValueError(f"{bad}: {_check_waves(cfg)}")
    if _tile_bn(cfg) != int(tile.group(2)):
        raise ValueError(f"{bad}: BN must be {_tile_bn(cfg)}")
    if options:
        cfg["diag"] = "+".join(options)
    seen = set()
    while rest and rest[0] in flags and rest[0] != "fc":
        seen.add(rest.pop(0))
    return cfg, seen, rest


def parse_knames(kname1: str, kname2: str) -> dict:
    """``MoERun`` keyword arguments for a (kernelName1, kernelName2) pair."""
    if not (is_a4w4c_kname(kname1) and str(kname2).startswith(G2_PREFIX)):
        raise ValueError(f"not an A4W4 compact pair: {kname1!r}, {kname2!r}")
    s1, f1, rest1 = _parse_launch(
        kname1, kname1[len(G1_PREFIX) :].split("_"), _OPTIONS, {"qp", "ht"}
    )
    s2, f2, rest2 = _parse_launch(
        kname2, kname2[len(G2_PREFIX) :].split("_"), _OPTIONS, {"ht", "fc"}
    )
    if rest1 or (rest2 and rest2[0] != "fc") or "qp" in f2:
        raise ValueError(f"bad A4W4 compact kernel names {kname1!r}, {kname2!r}")
    if ("ht" in f1) != ("ht" in f2):
        raise ValueError(f"ht must be on both stages: {kname1!r}, {kname2!r}")
    for stage, cfg in ((1, s1), (2, s2)):
        error = _check_launch(cfg, stage)
        if error:
            raise ValueError(
                f"bad A4W4 compact kernel names {kname1!r}, {kname2!r}: {error}"
            )
    if s2["pipe"] == "il4" and "ht" not in f2:
        raise ValueError(f"il4 stage 2 needs ht: {kname1!r}, {kname2!r}")
    kwargs = {f"{k}1": v for k, v in s1.items()}
    kwargs.update({f"{k}2": v for k, v in s2.items()})
    if "qp" in f1:
        kwargs["QP"] = 1
    if "ht" in f1:
        kwargs["HT"] = 1
    if rest2:
        fused, _, tail = _parse_launch(kname2, rest2[1:], _FUSED_OPTIONS, set())
        if tail:
            raise ValueError(f"bad A4W4 compact kernel name {kname2!r}: {tail}")
        extra = set(fused) - {*_FUSED_KEYS, "diag"}
        if extra:
            raise ValueError(
                f"bad A4W4 compact kernel name {kname2!r}: "
                f"fused launch cannot set {sorted(extra)}"
            )
        error = _check_launch(fused, 2, fused=True)
        if error:
            raise ValueError(f"bad A4W4 compact kernel name {kname2!r}: {error}")
        kwargs["FC"] = 1
        kwargs.update({k + "F": fused.get(k, 1) for k in ("BM", "NW", "WM", "D")})
        kwargs["pipeF"] = fused["pipe"]
        if "diag" in fused:
            kwargs["diagF"] = fused["diag"]
    return kwargs


def is_a4w4c_kname(kname) -> bool:
    return isinstance(kname, str) and kname.startswith(G1_PREFIX)


def _cell(kwargs):
    """Tile-table cell for parsed ``MoERun`` kwargs (inverse of ``cfg_kwargs``)."""
    cell = {"s1": {}, "s2": {}, "global": {}}
    for key, value in kwargs.items():
        if key[-1] in "12" and key[:-1] in (*_FUSED_KEYS, "PERS", "EF", "MV", "diag"):
            cell[f"s{key[-1]}"][key[:-1]] = value
        else:
            cell["global"][key] = value
    return cell


def impl_config(kname1: str, kname2: str) -> str:
    """Registry config string carrying both stage names. Raises ValueError unless
    the pair parses and is in canonical form (no repeated or reordered tokens)."""
    kwargs = parse_knames(kname1, kname2)
    if kernel_names(_cell(kwargs)) != (kname1, kname2):
        raise ValueError(
            f"non-canonical A4W4 compact kernel names {kname1!r}, {kname2!r}"
        )
    return f"{kname1}+{kname2}"


def split_impl_config(config: str) -> tuple[str, str]:
    kname1, _, kname2 = config.partition("+")
    return kname1, kname2
