# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Runtime capability probe. Every probe is guarded; none may raise. The `mori` Python package is NEVER imported:
the sp_qkv variants use the MORI CCO C++ library through ctypes (MORI_CCO_SO / MORI_ROOT, as the Phase-1 drivers).

mori_cco         : the CCO shared library resolves (MORI_CCO_SO, then MORI_ROOT/lib, MORI_ROOT/build/src/cco, then the default
                   library search), is a loadable ELF (dlopen succeeds) whose .dynsym defines every symbol gf_comm.cpp calls
                   (ccoGetUniqueId, ccoCommCreate, ccoWindowRegister, ccoGetPeerPtr, ccoDevCommCreate, ccoDevCommDestroy,
                   ccoWindowDeregister, ccoCommDestroy), AND the JIT header MORI_ROOT/include/mori/cco/cco.hpp exists.
sdma             : mori_cco AND the installed header defines the SDMA device API the large-M kernels compile against (BUILD_CCO_SDMA gate,
                   ccoSdmaContext, sdmaQueueCount; the r11 device/host sources include only <mori/cco/cco.hpp>) and ccoDevCommCreate is exported
                   (already required by mori_cco). The installed libmori_cco.so exports NO sdma/anvil symbol names (Evidence, nm -D, lib 87b7322f),
                   so no symbol heuristic is used; lg_attach_v1 additionally fails hard if devcomm sdmaNumQueue != the requested count.
gfx950_sc_bypass : arch is gfx950 (sc0/sc1 cache-control semantics of that ISA).
CCO functions are C++ (Itanium-mangled in namespace mori::cco), so symbols are matched by their mangled `<len><name>` token.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

CCO_REQUIRED_SYMBOLS = (
    "ccoGetUniqueId", "ccoCommCreate", "ccoCommDestroy", "ccoWindowRegister", "ccoWindowDeregister",
    "ccoGetPeerPtr", "ccoDevCommCreate", "ccoDevCommDestroy",
)
# Header tokens the SDMA path of the large-M kernels is compiled against (cco.hpp: BUILD_CCO_SDMA gate, ccoSdmaContext, sdmaQueueCount).
SDMA_HEADER_TOKENS = ("BUILD_CCO_SDMA", "ccoSdmaContext", "sdmaQueueCount")
CCO_HEADER_REL = os.path.join("include", "mori", "cco", "cco.hpp")
_DEFAULT_ROOT = "/reference-mori"
_SYM_CACHE: Dict[Tuple[str, int, int], Optional[Set[str]]] = {}


# ----------------------------------------------------------------------------- ELF .dynsym reader (no external tools)
def read_dynsym_names(path: str) -> Optional[Set[str]]:
    """Names of the DEFINED symbols in .dynsym of a little-endian ELF64 shared object; None if not such a file / unreadable."""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
        if key in _SYM_CACHE:
            return _SYM_CACHE[key]
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    names = _parse_dynsym(data)
    _SYM_CACHE[key] = names
    return names


def _parse_dynsym(data: bytes) -> Optional[Set[str]]:
    try:
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:  # ELF64, little endian
            return None
        e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
        e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
        if e_shoff == 0 or e_shnum == 0 or e_shentsize < 64:
            return None
        secs: List[Tuple[int, int, int, int, int]] = []  # (type, offset, size, link, entsize)
        for i in range(e_shnum):
            o = e_shoff + i * e_shentsize
            _name, sh_type, _flags, _addr, sh_off, sh_size, sh_link, _info, _align, sh_entsize = struct.unpack_from("<IIQQQQIIQQ", data, o)
            secs.append((sh_type, sh_off, sh_size, sh_link, sh_entsize))
        out: Set[str] = set()
        found = False
        for sh_type, off, size, link, entsize in secs:
            if sh_type != 11:  # SHT_DYNSYM
                continue
            found = True
            stroff, strsize = secs[link][1], secs[link][2]
            strtab = data[stroff:stroff + strsize]
            ent = entsize or 24
            for j in range(size // ent):
                st_name, _i, _o, st_shndx = struct.unpack_from("<IBBH", data, off + j * ent)
                if st_shndx == 0 or st_name == 0:  # undefined import / null
                    continue
                end = strtab.find(b"\0", st_name)
                out.add(strtab[st_name:end if end >= 0 else None].decode("utf-8", "replace"))
        return out if found else None
    except (struct.error, IndexError):
        return None


def _has_cco_symbol(names: Iterable[str], func: str) -> bool:
    tok = f"{len(func)}{func}"  # Itanium: <source-name> = <length><identifier>
    return any(tok in n or n == func for n in names)


# ----------------------------------------------------------------------------- resolution
def mori_root(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("MORI_ROOT") or _DEFAULT_ROOT


def resolve_cco_library(env=None) -> Optional[str]:
    """MORI_CCO_SO, then MORI_ROOT/lib, MORI_ROOT/build/src/cco (Phase-1 layout), then the default library search. None if not found."""
    env = os.environ if env is None else env
    so = env.get("MORI_CCO_SO")
    if so:
        return so if os.path.isfile(so) else None  # an explicit path is authoritative: no silent fallthrough
    root = mori_root(env)
    for cand in (os.path.join(root, "lib", "libmori_cco.so"), os.path.join(root, "build", "src", "cco", "libmori_cco.so")):
        if os.path.isfile(cand):
            return cand
    found = ctypes.util.find_library("mori_cco")
    return found if found and os.path.isabs(found) and os.path.isfile(found) else (found or None)


def cco_header_available(env=None) -> bool:
    return os.path.isfile(os.path.join(mori_root(env), CCO_HEADER_REL))


def cco_header_has_sdma(env=None) -> bool:
    try:
        with open(os.path.join(mori_root(env), CCO_HEADER_REL), "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        return all(tok in txt for tok in SDMA_HEADER_TOKENS)
    except OSError:
        return False


def _dlopen_ok(path: str) -> bool:
    try:
        ctypes.CDLL(path)  # loads the library and its DT_NEEDED deps; CCO runtime state is created only by ccoCommCreate
        return True
    except OSError:
        return False


def probe_mori_cco(env=None, check_dlopen: bool = True) -> Tuple[bool, bool, str]:
    """(mori_cco, sdma, reason). Never raises."""
    try:
        lib = resolve_cco_library(env)
        if not lib:
            return False, False, "CCO library not found (MORI_CCO_SO / MORI_ROOT/lib / MORI_ROOT/build/src/cco / default search)"
        if not cco_header_available(env):
            return False, False, f"CCO header missing: {os.path.join(mori_root(env), CCO_HEADER_REL)}"
        names = read_dynsym_names(lib)
        if names is None:
            return False, False, f"{lib}: not a readable ELF64 shared object with .dynsym"
        missing = [f for f in CCO_REQUIRED_SYMBOLS if not _has_cco_symbol(names, f)]
        if missing:
            return False, False, f"{lib}: missing symbols {missing}"
        if check_dlopen and not _dlopen_ok(lib):
            return False, False, f"{lib}: dlopen failed"
        sdma = cco_header_has_sdma(env)
        return True, sdma, "ok" if sdma else "cco.hpp lacks the SDMA device API (BUILD_CCO_SDMA / ccoSdmaContext / sdmaQueueCount)"
    except Exception as e:  # noqa: BLE001  (probe contract: never raise)
        return False, False, f"probe error {e!r}"


# ----------------------------------------------------------------------------- public API
def detect_arch() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx()
    except Exception:
        return "unknown"


def detect_cu() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return int(get_cu_num())
    except Exception:
        return 0


def detect_capabilities(group=None, arch: str = None, env=None, check_dlopen: bool = True) -> FrozenSet[str]:
    """Capabilities usable by variants' Contract.requires. ``group`` is accepted for API symmetry; no collective is run here."""
    caps = set()
    arch = arch or detect_arch()
    if arch.startswith("gfx950"):
        caps.add("gfx950_sc_bypass")
    cco, sdma, _why = probe_mori_cco(env, check_dlopen)
    if cco:
        caps.add("mori_cco")
        if sdma:
            caps.add("sdma")
    return frozenset(caps)
