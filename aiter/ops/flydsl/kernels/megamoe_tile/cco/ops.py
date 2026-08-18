# SPDX-License-Identifier: MIT
"""Legacy/RAIL-async FlyDSL bridge over CCO GDA.

Canonical MegaMoE Tile kernels use MORI's public ``Window.lsa_ptr`` for all
node-local data traffic.  This compatibility bridge remains only for APIs the
current public MORI FlyDSL binding does not yet expose: RAIL team selection,
runtime QP contexts, aggregate-without-doorbell posting, asynchronous flush
requests, and arbitrary-address generation waits.
"""

from __future__ import annotations

from .bitcode import get_bitcode_path


class _LazyExtern:
    def __init__(self, symbol, args, ret):
        self.symbol, self.args, self.ret = symbol, args, ret
        self.fn = None

    def __call__(self, *args):
        if self.fn is None:
            from flydsl.expr.extern import ffi
            from flydsl.compiler.extern_link import link_extern

            self.fn = link_extern(
                ffi(self.symbol, self.args, self.ret),
                bitcode_path=get_bitcode_path(),
                module_init_fn=None,
            )
        return self.fn(*args)


_U64, _I32 = "uint64", "int32"
_XFER = [_U64, _I32, _I32, _U64, _U64, _U64, _U64, _U64, _I32]
_PUT_VALUE = [_U64, _I32, _I32, _U64, _U64, _U64, _I32]

TEAM_WORLD = "world"
TEAM_RAIL = "rail"
_TEAMS = (TEAM_WORLD, TEAM_RAIL)
_SCOPES = ("warp", "block")


def _key(team, scope):
    if team not in _TEAMS:
        raise ValueError(f"team must be one of {_TEAMS}, got {team!r}")
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    return f"{team}_{scope}"


_PUT = {
    f"{team}_{scope}": _LazyExtern(
        f"aiter_cco_put_{team}_{scope}", _XFER, "void"
    )
    for team in _TEAMS
    for scope in _SCOPES
}
_PUT_VALUE_OP = {
    f"{team}_{scope}": _LazyExtern(
        f"aiter_cco_put_value_{team}_{scope}", _PUT_VALUE, "void"
    )
    for team in _TEAMS
    for scope in _SCOPES
}
_FLUSH_ASYNC = {
    f"{team}_{scope}": _LazyExtern(
        f"aiter_cco_flush_async_{team}_{scope}", [_U64, _I32, _I32], _U64
    )
    for team in _TEAMS
    for scope in _SCOPES
}
_WAIT = {
    s: _LazyExtern(f"aiter_cco_wait_request_{s}", [_U64, _I32, _U64], "void")
    for s in _SCOPES
}
_WAIT_READY = _LazyExtern(
    "aiter_cco_wait_u64_ge_system", [_U64, _U64], _U64
)
_LSA_PTR = _LazyExtern("aiter_cco_lsa_ptr", [_U64, _I32, _U64], _U64)


def put(
    dev_comm,
    ctx,
    peer,
    dst_win,
    dst_off,
    src_win,
    src_off,
    nbytes,
    *,
    aggregate=True,
    scope="warp",
    team=TEAM_WORLD,
):
    return _PUT[_key(team, scope)](
        dev_comm,
        ctx,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        int(aggregate),
    )


def put_value(
    dev_comm,
    ctx,
    peer,
    dst_win,
    dst_off,
    value,
    *,
    aggregate=True,
    scope="warp",
    team=TEAM_WORLD,
):
    return _PUT_VALUE_OP[_key(team, scope)](
        dev_comm, ctx, peer, dst_win, dst_off, value, int(aggregate)
    )


def flush_async(dev_comm, ctx, peer, *, scope="warp", team=TEAM_WORLD):
    return _FLUSH_ASYNC[_key(team, scope)](dev_comm, ctx, peer)


def wait_request(dev_comm, ctx, request, *, scope="warp"):
    return _WAIT[scope](dev_comm, ctx, request)


def wait_ready(address, expected):
    """Poll a monotonic u64 ready/credit word with system-acquire semantics."""

    return _WAIT_READY(address, expected)


def lsa_ptr(window, peer_lsa_rank, offset=0):
    """Return a peer LSA rank's directly addressable VA in a CCO window."""

    return _LSA_PTR(window, peer_lsa_rank, offset)
