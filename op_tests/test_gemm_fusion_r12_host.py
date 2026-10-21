# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the R12 host API (design card r2): the ERR/STEP/DIAG page, the async readback ring, the collective reset(), the peer-ERR table and the static call_id
uniqueness check (KEY gate). A fake backend / fake torch stand in for the device. Run: python -m pytest op_tests/test_gemm_fusion_r12_host.py
"""

import logging
import os
import sys
import types
from types import SimpleNamespace as NS

import pytest

try:
    import aiter.gemm_fusion as gf
except Exception:  # noqa: BLE001
    # no torch / GPU: load the pure-Python subpackage without aiter/__init__
    for _k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[_k]
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aiter")
    _pkg = types.ModuleType("aiter")
    _pkg.__path__ = [os.path.abspath(_root)]
    sys.modules["aiter"] = _pkg
    import aiter.gemm_fusion as gf  # noqa: F401

from aiter.gemm_fusion import callid, catalog, errpage
from aiter.gemm_fusion import profile as profmod
from aiter.gemm_fusion.plan import WindowReq
from aiter.gemm_fusion.resources import ResourceError
from aiter.ops.gemm_fusion.sp_qkv import specs
from aiter.ops.gemm_fusion.sp_qkv.test_utils import (
    ArenaBackend,
    FakeDist,
    RealisticFirstFit,
    clear_profile_cache,
    make_res,
)

PROFILE = "minimax_m3_sp4_gfx950_v1"
MiB = 1 << 20


# ----------------------------------------------------------------------------------------------------------------------- ERR word
def test_err_word_encode_decode_roundtrip_and_names():
    for reason, origin, kind, prop in [
        (0x20, 0, 1, False),
        (0x30, 3, 1, True),
        (2, 1, 2, False),
        (0x32, 2, 2, True),
        (0x41, 0, 1, False),
    ]:
        w = errpage.encode_err(reason, origin, kind, prop)
        e = errpage.decode_err(w)
        assert (e.reason, e.origin_rank, e.kind, e.propagated, e.word) == (
            reason,
            origin,
            kind,
            prop,
            w,
        )
        assert not e.ok and e.kind_name in ("small", "large")
    # the card's field positions: reason[7:0] | origin[9:8] | kind[11:10] | propagated[12]
    assert errpage.encode_err(0x20, 2, 1, True) == 0x20 | 2 << 8 | 1 << 10 | 1 << 12
    assert errpage.decode_err(0).ok and errpage.decode_err(0).describe() == "ok"
    assert (
        errpage.decode_err(errpage.encode_err(0x30, 1, 1)).reason_name == "TAG_MISMATCH"
    )
    assert (
        errpage.decode_err(errpage.encode_err(0x20, 1, 1)).reason_name
        == "B_POLL_TIMEOUT"
    )
    assert (
        errpage.decode_err(errpage.encode_err(0x42, 0, 1)).reason_name
        == "CONTRACT(0x42)"
    )
    assert "W2_TIMEOUT" in errpage.decode_err(errpage.encode_err(2, 0, 2)).reason_name
    d = errpage.decode_err(errpage.encode_err(0x30, 3, 1, True)).describe()
    assert "small:TAG_MISMATCH" in d and "origin_rank=3" in d and "propagated=1" in d
    for bad in (
        {"reason": 256, "origin_rank": 0, "kind": 1},
        {"reason": 1, "origin_rank": 4, "kind": 1},
        {"reason": 1, "origin_rank": 0, "kind": 4},
    ):
        with pytest.raises(ValueError):
            errpage.encode_err(**bad)
    assert errpage.decode_err(0x1_0000_0020).word == 0x20  # 32-bit word


def test_global_page_layout_constants_match_the_card_and_the_large_driver():
    assert (
        errpage.ERR_OFF,
        errpage.STEP_OFF,
        errpage.DIAG_OFF,
        errpage.DIAG_WORDS,
    ) == (0, 8, 64, 8)
    assert (
        errpage.GLOBAL_PAGE_BYTES == 4 << 20
        and errpage.SHARE_GLOBAL == "gf.global"
        and errpage.READBACK_SLOTS >= 2
    )
    pg = errpage.GlobalPage(0x7F00_0000_0000, errpage.GLOBAL_PAGE_BYTES, 3)
    assert (pg.err_addr, pg.step_addr, pg.diag_addr) == (
        0x7F00_0000_0000,
        0x7F00_0000_0008,
        0x7F00_0000_0040,
    )
    with pytest.raises(ValueError):
        errpage.GlobalPage(0x1000, 64, 0)  # smaller than its own layout
    # the delivered large/r12 header agrees (LG_GF_*_OFF, LGC_CTAG/ATAG): read from the source, not retyped
    root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "csrc",
        "gemm_fusion",
        "sp_qkv",
        "large",
        "r12",
        "interface",
        "lg_comm_abi_v3.h",
    )
    if os.path.exists(root):
        with open(root, encoding="utf-8") as f:
            text = f.read()
        import re

        def val(name):
            m = re.search(rf"#define\s+{name}\s+(\d+)", text) or re.search(
                rf"{name}\s*=\s*(\d+)", text
            )
            assert m, name
            return int(m.group(1))

        assert (val("LG_GF_ERR_OFF"), val("LG_GF_STEP_OFF"), val("LG_GF_DIAG_OFF")) == (
            errpage.ERR_OFF,
            errpage.STEP_OFF,
            errpage.DIAG_OFF,
        )
        from aiter.ops.gemm_fusion.sp_qkv import large_rt

        assert (val("LGC_CTAG_OFF"), val("LGC_ATAG_OFF")) == (
            large_rt.LG_CTAG_OFF,
            large_rt.LG_ATAG_OFF,
        )


def test_large_reason_table_is_pinned_to_the_delivered_header():
    root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "csrc",
        "gemm_fusion",
        "sp_qkv",
        "large",
        "r12",
        "interface",
        "lg_comm_abi_v3.h",
    )
    if not os.path.exists(root):
        pytest.skip("large/r12 header not present")
    import re

    with open(root, encoding="utf-8") as f:
        text = f.read()
    codes = {
        int(v, 0)
        for v in re.findall(r"LGC_POISON_[A-Z0-9_]+\s*=\s*(0x[0-9a-fA-F]+|\d+)", text)
    }
    assert codes == set(errpage.LARGE_REASONS), (
        sorted(codes),
        sorted(errpage.LARGE_REASONS),
    )  # exactly the header's reason codes
    assert errpage.decode_err(errpage.encode_err(0x31, 1, 2)).reason_name.startswith(
        "TAG_CREDIT"
    )
    assert errpage.decode_err(
        errpage.encode_err(0x32, 1, 2, True)
    ).reason_name.startswith("TAG_ARRIVAL")
    assert errpage.decode_err(errpage.encode_err(8, 0, 2)).reason_name.startswith(
        "NO_GPAGE"
    )


def test_devview_is_a_cuda_array_interface_without_copy():
    v = errpage.DevView(0x1234000, (1,), "<i4")
    cai = v.__cuda_array_interface__
    assert (
        cai["data"] == (0x1234000, False)
        and cai["shape"] == (1,)
        and cai["typestr"] == "<i4"
        and cai["strides"] is None
        and cai["version"] == 3
    )


# ----------------------------------------------------------------------------------------------------------------------- readback ring
class FakeRingOps:
    """Records every device interaction: the ring must never synchronise the host."""

    def __init__(self):
        self.log, self.capture = [], False
        self.complete = set()

    def alloc(self, slots):
        self.log.append(("alloc", slots))
        return [0] * slots

    def capturing(self):
        return self.capture

    def copy(self, buf, slot, src, stream):
        self.log.append(("copy", slot, src, stream))
        buf[slot] = int(
            getattr(src, "value", src)
            if not hasattr(src, "cai")
            else getattr(src, "value", 0)
        )  # "the D2H": the value the fake ERR tensor holds
        ev = NS(slot=slot, done=False)
        ev.query = lambda: ev.slot in self.complete
        ev.synchronize = lambda: (
            self.log.append(("event_sync", slot)),
            self.complete.add(ev.slot),
        )
        return ev

    def read(self, buf, slot):
        return buf[slot]


def test_ring_enqueue_slot_is_step_mod_r_and_consume_is_nonblocking_until_complete():
    ops = FakeRingOps()
    ring = errpage.ReadbackRing(slots=4, ops=ops)
    assert (
        ring.enqueue(0, 0x20) == 0
        and ring.enqueue(5, 0x30) == 1
        and ring.enqueue(7, 0) == 3
    )  # slot = step % 4
    assert ring.enqueued == 3 and [e[0] for e in ops.log] == [
        "alloc",
        "copy",
        "copy",
        "copy",
    ]
    assert ring.consume(5) is None  # its copy has not completed: no value, and NO wait
    assert not any(e[0] == "event_sync" for e in ops.log)
    ops.complete.add(1)
    assert (
        ring.consume(5) == 0x30 and ring.consume(5) == 0x30
    )  # consuming does not clear the slot
    ops.complete.update({0, 3})
    assert ring.consume(0) == 0x20 and ring.consume(7) == 0
    with pytest.raises(RuntimeError, match="never enqueued"):
        ring.consume(6)  # slot 2 was never enqueued


def test_ring_overrun_never_enqueued_wait_and_reset():
    ops = FakeRingOps()
    ring = errpage.ReadbackRing(slots=2, ops=ops)
    ring.enqueue(0, 1)
    ring.enqueue(1, 0)
    ring.enqueue(2, 7)  # overwrites slot 0: step 0 is gone
    ops.complete.update({0, 1})
    with pytest.raises(RuntimeError, match="slot 0 holds step 2, not 0"):
        ring.consume(0)
    assert ring.consume(2) == 7
    with pytest.raises(RuntimeError, match="never enqueued|holds step"):
        ring.consume(9)
    ops2 = FakeRingOps()
    r2 = errpage.ReadbackRing(slots=2, ops=ops2)
    r2.enqueue(4, 5)
    assert (
        r2.consume(4, wait=True) == 5 and ("event_sync", 0) in ops2.log
    )  # the explicit opt-in is the ONLY sync
    r2.reset()
    with pytest.raises(RuntimeError):
        r2.consume(4)  # the old readings are forgotten after the collective reset


def test_ring_refuses_capture_and_bad_arguments():
    ops = FakeRingOps()
    ring = errpage.ReadbackRing(slots=2, ops=ops)
    ops.capture = True
    with pytest.raises(RuntimeError, match="never enqueue it inside graph capture"):
        ring.enqueue(1, 0)
    assert ring.enqueued == 0
    ops.capture = False
    with pytest.raises(ValueError):
        ring.enqueue(-1, 0)
    with pytest.raises(ValueError, match="at least 2 slots"):
        errpage.ReadbackRing(slots=1, ops=ops)


# ----------------------------------------------------------------------------------------------------------------------- call_id (KEY gate)
ATOM_VECTORS = [  # computed from the real atom.model_ops.gemm_fusion.consensus.decision_key
    (("model.layers.0.self_attn", 8192, "sp_qkv.large.v1", 1), 1730581148),
    (("model.layers.61.self_attn", 64, "sp_qkv.small.v1", 2), 1012483901),
    (("mtp.layers.0.self_attn", 128, "sp_qkv.small.v1", 2), 319221034),
    (("a", 0, "native", 0), 796906824),
]


def test_decision_key_is_pinned_to_atoms_real_values_and_is_31_bit():
    for key, want in ATOM_VECTORS:
        assert callid.decision_key(*key) == want
        assert 0 <= want < 1 << 31
    assert callid.decision_key("s", 64, "v", 2) == callid.decision_key(
        "s", 64, "v", 2
    )  # process-independent: no salted hash
    assert len({callid.decision_key("s", t, "v", 2) for t in range(1, 500)}) == 499


def test_check_call_ids_detects_collisions_zero_ids_and_ignores_duplicates_of_one_key():
    keys = [
        ("a", 64, "v", 2),
        ("b", 64, "v", 2),
        ("a", 64, "v", 2),
    ]  # the same key twice is ONE key
    rep = callid.check_call_ids(keys)
    assert (
        rep.ok
        and rep.keys == 2
        and rep.refused_keys == frozenset()
        and rep.refused_sites == frozenset()
    )
    tiny = lambda site, tokens, vid, n: (tokens + n) % 4 + 1
    many = [(f"s{i}", 64, "v", 2) for i in range(9)]
    rep = callid.check_call_ids(many, key_fn=tiny)
    assert (
        not rep.ok
        and rep.collisions
        and all(len(ks) >= 2 for ks in rep.collisions.values())
    )
    assert (
        rep.refused_sites
        == frozenset(k[0] for ks in rep.collisions.values() for k in ks)
        == frozenset(f"s{i}" for i in range(9))
    )
    zero = lambda site, tokens, vid, n: (0 if site == "z" else 5 + len(site))
    rep = callid.check_call_ids([("z", 1, "v", 1), ("yy", 1, "v", 1)], key_fn=zero)
    assert (
        not rep.ok
        and rep.zero_id == [("z", 1, "v", 1)]
        and rep.refused_sites == frozenset({"z"})
    )  # call_id 0 == the content of a zeroed tag word


def test_registry_freezes_refuses_colliding_keys_and_logs():
    tiny = lambda site, tokens, vid, n: len(site) % 2 + 1
    reg = callid.CallIdRegistry(key_fn=tiny)
    for s in ("aa", "bb", "c"):
        reg.add(s, 64, "v", 2)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        rep = reg.freeze()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert reg.freeze() is rep  # idempotent
    assert (
        reg.is_refused("aa", 64, "v", 2)
        and reg.is_refused("bb", 64, "v", 2)
        and not reg.is_refused("c", 64, "v", 2)
    )
    assert reg.call_id("c", 64, "v", 2) == 2
    with pytest.raises(ValueError, match="refused.*run site 'aa' natively"):
        reg.call_id("aa", 64, "v", 2)
    with pytest.raises(KeyError):
        reg.call_id("never", 1, "v", 1)
    with pytest.raises(RuntimeError, match="frozen"):
        reg.add("late", 1, "v", 1)
    assert any(
        "event=call_id collision" in m and "fusion refused for sites ['aa', 'bb']" in m
        for m in msgs
    )


MODEL_SITES = [f"model.layers.{i}.self_attn" for i in range(62)]
DRAFT_SITES = [
    f"mtp.layers.{i}.self_attn" for i in range(3)
]  # draft/MTP forwards use DISTINCT site names (card D-8(c))


def test_KEY_every_profile_key_of_the_shipped_profile_has_a_distinct_nonzero_call_id():
    """The KEY gate: all (site, tokens, vid, n_fused) the shipped profile can dispatch (exact rows AND every m a rule covers) x the model's and the draft's fused sites map
    to distinct non-zero 31-bit call_ids."""
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    keys, unresolved = callid.profile_call_id_keys(
        prof, MODEL_SITES + DRAFT_SITES, world=4
    )
    assert unresolved == []
    tokens = {k[1] for k in keys}
    assert {64, 128, 8192, 16384, 32768} <= tokens  # exact rows (m * world) ...
    assert {
        4096,
        5120,
        12288,
        28672,
    } <= tokens  # ... and the m values the range rules add (m = 1024..8192 step 256, tokens = 4 m)
    n_entries = len({(k[2], k[1], k[3]) for k in keys})
    # the key has no mode: dense and indexer at the same token count are the same key (a site is one mode). 2 small token counts + 29 large (m = 1024..8192 step 256)
    assert (
        len(keys) == n_entries * len(MODEL_SITES + DRAFT_SITES) and n_entries == 2 + 29
    )
    rep = callid.check_profile_call_ids(prof, MODEL_SITES + DRAFT_SITES, world=4)
    assert (
        rep.ok
        and rep.keys == len(keys)
        and rep.refused_sites == frozenset()
        and rep.unresolved == []
    )
    ids = {callid.decision_key(*k) for k in keys}
    assert (
        len(ids) == len(keys) and 0 not in ids
    )  # distinct and non-zero, checked directly as well
    # n_fused is the stage count of the variant: small rows 2, large rows 1
    assert {(k[2], k[3]) for k in keys} == {
        ("sp_qkv.small.v1", 2),
        ("sp_qkv.large.v1", 1),
    }


def test_profile_check_reports_unregistered_variants_and_logs_collisions(monkeypatch):
    prof = profmod.parse_profile(
        {
            "profile": "t",
            "rows": [
                {
                    "sig": "s",
                    "arch": "gfx950",
                    "cu": 256,
                    "world": 4,
                    "m": 16,
                    "vid": "nope.v1",
                    "params": {},
                }
            ],
            "rules": [],
        }
    )
    rep = callid.check_profile_call_ids(prof, ["x"])
    assert (
        rep.unresolved == ["nope.v1: variant not registered"]
        and rep.keys == 0
        and rep.ok
    )
    # a world filter drops other worlds' entries
    prof2 = profmod.parse_profile(
        {
            "profile": "t",
            "rows": [
                {
                    "sig": "s",
                    "arch": "gfx950",
                    "cu": 256,
                    "world": 8,
                    "m": 16,
                    "vid": "nope.v1",
                    "params": {},
                }
            ],
            "rules": [],
        }
    )
    assert callid.profile_call_id_keys(prof2, ["x"], world=4) == ([], [])


# ----------------------------------------------------------------------------------------------------------------------- FusionResources integration
def _fake_torch(monkeypatch, log=None):
    tm = types.ModuleType("torch")
    made = []

    def as_tensor(obj, device=None):
        t = NS(obj=obj, device=device, cai=obj.__cuda_array_interface__)
        made.append(t)
        return t

    tm.as_tensor = as_tensor
    tm.int64 = "i64"
    tm.tensor = lambda addrs, dtype=None, device=None: NS(
        addrs=list(addrs), device=device, data_ptr=lambda: 0xDEAD0000
    )
    monkeypatch.setitem(sys.modules, "torch", tm)
    return made


def _page_reqs(owner="p"):
    return [
        WindowReq(owner + "/a", 1000, "p2p_window"),
        WindowReq(
            owner + "/gpage",
            errpage.GLOBAL_PAGE_BYTES,
            "control",
            share=errpage.SHARE_GLOBAL,
            min_bytes=errpage.GLOBAL_PAGE_BYTES,
        ),
    ]


def _budget_profile(window_bytes=8 * MiB):
    """A profile whose arena budget has a small shared set + the global page."""
    return profmod.parse_profile(
        {
            "profile": "t",
            "rows": [],
            "window_budget": {
                "window_bytes": window_bytes,
                "max_windows": 6,
                "shared": {
                    "S": 4 * MiB,
                    errpage.SHARE_GLOBAL: errpage.GLOBAL_PAGE_BYTES,
                },
                "small_window_sizes": [4 * MiB],
            },
        }
    )


def test_page_is_one_shared_view_under_arena_and_one_extra_window_under_uniform(
    monkeypatch,
):
    for var in (
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_ARENA_MIB",
        "AITER_GEMM_FUSION_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    # arena: two owners both request the page; ONE shared view exists
    be = ArenaBackend()
    r = make_res(dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena")
    with pytest.raises(ResourceError, match="no global ERR/STEP page"):
        r.global_page()
    w1 = r.ensure("o1", _page_reqs("o1"))
    w2 = r.ensure("o2", [WindowReq("o2/a", 1000, "p2p_window"), _page_reqs("o2")[1]])
    assert (
        w1["o1/gpage"] is w2["o2/gpage"]
        and w1["o1/gpage"].shared
        and w1["o1/gpage"].bytes == errpage.GLOBAL_PAGE_BYTES
    )
    assert (
        r.run_bindings()["global_page"] is True and len(be.regs) == 1
    )  # still ONE registration
    page = r.global_page()
    assert (
        page.local == w1["o1/gpage"].local
        and page.bytes == errpage.GLOBAL_PAGE_BYTES
        and r.global_page() is page
    )
    # uniform: the page is ONE more equal-size window, however many plans request it
    be2 = RealisticFirstFit()
    r2 = make_res(
        dist=FakeDist(), backend=be2, profile=_budget_profile(), policy="uniform"
    )
    a = r2.ensure("o1", _page_reqs("o1"))
    b = r2.ensure("o2", [WindowReq("o2/a", 1000, "p2p_window"), _page_reqs("o2")[1]])
    assert a["o1/gpage"] is b["o2/gpage"] and a["o1/gpage"].bytes == r2.window_bytes
    assert len(r2._regs) == 3  # o1/a, o2/a and ONE page
    r2.finalize()  # zeroed with everything else
    assert r2.pending_zero == 0 and any(
        x[0] == "zero" and x[1] == a["o1/gpage"].local for x in be2.log
    )


def test_tensors_are_zero_copy_views_with_stable_identity(monkeypatch):
    made = _fake_torch(monkeypatch)
    be = ArenaBackend()
    r = (
        make_res(
            dist=FakeDist(),
            backend=be,
            profile=_budget_profile(),
            policy="arena",
            device=2,
        )
        if False
        else make_res(
            dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena"
        )
    )
    r.device = 2
    r.ensure("o1", _page_reqs("o1"))
    base = r.global_page().local
    e, s = r.error_flag(), r.step_tensor()
    assert (
        r.error_flag() is e and r.step_tensor() is s
    )  # the SAME tensor objects: ATOM captures the step tick, the address never moves
    assert (
        e.cai["data"][0] == base
        and e.cai["shape"] == (1,)
        and e.cai["typestr"] == "<i4"
        and e.device == "cuda:2"
    )
    assert s.cai["data"][0] == base + 8 and s.cai["typestr"] == "<i8"
    d = r.global_page().diag_tensor()
    assert d.cai["data"][0] == base + 64 and d.cai["shape"] == (8,)
    assert len(made) == 3


def test_readback_through_fusion_resources_is_one_async_copy_per_step_and_never_syncs(
    monkeypatch,
):
    _fake_torch(monkeypatch)
    ops = FakeRingOps()
    be = ArenaBackend()
    r = make_res(dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena")
    r.readback_ops = ops
    with pytest.raises(ResourceError, match="no global ERR/STEP page"):
        r.readback_enqueue(0)
    r.ensure("o1", _page_reqs("o1"))
    err = r.error_flag()
    for step in range(5):
        assert r.readback_enqueue(step, stream="S") == step % errpage.READBACK_SLOTS
    copies = [e for e in ops.log if e[0] == "copy"]
    assert len(copies) == 5 and all(
        c[2] is err and c[3] == "S" for c in copies
    )  # exactly ONE 4 B copy per step, of the ERR tensor, on the given stream
    assert r.readback_consume(3) is None
    ops.complete.add(3)
    assert r.readback_consume(3) == 0
    assert (
        not any(e[0] == "event_sync" for e in ops.log)
        and r.run_bindings()["readback_slots"] == errpage.READBACK_SLOTS
    )


def test_global_peers_table_holds_every_ranks_err_address_own_entry_local(monkeypatch):
    _fake_torch(monkeypatch)
    be = ArenaBackend()
    be.peer_ptr = lambda comm, local, peer: local + 0x1000_0000 * (
        peer + 1
    )  # what ccoGetPeerPtr would return for the SAME offset in rank `peer`'s window
    r = make_res(
        rank=1, dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena"
    )
    r.ensure("o1", _page_reqs("o1"))
    seen = []
    r.gpeers_factory = lambda addrs, dev: (
        seen.append((list(addrs), dev)),
        NS(data_ptr=lambda: 0xABCD),
    )[1]
    t = r.global_peers()
    local = r.global_page().local
    addrs, dev = seen[0]
    assert dev == r.device and len(addrs) == 4
    assert (
        addrs[1] == local
    )  # own entry = this rank's ERR word (ignored by the kernels)
    assert [addrs[i] for i in (0, 2, 3)] == [
        local + 0x1000_0000 * (i + 1) for i in (0, 2, 3)
    ]
    assert (
        r.global_peers() is t and len(seen) == 1
    )  # built ONCE, kept for the comm's lifetime
    # default factory: a device int64 tensor
    r.gpeers_factory, r._gpeers = None, None
    t2 = r.global_peers()
    assert t2.addrs == addrs and t2.device == f"cuda:{r.device}"


def test_collective_reset_logs_the_decoded_error_then_zeroes_everything_and_forgets_the_ring(
    monkeypatch,
):
    _fake_torch(monkeypatch)
    ops = FakeRingOps()
    be, d = ArenaBackend(), FakeDist()
    r = make_res(dist=d, backend=be, profile=_budget_profile(), policy="arena")
    r.readback_ops = ops
    r.ensure("o1", _page_reqs("o1"))
    r.finalize()
    word = errpage.encode_err(0x30, 2, 1, True)
    r.global_page().read = lambda: (word, 17, [0x111, 0x222, 0, 0, 0, 0, 0, 0])
    r.set_report_hook("large", lambda: {"epoch": 3, "poison": 0})
    inits = []
    r.set_init("o1", lambda: inits.append("init"))
    r.readback_enqueue(4)
    ops.complete.add(4)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    be.log.clear()
    d.calls.clear()
    inits.clear()
    try:
        r.reset()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    page = [m for m in msgs if "event=resources error_page" in m]
    assert (
        len(page) == 1
        and "where=reset" in page[0]
        and "step=17" in page[0]
        and "small:TAG_MISMATCH" in page[0]
        and "origin_rank=2" in page[0]
        and "propagated=1" in page[0]
    )
    assert "0x111" in page[0]
    assert any("event=resources error_report" in m and "'epoch': 3" in m for m in msgs)
    kinds = [x[0] for x in be.log]
    assert kinds == ["sync", "zero", "sync"] and d.calls == [
        ("barrier",),
        ("barrier",),
    ]  # sync -> barrier -> zero ALL windows (the arena, incl. the page) -> init -> sync -> barrier
    assert (
        be.log[1][1] == r._arena.local and be.log[1][2] == r._arena.bytes
    )  # the page lives inside the arena: ERR and STEP return to 0 with it
    assert inits == ["init"]
    with pytest.raises(RuntimeError):
        r.readback_consume(4)  # the old readings describe a state that no longer exists
    assert any("event=resources reset rank=0" in m for m in msgs)


def test_clean_reset_logs_no_error_page_and_the_legacy_name_still_works(monkeypatch):
    _fake_torch(monkeypatch)
    be = ArenaBackend()
    r = make_res(dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena")
    r.ensure("o1", _page_reqs("o1"))
    r.finalize()
    r.global_page().read = lambda: (0, 5, [0] * 8)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        r.quiesce_and_reset()  # the pre-R12 name: same collective, the log says where it was called from
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert not [m for m in msgs if "error_page" in m] and any(
        "event=resources quiesce_and_reset" in m for m in msgs
    )
    # a failing diagnostics read never blocks the recovery
    r.global_page().read = lambda: (_ for _ in ()).throw(RuntimeError("device gone"))
    r.reset()


def test_r12_plans_register_no_per_plan_poison_and_poison_local_is_zero(monkeypatch):
    _fake_torch(monkeypatch)
    r = make_res(
        dist=FakeDist(),
        backend=ArenaBackend(),
        profile=_budget_profile(),
        policy="arena",
    )
    r.ensure("o1", _page_reqs("o1"))
    assert (
        r._poison_hooks == {} and r.poison_local() == 0
    )  # nothing on the per-step path: the ERR word is read through the readback ring only
    assert r.run_bindings()["global_page"] is True


def test_catalog_untouched_by_these_tests():
    assert isinstance(catalog._CATALOG, dict) and specs.LARGE_PROFILE_SRC == "r12"
