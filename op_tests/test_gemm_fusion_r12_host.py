# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the R12 host API (design card r2): the ERR/STEP/DIAG page, the async readback ring, the collective reset(), the peer-ERR table and the static call_id
uniqueness check (KEY gate). A fake backend / fake torch stand in for the device. Run: python -m pytest op_tests/test_gemm_fusion_r12_host.py
"""

import json
import logging
import os
import re
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
    import aiter.gemm_fusion as gf

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
    assert {
        4,
        8,
        16,
        32,
        48,
        64,
        128,
        256,
        512,
        8192,
        16384,
        32768,
    } <= tokens  # exact rows (m * world): P2 (m 1..12), m16, m32, r20 m64, r21 m128, large ...
    assert {
        132,
        148,
        252,
        260,
        400,
        508,
        516,
        520,
        1020,
    } <= tokens  # ... the r20 (MP64) rule over local m 33..64 (tokens 4 m = 132..256), the r22 (MP128) rule over 65..128 (260..512), the r23 (MP256) rule over 129..255 (516..1020) ...
    assert (
        3072 in tokens and 2048 in tokens
    )  # ... both r18 large pad buckets below the swept grid (exact rows) ...
    assert not (
        {1028, 2052, 3076, 4092} & tokens
    )  # ... and nothing off the 256 grid of the exact rows / rules (bucket_for_m pads those: the pad universe below)
    assert {
        4096,
        5120,
        12288,
        28672,
    } <= tokens  # ... and the m values the range rules add (m = 1024..8192 step 256, tokens = 4 m)
    n_entries = len({(k[2], k[1], k[3]) for k in keys})
    # the key has no mode: dense and indexer at the same token count are the same key (a site is one mode). 256 small token counts (every local m 1..256: the exact rows plus
    # the r19 rules over 1..16 / 17..32, the MP64 rule over 33..64, the MP128 rule over 65..128, the MP256 rule over 129..255 and its bucket row at 256) + 31 large (m = 1024..8192 step 256 and the two r18 pad buckets 512 / 768)
    assert (
        len(keys) == n_entries * len(MODEL_SITES + DRAFT_SITES)
        and n_entries == 256 + 31
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
    r.err_reader = (
        lambda: 0
    )  # the verification read after the zeroing (a device read in production)
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
        ("allgather",),
        ("barrier",),
        ("allgather",),
    ]  # ... + the FINAL barrier and the re-read after a clean first verification  # sync -> barrier -> zero ALL windows (the arena, incl. the page) -> init -> sync -> barrier -> verify (all-gather of the ERR words)
    assert (
        be.log[1][1] == r._arena.local and be.log[1][2] == r._arena.bytes
    )  # the page lives inside the arena: ERR and STEP return to 0 with it
    assert inits == ["init"]
    with pytest.raises(RuntimeError):
        r.readback_consume(4)  # the old readings describe a state that no longer exists
    assert any("event=resources reset rank=0" in m and "rounds=1" in m for m in msgs)


def test_clean_reset_logs_no_error_page_and_the_legacy_name_still_works(monkeypatch):
    _fake_torch(monkeypatch)
    be = ArenaBackend()
    r = make_res(dist=FakeDist(), backend=be, profile=_budget_profile(), policy="arena")
    r.err_reader = lambda: 0
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


# ----------------------------------------------------------------------------------------------------------------------- reset(): verified, multi-round
GATHER_TAG = "gf_errpage"


class _ResetRig:
    """A FusionResources with a global page, a scriptable page (ERR / STEP / DIAG) and a shared event log of the backend / dist calls.
    late       = ERR words that LAND after the zeroing of successive rounds (a peer's propagation write still in flight): late[0] becomes visible at the next device sync.
    reread     = ERR words that land AFTER the first verification read (visible only to the re-read after the final barrier).
    page_script= (ERR, STEP, DIAG4) this rank reads at successive verification reads, the last one repeating (default: all zero): stale STEP / DIAG after the zero.
    peers      = what the other ranks report at successive gathers ({rank: ERR or (ERR, STEP, DIAG4)}), the last one repeating.
    in_flight  = an ERR a kernel writes while the first device sync drains.
    kernel     = zero with a device-side fill (events 'kernel_zero') instead of the H2D memcpy ('zero').
    """

    def __init__(
        self,
        monkeypatch,
        late=(),
        peers=(),
        reread=(),
        page_script=(),
        max_rounds=None,
        in_flight=0,
        page=True,
        kernel=False,
    ):
        _fake_torch(monkeypatch)
        monkeypatch.delenv("AITER_GEMM_FUSION_RESET_ZERO", raising=False)
        self.events, self.err, self.pending = [], 0, None
        self.zero_rounds = 0
        self.late, self._late_script = [], list(late)
        self.reread = list(reread)
        self.page_script, self.peers, self.gathers, self.reads = (
            list(page_script),
            list(peers),
            0,
            0,
        )
        self.fills = []
        be = ArenaBackend()
        ev = self.events
        real_zero, real_sync = be.zero, be.device_sync

        def cleared():
            self.err = 0
            self.zero_rounds += 1
            if self.late:
                self.pending = self.late.pop(
                    0
                )  # lands AFTER the zero, before the barrier completes

        def zero(device, local, nbytes):
            ev.append("zero")
            real_zero(device, local, nbytes)
            cleared()

        def sync(device):
            ev.append("sync")
            real_sync(device)
            if self.pending is not None:
                self.err, self.pending = (
                    self.pending,
                    None,
                )  # the in-flight remote write lands

        be.zero, be.device_sync = zero, sync

        def gather_hook(mine):
            if not (
                isinstance(mine, tuple) and mine and mine[0] == GATHER_TAG
            ):  # the registration-agreement gathers of ensure(): every rank agrees
                return [mine] * 4
            self.gathers += 1
            peer = (
                self.peers[min(self.gathers - 1, len(self.peers) - 1)]
                if self.peers
                else {}
            )
            out = []
            for r in range(4):
                if r == 0:
                    out.append(mine)
                else:
                    v = peer.get(r, 0)
                    out.append(
                        (GATHER_TAG, v, 0, 0, 0, 0, 0)
                        if isinstance(v, int)
                        else (GATHER_TAG, v[0], v[1], *v[2])
                    )
            return out

        d = FakeDist(gather_hook=gather_hook)
        bar, gat = d.barrier, d.all_gather_object

        def gather(out, obj, group=None):
            ev.append("allgather")
            gat(out, obj, group)
            if (
                self.reread
                and isinstance(obj, tuple)
                and obj[:1] == (GATHER_TAG,)
                and not obj[1]
            ):
                self.err = self.reread.pop(
                    0
                )  # lands right AFTER this (clean) read: only the re-read after the final barrier can see it

        d.barrier = lambda group=None: (ev.append("barrier"), bar(group))[1]
        d.all_gather_object = gather
        self.dist = d
        self.r = make_res(dist=d, backend=be, profile=_budget_profile(), policy="arena")
        if page:
            self.r.ensure("o1", _page_reqs("o1"))
        else:
            self.r.ensure("o1", [WindowReq("o1/a", 1000, "p2p_window")])
        self.r.finalize()

        def reader():
            self.reads += 1
            if self.page_script:
                w = self.page_script[min(self.reads - 1, len(self.page_script) - 1)]
                return w[0] or self.err, w[1], w[2]
            return self.err, 0, (0, 0, 0, 0)

        self.r.page_reader = reader
        if kernel:

            def fill(device, local, nbytes):
                ev.append("kernel_zero")
                self.fills.append((device, local, nbytes))
                cleared()

            self.r.zero_fill = fill
            self.r.zero_method = "kernel"
        else:
            self.r.zero_method = "h2d"
        self.inits = []
        self.r.set_init("o1", lambda: self.inits.append("init"))
        if max_rounds is not None:
            self.r.reset_max_rounds = max_rounds
        ev.clear()
        self.inits.clear()
        self.zero_rounds, self.reads = 0, 0
        self.late = self._late_script
        self._drain_write = in_flight
        if in_flight:
            first = {"done": False}
            sync_now = be.device_sync

            def sync_with_kernel(device):
                sync_now(device)
                if not first["done"]:
                    first["done"] = True
                    self.err = (
                        self._drain_write
                    )  # a kernel still waiting out its budget finishes during the drain and writes ERR

            be.device_sync = sync_with_kernel


class _Logs:
    def __init__(self):
        self.msgs = []
        outer = self

        class H(logging.Handler):
            def emit(self, rec):
                outer.msgs.append(rec.getMessage())

        self.h, self.lg = H(), logging.getLogger("aiter.gemm_fusion")
        self.old = self.lg.level

    def __enter__(self):
        self.lg.addHandler(self.h)
        self.lg.setLevel(logging.INFO)
        return self

    def __exit__(self, *a):
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.old)


CLEAN_TAIL = [
    "allgather",
    "barrier",
    "allgather",
]  # verify: gather, the FINAL barrier, the re-read


def test_reset_clean_is_one_round_in_the_specified_order_h2d(monkeypatch):
    rig = _ResetRig(monkeypatch)
    with _Logs() as lg:
        rig.r.reset()
    # (1) drain: device sync; (2) barrier; (3) zero; (4) sync + barrier; (5) verify: gather, final barrier, re-read
    assert rig.events == ["sync", "barrier", "zero", "sync", "barrier", *CLEAN_TAIL]
    assert (
        rig.zero_rounds == 1
        and rig.r.last_reset_rounds == 1
        and rig.inits == ["init"]
        and rig.gathers == 2
    )
    assert any(
        "event=resources reset rank=0" in m and "rounds=1" in m and "zero=h2d" in m
        for m in lg.msgs
    )
    assert not any("reset_retry" in m for m in lg.msgs)


def test_reset_with_the_device_side_zero_does_not_use_the_h2d_memcpy(monkeypatch):
    rig = _ResetRig(monkeypatch, kernel=True)
    rig.r.reset()
    # the device fill, ONE device sync right after it, then the usual (4) sync + barrier and the verification
    assert rig.events == [
        "sync",
        "barrier",
        "kernel_zero",
        "sync",
        "sync",
        "barrier",
        *CLEAN_TAIL,
    ]
    assert "zero" not in rig.events
    arena = rig.r._arena
    assert rig.fills == [
        (rig.r.device, arena.local, arena.bytes)
    ]  # the whole arena (the page is inside it) in one fill
    assert rig.inits == ["init"] and rig.r.last_reset_rounds == 1


def test_zero_method_resolution_auto_env_override_and_bad_values(monkeypatch):
    rig = _ResetRig(monkeypatch)
    r = rig.r
    r.zero_method, r.zero_fill = "auto", None
    assert r._resolve_zero_method() == "h2d"  # no torch / no device: MORI's memcpy
    r.zero_fill = lambda d, a, n: None
    assert r._resolve_zero_method() == "kernel"  # a device fill is available
    monkeypatch.setenv("AITER_GEMM_FUSION_RESET_ZERO", "h2d")
    assert r._resolve_zero_method() == "h2d"  # the env override wins
    monkeypatch.setenv("AITER_GEMM_FUSION_RESET_ZERO", "bogus")
    with pytest.raises(ResourceError, match="not in"):
        r._resolve_zero_method()
    monkeypatch.delenv("AITER_GEMM_FUSION_RESET_ZERO")
    r.zero_method, r.zero_fill = "kernel", None
    assert (
        r._resolve_zero_method() == "kernel"
    )  # explicit: the torch fill is used (and fails loudly without torch)


def test_a_late_propagated_write_after_the_zero_triggers_another_round_and_it_is_logged(
    monkeypatch,
):
    word = errpage.encode_err(
        6, 1, 2, True
    )  # reason 6, origin rank 1, propagated (0x1906 = the pattern Evidence saw)
    rig = _ResetRig(monkeypatch, late=[word])
    with _Logs() as lg:
        rig.r.reset()
    assert rig.events == [
        "sync",
        "barrier",  # drain, barrier
        "zero",
        "sync",
        "barrier",
        "allgather",  # round 1: the write lands between the zero and the barrier -> the FIRST read sees it (no re-read needed)
        "zero",
        "sync",
        "barrier",
        *CLEAN_TAIL,  # round 2: zeroed again, clean, final barrier, re-read
    ]
    assert (
        rig.zero_rounds == 2
        and rig.r.last_reset_rounds == 2
        and rig.err == 0
        and rig.inits == ["init", "init"]
    )
    retry = [m for m in lg.msgs if "reset_retry" in m]
    assert (
        len(retry) == 1
        and "round=1/3" in retry[0]
        and "large:" in retry[0]
        and "origin_rank=1" in retry[0]
        and "propagated=1" in retry[0]
        and "step=0" in retry[0]
    )
    assert any("event=resources reset rank=0" in m and "rounds=2" in m for m in lg.msgs)


def test_a_write_landing_after_the_first_read_is_caught_by_the_reread_after_the_final_barrier(
    monkeypatch,
):
    word = errpage.encode_err(
        6, 3, 2, False
    )  # 0xb06: the word that came back in Evidence's run
    assert word == 0xB06
    rig = _ResetRig(monkeypatch, reread=[word])
    with _Logs() as lg:
        rig.r.reset()
    assert rig.events == [
        "sync",
        "barrier",
        "zero",
        "sync",
        "barrier",
        "allgather",
        "barrier",
        "allgather",  # round 1: the first read is clean, the RE-READ after the final barrier is not
        "zero",
        "sync",
        "barrier",
        *CLEAN_TAIL,  # round 2
    ]
    assert rig.r.last_reset_rounds == 2 and rig.err == 0
    retry = [m for m in lg.msgs if "reset_retry" in m]
    assert (
        len(retry) == 1
        and "origin_rank=3" in retry[0]
        and "propagated=0" in retry[0]
        and "large:POSTER_WAIT" in retry[0]
    )


def test_stale_step_or_diag_after_the_zero_is_unclean_even_with_err_zero(monkeypatch):
    stale = (
        0,
        16,
        (0x1_0000_0001, 0x222, 0, 0),
    )  # ERR already 0 but the OLD page content (STEP 16, DIAG) is back: the zero did not stick
    rig = _ResetRig(monkeypatch, page_script=[stale, (0, 0, (0, 0, 0, 0))])
    with _Logs() as lg:
        rig.r.reset()
    assert rig.zero_rounds == 2 and rig.r.last_reset_rounds == 2
    retry = [m for m in lg.msgs if "reset_retry" in m]
    assert (
        len(retry) == 1
        and "step=16" in retry[0]
        and "0x100000001" in retry[0]
        and "0x222" in retry[0]
    )
    # DIAG alone (STEP and ERR zero) also counts, and only DIAG[0..3] are compared
    rig2 = _ResetRig(
        monkeypatch, page_script=[(0, 0, (0, 0, 0, 7)), (0, 0, (0, 0, 0, 0))]
    )
    rig2.r.reset()
    assert rig2.r.last_reset_rounds == 2


def test_a_peers_late_write_repeats_the_round_on_every_rank_the_decision_is_collective(
    monkeypatch,
):
    word = errpage.encode_err(0x31, 2, 2, True)
    rig = _ResetRig(
        monkeypatch, peers=[{2: word}, {}]
    )  # THIS rank is clean; rank 2's word is not after round 1
    rig.r.reset()
    assert (
        rig.zero_rounds == 2 and rig.r.last_reset_rounds == 2 and rig.gathers == 3
    )  # round 1: one gather (unclean), round 2: gather + re-read
    rig_stale = _ResetRig(
        monkeypatch, peers=[{1: (0, 9, (0, 0, 0, 0))}, {}]
    )  # a peer's stale STEP is enough
    rig_stale.r.reset()
    assert rig_stale.r.last_reset_rounds == 2
    # every rank gets the SAME gathered dict (it is what makes the repeat decision uniform)
    rig3 = _ResetRig(monkeypatch, peers=[{1: 5}])
    assert rig3.r._gather_err_words() == {0: 0, 1: 5, 2: 0, 3: 0}
    assert rig3.r._gather_page_words()[1] == (5, 0, (0, 0, 0, 0))


def test_persistent_errors_fail_fatally_after_k_rounds_with_the_decoded_words(
    monkeypatch,
):
    w_small = errpage.encode_err(0x30, 1, 1, True)
    w_large = errpage.encode_err(6, 3, 2, True)
    rig = _ResetRig(
        monkeypatch, late=[w_large] * 10, peers=[{1: (w_small, 4, (0x11, 0, 0, 0))}]
    )
    with _Logs() as lg, pytest.raises(gf.ResetError) as ei:
        rig.r.reset()
    e = ei.value
    assert rig.zero_rounds == 3 and rig.r.last_reset_rounds == 3  # K = 3 by default
    assert (
        e.words[1] == w_small and e.words[0] == w_large
    )  # ATOM's contract: {rank: ERR word}
    assert e.details[1] == (w_small, 4, (0x11, 0, 0, 0)) and e.details[0][0] == w_large
    msg = str(e)
    assert (
        "after 3 zeroing round(s)" in msg
        and "rank 1: small:TAG_MISMATCH" in msg
        and "step=4" in msg
        and "0x11" in msg
        and "rank 0: large:" in msg
        and "origin_rank=3" in msg
    )
    assert isinstance(e, gf.ResourceError)
    assert sum("reset_retry" in m for m in lg.msgs) == 3 and any(
        "round=3/3" in m for m in lg.msgs
    )
    assert not any(
        "event=resources reset rank=0" in m for m in lg.msgs
    )  # no success line
    rig1 = _ResetRig(monkeypatch, late=[w_large] * 10, max_rounds=1)
    with pytest.raises(gf.ResetError, match="after 1 zeroing round"):
        rig1.r.reset()
    assert rig1.zero_rounds == 1
    rig5 = _ResetRig(monkeypatch, late=[w_large] * 4, max_rounds=5)
    rig5.r.reset()
    assert (
        rig5.zero_rounds == 5 and rig5.r.last_reset_rounds == 5
    )  # four late writes, the fifth round is clean


def test_a_kernel_still_running_at_the_start_is_drained_before_anyone_zeroes(
    monkeypatch,
):
    """A kernel inside its wait budget finishes (and propagates ERR) during the drain; the zero that follows the barrier clears it: ONE round suffices."""
    word = errpage.encode_err(6, 1, 2, True)
    rig = _ResetRig(monkeypatch, in_flight=word)
    rig.r.set_drain("o1", lambda: rig.events.append("drain"))
    rig.events.clear()
    rig.r.reset()
    assert rig.events[:4] == ["sync", "drain", "sync", "barrier"] and rig.events.index(
        "zero"
    ) > rig.events.index("barrier")
    assert rig.zero_rounds == 1 and rig.err == 0 and rig.r.last_reset_rounds == 1


def test_without_a_global_page_there_is_one_round_and_no_verification(monkeypatch):
    rig = _ResetRig(monkeypatch, page=False)
    rig.r.reset()
    assert (
        rig.events == ["sync", "barrier", "zero", "sync", "barrier"]
        and rig.reads == 0
        and rig.r.last_reset_rounds == 1
    )
    rig.events.clear()
    rig.r.quiesce_and_reset()  # the legacy name is the same collective
    assert rig.events == ["sync", "barrier", "zero", "sync", "barrier"]


def test_an_unreadable_page_is_never_taken_for_zero_and_words_are_masked(monkeypatch):
    rig = _ResetRig(monkeypatch)
    rig.r.page_reader = lambda: (_ for _ in ()).throw(RuntimeError("device gone"))
    with pytest.raises(RuntimeError, match="device gone"):
        rig.r.reset()
    assert rig.r.launch_gate is False  # the gate is released even when the reset raises
    rig2 = _ResetRig(monkeypatch)
    rig2.r.page_reader = lambda: ((1 << 40) | 0x20, 1 << 3, (1, 2, 3, 4))
    assert rig2.r._gather_page_words()[0] == (
        0x20,
        8,
        (1, 2, 3, 4),
    )  # ERR is a 32-bit value
    # only an injected err_reader (fakes / old callers): ERR alone, STEP and DIAG taken as zero
    rig3 = _ResetRig(monkeypatch)
    rig3.r.page_reader, rig3.r.err_reader = None, (lambda: 0x40)
    assert rig3.r._read_page_words() == (0x40, 0, (0, 0, 0, 0))


def test_launch_gate_is_set_for_the_whole_reset_and_released_after_success_and_failure(
    monkeypatch,
):
    rig = _ResetRig(monkeypatch)
    seen = []
    rig.r.set_drain("o1", lambda: seen.append(("drain", rig.r.launch_gate)))
    rig.r.set_init(
        "o1", lambda: seen.append(("init", rig.r.launch_gate))
    )  # set_init runs the hook once NOW (gate off)
    seen.clear()
    assert rig.r.launch_gate is False
    rig.r.reset()
    assert seen == [("drain", True), ("init", True)] and rig.r.launch_gate is False
    # the gate also covers the legacy name
    seen.clear()
    rig.r.quiesce_and_reset()
    assert all(g for _n, g in seen) and rig.r.launch_gate is False
    # a failed reset releases the gate too (the caller then fails the worker, but the flag must not stay stuck)
    bad = _ResetRig(monkeypatch, late=[errpage.encode_err(6, 3, 2, True)] * 10)
    with pytest.raises(gf.ResetError):
        bad.r.reset()
    assert bad.r.launch_gate is False


def test_the_timing_line_has_timestamps_for_every_phase_also_when_the_reset_fails(
    monkeypatch,
):
    rig = _ResetRig(monkeypatch)
    t = {"now": 1_700_000_000.0}

    def clock():
        t["now"] += 0.002
        return t["now"]

    rig.r.reset_clock = clock
    with _Logs() as lg:
        rig.r.reset()
    line = next(m for m in lg.msgs if "event=resources reset_timing" in m)
    for token in (
        "where=reset",
        "zero=h2d",
        "rounds=1",
        "ok=1",
        "start=22:13:20.",
        "drain=[",
        "barrier0=[",
        "zero1=[",
        "sync_barrier1=[",
        "verify1=[",
        "total=",
    ):
        assert token in line, (token, line)
    assert re.search(
        r"drain=\[\d\d:\d\d:\d\d\.\d{3}\.\.\d\d:\d\d:\d\d\.\d{3} \d+\.\d\dms\]", line
    )
    # on failure the line is logged BEFORE the exception leaves, with ok=0 and every round's phases
    bad = _ResetRig(monkeypatch, late=[errpage.encode_err(6, 3, 2, True)] * 10)
    with _Logs() as lg2, pytest.raises(gf.ResetError):
        bad.r.reset()
    fail = next(m for m in lg2.msgs if "event=resources reset_timing" in m)
    assert (
        "ok=0" in fail
        and "rounds=3" in fail
        and "zero3=[" in fail
        and "verify3=[" in fail
    )


def test_devzero_fill_chunks_the_window_aligns_and_syncs(monkeypatch):
    from aiter.gemm_fusion import devzero

    log = []
    tm = types.ModuleType("torch")

    def as_tensor(obj, device=None):
        cai = obj.__cuda_array_interface__
        return NS(
            addr=cai["data"][0],
            n=cai["shape"][0],
            typestr=cai["typestr"],
            device=device,
            zero_=lambda a=cai["data"][0], n=cai["shape"][0]: log.append(
                ("zero_", a, n)
            ),
        )

    tm.as_tensor = as_tensor
    tm.cuda = NS(
        synchronize=lambda dev: log.append(("sync", dev)), is_available=lambda: True
    )
    monkeypatch.setitem(sys.modules, "torch", tm)
    assert devzero.torch_available() is True
    base = 0x7F00_0000_0000
    devzero.torch_fill_zero(
        2, base, 2_500_000_000 // 4 * 4
    )  # 2.5 GB: three chunks (1 GiB, 1 GiB, the rest)
    chunk = devzero.CHUNK_BYTES
    assert log[0] == ("zero_", base, chunk // 4) and log[1] == (
        "zero_",
        base + chunk,
        chunk // 4,
    )
    assert (
        log[2][0] == "zero_"
        and log[2][1] == base + 2 * chunk
        and log[-1] == ("sync", 2)
        and len(log) == 4
    )
    with pytest.raises(ValueError, match="4-byte aligned"):
        devzero.torch_fill_zero(0, base + 2, 4096)
    with pytest.raises(ValueError, match="4-byte aligned"):
        devzero.torch_fill_zero(0, base, 4094)
    # no usable device / no torch: not available (the "auto" zero method falls back to the H2D memcpy)
    tm.cuda = NS(is_available=lambda: False)
    assert devzero.torch_available() is False
    monkeypatch.setitem(
        sys.modules, "torch", types.ModuleType("torch")
    )  # a fake torch without .cuda
    assert devzero.torch_available() is False


# ======================================================================= R16: the call_id check on the low 19 bits (large r13 tags), per token count, over the pad-M universe
LG_VID, SM_VID = "sp_qkv.large.v1", "sp_qkv.small.v1"


def test_check_call_ids_low_bits_are_checked_per_token_count_and_only_for_the_tagged_variants():
    fixed = {
        ("a", 4096, LG_VID, 1): 0x1000_0005,
        (
            "b",
            4096,
            LG_VID,
            1,
        ): 0x2000_0005,  # same low 19 bits as "a", same tokens, both large -> collide on the tag
        (
            "c",
            4100,
            LG_VID,
            1,
        ): 0x3000_0005,  # same low bits, ANOTHER token count (m_real differs in the tag): fine
        (
            "d",
            4096,
            SM_VID,
            2,
        ): 0x4000_0005,  # the small plans' tag has no low-19 field: not checked
        ("e", 4096, LG_VID, 1): 0x5000_0000,  # low field 0
    }
    key_fn = lambda *k: fixed[k]
    rep = callid.check_call_ids(fixed, key_fn)  # the 31-bit check alone: all distinct
    assert rep.ok and rep.low_collisions == {} and rep.tag_bits is None
    rep = callid.check_call_ids(fixed, key_fn, tag_bits=19, tag_vids=(LG_VID,))
    assert not rep.ok and rep.tag_bits == 19
    assert (
        set(rep.low_collisions) == {(4096, 5)}
        and len(rep.low_collisions[(4096, 5)]) == 2
    )
    assert {k[0] for k in rep.low_collisions[(4096, 5)]} == {"a", "b"}
    assert [k[0] for k in rep.zero_low] == ["e"] and rep.collisions == {}
    assert {k[0] for k in rep.refused_keys} == {
        "a",
        "b",
        "e",
    }  # c (another token count) and d (small) stay fused
    # without the variant filter the small key counts too
    rep = callid.check_call_ids(fixed, key_fn, tag_bits=19)
    assert {k[0] for k in rep.low_collisions[(4096, 5)]} == {"a", "b", "d"}
    # per_tokens: the SAME 31-bit id in two token counts is not a collision (they never meet), inside one it is
    dup = {
        ("a", 4096, LG_VID, 1): 7,
        ("b", 4100, LG_VID, 1): 7,
        ("c", 4096, LG_VID, 1): 8,
        ("d", 4096, LG_VID, 1): 8,
    }
    kf = lambda *k: dup[k]
    assert set(callid.check_call_ids(dup, kf).collisions) == {7, 8}
    rep = callid.check_call_ids(dup, kf, per_tokens=True)
    assert set(rep.collisions) == {8} and {k[0] for k in rep.collisions[8]} == {
        "c",
        "d",
    }


def test_tag_unique_ids_are_distinct_nonzero_on_the_low_bits_by_construction_and_rank_uniform():
    sites = MODEL_SITES + DRAFT_SITES
    keys = [(s, 4 * m, LG_VID, 1) for s in sites for m in range(1024, 1400)]
    ids = callid.tag_unique_ids(keys)
    assert set(ids) == set(keys) and all(0 < v < (1 << 31) for v in ids.values())
    mask = (1 << 19) - 1
    for m in (1024, 1100, 1399):
        low = [ids[(s, 4 * m, LG_VID, 1)] & mask for s in sites]
        assert (
            len(set(low)) == len(sites) and 0 not in low
        )  # distinct and non-zero within one token count
    rep = callid.check_call_ids(
        keys, lambda *k: ids[k], tag_bits=19, tag_vids=(LG_VID,), per_tokens=True
    )
    assert rep.ok
    # a pure function of the key set: the same on every rank, whatever the order they were registered in
    assert callid.tag_unique_ids(list(reversed(keys))) == ids
    # the high bits keep the hash's entropy (two token counts of one site differ above the tag field)
    assert (ids[(sites[0], 4096, LG_VID, 1)] >> 19) != (
        ids[(sites[0], 4100, LG_VID, 1)] >> 19
    )
    with pytest.raises(ValueError, match="do not fit 1 tag bits"):
        callid.tag_unique_ids([("a", 4, LG_VID, 1), ("b", 4, LG_VID, 1)], tag_bits=1)


def test_registry_with_tag_bits_refuses_low_collisions_and_zero_low_fields():
    reg = callid.CallIdRegistry(
        lambda *k: {
            "a": 0x1000_0005,
            "b": 0x2000_0005,
            "c": 0x3000_0000,
            "d": 0x4000_0009,
        }[k[0]],
        tag_bits=19,
        tag_vids=(LG_VID,),
        per_tokens=True,
    )
    for site in "abcd":
        reg.add(site, 4096, LG_VID, 1)
    rep = reg.freeze()
    assert (
        reg.is_refused("a", 4096, LG_VID, 1)
        and reg.is_refused("b", 4096, LG_VID, 1)
        and reg.is_refused("c", 4096, LG_VID, 1)
    )
    assert (
        not reg.is_refused("d", 4096, LG_VID, 1)
        and reg.call_id("d", 4096, LG_VID, 1) == 0x4000_0009
    )
    with pytest.raises(ValueError, match="refused"):
        reg.call_id("a", 4096, LG_VID, 1)
    assert rep.tag_bits == 19


def test_the_pad_m_key_universe_covers_every_real_m_a_bucket_can_serve(monkeypatch):
    clear_profile_cache()
    prof = profmod.load_profile(
        PROFILE
    )  # the shipped layout: r18 large buckets 512 / 768 (floors 386 / 513) + the r13 1024 .. 8192 rows (1024: 769)
    sites = MODEL_SITES + DRAFT_SITES
    keys, unresolved = callid.profile_call_id_keys(prof, sites, world=4, pad_m=True)
    assert unresolved == []
    tokens = {k[1] for k in keys}
    # small: every m 1..256 (buckets); large: every real m a bucket serves, in the UNION over the modes (the r18 512 bucket from 386, the 768 from 513, then 1024..8192) -> m 386..8192
    assert tokens == {4 * m for m in range(1, 257)} | {4 * m for m in range(386, 8193)}
    assert len(tokens) == 256 + (8192 - 386 + 1) == 8063
    assert {k[2] for k in keys if k[1] >= 4 * 386} == {LG_VID} and {
        k[3] for k in keys if k[1] >= 4 * 386
    } == {1}
    assert {k[2] for k in keys if k[1] <= 1024} == {SM_VID} and {
        k[3] for k in keys if k[1] <= 1024
    } == {2}
    # per token count the 31-bit ids are distinct with the hash ids (the global check is impossible over ~500k keys and was never needed) ...
    rep = callid.check_profile_call_ids(prof, sites, world=4, pad_m=True)
    assert rep.ok and rep.keys == len(keys) == 8063 * len(sites)
    # ... but the HASH ids collide on the 19 tag bits of the r13 launches (~ n_tokens x pairs / 2^19 expected) and one has a zero low field: r13 needs tag-unique ids
    rep = callid.check_profile_call_ids(
        prof, sites, world=4, pad_m=True, tag_bits=19, tag_vids=(LG_VID,)
    )
    assert rep.collisions == {} and len(rep.low_collisions) > 0 and rep.refused_sites
    assert all(k[2] == LG_VID for ks in rep.low_collisions.values() for k in ks)
    # the assigned ordinal ids (what ATOM passes as call_id for the large launches) are clean for the whole universe; small launches keep decision_key
    ids = callid.tag_unique_ids(keys)
    rep = callid.check_call_ids(
        keys, lambda *k: ids[k], tag_bits=19, tag_vids=(LG_VID,), per_tokens=True
    )
    assert rep.ok and rep.keys == len(keys)
    # the r12 layout (rollback / A/B): exact large rows only (grid + the 768 bucket), and the hash ids pass the same checks (r12 tags carry the whole call_id)
    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # the r18 buckets are the outer layout
    monkeypatch.setattr(specs, "LARGE_R13_IN_PROFILE", False)
    from aiter.ops.gemm_fusion.sp_qkv import gen_profile

    p12 = profmod.parse_profile(json.loads(json.dumps(gen_profile.build_profile())))
    keys12, _u = callid.profile_call_id_keys(p12, sites, world=4, pad_m=True)
    assert {k[1] for k in keys12} == {4 * m for m in range(1, 257)} | {
        4 * m for m in (768, *range(1024, 8193, 256))
    }
    rep = callid.check_profile_call_ids(p12, sites, world=4, pad_m=True)
    assert rep.ok and rep.keys == 286 * len(sites)
