# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate / verify / golden-check a QSA index scorer profile.

Modes::

    # CPU: (re)write the profile from the sweep plan + rule spec (+ Phase-1 evidence);
    # --keep-golden carries golden/notes of unchanged rows/rules over from the existing file
    python -m aiter.ops.qsa_index_scorer.gen_profile emit  [--plan-file captured.json] [--out X.json]
    # CPU: assert the committed profile equals what `emit` would write (drift guard)
    python -m aiter.ops.qsa_index_scorer.gen_profile verify [--profile X.json] [--plan-file ...]
    # GPU: run every row and every rule sample through the public API vs the frozen golden;
    # write status back (file mode loads the package standalone: no `import aiter`)
    QSA_GOLDEN_DIR=/path/golden_baseline \
    python aiter/ops/qsa_index_scorer/gen_profile.py check [--profile X.json] [--plan-file cap.json] \
        [--only rows|rules|all] [--cells a,rule:b] [--out Y.json]

Range rules (:data:`RULE_SPEC`, schema 2) are verified over a deterministic
sample set: the rows/requests corners + midpoint (snapped onto the rule's
relation; per column under ``rows_x_columns_max``) crossed with the columns/table_width
corners + midpoint, plus every built-in shape (:data:`STAGE3_CAPTURE`, the exact bf16
rows, :data:`AGENTX_SHAPES`) and ``--plan-file`` entry the rule admits. A
rule is ``pass`` only when every sample passes the golden on the profiled kernel.

The golden (``qsa_golden.py`` / ``qsa_shapes.py``) is read from ``$QSA_GOLDEN_DIR``
and is never modified. The comparator and tolerance are the frozen Phase-1 ones:
``qsa_golden.compare_logits(rtol=1e-4, atol=1e-4)`` over ``valid_page & c < visible``,
``visible_groups`` exact, ``row_starts`` all zero, no NaN, and every element
outside the mask bit-identical to the golden's ``-inf``.

A ``--plan-file`` is a JSON list (or ``{"plan": [...]}``) of captured call classes
``{"cache_dtype", "rows", "requests", "columns", "table_width"[, "cache_pages", "max_ctx"]}``
(e.g. from the ATOM capture hook). Entries a rule admits become samples of that
rule; the rest are reported (``--plan-rows`` adds them as exact rows). New rows
and rules start as ``pending`` until ``check``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

if __package__ in (None, ""):  # executed as a file: make the sibling module importable
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import profiles as P  # type: ignore  # noqa: E402
else:
    from . import profiles as P

TOOL = "aiter.ops.qsa_index_scorer.gen_profile"
TOOL_VERSION = 1
GOLDEN_REVISION = "4b8888907e2bd3530bfde10eb081bf8cf60d49749aea4cac62ec486211b12ff8"
GOLDEN_RTOL = 1e-4
GOLDEN_ATOL = 1e-4
KERNEL_GIT_REV = "r2e-integrate"
#: Profiled kernel (ROUND-R2 4150ad40: P in {4, 8, 16} + decode-group group_rows /
#: row_mode / wg_waves + draft wg_waves). `emit` refuses any other source so a kernel
#: change can never silently inherit evidence.
R2_KERNEL_SHA256 = "d438d2b356ccd4c0e576a32cb0aeb52aafabc5ed4c25ad127f1e7b1dde6d67a9"
#: ROUND-R2B: kstat prefill + persist fallback loop + prefill route table.
R2B_KERNEL_SHA256 = "b82ef05be6666784a23ca27fd46f2233b8f2a1c4320d4bdea939491aac418e46"
#: ROUND-R2C: opt-in prefill variants (kstat early, ktile, fp8 r32 early); every default
#: specialization byte-identical to R2B (no-device gate base d438d2b3: all but r32_persist,
#: whose R2B fallback-loop change carries; kstat specializations vs R2B b82ef05b).
R2C_KERNEL_SHA256 = "4c45af10831dab674acc596a149ad33f52b0233cac9867c373376245306778cb"
#: ROUND-R2D: opt-in trace / lean / ipf / kconv prefill knobs; every R2C specialization
#: (defaults AND R2C opt-ins) byte-identical to R2C 4c45af10 (no-device gate base 4c45af10).
R2D_KERNEL_SHA256 = "685b0f604093a8ac673a016322a3ef2d4da7aa20670adc92f0bce97749960b37"
#: ROUND-R2E: router defaults only (prefill route presets r2e for bf16 / fp8, draft fp8
#: wg_waves auto=2, decode-group group_rows=3 'auto' slab 32); every R2D specialization
#: byte-identical to R2D 685b0f60 (no-device gate base 685b0f60: records/r2e).
KERNEL_SHA256 = "58051973626d598b36b0c51c4ac89a8830fa05b3a8f78d9628aeadd436c1f0ad"
#: The ISA-identity chain page-4 evidence travels along (each link one no-device gate:
#: records/page16, records/r2-kernel, records/r2c, records/r2d isa-evidence.json).
ISA_IDENTITY_CHAIN = ("bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93",
                      "536667d5997670a8b6d8fad54db438b2d958617d2ff5bd9b9edd359c5ea528a1",
                      R2_KERNEL_SHA256, R2C_KERNEL_SHA256, R2D_KERNEL_SHA256,   # R2D: R2C -> R2D
                      KERNEL_SHA256)                                     # R2E: R2D -> R2E
#: ROUND-R2B: variants whose default ISA changed on a link (final sha -> vids). Their
#: page-4 evidence does NOT carry over (pending until the GPU `gen_profile check`); every
#: other page-4 specialization must still be byte-identical on that link.
ISA_CHANGED_VIDS = {R2B_KERNEL_SHA256: ("qsa.page4.r32_persist.v1",),
                    R2C_KERNEL_SHA256: ("qsa.page4.r32_persist.v1",)}
#: kernel-name marker of the ISA-changed builds (mismatches allowed only for these)
_ISA_CHANGED_NAME = {R2B_KERNEL_SHA256: "_persist", R2C_KERNEL_SHA256: "_persist"}
_GATE_TEXT = {"536667d5997670a8b6d8fad54db438b2d958617d2ff5bd9b9edd359c5ea528a1":
              ("ROUND-PAGE16", "every page-4 specialization"),
              R2_KERNEL_SHA256: ("ROUND-R2", "every page-4 and page-16 specialization"),
              R2B_KERNEL_SHA256: ("ROUND-R2B", "every page-4/8/16 specialization except r32_persist"),
              R2C_KERNEL_SHA256: ("ROUND-R2C", "every page-4/8/16 specialization except r32_persist"),
              R2D_KERNEL_SHA256: ("ROUND-R2D", "every page-4/8/16 specialization incl. the R2C opt-ins"),
              KERNEL_SHA256: ("ROUND-R2E", "every page-4/8/16 bf16+fp8 R2D specialization incl. "
                                           "the R2D opt-ins; new defaults = R2D opt-in builds")}
#: The kernel every page-4 GPU evidence of this profile was recorded on (champion 82c4dec5 /
#: stage2 c1c0223b restamp). Page-4 evidence carries over to KERNEL_SHA256 only through the
#: ROUND-PAGE16 no-device ISA-identity gate (``emit --isa-identity-evidence``).
PAGE4_EVIDENCE_KERNEL_SHA256 = "bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93"
E2E_WIDTH = 16384
E2E_PAGES = 738964
ARCH, CU = "gfx1250", 256

#: The builtin sweep plan: target buckets (integration brief section 5) + the
#: Phase-1 GPU golden suite shapes. (cache_dtype, rows, requests, columns,
#: table_width, cache_pages, max_ctx, cell prefix)
def builtin_plan() -> list[dict]:
    plan = []
    for dt in P.CACHE_DTYPES:
        for cols in (65536, 32768):
            plan.append(dict(cache_dtype=dt, rows=64, requests=16, columns=cols))
            plan.append(dict(cache_dtype=dt, rows=16, requests=16, columns=cols))
        # prefill 2880 rows / 2 requests at c65536 = 5 x 512-row chunks + one 320-row chunk
        plan.append(dict(cache_dtype=dt, rows=512, requests=2, columns=65536))
        plan.append(dict(cache_dtype=dt, rows=320, requests=2, columns=65536))
    for p in plan:
        p.update(table_width=E2E_WIDTH, cache_pages=E2E_PAGES, max_ctx=16 * E2E_WIDTH)
    small = [  # Phase-1 run_gpu.py correctness shapes (I5/I6/I7/I8/tail)
        dict(cache_dtype="bf16", rows=4, requests=1, columns=64, table_width=16, cache_pages=64),
        dict(cache_dtype="fp8", rows=4, requests=1, columns=64, table_width=16, cache_pages=64),
        dict(cache_dtype="bf16", rows=16, requests=2, columns=128, table_width=32, cache_pages=128),
        dict(cache_dtype="fp8", rows=16, requests=2, columns=128, table_width=32, cache_pages=128),
        dict(cache_dtype="bf16", rows=64, requests=2, columns=256, table_width=64, cache_pages=256),
        dict(cache_dtype="bf16", rows=272, requests=1, columns=256, table_width=64, cache_pages=64),
        dict(cache_dtype="bf16", rows=8, requests=2, columns=1000, table_width=256, cache_pages=1024),
        dict(cache_dtype="fp8", rows=8, requests=2, columns=1000, table_width=256, cache_pages=1024),
    ]
    for s in small:
        s.update(max_ctx=16 * s["table_width"])
    return plan + small


def cell_name(e: dict) -> str:
    reg = P.regime_of(e["rows"], e["requests"])
    page = P.page_of(e)
    return (f"{reg}_{e['cache_dtype']}_r{e['rows']}q{e['requests']}"
            f"_c{e['columns']}_w{e['table_width']}" + ("" if page == P.PAGE_SIZE else f"_p{page}"))


def tokens_per_page(page_size: int) -> int:
    """KV tokens one index page covers (P compressed rows x compress ratio 4)."""
    return int(page_size) * P.COMPRESS_RATIO


#: Phase-1 GPU golden evidence, recorded on kernel a78c285c (a78c285c commit body:
#: "GPU (a08-1 GPU3, gpu-lock + idle gate): correctness 22/22 default").
#: cell -> run_gpu.py correctness case names that exercised exactly that row.
#: It carries over to 82c4dec5 only for bf16 rows: every bf16 kernel is ISA
#: byte-identical a78c285c -> 82c4dec5 (BF16_ISA_IDENTITY). Every fp8 kernel
#: changed (intrinsic pk8 converter, decode-group w64, draft qpre) -> fp8 rows are
#: `pending` until the root's GPU `gen_profile check` restamps them.
PHASE1_KERNEL_SHA256 = "30c6213d492ad50a9e063456628737ca993060d73dfa4ead248558ba5abb121c"
PHASE1_GIT_REV = "a78c285c"
BF16_ISA_IDENTITY = (
    "bf16 ISA byte-identical a78c285c -> 82c4dec5: 82c4dec5 commit body + _g2/ROUND_FP8PERF.md "
    "isa2 (38/38 bf16/legacy byte-identical vs 2e20f23b) + _g2/ROUND_W64FIX2.md (bf16 unchanged vs "
    "a78c285c); stage2r2 isa_audit_qsa_index_scorer.py --base-source a78c285c on the profiled "
    "bf16 rows")
STALE_FP8_NOTE = ("82c4dec5 changed every fp8 kernel (fp8 WMMA-hazard fix: intrinsic pk8 converter; "
                  "decode-group w64 + sched base; draft sched qpre): Phase-1 evidence on a78c285c "
                  "does not apply -> run `gen_profile check` on GPU")
PHASE1_EVIDENCE = {
    "decode_bf16_r64q16_c65536_w16384": ["I1_e2e_decode_bf16"],
    "decode_fp8_r64q16_c65536_w16384": ["I2_e2e_decode_fp8"],
    "draft_bf16_r16q16_c65536_w16384": ["I3_e2e_draft_bf16"],
    "draft_fp8_r16q16_c65536_w16384": ["I3_e2e_draft_fp8"],
    "prefill_bf16_r512q2_c65536_w16384": ["prefill_chunk512_bf16"],
    "prefill_fp8_r512q2_c65536_w16384": ["prefill_chunk512_fp8"],
    "decode_bf16_r64q16_c32768_w16384": ["I14_decode_c32768_bf16"],
    "decode_bf16_r4q1_c64_w16": ["I5_min_decode_bf16", "edge_visible_zero_bf16",
                                 "edge_t2r_invalid_bf16", "edge_bad_page_bf16",
                                 "edge_page_oob_bf16"],
    "decode_fp8_r4q1_c64_w16": ["I5_min_decode_fp8", "edge_bad_page_fp8", "edge_page_oob_fp8",
                                "edge_short_ctx_fp8"],
    "decode_bf16_r16q2_c128_w32": ["I6_2req_bf16"],
    "decode_fp8_r16q2_c128_w32": ["I6_2req_fp8"],
    "prefill_bf16_r64q2_c256_w64": ["I7_64rows_bf16"],
    "prefill_bf16_r272q1_c256_w64": ["I8_272rows_bf16"],
    "decode_bf16_r8q2_c1000_w256": ["tail_columns_1000_bf16"],
    "decode_fp8_r8q2_c1000_w256": ["tail_columns_1000_fp8"],
}
PHASE1_SOURCE = ("phase1: _g2/run_gpu.py --stage correctness on kernel a78c285c "
                 "(a08-1 GPU3, root, 2026-11-03, 22/22 PASS default env)")
EVIDENCE_NOTE = ("bf16 rows: Phase-1 GPU evidence on a78c285c carried by ISA identity; "
                 "fp8 rows: pending fresh GPU evidence on 82c4dec5")


def make_row(e: dict, evidence: dict | None = None) -> dict:
    regime = P.regime_of(e["rows"], e["requests"])
    if regime is None:
        raise ValueError(f"{e}: no profiled regime for rows={e['rows']} requests={e['requests']}")
    page = P.page_of(e)
    vid, params = P.expected_variant(e["cache_dtype"], regime, e["columns"], page,
                                     rows=int(e["rows"]))
    # ROUND-R2E: the default-env wrapper knobs (fp8 draft wg_waves 2): a re-pinned row gets
    # a new sig and never inherits the wg_waves 4 evidence
    params = dict(params, **P.default_knobs(e["cache_dtype"], regime))
    cell = cell_name(e)
    row: dict[str, Any] = {
        "sig": "",
        "arch": e.get("arch", ARCH), "cu": int(e.get("cu", CU)),
        "cache_dtype": e["cache_dtype"], "regime": regime, "page_size": page,
        "rows": {"min": int(e["rows"]), "max": int(e["rows"])},
        "requests": {"min": int(e["requests"]), "max": int(e["requests"])},
        "columns": int(e["columns"]), "table_width": int(e["table_width"]),
        "vid": vid, "params": params, "cell": cell,
        "fixture": {"cache_pages": int(e.get("cache_pages", E2E_PAGES)),
                    "max_ctx": int(e.get("max_ctx", tokens_per_page(page) * e["table_width"]))},
        "golden": {"status": "pending"},
        "generated_by": f"{TOOL}@{TOOL_VERSION}",
        "notes": e.get("notes", ""),
    }
    if evidence and cell in evidence and e["cache_dtype"] == "bf16":
        row["golden"] = {"status": "pass", "source": PHASE1_SOURCE, "cases": evidence[cell],
                         "kernel_sha256": PHASE1_KERNEL_SHA256, "kernel_git_rev": PHASE1_GIT_REV,
                         "isa_identity": BF16_ISA_IDENTITY,
                         "comparator": "qsa_golden.compare_logits",
                         "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL}
    elif evidence and cell in evidence:
        row["golden"] = {"status": "pending", "stale_cases": evidence[cell],
                         "stale_kernel_sha256": PHASE1_KERNEL_SHA256,
                         "stale_kernel_git_rev": PHASE1_GIT_REV}
        if not row["notes"]:
            row["notes"] = STALE_FP8_NOTE
    elif not row["notes"]:
        row["notes"] = "not yet golden-checked on GPU for this kernel: run `gen_profile check`"
    row["sig"] = P.row_signature(row)
    return row


# ---------------------------------------------------------------------------
# Range rules
# ---------------------------------------------------------------------------
_POW2_ALL = {"pow2": {"min": 64, "max": 65536}}
#: ATOM ``qsa_select_paged_tokens``: ``rows_per_chunk = logits_workspace_bytes // (columns * 4)``
#: with ``DEFAULT_LOGITS_WORKSPACE_BYTES = 128 MiB`` -> rows * columns <= 2^25 per scorer call.
ATOM_CHUNK_ROWS_X_COLUMNS = (128 * 1024 * 1024) // 4
#: Rule spec (bf16: the Stage-3 Qwen3.8-Flash-Next e2e index cache). Bounds are the
#: CUDA-graph decode batches (<= 512 requests), ATOM's prefill chunks (<= 16384 rows,
#: <= 32 requests), columns from one WMMA tile (64) to the e2e width (65536), table
#: width up to the e2e 16384; every rule maps to exactly one kernel variant.
RULE_SPEC: list[dict] = [
    dict(id="draft_bf16", cache_dtype="bf16", relation="rows<=requests",
         rows={"min": 1, "max": 512}, requests={"min": 1, "max": 512},
         columns=_POW2_ALL, table_width={"min": 16, "max": 16384},
         notes="no-MTP decode (rows == requests, CUDA-graph batches) + MTP draft"),
    dict(id="decode_bf16", cache_dtype="bf16", relation="decode_band",
         rows={"min": 4, "max": 2048, "multiple_of": 4}, requests={"min": 1, "max": 512},
         columns=_POW2_ALL, table_width={"min": 16, "max": 16384},
         notes="decode row-group kernel: requests < rows < 32*requests, rows % 4 == 0"),
    dict(id="tail_bf16", cache_dtype="bf16", relation="tail_band",
         rows={"min": 2, "max": 4095}, requests={"min": 1, "max": 512},
         columns=_POW2_ALL, table_width={"min": 16, "max": 16384},
         notes="r16 WMMA kernel: requests < rows < 32*requests, rows % 4 != 0 (short prefill)"),
    # ROUND-R2E: default prefill route 'r2e' (P.PREFILL_KT_MAX_COLUMNS): lean ktile RB32 x C2
    # for columns <= 2048, persist for 4096..65536 (was: r32 64..256, persist 512..8192).
    dict(id="prefill_bf16_kt", cache_dtype="bf16", relation="rows>=32*requests",
         rows={"min": 32, "max": 16384}, requests={"min": 1, "max": 32},
         columns={"pow2": {"min": 64, "max": P.PREFILL_KT_MAX_COLUMNS}},
         table_width={"min": 16, "max": 16384},
         notes="lean ktile r32 kernel RB32 x C2 (columns <= 2048; route preset r2e)"),
    dict(id="prefill_bf16_persist", cache_dtype="bf16", relation="rows>=32*requests",
         rows={"min": 32, "max": 16384}, requests={"min": 1, "max": 32},
         columns={"pow2": {"min": 2 * P.PREFILL_KT_MAX_COLUMNS, "max": 8192}},
         table_width={"min": 128, "max": 16384},
         notes="persistent r32 kernel (columns 4096..8192; 16384 x 8192 fp32 logits fit int32)"),
    dict(id="prefill_bf16_persist_wide", cache_dtype="bf16", relation="rows>=32*requests",
         rows={"min": 32, "max": 2048}, requests={"min": 1, "max": 32},
         columns={"pow2": {"min": 16384, "max": 65536}}, table_width={"min": 4096, "max": 16384},
         rows_x_columns_max=ATOM_CHUNK_ROWS_X_COLUMNS,
         notes="persistent r32 kernel at long-context columns (AgentX 256k): rows per call "
               "bounded by ATOM's 128 MiB logits chunk (rows <= 2^25 / columns: 2048 @ 16384, "
               "1024 @ 32768, 512 @ 65536)"),
]

#: ROUND-PAGE16: KV block 64 (``--block-size 64``, conc48 serving) -> index page 16.
#: The P = 4 rule set mirrored at P = 16: same relations / rows / requests / columns /
#: variants, page-table width bounds re-derived for 16 columns per page with the e2e
#: width W = max_model_len / 64 = 4096 (262144 tokens): columns <= 16 * W <= 65536.
#: Rules start ``pending`` (never selected at runtime) until the GPU ``gen_profile check``.
E2E_WIDTH_P16 = 4096
PAGE16 = 16


def _page_rule(spec: dict, page: int, tw: dict, note: str) -> dict:
    out = json.loads(json.dumps(spec))
    out.update(id=f"{spec['id']}_p{page}", page_size=int(page), table_width=dict(tw),
               notes=(spec.get("notes", "") + f" [{note}]").strip())
    return out


def _p16_rule(spec: dict, tw: dict) -> dict:
    return _page_rule(spec, PAGE16, tw, "page 16: KV block 64, W <= 4096")


RULE_SPEC_P16: list[dict] = [
    _p16_rule(RULE_SPEC[0], {"min": 4, "max": E2E_WIDTH_P16}),      # draft
    _p16_rule(RULE_SPEC[1], {"min": 4, "max": E2E_WIDTH_P16}),      # decode row-group
    _p16_rule(RULE_SPEC[2], {"min": 4, "max": E2E_WIDTH_P16}),      # r16 tail
    _p16_rule(RULE_SPEC[3], {"min": 4, "max": E2E_WIDTH_P16}),      # r32 (columns < 512)
    _p16_rule(RULE_SPEC[4], {"min": 32, "max": E2E_WIDTH_P16}),     # persist (512..8192)
    _p16_rule(RULE_SPEC[5], {"min": 1024, "max": E2E_WIDTH_P16}),   # persist wide (16384..65536)
]


#: R2: KV block 32 -> index page 8. The P = 4 rule set mirrored at P = 8 (as P = 16):
#: same relations / rows / requests / columns / variants, table-width bounds re-derived
#: for 8 columns per page with the e2e width W = 262144 / 32 = 8192. ``pending`` until
#: the page-8 kernel lands and the GPU ``gen_profile check`` passes (until then the op
#: also refuses page 8 with ``cache_shape_unsupported`` via the kernel capability probe).
E2E_WIDTH_P8 = 8192
PAGE8 = 8
_P8_NOTE = "page 8: KV block 32, W <= 8192"
RULE_SPEC_P8: list[dict] = [
    _page_rule(RULE_SPEC[0], PAGE8, {"min": 8, "max": E2E_WIDTH_P8}, _P8_NOTE),      # draft
    _page_rule(RULE_SPEC[1], PAGE8, {"min": 8, "max": E2E_WIDTH_P8}, _P8_NOTE),      # decode
    _page_rule(RULE_SPEC[2], PAGE8, {"min": 8, "max": E2E_WIDTH_P8}, _P8_NOTE),      # tail
    _page_rule(RULE_SPEC[3], PAGE8, {"min": 8, "max": E2E_WIDTH_P8}, _P8_NOTE),      # r32
    _page_rule(RULE_SPEC[4], PAGE8, {"min": 64, "max": E2E_WIDTH_P8}, _P8_NOTE),     # persist
    _page_rule(RULE_SPEC[5], PAGE8, {"min": 2048, "max": E2E_WIDTH_P8}, _P8_NOTE),   # wide
]

#: R2B-FP8: fp8 (E4M3 + fp32 per-row scale) range rules. The bf16 rule set mirrored with
#: the same relations / rows / requests / columns / table-width bounds per page size
#: (P = 4 / 8 / 16); each maps to the variant the kernel router picks for fp8
#: (``P.expected_variant``): draft (tdm, w32, sched qpre, pk8 intrin), decode-group (tdm,
#: w64 = QSA_DECODE_GROUP_COLS auto for fp8, sched base), tail r16 fp8 (raw K load:
#: ``_resolve_fp8_k_load`` keeps r16 raw) and prefill r32 fp8 (tdm, cvt fast) -- fp8 has
#: no persistent r32 kernel, so the three bf16 prefill column bands (64..256, 512..8192,
#: 16384..65536 with the ATOM chunk bound) all map to r32 fp8. ``pending`` until the GPU
#: ``gen_profile check``; samples = bounds corners/mid + the R2 intake matrix (fp8) + the
#: Phase-1 exact fp8 rows.
#: ROUND-R2E: fp8 prefill follows the fp8 route preset 'r2e': lean+kconv ktile RB32 x C2
#: (rows <= 2048) / x C4 (rows 2049..16384) for columns <= 2048, r32 fp8 above -> the bf16
#: prefill slot 3 (columns 64..2048) splits by rows into two fp8 rules.
_FP8_SLOTS = (
    (0, "draft_fp8", {}), (1, "decode_fp8", {}), (2, "tail_fp8", {}),
    (3, "prefill_fp8_kt", dict(rows={"min": 32, "max": P.FP8_KT_C2_MAX_ROWS},
                               notes="lean+kconv fp8 ktile RB32 x C2 (columns <= 2048, rows <= "
                                     "2048; route preset r2e)")),
    (3, "prefill_fp8_kt_tall", dict(rows={"min": P.FP8_KT_C2_MAX_ROWS + 1,
                                          "max": P.PREFILL_KT_MAX_ROWS},
                                    notes="lean+kconv fp8 ktile RB32 x C4 (columns <= 2048, rows "
                                          "2049..16384; route preset r2e)")),
    (4, "prefill_fp8_r32_mid", {}), (5, "prefill_fp8_r32_wide", {}),
)
_FP8_RULE_IDS = tuple(rid for _s, rid, _o in _FP8_SLOTS)
_FP8_NOTE = "fp8 E4M3 + fp32 per-row scale"


def _fp8_rule(spec: dict, rid: str, over: dict | None = None) -> dict:
    out = json.loads(json.dumps(spec))
    out.update(id=rid, cache_dtype="fp8",
               notes=(spec.get("notes", "").replace("persistent r32 kernel", "r32 fp8 kernel")
                      + f" [{_FP8_NOTE}]").strip())
    for k, v in (over or {}).items():
        out[k] = json.loads(json.dumps(v)) if k != "notes" else f"{v} [{_FP8_NOTE}]"
    return out


RULE_SPEC_FP8: list[dict] = [_fp8_rule(RULE_SPEC[s], rid, o) for s, rid, o in _FP8_SLOTS]
RULE_SPEC_FP8_P8: list[dict] = [
    _page_rule(_fp8_rule(RULE_SPEC[s], rid, o), PAGE8, dict(RULE_SPEC_P8[s]["table_width"]),
               _P8_NOTE)
    for s, rid, o in _FP8_SLOTS]
RULE_SPEC_FP8_P16: list[dict] = [
    _page_rule(_fp8_rule(RULE_SPEC[s], rid, o), PAGE16, dict(RULE_SPEC_P16[s]["table_width"]),
               "page 16: KV block 64, W <= 4096")
    for s, rid, o in _FP8_SLOTS]


def all_rule_specs() -> list[dict]:
    """Every committed rule spec, in profile order (bf16 P4/P16/P8, then fp8 P4/P8/P16)."""
    return (RULE_SPEC + RULE_SPEC_P16 + RULE_SPEC_P8
            + RULE_SPEC_FP8 + RULE_SPEC_FP8_P8 + RULE_SPEC_FP8_P16)


#: R2 intake matrix (``logits_indexer_param_list.csv``): concurrency C, contexts, prefill
#: rows. columns = ctx / 4, table_width = ctx / (4 * page) (KV block = 4 * page).
R2_CONCURRENCY = (1, 8, 16, 32, 64, 128, 192, 256, 384, 448)
R2_CONTEXTS = (4096, 16384, 65536)
R2_PREFILL_ROWS = (1024, 4096)
R2_PREFILL_CTX = 4096


def r2_matrix_shapes(page: int) -> list[tuple[int, int, int, int, str]]:
    """(rows, requests, columns, table_width, group) of the R2 intake matrix at ``page``:
    single/concurrent decode (rows = C), MTP verify bench3 (3C) and trace4 (4C), prefill."""
    out = []
    for ctx in R2_CONTEXTS:
        cols, w = ctx // 4, ctx // (4 * int(page))
        for c in R2_CONCURRENCY:
            out += [(c, c, cols, w, "decode"), (3 * c, c, cols, w, "verify3"),
                    (4 * c, c, cols, w, "verify4")]
    for r in R2_PREFILL_ROWS:
        out.append((r, 1, R2_PREFILL_CTX // 4, R2_PREFILL_CTX // (4 * int(page)), "prefill"))
    return list(dict.fromkeys(out))


def capture_shapes_p8(plan: list[dict] | None = None) -> list[tuple]:
    """Page-8 sample sources: the R2 intake matrix (bf16 and fp8) + page-8 plan entries."""
    out = [(dt, r, q, c, w, "intake_r2") for dt in P.CACHE_DTYPES
           for (r, q, c, w, _g) in r2_matrix_shapes(PAGE8)]
    for e in plan or []:
        if P.page_of(e) == PAGE8:
            out.append((e["cache_dtype"], int(e["rows"]), int(e["requests"]), int(e["columns"]),
                        int(e["table_width"])))
    return out


# ---------------------------------------------------------------------------
# R2 buckets (profiles.py "Buckets"): opt-in token-dimension padding
# ---------------------------------------------------------------------------
_DECODE_TW = {P.PAGE_SIZE: RULE_SPEC[1]["table_width"], PAGE8: RULE_SPEC_P8[1]["table_width"],
              PAGE16: {"min": 4, "max": E2E_WIDTH_P16}}


def bucket_spec() -> list[dict]:
    """group (MTP verify bench3: 3 rows/request -> 4) and rows (tail band rounded up to the
    decode band), bf16, at every page size; fp8 (R2B-FP8): the kernel-native verify3 group
    bucket only (target = the fp8 decode rule; a bucket's target shape must be served by a
    verified row/rule)."""
    out = []
    for page in (P.PAGE_SIZE, PAGE8, PAGE16):
        sfx = "" if page == P.PAGE_SIZE else f"_p{page}"
        for dt in ("bf16",):
            # R2-MERGE default: kernel-native 3-row groups (decode-group group_rows=3, the
            # caller's tensors, zero copy). Listed first: preferred whenever the kernel
            # has the capability; the staging bucket below is the fallback.
            out.append(dict(id=f"verify3_g3_{dt}{sfx}", kind="group", cache_dtype=dt,
                            page_size=page, rows={"min": 3, "max": 1536},
                            requests={"min": 1, "max": 512},
                            group={"real": 3, "padded": 4, "kernel_group_rows": 3},
                            knobs={"group_rows": 3},
                            columns=_POW2_ALL, table_width=dict(_DECODE_TW[page]),
                            notes="MTP verify bench3 (3 rows per request): decode-group kernel "
                                  "with native 3-row groups (group_rows=3, one pass per request, "
                                  "no padding/staging); target = the 4-row decode cell"))
            out.append(dict(id=f"verify3_pad4_{dt}{sfx}", kind="group", cache_dtype=dt,
                            page_size=page, rows={"min": 3, "max": 1536},
                            requests={"min": 1, "max": 512}, group={"real": 3, "padded": 4},
                            columns=_POW2_ALL, table_width=dict(_DECODE_TW[page]),
                            notes="MTP verify bench3 (3 rows per request) padded to the "
                                  "decode-group kernel's 4-row group"))
        out.append(dict(id=f"tail_round4_bf16{sfx}", kind="rows", cache_dtype="bf16",
                        page_size=page, rows={"min": 2, "max": 2047},
                        requests={"min": 1, "max": 512}, round_to=4, columns=_POW2_ALL,
                        table_width=dict(_DECODE_TW[page]),
                        notes="tail band (rows % 4 != 0) served by the decode-group kernel with "
                              "the real row count as its row bound (no padded rows written)"))
        # R2B-FP8: the fp8 decode-group kernel takes group_rows=3 like bf16
        # (``_check_decg_knobs`` / ``_build_decode_group_kernel`` are dtype-independent):
        # kernel-native 3-row groups at the fp8 decode-group slab (w64). No fp8 staging
        # (pad4) bucket: every R2 verify3 shape (rows = 3C, C % 4 == 0) is already in the
        # fp8 decode band, so without the native capability the unpadded decode_fp8 rule
        # serves it; no fp8 tail_round4 bucket (fp8 tail stays on the r16 fp8 rule).
        out.append(dict(id=f"verify3_g3_fp8{sfx}", kind="group", cache_dtype="fp8",
                        page_size=page, rows={"min": 3, "max": 1536},
                        requests={"min": 1, "max": 512},
                        group={"real": 3, "padded": 4, "kernel_group_rows": 3},
                        knobs={"group_rows": 3},
                        columns=_POW2_ALL, table_width=dict(_DECODE_TW[page]),
                        notes="MTP verify bench3 (3 rows per request), fp8 cache: decode-group "
                              "kernel (w32, ROUND-R2E) with native 3-row groups (group_rows=3, one pass per "
                              "request, no padding/staging); target = the 4-row fp8 decode cell"))
    return out


def bucket_samples(b: dict, extra: list[tuple] = ()) -> list[dict]:
    R = P.ProfileBucket(b)
    page = P.page_of(b)
    cols = P.column_values(b["columns"])
    tw = b["table_width"]
    twlo, twhi = int(tw["min"]), int(tw["max"])

    def tw_for(c, want):
        return min(max(want, twlo, -(-c // page)), twhi)

    cmid = cols[len(cols) // 2]
    ct = [(cols[0], tw_for(cols[0], twlo)), (cmid, tw_for(cmid, _mid(tw))),
          (cols[-1], twhi)]
    qq = b["requests"]
    rq = []
    for q in (int(qq["min"]), _mid(qq), int(qq["max"])):
        if b["kind"] == "group":
            rq.append((3 * q, q))
        else:
            for r in (q + 1, 4 * q + 1, 32 * q - 5):
                r += (r % 4 == 0)
                rq.append((r, q))
    out, seen = [], set()

    def add(k, src):
        k = tuple(int(x) for x in k)
        t = P.bucket_target(b, k[0], k[1]) if R.admits(*k) else None
        if (k not in seen and t is not None and t[0] * k[2] * 4 <= P.MAX_PLANE_BYTES
                and k[1] * min(k[3], sample_max_pages(page) // k[1]) >= 1):
            seen.add(k)
            out.append({"rows": k[0], "requests": k[1], "columns": k[2], "table_width": k[3],
                        "source": src})

    for (r, q) in rq:
        for (c, w) in ct:
            add((r, q, c, w), "bounds")
    for k in extra:
        add(k[:4], k[4] if len(k) > 4 else "capture")
    return out


def make_bucket(spec: dict, extra: list[tuple] = ()) -> dict:
    page = P.page_of(spec)
    cols = P.column_values(spec["columns"])
    # ROUND-R2E: group_rows 3 -> the router's 3-row slab (fp8 w32)
    vid, params = P.expected_variant(spec["cache_dtype"], "decode", cols[0], page,
                                     group_rows=int(spec.get("knobs", {}).get("group_rows", 4)))
    params = dict(params, **spec.get("knobs", {}))       # R2-MERGE pinned launch knobs
    b: dict[str, Any] = {
        "id": spec["id"], "sig": "", "kind": spec["kind"],
        "arch": spec.get("arch", ARCH), "cu": int(spec.get("cu", CU)),
        "cache_dtype": spec["cache_dtype"], "relation": P.BUCKET_KINDS[spec["kind"]],
        "rows": dict(spec["rows"]), "requests": dict(spec["requests"]),
        "columns": json.loads(json.dumps(spec["columns"])),
        "table_width": dict(spec["table_width"]), "page_size": page,
        "divisor": P.RULE_DIVISOR, "target_regime": "decode",
        "vid": vid, "params": params, "samples": [],
        "golden": {"status": "pending"},
        "generated_by": f"{TOOL}@{TOOL_VERSION}",
        "notes": spec.get("notes", ""),
    }
    for k in P.BUCKET_OPTIONAL_KEYS:
        if k in spec:
            b[k] = json.loads(json.dumps(spec[k]))
    b["sig"] = P.bucket_signature(b)
    b["samples"] = bucket_samples(b, [k for d, *k in extra if d == spec["cache_dtype"]])
    return b


def bucket_extra(page: int) -> list[tuple]:
    """Bucket sample sources: the R2 intake matrix at ``page`` (both dtypes)."""
    return [(dt, r, q, c, w, "intake_r2") for dt in P.CACHE_DTYPES
            for (r, q, c, w, _g) in r2_matrix_shapes(page)]


def agentx_p16_shapes() -> list[tuple[int, int, int, int]]:
    """Block-64 e2e call classes (W = 4096): MTP3 verify decode rows = 4*bs and MTP draft
    rows = bs for bs in 1..64 at the full 65536 columns (and 32768 for bs in powers of two),
    ATOM prefill chunks (rows <= 2^25 / columns) at columns 4096..65536, chunk tails."""
    W = E2E_WIDTH_P16
    out: list[tuple[int, int, int, int]] = []
    for bs in range(1, 65):
        out += [(4 * bs, bs, 65536, W), (bs, bs, 65536, W)]
    for bs in (1, 2, 4, 8, 16, 32, 48, 64):
        out += [(4 * bs, bs, 32768, W), (bs, bs, 32768, W)]
    for cols in (4096, 8192, 16384, 32768, 65536):
        chunk = min(16384, ATOM_CHUNK_ROWS_X_COLUMNS // cols)
        for q in (1, 4, 16, 32):
            if chunk >= 32 * q:
                out.append((chunk, q, cols, W))
        out += [(chunk - 1, 1, cols, W), (33, 1, cols, W)]
    out += [(88, 16, 65536, W), (87, 16, 65536, W), (3, 16, 65536, W), (501, 32, 32768, W)]
    return list(dict.fromkeys(out))


def builtin_sample_shapes_p16() -> list[tuple]:
    return [(*k, "agentx_p16") for k in agentx_p16_shapes()]


#: Stage-3 ATOM capture (Qwen3.8-Flash-Next MXFP4, gfx1250, no MTP, opt-in scorer):
#: every bf16 call class as (rows, requests, columns, table_width). Rule samples
#: include each class the rule admits.
STAGE3_CAPTURE: list[tuple[int, int, int, int]] = [
    (8192, 1, 4096, 1024), (512, 512, 65536, 16384), (256, 256, 65536, 16384),
    (128, 128, 65536, 16384), (64, 64, 65536, 16384), (48, 48, 65536, 16384),
    (32, 32, 65536, 16384), (16, 16, 65536, 16384), (8, 8, 65536, 16384), (7, 7, 65536, 16384),
    (6, 6, 65536, 16384), (5, 5, 65536, 16384), (4, 4, 65536, 16384), (3, 3, 65536, 16384),
    (2, 2, 65536, 16384), (1, 1, 65536, 16384), (16, 1, 4, 16384), (11, 1, 8, 16384),
    (1136, 1, 512, 16384), (16359, 16, 512, 16384), (16349, 31, 512, 16384),
    (16357, 31, 512, 16384), (16378, 30, 512, 16384), (16326, 30, 512, 16384),
    (16344, 30, 512, 16384), (16327, 31, 512, 16384), (16344, 32, 512, 16384),
    (16351, 31, 512, 16384), (16370, 30, 512, 16384), (16353, 29, 512, 16384),
    (16355, 30, 512, 16384), (16322, 30, 512, 16384), (16329, 31, 512, 16384),
    (16383, 32, 512, 16384), (16335, 32, 512, 16384), (16330, 30, 512, 16384),
    (4578, 19, 512, 16384), (50, 5, 512, 16384), (1104, 1, 512, 16384), (2, 1, 512, 16384),
    (1248, 1, 512, 16384), (4, 1, 512, 16384), (1968, 2, 256, 16384), (20, 2, 512, 16384),
    (1152, 1, 512, 16384), (719, 2, 512, 16384), (4, 1, 256, 16384), (1056, 1, 512, 16384),
    (13, 1, 512, 16384), (3456, 3, 512, 16384), (1120, 4, 512, 16384), (1044, 2, 512, 16384),
    (12, 1, 512, 16384), (2272, 2, 512, 16384), (22, 2, 512, 16384), (2032, 2, 512, 16384),
    (4887, 7, 512, 16384), (37, 5, 512, 16384), (15, 1, 512, 16384), (3360, 3, 512, 16384),
    (1180, 4, 512, 16384), (1168, 1, 512, 16384), (7, 1, 512, 16384), (1696, 2, 256, 16384),
    (1836, 4, 256, 16384), (23, 2, 256, 16384), (1296, 1, 512, 16384), (2016, 2, 512, 16384),
    (3105, 5, 512, 16384), (15, 3, 512, 16384), (1008, 1, 256, 16384), (10, 1, 256, 16384),
    (2240, 2, 512, 16384), (1134, 3, 512, 16384), (1, 1, 512, 16384),
]
STAGE3_CAPTURE_DTYPE = "bf16"

#: AgentX recipe (Qwen3.8-Flash-Next, TP1, MTP3, CONC16, contexts up to ~256k, BF16 index
#: cache, max-num-batched-tokens 16384, table_width 16384): scorer calls of ATOM's select
#: loop (rows chunked to 2^25 / columns, requests = the whole batch's page-table rows):
#: full and partial long-context prefill chunks, MTP3 verify decode (rows = 4*bs,
#: bs = CUDA-graph sizes), MTP draft (rows == bs) and the decode/tail/draft-shaped
#: last chunk of a chunked prefill (rows < 32*requests).
AGENTX_SHAPES: list[tuple[int, int, int, int]] = [
    # prefill chunks @ 65536 (512 rows / chunk)
    (512, 1, 65536, 16384), (512, 4, 65536, 16384), (512, 8, 65536, 16384),
    (512, 16, 65536, 16384), (511, 1, 65536, 16384), (480, 15, 65536, 16384),
    (33, 1, 65536, 16384),
    # @ 32768 (1024 rows / chunk)
    (1024, 1, 32768, 16384), (1024, 16, 32768, 16384), (1024, 32, 32768, 16384),
    (1023, 1, 32768, 16384), (777, 3, 32768, 16384),
    # @ 16384 (2048 rows / chunk)
    (2048, 1, 16384, 16384), (2048, 16, 16384, 16384), (2048, 32, 16384, 16384),
    (2047, 1, 16384, 16384), (1500, 7, 16384, 16384),
    # @ 8192 / 4096 (4096 / 8192 rows / chunk)
    (4096, 1, 8192, 16384), (4096, 16, 8192, 16384), (8192, 1, 4096, 16384),
    (8192, 16, 4096, 16384),
    # MTP3 verify decode (rows == 4*bs) and MTP draft (rows == bs) @ 65536
    (4, 1, 65536, 16384), (8, 2, 65536, 16384), (12, 3, 65536, 16384),
    (20, 5, 65536, 16384), (32, 8, 65536, 16384), (64, 16, 65536, 16384),
    (96, 24, 65536, 16384), (128, 32, 65536, 16384), (24, 24, 65536, 16384),
    # last chunk of a chunked prefill with rows < 32*requests (decode / tail / draft band)
    (88, 16, 65536, 16384), (87, 16, 65536, 16384), (3, 16, 65536, 16384),
    (500, 32, 32768, 16384), (501, 32, 32768, 16384),
]


def builtin_sample_shapes() -> list[tuple]:
    """(rows, requests, columns, table_width, source) of every builtin bf16 sample source:
    Stage-3 capture, the profile's exact bf16 rows, the AgentX recipe shapes."""
    out = [(*k, "capture") for k in STAGE3_CAPTURE]
    out += [(int(e["rows"]), int(e["requests"]), int(e["columns"]), int(e["table_width"]),
             "profile_row") for e in builtin_plan() if e["cache_dtype"] == STAGE3_CAPTURE_DTYPE]
    out += [(*k, "agentx") for k in AGENTX_SHAPES]
    return out
#: Physical-page budget of one sample fixture. The op refuses a cache plane >= 2 GiB
#: (int32 byte addressing, reason plane_too_large), so the bf16 fixture stays below
#: 2^21 pages (1 KiB per page). Up to 127 requests get full 16384-page tables; larger
#: batches get shorter contexts (the remaining columns are checked as -inf).
SAMPLE_MAX_PAGES = (1 << 21) - 1


def sample_max_pages(page_size: int = P.PAGE_SIZE) -> int:
    """Same < 2 GiB bf16 cache-plane budget in pages of ``page_size`` rows."""
    return SAMPLE_MAX_PAGES * P.PAGE_SIZE // int(page_size)


def _mid(rng: dict) -> int:
    lo, hi, m = int(rng["min"]), int(rng["max"]), int(rng.get("multiple_of", 1))
    v = (lo + hi) // 2
    v -= v % m
    return max(v, lo)


def _snap_rows(rule: dict, rows: int, requests: int, hi_cap: int | None = None) -> int | None:
    """Nearest rows (to ``rows``) inside the rule's rows range (``<= hi_cap``) satisfying
    its relation."""
    rr = rule["rows"]
    lo, hi = int(rr["min"]), int(rr["max"])
    if hi_cap is not None:
        hi = min(hi, int(hi_cap))
    rel, q = rule["relation"], int(requests)
    bounds = {"rows<=requests": (1, q), "rows==requests": (q, q), "rows==4*requests": (4 * q, 4 * q),
              "decode_band": (q + 1, 32 * q - 1), "tail_band": (q + 1, 32 * q - 1),
              "rows>=32*requests": (max(32, 32 * q), hi)}[rel]
    lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    if lo > hi:
        return None
    c = min(max(int(rows), lo), hi)
    for d in range(0, 8):
        for v in (c - d, c + d):
            if lo <= v <= hi and P._range_ok(rr, v) and P.relation_ok(rel, v, q):
                return v
    return None


def rule_samples(rule: dict, extra: list[tuple] = ()) -> list[dict]:
    """Deterministic golden sample set of a rule (bounds corners/mid x captured shapes)."""
    R = P.ProfileRule(rule)
    rq = []
    rr, qq = rule["rows"], rule["requests"]
    for q in (int(qq["min"]), int(qq["max"]), _mid(qq)):
        for r in (int(rr["min"]), int(rr["max"]), _mid(rr)):
            s = _snap_rows(rule, r, q)
            if s is not None and (s, q) not in rq:
                rq.append((s, q))
    cols = P.column_values(rule["columns"])
    tw = rule["table_width"]
    twlo, twhi = int(tw["min"]), int(tw["max"])
    page = P.page_of(rule)

    def tw_for(c, want):
        return min(max(want, twlo, -(-c // page)), twhi)

    cmid = cols[len(cols) // 2]
    ct = [(cols[0], tw_for(cols[0], twlo)), (cols[0], twhi), (cmid, tw_for(cmid, _mid(tw))),
          (cols[-1], tw_for(cols[-1], twlo)), (cols[-1], twhi)]
    out, seen = [], set()

    def add(k, src):
        if k not in seen and R.admits(*k) and k[0] * k[2] * 4 <= P.MAX_PLANE_BYTES:
            seen.add(k)
            out.append({"rows": k[0], "requests": k[1], "columns": k[2], "table_width": k[3],
                        "source": src})

    if "rows_x_columns_max" not in rule:
        for (r, q) in rq:
            for (c, w) in ct:
                add((r, q, c, w), "bounds")
    else:
        # rows bound tied to columns: rows corners/mid per column (min, rows_max_at(c), mid)
        for (c, w) in ct:
            hi_c = R.rows_max_at(c)
            mid_c = _mid({"min": rr["min"], "max": hi_c, **({"multiple_of": rr["multiple_of"]}
                                                           if "multiple_of" in rr else {})})
            for q in (int(qq["min"]), int(qq["max"]), _mid(qq)):
                for r in (int(rr["min"]), hi_c, mid_c):
                    s = _snap_rows(rule, r, q, hi_c)
                    if s is not None:
                        add((s, q, c, w), "bounds")
    for k in extra:
        add(tuple(int(x) for x in k[:4]), k[4] if len(k) > 4 else "capture")
    return out


def make_rule(spec: dict, extra: list[tuple] = ()) -> dict:
    regime = P.RELATIONS[spec["relation"]]
    cols = P.column_values(spec["columns"])
    page = P.page_of(spec)
    vid, params = P.expected_variant(spec["cache_dtype"], regime, cols[0], page,
                                     rows=int(spec["rows"]["min"]))
    # R2-MERGE: a GPU sweep winner may pin launch knobs per rule (P.LAUNCH_KNOBS, e.g.
    # {"wg_waves": 2}); a pinned rule gets a new sig, hence never inherits old evidence.
    # ROUND-R2E: the default-env wrapper knobs (P.default_knobs: fp8 draft wg_waves 2).
    params = dict(params, **P.default_knobs(spec["cache_dtype"], regime), **spec.get("knobs", {}))
    rule: dict[str, Any] = {
        "id": spec["id"], "sig": "",
        "arch": spec.get("arch", ARCH), "cu": int(spec.get("cu", CU)),
        "cache_dtype": spec["cache_dtype"], "regime": regime, "relation": spec["relation"],
        "rows": dict(spec["rows"]), "requests": dict(spec["requests"]),
        "columns": json.loads(json.dumps(spec["columns"])),
        "table_width": dict(spec["table_width"]),
        "page_size": page, "divisor": P.RULE_DIVISOR,
        "vid": vid, "params": params,
        "samples": [],
        "golden": {"status": "pending"},
        "generated_by": f"{TOOL}@{TOOL_VERSION}",
        "notes": spec.get("notes", ""),
    }
    if "rows_x_columns_max" in spec:
        rule["rows_x_columns_max"] = int(spec["rows_x_columns_max"])
    rule["sig"] = P.rule_signature(rule)
    rule["samples"] = rule_samples(rule, [k for d, *k in extra if d == spec["cache_dtype"]])
    return rule


def fp8_intake_shapes(page: int) -> list[tuple]:
    """R2B-FP8 sample sources of the fp8 rules at ``page``: the R2 intake matrix (fp8)."""
    return [("fp8", r, q, c, w, "intake_r2") for (r, q, c, w, _g) in r2_matrix_shapes(page)]


def capture_shapes_p16(plan: list[dict] | None = None) -> list[tuple]:
    """Page-16 sample sources: the block-64 e2e shapes (bf16), the R2 intake matrix (fp8)
    + page-16 plan entries."""
    out = [(STAGE3_CAPTURE_DTYPE, *k) for k in builtin_sample_shapes_p16()]
    out += fp8_intake_shapes(PAGE16)
    for e in plan or []:
        if P.page_of(e) == PAGE16:
            out.append((e["cache_dtype"], int(e["rows"]), int(e["requests"]), int(e["columns"]),
                        int(e["table_width"])))
    return out


def capture_shapes(plan: list[dict] | None = None) -> list[tuple]:
    """(cache_dtype, rows, requests, columns, table_width[, source]) of the builtin sample
    sources (:func:`builtin_sample_shapes`), the fp8 sources (R2 intake matrix + Phase-1
    exact fp8 rows) + plan."""
    out = [(STAGE3_CAPTURE_DTYPE, *k) for k in builtin_sample_shapes()]
    out += fp8_intake_shapes(P.PAGE_SIZE)
    out += [("fp8", int(e["rows"]), int(e["requests"]), int(e["columns"]), int(e["table_width"]),
             "profile_row") for e in builtin_plan() if e["cache_dtype"] == "fp8"]
    for e in plan or []:
        if P.page_of(e) != P.PAGE_SIZE:
            continue
        out.append((e["cache_dtype"], int(e["rows"]), int(e["requests"]), int(e["columns"]),
                    int(e["table_width"])))
    return out


def classify_plan(data: dict, plan: list[dict], allow_unverified: bool = True) -> list[dict]:
    """Match every plan entry against a profile dict -> [{entry, ok, reason, cell}]."""
    prof = P.profile_from_data(data)
    out = []
    for e in plan:
        m = P.match(prof, arch=e.get("arch", ARCH), cu=int(e.get("cu", CU)),
                    cache_dtype=e["cache_dtype"], rows=int(e["rows"]),
                    requests=int(e["requests"]), columns=int(e["columns"]),
                    table_width=int(e["table_width"]), allow_unverified=allow_unverified,
                    page_size=P.page_of(e))
        out.append({"entry": e, "ok": m.ok, "reason": m.reason.value,
                    "cell": m.row.cell if m.row is not None else None, "detail": m.detail})
    return out


def _kernel_sha() -> str:
    with open(P.KERNEL_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _rule_admits_entry(rules: list[dict], e: dict) -> bool:
    return any(r["cache_dtype"] == e["cache_dtype"] and P.page_of(r) == P.page_of(e)
               and P.ProfileRule(r).admits(
        int(e["rows"]), int(e["requests"]), int(e["columns"]), int(e["table_width"]))
        for r in rules)


def build_profile(plan: list[dict], name: str = P.DEFAULT_PROFILE,
                  evidence: dict | None = PHASE1_EVIDENCE, rule_spec: list[dict] | None = None,
                  capture_plan: list[dict] | None = None, plan_rows: bool = False) -> dict:
    """Profile dict: exact rows from ``plan``; rules from ``rule_spec`` (default
    :data:`RULE_SPEC`) sampled with the builtin capture + ``capture_plan``; with
    ``plan_rows``, capture entries no rule admits become exact rows."""
    sha = _kernel_sha()
    if sha != KERNEL_SHA256:
        raise ValueError(f"kernel source sha256 {sha} != profiled {KERNEL_SHA256} "
                         f"({KERNEL_GIT_REV}): update KERNEL_SHA256/evidence for the new kernel")
    caps = {P.PAGE_SIZE: capture_shapes(capture_plan), PAGE16: capture_shapes_p16(capture_plan),
            PAGE8: capture_shapes_p8(capture_plan)}
    rules = [make_rule(s, caps[P.page_of(s)])
             for s in (all_rule_specs() if rule_spec is None else rule_spec)]
    buckets = [make_bucket(s, bucket_extra(P.page_of(s))) for s in bucket_spec()]
    if plan_rows:
        plan = list(plan) + [e for e in capture_plan or []
                             if not _rule_admits_entry(rules, e)
                             and P.regime_of(e["rows"], e["requests"]) is not None]
    rows, seen = [], set()
    for e in plan:
        r = make_row(e, evidence)
        if r["sig"] in seen:
            continue
        seen.add(r["sig"])
        rows.append(r)
    return {
        "schema_version": P.SCHEMA_VERSION,
        "feature": P.FEATURE,
        "profile": name,
        "api_version": P.PROFILE_API_VERSION,
        "backend": P.BACKEND,
        "kernel": {"module": P.KERNEL_MODULE, "path": P.KERNEL_REL, "sha256": sha,
                   "git_rev": KERNEL_GIT_REV, "toolchain": "flydsl 0.3.3 (gfx1250-atom-20261015)"},
        "kernel_env": dict(P.KERNEL_ENV_DEFAULTS),
        "contract": {"index_heads": P.INDEX_HEADS, "head_dim": P.HEAD_DIM,
                     "page_size": P.PAGE_SIZE, "page_sizes": list(P.PAGE_SIZES),
                     "compress_ratio": P.COMPRESS_RATIO,
                     "q_dtype": "bf16", "score_divisor": "sqrt(head_dim)",
                     "cache_align_bytes": P.CACHE_ALIGN_BYTES,
                     "metadata_dtypes": ["int32", "int64"]},
        "golden": {"revision": GOLDEN_REVISION, "module": "qsa_golden.py",
                   "comparator": "compare_logits", "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL,
                   "extra_checks": ["visible_groups exact", "row_starts == 0", "no NaN",
                                    "outside-mask bit-identical (-inf)"]},
        "generated_by": {"tool": TOOL, "version": TOOL_VERSION, "mode": "emit",
                         "plan": "builtin (target buckets + Phase-1 GPU suite shapes)",
                         "rules": "RULE_SPEC (bounds corners/mid x Stage-3 ATOM capture, exact "
                                  "bf16 rows and AgentX recipe samples) at P 4/16/8; "
                                  "RULE_SPEC_FP8 (bounds corners/mid x R2 intake matrix, exact "
                                  "fp8 rows) at P 4/8/16",
                         "evidence": PHASE1_SOURCE, "evidence_policy": EVIDENCE_NOTE,
                         "note": "exact rows take precedence over range rules; pending rows "
                                 "and rules are never selected at runtime"},
        "rows": rows,
        "rules": rules,
        "buckets": buckets,
    }


def _isa_link(path: str) -> tuple[str, str, str]:
    """(base sha, final sha, justification) of one no-device ISA-identity gate evidence."""
    with open(path, "rb") as fh:
        blob = fh.read()
    ev = json.loads(blob)
    n, tot = int(ev.get("p4_identical", -1)), int(ev.get("p4_total", 0))
    mark = _ISA_CHANGED_NAME.get(ev.get("final_kernel_sha256"))
    mism = list(ev.get("p4_mismatch") or [])
    if mark:   # ROUND-R2B: only the declared ISA-changed builds may differ
        ident = ev.get("p4_identity") or {}
        mism = [c for c in mism if mark not in str((ident.get(c) or {}).get("final_kernel_name"))]
        n = tot - len(mism)
    if tot < 1 or n != tot or mism or ev.get("hazard_violations"):
        raise ValueError(f"{path}: page-4 ISA identity {n}/{tot}, mismatch {ev.get('p4_mismatch')}, "
                         f"hazards {ev.get('hazard_violations')}: evidence does not carry over")
    base, final = ev.get("base_kernel_sha256"), ev.get("final_kernel_sha256")
    gate, what = _GATE_TEXT.get(final, ("no-device", "every page-4 specialization"))
    return base, final, (
        f"page-4 ISA byte-identical {str(base)[:12]} -> {str(final)[:12]} "
        f"({gate} no-device gate: {n}/{tot} builds = {what} x int32/int64 positions, "
        f"normalized final ISA + kernel name; evidence sha256 "
        f"{hashlib.sha256(blob).hexdigest()[:16]})")


def load_isa_identity(paths) -> list[str] | None:
    """ISA-identity justification chain from the no-device gate evidences, or None.

    ``paths``: one path or a list (``emit --isa-identity-evidence`` may repeat). Each
    evidence (``summarize_*_isa_cpu.py`` -> ``isa-evidence.json``) must show every
    page-4 specialization (normalized final ISA + kernel name) identical between its
    base and final kernel and no WMMA hazard; the links must chain
    :data:`PAGE4_EVIDENCE_KERNEL_SHA256` -> ... -> :data:`KERNEL_SHA256`."""
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]
    links = [_isa_link(p) for p in paths]
    at = PAGE4_EVIDENCE_KERNEL_SHA256
    for (base, final, _t), p in zip(links, paths):
        if base != at:
            raise ValueError(f"{p}: evidence base {str(base)[:12]} does not chain from {at[:12]} "
                             f"(order the evidences {PAGE4_EVIDENCE_KERNEL_SHA256[:12]} -> "
                             f"{KERNEL_SHA256[:12]})")
        at = final
    if at != KERNEL_SHA256:
        raise ValueError(f"evidence chain ends at {str(at)[:12]}, not the profiled kernel "
                         f"{KERNEL_SHA256[:12]}")
    return [t for _b, _f, t in links]


def load_isa_links(paths) -> list[tuple[str, str, str]]:
    """ROUND-R2E: the (base, final, justification) links of ``paths`` (chain-checked like
    :func:`load_isa_identity`)."""
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    texts = load_isa_identity(paths)
    return [(b, f, t) for (b, f, _t), t in zip((_isa_link(p) for p in paths), texts)]


def carry_chain_evidence(data: dict, links) -> dict:
    """ROUND-R2E: every ``pass`` record (rows, rules, buckets; any page size / dtype) verified
    on a kernel of :data:`ISA_IDENTITY_CHAIN` gets the identity links from its kernel onward;
    a record whose variant kind changed ISA on one of those links, or whose kernel is not in
    the chain / not covered by ``links``, becomes ``pending`` (stale record kept)."""
    out = json.loads(json.dumps(data))
    chain = list(ISA_IDENTITY_CHAIN)
    by_base = {b: (f, t) for b, f, t in links}
    for r in out["rows"] + (out.get("rules") or []) + (out.get("buckets") or []):
        g = r["golden"]
        if g.get("status") != "pass":
            continue
        sha = g.get("kernel_sha256")
        if sha == KERNEL_SHA256:
            continue
        why = None
        if sha not in chain:
            why = f"evidence kernel {str(sha)[:12]} not on the ISA-identity chain"
        else:
            at, texts = sha, []
            while at != KERNEL_SHA256 and why is None:
                nxt = by_base.get(at)
                if nxt is None:
                    why = f"no ISA-identity link from {at[:12]}"
                    break
                fin, txt = nxt
                kinds = {P.variant_kind(v) for v in ISA_CHANGED_VIDS.get(fin, ())}
                if P.variant_kind(r.get("vid", "")) in kinds:
                    why = f"variant ISA changed on the link -> {fin[:12]}"
                    break
                texts.append(txt)
                at = fin
        if why is not None:
            r["golden"] = {"status": "pending", "stale": g,
                           "reason": f"ROUND-R2E: {why}; run the GPU gen_profile check"}
            continue
        for txt in texts:
            if txt not in g.get("isa_identity", ""):
                g["isa_identity"] = (g["isa_identity"] + "; " + txt) if g.get("isa_identity") \
                    else txt
    return out


def carry_page4_evidence(data: dict, identity) -> dict:
    """Page-4 ``pass`` records verified on another kernel: chain ``identity`` (one link or
    a list of links) onto them, or (no identity) downgrade them to ``pending`` with the
    stale record kept."""
    if isinstance(identity, (list, tuple)) and not identity:
        identity = None
    if isinstance(identity, (list, tuple)):
        for link in identity:
            data = carry_page4_evidence(data, link)
        return data
    out = json.loads(json.dumps(data))
    # ROUND-R2D: the ISA-changed vids of whichever link this is (not only the last link)
    changed = tuple(v for sha, vids in ISA_CHANGED_VIDS.items()
                    if identity and "-> " + sha[:12] in identity for v in vids)
    for r in out["rows"] + (out.get("rules") or []):
        g = r["golden"]
        if g.get("status") != "pass" or P.page_of(r) != P.PAGE_SIZE:
            continue
        if r.get("vid") in changed:
            r["golden"] = {"status": "pending", "stale": g,
                           "reason": "ROUND-R2B: variant ISA changed (persist fallback loop); "
                                     "run the GPU gen_profile check"}
            continue
        if g.get("kernel_sha256") == KERNEL_SHA256 and not identity:
            continue
        if identity:
            if identity not in g.get("isa_identity", ""):
                g["isa_identity"] = (g["isa_identity"] + "; " + identity) if g.get("isa_identity") \
                    else identity
        else:
            r["golden"] = {"status": "pending", "stale": g,
                           "reason": "kernel changed; run the ROUND-PAGE16 ISA-identity gate"}
    return out


def _same_variant(a: dict, b: dict) -> bool:
    return a.get("vid") == b.get("vid") and dict(a.get("params") or {}) == dict(b.get("params") or {})


def merge_golden(new: dict, old: dict, only_unverified: bool = False) -> dict:
    """Carry golden/notes from ``old`` onto unchanged rows (same sig and variant) and rules /
    buckets (same sig and variant, and every new sample already verified by the old pass).
    ``only_unverified`` (ROUND-R2E ``--keep-golden-from``): touch only records that are not
    ``pass`` yet, and only with ``old`` records that are not ``pending``."""
    out = json.loads(json.dumps(new))
    rows = {r["sig"]: r for r in old.get("rows", [])}

    def skip(r, o):
        return only_unverified and (r["golden"].get("status") == "pass"
                                    or o["golden"].get("status") == "pending")

    for r in out["rows"]:
        o = rows.get(r["sig"])
        if o is not None and _same_variant(r, o) and not skip(r, o):
            r["golden"], r["notes"] = o["golden"], o["notes"]
    rules = {r["sig"]: r for r in (old.get("rules") or []) + (old.get("buckets") or [])}
    for r in out["rules"] + (out.get("buckets") or []):
        o = rules.get(r["sig"])
        if o is None or not _same_variant(r, o) or skip(r, o):   # R2E: re-routed = new evidence
            continue
        old_keys = {P.sample_key(s) for s in o["samples"]}
        if all(P.sample_key(s) in old_keys for s in r["samples"]):
            g = json.loads(json.dumps(o["golden"]))
            if g.get("checked"):
                want = {P.sample_key(s) for s in r["samples"]}
                g["checked"] = [c for c in g["checked"] if P.sample_key(c) in want]
                g["samples_n"] = len(g["checked"])
            r["golden"], r["notes"] = g, o["notes"]
    return out


def _dump(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _strip_dynamic(data: dict) -> dict:
    """Fields `check` may legitimately update (golden status/stats, generated_by)."""
    out = json.loads(json.dumps(data))
    out.pop("generated_by", None)
    for r in out["rows"] + (out.get("rules") or []) + (out.get("buckets") or []):
        r.pop("golden", None)
        r.pop("notes", None)
    return out


def verify(path: str, plan: list[dict] | None = None, capture_plan: list[dict] | None = None,
           plan_rows: bool = False) -> list[str]:
    """Differences between the profile file and a fresh `emit` (empty = in sync)."""
    with open(path, encoding="utf-8") as fh:
        have = json.load(fh)
    want = build_profile(plan or builtin_plan(), have.get("profile", P.DEFAULT_PROFILE),
                         capture_plan=capture_plan, plan_rows=plan_rows)
    a, b = _strip_dynamic(have), _strip_dynamic(want)
    errs = []
    for key in sorted(set(a) | set(b)):
        if key not in ("rows", "rules", "buckets") and a.get(key) != b.get(key):
            errs.append(f"top-level {key}: {a.get(key)!r} != {b.get(key)!r}")
    qa = {r["id"]: r for r in (a.get("rules") or []) + (a.get("buckets") or [])}
    qb = {r["id"]: r for r in (b.get("rules") or []) + (b.get("buckets") or [])}
    for rid in sorted(set(qa) ^ set(qb)):
        errs.append(f"rule {rid} " + ("not produced by the spec" if rid in qa else
                                      "missing from the file"))
    for rid in sorted(set(qa) & set(qb)):
        if qa[rid] != qb[rid]:
            diff = sorted(k for k in set(qa[rid]) | set(qb[rid]) if qa[rid].get(k) != qb[rid].get(k))
            errs.append(f"rule {rid} differs in {diff}")
    try:
        P.validate_profile(have)
    except P.ProfileError as exc:
        errs.append(f"invalid: {exc}")
    ra = {r["sig"]: r for r in a.get("rows", [])}
    rb = {r["sig"]: r for r in b.get("rows", [])}
    for sig in sorted(set(ra) - set(rb)):
        errs.append(f"row {ra[sig]['cell']} not produced by the plan")
    for sig in sorted(set(rb) - set(ra)):
        errs.append(f"row {rb[sig]['cell']} missing from the file")
    for sig in sorted(set(ra) & set(rb)):
        if ra[sig] != rb[sig]:
            errs.append(f"row {ra[sig]['cell']} differs")
    ksha = have.get("kernel", {}).get("sha256")
    for r in have.get("rows", []):
        g = r["golden"]
        if g["status"] != "pass":
            continue
        if not (g.get("cases") or g.get("checked")):
            errs.append(f"row {r['cell']} is 'pass' without evidence")
        if not g.get("kernel_sha256"):
            errs.append(f"row {r['cell']} is 'pass' without the verified kernel_sha256")
        elif g["kernel_sha256"] != ksha and not g.get("isa_identity"):
            errs.append(f"row {r['cell']} 'pass' on kernel {g['kernel_sha256'][:12]} != {str(ksha)[:12]} "
                        "without isa_identity")
    return errs


# ---------------------------------------------------------------------------
# GPU golden check (root runs this; never on a no-device host)
# ---------------------------------------------------------------------------
def _golden_modules():
    gdir = os.environ.get("QSA_GOLDEN_DIR")
    if not gdir:
        raise SystemExit("set QSA_GOLDEN_DIR to the frozen golden_baseline directory")
    sys.path.insert(0, os.path.abspath(gdir))
    import qsa_golden as G  # noqa: E402
    import qsa_shapes as S  # noqa: E402
    return G, S


def golden_regime(rows: int, requests: int) -> str:
    """Fixture layout of qsa_golden.build_case for a row shape."""
    if rows == requests:
        return "draft"
    if rows == 4 * requests:
        return "decode"
    return "prefill"   # two-way split of the rows (qsa_golden._row_layout)


def sample_fixture(s: dict) -> dict:
    """Deterministic fixture of a rule sample: full-width context when the page budget
    allows (else the longest that fits), golden-native layout for rows == requests /
    rows == 4*requests, otherwise rows split evenly over min(rows, requests) requests."""
    rows, reqs, width = int(s["rows"]), int(s["requests"]), int(s["table_width"])
    page = P.page_of(s)
    pages_per_req = min(width, sample_max_pages(page) // reqs)
    max_ctx = tokens_per_page(page) * pages_per_req
    layout = "draft" if rows == reqs else "decode" if rows == 4 * reqs else "even"
    return {"max_ctx": max_ctx, "cache_pages": reqs * pages_per_req, "layout": layout}


def _even_layout(rows: int, reqs: int, ctx: int):
    """(token_to_request, query_positions): consecutive chunks over min(rows, reqs)
    requests, each ending at ``ctx`` (positions wrap mod ctx, never negative)."""
    import torch
    active = min(rows, reqs)
    counts = [rows // active + (i < rows % active) for i in range(active)]
    t2r = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(counts)])
    pos = torch.cat([torch.remainder(torch.arange(n, dtype=torch.int64) + (ctx - n), ctx)
                     for n in counts])
    return t2r, pos


def check_row(row: dict, api, G, S, seed: int = 0, profile: str | None = None) -> dict:
    rows, reqs = row["rows"]["min"], row["requests"]["min"]
    fx = row.get("fixture", {})
    width = row["table_width"]
    page = P.page_of(row)
    return _golden_run(api, G, S, name=row["cell"], cache_dtype=row["cache_dtype"], rows=rows,
                       reqs=reqs, cols=row["columns"], width=width,
                       max_ctx=int(fx.get("max_ctx", tokens_per_page(page) * width)),
                       cache_pages=int(fx.get("cache_pages", E2E_PAGES)),
                       layout=golden_regime(rows, reqs), seed=seed, profile=profile,
                       page_size=page)


def check_sample(rule: dict, s: dict, api, G, S, seed: int = 0,
                 profile: str | None = None) -> dict:
    """Golden-check one rule sample; the call must be served with the rule's variant."""
    page = P.page_of(rule)
    fx = sample_fixture(dict(s, page_size=page))
    name = f"{rule['id']}_r{s['rows']}q{s['requests']}_c{s['columns']}_w{s['table_width']}"
    rec = _golden_run(api, G, S, name=name, cache_dtype=rule["cache_dtype"], rows=int(s["rows"]),
                      reqs=int(s["requests"]), cols=int(s["columns"]),
                      width=int(s["table_width"]), max_ctx=fx["max_ctx"],
                      cache_pages=fx["cache_pages"], layout=fx["layout"], seed=seed,
                      profile=profile, page_size=page)
    if rec.get("vid") is not None and rec["vid"] != rule["vid"]:
        rec.update(pass_=False, error=f"served by {rec['vid']} != rule vid {rule['vid']}")
    out = {k: int(s[k]) for k in ("rows", "requests", "columns", "table_width")}
    out.update(fixture=fx, **rec)
    return out


def check_bucket_sample(b: dict, s: dict, api, G, S, seed: int = 0,
                        profile: str | None = None) -> dict:
    """Golden-check one bucket sample through the bucket path (``buckets=True``): served by
    the bucket's variant via the bucket, golden-exact; also records whether the real rows
    are bitwise equal to the unpadded call (``bitwise_unpadded``, informational: the
    unpadded call may route to another variant)."""
    page = P.page_of(b)
    fx = sample_fixture(dict(s, page_size=page))
    name = f"{b['id']}_r{s['rows']}q{s['requests']}_c{s['columns']}_w{s['table_width']}"
    rec = _golden_run(api, G, S, name=name, cache_dtype=b["cache_dtype"], rows=int(s["rows"]),
                      reqs=int(s["requests"]), cols=int(s["columns"]),
                      width=int(s["table_width"]), max_ctx=fx["max_ctx"],
                      cache_pages=fx["cache_pages"], layout=fx["layout"], seed=seed,
                      profile=profile, page_size=page, buckets=True)
    if rec.get("cell") is not None and not str(rec["cell"]).startswith(f"bucket:{b['id']}>"):
        rec.update(pass_=False, error=f"served by {rec['cell']}, not bucket {b['id']}")
    out = {k: int(s[k]) for k in ("rows", "requests", "columns", "table_width")}
    out.update(fixture=fx, **rec)
    return out


def _golden_run(api, G, S, *, name, cache_dtype, rows, reqs, cols, width, max_ctx, cache_pages,
                layout, seed=0, profile=None, page_size=P.PAGE_SIZE, buckets=None) -> dict:
    import torch

    profile = profile or os.environ.get("QSA_PROFILE_PATH") or P.DEFAULT_PROFILE
    case = S.make_case(name, layout if layout != "even" else "prefill", rows=rows, requests=reqs,
                       max_ctx=max_ctx, index_dtype=cache_dtype, table_width=width,
                       cache_pages=cache_pages, page_size=int(page_size))
    d = G.build_case(case, device="cuda", seed=seed)
    if layout == "even":
        t2r, pos = _even_layout(rows, reqs, max_ctx)
        d["token_to_request"] = t2r.to(device="cuda", dtype=d["token_to_request"].dtype)
        d["query_positions"] = pos.to(device="cuda", dtype=d["query_positions"].dtype)
    rs = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    args = (d["q"], d["compressed_k_cache"], d["page_table"], d["token_to_request"],
            d["query_positions"], d["context_lens"])
    # ROUND-R2E: buckets are ON by default in the op -> row / rule checks run unbucketed
    # explicitly (bucket samples pass buckets=True), so a check measures the entry's variant
    bkw = {"buckets": bool(buckets)}
    dec = api.decide(*args, 4, None, cols, rs, False, d["compressed_k_scale"],
                     profile=profile, allow_unverified=True, **bkw)
    rec: dict[str, Any] = {"decision": dec.reason.value, "vid": dec.vid, "cell": dec.cell}
    if not dec.ok:
        rec.update(pass_=False, error=f"not eligible: {dec.reason.value}: {dec.detail}")
        return rec
    out, d2 = api.try_qsa_paged_mqa_logits(*args, 4, None, cols, rs, False,
                                           d["compressed_k_scale"],
                                           profile=profile, allow_unverified=True, **bkw)
    logits, vis = out
    if buckets:
        ref = api.try_qsa_paged_mqa_logits(*args, 4, None, cols, None, False,
                                           d["compressed_k_scale"], profile=profile,
                                           allow_unverified=True, buckets=False)[0]
        if ref is not None:
            same = (torch.equal(vis, ref[1]) and bool(
                ((logits == ref[0]) | (torch.isnan(logits) & torch.isnan(ref[0]))).all()))
            rec["bitwise_unpadded"] = same
    torch.cuda.synchronize()
    gold, gvis, valid = G.reference_logits(*args, compress_ratio=4, max_columns=cols,
                                           compressed_k_scale=d["compressed_k_scale"])
    res = G.compare_logits(logits, vis, gold, gvis, valid, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)
    colix = torch.arange(cols, device=gold.device)
    inside = valid & (colix[None, :] < gvis.to(torch.int64)[:, None])
    oa, ob = logits[~inside], gold[~inside]
    outside_diff = int((~((oa == ob) | (torch.isnan(oa) & torch.isnan(ob)))).sum())
    nan = int(torch.isnan(logits).sum())
    rs_zero = bool((rs == 0).all())
    ok = bool(res.get("pass")) and outside_diff == 0 and nan == 0 and rs_zero
    rec.update(pass_=ok, golden=res, outside_diff=outside_diff, nan=nan, row_starts_zero=rs_zero)
    return rec


def _load_api():
    if __package__ in (None, ""):
        # file mode: load this package standalone (no `import aiter`, which touches the GPU /
        # JIT at import time), exactly like op_tests/test_qsa_index_scorer.py does
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "qsa_isc_gen", os.path.join(here, "__init__.py"), submodule_search_locations=[here])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["qsa_isc_gen"] = pkg
        spec.loader.exec_module(pkg)
        return sys.modules["qsa_isc_gen.api"]
    from . import api
    return api


def add_plan_samples(data: dict, plan: list[dict]) -> int:
    """Append every plan entry a rule admits to that rule's samples (source 'capture')."""
    n = 0
    for rule in data.get("rules") or []:
        R = P.ProfileRule(rule)
        have = {P.sample_key(s) for s in rule["samples"]}
        for e in plan:
            k = (int(e["rows"]), int(e["requests"]), int(e["columns"]), int(e["table_width"]))
            if (e["cache_dtype"] == rule["cache_dtype"] and P.page_of(e) == P.page_of(rule)
                    and k not in have and R.admits(*k)
                    and k[0] * k[2] * 4 <= P.MAX_PLANE_BYTES):
                rule["samples"].append({"rows": k[0], "requests": k[1], "columns": k[2],
                                        "table_width": k[3], "source": "capture"})
                have.add(k)
                n += 1
    return n


def check(path: str, out: str | None, cells: set | None, plan: list[dict] | None = None,
          only: str = "all") -> int:
    """GPU: golden-check rows and/or every sample of every rule; restamp ``golden``.

    The API matches against the profile being written: with ``--plan-file`` extra
    samples, the extended profile is staged at ``out`` (or in place) before checking.
    """
    import torch
    api = _load_api()
    G, S = _golden_modules()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    target = out or path
    if plan:
        print(f"[CHECK] plan: +{add_plan_samples(data, plan)} rule samples", flush=True)
    _dump(data, target)              # stage (rules stay pending until this run restamps them)
    os.environ["QSA_PROFILE_PATH"] = os.path.abspath(target)
    dev = torch.cuda.get_device_properties(0)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    meta = {"checked_at": stamp, "device": f"{dev.gcnArchName} cu={dev.multi_processor_count}",
            "kernel_sha256": data["kernel"]["sha256"], "golden_revision": GOLDEN_REVISION,
            "comparator": "qsa_golden.compare_logits", "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL}
    fails = 0
    for row in data["rows"] if only in ("all", "rows") else []:
        if cells and row["cell"] not in cells:
            continue
        print(f"[CHECK] {row['cell']} ...", flush=True)
        try:
            rec = check_row(row, api, G, S, profile=target)
        except Exception as exc:  # noqa: BLE001
            rec = {"pass_": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        ok = rec.pop("pass_")
        fails += not ok
        prev = row["golden"]
        row["golden"] = dict({"status": "pass" if ok else "fail", "checked": rec}, **meta)
        if prev.get("cases") or prev.get("stale_cases"):
            row["golden"]["phase1_cases"] = prev.get("cases") or prev.get("stale_cases")
            row["golden"]["phase1_kernel_sha256"] = (prev.get("kernel_sha256")
                                                     or prev.get("stale_kernel_sha256"))
        elif prev.get("phase1_cases"):
            row["golden"]["phase1_cases"] = prev["phase1_cases"]
            row["golden"]["phase1_kernel_sha256"] = prev.get("phase1_kernel_sha256")
        if ok and row.get("notes") == STALE_FP8_NOTE:
            row["notes"] = ""
        print(f"[{'PASS' if ok else 'FAIL'}] {row['cell']} {json.dumps(rec)[:400]}", flush=True)
        torch.cuda.empty_cache()
    for rule in (data.get("rules") or []) if only in ("all", "rules") else []:
        if cells and f"rule:{rule['id']}" not in cells and rule["id"] not in cells:
            continue
        recs, nfail = [], 0
        for i, s in enumerate(rule["samples"]):
            tag = (f"rule:{rule['id']} [{i + 1}/{len(rule['samples'])}] "
                   f"r{s['rows']}q{s['requests']} c{s['columns']} w{s['table_width']}")
            try:
                rec = check_sample(rule, s, api, G, S, profile=target)
            except Exception as exc:  # noqa: BLE001
                rec = {k: int(s[k]) for k in ("rows", "requests", "columns", "table_width")}
                rec.update(pass_=False, error=f"{type(exc).__name__}: {exc}"[:2000])
            ok = bool(rec.pop("pass_"))
            rec["pass"] = ok
            nfail += not ok
            recs.append(rec)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} "
                  f"{json.dumps(rec.get('golden', rec.get('error')))[:300]}", flush=True)
            torch.cuda.empty_cache()
        fails += nfail
        rule["golden"] = dict({"status": "fail" if nfail else "pass", "samples_n": len(recs),
                               "samples_failed": nfail, "checked": recs}, **meta)
        print(f"[RULE {'FAIL' if nfail else 'PASS'}] {rule['id']} {len(recs) - nfail}/{len(recs)}",
              flush=True)
    for b in (data.get("buckets") or []) if only in ("all", "buckets") else []:
        if cells and f"bucket:{b['id']}" not in cells and b["id"] not in cells:
            continue
        recs, nfail = [], 0
        for i, s in enumerate(b["samples"]):
            tag = (f"bucket:{b['id']} [{i + 1}/{len(b['samples'])}] "
                   f"r{s['rows']}q{s['requests']} c{s['columns']} w{s['table_width']}")
            try:
                rec = check_bucket_sample(b, s, api, G, S, profile=target)
            except Exception as exc:  # noqa: BLE001
                rec = {k: int(s[k]) for k in ("rows", "requests", "columns", "table_width")}
                rec.update(pass_=False, error=f"{type(exc).__name__}: {exc}"[:2000])
            ok = bool(rec.pop("pass_"))
            rec["pass"] = ok
            nfail += not ok
            recs.append(rec)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} bitwise_unpadded="
                  f"{rec.get('bitwise_unpadded')} "
                  f"{json.dumps(rec.get('golden', rec.get('error')))[:300]}", flush=True)
            torch.cuda.empty_cache()
        fails += nfail
        b["golden"] = dict({"status": "fail" if nfail else "pass", "samples_n": len(recs),
                            "samples_failed": nfail, "checked": recs}, **meta)
        print(f"[BUCKET {'FAIL' if nfail else 'PASS'}] {b['id']} {len(recs) - nfail}/{len(recs)}",
              flush=True)
    data["generated_by"] = dict(data["generated_by"], mode="check", checked_at=stamp)
    _dump(data, target)
    print(f"[CHECK] fails={fails} -> {target}")
    return 1 if fails else 0


def _read_plan(path: str | None) -> list[dict]:
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    return plan["plan"] if isinstance(plan, dict) else plan


def coverage(data: dict, plan: list[dict]) -> dict:
    """How every plan entry resolves (rules counted as if verified)."""
    res = classify_plan(data, plan, allow_unverified=True)
    by: dict = {}
    for c in res:
        key = c["cell"] if c["ok"] else f"uncovered:{c['reason']}"
        by[key] = by.get(key, 0) + 1
    return {"classes": len(res), "by": by,
            "uncovered": [dict(c["entry"], reason=c["reason"]) for c in res if not c["ok"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("emit", "verify", "check", "coverage"))
    ap.add_argument("--profile", default=os.path.join(P.CONFIG_DIR, f"{P.DEFAULT_PROFILE}.json"))
    ap.add_argument("--out")
    ap.add_argument("--plan-file", help="captured call classes: JSON list or {'plan': [...]}; "
                                        "rule-admitted entries become rule samples")
    ap.add_argument("--plan-rows", action="store_true",
                    help="emit/verify: plan entries no rule admits become exact rows")
    ap.add_argument("--keep-golden", action="store_true",
                    help="emit: keep golden/notes of unchanged rows/rules from the existing file")
    ap.add_argument("--keep-golden-from", action="append", default=[],
                    help="emit (repeatable, after --keep-golden): another profile file (e.g. a GPU "
                         "restamp on an earlier chain kernel) whose non-pending golden records "
                         "fill records that are not 'pass' yet (same sig + variant + samples)")
    ap.add_argument("--cells", help="comma-separated row cells / rule ids (check mode)")
    ap.add_argument("--isa-identity-evidence", action="append",
                    help="emit (repeatable, in chain order): no-device isa-evidence.json links "
                         "PAGE4_EVIDENCE_KERNEL_SHA256 -> ... -> KERNEL_SHA256 (records/page16, "
                         "then records/r2-kernel); chains page-4 evidence recorded on the "
                         "evidence kernel onto the profiled kernel (else it becomes pending)")
    ap.add_argument("--only", choices=("all", "rows", "rules", "buckets"), default="all",
                    help="check mode: what to golden-check")
    a = ap.parse_args(argv)
    cap = _read_plan(a.plan_file)
    if a.mode == "emit":
        out = a.out or a.profile
        data = build_profile(builtin_plan(), os.path.splitext(os.path.basename(out))[0],
                             capture_plan=cap, plan_rows=a.plan_rows)
        if a.keep_golden and os.path.isfile(out):
            with open(out, encoding="utf-8") as fh:
                data = merge_golden(data, json.load(fh))
            data["generated_by"] = dict(data["generated_by"], mode="emit+keep-golden")
        for extra in a.keep_golden_from:
            with open(extra, encoding="utf-8") as fh:
                data = merge_golden(data, json.load(fh), only_unverified=True)
        data = carry_chain_evidence(data, load_isa_links(a.isa_identity_evidence))
        P.validate_profile(data)
        _dump(data, out)
        print(f"wrote {out}: {len(data['rows'])} rows "
              f"({sum(r['golden']['status'] == 'pass' for r in data['rows'])} pass), "
              f"{len(data['rules'])} rules "
              f"({sum(r['golden']['status'] == 'pass' for r in data['rules'])} pass, "
              f"{sum(len(r['samples']) for r in data['rules'])} samples), "
              f"{len(data.get('buckets') or [])} buckets "
              f"({sum(b['golden']['status'] == 'pass' for b in data.get('buckets') or [])} pass)")
        for c in classify_plan(data, cap):
            if not c["ok"] and c["reason"] != P.Reason.ROW_UNVERIFIED.value:
                print(f"UNCOVERED {c['reason']} {json.dumps(c['entry'])}")
        return 0
    if a.mode == "verify":
        errs = verify(a.profile, builtin_plan(), capture_plan=cap, plan_rows=a.plan_rows)
        for e in errs:
            print("DRIFT", e)
        print("VERIFY", "PASS" if not errs else "FAIL")
        return 1 if errs else 0
    if a.mode == "coverage":
        with open(a.profile, encoding="utf-8") as fh:
            print(json.dumps(coverage(json.load(fh), cap), indent=1))
        return 0
    return check(a.profile, a.out, set(a.cells.split(",")) if a.cells else None, cap, a.only)


if __name__ == "__main__":
    sys.exit(main())
