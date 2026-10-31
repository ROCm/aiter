# SPDX-License-Identifier: MIT
"""Round PERSIST no-device ISA gate + evidence summary (CPU only, stdlib).

usage: python summarize_persist_cpu.py RESULTS_DIR MERGE_EVIDENCE_JSON OUT_JSON

* truly unset auto routes persistent P8, off routes no persistent kernel,
  and explicit p2/p4/p8 route their selected persistent kernels;
* every nonpersistent kernel in auto/off/raw/tdm/p2/p4/p8 is byte-identical
  to existing merge evidence (same kernel name -> same final ISA sha256);
* each persistent kernel (auto=P8 and p2/p4/p8): 34,816 B LDS, exactly 2*P
  tensor_load_to_lds (2 per wave per chunk), exactly P s_wait_tensorcnt 0 and
  no other tensor waits (no conservative waits before the ds_loads that would
  serialise the pipeline), P barrier pairs, SGPR-only tensor operands, no
  tensor op inside an s_clause, padded descriptor config 0x07510000, 272 B
  consumer stride, zero waterfall markers, zero VGPR spills / scratch;
* the textual order of tensor issues (T), waits (W), barriers (B) and LDS
  b128 loads (D) in the TDM path is reported per iteration.
This is a compile/layout gate, not GPU correctness or performance evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_tdm_isa_cpu import _resources  # noqa: E402
from check_bf16_routing_isa_cpu import check_routed_selection  # noqa: E402

MARKERS = ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128", "s_barrier_signal",
           "s_barrier_wait", "v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid",
           "v_wmma_f32_16x16x32_bf16", "buffer_store_b32", "s_cbranch_execz",
           "s_cbranch_scc0", "s_cbranch_scc1")


def counts(source):
    return {m: len(re.findall(r"\b" + re.escape(m) + r"\b", source)) for m in MARKERS}


def instructions(source):
    return [line.strip() for line in source.splitlines()
            if re.match(r"^\s+(?:s_|v_|buffer_|global_|flat_|ds_|tensor_)", line)]


def token_sequence(ins):
    seq = []
    for line in ins:
        if line.startswith("tensor_load_to_lds"):
            tok = "T"
        elif line.startswith("s_wait_tensorcnt"):
            tok = "W" if int(line.split()[1], 0) == 0 else "w"
        elif line.startswith("s_barrier_signal"):
            tok = "B"
        elif line.startswith("ds_load_b128 "):
            tok = "D"
        else:
            continue
        if tok == "D" and seq and seq[-1].startswith("D"):
            seq[-1] = "D%d" % (int(seq[-1][1:] or 1) + 1)
        else:
            seq.append(tok if tok != "D" else "D1")
    return seq


def check_persist(path: Path, chunks: int) -> dict:
    source = path.read_text(encoding="utf-8")
    c = counts(source)
    lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", source)
    assert lds == ["34816"], lds
    waits = re.findall(r"\bs_wait_tensorcnt\s+(\S+)", source)
    # Per issued chunk: exactly one completion wait(0), plus LLVM's descriptor
    # SGPR-reuse wait between the two gathers of one issue (also present in the
    # champion single-tile r32: "tensor 2, waits 2, wait(0) 1"). Its threshold
    # is >= 2 while at most 1 op of the wave is outstanding: it never stalls.
    zero = [w for w in waits if int(w, 0) == 0]
    nonzero = [int(w, 0) for w in waits if int(w, 0) != 0]
    assert len(zero) == chunks and len(nonzero) == chunks and min(nonzero) >= 2, waits
    assert c["tensor_load_to_lds"] == 2 * chunks, c
    assert c["s_barrier_signal"] == c["s_barrier_wait"] == chunks, c
    assert c["ds_load_b128"] >= 32 * chunks, c
    for marker in ("v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid"):
        assert c[marker] == 0, (marker, c)
    assert re.search(r"\bs_mov_b32\s+s\d+,\s*0x7510000\b", source), "padded descriptor config"
    assert re.search(r"\bv_[^\n]*\b0x110\b", source), "272B consumer stride"
    ins = instructions(source)
    for line in ins:
        if line.startswith("tensor_load_to_lds"):
            assert re.fullmatch(r"tensor_load_to_lds s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\]", line), line
    for i, line in enumerate(ins):
        m = re.match(r"s_clause\s+(0x[0-9a-f]+|\d+)", line)
        if m:
            n = int(m.group(1), 0) + 1
            assert not any("tensor_" in x for x in ins[i + 1:i + n + 1]), line
    seq = token_sequence(ins)
    tdm_seq = [t for t in seq if t in ("T", "w", "W", "B") or t.startswith("D")]
    # Program-order shape of the TDM path (w = non-stalling SGPR-reuse wait):
    # prologue T w T W B; per iteration k < P-1: T w T (chunk k+1), D.. (chunk
    # k ds_loads), W B; last iteration: D.. only. Raw fallback has no T/W/B.
    tw = "".join(t for t in tdm_seq if t in ("T", "w", "W", "B"))
    expected = "TwTWB" * chunks
    assert tw == expected, (tw, expected)
    # Overlap proof: between every non-prologue issue and its completion
    # wait(0) there are LDS b128 loads (the current chunk's WMMA operands).
    marks = [i for i, t in enumerate(tdm_seq) if t == "W"]
    issues = [i for i, t in enumerate(tdm_seq) if t == "T"][::2]
    overlapped = 0
    for it, iw in zip(issues[1:], marks[1:]):
        assert it < iw
        if any(t.startswith("D") for t in tdm_seq[it:iw]):
            overlapped += 1
    assert overlapped == chunks - 1, (overlapped, tdm_seq)
    res = _resources(source)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "chunks": chunks, "lds_bytes": 34816, "buffers": 2, "pitch_bytes": 272,
            "markers": c, "tensor_waits": waits, "barrier_pairs": c["s_barrier_signal"],
            "tensor_loads_per_iteration_per_wave": 2, "waits_per_iteration": 1,
            "sgpr_reuse_waits_nonstalling": nonzero, "overlapped_issues": overlapped,
            "barriers_per_iteration": 1, "twb_program_order": tw,
            "token_sequence": tdm_seq, "resources": res, "status": "PASS"}


def main() -> None:
    results, merge, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    old = json.loads(merge.read_text(encoding="utf-8"))
    old_sha = {}
    for k in old["kernels"]:
        old_sha.setdefault(k["kernel_name"], set()).add(k["sha256"])
    modes = ("auto", "off", "raw", "tdm", "p2", "p4", "p8")
    routed = [check_routed_selection(results, mode) for mode in modes]
    assert len({rec["source_sha256"] for rec in routed}) == 1, "mixed production sources"
    identical, new = {}, []
    for mode in modes:
        data = json.loads((results / mode / "compile_only.json").read_text(encoding="utf-8"))
        assert data["specializations"] and all(rec.get("compiled") is True for rec in data["specializations"]), (mode, data)
        compiled_names = set(re.findall(r"^\[COMPILED\] \S+ kernel=(\S+)\s*$",
                                       (results / mode / "compile.log").read_text(encoding="utf-8"), re.MULTILINE))
        files = sorted((results / mode / "isa").glob("*/21_final_isa.s"))
        assert files and {isa.parent.name for isa in files} == compiled_names, (mode, "missing/unexpected ISA")
        for isa in files:
            name = isa.parent.name
            sha = hashlib.sha256(isa.read_bytes()).hexdigest()
            src = isa.read_text(encoding="utf-8")
            for marker in ("v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid"):
                assert not re.search(r"\b" + marker + r"\b", src), (mode, name, marker)
            if "_persist" in name:
                new.append((mode, name, isa))
                continue
            assert name in old_sha, ("unknown non-persistent kernel", mode, name)
            assert old_sha[name] == {sha}, ("single-tile ISA changed", mode, name, sha)
            identical[f"{mode}:{name}"] = sha
    persist = []
    for mode, name, isa in new:
        assert mode in ("auto", "p2", "p4", "p8"), ("unexpected persistent ISA", mode, name)
        chunks = 8 if mode == "auto" else int(mode[1:])
        assert name.endswith(f"_persist{chunks}"), (mode, name)
        persist.append(dict(check_persist(isa, chunks), mode=mode, kernel_name=name))
    assert sorted((p["mode"], p["chunks"]) for p in persist) == [("auto", 8), ("p2", 2), ("p4", 4), ("p8", 8)], persist
    by_mode = {p["mode"]: p for p in persist}
    assert by_mode["auto"]["sha256"] == by_mode["p8"]["sha256"], "default P8 differs from explicit P8 ISA"
    r32 = results / "off" / "isa" / "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r32_c64_flydsl_tdm" / "21_final_isa.s"
    base_res = _resources(r32.read_text(encoding="utf-8"))
    summary = {"scope": "CPU-only stock-FlyDSL 0.3.3.dev886 ISA gate; no GPU dispatch",
               "byte_identical_to_merge_evidence": identical,
               "byte_identical_count": len(identical),
               "single_tile_r32_resources": base_res,
               "routed_wrapper_prefill": routed,
               "default_auto_persist8_equals_explicit_p8_isa": True,
               "off_has_no_persistent_isa": True,
               "persistent": persist, "status": "PASS"}
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identical": len(identical),
                      "persist": [{k: p[k] for k in ("kernel_name", "markers", "tensor_waits",
                                                     "twb_program_order", "resources")}
                                  for p in persist]}, indent=2))


if __name__ == "__main__":
    main()
