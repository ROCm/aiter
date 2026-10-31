# SPDX-License-Identifier: MIT
"""CPU-only round-F fp8-cache Q LDS checks (no FlyDSL import or GPU).

Q remains bf16 for both cache dtypes.  Only the fp8-cache specialization stages
its packed dwords into LDS; bf16 keeps the original global vector Q loads.
Adapted from round B: all 65536 bf16 encodings (including NaN payloads), one
writer per dword, complete aligned vector reads, clamped scalar staging, and
build-time helper selection with one unconditional fp8-only allocator/barrier.

These models and AST checks do not prove actual FlyDSL lowering, hardware
synchronization, GPU FMA bit-equivalence, or performance.  GPU compile/trace and
correctness gates are still required; the separate WMMA builder is unchanged.

    python test_q_lds_cpu.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

import audit_addressing_cpu as A

HEADS = 4
HDIM = 128
Q_DWORDS = HEADS * HDIM // 2
STEPS = HDIM // 8
THREAD_CHOICES = (32, 64, 128, 256, 512, 1024, 96)


def stage_q_cpu(q, threads=256, is_fp8=True):
    """Model fp8 cooperative scalar staging, or the unchanged bf16 direct path."""
    assert q.device.type == "cpu" and q.dtype == torch.bfloat16 and q.is_contiguous()
    assert threads > 0
    source = q.view(torch.int32).reshape(-1, Q_DWORDS)
    stage_steps = (Q_DWORDS + threads - 1) // threads if is_fp8 else 0
    owners = [0] * Q_DWORDS
    staged = torch.empty_like(source) if is_fp8 else source
    for qi in range(stage_steps):
        for tid in range(threads):
            q_dw = tid + qi * threads
            safe_q_dw = min(q_dw, Q_DWORDS - 1)
            assert 0 <= safe_q_dw < Q_DWORDS, "unconditional scalar load leaves Q row"
            word = source[:, safe_q_dw]
            if q_dw < Q_DWORDS:
                staged[:, safe_q_dw] = word
                owners[safe_q_dw] += 1
    assert owners == [int(is_fp8)] * Q_DWORDS, "missing/duplicate writer or bf16 staged Q"
    assert torch.equal(source, staged), "Q staging changed packed bf16 bits"
    restored = torch.empty_like(source)
    reads = [0] * Q_DWORDS
    for s in range(STEPS):
        for h in range(HEADS):
            # LDS offsets are dwords; bf16 global offsets are the same * 4,
            # with row * 1024 supplied by q_base.
            idx = h * (HDIM // 2) + s * 4
            assert idx * 4 % 16 == 0 and 0 <= idx <= Q_DWORDS - 4
            restored[:, idx:idx + 4] = staged[:, idx:idx + 4]
            for dw in range(idx, idx + 4):
                reads[dw] += 1
    assert reads == [1] * Q_DWORDS, "missing or duplicate consumer dword"
    return restored.view(torch.bfloat16).reshape_as(q)


def _calls(root, name):
    return [n for n in ast.walk(root) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == name]


def _ancestors(node, root):
    parents = {child: parent for parent in ast.walk(root)
               for child in ast.iter_child_nodes(parent)}
    result = []
    while node in parents:
        node = parents[node]
        result.append(node)
    return result


def _assignments(root, name):
    return [n for n in ast.walk(root) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def _const_eval(node, values):
    """Evaluate only the arithmetic/conditional subset used by q_stage_steps."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.IfExp):
        return _const_eval(node.body if _const_eval(node.test, values) else node.orelse, values)
    if isinstance(node, ast.BinOp):
        left, right = _const_eval(node.left, values), _const_eval(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise AssertionError("unsupported build-time stage arithmetic: " + ast.unparse(node))


def _selected_q_helpers(build, is_fp8):
    """Resolve the top-level Python dtype branch, without importing FlyDSL."""
    choices = [n for n in build.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "is_fp8"]
    assert len(choices) == 1, "Q helper selection must be one build-time dtype branch"
    selected = {}
    for node in choices[0].body if is_fp8 else choices[0].orelse:
        if not isinstance(node, ast.Assign):
            continue
        assert len(node.targets) == 1
        targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else [node.targets[0]]
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        for target, value in zip(targets, values):
            if isinstance(target, ast.Name) and target.id in ("q_storage", "q_load", "q_barrier"):
                assert isinstance(value, ast.Name), "helpers must be selected Python closures"
                selected[target.id] = value.id
    return selected


def _structural_checks(check):
    path = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_kernel")
    helpers = {n.name: n for n in build.body if isinstance(n, ast.FunctionDef)}
    kernel = helpers["kernel"]
    fp8 = _selected_q_helpers(build, True)
    bf16 = _selected_q_helpers(build, False)
    check("build-time fp8/bf16 selection binds all three Q closures; no traced dtype branch",
          fp8 == {"q_storage": "_q_lds_fp8", "q_load": "_q_load_fp8", "q_barrier": "_q_barrier_fp8"}
          and bf16 == {"q_storage": "_q_direct_bf16", "q_load": "_q_load_bf16", "q_barrier": "_q_barrier_bf16"}
          and not any(isinstance(n, ast.Name) and n.id == "is_fp8" for n in ast.walk(kernel)))
    storage_fp8, storage_bf16 = helpers[fp8["q_storage"]], helpers[bf16["q_storage"]]
    load_fp8, load_bf16 = helpers[fp8["q_load"]], helpers[bf16["q_load"]]
    barrier_fp8, barrier_bf16 = helpers[fp8["q_barrier"]], helpers[bf16["q_barrier"]]

    alloc = _calls(build, "fx.SharedAllocator")
    check("exactly one static 16-byte-aligned allocator, only in selected fp8 storage",
          len(alloc) == 1 and len(_calls(storage_fp8, "fx.SharedAllocator")) == 1
          and ast.unparse(alloc[0]) == "fx.SharedAllocator(base_alignment=16, static=True)"
          and [ast.unparse(n) for n in storage_fp8.body]
          == ["return fx.SharedAllocator(base_alignment=16, static=True).allocate(fx.Array[fx.Int32, q_dwords, 16]).peek().ptr"]
          and not any(isinstance(n, (ast.If, ast.For, ast.While)) for n in _ancestors(alloc[0], storage_fp8))
          and [ast.unparse(n) for n in storage_bf16.body] == ["return q_rsrc"])
    check("LDS allocation is one 1024-byte row, q_dwords = q_row_bytes // 4",
          any(ast.unparse(n) == "q_row_bytes = NUM_HEADS * HEAD_DIM * 2" for n in build.body)
          and any(ast.unparse(n) == "per_page = NUM_HEADS * HEAD_DIM" for n in build.body)
          and any(ast.unparse(n) == "q_dwords = q_row_bytes // 4" for n in build.body))

    check("fp8 Q arithmetic uses exactly one aligned int32x4 LDS read, no global Q vector load",
          [ast.unparse(n) for n in load_fp8.body]
          == ["return Vec(fx.ptr_load(q_ptr + h * (HEAD_DIM // 2) + s * 4, result_type=Vec.make_type(4, fx.Int32)))"]
          and len(_calls(load_fp8, "fx.ptr_load")) == 1
          and not _calls(load_fp8, "_ld_vec")
          and not any(isinstance(n, ast.Name) and n.id == "q_rsrc"
                      for call in _calls(kernel, "_ld_vec") for n in ast.walk(call))
          and not _calls(kernel, "fx.ptr_load"))
    check("bf16 Q helper retains original global int32x4 address, no LDS allocator/read/barrier",
          [ast.unparse(n) for n in load_bf16.body]
          == ["return _ld_vec(q_ptr, q_base + h * (HEAD_DIM * 2) + s * 16, 4, fx.Int32)"]
          and not any(_calls(h, name) for h in (storage_bf16, load_bf16, barrier_bf16)
                      for name in ("fx.SharedAllocator", "fx.ptr_load", "fx.ptr_store", "fx.gpu.barrier"))
          and [ast.unparse(n) for n in barrier_bf16.body] == ["pass"])

    stage = [n for n in kernel.body if isinstance(n, ast.For) and ast.unparse(n.target) == "qi"]
    stage_steps = _assignments(build, "q_stage_steps")
    tripcount_ok = len(stage_steps) == 1 and len(stage) == 1
    if tripcount_ok:
        value = stage_steps[0].value
        tripcount_ok = ast.unparse(value) == "(q_dwords + threads - 1) // threads if is_fp8 else 0"
        for threads in THREAD_CHOICES:
            for is_fp8 in (True, False):
                tripcount_ok &= _const_eval(value, {"q_dwords": Q_DWORDS, "threads": threads, "is_fp8": is_fp8}) \
                    == ((Q_DWORDS + threads - 1) // threads if is_fp8 else 0)
        tripcount_ok &= ast.unparse(stage[0].iter) == "range_constexpr(q_stage_steps)"
    check("staging tripcount selected at build time: ceil(dwords/threads) fp8, zero bf16", tripcount_ok)

    scalar = [n for n in _calls(kernel, "_ld_i32")
              if isinstance(n.args[0], ast.Name) and n.args[0].id == "q_rsrc"]
    stores = _calls(kernel, "fx.ptr_store")
    check("one unconditional clamped scalar Q stage load, one predicated clamped LDS store",
          len(stage) == len(scalar) == len(stores) == 1
          and [ast.unparse(n) for n in stage[0].body]
          == ["q_dw = tid + qi * threads", "safe_q_dw = _imin(q_dw, fx.Int32(q_dwords - 1))",
              "q_word = _ld_i32(q_rsrc, q_base + safe_q_dw * 4)",
              "if q_dw < q_dwords:\n    fx.ptr_store(q_word, q_ptr + safe_q_dw)"]
          and not any(isinstance(n, ast.If) for n in _ancestors(scalar[0], kernel))
          and any(isinstance(n, ast.If) and ast.unparse(n.test) == "q_dw < q_dwords"
                  for n in _ancestors(stores[0], kernel)))

    q_storage, q_barrier, q_load = (_calls(kernel, name) for name in ("q_storage", "q_barrier", "q_load"))
    barriers = _calls(build, "fx.gpu.barrier")
    check("one unconditional fp8-only barrier helper; kernel calls after stage before all consumers",
          len(barriers) == len(_calls(barrier_fp8, "fx.gpu.barrier")) == 1
          and [ast.unparse(n) for n in barrier_fp8.body] == ["fx.gpu.barrier()"]
          and len(q_storage) == len(q_barrier) == len(q_load) == len(stores) == 1
          and q_storage[0].lineno < stores[0].lineno < q_barrier[0].lineno < q_load[0].lineno
          and not any(isinstance(n, (ast.If, ast.For, ast.While))
                      for n in _ancestors(q_barrier[0], kernel) + _ancestors(barriers[0], barrier_fp8))
          and not _calls(kernel, "fx.gpu.barrier"))
    check("Q pointer initialized outside branches after every descriptor, before stage",
          any(ast.unparse(n) == "q_ptr = q_storage(q_rsrc)" for n in kernel.body)
          and any(ast.unparse(n) == "q_base = row * per_page * 2" for n in kernel.body)
          and len(q_storage) == 1
          and all(n.lineno < q_storage[0].lineno for n in _calls(kernel, "_plane"))
          and not any(isinstance(n, (ast.If, ast.For, ast.While))
                      for n in _ancestors(q_storage[0], kernel)))
    check("arithmetic Q call uses selected pointer/helper in unchanged head/step loops",
          len(q_load) == 1 and ast.unparse(q_load[0]) == "q_load(q_ptr, q_base, h, s)"
          and [ast.unparse(n.iter) for n in _ancestors(q_load[0], kernel) if isinstance(n, ast.For)]
          == ["range_constexpr(NUM_HEADS)", "range_constexpr(steps)"]
          and [ast.unparse(n) for n in _calls(kernel, "_ld_vec")]
          == ["_ld_vec(kv_rsrc, k_base + s * k_bytes, k_dwords, fx.Int32)"])
    check("one-thread-per-column mapping and output safety predicates/clamps retained",
          any(ast.unparse(n) == "col = ptile * threads + tid" for n in kernel.body)
          and any(ast.unparse(n) == "out_idx = _imin(row * columns + col, rows * columns - 1)"
                  for n in kernel.body)
          and any(isinstance(n, ast.If) and ast.unparse(n.test) == "col_ok"
                  and len(_calls(n, "_st_f32")) == 1 for n in kernel.body)
          and any(isinstance(n, ast.If) and ast.unparse(n.test) == "writer"
                  and len(_calls(n, "_st_i32")) == 2 for n in kernel.body))


def _audit_checks(check):
    fp8_ok = bf16_ok = True
    for threads in THREAD_CHOICES:
        for rows, columns in ((1, 1), (2, 3), (2, 255), (2, 257)):
            width = (columns + A.PAGE - 1) // A.PAGE
            blocks = (columns + threads - 1) // threads
            for fmt in ("fp8", "bf16"):
                a = A.audit_analytic(rows, 2, width, 8, fmt, columns)
                e = A.audit_enumerate(rows, 2, width, 8, fmt, columns, threads=threads)
                ok = a["ok"] and e["ok"] and e["full"] and not e["unaligned"]
                ok &= e["stores_by_plane"]["logits"] == rows * columns
                ok &= e["stores_by_plane"]["visible_groups"] == e["stores_by_plane"]["row_starts"] == rows
                if fmt == "fp8":
                    expected_loads = rows * blocks * threads * ((Q_DWORDS + threads - 1) // threads)
                    expected_stores = rows * blocks * Q_DWORDS
                    expected_reads = rows * e["cols_visited"] * HEADS * STEPS
                    ok &= a["sizes"]["q_lds"] == e["sizes"]["q_lds"] == 1024
                    ok &= a["max_offset_plus_width"]["q_lds"] == e["max_offset_plus_width"]["q_lds"] == 1024
                    ok &= a["max_offset_plus_width"]["q"] == e["max_offset_plus_width"]["q"] == rows * 1024
                    ok &= e["accesses_by_plane"]["q"] == expected_loads
                    ok &= e["stores_by_plane"]["q_lds"] == expected_stores
                    ok &= e["accesses_by_plane"]["q_lds"] == expected_stores + expected_reads
                    fp8_ok &= ok
                else:
                    ok &= "q_lds" not in a["sizes"] and "q_lds" not in e["sizes"]
                    ok &= "q_lds" not in e["accesses_by_plane"] and "q_lds" not in e["stores_by_plane"]
                    ok &= e["accesses_by_plane"]["q"] == rows * e["cols_visited"] * HEADS * STEPS
                    bf16_ok &= ok
    check("fp8 audits count scalar surplus-lane loads, 256 LDS writers/block and tail-lane reads", fp8_ok)
    check("bf16 scalar audit retains direct vector Q loads, unchanged stores, no LDS plane/access", bf16_ok)

    boundary_ok = True
    for threads in (96, 256, 512):
        e = A.audit_enumerate(65535, 2, 65, 8, "fp8", 257, threads=threads, full_limit=0)
        a = A.audit_analytic(65535, 2, 65, 8, "fp8", 257)
        blocks = (257 + threads - 1) // threads
        boundary_ok &= e["ok"] and a["ok"] and not e["full"]
        boundary_ok &= e["max_offset_plus_width"]["q"] == 65535 * 1024
        boundary_ok &= e["stores_by_plane"]["q_lds"] == e["rows_visited"] * blocks * Q_DWORDS
    check("boundary audits retain full stage coverage and clamped last-row loads", boundary_ok)

    checker = A.Checker({"q_lds": 1024})
    checker.acc("q_lds", 4, 16, alignment=16)
    checker.store("q_lds", 1024, 4, alignment=4)
    check("audit rejects misaligned LDS vector reads and out-of-bounds LDS stores",
          len(checker.unaligned) == 1 and len(checker.oob) == 1)
    check("round-C WMMA audit and partial-wave/i32-cap gate cases preserved",
          A.audit_wmma(7, 2, 32, 128, 70)["ok"]
          and all(name in {c[0] for c in A.cases()} for name in (
              "wm_rows7_cols70_bf16", "wm_rows13_bf16", "wm_rows70_bf16",
              "wm_draft_16x65536_bf16", "wm_prefill_512x65536_bf16", "wm_i32_cap_8191x65536_bf16")))


def check_q_lds_cpu():
    """Return failure count, printing one PASS line per independent check."""
    fails = 0

    def check(tag, condition):
        nonlocal fails
        print("[%s] %s" % ("PASS" if condition else "FAIL", tag))
        fails += int(not condition)

    print("== fp8-only Q LDS: bf16 bits, one writer, aligned coverage, build-time selection ==")
    # Compare integer views: floating NaN equality loses the bit-preservation proof.
    bits = torch.arange(65536, dtype=torch.int32, device="cpu").to(torch.int16)
    fixtures = (bits, bits.roll(1))  # every encoding in both halves of a packed dword
    for threads in THREAD_CHOICES:
        for is_fp8 in (True, False):
            ok = True
            for fixture in fixtures:
                q = fixture.view(torch.bfloat16).reshape(-1, HEADS, HDIM)
                restored = stage_q_cpu(q, threads, is_fp8)
                ok &= torch.equal(q.view(torch.int16), restored.view(torch.int16))
            check("%s all 65536 bf16 encodings, both packed halves; threads=%d"
                  % ("fp8 LDS" if is_fp8 else "bf16 direct / zero stage", threads), ok)

    address_ok = True
    for rows in (1, 4, 16, 64, 512, 65535):
        q_extent = rows * Q_DWORDS * 4
        for row in (0, rows - 1):
            for threads in THREAD_CHOICES:
                for qi in range((Q_DWORDS + threads - 1) // threads):
                    for tid in range(threads):
                        safe = min(tid + qi * threads, Q_DWORDS - 1)
                        off = row * Q_DWORDS * 4 + safe * 4
                        address_ok &= 0 <= off and off + 4 <= q_extent <= 2**31 - 1
    check("all staged scalar loads, including inactive stores, stay in row/plane and fit i32", address_ok)
    _audit_checks(check)
    _structural_checks(check)
    print("Q LDS CPU checks: %s" % ("PASS" if not fails else "%d FAILURE(S)" % fails))
    return fails


if __name__ == "__main__":
    raise SystemExit(bool(check_q_lds_cpu()))
