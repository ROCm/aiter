# SPDX-License-Identifier: MIT
"""Round PERSIST: CPU-only proofs of the bf16 r32 persistent prefill kernel.

Stdlib only (no aiter/torch/FlyDSL import, no GPU). Covers:

* the single-tile A/B path and every other specialization are source-identical
  to champion adea96d5 (per-function source-segment SHA-256);
* the two-buffer plan (``persist_schedule``) and that the REAL kernel body,
  abstractly interpreted from its AST for 2/4/8 chunks, emits exactly that
  plan: per iteration issue(k+1 -> buf (k+1)%2), consume(k <- buf k%2),
  tensor_wait(0), barrier; prologue issue(0 -> 0), wait, barrier;
* a 4-wave adversarial interleaving simulator (random wave order, TDM data
  landing anywhere between issue and the issuing wave's wait): no read of an
  incomplete/foreign tile, no overwrite of an in-flight or being-read buffer,
  <= 2 outstanding tensor ops per wave; a schedule with the end-of-iteration
  barrier removed is caught (negative control);
* byte-exact K fragments for every chunk/wave/lane/n-tile/k-chunk, by
  executing the REAL ``_persist_issue_tdm`` helper against a mock and reading
  with the REAL consumer address expressions, versus an independent per-lane
  global-load reference (all uint16 encodings, invalid pages, width tails);
* store coverage exactly once over the real grid, LDS bounds/alignment, the
  SGPR-only issue guard and the visibility bound, and host routing/env switch.

Run: python -B aiter/ops/flydsl/kernels/mqa_logits/_g2/test_persist_cpu.py -v
"""
from __future__ import annotations

import sys as _n64_sys, pathlib as _n64_path  # ROUND-NATIVE64 seam support
_n64_sys.path.insert(0, str(_n64_path.Path(__file__).resolve().parent))
from native64_seam import ensure_i32, unseam  # noqa: E402

import ast
import hashlib
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_k_tdm_cpu import (PADDING_SENTINEL, fallback_fragments,  # noqa: E402
                            padded_fragment_offsets)
import test_bf16_routing_cpu as R  # noqa: E402

SOURCE = HERE.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)
TOP = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
BUILD = TOP["_build_wmma_kernel"]
HELPERS = {n.name: n for n in BUILD.body if isinstance(n, ast.FunctionDef)}
KERNEL = HELPERS["kernel_r32_persist"]

ROW = 272
TILE = 17408
LDS = 2 * TILE
CONSTS = dict(PAGE_SIZE=4, WMMA_COLS=64, WMMA_N=16, WMMA_N_TILES=4, WMMA_WAVES=4,
              WMMA_ROWS_PER_WAVE=4, TDM_K_ROW_BYTES=ROW, TDM_K_TILE_BYTES=TILE,
              TDM_PERSIST_BUFFERS=2, TDM_PERSIST_LDS_BYTES=LDS)

# Source-segment SHA-256 of every function of champion adea96d5 that this round
# must not touch (generated from `git show adea96d5:<kernel>`).
BASE_SOURCE = {
    "_build_kernel":
        "3e51cf5cfdf135bcd69202f81a7bacf451c35bbcab14bf57480c83c5ae00d74b",
    "_build_wmma_fp8_kernel._frag":
        "567e4791becb96ac9778eca3837a38f9ac08c697e9585a0a6bcebefe9b30002f",
    "_build_wmma_fp8_kernel._i01":
        "7a34767c4b2fecb6e1e2653b7d8fe56af957fe19358c3d8ef824ec2b2174d4ed",
    "_build_wmma_fp8_kernel._kfrag_fp8":
        "c9cb4269edf1c72417f335765ee28988dfc44a6f3b47dd5fc8040246e0e8bac3",
    "_build_wmma_fp8_kernel._lane_cols":
        "207c1621f2cb47bf7cfaed40a035475c7dcb2f6b389447a592b90ddc9d8771f4",
    "_build_wmma_fp8_kernel._pair_base_raw":
        "50d248cfdd4b747d09fb46a1eb685063e705e41632fa4b5123d7398e4fa900ec",
    "_build_wmma_fp8_kernel._pair_base_tdm":
        "4a765b383ac71b423dc43f5647e4f405cc0bf13a4409fd81583b7c8ddad22451",
    "_build_wmma_fp8_kernel._pair_frag_raw":
        "def46f8972213684b5ee1de560f3e031c89ca783cffcbdc4fc643f7df6bccaab",
    "_build_wmma_fp8_kernel._pair_frag_tdm":
        "2b11ab9d6ae7f8bb167259def5085b3e65408e776d4b7d94d61b3bdffcfefbfe",
    "_build_wmma_fp8_kernel._pair_stage_raw":
        "6e52619e348b864aee8d59f5f7fbd6622088a462ec82203b0f81dc1509195e68",
    "_build_wmma_fp8_kernel._pair_stage_tdm":
        "852cebeb930a6fe3936d294ec796a76cac85ab3cf309ec465c3827f5c222a189",
    "_build_wmma_fp8_kernel._pair_storage_raw":
        "40afec7d3d1a5062b1a2f5996edaf27d4b88ec5d93853a1d2a752bd13ce4b24c",
    "_build_wmma_fp8_kernel._pair_storage_tdm":
        "33a66c5aa3b6e9ba33950378849223f42b21c9a143ee40ba0949215e69313b03",
    "_build_wmma_fp8_kernel._pair_values":
        "5255fd1c690fc447b75e8a4caf7c866f8ae42d927a912fe71227a30d41e22624",
    "_build_wmma_fp8_kernel._pass_masks":
        "85c62eb30be0c7718ce33348cf6f4cca3be41e48ae479d4959a15339258bfd96",
    "_build_wmma_fp8_kernel._pass_plan":
        "23cb6846260e45a533130a170b5fe686f63139c05486f742aa23caa7c4fb2901",
    "_build_wmma_fp8_kernel._pass_values":
        "d204ee271214ea822dd5ef72e3defba2d162e583559d225d4368659a1e450d29",
    "_build_wmma_fp8_kernel._pick_vis":
        "18f48d6d91166188574ca2d53a050448b9061b51c6182bc6ce0a7ab0ddfc0fdf",
    "_build_wmma_fp8_kernel._q_frags":
        "c0c15724c33685bdba99fe4f774fe645868e27521903a5ff1e646ad15f04ddf1",
    "_build_wmma_fp8_kernel._r32_guard":
        "d317b9f0f4baabcfee5148bf5a432fb7399bf2aeba5d9d0d84a4c3d44c9a7dfa",
    "_build_wmma_fp8_kernel._rmem8":
        "e529633da2a64de525afd9925e816b20aab8315794237676a8c89fb815b53c82",
    "_build_wmma_fp8_kernel._rows_meta":
        "5dccc4b5424730c4167029e9ec47d3758385e474f4ef7143e3acd823c4d78338",
    "_build_wmma_fp8_kernel._store_offsets":
        "869ecf0099228d237795bd4e99063ae1a2f4293cd6796ed6fd9efdab4ed27597",
    "_build_wmma_fp8_kernel._tile":
        "79166bf918861885af494ac158d6359382c25f2dbfb14170cddea6f57b9559fb",
    "_build_wmma_fp8_kernel._tile_pair":
        "1d63bbe78170dd3586bf10f140d4967a1ffd41ee86e8e9b1e5dbaf00817fed2c",
    "_build_wmma_fp8_kernel.kernel":
        "97cda5e1d8419b70f59e650e266b8a8199b5b4e00f6cb483a067c0ecb9f8759c",
    "_build_wmma_fp8_kernel.kernel_r32":
        "3e7aadbf867dd927a3babcd67c488f9b9c78824692f5670a0fc0c788738f48f5",
    "_build_wmma_fp8_kernel.launch":
        "55706805924bede04b541c9faa9cc0e26f5697f7b43d417d86373a87787d40c3",
    "_build_wmma_kernel._frag":
        "567e4791becb96ac9778eca3837a38f9ac08c697e9585a0a6bcebefe9b30002f",
    "_build_wmma_kernel._i01":
        "7a34767c4b2fecb6e1e2653b7d8fe56af957fe19358c3d8ef824ec2b2174d4ed",
    "_build_wmma_kernel._lane_cols":
        "8f4ca28984203f73582e48f6b389a035b20e002a2619dfd17889564e58428800",
    "_build_wmma_kernel._pair_base_raw":
        "82e5ae7787a707637e0ea0c639f6ae4770b3ac73ad6e3d25ade10d95f37241d7",
    "_build_wmma_kernel._pair_base_tdm":
        "a4bf3447dd3e3bdc45b2c53be5285e94a1d3a753112d4272d493d315717ba948",
    "_build_wmma_kernel._pair_frag_raw":
        "385973e6df4e37d6b70930ba7556b8ccc883033e4345214b18eee6150e6d5ca0",
    "_build_wmma_kernel._pair_frag_tdm":
        "c64a5b2b9e4cce67ff021f548792ba33b3b9fb20ad5978840027f36f736bd6b0",
    "_build_wmma_kernel._pair_stage_raw":
        "6e52619e348b864aee8d59f5f7fbd6622088a462ec82203b0f81dc1509195e68",
    "_build_wmma_kernel._pair_stage_tdm":
        "448682fb44ac8f306e4a3c276ddec16ba680e4e9c465d216b5ad25b39bdbb3e7",
    "_build_wmma_kernel._pair_storage_raw":
        "40afec7d3d1a5062b1a2f5996edaf27d4b88ec5d93853a1d2a752bd13ce4b24c",
    "_build_wmma_kernel._pair_storage_tdm":
        "e5b934bea79f8da5242d200001c39746af27ff6b0f62aee8b374f1c08d72880b",
    "_build_wmma_kernel._pair_values":
        "4f4b32fb4f2205d064c6df52251731407947c9671567207f92654648c75093b0",
    "_build_wmma_kernel._pass_masks":
        "4591616610a045188610edfa75d93c99e0f491c0da9fdba2b54abff426970a08",
    "_build_wmma_kernel._pass_plan":
        "23cb6846260e45a533130a170b5fe686f63139c05486f742aa23caa7c4fb2901",
    "_build_wmma_kernel._pass_values":
        "756670a8973bfb3f20cecf932d5b136cbcd26f8091b7bbe2426266526b19115d",
    "_build_wmma_kernel._pick_vis":
        "18f48d6d91166188574ca2d53a050448b9061b51c6182bc6ce0a7ab0ddfc0fdf",
    "_build_wmma_kernel._q_frags":
        "c0c15724c33685bdba99fe4f774fe645868e27521903a5ff1e646ad15f04ddf1",
    "_build_wmma_kernel._r16_stage_active":
        "4f3bbdc202a423f6fa800cbd2e6a7700c066179d54cef8579a3f8bf2d7907217",
    "_build_wmma_kernel._r16_stage_tdm":
        "3eddab57371216191830120020e67335defe639aea8f7072925c03da1bd1e2cd",
    "_build_wmma_kernel._r16_storage_tdm":
        "0ee1af2ed64b8abb8f65a05957aa2cf6eaa72772e39aa9246fc70ff61ccef7da",
    "_build_wmma_kernel._r32_guard":
        "d317b9f0f4baabcfee5148bf5a432fb7399bf2aeba5d9d0d84a4c3d44c9a7dfa",
    "_build_wmma_kernel._r64_guard":
        "850fd86abd8f90a6c9b891559b87511d8d099fb8bf13a3e0f0c41440ebd1ee82",
    "_build_wmma_kernel._rmem8":
        "e529633da2a64de525afd9925e816b20aab8315794237676a8c89fb815b53c82",
    "_build_wmma_kernel._rows_meta":
        "5dccc4b5424730c4167029e9ec47d3758385e474f4ef7143e3acd823c4d78338",
    "_build_wmma_kernel._rows_meta_uniform":
        "9e00540fa64b4c8a6326ef9c70f30939723f76f8baca7c20551d51fb0adc5574",
    "_build_wmma_kernel._store_offsets":
        "869ecf0099228d237795bd4e99063ae1a2f4293cd6796ed6fd9efdab4ed27597",
    "_build_wmma_kernel._tile":
        "d0baa85675f85b4ed69d750d3314e5fa3046a8a935e48f710e49d5ab0de6eeec",
    "_build_wmma_kernel._tile_pair":
        "b983f2e30cf76aef5e6833add4241ca2fc1365b6462f9e127c52f8f462482a9d",
    "_build_wmma_kernel.kernel":
        "60a96f2f65d4c47c77ab122adb1520341eaf38cf23c113c8c1e25d637a84ab42",
    "_build_wmma_kernel.kernel_r32":
        "bcc06222873c6bf9ca02f6bed96e09ebaa07a00d17b6b8fe1d84e115e0b23c6c",
    "_build_wmma_kernel.kernel_r64":
        "48285b3551b9a1a6283c4f613fdd751ba9dc1cf7ad8898c2632655ed827b28bb",
    "_build_wmma_kernel.kernel_tdm_r16":
        "9cf2202e871e923b93ed2e4e612bc4b5f32c25b16a9cd7078f15d673aebe743d",
    "_build_wmma_kernel.launch":
        "4271c946759439f69abb5c720fdd328686c6500c45e63d4d1caf063f8bdb0929",
    "_build_wmma_kernel_for":
        "a58e3e67c88eee4a42d6911678efb9695ac6093a3388341f58cd2a9ae5082cb6",
    "_check_align":
        "9246b25e0409cf1d5050fdc7d98f4d2ca8ae9237081cdab1538402b134d149b4",
    "_check_plane":
        "da447e2afb483a0277905fbe288d688ac5ecdfdcff12db2a44f066966cd25f42",
    "_dummy_scale":
        "efc8bee603627d2297b7c62b8521be0bfca1a66d5235d45e0af9608801318fef",
    "_e4m3_byte_to_f32":
        "578fa851b89b1f4d8e5c1a84462d063f566486a8a75eaa155a222468dbe7c823",
    "_e4m3x4_to_bf16x2":
        "503283545bba52d1ca93d810def579155aec4aa23b9b764f6598f0150fc9a539",
    "_imax":
        "c3d22c14aba3e5f15e3a10fbdec5bc4da9e08be326fc313fbbb68f505d785108",
    "_imin":
        "25f7f0d3519dd5709ed8ca9c6948b4c8ca48f4ecf7b82ce63be77a62ee7ae633",
    "_ld_f32":
        "31e7563f40845d9fa13c7f7e0aff2fd77e7aa17de24eb3cfeac1e08d29b0ff20",
    "_ld_i32":
        "a87e3284fe49971913dd7c3cb48ce622a230de152de60857dcf495e1765aca66",
    "_ld_scalar":
        "cd4990dd2d339fecc740e12f25fcf0f6539fd246793d45dd65bb34c6c3a49579",
    "_ld_uniform_i32":
        "39a37aae1da394361114d2e73ff505fa74e78b34ef0b03a7a68c9f008ff50221",
    "_ld_vec":
        "c0c1bb5b7f0095bd168c05ad8b4a08975cbe9b45397837e51d6952bffd19ff98",
    "_plane":
        "4a4fde949be01e3e33a6063c40375f5e99f07ff7552f31d0be05d71e263b02c3",
    "_plane_bytes":
        "3fd06db4a33f93c143a644789e89345b29d39e512ed8911816abbd1d960d3137",
    "_raw":
        "508629f67795e4f4c41471fbc3e934184e1ee04b0de8648bb9efa3b89dd4d65d",
    "_resolve_bf16_k_load":
        "b6146c56be9055f0170b1f6171998789c4abc33382b41fa239e0ee7ac08e7c6b",
    "_resolve_fp8_k_load":
        "d613c44fb950d731b463cd6054bd3581a96358f919a5dbc13b90814fd2a05205",
    "_run":
        "be4a11379884f09698723f829797cbbff7dc1c2e4950375e93b1346133fcc0a3",
    "_s32":
        "99a9efd33bb2adb2c8587e2c7e7a9b892b839f9a22929e6f39014319c14506a8",
    "_st_f32":
        "bd6398990afa851ca0cc941135682ad3ec76e83d9c64b445b7da6c90582b5b8a",
    "_st_i32":
        "2be66c5096b72933e07bdcda8ae918a01c09b1360c34ca9c4cfb975f334aaf22",
    "_st_scalar":
        "f37038bd65c208dc8af161f81a0140340bc60a4e82232928a70b1d74003ac269",
    "_tdm_fp8_k_gather":
        "cf5692a30ead227b3cb997ea0038d42f24f72e47ce4c57cb57a652e11932f90f",
    "_tdm_k_gather":
        "faa10625c733a97c3ce0b7454905708f4e051e31843b6c330b981bd39e396b97",
    "_to_i32_checked":
        "7c6ece6cb82845d7a72c2b640d5406f2f6adec3eac974f51c1fdf09b1917752a",
    "wmma_bf16_rows_per_block":
        "5f883871ef8362847b38b7301142b3cd5072b6a00f6dd286ec100bd56caaf24e",
    "wmma_rows_per_block":
        "7da67fdd4f536ff39645c9726d2f8b1e797cda4d8edad5809ca7b3d372b404dc",
}


def _segment_sha(node):
    # ROUND-NATIVE64: hash after mapping the query_positions seam back to its
    # pre-NATIVE64 text (native64_seam.unseam), i.e. the pinned functions may
    # differ from adea96d5 by exactly that seam and nothing else.
    return hashlib.sha256(unseam(ast.get_source_segment(TEXT, node)).encode()).hexdigest()


def _exec(nodes, ns):
    module = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    ensure_i32(ns)
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns


def _schedule_fn():
    return _exec([TOP["persist_schedule"]], dict(CONSTS))["persist_schedule"]


def _calls(root, name):
    return [n for n in ast.walk(root) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _reuse_branch():
    return next(n for n in KERNEL.body if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse")


def _fallback_branch():
    return next(n for n in KERNEL.body if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse == 0")


def _kernel_assign(name, root=KERNEL):
    values = [n.value for n in ast.walk(root) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(values) == 1, (name, len(values))
    return values[0]


def _eval(node, env):
    return eval(compile(ast.Expression(body=node), "<expr>", "eval"), dict(env))  # noqa: S307


def _persist_next(chunks):
    ns = _exec([HELPERS["_persist_next"]], {"persist_chunks": chunks})
    return ns["_persist_next"]


def interpret_kernel_schedule(chunks):
    """Abstractly execute the real `if reuse:` body; return TDM/LDS events."""
    env = dict(CONSTS, persist_chunks=chunks, range_constexpr=range, span0=0, kbase=0,
               _persist_next=_persist_next(chunks))
    events = []

    def run(stmts, guard):
        for st in stmts:
            if isinstance(st, ast.For):
                assert ast.unparse(st.iter.func) == "range_constexpr", ast.unparse(st)
                for v in _eval(st.iter, env):
                    env[st.target.id] = v
                    run(st.body, guard)
            elif isinstance(st, ast.Assign):
                try:
                    env[st.targets[0].id] = _eval(st.value, env)
                except Exception:  # noqa: BLE001 - DSL values are not modelled
                    env.pop(getattr(st.targets[0], "id", None), None)
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                name = ast.unparse(st.value.func)
                if name == "fx.rocdl.tdm_ops.tensor_wait":
                    assert guard is None, f"wait nested in guard {guard}"
                    assert ast.unparse(st.value) == "fx.rocdl.tdm_ops.tensor_wait(0)"
                    events.append(("wait",))
                elif name == "fx.gpu.barrier":
                    assert guard is None, f"barrier nested in guard {guard}"
                    events.append(("barrier",))
                elif name == "_persist_issue_tdm":
                    assert guard is not None and guard.startswith("issue["), guard
                    col0 = _eval(st.value.args[4], env)
                    buf = _eval(st.value.args[-1], env)
                    assert col0 % 64 == 0
                    # The SGPR guard of the issue is the same chunk's flag.
                    assert guard == f"issue[{col0 // 64}]", (guard, col0)
                    events.append(("issue", col0 // 64, buf))
            elif isinstance(st, ast.If):
                test = ast.unparse(st.test)
                if test.startswith("issue["):
                    idx = _eval(st.test.slice, env)
                    run(st.body, f"issue[{idx}]")
                elif test == "col0 < vmax":
                    values = _calls(st, "_pair_values")
                    assert len(values) == 1 and ast.unparse(values[0].args[1]) == "kbuf"
                    assert not _calls(st, "fx.gpu.barrier") and not _calls(st, "fx.rocdl.tdm_ops.tensor_wait")
                    assert not _calls(st, "_persist_issue_tdm")
                    dwords = env["kbuf"]
                    assert (dwords * 4) % TILE == 0
                    events.append(("consume", env["col0"] // 64, dwords * 4 // TILE))
                elif test == "col0 >= vmax":
                    assert not _calls(st, "_pair_values")
                    assert not _calls(st, "fx.gpu.barrier") and not _calls(st, "_persist_issue_tdm")
                else:
                    raise AssertionError("unexpected guard " + test)

    run(_reuse_branch().body, None)
    return events


# ---------------------------------------------------------------------------
# 4-wave adversarial simulator of a schedule (events per wave, shared LDS)
# ---------------------------------------------------------------------------
class Hazard(AssertionError):
    pass


def simulate(schedule, rng, waves=4, groups_per_wave=2, land="random"):
    """Every wave runs `schedule`. Buffer content = per-group chunk ids."""
    content = {b: [None] * (waves * groups_per_wave) for b in (0, 1)}
    pending = {w: [] for w in range(waves)}          # in-flight ops (buf, group, chunk)
    reading = {w: None for w in range(waves)}
    pc = {w: 0 for w in range(waves)}
    at_barrier = set()
    max_out = 0

    def land_op(op):
        b, g, c = op
        if any(r is not None and r[0] == b for r in reading.values()):
            raise Hazard(f"TDM write into buffer {b} while it is being read")
        content[b][g] = c

    def check_read(w, b, c):
        if content[b] != [c] * len(content[b]):
            raise Hazard(f"wave {w} reads chunk {c} from buffer {b} holding {content[b]}")

    steps = 0
    while True:
        live = [w for w in range(waves) if pc[w] < len(schedule) or reading[w] is not None]
        if not live:
            break
        # Random landing of any in-flight op (TDM completes asynchronously).
        if land in ("random", "early"):
            for w in range(waves):
                for op in list(pending[w]):
                    if land == "early" or rng.random() < 0.3:
                        pending[w].remove(op)
                        land_op(op)
        runnable = [w for w in live if w not in at_barrier]
        if not runnable:
            raise Hazard("deadlock")
        w = rng.choice(runnable)
        steps += 1
        if reading[w] is not None:
            b, c = reading[w]
            check_read(w, b, c)          # data must be unchanged for the whole read
            reading[w] = None
            continue
        ev = schedule[pc[w]]
        pc[w] += 1
        if ev[0] == "issue":
            _, c, b = ev
            for g in range(groups_per_wave):
                pending[w].append((b, w * groups_per_wave + g, c))
            max_out = max(max_out, len(pending[w]))
            if max_out > 3:
                raise Hazard("more than 3 outstanding tensor ops per wave")
        elif ev[0] == "wait":
            for op in pending[w]:
                land_op(op)
            pending[w].clear()
        elif ev[0] == "barrier":
            at_barrier.add(w)
            if len(at_barrier) == waves:      # every wave arrived: release all
                at_barrier.clear()
        elif ev[0] == "consume":
            _, c, b = ev
            check_read(w, b, c)
            reading[w] = (b, c)
    for w in range(waves):
        if pending[w]:
            raise Hazard("kernel exits with in-flight tensor ops")
    return max_out, steps


# ---------------------------------------------------------------------------
# Mock DSL values for executing the real producer helper
# ---------------------------------------------------------------------------
class B:
    def __init__(self, v):
        self.v = bool(v)

    def __and__(self, o):
        return B(self.v and (o.v if isinstance(o, B) else bool(o)))

    def select(self, a, b):
        return a if self.v else b


class V:
    def __init__(self, v):
        self.v = int(v.v if isinstance(v, V) else v)

    @staticmethod
    def _o(o):
        return o.v if isinstance(o, V) else int(o)

    def __add__(self, o): return V(self.v + self._o(o))
    __radd__ = __add__
    def __sub__(self, o): return V(self.v - self._o(o))
    def __mul__(self, o): return V(self.v * self._o(o))
    __rmul__ = __mul__
    def __floordiv__(self, o):
        assert self.v >= 0
        return V(self.v // self._o(o))
    def __mod__(self, o): return V(self.v % self._o(o))
    def __lt__(self, o): return B(self.v < self._o(o))
    def __le__(self, o): return B(self.v <= self._o(o))
    def __ge__(self, o): return B(self.v >= self._o(o))
    def __gt__(self, o): return B(self.v > self._o(o))


class LDSModel:
    def __init__(self):
        self.mem = bytearray(PADDING_SENTINEL * (LDS // len(PADDING_SENTINEL)))
        self.starts = []

    def gather(self, source, row_ids, byte_off):
        """Padded TDM write model: 64 data DWORDs then 4 skipped, per row."""
        assert len(row_ids) == 8 and byte_off % 128 == 0, (len(row_ids), byte_off)
        assert 0 <= byte_off and byte_off + 8 * ROW <= LDS, byte_off
        self.starts.append(byte_off)
        for dw in range(8 * 64):
            row_id = row_ids[dw // 64]
            src = row_id * 256 + (dw % 64) * 4
            assert 0 <= src <= len(source) - 4
            dst = byte_off + (dw + (dw // 64) * 4) * 4
            self.mem[dst:dst + 4] = source[src:src + 4]


def run_issue(lds, source, table_flat, width, req, col0, num_pages, buf, wave):
    gathered = []

    def ld_uniform(rsrc, off):
        off = off.v if isinstance(off, V) else off
        assert off % 4 == 0 and 0 <= off < 4 * len(table_flat), off
        return V(table_flat[off // 4])

    def gather(KV, ptr, row_ids, num_pages_, byte_off):
        ids = [r.v if isinstance(r, V) else r for r in row_ids]
        off = byte_off.v if isinstance(byte_off, V) else byte_off
        gathered.append((ids, off))
        lds.gather(source, ids, off)

    ns = dict(CONSTS, range_constexpr=range, _ld_uniform_i32=ld_uniform, _tdm_k_gather=gather,
              fx=SimpleNamespace(Int32=lambda x: x,
                                 rocdl=SimpleNamespace(wave_id=lambda: V(wave))))
    _exec([HELPERS["_persist_issue_tdm"]], ns)["_persist_issue_tdm"](
        None, None, None, V(req), V(col0), V(width), V(num_pages), buf)
    return gathered


class PersistSourceIdentityCPU(unittest.TestCase):
    def test_every_base_function_is_source_identical(self):
        for name, digest in BASE_SOURCE.items():
            owner, _, inner = name.partition(".")
            node = (({n.name: n for n in TOP[owner].body if isinstance(n, ast.FunctionDef)}[inner])
                    if inner else TOP[owner])
            with self.subTest(name=name):
                self.assertEqual(_segment_sha(node), digest)

    def test_default_selection_and_builder_prefix_unchanged(self):
        selection = next(n.value for n in BUILD.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "selected_kernel" for t in n.targets))
        self.assertEqual(ast.unparse(selection),
                         "(kernel_tdm_r16 if use_tdm else kernel) if rows_per_block == 16 else kernel_r64 if rows_per_block == 64 else kernel_r32")
        # Every persistent override is gated by the trace-time `persist` flag
        # (False unless k_load == "persist"): validation, name, selection, grid.
        override = [ast.unparse(n.body[-1]) for n in BUILD.body
                    if isinstance(n, ast.If) and ast.unparse(n.test) == "persist"]
        self.assertEqual(override, ["k_load = 'tdm'",
                                    "kernel_name = kernel_name + f'_persist{persist_chunks}'",
                                    "selected_kernel = kernel_r32_persist",
                                    "launch.cols_per_block = cols_per_block * persist_chunks"])
        self.assertEqual(ast.unparse(_kernel_assign("persist", BUILD)), "k_load == 'persist'")


class PersistScheduleCPU(unittest.TestCase):
    def test_two_buffer_plan(self):
        sched = _schedule_fn()
        for chunks in (2, 4, 8):
            ev = sched(chunks)
            issues = [e for e in ev if e[0] == "issue"]
            consumes = [e for e in ev if e[0] == "consume"]
            self.assertEqual(issues, [("issue", k, k % 2) for k in range(chunks)])
            self.assertEqual(consumes, [("consume", k, k % 2) for k in range(chunks)])
            self.assertEqual(ev.count(("wait",)), chunks)
            self.assertEqual(ev.count(("barrier",)), chunks)
            for k in range(chunks):
                ci = ev.index(("consume", k, k % 2))
                ii = ev.index(("issue", k, k % 2))
                between = ev[ii:ci]
                # RAW: the chunk's gathers are waited and published before use.
                self.assertIn(("wait",), between)
                self.assertGreater(len(between) - 1 - between[::-1].index(("barrier",)),
                                   between.index(("wait",)))
                if k + 2 < chunks:
                    # WAR: buffer k%2 is reissued only after a barrier following consume k.
                    ni = ev.index(("issue", k + 2, k % 2))
                    self.assertIn(("barrier",), ev[ci:ni])
                if k + 1 < chunks:
                    # Overlap: chunk k+1 is issued BEFORE chunk k is consumed.
                    self.assertLess(ev.index(("issue", k + 1, (k + 1) % 2)), ci)
            self.assertEqual(ev[-1], ("consume", chunks - 1, (chunks - 1) % 2))
        self.assertEqual(sched(1), [("issue", 0, 0), ("wait",), ("barrier",), ("consume", 0, 0)])
        for bad in (0, 3, 16):
            with self.assertRaises(ValueError):
                sched(bad)

    def test_real_kernel_body_emits_exactly_the_plan(self):
        sched = _schedule_fn()
        for chunks in (2, 4, 8):
            with self.subTest(chunks=chunks):
                self.assertEqual(interpret_kernel_schedule(chunks), sched(chunks))

    def test_per_iteration_wait_and_barrier_counts(self):
        for chunks in (2, 4, 8):
            ev = interpret_kernel_schedule(chunks)
            # Split at consumes: iteration k = events after consume k-1 up to consume k.
            cuts = [i for i, e in enumerate(ev) if e[0] == "consume"]
            prologue = ev[:cuts[0]]
            self.assertEqual(prologue, [("issue", 0, 0), ("wait",), ("barrier",), ("issue", 1, 1)])
            for k in range(1, chunks):
                body = ev[cuts[k - 1] + 1:cuts[k]]
                expect = [("wait",), ("barrier",)] + ([("issue", k + 1, (k + 1) % 2)] if k + 1 < chunks else [])
                self.assertEqual(body, expect, (chunks, k))

    def test_adversarial_four_wave_interleavings(self):
        sched = _schedule_fn()
        for chunks in (2, 4, 8):
            for seed in range(150):
                for land in ("random", "early", "late"):
                    max_out, _ = simulate(sched(chunks), random.Random(seed * 7 + chunks), land=land)
                    self.assertEqual(max_out, 2)

    def test_simulator_catches_missing_barrier_and_wait(self):
        sched = _schedule_fn()
        ev = sched(4)
        # Drop the end-of-iteration-0 barrier: the next issue may overwrite a read.
        cut = ev.index(("consume", 0, 0)) + 2
        assert ev[cut] == ("barrier",)
        no_barrier = ev[:cut] + ev[cut + 1:]
        no_wait = [e for i, e in enumerate(ev) if not (e == ("wait",) and i > 3)]
        for broken in (no_barrier, no_wait):
            caught = 0
            for seed in range(300):
                try:
                    simulate(broken, random.Random(seed), land="random")
                except Hazard:
                    caught += 1
            self.assertGreater(caught, 0)


class PersistBytesCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patterns = list(range(1 << 16))
        random.Random(0x9E5).shuffle(patterns)
        cls.source = struct.pack("<65536H", *patterns)     # 128 pages of every uint16
        cls.num_pages = len(cls.source) // 1024

    def _check_block(self, chunks, table, req, span0, columns, rng):
        width = len(table[0])
        flat = [p for row in table for p in row]
        lds = LDSModel()
        kbuf_expr = _kernel_assign("kbuf", _reuse_branch())
        base_expr = HELPERS["_pair_base_tdm"].body[-1].value
        sched = _schedule_fn()(chunks)
        checked = 0
        for ev in sched:
            if ev[0] == "issue":
                _, k, buf = ev
                order = list(range(4))
                rng.shuffle(order)
                for wave in order:      # producer order is irrelevant (disjoint rows)
                    run_issue(lds, self.source, flat, width, req, span0 + k * 64,
                              self.num_pages, buf, wave)
            elif ev[0] == "consume":
                _, k, buf = ev
                col0 = span0 + k * 64
                kbuf = _eval(kbuf_expr, dict(CONSTS, kbase=0, k=k))
                ref, _ = fallback_fragments(self.source, table[req], col0, columns, columns)
                for nt in range(4):
                    for lane in range(32):
                        c = col0 + nt * 16 + lane % 16
                        kgrp = lane // 16
                        base = _eval(base_expr, dict(CONSTS, c=c, kgrp=kgrp, tdm_cols=64))
                        for s in range(4):
                            lo = kbuf * 4 + base + s * 64        # _pair_frag_tdm: ptr + byte_off // 4
                            hi = lo + 32
                            self.assertEqual(lo, buf * TILE + padded_fragment_offsets(nt, s, lane)[0])
                            self.assertTrue(0 <= lo and hi + 16 <= LDS)
                            got = bytes(lds.mem[lo:lo + 16] + lds.mem[hi:hi + 16])
                            for _wave in range(4):   # B is shared: identical for all waves
                                self.assertEqual(got, ref[nt, s, lane], (k, nt, s, lane))
                            checked += 1
        # Padding DWORDs are never written.
        for r in range(2 * 64):
            pad = lds.mem[r * ROW + 256:r * ROW + 272]
            self.assertEqual(bytes(pad), PADDING_SENTINEL)
        self.assertEqual(sorted(set(lds.starts)), [b * TILE + g * 8 * ROW for b in (0, 1) for g in range(8)])
        return checked

    def test_byte_exact_fragments_every_chunk_wave_lane_kchunk(self):
        rng = random.Random(5)
        total = 0
        for chunks in (2, 4, 8):
            width = chunks * 16 + 3           # > one span: tails exercised by span1
            table = []
            for _ in range(3):
                row = list(range(self.num_pages))
                rng.shuffle(row)
                row = (row * 4)[:width]
                for j in rng.sample(range(width), 5):
                    row[j] = rng.choice((-1, -7, self.num_pages, self.num_pages + 9))
                table.append(row)
            for req in (0, 2):
                for span in (0, 1):
                    span0 = span * chunks * 64
                    total += self._check_block(chunks, table, req, span0, width * 4, rng)
        self.assertEqual(total, sum(2 * 2 * c * 4 * 32 * 4 for c in (2, 4, 8)))

    def test_lds_allocator_bounds_and_alignment(self):
        storage = HELPERS["_persist_storage_tdm"]
        self.assertEqual(len(_calls(storage, "fx.SharedAllocator")), 1)
        arr = next(n for n in ast.walk(storage) if isinstance(n, ast.Subscript)
                   and ast.unparse(n.value) == "fx.Array")
        etype, count, align = arr.slice.elts
        self.assertEqual(ast.unparse(etype), "fx.Int32")
        self.assertEqual(_eval(count, CONSTS) * 4, 34816)
        self.assertEqual(_eval(align, CONSTS), 128)
        consts = {t.id: n.value for n in TREE.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name)}
        self.assertEqual(_eval(consts["TDM_PERSIST_LDS_BYTES"], CONSTS), 34816)
        self.assertEqual(len(_calls(KERNEL, "_persist_storage_tdm")), 1)
        self.assertFalse(_calls(KERNEL, "pair_storage") or _calls(KERNEL, "_pair_storage_tdm"))
        for k in range(8):
            self.assertEqual(_eval(_kernel_assign("kbuf", _reuse_branch()), dict(CONSTS, kbase=0, k=k)) * 4,
                             (k % 2) * TILE)


class PersistGuardsCPU(unittest.TestCase):
    def test_issue_helpers_are_sgpr_only(self):
        for name in ("_persist_issue_tdm", "_persist_block_vis"):
            helper = HELPERS[name]
            self.assertFalse(_calls(helper, "_ld_i32"), name)
            self.assertFalse(any(isinstance(n, ast.Name) and n.id in ("lane", "kgrp", "pp", "plans", "n_in", "tid")
                                 for n in ast.walk(helper)), name)
        issue = HELPERS["_persist_issue_tdm"]
        self.assertEqual(ast.unparse(_kernel_assign("wave_id", issue)), "fx.Int32(fx.rocdl.wave_id())")
        self.assertFalse(_calls(issue, "fx.rocdl.tdm_ops.tensor_wait") or _calls(issue, "fx.gpu.barrier"))
        reuse = _reuse_branch()
        self.assertEqual(ast.unparse(_kernel_assign("preq", reuse)), "_ld_uniform_i32(t2r_rsrc, rblk * 32 * 4)")
        self.assertEqual(ast.unparse(_kernel_assign("issue", reuse)),
                         "[(span0 + k0 * WMMA_COLS < bvis) & (span0 + k0 * WMMA_COLS < columns) for k0 in range_constexpr(persist_chunks)]")
        self.assertEqual(ast.unparse(_kernel_assign("reuse")), "_r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)")

    def test_block_visibility_bounds_every_row(self):
        rng = random.Random(11)
        for _ in range(300):
            ctx = rng.randint(-5, 5000)
            qp = [rng.randint(-10, 6000) for _ in range(64)]
            cl = [rng.randint(0, 9000) for _ in range(3)]
            cl[1] = ctx
            rblk = rng.randint(0, 1)

            def ld(rsrc, off):
                off = off.v if isinstance(off, V) else off
                plane = qp if rsrc == "qp" else cl
                assert off % 4 == 0 and 0 <= off < 4 * len(plane)
                return V(plane[off // 4])

            ns = dict(CONSTS, range_constexpr=range, _ld_uniform_i32=ld,
                      _imin=lambda a, b: a if V._o(a) <= V._o(b) else b,
                      _imax=lambda a, b: a if V._o(a) >= V._o(b) else b,
                      fx=SimpleNamespace(Int32=V))
            bvis = _exec([HELPERS["_persist_block_vis"]], ns)["_persist_block_vis"]("cl", V(1)).v
            self.assertEqual(bvis, max(ctx, 0) // 4)
            # `_rows_meta` vis of every row of the block is <= the bound, so a
            # skipped chunk (col0 >= bvis) is never read by a stored lane.
            rows_vis = [max(min(p + 1, ctx), 0) // 4 for p in qp[rblk * 32:rblk * 32 + 32]]
            self.assertGreaterEqual(bvis, max(rows_vis))
            for col0 in range(0, 6000, 64):
                if col0 >= bvis:
                    self.assertTrue(all(c >= v for v in rows_vis for c in range(col0, col0 + 64)))

    def test_fallback_is_raw_and_barrier_free(self):
        fb = _fallback_branch()
        for name in ("_persist_issue_tdm", "fx.rocdl.tdm_ops.tensor_wait", "fx.gpu.barrier", "_pair_values"):
            self.assertFalse(_calls(fb, name), name)
        values = _calls(fb, "_pass_values")
        self.assertEqual(len(values), 1)
        self.assertEqual(ast.unparse(values[0].args[1]), "kv_rsrc")
        self.assertEqual(len(values[0].args), 10)        # raw layout (staged=False)
        self.assertEqual(ast.unparse(_kernel_assign("fcol0")), "span0 + kf * WMMA_COLS")
        self.assertEqual(ast.unparse(_kernel_assign("col0")), "span0 + k * WMMA_COLS")
        self.assertEqual(ast.unparse(_kernel_assign("span0")), "chunk * (WMMA_COLS * persist_chunks)")
        writer = _kernel_assign("writer")
        self.assertEqual(ast.unparse(writer), "(chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)")


class PersistCoverageCPU(unittest.TestCase):
    def test_every_logit_written_exactly_once_over_the_real_grid(self):
        ns = R._namespace({"QSA_BF16_PERSIST": "1"})
        lane_cols = _exec([HELPERS["_lane_cols"]], dict(CONSTS, range_constexpr=range))["_lane_cols"]
        offsets = _exec([HELPERS["_store_offsets"]], dict(CONSTS, _imin=min))["_store_offsets"]
        span_expr = _kernel_assign("span0")
        for chunks in (2, 4, 8):
            for rows, columns in ((64, 64 * chunks), (96, 64 * chunks + 1), (100, 1000),
                                  (33, 64 * chunks * 3 - 5)):
                gx, gy = ns["wmma_grid"](rows, columns, 32, 64 * chunks)
                self.assertEqual((gx, gy), (-(-columns // (64 * chunks)), -(-rows // 32)))
                count = {}
                for by in range(gy):
                    full = by * 32 + 32 <= rows                 # single request here
                    for bx in range(gx):
                        span0 = _eval(span_expr, dict(CONSTS, chunk=bx, persist_chunks=chunks))
                        for wave in range(4):
                            t0 = (by * 4 + wave) * 8
                            for k in range(chunks):
                                col0 = span0 + k * 64
                                for lane in range(32):
                                    n_in, kgrp = lane % 16, lane // 16
                                    cols = lane_cols(col0, n_in)
                                    for mi in range(2):
                                        sub = t0 + mi * 4
                                        if not full and sub >= rows:
                                            continue
                                        offs = offsets(sub, kgrp, cols, rows, columns)
                                        for nt, c in enumerate(cols):
                                            for half in range(2):
                                                row = sub + 2 * kgrp + half
                                                if row < rows and c < columns:     # own & col_ok
                                                    o = offs[2 * nt + half]
                                                    self.assertTrue(0 <= o < rows * columns * 4)
                                                    self.assertEqual(o, (row * columns + c) * 4)
                                                    count[o] = count.get(o, 0) + 1
                self.assertEqual(len(count), rows * columns, (chunks, rows, columns))
                self.assertEqual(set(count.values()), {1})


class PersistRoutingCPU(unittest.TestCase):
    def test_default_is_persist8_and_off_is_single_tile(self):
        ns = R._namespace()
        self.assertIs(ns["TDM_PERSIST"], True)
        self.assertEqual(ns["TDM_PERSIST_CHUNKS"], 8)
        launch = ns["_build_wmma_kernel"](1.0, 32, "persist")
        self.assertEqual(R._selected(launch)._func.__name__, "kernel_r32_persist")
        self.assertTrue(launch.kernel_name.endswith("_tdm_persist8"))
        self.assertEqual((launch.cols_per_block, launch.persist_chunks), (512, 8))
        off = R._namespace({"QSA_BF16_PERSIST": "0"})
        self.assertIs(off["TDM_PERSIST"], False)
        self.assertEqual(off["TDM_PERSIST_CHUNKS"], 8)
        single = off["_build_wmma_kernel"](1.0, 32, None)
        self.assertEqual(R._selected(single)._func.__name__, "kernel_r32")
        self.assertEqual((single.cols_per_block, single.persist_chunks), (64, 1))
        for rows, requests in ((512, 1), (2880, 2)):
            self.assertEqual(R._wrapper_route(ns, rows, requests, 0x1000, columns=65536),
                             (32, "persist", 128, -(-rows // 32)))
            self.assertEqual(R._wrapper_route(off, rows, requests, 0x1000, columns=65536),
                             (32, "tdm", 1024, -(-rows // 32)))

    def test_env_validation(self):
        validators = [n for n in TREE.body if isinstance(n, ast.If)
                      and any(isinstance(x, ast.Constant) and x.value in ("QSA_BF16_PERSIST", "QSA_PERSIST_COLS")
                              for x in ast.walk(n.test))]
        self.assertEqual(len(validators), 2)
        for env in ({"QSA_BF16_PERSIST": "2"}, {"QSA_BF16_PERSIST": "yes"}, {"QSA_PERSIST_COLS": "3"},
                    {"QSA_PERSIST_COLS": "16"}, {"QSA_PERSIST_COLS": ""}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                _exec(validators, {"os": SimpleNamespace(environ=env)})
        for enabled in ("0", "1", " 0 ", " 1 "):
            for chunks in ("2", "4", "8", " 8 "):
                ns = R._namespace({"QSA_BF16_PERSIST": enabled, "QSA_PERSIST_COLS": chunks})
                self.assertIs(ns["TDM_PERSIST"], enabled.strip() == "1")
                self.assertEqual(ns["TDM_PERSIST_CHUNKS"], int(chunks))

    def test_default_and_off_keep_exact_fallback_routes(self):
        cases = ((512, 1, 511, 0x1000, (32, "tdm", 8, 16)),
                 (512, 1, 512, 0x1010, (16, "raw", 8, 32)),
                 (64, 16, 65536, 0x1000, (16, "tdm", 2048, 4)),
                 (31, 1, 65536, 0x1000, (16, "tdm", 2048, 2)),
                 (63, 2, 65536, 0x1000, (16, "tdm", 2048, 4)))
        for env in ({}, {"QSA_BF16_PERSIST": "0"}):
            ns = R._namespace(env)
            for rows, requests, columns, pointer, expected in cases:
                with self.subTest(env=env, rows=rows, columns=columns, pointer=pointer):
                    self.assertEqual(R._wrapper_route(ns, rows, requests, pointer, columns), expected)
        for env in ({"QSA_BF16_K_LOAD": "raw"}, {"QSA_G2_ACCESS": "cop"},
                    {"QSA_BF16_R64": "1"}):
            ns = R._namespace(env)
            expected = ((64, "tdm", 1024, 8) if "QSA_BF16_R64" in env else
                        (32, "raw" if "QSA_BF16_K_LOAD" in env else "tdm", 1024, 16))
            self.assertEqual(R._wrapper_route(ns, 512, 1, 0x1000, 65536), expected)
            if "QSA_G2_ACCESS" in env:
                # The existing wrapper label stays tdm; the actual cop builder
                # resolves raw internally. Neither layer selects persistent.
                R.Bf16RoutingCPU().assert_selection(ns, 32, "tdm", False)

    def test_supported_spans_route_only_guarded_r32_tdm_prefill(self):
        for chunks in ("2", "4", "8"):
            n = int(chunks)
            for mode in ("auto", "tdm", "raw"):
                for access in ("raw", "cop"):
                    ns = R._namespace({"QSA_BF16_PERSIST": "1", "QSA_PERSIST_COLS": chunks,
                                       "QSA_BF16_K_LOAD": mode, "QSA_G2_ACCESS": access})
                    for rows, requests, columns, pointer in (
                            (512, 1, 65536, 0x1000), (2880, 2, 65536, 0x1000),
                            (512, 1, 64 * n, 0x1000), (512, 1, 64 * n - 1, 0x1000),
                            (512, 1, 65536, 0x1010), (64, 16, 65536, 0x1000), (31, 1, 65536, 0x1000)):
                        got = R._wrapper_route(ns, rows, requests, pointer, columns=columns)
                        r32 = rows >= 32 and rows >= 32 * requests
                        tdm = mode != "raw"
                        aligned = pointer % 128 == 0
                        persist = (r32 and tdm and aligned and access == "raw" and columns >= 64 * n)
                        with self.subTest(chunks=n, mode=mode, access=access, rows=rows,
                                          columns=columns, pointer=pointer):
                            if persist:
                                self.assertEqual(got, (32, "persist", -(-columns // (64 * n)), -(-rows // 32)))
                            else:
                                self.assertNotEqual(got[1], "persist")
                            if persist:
                                launch = ns["_build_wmma_kernel"](1.0, 32, "persist")
                                self.assertEqual(R._selected(launch)._func.__name__, "kernel_r32_persist")
                                self.assertEqual(launch.cols_per_block, 64 * n)
                                self.assertEqual(launch.kernel_name,
                                                 "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r32_c64_flydsl_tdm_persist%d" % n)
                                self.assertEqual(R._selected(launch).name, launch.kernel_name)

    def test_builder_rejects_persist_outside_r32_tdm(self):
        ns = R._namespace({"QSA_BF16_PERSIST": "1"})
        for rows in (16, 64):
            with self.assertRaises(ValueError):
                ns["_build_wmma_kernel"](1.0, rows, "persist")
        with self.assertRaises(ValueError):
            ns["_build_wmma_kernel"](1.0, 32, "persist", 3)
        cop = R._namespace({"QSA_BF16_PERSIST": "1", "QSA_G2_ACCESS": "cop"})
        with self.assertRaises(ValueError):
            cop["_build_wmma_kernel"](1.0, 32, "persist")
        explicit = ns["_build_wmma_kernel"](1.0, 32, "persist", 8)
        self.assertEqual((explicit.cols_per_block, explicit.persist_chunks), (512, 8))
        for args in ((32, 65, 32, 1024), (16, 65, 16, 256), (64, 65, 64, 256)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ns["wmma_grid"](*args)


if __name__ == "__main__":
    unittest.main()
