# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Framework tuner for the opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a).

Wired into the canonical :class:`GemmCommonTuner`, so it runs like the other
aiter GEMM tuners: multi-GPU via ``mp_tuner``, standard ``-i/--untune_file`` /
``-o/--tune_file`` CLI, batching, and the shared post-process / CSV writer.

The candidate pool lives here (``_TUNE_POLICY``). Per kid it holds only the
split-K factors to sweep; tile geometry, kernelName and the M alignment come from
the codegen instance table, so a kid cannot be tuned on a shape its launcher
rejects. That alignment used to be a second hand-maintained column and was wrong
in both directions -- it hid kid326, which is really arbitrary-M, from every
unaligned shape while the runtime dispatched it there anyway.

Runtime schema (what the tuner emits, and what the runtime reads back):
    gfx,b,m,n,k,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
``aiter/ops/batched_gemm_op_a8w8.py:lookup_mxscale_bmm_config`` indexes on
``["gfx","b","m","n","k"]``, dispatches to a backend on the winning row's
``libtype``, and the existing A8W8 caller passes ``kernelId`` / ``splitK`` to
the batch-first ``opus_bmm`` entry, so
those columns must match exactly.

Verification (the part that catches column-transpose / scale defects):
  * inputs are *signed* and have *per-128-K-block varied magnitude*
    (``randn * 2**randint(-4,4)`` per block) so the e8m0 128-block scales span
    many exponents. Uniform non-negative ``rand()/10`` data hides a pure output
    column permutation (kid312/313 measured ~0.007 there but ~0.7-1.0 on real
    signed data) -- see the opus_bmm.md root-cause note.
  * reference is a dequantized fp32 einsum.
  * gate: ``mp_tuner`` runs ``checkAllclose(rtol=1e-2, atol=1e-2)`` and
    ``post_process`` keeps the fastest candidate whose mismatch fraction is
    ``<= --errRatio`` (default 0.02). A still-broken tileN COM_REP_N>1 kernel
    measures ~0.5 here and is rejected; the fp8 e8m0 quant floor is ~1e-4.

Usage (gfx950 only; the repo root must be on PYTHONPATH so the edited/rebuilt
tree wins over any installed aiter):
    cd <repo> && PYTHONPATH=$PWD \\
        python3 csrc/opus_gemm/opus_bmm_mxscale_tune.py -g 16 -m 1,16,64 -n 1024 -k 4096

    # re-tune every shape already in the shipped CSV, write a diffable copy:
    ... opus_bmm_mxscale_tune.py

    # overwrite the shipped tuned CSV in place:
    ... opus_bmm_mxscale_tune.py --apply

    # from an untuned CSV (columns: b,m,n,k -- or g,m,n,k), 8-way parallel:
    ... opus_bmm_mxscale_tune.py -i my_untuned.csv -o /tmp/out.csv --mp 8
"""

import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus import opus_bmm
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Neither op_tests nor this directory is a package, so put both on sys.path. This
# also has to hold in the spawned mp_tuner subprocesses, which re-import this
# module top-to-bottom.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OPTESTS = os.path.join(_REPO, "op_tests")
for _p in (_HERE, _OPTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# opus_gemm_common is pure python (stdlib only), so importing the codegen kid
# table here does not pull in the build.
from opus_gemm_common import a8w8_mxscale_bmm_kernel_lists
from test_opus_a8w8_bmm import (
    GROUP,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

# kid -> OpusGemmInstance. Kids are disjoint across the BMM families today;
# assert so a future collision (which the codegen dedups by launcher name
# downstream) is caught here instead of silently tuning one of the two.
_CODEGEN_BMM = {}
for _fam in a8w8_mxscale_bmm_kernel_lists:
    for _kid, _inst in _fam.items():
        assert (
            _kid not in _CODEGEN_BMM
        ), f"bmm kid {_kid} collides across codegen families; disambiguate by name"
        _CODEGEN_BMM[_kid] = _inst

# Split-K sweep for the flatmm_splitk family. Small-M / few-tile shapes (the G16
# wo_a decode: 16 batch * n1024 * k4096) underfill the CUs at splitK=1, so split-K
# (fp32-workspace partials + fused reduce tail) can win by exposing parallelism
# along K. The correctness gate drops any combo a kernel mishandles, so an
# over-broad sweep is safe, just slower.
_SK = [1, 2, 4, 8]

# Tuning policy: kid -> splitK list. The ONLY hand-maintained per-kid metadata --
# it decides which kids to sweep and with which split-K factors, not their
# geometry and not their M alignment. Tile shape, kernelName and m_align all come
# from the codegen instance, so this cannot drift from what compiles. kid 8000 (the
# heuristic default) is intentionally not tuned.
_TUNE_POLICY = {
    # flatmm_splitk family: the M=16/32 last-mile tiles, the mid-M SFA/SFB-preload
    # tiles and the 64x* tiles. All are split-K capable via the fused reduce tail,
    # except kid646 whose persistent DIRECT_ONLY schedule requires splitK == 1.
    8032: _SK,
    8064: _SK,
    8138: _SK,
    8139: _SK,
    8256: _SK,
    8311: _SK,
    8312: _SK,
    8313: _SK,
    8314: _SK,
    8316: _SK,
    8317: _SK,
    8318: _SK,
    8319: _SK,
    8320: _SK,
    8321: _SK,
    8322: _SK,
    8323: _SK,
    8324: _SK,
    8326: _SK,
    8327: _SK,
    8640: _SK,
    8642: _SK,
    8646: [1],
    8650: _SK,
    8653: _SK,
    # fused single-tile launcher.
    8100: [1],
    # pipeline family; kid158 preloads both the per-token SFA and the block SFB
    # panel into LDS.
    8149: [1],
    8150: [1],
    8151: [1],
    8152: [1],
    8158: [1],
    # monolithic mouter / wave pipelines.
    8131: [1],
    8132: [1],
    8134: [1],
    8142: [1],
    8144: [1],
    8148: [1],
    8160: [1],
    8161: [1],
    # minterleave only exists in split-K form.
    8162: [1],
    8163: [1],
    # 128x128x128 tiles, splitK=1 only and deliberately so. They are the largest
    # BMM tile (COM_REP_M=4 x COM_REP_N=8 -> 32 C fragments, 128 fp32 C values per
    # lane) at 512 VGPRs / occupancy 1. At splitK=1 they run the Cbf16
    # direct-output kernel and are strong -- kid325 wins 5 shipped wo_a rows. At
    # splitK>1 they switch to the Cvoid fp32-workspace kernel, which spills and has
    # never won (g2/m256: best kid325 split-K is 23.6us against the 14.5us winner),
    # and which is also where the clang-22 gfx950 greedy-VGPR miscompile lives (one
    # C-fragment dword left unmaterialized under --amdgpu-mfma-vgpr-form).
    8128: [1],
    8137: [1],
    8325: [1],
}

# Only non-direct flatmm split-K launchers are swept with splitK>1.
# Any other family sweeping it is a policy bug, so fail loudly at import.
for _kid, _sks in _TUNE_POLICY.items():
    if any(s > 1 for s in _sks):
        _tag = _CODEGEN_BMM[_kid].kernel_tag
        assert (
            _tag == "a8w8_mxscale_bmm_flatmm_splitk"
            and not getattr(_CODEGEN_BMM[_kid], "direct_only", False)
        ), (
            f"kid {_kid} ({_tag}) is not split-K capable but sweeps {_sks}"
        )


def _applicable(kid, g, m, n, k):
    """Split-K factors worth trying for this kid on this shape ([] == skip it)."""
    k_inst = _CODEGEN_BMM[kid]
    if n % k_inst.B_N or k % k_inst.B_K or m % k_inst.m_align:
        return []
    return _TUNE_POLICY[kid]


SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_retuned.csv")


# ---------------------------------------------------------------------------
# mp_tuner hooks (module-level so the spawn workers can import them by name).
# ---------------------------------------------------------------------------
def _gen_varied(shape, k, device):
    """Signed, per-128-K-block varied-magnitude bf16 (mirrors _block_varied)."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=device).float())
    return (x * amp.repeat_interleave(GROUP)).to(dtypes.bf16)


def _workspace_numel(kernel_id, split_k, batch, m, n):
    instance = _CODEGEN_BMM[int(kernel_id)]
    if split_k <= 1 or instance.kernel_tag not in {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    }:
        return 0
    tiles_m = (m + instance.B_M - 1) // instance.B_M
    tiles_n = (n + instance.B_N - 1) // instance.B_N
    partial_numel = (
        split_k
        * batch
        * tiles_m
        * instance.B_M
        * tiles_n
        * instance.B_N
    )
    if instance.kernel_tag != "a8w8_mxscale_bmm_fused":
        return partial_numel
    counter_offset = (partial_numel * 4 + 255) & ~255
    counter_bytes = batch * tiles_m * tiles_n * 4
    return (counter_offset + counter_bytes + 3) // 4


def gen_bmm_mxscale_data(
    batch, m, n, k, seed, out_dtype, kernel_id, split_k, device="cuda"
):
    """Return the 7-tuple mp_tuner indexes into:

    0 O_mx   [g,m,k]     fp8 (batch-first, K contiguous)
    1 W_mx   [g,n,k]     fp8 (batch-major)
    2 Y       [g,m,n]     out_dtype output buffer
    3 xs_mx  [g,m,k/128] uint8 e8m0 per-token scale
    4 ws_mx  [g,n/128,k/128] uint8 e8m0 128x128-block scale
    5 workspace optional caller-owned FP32 split-K buffer
    6 ref     [g,m,n]     out_dtype dequant fp32 einsum reference
    """
    torch.manual_seed(seed)
    O_bf16 = _gen_varied((batch, m, k), k, device)
    W_bf16 = _gen_varied((batch, n, k), k, device)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)
    Y = torch.empty((batch, m, n), dtype=out_dtype, device=device)
    workspace_numel = _workspace_numel(kernel_id, split_k, batch, m, n)
    workspace = (
        torch.empty(workspace_numel, dtype=torch.float32, device=device)
        if workspace_numel
        else None
    )
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).to(out_dtype)
    return (O_mx, W_mx, Y, xs_mx, ws_mx, workspace, ref)


def run_bmm_mxscale_bench(
    O_mx, W_mx, Y, xs_mx, ws_mx, workspace, kernelId, splitK
):
    """Tuner bench func: run the kid in-place, return Y for checkAllclose."""
    return opus_bmm(
        O_mx,
        W_mx,
        Y,
        kid=int(kernelId),
        layout="mxscale_bmm",
        x_scale=xs_mx,
        w_scale=ws_mx,
        split_k=int(splitK),
        workspace=workspace,
    )


def _bmm_ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 6)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
class OpusBmmMxscaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": DEFAULT_OUT,
        "untune_file": "",
        # Fraction-of-mismatch (rtol=atol=1e-2) accept threshold. Correct kids
        # sit at the ~1e-4 fp8 e8m0 quant floor; a column-transposed kid is ~0.5.
        "errRatio": 0.02,
        "batch": 100,
    }

    KEYS: ClassVar[list[str]] = ["gfx", "b", "m", "n", "k"]
    RESULTS: ClassVar[list[str]] = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]

    def __init__(self):
        # Bypass GemmCommonTuner.__init__ (it force-swaps "M"/"N" in the key,
        # which assumes the uppercase gptoss schema). Go straight to the
        # grandparent with our lowercase batched schema.
        TunerCommon.__init__(
            self,
            "OpusBmmMxscaleTuner",
            self.KEYS,
            self.RESULTS,
            description="Tune opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a)",
        )
        # sort N before M like the GEMM tuners (cosmetic ordering of the CSV).
        self.sort_keys = ["gfx", "b", "n", "m", "k"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId):
        k_inst = _CODEGEN_BMM.get(int(kernelId))
        return k_inst.name if k_inst else None

    def calculate(self, results, bpes=None):
        info, time, _err = results
        if time == self.INVALID_TIME:
            return 0, 0
        _gfx, b, m, n, k = info[0]
        us_s = time * 1e-6
        tflops = round(2 * b * m * n * k / us_s / 1e12, 1)
        # fp8 A + fp8 W + bf16 out.
        bw = round((b * m * k + b * n * k + 2 * b * m * n) / us_s / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for info, time, err in results:
            keys, kernelId, splitK, kernelName = info
            resolved = kernelName or self.getKernelName(kernelId)
            tflops, bw = self.calculate((info, time, err))
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": "opus",
                    "kernelId": int(kernelId),
                    "splitK": int(splitK),
                    "us": time,
                    "kernelName": "None" if resolved is None else str(resolved),
                    "tflops": tflops,
                    "bw": bw,
                    "errRatio": err,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    # --- CLI ----------------------------------------------------------------
    def _setup_specific_arguments(self):
        # Free the base "-k/--splitK" store_true so we can reuse -k for the K dim.
        for action in list(self.parser._actions):
            if "-k" in action.option_strings or "--splitK" in action.option_strings:
                self.parser._actions.remove(action)
                for s in action.option_strings:
                    self.parser._option_string_actions.pop(s, None)
                for grp in self.parser._action_groups:
                    if action in grp._group_actions:
                        grp._group_actions.remove(action)
                break

        def _intlist(s):
            return [int(x) for x in str(s).split(",") if x != ""]

        self.parser.add_argument(
            "-g",
            "--batch_g",
            type=_intlist,
            default=None,
            help="comma list of batch g (e.g. 2,8,16)",
        )
        self.parser.add_argument(
            "-m",
            "--M",
            type=_intlist,
            default=None,
            help="comma list of M (e.g. 1,16,64)",
        )
        self.parser.add_argument(
            "-n",
            "--N",
            type=_intlist,
            default=[1024],
            help="comma list of N (default 1024)",
        )
        self.parser.add_argument(
            "-k",
            "--K",
            type=_intlist,
            default=[4096],
            help="comma list of K (default 4096)",
        )
        self.parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="overwrite the shipped tuned CSV in place",
        )

    # --- shape sourcing -----------------------------------------------------
    def _shapes_from_shipped(self):
        try:
            df = pd.read_csv(SHIPPED_CSV)
        except FileNotFoundError:
            return []
        return sorted(
            {(int(r.b), int(r.m), int(r.n), int(r.k)) for _, r in df.iterrows()}
        )

    def pre_process(self, args):
        if args.apply:
            args.tune_file = SHIPPED_CSV

        gfx = self.get_gfx()
        if args.batch_g and args.M:
            shapes = [
                (g, m, n, k)
                for g in args.batch_g
                for m in args.M
                for n in args.N
                for k in args.K
            ]
        elif args.untune_file and os.path.exists(args.untune_file):
            df = pd.read_csv(args.untune_file)
            df.columns = [c.strip().lower() for c in df.columns]
            bcol = "b" if "b" in df.columns else "g"
            shapes = [
                (int(r[bcol]), int(r["m"]), int(r["n"]), int(r["k"]))
                for _, r in df.iterrows()
            ]
        else:
            logger.info(
                "no -g/-m and no untune_file; re-tuning shapes from %s", SHIPPED_CSV
            )
            shapes = self._shapes_from_shipped()

        self.untunedf = pd.DataFrame(
            [{"gfx": gfx, "b": g, "m": m, "n": n, "k": k} for (g, m, n, k) in shapes],
            columns=self.keys,
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)

        # Skip shapes already present in the tuned CSV (unless --all forces retune).
        if not args.all and len(self.tunedf) and len(self.untunedf):
            td = self.tunedf
            if "gfx" not in td.columns:
                td = td.assign(gfx=gfx)
            have = set(td[self.keys].apply(lambda r: tuple(r), axis=1).tolist())
            mask = self.untunedf.apply(lambda r: tuple(r) in have, axis=1)
            if args.verbose and mask.any():
                logger.info("skipping %d already-tuned shapes", int(mask.sum()))
            self.untunedf = self.untunedf[~mask].reset_index(drop=True)

    # --- tuning -------------------------------------------------------------
    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        out_dtype = dtypes.bf16
        perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}

        task = []
        tasks_data = []
        for seed, i in enumerate(range(len(untunedf)), start=1):
            b = int(untunedf.loc[i, "b"])
            m = int(untunedf.loc[i, "m"])
            n = int(untunedf.loc[i, "n"])
            k = int(untunedf.loc[i, "k"])
            info_keys = (gfx, b, m, n, k)

            n_cand = 0
            for kid in _TUNE_POLICY:
                for sk in _applicable(kid, b, m, n, k):
                    info = (info_keys, kid, sk, "")
                    task.append(
                        (
                            info,
                            gen_bmm_mxscale_data,
                            (b, m, n, k, seed, out_dtype, kid, sk),
                            run_bmm_mxscale_bench,
                            ([0, 1, 2, 3, 4, 5], kid, sk),
                            perf_kwargs,
                            _bmm_ref_passthrough,
                            ([6],),
                            {},
                            None,
                            1e-2,  # rtol
                            1e-2,  # atol
                            None,  # compare_fn
                            None,  # max_abs_delta
                            [2],  # output_keys: NaN-init Y to catch partial writes
                        )
                    )
                    n_cand += 1
            tasks_data.append((n_cand, ()))

        if not task:
            return []
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = OpusBmmMxscaleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
