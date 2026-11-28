"""Exercise the review helper against real Git trees, without importing aiter."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("tuning_coverage.py")

CFG = "aiter/configs"
MODEL = "aiter/configs/model_configs"
GEMM_IN = "M,N,K,bias\n"
GEMM_OUT = "gfx,cu_num,M,N,K,bias,kernelName,us\n"


class TuningCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.write("README", "baseline\n")
        self.commit("initial")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def remove(self, name):
        (self.repo / name).unlink()

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def scan(self, base, head, success=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.repo), base, head],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result.stdout

    def gemm_pair(self, inputs, tuned, prefix=f"{CFG}/bf16"):
        self.write(f"{prefix}_untuned_gemm.csv", GEMM_IN + "".join(inputs))
        self.write(f"{prefix}_tuned_gemm.csv", GEMM_OUT + "".join(tuned))

    # -- the four mismatch kinds -------------------------------------------------

    def test_added_tuned_row_needs_untuned_row(self):
        self.gemm_pair(
            ["1,2,3,False\n"],
            ["gfx950,256,1,2,3,False,k,4\n", "gfx950,256,9,9,9,False,k,4\n"],
        )
        base = self.commit("legacy: (9,9,9) has no input, not this PR's problem")
        self.gemm_pair(
            ["1,2,3,False\n", "4,5,6,False\n"],
            [
                "gfx950,256,1,2,3,False,k2,2\n",  # retune: result columns only
                "gfx950,256,9,9,9,False,k,4\n",
                "gfx942,304,1,2,3,FALSE,k,3\n",  # new GPU, shape already an input
                "gfx950,256,4,5,6.0,False,k,2\n",  # covered; 6.0 == 6
                "gfx950,256,7,8,9,False,k,2\n",  # missing
            ],
        )
        out = self.scan(base, self.commit("new rows"))
        self.assertIn("TUNED_NOT_IN_UNTUNED", out)
        self.assertIn("1 added tuned workloads", out)
        self.assertIn(f"{CFG}/bf16_tuned_gemm.csv:6:", out)
        self.assertIn('"M": "7"', out)
        self.assertNotIn('"M": "9"', out)
        self.assertNotIn("CHECKED", out)

    def test_added_untuned_row_needs_tuned_row(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.gemm_pair(["1,2,3,False\n", "4,5,6,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        out = self.scan(base, self.commit("input only"))
        self.assertIn("UNTUNED_NOT_IN_TUNED", out)
        self.assertIn(f"{CFG}/bf16_untuned_gemm.csv:3:", out)

    def test_removed_tuned_row_should_remove_untuned_row(self):
        self.gemm_pair(
            ["1,2,3,False\n", "4,5,6,False\n"],
            [
                "gfx950,256,1,2,3,False,k,4\n",
                "gfx942,80,4,5,6,False,k,4\n",
                "gfx942,304,4,5,6,False,k,4\n",
            ],
        )
        base = self.commit("pair")
        # The gfx942/80 row goes but gfx942/304 still tunes (4,5,6): its input stays.
        self.gemm_pair(
            ["1,2,3,False\n", "4,5,6,False\n"], ["gfx942,304,4,5,6,False,k,4\n"]
        )
        out = self.scan(base, self.commit("drop rows"))
        self.assertIn("TUNED_REMOVED_UNTUNED_KEPT", out)
        self.assertIn("1 removed tuned workloads", out)
        self.assertIn(f"{CFG}/bf16_tuned_gemm.csv:2:", out)
        self.assertNotIn('"M": "4"', out)

    def test_removed_untuned_row_should_remove_tuned_row(self):
        self.gemm_pair(
            ["1,2,3,False\n", "4,5,6,False\n"],
            ["gfx950,256,1,2,3,False,k,4\n", "gfx950,256,4,5,6,False,k,4\n"],
        )
        base = self.commit("pair")
        self.write(f"{CFG}/bf16_untuned_gemm.csv", GEMM_IN + "1,2,3,False\n")
        out = self.scan(base, self.commit("drop input"))
        self.assertIn("UNTUNED_REMOVED_TUNED_KEPT", out)
        self.assertIn(f"{CFG}/bf16_untuned_gemm.csv:3:", out)

    def test_matching_changes_on_both_sides_are_checked_clean(self):
        self.gemm_pair(
            ["1,2,3,False\n", "4,5,6,False\n"],
            ["gfx950,256,1,2,3,False,k,4\n", "gfx950,256,4,5,6,False,k,4\n"],
        )
        base = self.commit("pair")
        self.gemm_pair(
            ["1,2,3,False\n", "7,8,9,False\n"],
            ["gfx950,256,1,2,3,False,k,4\n", "gfx950,256,7,8,9,False,k,4\n"],
        )
        out = self.scan(base, self.commit("swap a workload"))
        self.assertIn("CHECKED", out)
        self.assertIn("tuned +1/-1 rows (+1/-1 workloads), untuned +1/-1 workloads", out)
        self.assertIn("[key: M, N, K, bias]", out)

    # -- hints, pairing and what is deliberately not checked ---------------------

    def test_move_to_another_changed_csv_is_hinted(self):
        model_in = f"{MODEL}/foo_bf16_untuned_gemm.csv"
        model_tuned = f"{MODEL}/foo_bf16_tuned_gemm.csv"
        canon = f"{CFG}/bf16_tuned_gemm.csv"
        self.write(model_in, GEMM_IN + "1,2,3,False\n")
        self.write(model_tuned, GEMM_OUT + "gfx950,256,1,2,3,False,k,4\n")
        self.write(canon, GEMM_OUT)
        base = self.commit("model rows")
        self.write(model_tuned, GEMM_OUT)
        self.write(canon, GEMM_OUT + "gfx950,256,1,2,3,False,k,4\n")
        out = self.scan(base, self.commit("move to canonical"))
        self.assertIn("TUNED_REMOVED_UNTUNED_KEPT", out)
        self.assertIn(f"[also at head in {canon}]", out)
        self.assertIn(f"UNPAIRED {canon}", out)

    def test_model_name_pairing_token_order_and_arch(self):
        self.write(f"{MODEL}/llama_untuned_gemm_bf16.csv", GEMM_IN + "1,2,3,False\n")
        self.write(f"{MODEL}/qwen_untuned_gemm.csv", GEMM_IN + "1,2,3,False\n")
        base = self.commit("inputs")
        self.write(
            f"{MODEL}/llama_bf16_tuned_gemm.csv",
            GEMM_OUT + "gfx950,256,1,2,3,False,k,4\n",
        )
        self.write(
            f"{MODEL}/qwen_gfx950_tuned_gemm.csv",
            GEMM_OUT + "gfx950,256,7,7,7,False,k,4\n",
        )
        self.write(f"{CFG}/bf16_untuned_gemm.csv", GEMM_IN + "7,7,7,False\n")
        out = self.scan(base, self.commit("model tables"))
        self.assertIn(f"CHECKED {MODEL}/llama_bf16_tuned_gemm.csv <-> "
                      f"{MODEL}/llama_untuned_gemm_bf16.csv", out)
        # Pairs with the arch-neutral model input, never the canonical one.
        self.assertIn(f"{MODEL}/qwen_gfx950_tuned_gemm.csv <-> "
                      f"{MODEL}/qwen_untuned_gemm.csv", out)
        self.assertIn("TUNED_NOT_IN_UNTUNED", out)

    def test_skips_and_informational_lines(self):
        shared_in = f"{MODEL}/shared_untuned_gemm.csv"
        self.write(shared_in, GEMM_IN + "1,2,3,False\n")
        self.write(f"{MODEL}/amb_untuned_gemm_bf16.csv", GEMM_IN)
        self.write(f"{MODEL}/amb_bf16_untuned_gemm.csv", GEMM_IN)
        self.write(f"{MODEL}/lonely_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2,3,False,k,4\n")
        self.write(f"{CFG}/paged_mqa_logits_untuned_glm5.csv", "context_len\n8192\n")
        base = self.commit("setup")
        self.write(shared_in, GEMM_IN + "1,2,3,False\n4,5,6,False\n")
        self.write(f"{MODEL}/amb_bf16_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,1,1,False,k,4\n")
        self.write(f"{MODEL}/lonely_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2,3,False,k2,3\n")
        self.write(f"{MODEL}/new_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2,3,False,k,4\n")
        self.write(f"{CFG}/paged_mqa_logits_untuned_glm5.csv", "context_len\n8192\n60000\n")
        self.write(f"{CFG}/notes.json", "{}\n")
        out = self.scan(base, self.commit("misc"))
        self.assertNotIn("shared_untuned", out)  # input with no name-paired tuned file
        self.assertIn(f"NOT_CHECKED {MODEL}/amb_bf16_tuned_gemm.csv: several untuned", out)
        self.assertNotIn("lonely", out)  # retune only
        self.assertIn(f"UNPAIRED {MODEL}/new_tuned_gemm.csv", out)
        self.assertIn("NOT_CHECKED aiter/configs/paged_mqa_logits", out)
        self.assertNotIn("notes.json", out)

    def test_extra_or_blank_input_columns_are_ignored_and_named(self):
        self.write(f"{CFG}/x_untuned_fmoe.csv", "cu_num,token,gate_mode\n,1,A\n")
        self.write(f"{CFG}/x_tuned_fmoe.csv", "gfx,cu_num,token,us\ngfx950,256,1,4\n")
        base = self.commit("pair")
        self.write(f"{CFG}/x_tuned_fmoe.csv",
                   "gfx,cu_num,token,us\ngfx950,256,1,4\ngfx950,256,2,4\n")
        out = self.scan(base, self.commit("new token"))
        self.assertIn("[key: token; ignored: cu_num, gate_mode]", out)
        self.assertIn("TUNED_NOT_IN_UNTUNED", out)
        self.assertIn('{"token": "2"}', out)

    def test_tuner_spellings_are_normalised(self):
        hdr_in = "token,q_dtype_a,q_type,use_g1u1\n"
        hdr_out = "gfx,cu_num,token,q_dtype_a,q_type,use_g1u1,us\n"
        self.write(f"{CFG}/x_untuned_fmoe.csv",
                   hdr_in + "1,torch.float8_e4m3fn,QuantType.per_128x128,TRUE\n")
        self.write(f"{CFG}/x_tuned_fmoe.csv", hdr_out)
        base = self.commit("input")
        self.write(f"{CFG}/x_tuned_fmoe.csv",
                   hdr_out + "gfx942,304,1,torch.float8_e4m3fnuz,QuantType.per_1x128,True,4\n")
        out = self.scan(base, self.commit("tuned on gfx942"))
        self.assertIn("CHECKED", out)
        self.assertNotIn("TUNED_NOT_IN_UNTUNED", out)

    def test_rename_is_a_removal_plus_an_addition(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.git("mv", f"{CFG}/bf16_tuned_gemm.csv", f"{CFG}/bf16_tuned_gemm_old.csv")
        out = self.scan(base, self.commit("rename tuned away from its input"))
        self.assertIn("TUNED_REMOVED_UNTUNED_KEPT", out)
        self.assertIn(f"[also at head in {CFG}/bf16_tuned_gemm_old.csv]", out)

    def test_legacy_rows_stay_quiet_through_pair_reshuffles(self):
        legacy = "gfx950,256,9,9,9,False,k,4\n"  # tuned, never had an input
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n", legacy])
        base = self.commit("pair")
        # A rename that keeps the pairing (arch-suffixed name -> arch-neutral input).
        self.git("mv", f"{CFG}/bf16_tuned_gemm.csv", f"{CFG}/bf16_gfx950_tuned_gemm.csv")
        out = self.scan(base, self.commit("rename within pair"))
        self.assertIn("CHECKED", out)
        self.assertNotIn('"M": "9"', out)

    def test_renamed_input_keeps_its_legacy_rows(self):
        # (5,5,5) is an untuned-only legacy row: never tuned, not this PR's problem.
        inputs = ["1,2,3,False" + chr(10), "5,5,5,False" + chr(10)]
        self.write(f"{CFG}/bf16_untuned_gemm.csv", GEMM_IN + "".join(inputs))
        self.write(f"{CFG}/bf16_gfx950_untuned_gemm.csv", GEMM_IN + "".join(inputs))
        self.write(f"{CFG}/bf16_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2,3,False,k,4" + chr(10))
        self.write(f"{CFG}/bf16_gfx950_tuned_gemm.csv",
                   GEMM_OUT + "gfx950,256,1,2,3,False,k,4" + chr(10))
        base = self.commit("pairs with untuned-only legacy rows")
        self.git("mv", f"{CFG}/bf16_untuned_gemm.csv", f"{CFG}/untuned_bf16_gemm.csv")
        self.git("rm", "-q", f"{CFG}/bf16_gfx950_tuned_gemm.csv")
        self.git("mv", f"{CFG}/bf16_gfx950_untuned_gemm.csv", f"{CFG}/gfx950_untuned_gemm.csv")
        self.write(f"{CFG}/gfx950_tuned_gemm.csv",
                   GEMM_OUT + "gfx950,256,1,2,3,False,k,4" + chr(10))
        out = self.scan(base, self.commit("renames that keep the pairing"))
        self.assertNotIn("UNTUNED_NOT_IN_TUNED", out)
        self.assertNotIn('"M": "5"', out)
        self.assertIn(f"<-> {CFG}/untuned_bf16_gemm.csv", out)

    def test_per_gpu_rows_for_existing_shapes_are_counted(self):
        self.gemm_pair(["1,2,3,False" + chr(10)], ["gfx950,256,1,2,3,False,k,4" + chr(10)])
        base = self.commit("gfx950 only")
        self.gemm_pair(["1,2,3,False" + chr(10)],
                       ["gfx950,256,1,2,3,False,k,4" + chr(10),
                        "gfx942,304,1,2,3,False,k,5" + chr(10)])
        out = self.scan(base, self.commit("add gfx942 row"))
        self.assertIn("tuned +1/-0 rows (+0/-0 workloads), untuned +0/-0 workloads", out)

    def test_new_hardware_column_does_not_make_legacy_rows_new(self):
        self.write(f"{CFG}/bf16_untuned_gemm.csv", GEMM_IN + "1,2,3,False\n")
        self.write(f"{CFG}/bf16_tuned_gemm.csv",
                   "cu_num,M,N,K,bias,us\n256,1,2,3,False,4\n256,9,9,9,False,4\n")
        base = self.commit("pair without gfx")
        self.write(f"{CFG}/bf16_tuned_gemm.csv",
                   "gfx,cu_num,M,N,K,bias,us\n"
                   "gfx950,256,1,2,3,False,4\ngfx950,256,9,9,9,False,4\n")
        out = self.scan(base, self.commit("add gfx column"))
        self.assertNotIn("TUNED_NOT_IN_UNTUNED", out)

    def test_deleting_the_untuned_file_is_a_removal(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.remove(f"{CFG}/bf16_untuned_gemm.csv")
        out = self.scan(base, self.commit("drop the input file"))
        self.assertIn("UNTUNED_REMOVED_TUNED_KEPT", out)

    def test_new_ambiguous_input_does_not_fake_removals(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.write(f"{CFG}/untuned_bf16_gemm.csv", GEMM_IN)
        self.write(f"{CFG}/bf16_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2,3,False,k2,3\n")
        out = self.scan(base, self.commit("retune and add a look-alike input"))
        self.assertIn("CHECKED", out)  # keeps the base pairing; no contradictory line
        self.assertNotIn("NOT_CHECKED", out)
        self.assertNotIn("TUNED_REMOVED_UNTUNED_KEPT", out)

    def test_tuned_file_switching_to_a_better_input_is_not_a_change(self):
        nl = chr(10)
        row_in, row_out = "1,2,3,False" + nl, "gfx950,256,1,2,3,False,k,4" + nl
        self.write(f"{MODEL}/qwen_untuned_gemm.csv", GEMM_IN + row_in)
        self.write(f"{MODEL}/qwen_gfx950_tuned_gemm.csv", GEMM_OUT + row_out)
        self.write(f"{CFG}/bf16_gfx950_untuned_gemm.csv", GEMM_IN + row_in)
        self.write(f"{CFG}/bf16_gfx950_tuned_gemm.csv", GEMM_OUT + row_out)
        base = self.commit("arch-fallback pair and exact pair")
        self.write(f"{MODEL}/qwen_gfx950_untuned_gemm.csv",
                   GEMM_IN + row_in + "7,7,7,False" + nl)
        self.write(f"{MODEL}/qwen_gfx950_tuned_gemm.csv",
                   GEMM_OUT + row_out + "gfx950,256,7,7,7,False,k,4" + nl)
        self.git("mv", f"{CFG}/bf16_gfx950_untuned_gemm.csv", f"{CFG}/bf16_untuned_gemm.csv")
        out = self.scan(base, self.commit("exact input added; input renamed"))
        for tag in ("TUNED_NOT_IN_UNTUNED", "UNTUNED_NOT_IN_TUNED",
                    "TUNED_REMOVED_UNTUNED_KEPT", "UNTUNED_REMOVED_TUNED_KEPT", "NOT_CHECKED"):
            self.assertNotIn(tag, out)
        self.assertIn(f"<-> {MODEL}/qwen_gfx950_untuned_gemm.csv", out)
        self.assertIn(f"<-> {CFG}/bf16_untuned_gemm.csv", out)

    # -- failures ---------------------------------------------------------------

    def test_unreadable_blob_is_skipped_loudly(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k2,3\n"])
        head = self.commit("retune")
        blob = self.git("rev-parse", f"{base}:{CFG}/bf16_tuned_gemm.csv")
        obj = self.repo / ".git" / "objects" / blob[:2] / blob[2:]
        obj.chmod(0o644)  # loose objects are read-only
        obj.unlink()
        out = self.scan(base, head, success=False)
        self.assertIn("SKIPPED:", out)

    def test_malformed_paired_csv_is_skipped_loudly(self):
        self.gemm_pair(["1,2,3,False\n"], ["gfx950,256,1,2,3,False,k,4\n"])
        base = self.commit("pair")
        self.write(f"{CFG}/bf16_tuned_gemm.csv", GEMM_OUT + "gfx950,256,1,2\n")
        out = self.scan(base, self.commit("broken"), success=False)
        self.assertIn("SKIPPED: tuning coverage could not parse", out)

    def test_disconnected_merge_base_emits_single_clear_skipped_line(self):
        base = self.git("rev-parse", "HEAD")
        self.git("checkout", "--orphan", "orphan-branch")
        self.write(f"{CFG}/example_tuned.csv", "M,us\n1,2\n")
        head = self.commit("orphan")
        out = self.scan(base, head, success=False)
        self.assertEqual(out.count("SKIPPED:"), 1)
        self.assertIn("review these files manually", out)

    def test_fetch_runs_scanner_and_degrades_on_failure(self):
        self.write(f"{CFG}/example_untuned.csv", "M\n1\n")
        self.write(f"{CFG}/example_tuned.csv", "M,us\n1,3\n")
        base = self.commit("pair")
        self.write(f"{CFG}/example_tuned.csv", "M,us\n1,3\n2,3\n")
        head = self.commit("new workload")
        fetch = SCRIPT.with_name("fetch.sh").read_text(encoding="utf-8")
        start = fetch.index("# Compare the PR's tuned/untuned row changes")
        end = fetch.index("# Day-old worktrees", start)
        evidence = self.repo / "evidence"
        evidence.mkdir()
        env = os.environ | {
            "SKILLS_ROOT": str(SCRIPT.parent.parent),
            "PROJECT_ROOT": str(self.repo),
            "WORK": str(evidence),
            "BASE_SHA": base,
            "HEAD_SHA": head,
        }
        snippet = "set -euo pipefail\n" + fetch[start:end]
        subprocess.run(["bash", "-c", snippet], env=env, check=True, capture_output=True)
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertIn("TUNED_NOT_IN_UNTUNED aiter/configs/example_tuned.csv", output)

        env["HEAD_SHA"] = "missing-commit"
        subprocess.run(["bash", "-c", snippet], env=env, check=True, capture_output=True)
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertEqual(output.count("SKIPPED:"), 1)


if __name__ == "__main__":
    unittest.main()
