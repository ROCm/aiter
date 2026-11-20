"""Exercise the review scanner against real Git trees, without importing aiter."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("tuning_coverage.py")


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

    def test_only_new_full_keys_are_checked_against_head_input(self):
        untuned = "aiter/configs/bf16_untuned_gemm.csv"
        tuned = "aiter/configs/bf16_tuned_gemm.csv"
        self.write(untuned, "M,N,dtype,bias\n1,2,bf16,False\n")
        self.write(
            tuned,
            "gfx,M,N,dtype,bias,us\n"
            "gfx1,1,2,bf16,False,4\n"
            "gfx1,9,9,bf16,False,4\n",
        )
        base = self.commit("existing workloads")
        self.write(untuned, "M,N,dtype,bias\n1,2,bf16,False\n3,4,bf16,False\n")
        self.write(
            tuned,
            "gfx,M,N,dtype,bias,us\n"
            "gfx2,1,2,bf16,False,2\n"
            "gfx1,9,9,bf16,False,4\n"
            "gfx1,3,4,bf16,False,2\n"
            "gfx1,3,4,bf16,True,2\n",
        )
        head = self.commit("new workloads")
        out = self.scan(base, head)
        self.assertIn("1 of 2 new keys absent", out)
        self.assertIn(f"{tuned}:5:", out)
        self.assertIn('"bias": "True"', out)
        self.assertNotIn('"M": "9"', out)

    def test_model_file_does_not_fallback_to_canonical_or_mismatched_arch(self):
        self.write("aiter/configs/example_untuned.csv", "token\n8\n")
        self.write(
            "aiter/configs/model_configs/foo_untuned_fmoe_gfx942.csv", "token\n8\n"
        )
        base = self.commit("canonical and arch-specific inputs")
        self.write(
            "aiter/configs/model_configs/variant_example_tuned.csv", "token,us\n99,1\n"
        )
        self.write(
            "aiter/configs/model_configs/foo_tuned_fmoe_gfx950.csv", "token,us\n99,1\n"
        )
        head = self.commit("unpaired model tuning")
        out = self.scan(base, head)
        self.assertIn("No paired tuning CSVs to check.", out)
        self.assertNotIn("COVERED", out)
        self.assertNotIn("MISSING", out)

    def test_architecture_is_compared_when_it_is_part_of_the_input(self):
        self.write("aiter/configs/example_untuned.csv", "gfx,M\ngfx1,16\n")
        base = self.commit("architecture-specific input")
        self.write(
            "aiter/configs/example_tuned.csv",
            "gfx,M,us\ngfx1,16,1\ngfx2,16,1\n",
        )
        head = self.commit("architecture-specific tuning")
        out = self.scan(base, head)
        self.assertIn("1 of 2 new keys absent", out)
        self.assertIn('"gfx": "gfx2"', out)

    def test_exact_input_takes_precedence_and_numeric_values_match(self):
        self.write("aiter/configs/example_untuned.csv", "M\n32\n")
        base = self.commit("canonical input")
        self.write(
            "aiter/configs/model_configs/variant_example_untuned.csv",
            "M,mode\n16.0,fast\n",
        )
        self.write(
            "aiter/configs/model_configs/variant_example_tuned.csv",
            "M,mode,us\n16,fast,1\n",
        )
        head = self.commit("paired input")
        out = self.scan(base, head)
        self.assertIn("COVERED", out)
        self.assertIn("1 new workload keys", out)
        self.assertIn("variant_example_untuned.csv", out)

    def test_unmatched_unsupported_and_non_csv_configs_are_silently_skipped(self):
        base = self.git("rev-parse", "HEAD")
        self.write("aiter/configs/unknown_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/unknown_untuned.csv", "M\n1\n")
        self.write("aiter/configs/no_input_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/unclassified.csv", "M,us\n1,2\n")
        self.write(
            "aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_opt_untuned.csv",
            "dtype,K,V\nbf16,128,128\n",
        )
        self.write(
            "aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_opt_tuned.csv",
            "dtype,K,V,total_chunks,us\nbf16,128,128,10,1.0\n",
        )
        self.write(
            "aiter/configs/model_configs/paged_mqa_logits_untuned_glm5.csv",
            "batch_size,context_len\n1,8192\n",
        )
        self.write(
            "aiter/configs/model_configs/paged_mqa_logits_tuned_glm5.csv",
            "gfx,batch_size,context_len,us\ngfx950,1,8192;60000,4.1\n",
        )
        self.write("aiter/ops/triton/configs/gfx950/op/DEFAULT.json", "{}\n")
        self.write("aiter/configs/other.yaml", "key: value\n")
        self.write("aiter/configs/other.toml", "key = 1\n")
        head = self.commit("unsupported and non-csv formats")
        out = self.scan(base, head)
        self.assertIn("COVERED aiter/configs/unknown_tuned.csv", out)
        self.assertNotIn("UNVERIFIED", out)
        self.assertNotIn("no_input_tuned.csv", out)
        self.assertNotIn("chunk_gdn_h_opt", out)
        self.assertNotIn("paged_mqa_logits", out)
        self.assertNotIn("DEFAULT.json", out)
        self.assertNotIn("other.toml", out)

    def test_differing_valid_schemas_are_silently_skipped(self):
        self.write("aiter/configs/example_untuned.csv", "M,dtype\n1,bf16\n")
        self.write("aiter/configs/schema_untuned.csv", "M\n1\n")
        self.write("aiter/configs/schema_tuned.csv", "M,us\n1,2\n")
        base = self.commit("input")
        self.write("aiter/configs/example_tuned.csv", "M,us\n1,2\n")
        self.write("aiter/configs/schema_untuned.csv", "M,dtype\n1,bf16\n")
        self.write("aiter/configs/schema_tuned.csv", "M,dtype,us\n1,bf16,2\n")
        head = self.commit("differing schemas")
        out = self.scan(base, head)
        self.assertIn("No paired tuning CSVs to check.", out)
        self.assertNotIn("COVERED", out)
        self.assertNotIn("MISSING", out)

    def test_malformed_paired_csv_emits_skipped_diagnostic_and_fails(self):
        base = self.git("rev-parse", "HEAD")
        self.write("aiter/configs/second_untuned.csv", "M\n1,extra\n")
        self.write("aiter/configs/second_tuned.csv", "M,us\n1,2\n")
        head = self.commit("malformed paired csv")
        out = self.scan(base, head, success=False)
        self.assertIn("SKIPPED:", out)
        self.assertIn("row has 2 fields, expected 1", out)
        self.assertNotIn("No paired tuning CSVs to check.", out)

    def test_fp8_arch_native_normalization_and_conservative_rejection(self):
        untuned = "aiter/configs/a8w8_untuned_gemm.csv"
        tuned = "aiter/configs/a8w8_tuned_gemm.csv"
        self.write(
            untuned,
            "M,N,K,q_dtype_a,q_dtype_w\n"
            "16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn\n"
            "32,64,128,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz\n",
        )
        self.write(tuned, "gfx,M,N,K,q_dtype_a,q_dtype_w,us\n")
        base = self.commit("fp8 base")
        self.write(
            tuned,
            "gfx,M,N,K,q_dtype_a,q_dtype_w,us\n"
            "gfx942,16,64,128,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,1.0\n"
            "gfx950,16,64,128,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,1.0\n"
            "gfx950,32,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn,1.0\n"
            "gfx942,16,64,128,torch.float8_e5m2,torch.float8_e5m2,1.0\n"
            "gfx942,16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn,1.0\n",
        )
        head = self.commit("fp8 tuned")
        out = self.scan(base, head)
        self.assertIn("3 of 5 new keys absent", out)
        self.assertIn(f"{tuned}:3:", out)
        self.assertIn(f"{tuned}:5:", out)
        self.assertIn(f"{tuned}:6:", out)

    def test_fp8_old_arch_does_not_mask_new_arch_and_gfxless_is_skipped(self):
        untuned = "aiter/configs/a8w8_untuned_gemm.csv"
        tuned = "aiter/configs/a8w8_tuned_gemm.csv"
        self.write(
            untuned,
            "M,N,K,q_dtype_a,q_dtype_w\n"
            "16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn\n",
        )
        self.write(
            tuned,
            "gfx,M,N,K,q_dtype_a,q_dtype_w,us\n"
            "gfx950,16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn,1.0\n",
        )
        self.write(
            "aiter/configs/untuned_fmoe.csv",
            "token,q_dtype_a,q_dtype_w\n16,torch.float8_e4m3fn,torch.float8_e4m3fn\n",
        )
        base = self.commit("fp8 legacy gfx950")
        self.write(
            tuned,
            "gfx,M,N,K,q_dtype_a,q_dtype_w,us\n"
            "gfx950,16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn,1.0\n"
            "gfx942,16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn,1.0\n",
        )
        self.write(
            "aiter/configs/tuned_fmoe.csv",
            "cu_num,token,q_dtype_a,q_dtype_w,us\n"
            "256,16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,1.0\n",
        )
        head = self.commit("fp8 new gfx942 and gfx-less fmoe")
        out = self.scan(base, head)
        self.assertIn(f"MISSING {tuned}: 1 of 1 new keys absent", out)
        self.assertIn(f"{tuned}:3:", out)
        self.assertNotIn("tuned_fmoe.csv", out)

    def test_moe_quant_remap_and_boolean_case(self):
        untuned = "aiter/configs/model_configs/mimo_untuned_fmoe.csv"
        tuned = "aiter/configs/model_configs/mimo_tuned_fmoe.csv"
        self.write(untuned, "token,q_type,bias\n16,QuantType.per_128x128,FALSE\n")
        base = self.commit("moe base")
        self.write(
            tuned,
            "gfx,token,q_type,bias,us\ngfx950,16,QuantType.per_1x128,False,1.2\n",
        )
        head = self.commit("moe tuned")
        out = self.scan(base, head)
        self.assertIn("COVERED", out)
        self.assertIn("1 new workload keys", out)

    def test_model_configs_alternate_token_order_and_arch_suffix(self):
        self.write("aiter/configs/bf16_untuned_gemm.csv", "M,N\n4352,1024\n")
        self.write("aiter/configs/untuned_fmoe.csv", "token\n999\n")
        self.write(
            "aiter/configs/model_configs/llama70B_untuned_gemm_bf16.csv",
            "M,N\n1,1280\n",
        )
        self.write(
            "aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv",
            "token\n256\n",
        )
        base = self.commit("model untuned")
        self.write(
            "aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv",
            "M,N,us\n4352,1024,1.0\n",
        )
        self.write(
            "aiter/configs/model_configs/gptoss_fp4_gfx950_tuned_fmoe.csv",
            "gfx,token,us\ngfx950,256,1.0\n",
        )
        head = self.commit("model tuned")
        out = self.scan(base, head)
        self.assertIn(
            "MISSING aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv", out
        )
        self.assertIn("llama70B_untuned_gemm_bf16.csv", out)
        self.assertIn(
            "COVERED aiter/configs/model_configs/gptoss_fp4_gfx950_tuned_fmoe.csv", out
        )
        self.assertIn("gptoss_fp4_untuned_fmoe.csv", out)

    def test_ambiguous_model_configs_sources_are_skipped(self):
        self.write("aiter/configs/model_configs/foo_untuned_gemm_gfx942.csv", "M\n1\n")
        self.write("aiter/configs/model_configs/foo_untuned_gemm_gfx950.csv", "M\n1\n")
        base = self.commit("ambiguous untuned")
        self.write(
            "aiter/configs/model_configs/foo_tuned_gemm_gfx1250.csv", "M,us\n1,2\n"
        )
        head = self.commit("tuned")
        out = self.scan(base, head)
        self.assertIn("No paired tuning CSVs to check.", out)
        self.assertNotIn("COVERED", out)
        self.assertNotIn("MISSING", out)

    def test_renamed_file_and_same_source_split_provenance(self):
        self.write("aiter/configs/model_configs/foo_untuned_gemm.csv", "M,N\n10,20\n")
        self.write(
            "aiter/configs/model_configs/foo_gfx942_tuned_gemm.csv",
            "M,N,us\n1,2,1.0\n5,6,1.0\n10,20,1.0\n",
        )
        base = self.commit("model legacy")
        # Row (1,2) is removed from gfx942 and moved to gfx950; (5,6) stays in
        # gfx942 AND is copied to gfx950 (so (5,6) in gfx950 is a new gap).
        self.write(
            "aiter/configs/model_configs/foo_gfx942_tuned_gemm.csv",
            "M,N,us\n5,6,1.0\n10,20,1.0\n",
        )
        self.write(
            "aiter/configs/model_configs/foo_gfx950_tuned_gemm.csv",
            "M,N,us\n1,2,1.0\n5,6,1.0\n",
        )
        head = self.commit("split legacy row")
        out = self.scan(base, head)
        self.assertIn(
            "MISSING aiter/configs/model_configs/foo_gfx950_tuned_gemm.csv: 1 of 1",
            out,
        )
        self.assertIn('"M": "5"', out)
        self.assertNotIn('"M": "1"', out)

    def test_cross_source_rename_does_not_suppress_missing(self):
        self.write(
            "aiter/configs/model_configs/model_a_untuned_gemm.csv",
            "M,N\n1,2\n3,4\n",
        )
        self.write(
            "aiter/configs/model_configs/model_a_tuned_gemm.csv",
            "M,N,us\n1,2,1\n3,4,1\n",
        )
        self.write("aiter/configs/model_configs/model_b_untuned_gemm.csv", "M,N\n1,2\n")
        base = self.commit("two models")
        (self.repo / "aiter/configs/model_configs/model_a_tuned_gemm.csv").rename(
            self.repo / "aiter/configs/model_configs/model_b_tuned_gemm.csv"
        )
        head = self.commit("cross-source rename")
        out = self.scan(base, head)
        self.assertIn(
            "MISSING aiter/configs/model_configs/model_b_tuned_gemm.csv: 1 of 2", out
        )
        self.assertIn('"M": "3"', out)

    def test_hyphen_delimited_tuned_and_untuned_names(self):
        base = self.git("rev-parse", "HEAD")
        self.write("aiter/configs/model_configs/foo-untuned.csv", "M\n1\n")
        self.write("aiter/configs/model_configs/foo-tuned.csv", "M,us\n1,2\n")
        head = self.commit("hyphenated")
        out = self.scan(base, head)
        self.assertIn("COVERED aiter/configs/model_configs/foo-tuned.csv", out)

    def test_disconnected_merge_base_emits_single_clear_skipped_line(self):
        base = self.git("rev-parse", "HEAD")
        self.git("checkout", "--orphan", "orphan-branch")
        self.write("aiter/configs/example_tuned.csv", "M,us\n1,2\n")
        head = self.commit("orphan")
        out = self.scan(base, head, success=False)
        self.assertIn("SKIPPED:", out)
        self.assertIn("no common ancestor", out)
        self.assertIn("review these files manually", out)
        self.assertEqual(out.count("SKIPPED:"), 1)

    def test_fetch_runs_scanner_and_degrades_on_failure(self):
        self.write("aiter/configs/example_untuned.csv", "M\n1\n")
        base = self.commit("input")
        self.write("aiter/configs/example_tuned.csv", "M,us\n2,3\n")
        head = self.commit("new workload")
        fetch = SCRIPT.with_name("fetch.sh").read_text(encoding="utf-8")
        start = fetch.index("# Compare the PR's new workload keys")
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
        subprocess.run(
            ["bash", "-c", snippet], env=env, check=True, capture_output=True
        )
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertIn("MISSING aiter/configs/example_tuned.csv", output)

        env["HEAD_SHA"] = "missing-commit"
        subprocess.run(
            ["bash", "-c", snippet], env=env, check=True, capture_output=True
        )
        output = (evidence / "tuning_coverage.txt").read_text(encoding="utf-8")
        self.assertEqual(output.count("SKIPPED:"), 1)
        self.assertIn("review these files manually", output)

    def test_fp8_migration_without_base_or_sibling_gfx_is_skipped(self):
        untuned = "aiter/configs/a8w8_untuned_gemm.csv"
        tuned = "aiter/configs/a8w8_tuned_gemm.csv"
        self.write(
            untuned,
            "M,N,K,q_dtype_a,q_dtype_w\n"
            "16,64,128,torch.float8_e4m3fn,torch.float8_e4m3fn\n",
        )
        # Base tuned file has no gfx column and carries an uncovered legacy row (99,64,128).
        self.write(
            tuned,
            "cu_num,M,N,K,q_dtype_a,q_dtype_w,us\n"
            "256,99,64,128,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,1.0\n",
        )
        base = self.commit("gfxless fp8 base")
        self.write(
            tuned,
            "gfx,cu_num,M,N,K,q_dtype_a,q_dtype_w,us\n"
            "gfx942,256,99,64,128,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,1.0\n",
        )
        head = self.commit("migrate fp8 to gfx")
        out = self.scan(base, head)
        self.assertIn("No paired tuning CSVs to check.", out)
        self.assertNotIn("MISSING", out)

    def test_untuned_with_kernelid_placeholder_is_skipped(self):
        untuned = "aiter/configs/model_configs/foo_untuned_gemm.csv"
        tuned = "aiter/configs/model_configs/foo_tuned_gemm.csv"
        self.write(untuned, "M,N,K,kernelId\n16,64,128,\n")
        base = self.commit("untuned with kernelId")
        self.write(tuned, "M,N,K,kernelId,us\n16,64,128,42,1.0\n")
        head = self.commit("tuned with kernelId")
        out = self.scan(base, head)
        self.assertIn("No paired tuning CSVs to check.", out)
        self.assertNotIn("MISSING", out)


if __name__ == "__main__":
    unittest.main()
