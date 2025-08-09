import argparse
import os
import shutil
from pathlib import Path
import json
from aiter.jit.core import (
    CK_DIR,
    AITER_CSRC_DIR,
    build_module,
    get_args_of_build,
    get_user_jit_dir,
)

root_dir = Path(__file__).resolve().parents[2]


def copy_built_kernels(out_dir: Path, module_names: list) -> None:
    """Copy built kernel files to output directory"""
    jit_dir = Path(get_user_jit_dir())
    if out_dir == jit_dir:
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for module_name in module_names:
        so_file = jit_dir / f"{module_name}.so"
        if so_file.exists():
            dst = out_dir / f"{module_name}.so"
            shutil.copy2(so_file, dst)


def get_filter_fwd_from_md_name(md_name):
    filter_fwd = "*"
    parts = md_name.split("_")[3:]  # skip 'mha', 'varlen', 'fwd'

    for p in parts:
        if p == "fp16":
            filter_fwd += "fp16*"
        elif p == "bf16":
            filter_fwd += "bf16*"
        elif p == "logits":
            filter_fwd += "_logits*"
        elif p == "nlogits":
            filter_fwd += "_nlogits*"
        elif p == "bias":
            filter_fwd += "_bias*"
        elif p == "alibi":
            filter_fwd += "_alibi*"
        elif p == "nbias":
            filter_fwd += "_nbias*"
        elif p == "nmask":
            filter_fwd += "_nmask*"
        elif p == "mask":
            filter_fwd += "_mask*"
        elif p == "lse":
            filter_fwd += "_lse*"
        elif p == "nlse":
            filter_fwd += "_nlse*"
        elif p == "ndropout":
            filter_fwd += "_ndropout*"
        elif p == "dropout":
            filter_fwd += "_dropout*"
        elif p == "nskip":
            filter_fwd += "_nskip*"
        elif p == "skip":
            filter_fwd += "_skip*"
    return filter_fwd


def main():
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build modules for Aiter"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for built kernels",
        default=os.path.join(root_dir, "aot-ops"),
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        type=str,
        help="List of module names to build (new or modified). If not provided, uses default set",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild all modules regardless of modification status",
    )
    args = parser.parse_args()

    # Use provided modules or default set
    modules_to_build = args.modules or [
        "module_mha_varlen_fwd",
        "module_moe_ck2stages",
        # "moduel_fmla_asm_fwd" # TODO: waiting on aiter merge
        "module_quant",
        "module_rmsnorm",
        "module_rocsolgemm",
        "module_hipbsolgemm",
        "module_rope_general_fwd",
        "module_rope_pos_fwd",
        "module_moe_asm",
        "module_moe_sorting",
        "module_fused_moe_bf16_asm",
        "module_gemm_a8w8",
        "module_gemm_a8w8_bpreshuffle",
        "module_custom_all_reduce",
    ]
    print(f"modules_to_build: {modules_to_build}")
    # Load module configurations
    config_path = Path(__file__).resolve().parents[1] / "jit/optCompilerConfig.json"
    with open(config_path, "r") as f:
        opt_config = json.load(f)
        valid_modules = [m for m in modules_to_build if m in opt_config]

        # Filter unchanged modules unless force-rebuild is specified
        if not args.force_rebuild:
            jit_dir = Path(get_user_jit_dir())
            valid_modules = [
                m for m in valid_modules if not (jit_dir / f"{m}.so").exists()
            ]

        if not valid_modules:
            print("No modules to build. All requested modules are up-to-date.")
            return

        print(f"Building {len(valid_modules)} modules...")
        for module_name in valid_modules:
            so_file = jit_dir / f"{module_name}.so"
            status = "Rebuilding" if so_file.exists() else "Building"
            print(f"{status} module: {module_name}")
            build_args = get_args_of_build(module_name)
            filtered_args = {
                "srcs": [el for el in build_args.get("srcs", [])],
                "flags_extra_cc": build_args.get("flags_extra_cc", []),
                "flags_extra_hip": build_args.get("flags_extra_hip", []),
                "blob_gen_cmd": build_args.get("blob_gen_cmd", ""),
                "extra_include": build_args.get("extra_include", [])
                + [os.path.join(root_dir, "csrc", "include")],
                "extra_ldflags": build_args.get("extra_ldflags", None),
                "verbose": build_args.get("verbose", False),
                "is_python_module": build_args.get("is_python_module", True),
                "is_standalone": build_args.get("is_standalone", False),
                "torch_exclude": build_args.get("torch_exclude", False),
            }

            if module_name == "module_mha_varlen_fwd":
                # Build runtime variants explicitly
                rt_module_names = [
                    "mha_varlen_fwd_bf16_nlogits_nbias_mask_lse_ndropout_nskip",
                    "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_nskip",
                    "mha_varlen_fwd_bf16_nlogits_nbias_nmask_lse_ndropout_nskip",
                ]

                for rt_module_name in rt_module_names:
                    so_file = jit_dir / f"{rt_module_name}.so"
                    if not so_file.exists() or args.force_rebuild:
                        variant_args = filtered_args.copy()
                        filter_fwd = get_filter_fwd_from_md_name(rt_module_name)
                        blob_gen_cmd = [
                            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd "
                            "--receipt 200 --filter {} --output_dir {{}}".format(
                                filter_fwd
                            )
                        ]
                        blob_gen_cmd.append(
                            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv "
                            "--receipt 200 --filter {} --output_dir {{}}".format(
                                '" @ "'
                            )
                        )
                        blob_gen_cmd.append(
                            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_generate.py --receipt 3 --output_dir {{}}"
                        )
                        variant_args["blob_gen_cmd"] = blob_gen_cmd
                        build_module(rt_module_name, **variant_args)
            elif module_name == "module_moe_ck2stages":
                # Build runtime variants explicitly
                rt_module_names = [
                    "module_moe_ck2stages_f8_f8_b16_silu_per_token_mulWeightStage2",
                ]
                for rt_module_name in rt_module_names:
                    so_file = jit_dir / f"{rt_module_name}.so"
                    if not so_file.exists() or args.force_rebuild:
                        variant_args = filtered_args.copy()
                        parts = rt_module_name.split("_")
                        # prefix = "_".join(parts[:3])  # 'module_moe_ck2stages'
                        Adtype = parts[3]
                        Bdtype = parts[4]
                        Cdtype = parts[5]
                        act = parts[6]
                        quant_type = "_".join(parts[7:-1])  # "per_token"
                        mul_routed_weight_stage = int(
                            parts[-1].replace("mulWeightStage", "")
                        )
                        blob_gen_cmd = [
                            f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -a {Adtype} -b {Bdtype} -c {Cdtype} -q {quant_type} -act {act} -m {mul_routed_weight_stage} -w {{}}"
                        ]
                        variant_args["blob_gen_cmd"] = blob_gen_cmd
                        build_module(rt_module_name, **variant_args)
            else:
                build_module(module_name, **filtered_args)

    # Copy built kernels to output directory
    copy_built_kernels(args.out_dir, valid_modules)
    print(f"AOT kernels saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
