import argparse
import os
import shutil
from pathlib import Path
import json
from aiter.jit.core import build_module, get_args_of_build, get_user_jit_dir

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"root_dir:{root_dir}")

def copy_built_kernels(out_dir: Path, module_names: list) -> None:
    """Copy built kernel files to output directory"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jit_dir = Path(get_user_jit_dir())
    for module_name in module_names:
        so_file = jit_dir / f"{module_name}.so"
        if so_file.exists():
            dst = out_dir / f"{module_name}.so"
            shutil.copy2(so_file, dst)

def main():
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build modules for Aiter"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for built kernels",
        default=os.path.join(root_dir, "aot-ops")
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        type=str,
        help="List of module names to build (new or modified). If not provided, uses default set"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild all modules regardless of modification status"
    )
    args = parser.parse_args()
        
    # Print build configuration
    print("Aiter AOT build summary:")
    print(f"  out_dir: {args.out_dir}")
    
    # Use provided modules or default set
    modules_to_build = args.modules or [
        # "module_mha_varlen_fwd",
        "module_rope_general_fwd",
        "module_rope_pos_fwd",
        "module_moe_asm",
        "module_fused_moe_bf16_asm",
        "module_gemm_a8w8",
        "module_gemm_a8w8_bpreshuffle",
    ]
    print(f"modules_to_build: {modules_to_build}")
    # Load module configurations
    config_path = Path(__file__).resolve().parents[1] / "jit/optCompilerConfig.json"
    with open(config_path, 'r') as f:
        opt_config = json.load(f)
        valid_modules = [m for m in modules_to_build if m in opt_config]
        
        # Filter unchanged modules unless force-rebuild is specified
        if not args.force_rebuild:
            jit_dir = Path(get_user_jit_dir())
            valid_modules = [m for m in valid_modules 
                            if not (jit_dir / f"{m}.so").exists()]
            
        if not valid_modules:
            print("No modules to build. All requested modules are up-to-date.")
            return
            
        print(f"Building {len(valid_modules)} modules...")
        for module_name in valid_modules:
            so_file = jit_dir / f"{module_name}.so"
            status = "Rebuilding" if so_file.exists() else "Building"
            print(f"{status} module: {module_name}")
            build_args = get_args_of_build(module_name)
            
            print(f"srcs: {build_args.get("srcs")}")

            filtered_args = {
                'srcs': [el for el in build_args.get('srcs', []) if "pybind.cu" not in el],
                'flags_extra_cc': build_args.get('flags_extra_cc', []) + ['-DPREBUILD_KERNELS'],
                'flags_extra_hip': build_args.get('flags_extra_hip', []) + ['-DPREBUILD_KERNELS'],
                'blob_gen_cmd': build_args.get('blob_gen_cmd', ''),
                'extra_include': build_args.get('extra_include', []) + [os.path.join(root_dir, "csrc")],
                'extra_ldflags': build_args.get('extra_ldflags', None),
                'verbose': build_args.get('verbose', False),
                'is_python_module': build_args.get('is_python_module', True),
                'is_standalone': build_args.get('is_standalone', False),
                'torch_exclude': build_args.get('torch_exclude', False)
            }
            
            build_module(module_name, **filtered_args)
    
    # Copy built kernels to output directory
    copy_built_kernels(args.out_dir, valid_modules)
    print(f"AOT kernels saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
