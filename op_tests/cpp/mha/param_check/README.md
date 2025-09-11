# mha_v3_param_check_codegen.py script user guide
1. firstly, the `codegen_info_list` int mha_v3_param_check_codegen.py should be updated based on the latest aiter codegen.py version:
    ```
    CodegenInfo("gfx942", "fmha_v3_fwd", "fmha_fwd_v3(", 226, 394, "mha_fwd_param_v3_check.cpp"),
    # the `fmha_fwd_v3_gfx942_check` fucntion will be generated based on the code snippets of function `fmha_fwd_v3` from 226 line to 394 line in /aiter/hsa/gfx942/fmha_v3_fwd/codegen.py, and will be written into the mha_fwd_param_v3_check.cpp
    ```

2. secondly, use the script `mha_v3_param_check_codegen.py` to generate param check function with aiter working directory as input param `--work_path`:
    `python3 mha_v3_param_check_codegen.py --work_path /home/xxx/xxx/aiter/hsa/`

3. thirdly, if new function generated success, the delete original function `fmha_xxx_v3_gfxxxx_check` in mha_fwd_param_v3_check.cpp or mha_bwd_param_v3_check.cpp
