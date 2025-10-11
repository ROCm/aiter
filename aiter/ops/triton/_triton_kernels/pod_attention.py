import torch
import triton
import triton.language as tl

import importlib.util
from pathlib import Path


file_path = Path("./aiter/ops/triton/lean_atten.py").resolve()
module_name = "la_persistent"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp


@triton.jit
def get_cu_id():
    # HW_ID Register bit structure for GCN and CDNA
    #   WAVE_ID     3:0     Wave buffer slot number. 0-9.
    #   SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
    #   PIPE_ID     7:6     Pipeline from which the wave was dispatched.
    #   CU_ID       11:8    Compute Unit the wave is assigned to.
    #   SH_ID       12      Shader Array (within an SE) the wave is assigned to.
    #   SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a
    #               14:13   Shader Engine the wave is assigned to for 942
    #   TG_ID       19:16   Thread-group ID
    #   VM_ID       23:20   Virtual Memory ID
    #   QUEUE_ID    26:24   Queue from which this wave was dispatched.
    #   STATE_ID    29:27   State ID (graphics only, not compute).
    #   ME_ID       31:30   Micro-engine ID.

    # XCC_ID Register bit structure for 942/950
    #   XCC_ID      3:0     XCC the wave is assigned to.

    (cu_id, se_id, xcc_id) = tl.inline_asm_elementwise(
        asm="""
        s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)
        s_getreg_b32 $1, hwreg(HW_REG_HW_ID, 13, 2)
        s_getreg_b32 $2, hwreg(HW_REG_XCC_ID, 0, 4)
        s_waitcnt lgkmcnt(0)
        """,
        constraints=("=s,=s,=s"),  # Three scalar output
        args=[],  # No inputs
        dtype=(tl.int32, tl.int32, tl.int32),  # Output type is int32
        is_pure=False,
        pack=1,
    )
    return (cu_id, se_id, xcc_id)


@triton.jit
def pod_persistent(
    # Prefill/Decode Communication
    cu_ctr,
    # Decode
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    # Prefill
    Q_pf,
    K_pf,
    V_pf,
    Mp_pf,
    Lp_pf,
    Op_pf,
    Out_pf,
    batch_num_block_n_pf,
    locks_pf,
    stride_qm_pf,
    stride_qh_pf,
    stride_qk_pf,
    stride_kn_pf,
    stride_kh_pf,
    stride_kk_pf,
    stride_vn_pf,
    stride_vh_pf,
    stride_vk_pf,
    stride_om_pf,
    stride_oh_pf,
    stride_on_pf,
    stride_oph_pf,
    stride_opm_pf,
    stride_opn_pf,
    # Decode
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    # Prefill
    # HEAD_DIM: tl.constexpr,
    BLOCK_M_pf: tl.constexpr,
    BLOCK_N_pf: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    batch_size_pf: tl.constexpr,
    # causal: tl.constexpr,
    num_m_blocks_pf: tl.constexpr,
    num_n_blocks_pf: tl.constexpr,
    # leanAttention params
    high_load_wgs_pf: tl.constexpr,
    max_tiles_per_wg_pf: tl.constexpr,
    tiles_per_head_pf: tl.constexpr,
    num_splits_pf: tl.constexpr,
    # Prefill/Decode common
    prefill_ratio: tl.constexpr,
    decode_ratio: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
):

    # cu_id: 4 bits, se_id: 2 bits, xcc_id: 4 bits
    (cu_id, se_id, xcc_id) = get_cu_id()
    gcu_id = (xcc_id << 6) + (se_id << 4) + cu_id
    # tl.device_print("gcu_id is ", gcu_id)

    # cu_ctr is initialized to zero
    # tl.atomic_add(cu_ctr + gcu_id, 1)
    ratio = prefill_ratio + decode_ratio
    op = 0  # 0 - decode
    ticket = (tl.atomic_add(cu_ctr + gcu_id, 1)) % ratio
    # ticket=tl.atomic_add(cu_ctr,1)
    # if ticket >= 304:
    #    op=1
    if ticket < prefill_ratio:
        op = 1  # 1 - prefill

    current_pid = tl.program_id(0) % 304
    # if gcu_id==352:
    #    tl.device_print("ticket is", ticket)
    #    tl.device_print("op is ", op)
    # tl.device_print("op is:", op)
    if op == 0:  # 0 - decode
        # decode_time = read_realtime()
        # if gcu_id==0:
        #    tl.device_print("time to start decode kernel", decode_time)
        module.la_persistent(
            True,
            current_pid,
            Q,
            K,
            V,
            qk_scale,
            Mp,
            Lp,
            Op,
            Out,
            batch_num_block_n,
            locks,
            stride_qm,
            stride_qh,
            stride_qk,
            stride_kn,
            stride_kh,
            stride_kk,
            stride_vn,
            stride_vh,
            stride_vk,
            stride_om,
            stride_oh,
            stride_on,
            stride_oph,
            stride_opm,
            stride_opn,
            HEAD_DIM,  #: tl.constexpr,
            BLOCK_M,  #: tl.constexpr,
            BLOCK_N,  #: tl.constexpr,
            MASKED_BLOCKS,
            batch_size,  #: tl.constexpr,
            False,  # tl.constexpr,
            num_m_blocks,  #: tl.constexpr,
            num_n_blocks,
            # leanAttention params
            high_load_wgs,  #: tl.constexpr,
            max_tiles_per_wg,  #: tl.constexpr,
            tiles_per_head,  #: tl.constexpr,
            num_splits,  #: tl.constexpr,
            max_output_tile_cnt,
        )
        tl.debug_barrier()
        # decode_time = read_realtime() - decode_time
        # if gcu_id==0:
        #    tl.device_print("time to run decode", decode_time)

        # tl.device_print("gcu_id for decode", gcu_id)
    else:
        # prefill_time = read_realtime()
        # if gcu_id==0:
        #    tl.device_print("time to start prefill kernel", prefill_time)
        # tl.device_print("gcu_id start prefill kernel", gcu_id)
        module.la_persistent(
            True,
            current_pid,
            Q_pf,
            K_pf,
            V_pf,
            qk_scale,
            Mp_pf,
            Lp_pf,
            Op_pf,
            Out_pf,
            batch_num_block_n_pf,
            locks_pf,
            stride_qm_pf,
            stride_qh_pf,
            stride_qk_pf,
            stride_kn_pf,
            stride_kh_pf,
            stride_kk_pf,
            stride_vn_pf,
            stride_vh_pf,
            stride_vk_pf,
            stride_om_pf,
            stride_oh_pf,
            stride_on_pf,
            stride_oph_pf,
            stride_opm_pf,
            stride_opn_pf,
            HEAD_DIM,  #: tl.constexpr,
            BLOCK_M_pf,  #: tl.constexpr,
            BLOCK_N_pf,  #: tl.constexpr,
            MASKED_BLOCKS,
            batch_size_pf,  #: tl.constexpr,
            True,  # causaltl.constexpr,
            num_m_blocks_pf,  #: tl.constexpr,
            num_n_blocks_pf,
            # leanAttention params
            high_load_wgs_pf,  #: tl.constexpr,
            max_tiles_per_wg_pf,  #: tl.constexpr,
            tiles_per_head_pf,  #: tl.constexpr,
            num_splits_pf,  #: tl.constexpr,
            max_output_tile_cnt,
        )
        tl.debug_barrier()
        # prefill_time = read_realtime() - prefill_time
        # if gcu_id==0:
        #    tl.device_print("time to run prefill kernel", prefill_time)
        # tl.device_print("gcu_id for prefill", gcu_id)
