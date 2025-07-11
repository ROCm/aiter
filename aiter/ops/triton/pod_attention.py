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


def pod_attention(
    cu_ctr: torch.Tensor,
    # Decode
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    # causal: bool,
    batch_size: int,
    sm_scale: torch.float16,
    num_warps,
    waves_per_eu,
    # Prefill
    q_pf: torch.Tensor,
    k_pf: torch.Tensor,
    v_pf: torch.Tensor,
    Mp_pf: torch.Tensor,
    Lp_pf: torch.Tensor,
    Op_pf: torch.Tensor,
    locks_pf: torch.Tensor,
    batch_num_block_n_pf: torch.Tensor,
    BLOCK_M_pf: int,
    BLOCK_N_pf: int,
    # causal_pf: bool,
    batch_size_pf: int,
    prefill_ratio: int,
    decode_ratio: int,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # Calculate Decode Params
    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]

    qk_scale = sm_scale * 1.44269504

    # We assume the kernel functions fused by pod attention are persistent kernel functions
    # For MI300, we launch total 608 WGs. Each CU will get 2 WG --- one WG will be doing decode and one prefill
    # For different decode:prefill ratios, assign (decode+prefill)*304 number of WGs
    total_wgs = total_programs // 2

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        False,  # causal
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H,
        BLOCK_M,
        BLOCK_N,
        total_wgs,
    )
    # print(" Decode LA params")
    # print(f" num_m_blocks={num_m_blocks}, high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}")
    # print(f" tiles_per_head={tiles_per_head}, total_wgs={total_wgs}")

    o = torch.empty_like(q, dtype=v.dtype)

    # Calculate Prefill Params
    N_CTX_Q_pf = q_pf.shape[0] // batch_size
    N_CTX_K_pf = k_pf.shape[0]  # This is the sum of all ctx_n in a batch

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For MI300, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M_pf // BLOCK_N_pf

    # if causal_pf:
    # Only support BLOCK_M is multiple of BLOCK_N
    # TODO: add other scenarios
    assert BLOCK_M_pf % BLOCK_N_pf == 0

    #    num_m_blocks_pf, high_load_wgs_pf, max_tiles_per_wg_pf, tiles_per_head_pf, num_splits_pf, even_split_pf = (
    #        get_num_splits_and_buffer_sizes(causal_pf, N_CTX_Q_pf, N_CTX_K_pf, H, H, HEAD_DIM_Q, BLOCK_M_pf, BLOCK_N_pf, total_programs)
    #    )
    (
        num_m_blocks_pf,
        num_n_blocks_pf,
        high_load_wgs_pf,
        max_tiles_per_wg_pf,
        tiles_per_head_pf,
        num_splits_pf,
        even_split_pf,
    ) = get_num_splits_and_buffer_sizes(
        True,  # causal,
        batch_size_pf,
        N_CTX_Q_pf,
        N_CTX_K_pf,
        H,
        H,
        BLOCK_M_pf,
        BLOCK_N_pf,
        total_wgs,
    )
    print("\n Prefill LA params")
    print(
        f" num_m_blocks={num_m_blocks_pf}, high_load_wgs={high_load_wgs_pf}, max_tiles_per_wg={max_tiles_per_wg_pf}"
    )
    print(f" tiles_per_head={tiles_per_head_pf}, total_wgs={total_wgs}")
    print(
        f" BLOCK_M_pf={BLOCK_M_pf}, BLOCK_N_pf={BLOCK_N_pf}, MASKED_BLOCKS={MASKED_BLOCKS}"
    )
    print(
        f" batch_size_pf={batch_size_pf}, num_m_blocks_pf={num_m_blocks_pf}, num_n_blocks_pf={num_n_blocks_pf}"
    )

    print(f" Launching {total_programs} of kernels")

    grid = (total_programs, 1, 1)

    o_pf = torch.empty_like(q_pf, dtype=v_pf.dtype)

    pod_kernel = pod_persistent[grid](
        cu_ctr,
        # Decode positional arguments
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
        batch_num_block_n,
        locks,
        q.stride(0),  # N_CTX_Q
        q.stride(1),  # H
        q.stride(2),  # HEAD_DIM
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        Op.stride(0),  # total_programs
        Op.stride(1),  # N_CTX_Q
        Op.stride(2),  # HEAD_DIM
        # Prefill positional arguments
        q_pf,
        k_pf,
        v_pf,
        Mp_pf,
        Lp_pf,
        Op_pf,
        o_pf,
        batch_num_block_n_pf,
        locks_pf,
        q_pf.stride(0),
        q_pf.stride(1),
        q_pf.stride(2),
        k_pf.stride(0),
        k_pf.stride(1),
        k_pf.stride(2),
        v_pf.stride(0),
        v_pf.stride(1),
        v_pf.stride(2),
        o_pf.stride(0),
        o_pf.stride(1),
        o_pf.stride(2),
        Op_pf.stride(0),
        Op_pf.stride(1),
        Op_pf.stride(2),
        # Decode keyword argument
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        # Prefill keyword argument
        # HEAD_DIM=HEAD_DIM_K,
        BLOCK_M_pf=BLOCK_M_pf,
        BLOCK_N_pf=BLOCK_N_pf,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size_pf=batch_size_pf,
        # causal_pf=causal_pf,
        num_m_blocks_pf=num_m_blocks_pf,
        num_n_blocks_pf=num_n_blocks_pf,
        # leanAttention params
        high_load_wgs_pf=high_load_wgs_pf,
        max_tiles_per_wg_pf=max_tiles_per_wg_pf,
        tiles_per_head_pf=tiles_per_head_pf,
        num_splits_pf=num_splits_pf,
        prefill_ratio=prefill_ratio,
        decode_ratio=decode_ratio,
    )
    # torch.cuda.synchronize()
    print(
        f"pod kernel {pod_kernel.n_regs} registers used, {pod_kernel.n_spills} spills"
    )

    return o, o_pf


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    ##### Lean Atteion: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    # print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")
    # print(f"num_SMs: {num_SMs}")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 2 tiles per CTA and 2 CTAs per SM
    lean_griddimz = num_SMs  # CTA launch grid
    # if (total_tiles <= 2 * 2 * num_SMs):
    #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
    # else:
    #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

    # Max number lean tiles per task block (CTA)
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    # Find max number of splits
    num_splits = 0
    even_split = False
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    # print(f"total_tiles={total_tiles}, max_tiles_per_tb={max_tiles_per_tb}, high_load_tbs={high_load_tbs}")
    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        num_splits,
        even_split,
    )


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
        )
        tl.debug_barrier()
        # prefill_time = read_realtime() - prefill_time
        # if gcu_id==0:
        #    tl.device_print("time to run prefill kernel", prefill_time)
        # tl.device_print("gcu_id for prefill", gcu_id)
