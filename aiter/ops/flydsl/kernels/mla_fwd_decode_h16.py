import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size

BLOCK_THREADS: int = 512
WAVE_SIZE: int = get_warp_size()
NUM_WAVES: int = BLOCK_THREADS // WAVE_SIZE

# Packed MLA head: 512 NOPE + 64 RoPE
NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
QKV_LORA_DIM: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QKV_LORA_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM: int = QKV_LORA_DIM
PAGE_SIZE: int = 16
KV_COL_BLOCK: int = 64
SIZE_MLA_WORK_INFO_IN_DW: int = 8



@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
def flydsl_mla_fwd_decode_m16_bf16_fp8_16head(
   query: fx.Tensor,  # [num_q, H, 512 + 64] bf16
   kv: fx.Tensor,  # [num_pages, page_size, 1, 512 + 64] fp8
   kv_page_indices: fx.Tensor,  # [num_page_used] i32
   work_indptr: fx.Tensor,  # [num_workers + 1] i32
   work_info_set: fx.Tensor,  # [num_work_items, 8] i32
   split_output: fx.Tensor,  # [num_partial, 1, H, 512] f32
   split_lse: fx.Tensor,  # [num_partial, 1, H, 1] f32
   softmax_scale: fx.Float32,
):
    """Specialized MLA forward decode for heads <= 16."""

    def masked_attention(
        q: fx.Tensor, k: fx.Tensor, scale: fx.Tensor, out: fx.Tensor
    ):
        """Compute bf16 masked attention for the given tensors"""
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        q = fx.rocdl.make_buffer_tensor(A)
        k = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA = fx.zipped_divide(q, (block_m, block_k))
        bB = fx.zipped_divide(k, (block_n, block_k))
        bC = fx.zipped_divide(out, (block_m, block_n))

        bA = fx.slice(bA, (None, bid))
        bB = fx.slice(bB, (None, bid))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(4, 4, 4, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout())
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        copy_src_A = thr_copy_A.partition_S(bA)
        copy_src_B = thr_copy_B.partition_S(bB)
        copy_dst_C = thr_copy_C.partition_S(bC)

        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        copy_frag_A = thr_copy_A.retile(frag_A)
        copy_frag_B = thr_copy_B.retile(frag_B)
        copy_frag_C = thr_copy_C.retile(frag_C)

        fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
        fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


    # Work loop

    q = fx.rocdl.make_buffer_tensor(query)
    kv = fx.rocdl.make_buffer_tensor(kv)
    page_idx = fx.rocdl.make_buffer_tensor(kv_page_indices)
    work_indptr = fx.rocdl.make_buffer_tensor(work_indptr)
    work_info_set = fx.rocdl.make_buffer_tensor(work_info_set)
    split_output = fx.rocdl.make_buffer_tensor(split_output)
    split_lse = fx.rocdl.make_buffer_tensor(split_lse)

    # Each work items is an 8 x uint32 struct (see mla.h)
    # Each persistent workgroup processes its assigned work-set interval.
    worker_idx = fx.gpu.block_idx.x
    work_start = work_indptr[worker_idx]
    work_end = work_indptr[worker_idx + 1]
    for work_idx in range(work_start, work_end):
        work_item = fx.slice(work_info_set, (work_idx, None))

        batch_idx = work_item[0]
        partial_qo_loc = work_item[1]
        qo_start = work_item[2]
        qo_end = work_item[3]
        kv_start_page = work_item[4]
        kv_end_page = work_item[5]
        kv_offset = work_item[6]

    

@flyc.jit
def launch(
    query: fx.Tensor,  # [num_q, H, 576] bf16
    kv_buffer: fx.Tensor,  # [num_pages, page_size, 1, 576] fp8
    kv_page_indices: fx.Tensor,  # [num_page_used] i32
    work_indptr: fx.Tensor,  # [num_workers + 1] i32
    work_info_set: fx.Tensor,  # [num_work_items, 8] i32
    split_output: fx.Tensor,  # [num_partial, 1, H, 512] f32
    split_lse: fx.Tensor,  # [num_partial, 1, H, 1] f32
    softmax_scale: fx.Float32,
    stream: fx.Stream = fx.Stream(None),
):
    grid = (1,1,1)
    block = (512,1,1)
    flydsl_mla_fwd_decode_m16_bf16_fp8_16head(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        split_output,
        split_lse,
        softmax_scale,
     ).launch(grid=grid, block=block, stream=stream)




def _make_inputs(batch, context_len):
    import torch
    from aiter import dtypes

    total_kv = batch * context_len
    query = torch.zeros((batch,  NUM_HEADS, QK_HEAD_DIM), dtype=dtypes.fp8)
    kv_buffer = torch.zeros((total_kv, QK_HEAD_DIM), dtype=dtypes.fp8)
    kv_page_indices = torch.arange(total_kv, dtype=torch.int32)
    work_indptr = torch.tensor([0, batch], dtype=torch.int32)
    work_info_set = torch.tensor(
        [
            [
                batch_idx,
                batch_idx,
                batch_idx,
                batch_idx + 1,
                batch_idx * context_len,
                (batch_idx + 1) * context_len,
                0,
                0,
            ]
            for batch_idx in range(batch)
        ],
        dtype=torch.int32,
    )
    split_output = torch.full(
        (batch * NUM_HEADS, V_HEAD_DIM), float("nan"), dtype=torch.float32
    )
    split_lse = torch.full(
        (batch * NUM_HEADS,), float("nan"), dtype=torch.float32
    )
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)

    return (
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        split_output,
        split_lse,
        softmax_scale,
    )


if __name__=="__main__":
    batch = 32
    ctx_len = 1
    args = _make_inputs(batch, ctx_len)
    launch(*args)