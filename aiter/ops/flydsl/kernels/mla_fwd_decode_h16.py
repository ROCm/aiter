import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.kernels_common import get_warp_size
# TODO: Extend the parameters to include following options:
# flydsl_mla_fwd_decode(
#     query,                # [num_q,H,576] bf16
#     kv_buffer,            # packed physical cache [...,576]
#     page_table,           # flat-token IDs, dense [B,max_blocks], or CSR page IDs
#     pages_kv_indptr,      # [B+1] logical page/token counts for metadata
#     context_lens,         # [B], clamps the partially filled last page
#     work_indptr,
#     work_info_set,        # AITER work ABI; DW0 is batch index
#     final_output,         # [num_q,H,512] bf16
#     split_output,         # [num_partial,1,H,512] f32
#     split_lse,            # [num_partial,1,H,1] f32
#     softmax_scale,        # K3: 192**-0.5 * mscale**2
#     kv_scale=1.0,
#     *,
#     metadata_flavor="pa_v1",      # compile-time: v1 token units | pa_v1 page units
#     page_table_mode="dense_block",# flat_token | dense_block | csr_block
#     page_size=16,                 # compile-time: 1 | 16 | 32 | 64
#     cache_strides=None,           # page/token strides; last dim contiguous
#     cache_layout="packed_576",    # split_512_64 only with a named consumer
#     k_pe_buffer=None,
# ) -> None
BLOCK_THREADS: int = 512
WAVE_SIZE: int = get_warp_size()
NUM_WAVES: int = BLOCK_THREADS // WAVE_SIZE

# Packed MLA head: 512 NOPE + 64 RoPE, last dim contiguous.
NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
QK_NOPE_HEAD_DIM: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM: int = QK_NOPE_HEAD_DIM
PAGE_SIZE: int = 16
KV_COL_BLOCK: int = 64
SIZE_MLA_WORK_INFO_IN_DW: int = 8


@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
def kn_mla_fwd_decode_m16_fp8_fp8_16head(
    query: fx.Tensor,  # [num_q, H, 576] fp8
    kv_buffer: fx.Tensor,  # [num_pages, page_size, 1, 576] fp8
    kv_page_indices: fx.Tensor,  # [num_page_used] i32
    work_indptr: fx.Tensor,  # [num_workers + 1] i32
    work_info_set: fx.Tensor,  # [num_work_items, 8] i32
    final_output: fx.Tensor,  # [num_q, H, 512] bf16
    split_output: fx.Tensor,  # [num_partial, 1, H, 512] f32
    split_lse: fx.Tensor,  # [num_partial, 1, H, 1] f32
    softmax_scale: float,
):
    """Specialized MLA forward decode for heads <= 16."""

    buf_q = fx.rocdl.make_buffer_tensor(query)
    buf_kv = fx.rocdl.make_buffer_tensor(kv_buffer)
    buf_page_idx = fx.rocdl.make_buffer_tensor(kv_page_indices)
    buf_indptr = fx.rocdl.make_buffer_tensor(work_indptr)
    buf_workinfo = fx.rocdl.make_buffer_tensor(work_info_set)
    buf_output = fx.rocdl.make_buffer_tensor(final_output)
    buf_split_output = fx.rocdl.make_buffer_tensor(split_output)
    buf_split_lse = fx.rocdl.make_buffer_tensor(split_lse)


    # Each work items is an 8 x uint32 struct (see mla.h)
    # Each persistent workgroup processes its assigned work-set interval.
    worker_idx = fx.gpu.block_idx.x
    work_start = buf_indptr[worker_idx]
    work_end = buf_indptr[worker_idx + 1]
    for work_idx in range(work_start, work_end):
        work_item = fx.slice(buf_workinfo, (work_idx, None))

        batch_idx = work_item[0]
        partial_qo_loc = work_item[1]
        qo_start = work_item[2]
        qo_end = work_item[3]
        kv_start_page = work_item[4]
        kv_end_page = work_item[5]
        kv_offset = work_item[6]

        print(work_item)

        compute_work_set()


        

    def compute_work_set(work_item):
        """Compute one MlaWorkInfo row.
        """
   
        pass