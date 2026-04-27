from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _gluon_rms_norm_kernel(
    input_ptr,
    output_ptr,
    weights_ptr,
    rsigma_ptr,
    n_rows,
    n_cols,
    epsilon,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: gl.constexpr,
    USE_BLOCK: gl.constexpr,
    NUM_PROG: gl.constexpr,
):

    # map program id to row start and column offsets
    row_start = gl.program_id(axis=0)

    # Rank-1D layout: used for col_offsets, masks, zeros, etc.
    col_layout: gl.constexpr = gl.BlockedLayout(
        [8],  # sizePerThread
        [32],  # threadsPerWarp  (warp size = 32 on gfx1250)
        [4],  # warpsPerCTA
        [0],  # order
    )

    # create a swizzled shared layout for the input
    sharedLayoutInput: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # create a swizzled shared layout for the output
    sharedLayoutOutput: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # create a swizzled shared layout for the weights
    sharedLayoutWeights: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    # create a swizzled shared layout for the output
    gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # tensor descriptor for input
    input_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        input_ptr,  # base pointer of tesnor
        [n_rows, n_cols],  # shape of tensor
        [input_row_stride, 1],  # strides of tensor
        [1, BLOCK_SIZE],  # block shape of tensor
        sharedLayoutInput,  # layout of tensor
    )

    # tensor descript for weights
    weights_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        weights_ptr,  # base pointer of tensor
        [n_cols],  # shape of tensor
        [1],  # strides of tensor
        [BLOCK_SIZE],  # block shape of tensor
        sharedLayoutWeights,  # layout of tensor
    )

    # tensor descriptor for output
    output_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        output_ptr,  # base pointer of tensor
        [n_rows, n_cols],  # shape of tensor
        [output_row_stride, 1],  # strides of tensor
        [1, BLOCK_SIZE],  # block shape of tensor
        sharedLayoutOutput,  # layout of tensor
    )

    nStages: gl.constexpr = 2
    smemInput = gl.allocate_shared_memory(
        input_ptr.dtype.element_ty, [nStages, 1, BLOCK_SIZE], sharedLayoutInput
    )
    smemOutput = gl.allocate_shared_memory(
        output_ptr.dtype.element_ty, [nStages, 1, BLOCK_SIZE], sharedLayoutOutput
    )

    if USE_BLOCK:
        # 2-stage software pipeline
        smemWeights = gl.allocate_shared_memory(
            weights_ptr.dtype.element_ty, [nStages, BLOCK_SIZE], sharedLayoutWeights
        )

        # one time creation for handling the last block if n_cols is not a multiple of BLOCK_SIZE
        n_col_blk = gl.cdiv(n_cols, BLOCK_SIZE) - 1
        last_col_idx = n_col_blk * BLOCK_SIZE

        # Loop through the rows of the input tensor by NUM_PROG blocks
        for row_idx in range(row_start, n_rows, NUM_PROG):
            # sum_sq store
            sum_sq = 0.0
            # preload the inputs and weights
            gl.amd.gfx1250.tdm.async_load(input_desc, [row_idx, 0], smemInput.index(0))
            gl.amd.gfx1250.tdm.async_load(weights_desc, [0], smemWeights.index(0))

            # secondary loop to accumulate the sum square across all the cols
            # calculate the number of whole blocks for column dimension
            for blk_idx in range(0, n_col_blk - 1):
                preload_idx = blk_idx % 2
                load_idx = 1 - preload_idx

                # load input_ptr, compute square of input, store in sum_sq
                gl.amd.gfx1250.tdm.async_load(
                    input_desc,
                    [row_idx, (blk_idx + 1) * BLOCK_SIZE],
                    smemInput.index(load_idx),
                )
                gl.amd.gfx1250.tdm.async_wait(1)
                smemInput_1d = smemInput.index(preload_idx).reshape([BLOCK_SIZE])
                a = smemInput_1d.load(col_layout).to(gl.float32)
                sum_sq += gl.sum(a * a, axis=0)

            # handle the last block if n_cols is not a multiple of BLOCK_SIZE
            last_block_idx = (n_col_blk - 1) % 2
            gl.amd.gfx1250.tdm.async_wait(0)
            smemInput_1d = smemInput.index(last_block_idx).reshape([BLOCK_SIZE])
            a = smemInput_1d.load(col_layout).to(gl.float32)
            sum_sq += gl.sum(a * a, axis=0)

            # handles last condition of the second stage software pipeline
            last_idx = 1 - last_block_idx
            gl.amd.gfx1250.tdm.async_load(
                input_desc, [row_idx, last_col_idx], smemInput.index(last_idx)
            )
            gl.amd.gfx1250.tdm.async_wait(0)
            smemInput_1d = smemInput.index(last_idx).reshape([BLOCK_SIZE])
            a = smemInput_1d.load(col_layout).to(gl.float32)
            sum_sq += gl.sum(a * a, axis=0)

            # compute norm factor
            norm_factor = gl.rsqrt((sum_sq / n_cols) + epsilon)

            # reload the inputs and weights from the beginning of the 1st stage
            gl.amd.gfx1250.tdm.async_load(input_desc, [row_idx, 0], smemInput.index(0))
            gl.amd.gfx1250.tdm.async_load(weights_desc, [0], smemWeights.index(0))

            # loop through the columns to normalize the output
            for blk_idx in range(0, n_col_blk - 1):
                preload_idx = blk_idx % 2
                load_idx = 1 - preload_idx
                # load input_ptr, compute square of input, store in sum_sq
                gl.amd.gfx1250.tdm.async_load(
                    input_desc,
                    [row_idx, (blk_idx + 1) * BLOCK_SIZE],
                    smemInput.index(load_idx),
                )
                gl.amd.gfx1250.tdm.async_load(
                    weights_desc,
                    [(blk_idx + 1) * BLOCK_SIZE],
                    smemWeights.index(load_idx),
                )
                gl.amd.gfx1250.tdm.async_wait(1)
                smemInput_1d = smemInput.index(preload_idx).reshape([BLOCK_SIZE])
                a = smemInput_1d.load(col_layout).to(gl.float32)
                weights = smemWeights.index(preload_idx).load(col_layout).to(gl.float32)
                rms_norm = a * weights * norm_factor
                # store rms_norm than use async_store
                smemOutput1d = smemOutput.index(preload_idx).reshape([BLOCK_SIZE])
                smemOutput1d.store(rms_norm.to(output_ptr.dtype.element_ty))
                gl.amd.gfx1250.tdm.async_store(
                    output_desc,
                    [row_idx, blk_idx * BLOCK_SIZE],
                    smemOutput.index(preload_idx),
                )

            # handle the last block if n_cols is not a multiple of BLOCK_SIZE
            last_block_idx = (n_col_blk - 1) % 2
            gl.amd.gfx1250.tdm.async_wait(0)
            smemInput_1d = smemInput.index(last_block_idx).reshape([BLOCK_SIZE])
            a = smemInput_1d.load(col_layout).to(gl.float32)
            weights = smemWeights.index(last_block_idx).load(col_layout).to(gl.float32)
            rms_norm = a * weights * norm_factor
            # store rms_norm then use async_store
            smemOutput1d = smemOutput.index(last_block_idx).reshape([BLOCK_SIZE])
            smemOutput1d.store(rms_norm.to(output_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(
                output_desc,
                [row_idx, (n_col_blk - 1) * BLOCK_SIZE],
                smemOutput.index(last_block_idx),
            )

            # handle the last condition of the second stage software pipeline
            last_idx = 1 - last_block_idx
            gl.amd.gfx1250.tdm.async_load(
                input_desc, [row_idx, last_col_idx], smemInput.index(last_idx)
            )
            gl.amd.gfx1250.tdm.async_load(
                weights_desc, [last_col_idx], smemWeights.index(last_idx)
            )
            gl.amd.gfx1250.tdm.async_wait(0)
            smemInput_1d = smemInput.index(last_idx).reshape([BLOCK_SIZE])
            a = smemInput_1d.load(col_layout).to(gl.float32)
            weights = smemWeights.index(last_idx).load(col_layout).to(gl.float32)
            rms_norm = a * weights * norm_factor
            smemOutput1d = smemOutput.index(last_idx).reshape([BLOCK_SIZE])
            smemOutput1d.store(rms_norm.to(output_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(
                output_desc, [row_idx, last_col_idx], smemOutput.index(last_idx)
            )
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.store(rsigma_ptr + row_idx, norm_factor.to(rsigma_ptr.dtype.element_ty))

    else:  # no blocking
        smemWeights = gl.allocate_shared_memory(
            weights_ptr.dtype.element_ty, [BLOCK_SIZE], sharedLayoutWeights
        )
        # preload the weights as they all fit within the shared memory
        gl.amd.gfx1250.tdm.async_load(weights_desc, [0], smemWeights)
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.amd.gfx1250.tdm.async_load(input_desc, [row_start, 0], smemInput.index(0))

        for row_idx in range(row_start, n_rows, NUM_PROG):
            # determine the current and next stage
            current_stage = ((row_idx - row_start) // NUM_PROG) % 2
            next_stage = 1 - current_stage
            ##acquire next incoming row
            gl.amd.gfx1250.tdm.async_load(
                input_desc, [row_idx + NUM_PROG, 0], smemInput.index(next_stage)
            )
            gl.amd.gfx1250.tdm.async_wait(1)

            smemInput_1d = smemInput.index(current_stage).reshape([BLOCK_SIZE])
            a = smemInput_1d.load(col_layout).to(gl.float32)
            weights = smemWeights.load(col_layout).to(gl.float32)
            # compute the square of the input
            sum_sq = a * a
            # compute the sum of the square of the input
            row_sq_sum = gl.sum(sum_sq, axis=0)
            # compute the norm factor
            norm_factor = gl.rsqrt((row_sq_sum / n_cols) + epsilon)

            # compute the rms norm
            rms_norm = a * norm_factor * weights
            # store rms norm and the norm factor
            gl.store(rsigma_ptr + row_idx, norm_factor.to(rsigma_ptr.dtype.element_ty))
            smemOutput1d = smemOutput.index(current_stage).reshape([BLOCK_SIZE])
            smemOutput1d.store(rms_norm.to(output_ptr.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(
                output_desc, [row_idx, 0], smemOutput.index(current_stage)
            )
            gl.amd.gfx1250.tdm.async_wait(0)
