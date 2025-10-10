import triton
import triton.language as tl


@triton.jit
def _softmax_kernel_online(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
): 
    
    row_id = tl.program_id(0) # program(instruction) row index, one program per row along the "0 axis"
    
    row_start_ptr = input_ptr + row_id * input_row_stride #find the ptr address of the start of the row

    row_end_ptr = output_ptr + row_id * output_row_stride # find ptr of the end of the row

    block_offset = tl.arange(0, BLOCK_SIZE) # size of block that threads will be working on

    #for the max so far at that iteration
    running_max = -float("inf")
    #for how much we need to rescale
    running_sum = 0
    
    #for i in tl.range(0, n_cols, BLOCK_SIZE):
    col_start = 0

    # max calc and denominator
    while col_start < n_cols:

        col = col_start + block_offset # getting the actual indicies of each of the elements in the tile

        valid = col < n_cols # used to ensure that the index is valid or needs to be masked (incase block has more threads than tile)

        tile_load = tl.load(row_start_ptr + col, mask = valid, other = -float("inf")) #loading tile of the row in, masking invalid or non existent areas
        
        tile_load_32 = tile_load.to(tl.float32) # incase of overflow since 16 may overflow


        max_block = tl.max(tile_load,axis = 0) #getting the max of each block

        new_max = tl.maximum(max_block, running_max) #placeholder for the new max so we can set the running max to at end of loop

        running_max = new_max

        rescale = tl.exp(running_max - new_max) # getting rescale value

        tile_centered = tile_load - new_max # kinda normalizing the tile

        exp_block = tl.exp(tile_centered) # exponentiating the values after max was subtracted

        sum_block = tl.sum(exp_block, axis = 0) #summing the block again

        running_sum = running_sum * sum_block + rescale #rescaling the old values

        col_start = col_start + BLOCK_SIZE # go to next tile
    

    col_start = 0
    inv_denom = 1.0 / running_sum #compute the value to normalize

    while col_start < n_cols:
        
        
        col = col_start + block_offset  #get the indicies again

        valid = col < n_cols #masking again

        tile_load = tl.load(row_start_ptr + col, mask=valid, other=-float("inf")) #loading in the tile, masking anything that needs to be masked

        tile_centered = tile_load - running_max #kinda normalizing it again to prevent large values

        exp_val = tl.exp(tile_centered) # exponentiate the smaller values

        n = exp_val * inv_denom #normalize actually by the denonminator by multiplying by precomputed inverse value

        ntype = n.to(tile_load.dtype) # go back to original data type or fp16

        tl.store(row_end_ptr + col, ntype, mask = valid) #storing the values

        col_start = col_start + BLOCK_SIZE #next tile
        

















    



    
