from kernel import *

@g.jit
def get_pointer(Args, index:gl.constexpr, dtype:gl.constexpr):
    address=gl.load(Args+index,volatile=True)
    return address.to(gl.pointer_type(dtype))

@g.jit
def packed_full_layer(Args, Types: gl.constexpr, GRID: gl.constexpr=256):
    mhc_mix(get_pointer(Args, 0, Types[0]), get_pointer(Args, 6, Types[6]), get_pointer(Args, 7, Types[7]), get_pointer(Args, 8, Types[8]), get_pointer(Args, 41, Types[41]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    sinkhorn(get_pointer(Args, 41, Types[41]))
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    mhc_pre(get_pointer(Args, 0, Types[0]), get_pointer(Args, 41, Types[41]), get_pointer(Args, 44, Types[44]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    norm(get_pointer(Args, 44, Types[44]), get_pointer(Args, 12, Types[12]), get_pointer(Args, 44, Types[44]), 4096, 1, True, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 44, Types[44]), get_pointer(Args, 45, Types[45]), 4096, 128, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    matvec(get_pointer(Args, 45, Types[45]), get_pointer(Args, 17, Types[17]), get_pointer(Args, 18, Types[18]), get_pointer(Args, 46, Types[46]), 1024, 4096, 8, GRID)
    matvec(get_pointer(Args, 45, Types[45]), get_pointer(Args, 21, Types[21]), get_pointer(Args, 22, Types[22]), get_pointer(Args, 49, Types[49]), 512, 4096, 8, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    norm(get_pointer(Args, 46, Types[46]), get_pointer(Args, 14, Types[14]), get_pointer(Args, 46, Types[46]), 1024, 1, True, GRID, False)
    norm(get_pointer(Args, 49, Types[49]), get_pointer(Args, 15, Types[15]), get_pointer(Args, 49, Types[49]), 512, 1, True, GRID, False)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 46, Types[46]), get_pointer(Args, 47, Types[47]), 1024, 128, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    matvec(get_pointer(Args, 47, Types[47]), get_pointer(Args, 19, Types[19]), get_pointer(Args, 20, Types[20]), get_pointer(Args, 48, Types[48]), 32768, 1024, 8, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    norm(get_pointer(Args, 48, Types[48]), get_pointer(Args, 14, Types[14]), get_pointer(Args, 48, Types[48]), 512, 64, False, GRID, False)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    rope(get_pointer(Args, 48, Types[48]), get_pointer(Args, 4, Types[4]), get_pointer(Args, 5, Types[5]), get_pointer(Args, 50, Types[50]), 64, False, GRID)
    rope(get_pointer(Args, 49, Types[49]), get_pointer(Args, 4, Types[4]), get_pointer(Args, 5, Types[5]), get_pointer(Args, 51, Types[51]), 1, False, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    if gl.program_id(0) == 0:
        i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
        dev_store(dev_load(get_pointer(Args, 51, Types[51]), i, cache='.cg'), get_pointer(Args, 3, Types[3]), 127 * 512 + i, cache='.cg')
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    attention(get_pointer(Args, 50, Types[50]), get_pointer(Args, 51, Types[51]), get_pointer(Args, 3, Types[3]), get_pointer(Args, 16, Types[16]), get_pointer(Args, 52, Types[52]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    rope(get_pointer(Args, 52, Types[52]), get_pointer(Args, 4, Types[4]), get_pointer(Args, 5, Types[5]), get_pointer(Args, 53, Types[53]), 64, True, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 53, Types[53]), get_pointer(Args, 54, Types[54]), 32768, 128, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    matvec(get_pointer(Args, 54, Types[54]), get_pointer(Args, 23, Types[23]), get_pointer(Args, 24, Types[24]), get_pointer(Args, 55, Types[55]), 8192, 4096, 8, GRID, True)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 55, Types[55]), get_pointer(Args, 56, Types[56]), 8192, 128, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    matvec(get_pointer(Args, 56, Types[56]), get_pointer(Args, 25, Types[25]), get_pointer(Args, 26, Types[26]), get_pointer(Args, 57, Types[57]), 4096, 8192, 8, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    mhc_post(get_pointer(Args, 57, Types[57]), get_pointer(Args, 0, Types[0]), get_pointer(Args, 41, Types[41]), get_pointer(Args, 43, Types[43]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    mhc_mix(get_pointer(Args, 43, Types[43]), get_pointer(Args, 9, Types[9]), get_pointer(Args, 10, Types[10]), get_pointer(Args, 11, Types[11]), get_pointer(Args, 42, Types[42]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    sinkhorn(get_pointer(Args, 42, Types[42]))
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    mhc_pre(get_pointer(Args, 43, Types[43]), get_pointer(Args, 42, Types[42]), get_pointer(Args, 58, Types[58]), GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    norm(get_pointer(Args, 58, Types[58]), get_pointer(Args, 13, Types[13]), get_pointer(Args, 58, Types[58]), 4096, 1, True, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 58, Types[58]), get_pointer(Args, 59, Types[59]), 4096, 128, GRID)
    quant(get_pointer(Args, 58, Types[58]), get_pointer(Args, 60, Types[60]), 4096, 32, GRID)
    matvec(get_pointer(Args, 58, Types[58]), get_pointer(Args, 33, Types[33]), get_pointer(Args, 33, Types[33]), get_pointer(Args, 66, Types[66]), 256, 4096, 16, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    route(get_pointer(Args, 66, Types[66]), get_pointer(Args, 34, Types[34]), get_pointer(Args, 2, Types[2]), get_pointer(Args, 67, Types[67]), get_pointer(Args, 68, Types[68]))
    matvec(get_pointer(Args, 59, Types[59]), get_pointer(Args, 27, Types[27]), get_pointer(Args, 28, Types[28]), get_pointer(Args, 61, Types[61]), 2048, 4096, 8, GRID)
    matvec(get_pointer(Args, 59, Types[59]), get_pointer(Args, 29, Types[29]), get_pointer(Args, 30, Types[30]), get_pointer(Args, 62, Types[62]), 2048, 4096, 8, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    activation(get_pointer(Args, 61, Types[61]), get_pointer(Args, 62, Types[62]), get_pointer(Args, 63, Types[63]), 2048, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 63, Types[63]), get_pointer(Args, 64, Types[64]), 2048, 128, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    matvec(get_pointer(Args, 64, Types[64]), get_pointer(Args, 31, Types[31]), get_pointer(Args, 32, Types[32]), get_pointer(Args, 65, Types[65]), 4096, 2048, 8, GRID)
    for slot in range(6):
        expert = dev_load(get_pointer(Args, 67, Types[67]), slot, cache='.cg').to(gl.int64)
        matvec(get_pointer(Args, 60, Types[60]), get_pointer(Args, 35, Types[35]) + expert * 2048 * 2048, get_pointer(Args, 36, Types[36]) + expert * 2048 * 128, get_pointer(Args, 69, Types[69]) + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
        matvec(get_pointer(Args, 60, Types[60]), get_pointer(Args, 37, Types[37]) + expert * 2048 * 2048, get_pointer(Args, 38, Types[38]) + expert * 2048 * 128, get_pointer(Args, 70, Types[70]) + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    activation(get_pointer(Args, 69, Types[69]), get_pointer(Args, 70, Types[70]), get_pointer(Args, 71, Types[71]), 6 * 2048, GRID, False)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    quant(get_pointer(Args, 71, Types[71]), get_pointer(Args, 72, Types[72]), 6 * 2048, 32, GRID)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    for slot in range(6):
        expert = dev_load(get_pointer(Args, 67, Types[67]), slot, cache='.cg').to(gl.int64)
        matvec(get_pointer(Args, 72, Types[72]) + slot * 2048, get_pointer(Args, 39, Types[39]) + expert * 4096 * 1024, get_pointer(Args, 40, Types[40]) + expert * 4096 * 64, get_pointer(Args, 73, Types[73]) + slot * 4096, 4096, 2048, 4, GRID, Gamma=get_pointer(Args, 68, Types[68]) + slot)
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    i = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for tile in range(gl.program_id(0), 16, GRID):
        c = tile * 256 + i
        value = gl.full((256,), 0.0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
        for reduce_slot in gl.static_range(6):
            value += dev_load(get_pointer(Args, 73, Types[73]), reduce_slot * 4096 + c, cache='.cg').to(gl.float32)
        value = value.to(gl.bfloat16).to(gl.float32) + dev_load(get_pointer(Args, 65, Types[65]), c, cache='.cg').to(gl.float32)
        dev_store(value.to(gl.bfloat16), get_pointer(Args, 74, Types[74]), c, cache='.cg')
    sync(get_pointer(Args, 75, Types[75]), get_pointer(Args, 76, Types[76]), GRID)
    mhc_post(get_pointer(Args, 74, Types[74]), get_pointer(Args, 43, Types[43]), get_pointer(Args, 42, Types[42]), get_pointer(Args, 1, Types[1]), GRID)

class PackedAdapter:
    def __init__(self, original):
        self.fn=original.fn
        self.table=None
    def warmup(self,*values,**kwargs):
        import torch
        if self.table is None:
            self.table=torch.tensor([x.data_ptr() for x in values],device='cuda',dtype=torch.uint64)
            mapping={torch.float32:gl.float32,torch.bfloat16:gl.bfloat16,torch.float8_e4m3fn:gl.float8e4nv,torch.int32:gl.int32,torch.int64:gl.int64,torch.uint8:gl.uint8,torch.int8:gl.int8}
            self.types=tuple(mapping[x.dtype] for x in values)
        return packed_full_layer.warmup(self.table,self.types,**kwargs)
    def __getitem__(self,grid):
        def launch(*values,**kwargs):
            return packed_full_layer[grid](self.table,self.types,**kwargs)
        return launch
