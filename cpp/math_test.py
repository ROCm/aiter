from ctypes import *
import torch


lib = CDLL('./math_test.so')
a = torch.tensor(0, dtype=torch.int32)
res = cast(a.data_ptr(), POINTER(c_int))
# res = c_int(res)
# lib.call.argtypes = [c_int, c_int, POINTER(c_int)] 
lib.call(1, 2, res)
print(a)