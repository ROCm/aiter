# python3 shared-rms-mixed.py

import time
import torch
import rms

input_tensors = []
output_tensors = []
weight_tensors = []
rsigma_tensors = []

ngpu=8
rows=64
cols=16384

shape=(rows,cols)
shape2=(rows)
type=torch.bfloat16

for i in range(ngpu):
    device = torch.device(f"cuda:{i}")
    input_tensor = torch.ones(shape, dtype=type, device=device)
    input_tensors.append(input_tensor)
    output_tensor = torch.zeros(shape, dtype=type, device=device)
    output_tensors.append(output_tensor)
    weight_tensor = torch.ones(cols, dtype=torch.bfloat16).cuda(i)
    weight_tensor2 = torch.mul(weight_tensor, 5)
    weight_tensors.append(weight_tensor2)
    rsigma_tensor = torch.zeros(shape2, dtype=type, device=device)
    rsigma_tensors.append(rsigma_tensor)

#print(input_tensors[0].numel())
epsilon = 17.0
exec = rms.create_executor("bf16")
#exec.initialize(ngpu, rows, cols, cols, cols, epsilon)
#print(input_tensors[0].size())
#print(input_tensors[0][0][0])
print(output_tensors[0][0][0])
exec.compute_with_pytorch_tensors(input_tensors, output_tensors,
        weight_tensors, rsigma_tensors, epsilon)

#print(f'input_data = ', input_tensors[7][4000][0])
#print(f'output_data = ', output_tensors[7][4000][0])
print(f'output_data = ', output_tensors[7][60][0])
print(f'risgma = ', rsigma_tensors[0][0])
#print(f'rsigma = ', rsigma_tensors[6][4095])

exec.release()

output_tensors = []
rsigma_tensors = []
output_type=torch.float8_e4m3fn
output_shape=(rows)
for i in range(ngpu):
    device = torch.device(f"cuda:{i}")
    output_tensor = torch.zeros(shape, dtype=output_type, device=device)
    output_tensors.append(output_tensor)
#    rsigma_tensor = torch.zeros(rows, dtype=output_type, device=device)
#    rsigma_tensors.append(rsigma_tensor)


epsilon = 17.0
exec_mixed = rms.create_executor("bf16", "fp8")
#exec_mixed.initialize(ngpu, rows, cols, cols, cols, epsilon)
print(output_tensors[0][0][0])
start_time = time.time()

for i in range(1000):
    exec_mixed.compute_with_pytorch_tensors(input_tensors, output_tensors,
        weight_tensors, rsigma_tensors, epsilon)
reusable_time = time.time() - start_time

print(f"Reusable executor: {reusable_time: .3f}s")

#print(f'input_data = ', input_tensors[7][4000][0])
#print(f'output_data = ', output_tensors[7][4095][7167])
#print(f'risgma = ', rsigma_tensors[0][0])
#print(f'rsigma = ', rsigma_tensors[6][4095])


#print(input_tensors[7][4000][0])
#print(output_tensors[7][4000][0])
print(output_tensors[7][40][0])
#print(output_tensors[0][0][0])
#print(rsigma_tensors[0][0])
#print(rsigma_tensors[6][7])
 
