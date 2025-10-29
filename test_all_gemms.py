shapes = """8192,57344,8192
16384,57344,8192
32768,57344,8192
8192,10240,16384
16384,10240,16384
32768,10240,16384
2048,57344,16384
4096,57344,16384
8192,57344,16384
16384,57344,16384
32768,57344,16384
4096,10240,16384
8192,8192,8192
16384,8192,8192
32768,8192,8192
16384,16384,16384
"""

from subprocess import check_call, check_output
shapes = shapes.strip().split('\n')
for shape in shapes:
    shape = shape.strip()
    cmd = f"python op_tests/test_gemm_a4w4.py -mnk {shape}"
    check_call(cmd.split(' '))
