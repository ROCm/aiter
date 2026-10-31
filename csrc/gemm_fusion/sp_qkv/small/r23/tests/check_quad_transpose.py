#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model check of smg::quad_transpose4 (SM_P_NT): emulates the 4 lanes of a quad with DPP quad_perm 0xB1 (xor1) / 0x4E (xor2) and verifies out[i] on lane j == in[j] on lane i for all inputs."""


def dpp(vals, ctrl):  # vals[lane] -> value of the lane selected by quad_perm
    perm = [(ctrl >> (2 * k)) & 3 for k in range(4)]
    return [vals[perm[k]] for k in range(4)]


def transpose(a):  # a[lane][k]
    p = [
        dpp([a[l][k] for l in range(4)], 0x4E) for k in range(4)
    ]  # p[k][lane] = a[lane^2][k]
    n = [[None] * 4 for _ in range(4)]
    for j in range(4):
        b1 = (j & 2) != 0
        n[j][0] = p[2][j] if b1 else a[j][0]
        n[j][1] = p[3][j] if b1 else a[j][1]
        n[j][2] = a[j][2] if b1 else p[0][j]
        n[j][3] = a[j][3] if b1 else p[1][j]
    q = [
        dpp([n[l][k] for l in range(4)], 0xB1) for k in range(4)
    ]  # q[k][lane] = n[lane^1][k]
    o = [[None] * 4 for _ in range(4)]
    for j in range(4):
        b0 = (j & 1) != 0
        o[j][0] = q[1][j] if b0 else n[j][0]
        o[j][1] = n[j][1] if b0 else q[0][j]
        o[j][2] = q[3][j] if b0 else n[j][2]
        o[j][3] = n[j][3] if b0 else q[2][j]
    return o


a = [[10 * l + k for k in range(4)] for l in range(4)]
o = transpose(a)
ok = all(o[j][i] == a[i][j] for i in range(4) for j in range(4))
print(
    ("PASS" if ok else "FAIL")
    + " quad_transpose4 model: out[i] on lane j == in[j] on lane i (4x4), DPP ctrl 0x4E / 0xB1"
)
raise SystemExit(0 if ok else 1)
