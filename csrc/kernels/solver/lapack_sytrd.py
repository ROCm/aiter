import numpy as np
from scipy.linalg.lapack import dsytrd
import scipy.linalg.blas as blas


def _copy_lower_to(A_src, A_dst):
    m, n = A_src.shape
    r = min(m, n)
    idx_i, idx_j = np.tril_indices(r, 0)
    A_dst[idx_i, idx_j] = A_src[idx_i, idx_j]


def symv_lower(A_sub, x):
    y = blas.dsymv(alpha=1.0, a=A_sub, x=x, lower=1, overwrite_y=0)
    return y


def syr2_inplace_lower(A_sub, alpha, x, y):
    C = A_sub.copy(order="F")
    C = blas.dsyr2(alpha=alpha, x=x, y=y, a=C, lower=1, overwrite_a=1)
    _copy_lower_to(C, A_sub)


def syr2k_inplace_lower(A_sub, alpha, V, W, beta=1.0):
    C = A_sub.copy(order="F")
    C = blas.dsyr2k(
        alpha=alpha, a=V, b=W, c=C, lower=1, trans=0, beta=beta, overwrite_c=1
    )
    _copy_lower_to(C, A_sub)


def larfg(alpha, x_tail):
    sigma = np.linalg.norm(x_tail)
    if sigma == 0.0:
        return alpha, 0.0, x_tail
    r = np.hypot(alpha, sigma)
    beta = -np.copysign(r, alpha if alpha != 0 else 1.0)
    tau = (beta - alpha) / beta
    x_tail /= alpha - beta
    return beta, tau, x_tail


def sytd2_lower(A):
    n = A.shape[0]
    E = np.zeros(n - 1, dtype=A.dtype, order="F")
    TAU = np.zeros(n - 1, dtype=A.dtype, order="F")

    for i in range(n - 2):
        beta, tau, _ = larfg(A[i + 1, i], A[i + 2 :, i])
        E[i] = beta
        TAU[i] = tau
        A[i + 1, i] = 1.0

        if tau != 0.0:
            v = A[i + 1 :, i]
            A22 = A[i + 1 :, i + 1 :]
            w = symv_lower(A22, v)
            alpha_corr = -0.5 * tau * np.dot(w, v)
            w = w + alpha_corr * v
            syr2_inplace_lower(A22, -tau, v, w)

    D = np.diag(A).copy(order="F")
    for i in range(n - 2):
        A[i + 1, i] = E[i]
    return D, E, TAU


def latrd_lower_panel(A_panel, nb):
    Np = A_panel.shape[0]
    nb = min(nb, max(0, Np - 1))

    E = np.zeros(nb, dtype=A_panel.dtype, order="F")
    TAU = np.zeros(nb, dtype=A_panel.dtype, order="F")
    W = np.zeros((Np, nb), dtype=A_panel.dtype, order="F")

    for i in range(nb):
        if i > 0:
            A_panel[i:, i] -= (
                A_panel[i:, :i] @ W[i, :i].T + W[i:, :i] @ A_panel[i, :i].T
            )

        beta, tau, _ = larfg(A_panel[i + 1, i], A_panel[i + 2 :, i])
        E[i] = beta
        TAU[i] = tau
        A_panel[i + 1, i] = 1.0

        if tau != 0.0:
            v = A_panel[i + 1 :, i]
            A22 = A_panel[i + 1 :, i + 1 :]
            w = symv_lower(A22, v)

            if i > 0:
                w -= A_panel[i + 1 :, :i] @ (W[i + 1 :, :i].T @ v) + W[i + 1 :, :i] @ (
                    A_panel[i + 1 :, :i].T @ v
                )

            w *= tau
            alpha_corr = -0.5 * tau * np.dot(w, v)
            w = w + alpha_corr * v
            W[i + 1 :, i] = w
        else:
            W[i + 1 :, i] = 0.0

    return E, TAU, W


def sytrd_lower(A, nb=32, nx=16):
    n = A.shape[0]
    D = np.zeros(n, dtype=A.dtype, order="F")
    E = (
        np.zeros(n - 1, dtype=A.dtype, order="F")
        if n >= 2
        else np.zeros(0, dtype=A.dtype, order="F")
    )
    TAU = (
        np.zeros(n - 1, dtype=A.dtype, order="F")
        if n >= 2
        else np.zeros(0, dtype=A.dtype, order="F")
    )

    j0 = 0
    while j0 < n - 1 and (n - j0) > nx:
        jb = min(nb, n - j0 - 1)
        if jb <= 0:
            break

        A_panel = A[j0:, j0:]
        E_panel, TAU_panel, W = latrd_lower_panel(A_panel, jb)
        E[j0 : j0 + jb] = E_panel
        TAU[j0 : j0 + jb] = TAU_panel

        if j0 + jb < n:
            V2 = A[j0 + jb :, j0 : j0 + jb]
            W2 = W[jb:, :jb]
            A22 = A[j0 + jb :, j0 + jb :]
            syr2k_inplace_lower(A22, alpha=-1.0, V=V2, W=W2, beta=1.0)

        D[j0 : j0 + jb] = np.diag(A)[j0 : j0 + jb]
        for i in range(jb):
            A[j0 + i + 1, j0 + i] = E[j0 + i]
        j0 += jb

    if j0 < n - 1:
        D_tail, E_tail, TAU_tail = sytd2_lower(A[j0:, j0:])
        D[j0:] = D_tail
        E[j0 : n - 1] = E_tail
        TAU[j0 : n - 1] = TAU_tail

    D[-1] = A[-1, -1]
    E[-1] = A[-1, -2]
    return D, E, TAU


def symmetric_random(n, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array(rng.standard_normal((n, n)), order="F")
    A = (A + A.T) / 2.0
    return A


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for n in [2, 4, 8, 16, 32, 64]:
        A = symmetric_random(n, seed=42)
        A_work = A.copy(order="F")
        D, E, TAU = sytrd_lower(A_work, nb=32, nx=16)

        A_lapack, D_lapack, E_lapack, tau_lapack, info = dsytrd(
            A.copy(order="F"), lower=True
        )
        if info == 0:
            diff_A = np.max(np.abs(A_work - A_lapack))
            diff_D = np.max(np.abs(D - D_lapack))
            diff_E = np.max(np.abs(E - E_lapack))
            diff_TAU = np.max(np.abs(TAU - tau_lapack))
            threshold = 1e-6
            if max(diff_A, diff_D, diff_E, diff_TAU) < threshold:
                print(f"n = {n:3d}   Pass")
            else:
                print(f"n = {n:3d}   Fail")
        else:
            print(f"   dsytrd failed info={info}")
