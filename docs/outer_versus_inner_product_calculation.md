# Mathematical Equivalence in Matrix Multiplication: Inner vs. Outer Product

Matrix multiplication (GEMM) can be computed using different loop orderings.
While the traditional approach calculates one scalar cell at a time, the
outer-product approach calculates an entire grid layer at a time. Because
addition is commutative and associative, both methods yield identical results.

Below is the general formulation and a proof of equivalence for matrices $A$
and $B$ of **arbitrary** (compatible) dimensions.

## Setup

Let $A$ be $M \times K$ and $B$ be $K \times N$. Their product $C = AB$ is
$M \times N$. The shared **inner dimension** $K$ is the one that gets summed
away (the contraction).

$$
A=\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1K}\\
a_{21} & a_{22} & \cdots & a_{2K}\\
\vdots & \vdots & \ddots & \vdots\\
a_{M1} & a_{M2} & \cdots & a_{MK}
\end{bmatrix},
\qquad
B=\begin{bmatrix}
b_{11} & b_{12} & \cdots & b_{1N}\\
b_{21} & b_{22} & \cdots & b_{2N}\\
\vdots & \vdots & \ddots & \vdots\\
b_{K1} & b_{K2} & \cdots & b_{KN}
\end{bmatrix}
$$

Regardless of strategy, every output element is defined by the same sum:

$$
C_{ij}=\sum_{k=1}^{K} a_{ik}\,b_{kj},
\qquad i = 1,\ldots,M;\quad j = 1,\ldots,N.
$$

The two strategies differ only in **which index is on the outside of the
loop** — i.e. the order in which the terms of this sum are produced.

---

## 1. The Traditional Approach (Inner Product / Dot Product)

The traditional approach focuses on **one cell of $C$ at a time**. To find an
individual cell $C_{ij}$, you take row $i$ of matrix $A$ and column $j$ of
matrix $B$ and compute their dot product over the inner dimension $K$:

$$
C_{ij}
= \underbrace{\begin{bmatrix} a_{i1} & a_{i2} & \cdots & a_{iK}\end{bmatrix}}_{\text{row } i \text{ of } A}
\;\cdot\;
\underbrace{\begin{bmatrix} b_{1j}\\ b_{2j}\\ \vdots\\ b_{Kj}\end{bmatrix}}_{\text{column } j \text{ of } B}
= a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots + a_{iK}b_{Kj}
= \sum_{k=1}^{K} a_{ik}\,b_{kj}.
$$

### Loop structure

The output positions $(i, j)$ are the **outer** loops; the contraction over $k$
is the **inner** loop:

```python
for i in range(M):
    for j in range(N):
        C[i, j] = 0
        for k in range(K):               # the inner product / dot product
            C[i, j] += A[i, k] * B[k, j]
        # C[i, j] is now COMPLETE
```

### Element lifecycle

Each $C_{ij}$ accumulates all $K$ of its terms back-to-back, then is written
**once, fully formed**, and never touched again. To produce the full output you
repeat this dot product $M \times N$ times.

---

## 2. The Outer-Product Approach

The outer-product approach loops through the inner dimension $K$ on the
**outside**. Instead of finishing one cell at a time, it calculates a full
$M \times N$ **matrix layer** at each step $k$ and accumulates these layers.

At step $k$, take **column $k$ of $A$** (length $M$) and **row $k$ of $B$**
(length $N$) and multiply them to form a rank-1 layer $\mathrm{Grid}_{k}$:

$$
\mathrm{Grid}_{k}
=
\underbrace{\begin{bmatrix} a_{1k}\\ a_{2k}\\ \vdots\\ a_{Mk}\end{bmatrix}}_{\text{column } k \text{ of } A}
\otimes
\underbrace{\begin{bmatrix} b_{k1} & b_{k2} & \cdots & b_{kN}\end{bmatrix}}_{\text{row } k \text{ of } B}
=
\begin{bmatrix}
a_{1k}b_{k1} & a_{1k}b_{k2} & \cdots & a_{1k}b_{kN}\\
a_{2k}b_{k1} & a_{2k}b_{k2} & \cdots & a_{2k}b_{kN}\\
\vdots & \vdots & \ddots & \vdots\\
a_{Mk}b_{k1} & a_{Mk}b_{k2} & \cdots & a_{Mk}b_{kN}
\end{bmatrix}.
$$

The $(i, j)$ entry of this single layer is
$\left(\mathrm{Grid}_{k}\right)_{ij} = a_{ik}\,b_{kj}$ — exactly **one term** of
the sum that defines $C_{ij}$.

### Accumulate the layers

Summing all $K$ independent layers forms the final matrix $C$:

$$
C=\sum_{k=1}^{K}\mathrm{Grid}_{k}
=\sum_{k=1}^{K}\bigl(A_{:,k}\otimes B_{k,:}\bigr).
$$

### Loop structure

The contraction $k$ is now the **outermost** loop; the full output grid
$(i, j)$ is updated inside it:

```python
C[:, :] = 0
for k in range(K):                       # contraction is OUTERmost
    for i in range(M):
        for j in range(N):
            C[i, j] += A[i, k] * B[k, j]  # add this layer's term
# C is complete only after the final k
```

### Element lifecycle

Each $C_{ij}$ is touched $K$ times — it receives one term per layer and is a
**running partial sum** that is only correct after the last step $k = K$. Every
element advances in lockstep.

---

## Proof of Equivalence

Take any fixed output position $(i, j)$ and read off its value from each method.

**Inner product.** The dot product for that cell is, directly,

$$
C_{ij}^{\text{(inner)}} = \sum_{k=1}^{K} a_{ik}\,b_{kj}.
$$

**Outer product.** Cell $(i, j)$ collects the $(i, j)$ entry of every layer:

$$
C_{ij}^{\text{(outer)}}
= \sum_{k=1}^{K} \left(\mathrm{Grid}_{k}\right)_{ij}
= \sum_{k=1}^{K} a_{ik}\,b_{kj}.
$$

The two expressions are the **same sum**; only the order in which its $K$ terms
are generated and added differs. Since real-number addition is commutative and
associative, the order is irrelevant:

$$
C_{ij}^{\text{(inner)}} = C_{ij}^{\text{(outer)}} \quad\text{for all } i, j
\;\;\Longrightarrow\;\; C^{\text{(inner)}} = C^{\text{(outer)}}. \qquad\blacksquare
$$

---

## Worked Example (arbitrary, non-square dimensions)

Let $M = 3,\; K = 2,\; N = 4$:

$$
A=\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}\\
a_{31} & a_{32}
\end{bmatrix}\;(3\times 2),
\qquad
B=\begin{bmatrix}
b_{11} & b_{12} & b_{13} & b_{14}\\
b_{21} & b_{22} & b_{23} & b_{24}
\end{bmatrix}\;(2\times 4).
$$

### Inner product — one cell at a time

$$
C_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j},\qquad C \text{ is } 3\times 4.
$$

For example $C_{23} = a_{21}b_{13} + a_{22}b_{23}$. All twelve cells are filled
this way, each completed before moving to the next.

### Outer product — one layer at a time ($K = 2$ layers)

$$
\mathrm{Grid}_{1}=
\begin{bmatrix} a_{11}\\ a_{21}\\ a_{31}\end{bmatrix}
\otimes
\begin{bmatrix} b_{11} & b_{12} & b_{13} & b_{14}\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} & a_{11}b_{12} & a_{11}b_{13} & a_{11}b_{14}\\
a_{21}b_{11} & a_{21}b_{12} & a_{21}b_{13} & a_{21}b_{14}\\
a_{31}b_{11} & a_{31}b_{12} & a_{31}b_{13} & a_{31}b_{14}
\end{bmatrix}
$$

$$
\mathrm{Grid}_{2}=
\begin{bmatrix} a_{12}\\ a_{22}\\ a_{32}\end{bmatrix}
\otimes
\begin{bmatrix} b_{21} & b_{22} & b_{23} & b_{24}\end{bmatrix}
=
\begin{bmatrix}
a_{12}b_{21} & a_{12}b_{22} & a_{12}b_{23} & a_{12}b_{24}\\
a_{22}b_{21} & a_{22}b_{22} & a_{22}b_{23} & a_{22}b_{24}\\
a_{32}b_{21} & a_{32}b_{22} & a_{32}b_{23} & a_{32}b_{24}
\end{bmatrix}
$$

$$
C = \mathrm{Grid}_{1} + \mathrm{Grid}_{2}.
$$

Reading position $(2,3)$ from the sum gives $a_{21}b_{13} + a_{22}b_{23}$ —
identical to the inner-product result for $C_{23}$ above.

---

## Conclusion

Comparing the final resulting matrices shows they are identical:

- **Traditional (Dot Product):** loops over $M$ and $N$ on the outside,
  computing a single scalar element to completion over $K$ steps.
- **Outer-Product:** loops over $K$ on the outside, computing a full
  $M \times N$ grid layer per step and accumulating the $K$ rank-1 layers.

Both realize $C_{ij} = \sum_{k=1}^{K} a_{ik} b_{kj}$; they differ only in the
order the terms are summed. Choosing between them in real code is a performance
decision (data reuse, memory-access pattern, how hardware accumulators and
matrix cores are fed), not a correctness one.
