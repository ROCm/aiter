// SPDX-License-Identifier: MIT
// Physical layout, preserving contiguous four-float vectors. Shared by S1/S2.
template <int Tile>
__host__ __device__ __forceinline__ int64_t partial_address(int64_t row, int col)
{
    static_assert(Tile == 0 || Tile == 16 || Tile == 32);
    if constexpr(Tile == 0)
        return row * 512 + col;
    else
        return (row & ~int64_t(15)) * 512 + int64_t(col / Tile) * (16 * Tile) + (row & 15) * Tile +
               col % Tile;
}
struct CtaTile
{
    int q, b, c;
};
template <bool Remap>
__host__ __device__ __forceinline__ uint32_t mapped_cta(uint32_t pid, uint32_t total)
{
    if constexpr(!Remap)
        return pid;
    // Balanced eight-XCD permutation also covers odd Q * S17, with no padding.
    const uint32_t quotient = total / 8, remainder = total % 8, xcd = pid % 8, local = pid / 8;
    return xcd * quotient + (xcd < remainder ? xcd : remainder) + local;
}
template <int B, int C, bool Remap>
__device__ __forceinline__ CtaTile cta_tile()
{
    if constexpr(!Remap)
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)};
    const uint32_t pid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const uint32_t j   = mapped_cta<true>(pid, gridDim.x * B * C);
    return {int(j / (B * C)), int((j / C) % B), int(j % C)};
}
