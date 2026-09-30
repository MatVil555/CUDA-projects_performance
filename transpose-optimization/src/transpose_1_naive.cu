// =====================================================================
// STEP 1 -- NAIVE: COALESCED READS, SCATTERED WRITES
// =====================================================================
// MOTIVATION
// Write the transpose straight from its definition: one thread per
// element, `threadIdx.x` mapped to the column of A. That mapping is the
// only sane choice for the read, and then the write index falls where it
// falls. The result establishes the baseline and exhibits, in a single
// line of code, the one difficulty the whole ladder exists to solve.
//
// THE READ IS PERFECT.
//     A[row*N + col],  col = blockIdx.x*blockDim.x + threadIdx.x
// Consecutive lanes hold consecutive `col`, so a warp's 32 lanes request
// 32 consecutive floats = 128 contiguous bytes. The memory system
// services that with 4 sectors of 32 bytes -- the minimum possible.
// Textbook coalescing; nothing to improve.
//
// THE WRITE IS BAD.
//     B[col*M + row]
// Now `col` is the *row* index of B, and it is the thing varying across
// the warp. Lane L writes to B[(col0+L)*M + row], so consecutive lanes are
// M floats = 4*M bytes apart. At the default M = 4096 that is 16 KiB
// between neighbouring lanes.
// =====================================================================

#include "transpose_common.cuh"

__global__ void transpose_l1_naive(const float* __restrict__ A,
                                   float* __restrict__ B,
                                   int M, int N)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // column of A, [0,N)
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // row of A,    [0,M)

    if (row < M && col < N) {
        // Coalesced load (lanes consecutive in col) ...
        // ... scattered store (lanes M floats apart).
        // The entire problem, in one line.
        B[static_cast<size_t>(col) * M + row] = A[static_cast<size_t>(row) * N + col];
    }
}

void launch_transpose_l1(const TransposeContext& c)
{
    transpose_l1_naive<<<naive_grid(c.M, c.N), naive_block()>>>(c.d_A, c.d_B, c.M, c.N);
}
