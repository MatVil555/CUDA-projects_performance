// =====================================================================
// STEP 4 -- cuBLAS: LIB BASELINE
// =====================================================================
// WHY THIS STEP.
// Compare custom kernel vs vendor lib. The library win depends on problem type:
//   GEMM      lib wins decisively -- deep blocking, arch-tuned, huge reuse.
//   softmax   lib LOSES -- no fused prim, extra DRAM passes required.
//   reducton  roughly a TIE -- single streaming pass, no data reuse. Read
//             mem fast, and step 3 already does that.
//
// Expect step 3 & 4 to be close. Knowing which regime you are in is the skill.
//
// TWO ROUTINES (NEITHER IS A PLAIN "SUM")
// 4a: cublasSasum computes sum(|x_i|) -- L1 norm, NOT plain sum. Works here
//     only because input is non-negative by construction. Hand it signed
//     data and it gives garbage. Common trap since BLAS lacks a plain sum.
//
// 4b: cublasSdot(x, ones) is honest signed sum (sum_i x_i * 1). Works for any
//     input, but reads TWO arrays -> moves 2x bytes. On a bandwidth-bound
//     task, expect ~2x runtime plus extra device mem for 1.0s.
//
// POINTER MODE
// By default cuBLAS returns scalars to HOST mem, forcing sync. We set
// CUBLAS_POINTER_MODE_DEVICE so results stay on device and pipeline like
// custom kernels -- true apples-to-apples comparsion.


#include "reduction_common.h"

// ---------------------------------------------------------------------
// 4a: sum of absolute values. Equal to the true sum here only because
// every element is non-negative.
// ---------------------------------------------------------------------
void launch_reduce_l4_asum(const ReduceContext& c)
{
    CUBLAS_CHECK(cublasSasum(c.cublas, static_cast<int>(c.N), c.d_x, 1, c.d_result));
}

// ---------------------------------------------------------------------
// 4b: dot product against a vector of ones -- a signed-safe sum that
// pays 2x the memory traffic for it.
// ---------------------------------------------------------------------
void launch_reduce_l4_dot(const ReduceContext& c)
{
    CUBLAS_CHECK(cublasSdot(c.cublas, static_cast<int>(c.N),
                            c.d_x, 1, c.d_ones, 1, c.d_result));
}

// ---------------------------------------------------------------------
// Builds the ones vector for 4b. Trivially bandwidth-bound and run once
// during setup, so it needs no optimization.
// ---------------------------------------------------------------------
__global__ void k_fill(float* __restrict__ p, float value, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}

void fill_ones(float* d_p, size_t n)
{
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    k_fill<<<blocks, threads>>>(d_p, 1.0f, n);
    CUDA_CHECK_LAUNCH();
}
