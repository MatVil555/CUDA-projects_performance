// =====================================================================
// softmax_l1_naive.cu -- LEVEL 1: ONE THREAD PER ROW
// =====================================================================
// Each thread owns an entire row and walks it three times with ordinary
// sequential loops. This is the version everybody writes first because
// it is a literal transcription of the math, and it is pathologically
// bad on a GPU for two independent reasons.
//
// REASON 1: THE MEMORY ACCESS PATTERN IS TRANSPOSED.
// This is the dominant cost. A warp is 32 threads executing one load
// instruction together. Here thread t reads xr[j] where its row base is
// t*N, so at any instant the 32 lanes of a warp are requesting
// addresses that are N*4 = 16 KiB apart.
//
// The memory system does not service addresses, it services 32-byte
// sectors. Thirty-two lanes landing in thirty-two *different* sectors
// means one load instruction becomes 32 separate memory transactions,
// each dragging in 32 bytes to deliver the 4 bytes actually wanted.
// That is 1/8 bus efficiency -- seven eighths of every byte crossing the
// DRAM bus is thrown away. A coalesced warp load, where 32 lanes cover
// 128 contiguous bytes, needs only 4 sectors. So this kernel issues
// roughly 8x the transactions and moves roughly 8x the bytes of the
// kernels below. In Nsight Compute this is the sectors-per-request
// metric reading ~32 instead of ~4.
//
// Caching does not save it either: the 32 rows a warp is walking are
// 16 KiB each, a 512 KiB working set per warp, far beyond L1. So each of
// the three passes really does go back to DRAM.
//
// REASON 2: THERE IS NOT ENOUGH PARALLELISM TO HIDE LATENCY.
// The grid has exactly M threads. At M = 8192 that is 256 warps for the
// whole GPU. Spread over the ~40-140 SMs of a modern device that is a
// handful of warps per SM, which is nowhere near enough to keep the
// scheduler busy while any one warp waits on a DRAM round trip.
// =====================================================================
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#include "softmax_kernels.h"

namespace {

constexpr int kL1BlockThreads = 256;  // Level 1: threads per block

__global__ void softmax_l1_naive(const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  int M, int N)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* xr = X + static_cast<size_t>(row) * N;
    float*       yr = Y + static_cast<size_t>(row) * N;

    // Pass 1: max. Seeded with -FLT_MAX rather than -INFINITY; see the
    // note in Level 3 for why that matters once rescaling is involved.
    float m = -FLT_MAX;
    for (int j = 0; j < N; ++j) {
        m = fmaxf(m, xr[j]);
    }

    // Pass 2: denominator, with the stability shift applied.
    // This serial accumulation is also the least accurate summation in
    // the file: O(N) error growth, measured at 5.6e-6 worst-case
    // relative error versus 2.4e-7 for the tree reductions elsewhere.
    float denom = 0.0f;
    for (int j = 0; j < N; ++j) {
        denom += expf(xr[j] - m);
    }

    // Pass 3: normalize. Note this re-reads the row a third time and
    // recomputes every exp, because a thread cannot cache 4096 floats.
    const float inv_denom = 1.0f / denom;
    for (int j = 0; j < N; ++j) {
        yr[j] = expf(xr[j] - m) * inv_denom;
    }
}

}  // namespace

void launch_softmax_l1(const float* d_X, float* d_Y, int M, int N)
{
    const int grid = (M + kL1BlockThreads - 1) / kL1BlockThreads;
    softmax_l1_naive<<<grid, kL1BlockThreads>>>(d_X, d_Y, M, N);
}
