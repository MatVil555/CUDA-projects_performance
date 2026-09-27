// =====================================================================
// softmax_l2_shared.cu -- LEVEL 2: ONE BLOCK PER ROW + SHARED-MEMORY
// TREE REDUCTION
// =====================================================================
// The fix for Level 1's two problems is the same single change: stop
// giving a row to a thread and give it to a whole block.
//
// COALESCING. Inside a row, thread t handles columns t, t+BLOCK,
// t+2*BLOCK, ... (a "strided" or grid-stride loop). Consecutive threads
// now hold consecutive addresses, so the 32 lanes of a warp cover 128
// contiguous bytes and every load is a perfectly coalesced 4-sector
// request. Sectors per request drops from ~32 to ~4 and the wasted
// 7/8 of the bus disappears. This one indexing change is usually worth
// most of the total speedup in this file.
//
// PARALLELISM. The grid is now M blocks of 256 threads = 2.1M threads
// instead of 8192. The machine is full, so DRAM latency is hidden by
// having other warps to run.
//
// THE REDUCTIONS. Each thread ends its strided walk holding a partial
// max (or partial sum) in a register. Those 256 partials must be
// combined, and that is what shared memory is for: it is the only
// cheap way for threads in a block to see each other's registers.
// The standard tree halves the number of live partials each step,
// finishing in log2(256) = 8 steps instead of 256 serial ones.
//
// The __syncthreads() inside the tree loop is not optional. Shared
// memory writes are only guaranteed visible to other threads after a
// barrier, and each step reads what the previous step wrote.
//
// A FREE BONUS: THE TREE IS ALSO MORE ACCURATE.
// This is usually overlooked. Sequential FP32 summation of n terms
// accumulates rounding error that grows like O(n), because the running
// total gets large while the addends stay small and low-order bits fall
// off the end of the mantissa. A tree performs pairwise summation, where
// addends are always of comparable magnitude, and the error grows like
// O(log n). Emulating both on this input at N = 4096 gives a worst-case
// relative error of 5.6e-6 for Level 1's sequential loop versus 2.4e-7
// for this tree -- about 20x better. So the parallel reduction is not a
// speed-for-accuracy trade; it wins on both. Level 3's warp shuffle
// reduction is the same tree and measures the same 2.4e-7.
//
// THE REUSE HAZARD -- the subtle bug this kernel is built to show.
// The same smem[] array is used for both reductions to save space. After
// the max tree, every thread reads smem[0]. Then every thread writes
// smem[tid] with its partial sum. Without a barrier in between, a fast
// thread can overwrite smem[0] while a slow thread has not yet read it,
// and that thread silently gets a wrong row max. It is a race, so it
// shows up as intermittent wrong answers on some GPUs and not others --
// the worst possible failure mode. The lone __syncthreads() after
// "const float row_max = smem[0];" is what makes this correct.
//
// WHAT IS STILL WRONG. Two things, both fixed at Level 3:
//   * Still 3 read passes + 1 write pass over 128 MiB. The read of X in
//     pass 3 is pure waste caused by the algorithm's structure.
//   * The last 5 steps of each tree (s = 16,8,4,2,1) operate entirely
//     within a single warp, yet still pay for a full block-wide barrier
//     and a shared-memory round trip. Threads in a warp can exchange
//     registers directly. That is exactly what Level 3 uses.
// =====================================================================
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#include "softmax_kernels.h"

namespace {

constexpr int kL2BlockThreads = 256;  // Level 2: threads per row (power of two!)

template <int BLOCK>
__global__ void softmax_l2_shared(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int M, int N)
{
    // One float per thread, reused for both reductions (1 KiB at
    // BLOCK=256). Keeping shared usage small matters: it is a per-SM
    // resource, and asking for a lot of it reduces how many blocks can
    // be resident, which costs you exactly the latency hiding this
    // level just bought.
    __shared__ float smem[BLOCK];

    const int row = blockIdx.x;          // one block <-> one row
    const int tid = threadIdx.x;
    if (row >= M) return;

    const float* xr = X + static_cast<size_t>(row) * N;
    float*       yr = Y + static_cast<size_t>(row) * N;

    // --- Pass 1: coalesced strided walk -> per-thread partial max ---
    float m = -FLT_MAX;
    for (int j = tid; j < N; j += BLOCK) {
        m = fmaxf(m, xr[j]);
    }

    // --- Tree reduction for the max ---
    smem[tid] = m;
    __syncthreads();
#pragma unroll
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    const float row_max = smem[0];   // broadcast read: all lanes hit the
                                     // same address, which shared memory
                                     // serves in one cycle with no bank
                                     // conflict.
    __syncthreads();                 // <-- THE REUSE HAZARD GUARD. Do not
                                     // remove: every thread must finish
                                     // reading smem[0] before anyone
                                     // overwrites smem[] below.

    // --- Pass 2: shifted exponential sum ---
    float denom = 0.0f;
    for (int j = tid; j < N; j += BLOCK) {
        denom += expf(xr[j] - row_max);
    }

    // --- Tree reduction for the sum ---
    smem[tid] = denom;
    __syncthreads();
#pragma unroll
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float inv_denom = 1.0f / smem[0];
    // No barrier needed here: smem is never written again.

    // --- Pass 3: normalize (third read of X, first write of Y) ---
    for (int j = tid; j < N; j += BLOCK) {
        yr[j] = expf(xr[j] - row_max) * inv_denom;
    }
}

}  // namespace

void launch_softmax_l2(const float* d_X, float* d_Y, int M, int N)
{
    softmax_l2_shared<kL2BlockThreads><<<M, kL2BlockThreads>>>(d_X, d_Y, M, N);
}
