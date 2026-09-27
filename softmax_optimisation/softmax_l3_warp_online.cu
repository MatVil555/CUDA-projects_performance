// =====================================================================
// softmax_l3_warp_online.cu -- LEVEL 3: ONE WARP PER ROW, ONLINE
// SOFTMAX, SHUFFLES, NO SHARED MEM
// =====================================================================
// Two independent optimizations, and the first one is the big one.
//
// (A) ONLINE SOFTMAX: FUSE THE MAX AND SUM PASSES INTO ONE.
// Levels 1 and 2 need max before they can start summing, because every
// term needs the shift. That data dependency is what forces a separate
// pass. The online formulation (Milakov & Gimelshein, 2018) breaks it by
// tracking a running max m and a running denominator d that is always
// expressed *relative to the current m*. When a new value v arrives:
//
//     m_new = max(m, v)
//     d_new = d * exp(m - m_new) + exp(v - m_new)
//
// The factor exp(m - m_new) retroactively re-bases the entire
// accumulated sum onto the new maximum. If v did not beat the max,
// exp(m - m_new) = exp(0) = 1 and the correction is free. If it did, the
// running total is scaled down by the amount the reference point moved.
// Either way d is exactly what a two-pass algorithm would have computed
// with the max known in advance -- and every exp argument is still <= 0,
// so the overflow safety of the max-subtraction trick is fully retained.
//
// This turns 3 read passes into 2. On a memory-bound kernel that is
// directly a ~33% reduction in traffic, and it is the single largest
// algorithmic win in this file. The price is one extra exp per update
// step (the rescale) -- pure profit when you are waiting on DRAM.
//
// Better still, the second pass re-reads a row the same warp just
// finished reading. A row is 16 KiB, so with many warps resident the
// re-read is very likely served by L2 (several MiB) rather than DRAM.
// Effective DRAM traffic therefore approaches the 2-pass ideal of
// 1 read + 1 write. Check l2_hit_rate in Nsight Compute to confirm.
//
// (B) WARP-LEVEL COMMUNICATION INSTEAD OF SHARED MEMORY.
// A row is now owned by one warp (32 threads), not a block. That means
// the 32 partial (m, d) pairs live in the registers of threads that are
// already implicitly synchronized -- a warp executes in lockstep. So:
//
//   * __shfl_down_sync reads another lane's register *directly*. No
//     shared memory round trip (store + barrier + load), no shared
//     memory capacity consumed at all, and no bank conflicts to reason
//     about. It is a register-file permute in the datapath.
//   * No __syncthreads() anywhere in the kernel. Level 2 pays ~18 block
//     barriers per row; a barrier makes every warp wait for the slowest,
//     so removing them removes a real serialization point.
//   * Because rows are independent and each is handled by one warp,
//     there is no cross-warp coordination left to do. Warp 3 can be on
//     its second pass while warp 4 is still on its first.
//
// REGISTER PRESSURE -- the thing to get right here.
// "Keep it in registers" is good advice that is easy to take too far.
// The tempting move is to cache the whole row in registers to eliminate
// the second read entirely. Do not: at N = 4096 with 32 lanes that is
// 128 floats per thread, on top of addressing and loop state. The
// hardware limit is 255 registers per thread, so the compiler would
// spill to "local" memory -- which is not local at all, it is DRAM with
// a fancy name. You would trade a likely-L2-hit re-read for guaranteed
// spill traffic, and lose. Occupancy would collapse too: at 128+
// registers a thread you get at most 2 warps per scheduler, and then
// there is nothing left to hide latency with.
//
// This kernel instead keeps only what must persist -- the two-float
// reduction state (m, d) plus one float4 in flight -- so the state is
// O(1) per thread regardless of N. Measured with
// `nvcc -arch=sm_75 -Xptxas -v`: 43 registers, **0 bytes spill stores,
// 0 bytes spill loads, 0 barriers, 0 bytes smem**. Compare Level 2 at
// 43 registers *plus* 1024 bytes of shared memory and its barriers. The
// register-cached variant, by contrast, would report a multi-hundred-byte
// stack frame and non-zero spill traffic -- always check that line before
// believing a "keep it in registers" claim.
//
// (C) VECTORIZED float4 ACCESS.
// Each lane loads 16 bytes at a time, so one warp instruction covers
// 512 contiguous bytes. Four times fewer load/store instructions for
// the same bytes, which matters once the addressing is already perfect:
// it reduces instruction issue pressure and lets the memory pipeline
// work on wider requests. Requires 16-byte alignment -- cudaMalloc is
// 256-byte aligned, so row starts are aligned iff N % 4 == 0. When that
// does not hold, nvec is set to 0 and the scalar tail loop handles the
// entire row. Correctness never depends on the fast path being taken.
// =====================================================================
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#include "softmax_kernels.h"

namespace {

constexpr int kWarpSize      = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kL3WarpsPerBlk = 8;   // Level 3: rows in flight per block

// Merge two (max, denominator) pairs. Both are expressed relative to
// their own max; the result is expressed relative to the combined max.
// This is the associative, commutative operator that makes the online
// formulation reducible in a tree at all.
__device__ __forceinline__ void online_combine(float& m, float& d,
                                                float m_other, float d_other)
{
    const float m_new = fmaxf(m, m_other);
    d = d * expf(m - m_new) + d_other * expf(m_other - m_new);
    m = m_new;
}

// Absorb four new values (one float4) into the running state.
// Taking the max of the four first means one rescale of d for four
// elements instead of four, which is why this is written as a block
// update rather than four scalar updates.
__device__ __forceinline__ void online_update4(float& m, float& d, const float4 v)
{
    const float vmax  = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    const float m_new = fmaxf(m, vmax);
    const float scale = expf(m - m_new);   // re-base the existing sum
    d = d * scale
      + expf(v.x - m_new) + expf(v.y - m_new)
      + expf(v.z - m_new) + expf(v.w - m_new);
    m = m_new;
}

template <int WARPS_PER_BLOCK>
__global__ void softmax_l3_warp_online(const float* __restrict__ X,
                                        float* __restrict__ Y,
                                        int M, int N)
{
    const int lane = threadIdx.x;                 // 0..31 within the warp
    const int warp = threadIdx.y;                 // which warp of the block
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;

    // This early exit is warp-uniform: `row` depends only on blockIdx.x
    // and threadIdx.y, both identical across the 32 lanes. So either the
    // whole warp returns or none of it does, which is what makes the
    // full-mask shuffles below legal -- a shuffle from an exited lane
    // would be undefined behaviour.
    if (row >= M) return;

    const float* xr = X + static_cast<size_t>(row) * N;
    float*       yr = Y + static_cast<size_t>(row) * N;

    // Seed with -FLT_MAX, NOT -INFINITY. With -INFINITY, the very first
    // update computes exp(-inf - (-inf)) = exp(NaN) if the row were also
    // all -inf, and inf arithmetic in the rescale is a minefield in
    // general. With -FLT_MAX the rescale is exp(-FLT_MAX - m_new), which
    // flushes cleanly to 0, and d is exactly 0 at that point so the
    // product is a clean 0 with no NaN. Same result, no special cases.
    float m = -FLT_MAX;
    float d = 0.0f;

    // Vectorized fast path, or 0 iterations if the row is not 16B-aligned.
    const int nvec = ((N & 3) == 0) ? (N >> 2) : 0;
    const float4* __restrict__ xr4 = reinterpret_cast<const float4*>(xr);

    // --- Single fused pass: max and denominator together ---
    for (int i = lane; i < nvec; i += kWarpSize) {
        online_update4(m, d, xr4[i]);
    }
    for (int j = (nvec << 2) + lane; j < N; j += kWarpSize) {   // scalar tail
        const float v = xr[j];
        const float m_new = fmaxf(m, v);
        d = d * expf(m - m_new) + expf(v - m_new);
        m = m_new;
    }

    // --- Warp reduction of the 32 (m, d) pairs, in registers ---
    // Standard shfl_down tree: after the offset-16 step lane i<16 holds
    // the merge of lanes i and i+16; upper lanes hold garbage (a shuffle
    // past the warp edge returns the lane's own value, which would
    // double-count). That garbage never flows back down, because each
    // subsequent step only pulls from lanes below the previous cut, so
    // lane 0 accumulates exclusively from valid partials.
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(kFullMask, m, off);
        const float d_o = __shfl_down_sync(kFullMask, d, off);
        online_combine(m, d, m_o, d_o);
    }
    // Lane 0 has the true row statistics; publish them to all 32 lanes.
    // Again a register-to-register broadcast, no memory involved.
    m = __shfl_sync(kFullMask, m, 0);
    d = __shfl_sync(kFullMask, d, 0);

    const float inv_denom = 1.0f / d;   // >= 1 denominator, always safe

    // --- Second pass: normalize and store ---
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = lane; i < nvec; i += kWarpSize) {
        const float4 v = xr4[i];
        float4 o;
        o.x = expf(v.x - m) * inv_denom;
        o.y = expf(v.y - m) * inv_denom;
        o.z = expf(v.z - m) * inv_denom;
        o.w = expf(v.w - m) * inv_denom;
        yr4[i] = o;                      // 16B coalesced store
    }
    for (int j = (nvec << 2) + lane; j < N; j += kWarpSize) {
        yr[j] = expf(xr[j] - m) * inv_denom;
    }
}

}  // namespace

void launch_softmax_l3(const float* d_X, float* d_Y, int M, int N)
{
    const int l3_grid = (M + kL3WarpsPerBlk - 1) / kL3WarpsPerBlk;
    const dim3 l3_block(kWarpSize, kL3WarpsPerBlk);
    softmax_l3_warp_online<kL3WarpsPerBlk><<<l3_grid, l3_block>>>(d_X, d_Y, M, N);
}
