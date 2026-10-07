// =====================================================================
// STEP 3 -- CASCADED LOAD + WARP SHUFFLE TREE
// =====================================================================
// MOTIVATION
// Steps 1 and 2 has the tree right and are still nowhere near the
// hardware limit. The reason is that a reduction is a *bandwidth*
// problem -- one add per 4-byte element is 0.25 flop/byte, and a modern
// GPU wants 10-40 flop/byte to be compute bound. The only thing that
// matters is reading teh array once at  fulll bus speed. Anyting the
// instruction stream does besides issuing loads is a cycle the memory
// system spends idle
//
// Steps 1 and 2 are instruction-bound wile teh DRAM bus starves. This
// step strips the instruction stream down to little more than loads and
// adds. Three changes..., in descending order of importance.
//
// =====================================================================

#include "reduction_common.h"

template <int BLOCK>
__global__ void reduce_l3_shuffle(const float* __restrict__ x,
                                  float* __restrict__ partials,
                                  size_t N)
{
    const size_t grid_stride = static_cast<size_t>(BLOCK) * gridDim.x;
    const size_t thread_id0  = static_cast<size_t>(blockIdx.x) * BLOCK + threadIdx.x;

    float sum = 0.0f;

    // --- (A)+(C): cascaded, coalesced, vectorized streaming pass ---
    const size_t nvec = N / 4;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    for (size_t i = thread_id0; i < nvec; i += grid_stride) {
        const float4 v = x4[i];
        // Two independent pairs rather than a serial chain: shortens teh
        // dependency graph inside the thread so the adds can overlap
        // instead of each waiting on the previous one.
        sum += (v.x + v.y) + (v.z + v.w);
    }
    // Scalar tail: at most 3 elements, but written as a grid-stride loop
    // so every remaining index is visited exactly once for any N.
    for (size_t i = nvec * 4 + thread_id0; i < N; i += grid_stride) {
        sum += x[i];
    }

    // --- (B): intra-warp tree, entirely in registers ---
    sum = warp_reduce_sum(sum);

    // --- Cross-warp combine: teh only shared memory, the only barrier ---
    __shared__ float warp_sums[BLOCK / kWarpSize];
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned wid  = threadIdx.x / kWarpSize;

    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();                    // exactly one, for the whole kernel

    if (wid == 0) {
        // Pad with the identity so the final warp tree stays uniform.
        sum = (lane < BLOCK / kWarpSize) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) partials[blockIdx.x] = sum;
    }
}

void launch_reduce_l3(const ReduceContext& c)
{
    const int blocks = blocks_cascaded(c.N, c.sm_count);
    reduce_l3_shuffle<kBlock><<<blocks, kBlock>>>(c.d_x, c.d_partials, c.N);
    launch_reduce_final(c.d_partials, blocks, c.d_result);
}
