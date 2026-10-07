// =====================================================================
// STEP 2 -- SEQUENTIAL ADDRESSING: divergence-FREE
// =====================================================================
// MOTIVATION
// Step 1 wastes ~80% of its issues slots on masked-off lanes. Fix that and
// nothing else. Identical tree depth, identical add count, identical
// barrier count (9), identical shared-memory footprint -- the only change
// is *which* threads do the work. The stride now shrinks instead of
// growing, and the guard becomes `tid < stride`:
//
//   stride=128: t0....t127 += t128..t255
//   stride=64 : t0..t63  += t64...t127
//   stride=32 : t0..t31  += t32....t63...
// This is the cleanest demonstration in the ladder that *how you map
// threads to data* can matter more than anything about the arithmetic.
//

#include "reduction_common.h"

template <int BLOCK>
__global__ void reduce_l2_sequential(const float* __restrict__ x,
                                     float* __restrict__ partials,
                                     size_t N)
{
    __shared__ float sdata[BLOCK];

    const unsigned tid = threadIdx.x;
    const size_t   i   = static_cast<size_t>(blockIdx.x) * BLOCK + tid;

    sdata[tid] = (i < N) ? x[i] : 0.0f;
    __syncthreads();

    // Sequential tree: stride shrinks, active threads stay contiguous.
    for (unsigned stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

void launch_reduce_l2(const ReduceContext& c)
{
    const int blocks = blocks_one_per_element(c.N);
    reduce_l2_sequential<kBlock><<<blocks, kBlock>>>(c.d_x, c.d_partials, c.N);
    launch_reduce_final(c.d_partials, blocks, c.d_result);
}
