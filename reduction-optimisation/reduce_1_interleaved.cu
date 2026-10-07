// =====================================================================
// STEP 1 -- INTERLEAVED ADDRESSING: THE DIVERGENT BASELINE
// =====================================================================
//
// THE BOTTLENECK: WARP DIVERGENCE.
// A wrap issues one instruction for all 32 of its lanes. Lanes failing a
// predicate are *masked off*, not skipped -- the instruction still
// occupies the issue slot and the cycle. Because active threads here are
// spread `2*stride` apart, the useful fraction of each issue collapses:
//
//     stride=1  -> lanes 0,2,4,...,30 active  = 16/32 = 50%
//     stride=2  -> lanes 0,4,8,...,28 active  =  8/32 = 25%
//     stride=4  -> lanes 0,8,16,24    active  =  4/32 = 12.5%
//     stride=8  -> lanes 0,16         active  =  2/32 = 6.25%
//     stride=16 -> lane  0            active  =  1/32 = 3.1%
//
// Averaged over those five steps, ~19% lane utilization: four fifths of
// the scheduler's work is discarded. EVERY wrap in the block also runs
// all 8 iterations and all 8 barriers even once none of its lanes can
// contribute anything.
//
// A CLAIM THIS FILE DELIBERATELY CONTRADICTS: "% IS VERY SLOW".
// Every treatment of this kernel (including NVIDIA's classic reduction
// slides) blames `tid % (2*stride)` for being an expensive integer
// remainder, since  nvd GPU synthesize integer division from ~20
// instructions. True in general; NOT tru as this compiles. Measured by
// dumping the PTX:
//
//   kBlock a compile-time constant (what we do):
//       0 x rem.u32,  8 x and.b32,  9 x bar.sync
//   the same loop written against runtime blockDim.x:
//       1 x rem.u32,  0 x and.b32,  2 x bar.sync
//
// Because kBlock is a compile-time constant the trip count is known, so
// nvcc fully unrolls all 8 iterations; each copy then has a *literal*
// stride, making `2*stride`  a literal power of two, and `%` is
// strength-reduced to a single `and.b32`. The expensive remainder only
// appears when the block size is a runtime value and the loop therefore
// cannot unroll
//
// So the honest accounting for step 1 vs step 2 is: divergence is real
// and dominant and  the modulo costs 8 extra ANDs, which is minor. Read the
// generated code before believing a stoy about instruction cost, even a
// story printed in a textbook.
//

// =====================================================================

#include "reduction_common.h"

template <int BLOCK>
__global__ void reduce_l1_interleaved(const float* __restrict__ x,
                                      float* __restrict__ partials,
                                      size_t N)
{
    __shared__ float sdata[BLOCK];

    const unsigned tid = threadIdx.x;
    const size_t   i   = static_cast<size_t>(blockIdx.x) * BLOCK + tid;

    // Pad with the additive identity so a ragged final block needs no
    // special case.
    sdata[tid] = (i < N) ? x[i] : 0.0f;
    __syncthreads();

    // Interleaved tree: stride grows, active threads spread apart.
    for (unsigned stride = 1; stride < BLOCK; stride *= 2) {
        // The divergent guard -- the bottleneck on display. Written with
        // % on purpose, though see the header: it compiles to an AND.
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();   // 8 loop barriers (9 in the kernel) at BLOCK=256
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

void launch_reduce_l1(const ReduceContext& c)
{
    const int blocks = blocks_one_per_element(c.N);
    reduce_l1_interleaved<kBlock><<<blocks, kBlock>>>(c.d_x, c.d_partials, c.N);
    launch_reduce_final(c.d_partials, blocks, c.d_result);
}
