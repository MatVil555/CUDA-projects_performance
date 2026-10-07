// ==========
// SHARED PLUMBING -- COLLAPSING THE PER-BLOCK PARTIALS
// ==============================
// Not a step of the ladder. This is the second half that steps 1, 2 and 3
// all need, factored out so they share it identicallly.
//
// WHY IT HAS TO EXIST.
// A block cannot see another block's results, so any single-kernel
// reduction can only get as far as one value per block. Somthing must
// combine those. The options, and why this one:
//
//   * atomicAdd from each block into one global float. One fewer launch,
//     but the summation ORDER becomes nondeterministic, so the answer
//     shifts slightly from run to run. For a benchmark that reports
//     relative error at the 1e-8 level that is disqualifiing -- you could
//     no longer distinguish a real bug from atomic jitter. Rejected on
//     reproducibility grounds, not speed.
//   * Copy the partials to the host and finish there. Adds a PCIe round
//     trip to every iteration, which would dominate the meassurement.
//   * A second kernel launch. Deterministic, and the data is tiny.
//
// So: one block, grid-stride cascade, warp shuffle tree -- structuraly
// step 3, applied to a very small array. The SAME function is used by
// steps 1, 2 and 3, so the differences the benchmark reports isolate
// stage 1, which is the entire point of the ladder.
//
// COST ASYMMETRY WORTH KNOWING.
// Steps 1 and 2 hand this 128,000 partials (500 KiB) at the default N;
// step 3 hands it ~1,300. Even the 500 KiB case is 0.4% of the 125 MiB
// input and runs on a single SM in tens of microseconds, so it does not
// distrot the comparison -- but it is a real asymmetry, and it is a
// second reason cascading helps: fewer blocks means a cheaper tail.


#include "reduction_common.h"
template <int BLOCK>
__global__ void reduce_final_kernel(const float* __restrict__ partials,
                                    float* __restrict__ out,
                                    int n)
									
{



    // Single block, so the grid-stride loop is just a block-stride loop.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) sum += partials[i];


    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[BLOCK / kWarpSize];
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned wid  = threadIdx.x / kWarpSize;

    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (lane < BLOCK / kWarpSize) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) *out = sum;
    }
}




void launch_reduce_final(const float* d_partials, int n, float* d_result)
{
    reduce_final_kernel<kFinalBlock><<<1, kFinalBlock>>>(d_partials, d_result, n);
}
