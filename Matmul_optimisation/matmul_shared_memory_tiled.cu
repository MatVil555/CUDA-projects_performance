// =====================================================================
// Exercise 3 / 8 -- shared memory tiling (data reuse)
// =====================================================================
// THE PROBLEM LEFT BY EXERCISE 2
//   The accesses are coalesced, but the kernel still issues 2N global
//   loads per output element. Every element of A is re-read by all N
//   threads in its column-block, and every element of B by all N threads
//   in its row-block. The L2 cache absorbs some of that, but the load
//   instructions themselves still have to be issued and the data still
//   has to travel through the memory pipeline.
//
// THE IDEA
//   Stage the data in shared memory -- an on-chip, software-managed
//   scratchpad, roughly an order of magnitude lower latency than L2 and
//   with its own bandwidth. Split the k dimension into chunks of TILE:
//
//     for each tile t:
//         every thread loads exactly ONE element of A and ONE of B
//         __syncthreads()
//         every thread does TILE multiply-adds out of shared memory
//         __syncthreads()
//
//   Each block loads a TILE x TILE square of A and of B once, and the
//   block's TILE x TILE threads then read those squares TILE times each.
//   Global traffic per output element drops from 2N loads to 2N/TILE:
//   with TILE = 32, a 32x reduction. That is the reuse factor, and it is
//   the only thing this step buys -- the instruction count for the FMAs
//   is unchanged.
//
// THE TWO BARRIERS (both are mandatory)
//   * The first __syncthreads() is the obvious one: no thread may read
//     the tile until every thread has finished writing its element.
//   * The second is the one people forget: without it, a fast warp can
//     loop around and overwrite As/Bs for tile t+1 while a slow warp is
//     still reading tile t. That produces a wrong answer that depends on
//     scheduling -- it will often pass on small inputs and fail
//     intermittently on big ones. The correctness routine in
//     matmul_common.cuh is there precisely so that class of bug cannot
//     hide behind a plausible-looking GFLOP/s number.
//
// A BANK-CONFLICT NOTE (free by construction here)
//   Shared memory has 32 banks of 4 bytes. Within a warp ty is constant
//   and tx runs 0..31, so:
//     * As[ty][k] -- all 32 threads read the same address: broadcast, no
//       conflict.
//     * Bs[k][tx] -- 32 consecutive floats: one per bank, no conflict.
//   This kernel is conflict-free without any padding tricks. Later steps
//   change the access shape and have to think about it again.
//
// WHAT TO LOOK AT IN NSIGHT COMPUTE
//   ncu --set full -o reports/matmul_3 ./matmul_3 1024 1
//
//   * dram__bytes_read.sum should be roughly TILE times smaller than in
//     exercise 2 -- the headline result of this step.
//   * launch__shared_mem_per_block_static: 2*32*32*4 = 8192 bytes.
//   * l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum: expect ~0.
//   * Warp State Statistics: "Barrier" now appears as a stall reason
//     (that is the cost of the two __syncthreads()) while "Long
//     Scoreboard" drops. You have traded memory stalls for
//     synchronization stalls -- a good trade here, and exercise 4
//     reduces the barrier count per unit of work.
//   * SpeedOfLight: still nowhere near compute-bound. The kernel now
//     does one FMA per two shared-memory loads; shared memory bandwidth
//     and the load/store pipeline are the new limit, which is what
//     exercises 4-5 fix by moving reuse into registers.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

#include "matmul_common.cuh"

#define TILE 32   // 32x32 threads = 1024 per block, 8 KiB of shared memory

__global__ void matmul_shared_kernel(const float* A, const float* B,
                                     float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // N is guaranteed to be a multiple of TILE by the harness, so no
    // boundary handling is needed inside the loop.
    for (int t = 0; t < N / TILE; ++t) {
        const int k0 = t * TILE;

        // Both loads are coalesced: tx is the fast dimension and sits in
        // the contiguous index of A and of B.
        As[ty][tx] = A[row * N + (k0 + tx)];
        Bs[ty][tx] = B[(k0 + ty) * N + col];

        __syncthreads();   // tile fully written before anyone reads it

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // everyone done reading before the tile is reused
    }

    C[row * N + col] = acc;
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    matmul_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK();
}

int main(int argc, char** argv) {
    ExerciseInfo info{
        "matmul_3",
        "shared-memory tiling: 32x32 tiles, ~32x less global traffic",
        TILE,     // N must be a multiple of the tile size
        1e-3
    };
    return run_exercise(argc, argv, info);
}
