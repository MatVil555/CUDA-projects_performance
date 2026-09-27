// =====================================================================
// Exercise 2 / 8 -- global memory coalescing (two lines changed)
// =====================================================================
// THE CHANGE
//   Compared with matmul_1, the algorithm, the block size, the grid
//   size, the instruction count and the occupancy are all identical.
//   Two lines are swapped:
//
//       exercise 1:  row = blockIdx.x*blockDim.x + threadIdx.x;
//                    col = blockIdx.y*blockDim.y + threadIdx.y;
//
//       exercise 2:  col = blockIdx.x*blockDim.x + threadIdx.x;
//                    row = blockIdx.y*blockDim.y + threadIdx.y;
//
//
// WHAT TO LOOK AT IN NSIGHT COMPUTE
//   ncu --set full -o reports/matmul_2 ./matmul_2 1024 1
//
//   * Compare sectors-per-request against exercise 1: it should drop
//     from ~32 to ~4 for both the load and the store:
//       l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
//       l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
//   * dram__bytes_read.sum should fall sharply while the instruction
//     count (smsp__inst_executed.sum) stays essentially unchanged --
//     proof that this is purely an access-pattern win, not less work.
//   * Memory throughput (dram__throughput.avg.pct_of_peak_sustained_elapsed)
//     should now be high. The kernel has moved from latency-bound to
//     bandwidth-bound, which is what exercise 3 attacks next.
//
// THE LESSON
//   Profile before optimizing. If you had jumped straight from
//   exercise 1 to shared-memory tiling, you would have credited tiling
//   with a speedup that mostly belonged to fixing the index mapping.
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

#define BLOCK_DIM 32

__global__ void matmul_coalesced_kernel(const float* A, const float* B,
                                        float* C, int N) {
    // threadIdx.x -> column: consecutive threads now touch consecutive
    // addresses in B and C.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_coalesced_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK();
}

int main(int argc, char** argv) {
    ExerciseInfo info{
        "matmul_2",
        "coalesced global access: threadIdx.x now maps to the column",
        1,
        1e-3
    };
    return run_exercise(argc, argv, info);
}
