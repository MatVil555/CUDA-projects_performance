// =====================================================================
// Exercise 4 / 7 -- one thread computes many outputs
// =====================================================================
// THE PROBLEM LEFT BY EXERCISE 3
//   Count the instructions in exercise 3's inner loop: two shared-memory
//   loads (As and Bs) feed exactly one FMA. The FMA pipes are idle most
//   of the time because the load/store unit cannot keep up, shared memory
//   removed the *global* bottleneck and became the new bottleneck itself.
//
//   The fix is not a faster load -- it is fewer loads per FMA. That means
//   reuse at the next level down the hierarchy: registers.
//
// THE IDEA
//   Give each thread a *column trip* of TM = 8 outputs instead of one.
//   In the inner loop, load one balue of b into a register and reuse it
//   against TM values of A:
//
//       tmp_b = Bs[k][thread_col];          // 1 shared load
//       for (i = 0; i < TM; ++i)            // TM shared loads + TM FMAs
//           acc[i] += As[thread_row*TM + i][k] * tmp_b;
//
//   That is TM+1 = 9 shared loads for TM = 8 FMAs, versus 2 loads per 1
//   FMA before: the load-to-FMA ratio improves ~1.8x. The TM
//   accumulators live in register for the entire kernel andare written
//   to global memory exactly once at the end.
//


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



#define BM 64                          // rows of C per block
#define BN 64                          // columns of C per block
#define BK 8                           // k-slice staged in shared memory
#define TM 8                           // outputs (rows) per thread
#define THREADS ((BM * BN) / TM)       // 512

__global__ __launch_bounds__(THREADS)
void matmul_1d_tiling_kernel(const float* A, const float* B,
                             float* C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;   // 0..511

    // --- mapping used for COMPUTE: chosen for register reuse ---------
    // Each thread owns column `thread_col` and rows
    // thread_row*TM .. thread_row*TM + TM-1 of the block's output tile.
    const int thread_col = tid % BN;   // 0..63
    const int thread_row = tid / BN;   // 0..7   (7*8+7 = 63 = BM-1, covers BM)

    // --- mapping used for LOADING: chosen for coalescing -------------
    // A tile is BM x BK
    const int a_row = tid / BK;        // 0..63
    const int a_col = tid % BK;        // 0..7
    // B tile is BK x BN


    const int b_row = tid / BN;        // 0..7
    const int b_col = tid % BN;        // 0..63

    float acc[TM] = {0.0f};

    for (int t = 0; t < N / BK; ++t) {
        const int k0 = t * BK;

        As[a_row][a_col] = A[(block_row + a_row) * N + (k0 + a_col)];
        Bs[b_row][b_col] = B[(k0 + b_row) * N + (block_col + b_col)];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // One shared-memory load of B, reused TM times from a
            // register. This singleline is the whole point of the step.
            const float tmp_b = Bs[k][thread_col];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[thread_row * TM + i][k] * tmp_b;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        C[(block_row + thread_row * TM + i) * N + (block_col + thread_col)] = acc[i];
    }
}
void launch_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(THREADS);
    dim3 grid(N / BN, N / BM);

    matmul_1d_tiling_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK();
}



int main(int argc, char** argv) {
    ExerciseInfo info{
        "matmul_4",
        "1D register tiling: 8 outputs per thread, ~2x fewer shared loads per FMA",
        BM,       // BM == BN == 64
        1e-3
    };
    return run_exercise(argc, argv, info);
}
