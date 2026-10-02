// =====================================================================
// Exercise 5 / 8 -- 2D register tiling (the outer product)
// =====================================================================

//   Exercise 4 reached TM+1 = 9 shared-memory loads per TM = 8 FMAs. good!
//   but the ratio is still ~1.1 loads per FMA. The strip of outputs was
//   one-dimensional, so a loaded value of B was reused TM times while a
//   loaded value of A was reused only unce.
//
//
// THE shape
//   BM = BN = 128, BK = 8, TM = TN = 8.
//   Threads per block = (BM*BN)/(TM*TN) = 16384/64 = 256.
//   Shared memory = (128*8 + 8*128)*4 = 8 KiB.
//   Each tile is 1024 floats and there are only 256 threads, so every
//   thread loads 4 elements per tile -- hence the strided load loops
//   (stride_a, stride_b) instead of exercise 4's single assignment.
/


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

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define THREADS ((BM * BN) / (TM * TN))   // 256

__global__ __launch_bounds__(THREADS)
void matmul_2d_tiling_kernel(const float* A, const float* B,
                             float* C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;   // 0..255

    // --- compute mapping: each thread owns a TM x TN sub-tile 
    const int thread_row = tid / (BN / TN);   // 0..15
    const int thread_col = tid % (BN / TN);   // 0..15

    // --- load mapping: 1024 elements per tile, 256 threads -> 4 eahc --
    const int a_row = tid / BK;          // 0...31
    const int a_col = tid % BK;          // 0...7
    const int stride_a = THREADS / BK;   // 32 rows of A per pass
    const int b_row = tid / BN;          // 0..1
    const int b_col = tid % BN;          // 0..127
    const int stride_b = THREADS / BN;   // 2 rows of B per pass

    float acc[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int t = 0; t < N / BK; ++t) {
        const int k0 = t * BK;

        #pragma unroll
        for (int off = 0; off < BM; off += stride_a) {
            As[a_row + off][a_col] = A[(block_row + a_row + off) * N + (k0 + a_col)]; //
        }
        #pragma unroll
        for (int off = 0; off < BK; off += stride_b) {
            Bs[b_row + off][b_col] = B[(k0 + b_row + off) * N + (block_col + b_col)]; //
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Pull the two strips into registers once then do TM*TN FMAs entirely out of registers.
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[thread_row * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k][thread_col * TN + j];

            //
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }



    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[row * N + (block_col + thread_col * TN + j)] = acc[i][j];
        }
    }
}



void launch_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(THREADS);
    dim3 grid(N / BN, N / BM);

    matmul_2d_tiling_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK();
}

int main(int argc, char** argv) {
    ExerciseInfo info{
        "matmul_5",
        "2D register tiling: 8x8 outputs per thread, 16 shared loads per 64 FMAs",
        BM,       // BM == BN == 128
        1e-3
    };
    return run_exercise(argc, argv, info);
}
