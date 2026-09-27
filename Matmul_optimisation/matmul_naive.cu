// =====================================================================
// Exercise 1 / 8 -- the naive baseline
// =====================================================================
// THE KERNEL
//   One thread per output element. Each thread walks the whole k
//   dimension, reading one row of A and one column of B straight out of
//   global memory, and writes a single element of C. This is the
//   textbook translation of the triple loop, and it is the honest
//   starting point: nothing here is wrong, it is just slow.
//
// THE DELIBERATE MISTAKE
//   The index mapping is intentionally the wrong way round:
//
//       row = blockIdx.x * blockDim.x + threadIdx.x    <-- threadIdx.x -> ROW
//       col = blockIdx.y * blockDim.y + threadIdx.y
//
//   threadIdx.x is the fast-varying dimension: threads 0..31 of a warp
//   differ only in threadIdx.x. Mapping it to the row means the 32
//   threads of a warp touch 32 *different rows* of A and C, i.e.
//   addresses N*4 bytes apart. Every warp-wide load of A and every
//   warp-wide store of C therefore needs 32 separate memory
//   transactions instead of 4. Exercise 2 changes these two lines and
//   nothing else.
//
// WHY IT IS SLOW (the arithmetic worth internalising)
//   The kernel reads 2N floats and writes 1 float per output element,
//   so for the whole matrix it moves 2*N^3 + N^2 floats. For N = 1024
//   that is about 8.6 GB of load traffic to do 2.1 GFLOP of work: an
//   arithmetic intensity of ~0.25 FLOP/byte. No GPU on the market can
//   feed that. A correct GEMM only *needs* to move 3*N^2 floats (12 MiB
//   at N = 1024) -- the harness prints that number. The whole ladder is
//   about closing the gap between 8.6 GB and 12 MiB.
//
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

#define BLOCK_DIM 32   // 32 x 32 = 1024 threads per block

__global__ void matmul_naive_kernel(const float* A, const float* B,
                                    float* C, int N) {
    // Deliberately uncoalesced mapping...
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;


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

    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK();   // must stay on the line directly after the launch: the
                    // macro reports __LINE__-1
}

int main(int argc, char** argv) {
    ExerciseInfo info{
        "matmul_1",
        "naive: one thread per output element, uncoalesced index mapping",
        1,        // any N
        1e-3      // FP32 tolerance
    };
    return run_exercise(argc, argv, info);
}
