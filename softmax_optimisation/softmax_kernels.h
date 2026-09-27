// =====================================================================
// softmax_kernels.h -- host-callable entry points for the three
// hand-written softmax kernels, one per .cu file:
//
//   launch_softmax_l1  ->  softmax_l1_naive.cu     (naive, thread/row)
//   launch_softmax_l2  ->  softmax_l2_shared.cu    (tiled, block/row)
//   launch_softmax_l3  ->  softmax_l3_warp_online.cu (online, warp/row)
//
// Each function launches its kernel with its own fixed launch
// configuration and returns immediately (asynchronous), exactly like
// the lambdas in the original single-file version. Callers are
// responsible for error checking (cudaPeekAtLastError) and
// synchronization (cudaDeviceSynchronize), same as before.
//
// This header has no CUDA-syntax (<<<>>>) in it, so it is safe to
// include from main.cpp, which is compiled as ordinary C++.
// =====================================================================
#pragma once

void launch_softmax_l1(const float* d_X, float* d_Y, int M, int N);
void launch_softmax_l2(const float* d_X, float* d_Y, int M, int N);
void launch_softmax_l3(const float* d_X, float* d_Y, int M, int N);
