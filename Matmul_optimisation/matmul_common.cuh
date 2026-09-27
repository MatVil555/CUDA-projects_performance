#pragma once
// =====================================================================
// matmul_common.cuh
// Shared support code for the matmul profiling exercises.
//
// This header deliberately contains NO kernel and NO optimization: it
// only provides the scaffolding that is identical for every step of the
// ladder, so each matmul_N.cu file contains nothing but the one idea it
// is teaching plus its own launch configuration.
//
//   create_known_matrices()  -> fills A and B with a fixed, reproducible
//                               pattern whose exact product is known in
//                               closed form.
//   check_result()           -> verifies a computed C against that closed
//                               form (O(N^2), no CPU GEMM needed).
//   cpu_matmul_reference()   -> brute-force O(N^3) double-precision GEMM,
//                               only used to cross-check the closed form.
//   run_exercise()           -> allocate / copy / warm up / time / verify
//                               harness shared by every exercise.
//
// IMPORTANT: this header does not define the CUDA_CALL / CUDA_CHECK
// error-handling macros. Every exercise file defines that block itself
// (see the #error below), because reproducing it and using it on every
// single CUDA API call is part of the exercise.
// =====================================================================

#if !defined(CUDA_CALL) || !defined(CUDA_CHECK)
#error "Put the CUDA_CALL / CUDA_CHECK macro block at the top of the .cu file, BEFORE including matmul_common.cuh."
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Every exercise file provides this. The harness below calls it.
// Timing happens around this call, so it must contain the kernel launch
// and nothing expensive besides it.
// ---------------------------------------------------------------------
void launch_matmul(const float* d_A, const float* d_B, float* d_C, int N);

// Describes one exercise to the harness.
struct ExerciseInfo {
    const char* id;               // e.g. "matmul_3"
    const char* title;            // one-line description of the optimization
    int         size_multiple_of; // N must be a multiple of this (1 = any N)
    double      rtol;             // 1e-3 is right for an FP32 kernel; only a
                                  // reduced-precision path (TF32 tensor cores)
                                  // needs anything looser
};

// =====================================================================
// 1. The two known matrices
// =====================================================================
// A and B are filled with a simple deterministic pattern (row-major,
// square, N x N, float):
//
//     A[i][k] = (i + 2k + 1) / N
//     B[k][j] = (3k + j + 1) / N
//
// Why this pattern instead of random numbers:
//
//   * Reproducible. Every exercise, every run, every GPU sees the exact
//     same input, so a timing or correctness difference is never the
//     input's fault.
//   * The product has a closed form (see known_reference_element), so
//     verification is O(N^2) instead of an O(N^3) CPU GEMM. Checking a
//     4096x4096 result costs microseconds, not a minute.
//   * Entries stay small (<= ~3.0) and strictly positive, so the FP32
//     accumulation in the kernel suffers no catastrophic cancellation
//     and a tight tolerance stays meaningful.
//   * The pattern is NOT symmetric and NOT rank-1: the i-coefficient (2)
//     differs from the j-coefficient (3), so C is not symmetric. A
//     kernel that transposes an index, swaps A and B, or mixes up row
//     and column will produce a wrong value instead of accidentally
//     landing on the right one. Rank-1 or constant inputs would hide
//     exactly those bugs.
//   * When N is a power of two the division by N is exact in binary
//     floating point, so A and B hold their mathematically exact values
//     and the only error in C comes from the kernel's own summation.
inline void create_known_matrices(float* A, float* B, int N) {
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            A[i * N + k] = static_cast<float>(i + 2 * k + 1) * inv_n;
        }
    }
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            B[k * N + j] = static_cast<float>(3 * k + j + 1) * inv_n;
        }
    }
}

// =====================================================================
// 2. The closed-form product of those two matrices
// =====================================================================
// With a = i+1 and b = j+1:
//
//   C[i][j] = sum_k A[i][k]*B[k][j]
//           = (1/N^2) * sum_{k=0}^{N-1} (a + 2k)(3k + b)
//           = (1/N^2) * sum_k [ 3ak + ab + 6k^2 + 2kb ]
//           = (1/N^2) * [ 3a*S1 + 2b*S1 + ab*N + 6*S2 ]
//
// where S1 = sum k = N(N-1)/2 and S2 = sum k^2 = (N-1)N(2N-1)/6.
//
// Evaluated in double so the reference is far more accurate than the
// FP32 kernel result it is compared against.
inline double known_reference_element(int i, int j, int N) {
    const double n  = static_cast<double>(N);
    const double S1 = n * (n - 1.0) / 2.0;
    const double S2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    const double a  = static_cast<double>(i) + 1.0;
    const double b  = static_cast<double>(j) + 1.0;
    return (3.0 * a * S1 + 2.0 * b * S1 + a * b * n + 6.0 * S2) / (n * n);
}

// =====================================================================
// 3. The correctness check
// =====================================================================
// Compares C against the closed form with a mixed absolute/relative
// tolerance: |err| <= atol + rtol * |ref|.
//
// A relative tolerance is required because the kernels are free to sum
// the k-dimension in any order they like (that reordering is the whole
// point of tiling), and FP32 addition is not associative -- bit-exact
// agreement is not a legitimate expectation. The default rtol of 1e-3 is
// comfortably above the worst-case FP32 rounding drift for N up to a few
// thousand, while a genuine indexing bug is off by O(100%) and is caught
// immediately. The function prints the observed worst-case relative
// error so the real margin (typically ~1e-6) is always visible rather
// than hidden behind the threshold.
inline bool check_result(const float* C, int N,
                         double rtol = 1e-3, double atol = 1e-3) {
    double worst_rel = 0.0;
    int    worst_i = -1, worst_j = -1;
    double worst_ref = 0.0, worst_got = 0.0;
    long long bad = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double ref = known_reference_element(i, j, N);
            const double got = static_cast<double>(C[i * N + j]);
            const double err = std::fabs(ref - got);
            if (err > atol + rtol * std::fabs(ref)) ++bad;

            const double rel = err / (std::fabs(ref) > 0.0 ? std::fabs(ref) : 1.0);
            if (worst_i < 0 || rel > worst_rel) {
                worst_rel = rel;
                worst_i = i; worst_j = j;
                worst_ref = ref; worst_got = got;
            }
        }
    }

    if (bad == 0) {
        printf("  correctness : PASS  (max relative error %.3e at C[%d][%d]: "
               "expected %.6f, got %.6f)\n",
               worst_rel, worst_i, worst_j, worst_ref, worst_got);
        return true;
    }

    printf("  correctness : FAIL  (%lld of %lld elements outside tolerance)\n",
           bad, static_cast<long long>(N) * N);
    printf("                worst: C[%d][%d] expected %.6f, got %.6f "
           "(relative error %.3e)\n",
           worst_i, worst_j, worst_ref, worst_got, worst_rel);
    return false;
}

// Brute-force CPU GEMM in double precision. This is NOT used by the
// harness (the closed form replaces it) and is NOT a performance
// baseline -- it exists so the closed form itself can be cross-checked,
// and so a reader who distrusts the algebra above can verify it. O(N^3)
// single-threaded: only sane for small N.
inline void cpu_matmul_reference(const float* A, const float* B,
                                 double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<double>(A[i * N + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// =====================================================================
// 4. Reporting helpers
// =====================================================================
// A GEMM performs 2*N^3 flops: one multiply and one add per inner-loop
// step. GFLOP/s is the number to compare across the ladder -- elapsed
// milliseconds alone says nothing without the problem size.
inline double gflops(int N, double ms) {
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    return (flops / (ms / 1000.0)) / 1e9;
}

// Minimum global memory traffic any correct GEMM must move: read A, read
// B, write C, once each. Comparing this against the dram__bytes metric
// in Nsight Compute is how you see how much redundant traffic a kernel
// is generating.
inline double min_dram_bytes(int N) {
    return 3.0 * sizeof(float) * static_cast<double>(N) * N;
}

// =====================================================================
// 5. The shared harness
// =====================================================================
// Usage: <exercise> [N] [iterations]
//
// Note on the macro usage below: CUDA_CALL wraps every CUDA runtime call
// so that a DEBUG build reports the exact file and line of a failure. In
// a release build (no -DDEBUG) it expands to the bare call, which is
// precisely why timing runs are built without DEBUG -- no extra
// synchronization or branching around the calls being measured.
inline int run_exercise(int argc, char** argv, const ExerciseInfo& info) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 10;

    if (N <= 0 || iters <= 0) {
        printf("usage: %s [N] [iterations]   (N > 0, iterations > 0)\n", argv[0]);
        return 2;
    }
    if (info.size_multiple_of > 1 && (N % info.size_multiple_of) != 0) {
        printf("%s requires N to be a multiple of %d (got N = %d).\n",
               info.id, info.size_multiple_of, N);
        printf("This kernel is written without boundary checks on purpose so the\n"
               "optimization being demonstrated is not buried under guard branches.\n"
               "Try N = %d.\n",
               ((N + info.size_multiple_of - 1) / info.size_multiple_of) *
                   info.size_multiple_of);
        return 2;
    }

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    printf("=====================================================================\n");
    printf("%s : %s\n", info.id, info.title);
    printf("=====================================================================\n");
    printf("  device      : %s (compute capability %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  problem     : C = A * B, N = %d (%.1f MFLOP per call), %d timed iterations\n",
           N, 2.0 * N * N * N / 1e6, iters);

    const size_t bytes = sizeof(float) * static_cast<size_t>(N) * N;

    // --- host buffers and the two known matrices -----------------------
    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        printf("host allocation of %.1f MiB x3 failed\n", bytes / 1048576.0);
        std::free(h_A); std::free(h_B); std::free(h_C);
        return 1;
    }
    create_known_matrices(h_A, h_B, N);

    // --- device buffers ------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CALL(cudaMalloc(&d_A, bytes));
    CUDA_CALL(cudaMalloc(&d_B, bytes));
    CUDA_CALL(cudaMalloc(&d_C, bytes));

    CUDA_CALL(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    // Zero C so a kernel that fails to write some outputs produces an
    // obvious wrong answer instead of reading whatever was in memory.
    CUDA_CALL(cudaMemset(d_C, 0, bytes));

    // --- warm-up ------------------------------------------------------
    // The first launch pays for module loading and JIT/context setup.
    // Timing it would attribute several milliseconds of one-time cost to
    // the kernel, which is the single most common way to produce a
    // nonsense CUDA benchmark.
    launch_matmul(d_A, d_B, d_C, N);
    CUDA_CALL(cudaDeviceSynchronize());

    // --- timed loop ---------------------------------------------------
    // CUDA events are used rather than a host clock: they are recorded in
    // the stream on the device, so they measure device execution time
    // instead of launch overhead plus host-side scheduling noise.
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        launch_matmul(d_A, d_B, d_C, N);
    }
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&total_ms, start, stop));
    const double ms = static_cast<double>(total_ms) / iters;

    CUDA_CALL(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- report -------------------------------------------------------
    printf("  time        : %8.3f ms per call\n", ms);
    printf("  throughput  : %8.2f GFLOP/s\n", gflops(N, ms));
    printf("  ideal DRAM  : %8.2f MiB of unavoidable A+B+C traffic per call\n",
           min_dram_bytes(N) / 1048576.0);
    const bool ok = check_result(h_C, N, info.rtol);
    printf("\n");

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);

    return ok ? 0 : 1;
}
