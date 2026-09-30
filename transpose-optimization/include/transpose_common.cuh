#pragma once
// =====================================================================
// transpose_common.cuh
//
// Everything identical across the four transpose steps lives here, so
// each src/transpose_*.cu file contains nothing but the one idea it is
// teaching plus its launch configuration.
//
// Provided:
//   TransposeContext         the buffers a launcher needs, in one struct
//   launch_transpose_*()     one launcher per step (defined in src/)
//   launch_copy_reference()  the copy upper bound (not a transpose)
//   naive_grid() / tile_grid()  grid geometry, shared by the launchers
//                            and by main's report so they cannot disagree
//   generate_input()         reproducible, self-labelling input
//   transpose_cpu_reference() the correctness oracle
//   verify()                 bit-exact comparison (see the note below)
//   time_ms()                CUDA-event timing of a launcher
//
// Single-file version of this same ladder, if you prefer one blob:
//   ../Small_projects/transpose.cu
//
// ---------------------------------------------------------------------
// THE PROBLEM IN ONE PARAGRAPH
// ---------------------------------------------------------------------
// B[j][i] = A[i][j] performs exactly ZERO floating-point operations.
// Arithmetic intensity is 0.0 flop/byte, so there is no arithmetic to
// optimize, no reuse to exploit and no algorithmic choice to make --
// every correct implementation moves exactly the same bytes. Any
// difference between the four steps is therefore caused purely by memory
// access pattern, which makes this an unusually honest teacher.
//
// The awkwardness is inherent: element (i,j) sits at A[i*N+j] and must
// land at B[j*M+i]. Walking A along a row is contiguous; writing those
// same values into B walks a column, striding by M floats. You can choose
// which side pays, but not whether anyone pays.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------
// Error checking. Always on rather than behind #ifdef DEBUG; applied to
// setup and teardown only, never inside a timed loop, so nothing being
// measured pays for it.
//
// The asynchronous caveat: a kernel launch reports only *launch* errors
// synchronously. A fault inside the kernel surfaces at the next
// synchronizing call, which is why cudaDeviceSynchronize and cudaMemcpy
// are wrapped too.
// ---------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",            \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        const cublasStatus_t st_ = (call);                                    \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                \
                         static_cast<int>(st_), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaPeekAtLastError())

// ---------------------------------------------------------------------
// Tile geometry.
//
// kTileDim = 32 is NOT a free parameter. It must equal the warp size so
// that a warp's 32 lanes cover exactly one tile row -- that is what makes
// a tile row one coalesced 128-byte global transaction, and it is also
// what makes the bank arithmetic in steps 2 and 3 work out as described.
//
// kBlockRows = 8 gives a 32x8 = 256-thread block where each thread moves
// kTileDim/kBlockRows = 4 elements. Why not 32x32 with one element each?
//   * 1024 threads is the hardware maximum per block, which caps resident
//     blocks per SM at one and leaves the scheduler no room.
//   * Four elements per thread amortizes index arithmetic and boundary
//     checks over four transfers instead of one.
//   * It gives each thread four independent loads to issue back to back,
//     overlapping memory latency within a thread rather than relying
//     entirely on other warps -- free instruction-level parallelism.
// ---------------------------------------------------------------------
static constexpr int kTileDim   = 32;
static constexpr int kBlockRows = 8;

// ---------------------------------------------------------------------
// Grid geometry, shared between the launchers and main's report.
// ---------------------------------------------------------------------

// Step 1: one thread per element, block 32x8. Deliberately the same
// thread count as the tiled steps so the comparison is not confounded by
// block size.
inline dim3 naive_block() { return dim3(kTileDim, kBlockRows); }
inline dim3 naive_grid(int M, int N)
{
    return dim3(static_cast<unsigned>((N + kTileDim   - 1) / kTileDim),
                static_cast<unsigned>((M + kBlockRows - 1) / kBlockRows));
}

// Steps 2 and 3: one block per 32x32 tile. grid.x spans A's columns,
// grid.y spans A's rows.
inline dim3 tile_block() { return dim3(kTileDim, kBlockRows); }
inline dim3 tile_grid(int M, int N)
{
    return dim3(static_cast<unsigned>((N + kTileDim - 1) / kTileDim),
                static_cast<unsigned>((M + kTileDim - 1) / kTileDim));
}

// ---------------------------------------------------------------------
// One struct so every launcher has the same signature and main can hold
// them in a table.
// ---------------------------------------------------------------------
struct TransposeContext {
    const float*   d_A      = nullptr;   // input,  M x N row-major
    float*         d_B      = nullptr;   // output, N x M row-major
    int            M        = 0;
    int            N        = 0;
    int            sm_count = 0;
    cublasHandle_t cublas   = nullptr;
};

// Each launcher performs one COMPLETE operation d_A -> d_B.
void launch_transpose_l1(const TransposeContext& c);   // naive, scattered writes
void launch_transpose_l2(const TransposeContext& c);   // tiled, bank conflicts
void launch_transpose_l3(const TransposeContext& c);   // tiled + padding
void launch_transpose_l4(const TransposeContext& c);   // cuBLAS Sgeam

// Reference point, NOT a transpose: a straight copy moving identical
// traffic. See src/copy_reference.cu for why it earns a row in the table.
void launch_copy_reference(const TransposeContext& c);

// =====================================================================
// INPUT GENERATION
// =====================================================================
// A[i][j] = (i*N + j) & 0xFFFFFF, row-major.
//
// Deterministic (no rand(), so reproducible across machines), and every
// value is an integer below 2^24 and therefore *exactly* representable in
// FP32. So the whole pipeline is free of rounding and the output can be
// required to match bit for bit.
//
// The masking preserves exactness for any M*N the user might pass; at the
// default size M*N = 2^23, so no masking occurs and all 8.4M values are
// distinct. Uniqueness is what makes verification strong: every element
// effectively carries a label saying where it came from, so a misindexed
// kernel cannot accidentally produce a plausible answer. A swapped axis,
// an off-by-one tile offset, a wrong leading dimension or a dropped edge
// tile each land a provably wrong, identifiable number in a known place.
// =====================================================================
inline void generate_input(float* A, int M, int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const uint32_t v = (static_cast<uint32_t>(i) * static_cast<uint32_t>(N)
                                + static_cast<uint32_t>(j)) & 0xFFFFFFu;
            A[static_cast<size_t>(i) * N + j] = static_cast<float>(v);
        }
    }
}

// ---------------------------------------------------------------------
// Host reference. A is M x N row-major; B is N x M row-major.
//
// The obvious nested loop, deliberately not cache-blocked: this is the
// correctness oracle, and the clearest possible expression of
// "B(j,i) = A(i,j)" is worth more here than host speed. It is also a
// demonstration of the very problem the GPU has -- the inner loop strides
// through B by M floats and thrashes the CPU cache for exactly the reason
// step 1 thrashes DRAM.
// ---------------------------------------------------------------------
inline void transpose_cpu_reference(const float* A, float* B, int M, int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            B[static_cast<size_t>(j) * M + i] = A[static_cast<size_t>(i) * N + j];
        }
    }
}

// =====================================================================
// VERIFICATION -- WHY BIT-EXACT IS THE RIGHT TEST HERE
// =====================================================================
// A transpose performs no arithmetic, so it has no rounding, so the
// output must match the reference in every bit. This is one of the few
// kernels where a tolerance check is the wrong instinct: a nonzero
// tolerance quietly passing a transpose is hiding a bug, not absorbing
// floating-point noise.
//
// So two numbers are reported, answering different questions:
//   exact_mismatches : elements differing in ANY bit. Must be 0. The
//                      real test.
//   max_abs_diff     : largest absolute difference, checked against a
//                      tolerance. Kept as a sensible net for the cuBLAS
//                      path (which technically computes 1.0*x + 0.0*y),
//                      but on a kernel with zero flops any nonzero value
//                      means a bug.
//
// Non-finite outputs are counted separately, because NaN compares false
// against everything -- without an explicit test a NaN slips past a naive
// difference check. NaN is also the exact signature of the
// 0.0f * uninitialized hazard described in transpose_4_cublas.cu.
// =====================================================================
struct VerifyResult {
    bool      ok           = false;
    long long exact_mismatches = 0;
    long long nonfinite    = 0;
    double    max_abs_diff = 0.0;
    size_t    first_bad    = static_cast<size_t>(-1);
};

inline VerifyResult verify(const float* got, const float* ref, size_t n, double atol)
{
    VerifyResult r;
    r.ok = true;

    for (size_t k = 0; k < n; ++k) {
        const float g = got[k], e = ref[k];

        if (!std::isfinite(g)) {
            ++r.nonfinite;
            r.ok = false;
            if (r.first_bad == static_cast<size_t>(-1)) r.first_bad = k;
            continue;
        }
        if (g != e) {                       // bit-exact comparison
            ++r.exact_mismatches;
            if (r.first_bad == static_cast<size_t>(-1)) r.first_bad = k;
        }
        const double d = std::fabs(static_cast<double>(g) - static_cast<double>(e));
        if (d > r.max_abs_diff) r.max_abs_diff = d;
    }

    if (r.exact_mismatches != 0 || r.max_abs_diff > atol) r.ok = false;
    return r;
}

// =====================================================================
// TIMING
// =====================================================================
// CUDA events, not a host clock: they are recorded in the stream on the
// device, so they measure device execution rather than host scheduling
// noise.
//
// Two details that separate a benchmark from a number:
//   * WARM-UP. The first launch pays module load, JIT and context setup;
//     cuBLAS additionally allocates workspace and selects a kernel on its
//     first call. Charging a kernel for milliseconds of one-time cost is
//     the most common way to produce a nonsense CUDA measurement.
//   * TIME A LOOP, NOT A CALL. One event pair around `iters` launches
//     amortizes the few-microsecond launch overhead and the event
//     resolution itself. At the default size a good transpose takes only
//     a few hundred microseconds, so launch overhead is not negligible
//     relative to a single call.
// =====================================================================
typedef void (*TransposeLaunchFn)(const TransposeContext&);

inline double time_ms(TransposeLaunchFn fn, const TransposeContext& c, int iters)
{
    for (int i = 0; i < 5; ++i) fn(c);             // warm-up, discarded
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAUNCH();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn(c);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(total_ms) / iters;
}
