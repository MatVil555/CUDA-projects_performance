#pragma once
// =====================================================================
// reduction_common.h
//
// Everything that is identical across the four reduction steps lives
// here, so each reduce_*.cu file contains nothing but the one idea it
// is teaching plus its launch configuration.
//
// WHY .h AND NOT .cuh: this header is included by BOTH the kernel
// translation units (compiled by nvcc) and main.cpp (compiled by the
// plain host compiler). Everything below is therefore host-compilable
// C++ -- declarations, host helpers and the runtime API -- except the
// one __device__ helper, which is fenced behind #ifdef __CUDACC__ so the
// host compiler never has to parse CUDA-only syntax. main.cpp needs the
// launcher declarations and the timing/reference helpers, nothing that
// runs on the device.
//
// Provided:
//   ReduceContext            the buffers a launcher needs, in one struct
//   launch_reduce_*()        one launcher per step (one reduce_*.cu each)
//   warp_reduce_sum()        the 5-step shuffle tree, used by steps 3
//                            and by the shared stage-2 kernel
//                            (device-only, nvcc sees it)
//   blocks_*()              grid-geometry helpers, shared by the
//                            launchers and by main's report
//   generate_input_and_exact_sum()  reproducible input with an EXACTLY
//                                   known answer
//   cpu_sum_fp32_naive() / cpu_sum_double()   the two host baselines
//   time_ms()               CUDA-event timing of a launcher
//
// Single-file version of this same ladder, if you prefer one blob:
//   ../Small_projects/reduction.cu
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------
// Error checking.
//
// Always on rather than behind #ifdef DEBUG: these wrap setup and
// teardown calls only, never anything inside a timed loop, so nothing
// being measured pays for them.
//
// The asynchronous caveat that catches everyone: a kernel launch reports
// only *launch* errors synchronously. A fault inside the kernel surfaces
// at the next synchronizing call, which is why cudaDeviceSynchronize and
// cudaMemcpy are wrapped too.
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
// Tunables.
//
// kBlock must be a power of two: steps 1 and 2 halve the active range
// each tree iteration and would silently drop elements otherwise. It is
// deliberately a compile-time constant rather than a runtime blockDim --
// that is what lets nvcc unroll the tree loops, and it has a measurable
// side effect documented in reduce_1_interleaved.cu.
// ---------------------------------------------------------------------
static constexpr int      kWarpSize    = 32;
static constexpr unsigned kFullMask    = 0xffffffffu;
static constexpr int      kBlock       = 256;   // steps 1-3 stage-1 block
static constexpr int      kFinalBlock  = 1024;  // shared stage-2 block
static constexpr int      kBlocksPerSM = 32;    // step 3 grid sizing

// ---------------------------------------------------------------------
// Grid geometry. Shared between the launchers and main's report so the
// two can never disagree about how many blocks were used.
// ---------------------------------------------------------------------

// Steps 1 and 2: one thread per element, so N dictates the grid.
inline int blocks_one_per_element(size_t N)
{
    return static_cast<int>((N + kBlock - 1) / kBlock);
}

// Step 3: size the grid to the *machine*, not to N, then let each thread
// absorb ~N/threads elements sequentially. Capped so a small N cannot
// ask for more blocks than there is work for.
inline int blocks_cascaded(size_t N, int sm_count)
{
    long long cap  = static_cast<long long>((N / 4 + kBlock - 1) / kBlock);
    long long want = static_cast<long long>(sm_count) * kBlocksPerSM;
    if (want > cap)  want = cap;
    if (want < 1)    want = 1;
    return static_cast<int>(want);
}

// ---------------------------------------------------------------------
// Sum across the 32 lanes of a warp; lane 0 ends up holding the total.
// Fully unrolled because every offset is a compile-time constant.
//
// Butterfly detail: __shfl_down_sync with an out-of-range source lane
// returns the caller's own value, so lanes 16..31 accumulate garbage
// after the first step. That garbage never reaches lane 0, because each
// later step only pulls from lanes below the previous cut. Only lane 0's
// result is ever used.
//
// Device-only, so it is fenced off from the host compiler: __CUDACC__ is
// defined only while nvcc is compiling a .cu file. main.cpp includes this
// header but never calls this, so it loses nothing.
// ---------------------------------------------------------------------
#ifdef __CUDACC__
__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(kFullMask, v, offset);
    }
    return v;
}
#endif  // __CUDACC__

// ---------------------------------------------------------------------
// One struct so every launcher has the same signature and main can hold
// them in a table.
// ---------------------------------------------------------------------
struct ReduceContext {
    const float*   d_x        = nullptr;  // input, N floats
    const float*   d_ones     = nullptr;  // N floats of 1.0f, step 4b only
    float*         d_partials = nullptr;  // one float per stage-1 block
    float*         d_result   = nullptr;  // the single output float
    size_t         N          = 0;
    int            sm_count   = 0;
    cublasHandle_t cublas     = nullptr;
};

// Each launcher performs a COMPLETE reduction of c.d_x into c.d_result.
// Implemented one per file, reduce_*.cu.
void launch_reduce_l1(const ReduceContext& c);        // interleaved, divergent
void launch_reduce_l2(const ReduceContext& c);        // sequential, divergence-free
void launch_reduce_l3(const ReduceContext& c);        // cascade + shuffles
void launch_reduce_l4_asum(const ReduceContext& c);   // cuBLAS Sasum
void launch_reduce_l4_dot(const ReduceContext& c);    // cuBLAS Sdot vs ones

// Shared plumbing, not a step of the ladder: collapses the per-block
// partials to one value. See reduce_final.cu for why this exists.
void launch_reduce_final(const float* d_partials, int n, float* d_result);

// Fills n floats with 1.0f. Lives with its only consumer, step 4b.
void fill_ones(float* d_p, size_t n);

// =====================================================================
// INPUT GENERATION AND THE EXACT REFERENCE
// =====================================================================
// A deterministic integer bit-mixer, not rand(): reproducible across
// machines and libc versions, and fast enough that filling 32.7M
// elements does not dominate startup.
//
// Every element is q/256 with q an integer in [0,512), which buys three
// things at once:
//   * the value is exactly representable in FP32 (9 mantissa bits);
//   * values average ~1.0, so the running total climbs past 2^24 and
//     exposes FP32 accumulation limits (see cpu_sum_fp32_naive);
//   * the exact sum is an integer multiple of 1/256, computable in pure
//     integer arithmetic.
//
// Why that last point matters: validating a reduction against a
// floating-point CPU sum is circular, because you end up comparing two
// differently-wrong numbers. Here the reference has *zero* error, so the
// per-step errors main prints are meaningful in absolute terms.
// =====================================================================
inline uint32_t hash_u32(uint32_t v)
{
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

// Fills x and returns the EXACT sum. The uint64 accumulator cannot
// overflow (N * 511 stays far below 2^64) and the final divide by a
// power of two is exact in double, so the returned value is the true
// sum to the last bit.
inline double generate_input_and_exact_sum(float* x, size_t N)
{
    uint64_t q_total = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint32_t q = hash_u32(static_cast<uint32_t>(i)) & 511u;
        q_total += q;
        x[i] = static_cast<float>(q) * (1.0f / 256.0f);
    }
    return static_cast<double>(q_total) / 256.0;
}

// ---------------------------------------------------------------------
// The naive host sum: a cautionary exhibit, not a reference.
//
// A float has 24 significand bits, so near a running total of T the gap
// between representable values is about T * 2^-23. Our total reaches
// ~32.7M, where that gap is exactly 2.0 -- so adding an element of ~1.0
// rounds to nearest-even and contributes 0 or 2, never 1. The total
// crosses 2^24 = 16,777,216 at element 16,810,014, i.e. 51.3% of the way
// in, so half the array is accumulated in this degraded regime.
//
// Measured at the default N against the exact reference:
//     sequential FP32 : rel err 1.23e-05
//     the GPU tree    : rel err 1.30e-08   -> ~940x more accurate
//
// Worth stating the size honestly: the damage is ~1e-5, not the tens of
// percent you might guess from "half the additions are wrong". Round-to-
// nearest is unbiased, so errors are as often +1 as -1 and largely
// cancel in a random walk. The structural point still holds: sequential
// error grows O(N), a balanced tree only O(log N), because the tree
// always adds operands of comparable magnitude and never lets a small
// addend be swallowed by a huge accumulator.
//
// So the parallel algorithm here is both faster AND more accurate -- a
// rare case where the two goals point the same direction.
// ---------------------------------------------------------------------
inline float cpu_sum_fp32_naive(const float* x, size_t N)
{
    float s = 0.0f;
    for (size_t i = 0; i < N; ++i) s += x[i];
    return s;
}

// Sequential accumulation in double -- what you should reach for by
// default. At this problem size it is not merely accurate but provably
// *exact*: every input is a multiple of 2^-8, so every partial sum is
// too, and the largest is under 2^25, needing at most 25 + 8 = 33
// significand bits. Double has 53, so no addition rounds at all and the
// measured relative error is 0.0e+00.
//
// By the same argument FP32's 24 bits stop sufficing once the total
// passes 2^24 * 2^-8 = 65,536, which is why the float loop above
// degrades so early.
//
// This exactness is a property of this input, not of double in general.
inline double cpu_sum_double(const float* x, size_t N)
{
    double s = 0.0;
    for (size_t i = 0; i < N; ++i) s += static_cast<double>(x[i]);
    return s;
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
//     cuBLAS additionally allocates workspace and picks kernels on its
//     first call. Charging the kernel for milliseconds of one-time cost
//     is the most common way to produce a nonsense CUDA timing.
//   * TIME A LOOP, NOT A CALL. One event pair around `iters` launches
//     amortizes launch overhead and event resolution, and for the
//     two-stage steps it correctly charges each iteration for BOTH
//     kernels -- which is what a caller actually pays.
// =====================================================================
typedef void (*ReduceLaunchFn)(const ReduceContext&);

inline double time_ms(ReduceLaunchFn fn, const ReduceContext& c, int iters)
{
    for (int i = 0; i < 5; ++i) fn(c);            // warm-up, discarded
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
