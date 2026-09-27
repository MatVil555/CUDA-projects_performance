// =====================================================================
// main.cpp -- row-wise softmax, three kernels, driven from one host file.
//
// This is a split version of the original single-file softmax.cu. The
// three hand-written GPU kernels now live in their own translation
// units, each compiled by nvcc because they contain <<<>>> launch
// syntax:
//
//     softmax_l1_naive.cu       Level 1: one thread per row
//     softmax_l2_shared.cu      Level 2: one block per row, shared-mem tree
//     softmax_l3_warp_online.cu Level 3: one warp per row, online softmax
//
// This file contains no kernel-launch syntax at all -- it only calls the
// plain host functions declared in softmax_kernels.h (launch_softmax_l1/
// l2/l3), plus ordinary CUDA runtime API calls (cudaMalloc, cudaMemcpy,
// cudaEventRecord, ...), which are just C functions. That means this
// file can be compiled with an ordinary C++ compiler and linked against
// libcudart, or compiled by nvcc alongside the .cu files -- either way
// works; the Makefile below uses nvcc for both, which is the simplest
// option.
//
// NOTE: the original file also included a 4th level (a cuBLAS
// Sgemv+Sdgmm hybrid) purely as a "the vendor library loses here"
// teaching contrast. It has been dropped here since the ask was for
// exactly the three hand-written kernels; say the word if you'd like it
// back as a 4th file (softmax_l4_cublas.cu) with its own launcher.
//
// Build and run:
//     nvcc -O3 main.cpp softmax_l1_naive.cu softmax_l2_shared.cu \
//          softmax_l3_warp_online.cu -o softmax
//     ./softmax [M] [N] [iterations]
// Defaults:  M = 8192 rows, N = 4096 columns, 20 timed iterations
//
// ---------------------------------------------------------------------
// WHY SOFTMAX IS AN INTERESTING KERNEL
// ---------------------------------------------------------------------
// Softmax is not GEMM. GEMM is compute bound: it does O(N^3) work on
// O(N^2) data, so the game is keeping the FMA pipes fed. Softmax does
// O(1) work per element on O(1) data per element, so the game is
// entirely different:
//
//   Unavoidable traffic : read X once, write Y once = 8 bytes / element
//   Unavoidable math    : ~1 exp + 1 multiply / element
//
// That is an arithmetic intensity well under 1 flop per byte. On any
// modern GPU (roofline ridge point around 10-40 flop/byte) this is
// *deeply* memory bound. No amount of clever arithmetic will help. The
// only thing that matters is:
//
//     (1) how many times you touch DRAM, and
//     (2) whether each touch is coalesced.
//
// So this is a ladder about *memory passes*, not about math:
//
//   Level 1  naive     one thread per row     4 passes, uncoalesced
//   Level 2  tiled     one block per row      4 passes, coalesced
//   Level 3  online    one warp per row       3 passes, coalesced, no smem
//
// ---------------------------------------------------------------------
// NUMERICAL STABILITY: WHY THE MAX IS SUBTRACTED
// ---------------------------------------------------------------------
// The textbook definition S_ij = exp(x_ij) / sum_k exp(x_ik) is
// unusable in FP32. The exact boundaries, measured:
//
//     expf(88.0)  = 1.65e38   (finite, just under FLT_MAX = 3.40e38)
//     expf(88.8)  = inf       <-- overflow
//     expf(-87.0) = 1.65e-38  (finite, just above FLT_MIN = 1.18e-38)
//     expf(-103)  = 1.40e-45  <-- denormal, ~1 significant bit left
//     expf(-104)  = 0         <-- flushed to exact zero
//
// So the usable input window for a bare expf is roughly (-87, +88.7).
// The fix is the algebraic identity
//
//     exp(x_j) / sum_k exp(x_k) == exp(x_j - c) / sum_k exp(x_k - c)
//
// for any constant c. Choosing c = m_i = max_k(x_ik) makes the largest
// exponent argument exactly 0, so nothing overflows and the denominator
// is always >= 1. The cost is an extra reduction pass to find m_i before
// you can sum -- getting rid of that extra pass is precisely what
// Level 3 does.
//
// generate_logits() below deliberately biases every 4th row by +95 to
// exercise this: a non-stable implementation returns NaN on this input;
// all three levels here return finite, correct values.
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "softmax_kernels.h"

// ---------------------------------------------------------------------
// Error checking.
//
// Always on, not hidden behind #ifdef DEBUG, because this file is meant
// to be run once and a silent failure there costs more than the branch
// does. Used on setup/teardown calls only, never inside the timed loop.
//
// The asynchronous caveat that trips everyone up: a kernel launch only
// reports *launch* configuration errors synchronously. A fault inside
// the kernel surfaces at the next synchronizing call, which is why
// cudaDeviceSynchronize and cudaMemcpy are also wrapped.
// ---------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",             \
                         cudaGetErrorName(err_), __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Call after a kernel launch to catch bad launch configurations.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaPeekAtLastError())

// =====================================================================
// INPUT GENERATION
// =====================================================================
// A deterministic integer hash, not rand(). Three reasons:
//
//   * Reproducible across runs, machines and libc implementations, so a
//     timing or correctness difference is never the input's fault.
//   * No host RNG call per element -- filling 33.5M floats with rand()
//     is slow enough to notice.
//   * The bit pattern is well mixed, so adjacent elements are
//     uncorrelated and no kernel can accidentally benefit from
//     structure in the data.
//
// Rows are biased by {0, +30, -30, +95, -95} cycling with the row index:
// rows 3, 8, 13, ... span [+87, +103] and overflow an unstable exp;
// rows 4, 9, 14, ... span [-103, -87] and go denormal instead. Every row
// has a different max, so a kernel that computes one global max instead
// of a per-row max is caught immediately.
// =====================================================================
__host__ inline uint32_t hash_u32(uint32_t v)
{
    // Fast integer bit-mixer (the "lowbias32" finalizer).
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

void generate_logits(float* x, int M, int N)
{
    static const float kRowBias[5] = { 0.0f, 30.0f, -30.0f, 95.0f, -95.0f };

    for (int i = 0; i < M; ++i) {
        const float bias = kRowBias[i % 5];
        float* xr = x + static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            const uint32_t h = hash_u32(static_cast<uint32_t>(i) * 2654435761u +
                                        static_cast<uint32_t>(j));
            // Top 24 bits -> [0,1) exactly representable in FP32.
            const float u = static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
            xr[j] = (u * 16.0f - 8.0f) + bias;   // [-8,8) + row bias
        }
    }
}

// =====================================================================
// HOST REFERENCE
// =====================================================================
// The numerically stable three-pass formulation, accumulated in double,
// so that any disagreement with a GPU level is the kernel's fault and
// not a coin flip between two equally wrong FP32 numbers. The GPU
// kernels all sum the row in a different order -- that reordering is
// the entire point of a parallel reduction -- and FP32 addition is not
// associative, so bit-exact agreement is not a legitimate expectation.
// Hence the tolerance check in verify().
// =====================================================================
void softmax_cpu_reference(const float* x, float* out, int M, int N)
{
    for (int i = 0; i < M; ++i) {
        const float* xr = x + static_cast<size_t>(i) * N;
        float* orow = out + static_cast<size_t>(i) * N;

        // Pass 1: row max, for the stability shift.
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            if (xr[j] > m) m = xr[j];
        }

        // Pass 2: denominator, shifted so the largest term is exp(0)=1.
        double denom = 0.0;
        for (int j = 0; j < N; ++j) {
            denom += std::exp(static_cast<double>(xr[j]) - static_cast<double>(m));
        }

        // Pass 3: normalize. Reciprocal once, then multiply.
        const double inv = 1.0 / denom;
        for (int j = 0; j < N; ++j) {
            orow[j] = static_cast<float>(
                std::exp(static_cast<double>(xr[j]) - static_cast<double>(m)) * inv);
        }
    }
}

// =====================================================================
// VERIFICATION
// =====================================================================
// A mixed absolute/relative tolerance: |err| <= atol + rtol * |ref|.
//
// Relative alone is wrong here. Softmax outputs span many orders of
// magnitude -- with a +/-8 logit spread the smallest entry is ~1e-7 of
// the largest -- and demanding 1e-4 relative accuracy on a value of
// 1e-11 asks for more precision than FP32 has anywhere near that
// exponent. Absolute alone is equally wrong: with N = 4096 the typical
// output is ~2.4e-4, so a generous absolute tolerance would happily
// accept a kernel that got every value wrong by 50%.
//
// NaN and inf are rejected explicitly -- they are the signature of a
// broken stability shift, and comparisons against NaN are false, so
// without an explicit test a NaN can slip through a naive check.
// =====================================================================
struct VerifyResult {
    bool   ok;
    double max_abs_err;
    double max_rel_err;
    long long bad;
    long long nonfinite;
};

VerifyResult verify(const float* got, const float* ref, size_t n,
                    double rtol = 1e-4, double atol = 1e-9)
{
    VerifyResult r{ true, 0.0, 0.0, 0, 0 };

    for (size_t i = 0; i < n; ++i) {
        const double g = static_cast<double>(got[i]);
        const double e = static_cast<double>(ref[i]);

        if (!std::isfinite(g)) { ++r.nonfinite; r.ok = false; continue; }

        const double abs_err = std::fabs(g - e);
        if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;

        const double denom = std::fabs(e) > 0.0 ? std::fabs(e) : 1.0;
        const double rel   = abs_err / denom;
        if (rel > r.max_rel_err) r.max_rel_err = rel;

        if (abs_err > atol + rtol * std::fabs(e)) { ++r.bad; r.ok = false; }
    }
    return r;
}

// =====================================================================
// TIMING
// =====================================================================
// CUDA events, not a host timer. Events are recorded in the stream on
// the device, so they measure device execution and not host-side launch
// scheduling noise.
//
//   * WARM-UP. The first launch pays for module load and JIT setup --
//     milliseconds of one-time cost which, attributed to the kernel,
//     would be pure fiction. Discard it.
//   * TIMING A LOOP, NOT A CALL. One event pair around `iters` launches
//     amortizes the ~5us launch overhead and event resolution.
// =====================================================================
template <typename LaunchFn>
double time_ms(LaunchFn&& launch, int iters)
{
    for (int i = 0; i < 3; ++i) launch();          // warm-up, discarded
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAUNCH();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(total_ms) / iters;
}

// Per-variant results for the summary table.
struct Level {
    const char* name;
    int         passes;      // algorithmic global-memory passes over the matrix
    double      ms;
    VerifyResult v;
};

int main(int argc, char** argv)
{
    // ---------------- configuration ----------------
    const int M     = (argc > 1) ? std::atoi(argv[1]) : 8192;
    const int N     = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 20;

    if (M <= 0 || N <= 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s [M] [N] [iterations]  (all > 0)\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(N);
    const size_t bytes = elems * sizeof(float);

    // ---------------- device report ----------------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int mem_clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    // DDR -> 2 transfers per clock. Reported in kHz and bits.
    const double peak_gbs =
        2.0 * static_cast<double>(mem_clk_khz) * 1e3 * (bus_bits / 8.0) / 1e9;

    std::printf("=======================================================================\n");
    std::printf(" Row-wise softmax: three levels of optimization (split build)\n");
    std::printf("=======================================================================\n");
    std::printf("  device        : %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("  peak DRAM BW  : %.1f GB/s (%d-bit bus @ %.0f MHz, theoretical)\n",
                peak_gbs, bus_bits, mem_clk_khz / 1000.0);
    std::printf("  problem       : M = %d rows x N = %d cols (%.1f MiB per matrix)\n",
                M, N, bytes / 1048576.0);
    std::printf("  minimum traffic: %.1f MiB per softmax (read X once + write Y once)\n",
                2.0 * bytes / 1048576.0);
    std::printf("  iterations    : %d timed (plus 3 warm-up, discarded)\n\n", iters);

    // ---------------- host buffers ----------------
    float* h_X   = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    float* h_got = static_cast<float*>(std::malloc(bytes));
    if (!h_X || !h_ref || !h_got) {
        std::fprintf(stderr, "host allocation failed (%.1f MiB x 3)\n", bytes / 1048576.0);
        return EXIT_FAILURE;
    }

    std::printf("  generating logits ... ");
    std::fflush(stdout);
    generate_logits(h_X, M, N);
    std::printf("done (rows biased by {0,+30,-30,+95,-95}; the +95 rows overflow\n");
    std::printf("                     FP32 exp unless the row max is subtracted)\n");

    std::printf("  CPU reference  ... ");
    std::fflush(stdout);
    softmax_cpu_reference(h_X, h_ref, M, N);
    std::printf("done\n\n");

    // ---------------- device buffers ----------------
    float *d_X = nullptr, *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

    // Each lambda performs one complete softmax X -> Y, using the
    // kernel-launcher functions declared in softmax_kernels.h. No
    // <<<>>> syntax appears in this file.
    auto run_l1 = [&] { launch_softmax_l1(d_X, d_Y, M, N); };
    auto run_l2 = [&] { launch_softmax_l2(d_X, d_Y, M, N); };
    auto run_l3 = [&] { launch_softmax_l3(d_X, d_Y, M, N); };

    Level levels[3] = {
        { "1. Naive (thread/row, uncoalesced)", 4, 0.0, {} },
        { "2. Tiled (block/row, shared tree)",  4, 0.0, {} },
        { "3. Online (warp/row, shuffles)",     3, 0.0, {} },
    };

    // ---------------- correctness, then timing ----------------
    // Correctness first, on a zeroed output buffer so a kernel that
    // fails to write part of Y produces obvious garbage (0 vs ~2.4e-4)
    // rather than reading back a previous level's correct answer -- a
    // trap that makes a broken kernel look fine.
    for (int lv = 0; lv < 3; ++lv) {
        CUDA_CHECK(cudaMemset(d_Y, 0, bytes));
        switch (lv) {
            case 0: run_l1(); break;
            case 1: run_l2(); break;
            case 2: run_l3(); break;
        }
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_got, d_Y, bytes, cudaMemcpyDeviceToHost));

        levels[lv].v = verify(h_got, h_ref, elems);

        std::printf("  [%s] %s", levels[lv].v.ok ? "PASS" : "FAIL", levels[lv].name);
        if (!levels[lv].v.ok) {
            std::printf("  -- %lld outside tolerance, %lld non-finite",
                        levels[lv].v.bad, levels[lv].v.nonfinite);
        }
        std::printf("\n");

        switch (lv) {
            case 0: levels[lv].ms = time_ms(run_l1, iters); break;
            case 1: levels[lv].ms = time_ms(run_l2, iters); break;
            case 2: levels[lv].ms = time_ms(run_l3, iters); break;
        }
    }

    // ---------------- report ----------------
    // Two bandwidth columns, because one is not enough to tell the story:
    //
    //   "useful"     = 2*bytes / time. The traffic an ideal softmax must
    //                  move -- the roofline-honest metric.
    //   "issued"     = passes*bytes / time. The traffic the algorithm
    //                  actually asks for.
    const double base_ms = levels[0].ms;

    std::printf("\n");
    std::printf("=======================================================================\n");
    std::printf(" Level                                ms    useful   issued   vs L1\n");
    std::printf("                                            GB/s     GB/s          \n");
    std::printf("-----------------------------------------------------------------------\n");
    for (int lv = 0; lv < 3; ++lv) {
        const double secs      = levels[lv].ms / 1e3;
        const double useful_gbs = (2.0 * static_cast<double>(bytes)) / secs / 1e9;
        const double issued_gbs =
            (static_cast<double>(levels[lv].passes) * static_cast<double>(bytes)) / secs / 1e9;
        std::printf(" %-34s %7.3f  %7.1f  %7.1f  %5.2fx\n",
                    levels[lv].name, levels[lv].ms, useful_gbs, issued_gbs,
                    base_ms / levels[lv].ms);
    }
    std::printf("-----------------------------------------------------------------------\n");
    std::printf(" passes over the matrix: L1=4, L2=4, L3=3\n");
    std::printf(" theoretical peak: %.1f GB/s -- 'useful' is capped near\n", peak_gbs);
    std::printf(" (2/passes) x peak, which is why the pass count is the whole game.\n");
    std::printf("=======================================================================\n\n");

    std::printf(" Accuracy (worst case over %.1fM elements, vs double-precision CPU):\n",
                elems / 1e6);
    for (int lv = 0; lv < 3; ++lv) {
        std::printf("   %-34s abs %.3e   rel %.3e   %s\n",
                    levels[lv].name, levels[lv].v.max_abs_err, levels[lv].v.max_rel_err,
                    levels[lv].v.ok ? "ok" : "FAILED");
    }
    std::printf("\n");
    std::printf(" Reminder: every level subtracts the row max before calling expf. On the\n");
    std::printf(" rows biased to +95 (range [87,103]), an unstable exp(x)/sum(exp(x)) would\n");
    std::printf(" overflow most terms to +inf (3709 of 4096 at the default N), giving\n");
    std::printf(" inf/inf = NaN. Finite, accurate values above are that subtraction working.\n\n");

    // ---------------- cleanup ----------------
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    std::free(h_X);
    std::free(h_ref);
    std::free(h_got);

    bool all_ok = true;
    for (int lv = 0; lv < 3; ++lv) all_ok = all_ok && levels[lv].v.ok;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
