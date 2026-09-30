// =====================================================================
// main.cu -- driver for the transpose optimization ladder
//
//   ./transpose_bench [M] [N] [iterations]
//   defaults: M = 4096, N = 2048, 50 timed iterations
//
// Responsibilities, all deliberately kept OUT of the kernel files so each
// of those contains one idea:
//   * generate the self-labelling input and the host reference
//   * verify every step BEFORE timing it, bit-exactly
//   * time each step and report ms, GB/s, % of peak and % of the copy
//
// THE CEILING THIS IS MEASURED AGAINST.
// A transpose performs exactly zero floating-point operations, so it is a
// pure data-movement problem:
//
//     traffic = 2 * M * N * 4 bytes   (read once, write once)
//     effective bandwidth = traffic / elapsed
//
// Every row of the table moves the same bytes, so every difference between
// them is caused by access pattern alone. The copy row is the empirical
// ceiling -- see src/copy_reference.cu for why it matters more than the
// datasheet peak.
// =====================================================================

#include "transpose_common.cuh"

struct Step {
    const char*       name;
    TransposeLaunchFn fn;
    bool              is_transpose;   // false for the copy reference point
    double            ms;
    VerifyResult      v;
};

int main(int argc, char** argv)
{
    // ---------------- configuration ----------------
    const int M     = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int N     = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 50;

    if (M <= 0 || N <= 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s [M] [N] [iterations]  (all > 0)\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(N);
    const size_t bytes = elems * sizeof(float);
    const double traffic_bytes = 2.0 * static_cast<double>(bytes);

    // ---------------- device report ----------------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int mem_clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    const double peak_gbs =
        2.0 * static_cast<double>(mem_clk_khz) * 1e3 * (bus_bits / 8.0) / 1e9;

    std::printf("=======================================================================\n");
    std::printf(" Matrix transpose: four optimization steps\n");
    std::printf("=======================================================================\n");
    std::printf("  device        : %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("  peak DRAM BW  : %.1f GB/s (%d-bit bus @ %.0f MHz, theoretical)\n",
                peak_gbs, bus_bits, mem_clk_khz / 1000.0);
    std::printf("  problem       : A is %d x %d -> B is %d x %d (%.1f MiB each)\n",
                M, N, N, M, bytes / 1048576.0);
    std::printf("  traffic       : %.1f MiB per transpose (read once + write once)\n",
                traffic_bytes / 1048576.0);
    std::printf("  arithmetic    : 0 flops. This is a pure memory benchmark.\n");
    std::printf("  ideal time    : %.3f ms at theoretical peak bandwidth\n",
                (traffic_bytes / (peak_gbs * 1e9)) * 1e3);
    std::printf("  tile          : %dx%d, block %dx%d (%d threads, %d elems/thread)\n",
                kTileDim, kTileDim, kTileDim, kBlockRows,
                kTileDim * kBlockRows, kTileDim / kBlockRows);
    std::printf("  shared/block  : step 2 %d B (32x32), step 3 %d B (32x33 padded)\n",
                static_cast<int>(kTileDim * kTileDim * sizeof(float)),
                static_cast<int>(kTileDim * (kTileDim + 1) * sizeof(float)));
    std::printf("  iterations    : %d timed (plus 5 warm-up, discarded)\n\n", iters);

    if (M == N) {
        std::printf("  NOTE: M == N. A square matrix cannot catch a swapped-axis bug,\n");
        std::printf("        since the wrong answer has the right shape. Prefer M != N.\n\n");
    }

    // ---------------- host data ----------------
    float* h_A   = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    float* h_got = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_ref || !h_got) {
        std::fprintf(stderr, "host allocation failed (%.1f MiB x3)\n", bytes / 1048576.0);
        std::free(h_A); std::free(h_ref); std::free(h_got);
        return EXIT_FAILURE;
    }

    std::printf("  generating input ... ");
    std::fflush(stdout);
    generate_input(h_A, M, N);
    std::printf("done (A[i][j] = i*N+j, every value a distinct exact integer)\n");

    std::printf("  CPU reference    ... ");
    std::fflush(stdout);
    transpose_cpu_reference(h_A, h_ref, M, N);
    std::printf("done\n\n");

    // ---------------- device buffers ----------------
    float *d_A = nullptr, *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));


    TransposeContext ctx;
    ctx.d_A      = d_A;
    ctx.d_B      = d_B;
    ctx.M        = M;
    ctx.N        = N;
    ctx.sm_count = prop.multiProcessorCount;

    Step steps[5] = {
        { "0. Copy (reference, NOT a transpose)", launch_copy_reference, false, 0.0, {} },
        { "1. Naive (scattered writes)",          launch_transpose_l1,   true,  0.0, {} },
        { "2. Tiled shared (bank conflicts)",     launch_transpose_l2,   true,  0.0, {} },
        { "3. Tiled shared + padding",            launch_transpose_l3,   true,  0.0, {} },
    };
    const int n_steps = 5;

    // ---------------- correctness, then timing ----------------
    // A transpose does no arithmetic, so the tolerance is nominal: the
    // meaningful figure is the bit-exact mismatch count, which must be 0.
    const double atol = 0.0;

    for (int s = 0; s < n_steps; ++s) {
        // Poison the output so a kernel that fails to write part of B is
        // caught, rather than inheriting a previous step's correct result --
        // a trap that makes a broken kernel look fine. 0xFF bytes form a
        // NaN pattern, which the non-finite counter detects explicitly.
        CUDA_CHECK(cudaMemset(d_B, 0xFF, bytes));

        steps[s].fn(ctx);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_got, d_B, bytes, cudaMemcpyDeviceToHost));

        // The copy is checked against the INPUT; the transposes against the
        // transposed reference.
        const float* expect = steps[s].is_transpose ? h_ref : h_A;
        steps[s].v = verify(h_got, expect, elems, atol);

        std::printf("  [%s] %-36s", steps[s].v.ok ? "PASS" : "FAIL", steps[s].name);
        if (!steps[s].v.ok) {
            std::printf(" %lld mismatches, %lld non-finite, max|diff| %.3g\n",
                        steps[s].v.exact_mismatches, steps[s].v.nonfinite,
                        steps[s].v.max_abs_diff);
        } else if (steps[s].is_transpose) {
            std::printf(" bit-exact\n");
        } else {
            std::printf(" bit-exact copy (bounds the problem, not a transpose)\n");
        }

        steps[s].ms = time_ms(steps[s].fn, ctx, iters);
    }

    // ---------------- report ----------------
    const double base_ms  = steps[1].ms;                             // step 1
    const double copy_gbs = traffic_bytes / (steps[0].ms / 1e3) / 1e9;

    std::printf("\n");
    std::printf("=======================================================================\n");
    std::printf(" Step                                     ms     GB/s  %% peak  %% copy  vs 1\n");
    std::printf("-------------------------------------------------------------------------------\n");
    for (int s = 0; s < n_steps; ++s) {
        const double gbs = traffic_bytes / (steps[s].ms / 1e3) / 1e9;
        std::printf(" %-36s %7.3f  %7.1f  %5.1f%%  %5.1f%%  %5.2fx\n",
                    steps[s].name, steps[s].ms, gbs,
                    100.0 * gbs / peak_gbs, 100.0 * gbs / copy_gbs,
                    base_ms / steps[s].ms);
    }
    std::printf("-------------------------------------------------------------------------------\n");
    std::printf(" All rows move the same %.1f MiB. With zero flops in the problem, every\n",
                traffic_bytes / 1048576.0);
    std::printf(" difference above is caused by memory access pattern alone.\n");
    std::printf(" '%% copy' is the more honest column: the copy kernel pays the memory\n");
    std::printf(" system's unavoidable overheads and nothing else, so it, not the\n");
    std::printf(" theoretical peak, is the ceiling a transpose can actually aim at.\n");
    std::printf("=======================================================================\n\n");


    // ---------------- cleanup ----------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    std::free(h_A);
    std::free(h_ref);
    std::free(h_got);

    bool all_ok = true;
    for (int s = 0; s < n_steps; ++s) all_ok = all_ok && steps[s].v.ok;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
