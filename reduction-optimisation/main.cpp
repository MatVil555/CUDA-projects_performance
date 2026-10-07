// =====================================================================
// main.cpp -- driver for the reduction optimization ladder
//
//   ./reduction_bench [N] [iterations]
//   defaults: N = 32,768,000 floats (125 MiB), 50 timed iterations
//
// Responsibilities, all of them deliberately kept OUT of the kernel
// files so each of those contains one idea:
//   * generate the input and its exactly-known sum
//   * show the two host baselines (naive FP32, double) for contrast
//   * demonstrate the cublasSasum signed-data caveat
//   * verify every step against the exact answer BEFORE timing it
//   * time each step and report ms, GB/s and % of theoretical peak
//
// THE PERFORMANCE CEILING, which is what the table is measured against:
// one add per 4-byte element is 0.25 flop/byte, so this is purely a
// bandwidth problem. The target is "read the array once at full bus
// speed": effective GB/s = N*4 / elapsed, and a good kernel lands within
// a few percent of the DRAM peak. Levels 1 and 2 are not slow because
// their trees are badly shaped -- the trees are fine -- they are slow
// because they burn issue slots while the memory bus sits idle.
// =====================================================================

#include "reduction_common.h"

struct Step {
    const char*    name;
    ReduceLaunchFn fn;
    double         traffic_mult;   // multiples of N*4 bytes actually read
    double         ms;
    double         result;
    double         rel_err;
    bool           ok;
};

int main(int argc, char** argv)
{
    // ---------------- configuration ----------------
    const long long N_arg = (argc > 1) ? std::atoll(argv[1]) : 32768000LL;
    const int       iters = (argc > 2) ? std::atoi(argv[2]) : 50;

    if (N_arg <= 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s [N] [iterations]  (both > 0)\n", argv[0]);
        return EXIT_FAILURE;
    }
    // cuBLAS level-1 routines take the length as int.
    if (N_arg > 2147483647LL) {
        std::fprintf(stderr, "N must fit in a 32-bit int (cuBLAS API limit)\n");
        return EXIT_FAILURE;
    }
    const size_t N     = static_cast<size_t>(N_arg);
    const size_t bytes = N * sizeof(float);

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
    std::printf(" Parallel sum reduction: four optimization steps\n");
    std::printf("=======================================================================\n");
    std::printf("  device        : %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("  peak DRAM BW  : %.1f GB/s (%d-bit bus @ %.0f MHz, theoretical)\n",
                peak_gbs, bus_bits, mem_clk_khz / 1000.0);
    std::printf("  problem       : N = %zu floats (%.1f MiB)\n", N, bytes / 1048576.0);
    std::printf("  tree depth    : ceil(log2 N) = %d levels vs %zu sequential steps\n",
                static_cast<int>(std::ceil(std::log2(static_cast<double>(N)))), N - 1);
    std::printf("  ideal time    : %.3f ms at peak bandwidth (read the array once)\n",
                (bytes / (peak_gbs * 1e9)) * 1e3);
    std::printf("  iterations    : %d timed (plus 5 warm-up, discarded)\n\n", iters);

    // ---------------- host data and the exact answer ----------------
    float* h_x = static_cast<float*>(std::malloc(bytes));
    if (!h_x) {
        std::fprintf(stderr, "host allocation of %.1f MiB failed\n", bytes / 1048576.0);
        return EXIT_FAILURE;
    }

    std::printf("  generating input ... ");
    std::fflush(stdout);
    const double exact = generate_input_and_exact_sum(h_x, N);
    std::printf("done\n");
    std::printf("  exact sum (integer arithmetic, zero rounding) : %.6f\n", exact);

    // Guard the denominator: at very small N the exact sum can legitimately
    // be 0 (element 0 of the hash sequence is 0), and dividing by it would
    // report NaN and a spurious failure for a perfectly correct answer.
    const double exact_den = (exact != 0.0) ? exact : 1.0;

    const float  naive32   = cpu_sum_fp32_naive(h_x, N);
    const double dbl       = cpu_sum_double(h_x, N);
    std::printf("  CPU sequential FP32 accumulator              : %.6f  (rel err %.3e)\n",
                static_cast<double>(naive32),
                std::fabs(static_cast<double>(naive32) - exact) / exact_den);
    std::printf("  CPU sequential double accumulator            : %.6f  (rel err %.3e)\n",
                dbl, std::fabs(dbl - exact) / exact_den);
    std::printf("      ^ the FP32 loop is ~1000x less accurate than the GPU tree below.\n");
    std::printf("        Once its running total passes 2^24 = 16777216 the representable\n");
    std::printf("        spacing is 2.0, so adding elements of ~1.0 contributes 0 or 2.\n");
    std::printf("        Sequential error grows O(N); a balanced tree grows O(log N),\n");
    std::printf("        because it only ever adds operands of similar magnitude. The\n");
    std::printf("        parallel algorithm is more accurate here, not less.\n");
    std::printf("        (The double loop is bit-exact at this size: every value is a\n");
    std::printf("        multiple of 2^-8 and totals stay under 2^25, needing 33 of its\n");
    std::printf("        53 mantissa bits, so no addition rounds at all.)\n\n");

    // The cublasSasum caveat, demonstrated rather than asserted.
    {
        const int n_demo = 8;
        double true_sum = 0.0, abs_sum = 0.0;
        for (int i = 0; i < n_demo; ++i) {
            const double v = (i % 2 == 0) ? 1.5 : -1.5;
            true_sum += v;
            abs_sum  += std::fabs(v);
        }
        std::printf("  cublasSasum caveat: on signed data [+1.5,-1.5,...] x%d\n", n_demo);
        std::printf("      true sum = %.1f   but sum(|x|) = %.1f   <- Sasum returns the\n",
                    true_sum, abs_sum);
        std::printf("      latter. It is the L1 norm, not a sum. It is only a valid sum\n");
        std::printf("      below because this input is non-negative by construction;\n");
        std::printf("      step 4b (Sdot vs ones) is the signed-safe choice.\n\n");
    }

    // ---------------- device buffers ----------------
    float* d_x      = nullptr;
    float* d_ones   = nullptr;
    float* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_ones, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    fill_ones(d_ones, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // The partials buffer must hold the largest stage-1 grid.
    const int l12_blocks = blocks_one_per_element(N);
    const int l3_blocks  = blocks_cascaded(N, prop.multiProcessorCount);
    const int max_parts  = (l12_blocks > l3_blocks) ? l12_blocks : l3_blocks;
    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, static_cast<size_t>(max_parts) * sizeof(float)));

    std::printf("  geometry      : steps 1-2  %d blocks x %d thr (1 element/thread)\n",
                l12_blocks, kBlock);
    std::printf("                  step 3     %d blocks x %d thr (~%.0f elements/thread)\n",
                l3_blocks, kBlock,
                static_cast<double>(N) / (static_cast<double>(l3_blocks) * kBlock));
    std::printf("                  stage 2 collapses %d partials (steps 1-2) or %d (step 3)\n\n",
                l12_blocks, l3_blocks);

    // ---------------- cuBLAS setup ----------------
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep scalar results on the device so the calls stay asynchronous and
    // pipeline like the custom kernels. See reduce_4_cublas.cu.
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    ReduceContext ctx;
    ctx.d_x        = d_x;
    ctx.d_ones     = d_ones;
    ctx.d_partials = d_partials;
    ctx.d_result   = d_result;
    ctx.N          = N;
    ctx.sm_count   = prop.multiProcessorCount;
    ctx.cublas     = handle;

    Step steps[5] = {
        { "1. Interleaved (divergent)",     launch_reduce_l1,      1.0, 0, 0, 0, false },
        { "2. Sequential (divergence-free)",launch_reduce_l2,      1.0, 0, 0, 0, false },
        { "3. Shuffle + cascade + float4",  launch_reduce_l3,      1.0, 0, 0, 0, false },
        { "4a. cuBLAS Sasum (sum of |x|)",  launch_reduce_l4_asum, 1.0, 0, 0, 0, false },
        { "4b. cuBLAS Sdot vs ones",        launch_reduce_l4_dot,  2.0, 0, 0, 0, false },
    };
    const int n_steps = 5;

    // ---------------- correctness, then timing ----------------
    // Mixed absolute/relative acceptance: |err| <= atol + rtol*|exact|.
    // Relative alone breaks when the exact sum is 0 (possible at tiny N);
    // absolute alone would be meaningless across the range of N accepted.
    const double rtol = 1e-5;
    const double atol = 1e-3;

    for (int s = 0; s < n_steps; ++s) {
        // Poison the output so a step that never writes it fails loudly
        // instead of inheriting the previous step's correct answer -- a
        // trap that makes a broken kernel look fine.
        const float nan_poison = std::nanf("");
        CUDA_CHECK(cudaMemcpy(d_result, &nan_poison, sizeof(float), cudaMemcpyHostToDevice));

        steps[s].fn(ctx);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(float), cudaMemcpyDeviceToHost));

        const double g   = static_cast<double>(got);
        const double err = std::fabs(g - exact);
        steps[s].result  = g;
        steps[s].rel_err = std::isfinite(g) ? err / exact_den : INFINITY;
        steps[s].ok      = std::isfinite(g) && err <= atol + rtol * std::fabs(exact);

        std::printf("  [%s] %-34s sum = %.4f  (rel err %.3e)\n",
                    steps[s].ok ? "PASS" : "FAIL", steps[s].name,
                    steps[s].result, steps[s].rel_err);

        steps[s].ms = time_ms(steps[s].fn, ctx, iters);
    }

    // ---------------- report ----------------
    const double base_ms = steps[0].ms;
    std::printf("\n");
    std::printf("=======================================================================\n");
    std::printf(" Step                                     ms     GB/s  %% peak   vs 1\n");
    std::printf("-----------------------------------------------------------------------\n");
    for (int s = 0; s < n_steps; ++s) {
        const double secs = steps[s].ms / 1e3;
        const double gbs  = (steps[s].traffic_mult * static_cast<double>(bytes)) / secs / 1e9;
        std::printf(" %-36s %7.3f  %7.1f  %5.1f%%  %5.2fx\n",
                    steps[s].name, steps[s].ms, gbs,
                    100.0 * gbs / peak_gbs, base_ms / steps[s].ms);
    }
    std::printf("-----------------------------------------------------------------------\n");
    std::printf(" GB/s counts bytes actually read: N*4 everywhere except 4b, which also\n");
    std::printf(" reads the ones vector (2x traffic for identical math -- a structural\n");
    std::printf(" ceiling, not a tuning problem).\n");
    std::printf("=======================================================================\n\n");

    std::printf(" What each step bought:\n");
    std::printf("   1 -> 2 : removed warp divergence (~19%% -> 100%% lane utilization).\n");
    std::printf("            Same tree, same 9 barriers, same adds -- pure issue\n");
    std::printf("            efficiency. NOTE: the often-cited 'removed the slow %%'\n");
    std::printf("            is NOT the reason; with a compile-time block size nvcc\n");
    std::printf("            unrolls and strength-reduces %% to one AND (PTX: 0 rem,\n");
    std::printf("            8 and.b32). See reduce_1_interleaved.cu.\n");
    std::printf("   2 -> 3 : cascading (1 element/thread -> ~%.0f), 9 barriers -> 1,\n",
                static_cast<double>(N) / (static_cast<double>(l3_blocks) * kBlock));
    std::printf("            shared tree -> register shuffles, scalar -> float4. This is\n");
    std::printf("            where a bandwidth-bound kernel actually gets fast.\n");
    std::printf("   3 vs 4 : a reduction is one streaming pass with no reuse and no\n");
    std::printf("            blocking to tune, so there is little for a library to do\n");
    std::printf("            better. Unlike GEMM, hand-written code can match it here.\n\n");

    // ---------------- cleanup ----------------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_ones));
    CUDA_CHECK(cudaFree(d_partials));
    CUDA_CHECK(cudaFree(d_result));
    std::free(h_x);

    bool all_ok = true;
    for (int s = 0; s < n_steps; ++s) all_ok = all_ok && steps[s].ok;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
