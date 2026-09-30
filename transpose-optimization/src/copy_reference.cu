// =====================================================================
// REFERENCE POINT (not a step) -- PLAIN COPY
// =====================================================================
// B[idx] = A[idx]. Same byte count as a transpose, no transposition. It is
// not a transpose, is never verified as one, and does not compete in the
// ladder. It exists to bound the problem from above.
//
// WHY IT EARNS A ROW IN THE TABLE.
// "% of theoretical peak" is a slightly dishonest yardstick, because no
// real kernel reaches the theoretical number. DRAM refresh cycles,
// read/write turnaround on the bus, and imperfect access scheduling all
// take a cut before your code is even involved. The copy pays all of those
// costs and nothing else. So:
//
//     copy vs theoretical peak  = what the memory system costs anyway
//     best transpose vs copy    = what transposition actually costs
//
// If step 3 reaches 85% of the copy, the remaining 15% is the price of the
// scatter, and closing it means attacking partition camping rather than
// the kernel body. If step 3 reaches 99% of the copy while both sit at 60%
// of theoretical peak, the kernel is finished and the remaining gap is the
// hardware's. Those are completely different situations that a single
// "% of peak" column cannot distinguish -- which is the entire argument
// for measuring against an empirical ceiling instead of a datasheet one.
//
// Implemented as a grid-stride loop sized to the machine rather than one
// thread per element: fewer, longer-lived threads amortize launch and
// index overhead, and consecutive lanes still hold consecutive addresses
// so every access is perfectly coalesced.
// =====================================================================

#include "transpose_common.cuh"

__global__ void copy_kernel(const float* __restrict__ A,
                            float* __restrict__ B,
                            size_t total)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        B[i] = A[i];
    }
}

void launch_copy_reference(const TransposeContext& c)
{
    const size_t total   = static_cast<size_t>(c.M) * static_cast<size_t>(c.N);
    const int    threads = 256;

    // 32 blocks per SM is comfortably enough resident warps to saturate
    // memory on every architecture from Volta on, capped so a small matrix
    // cannot ask for more blocks than there is work for.
    long long blocks = static_cast<long long>(c.sm_count) * 32;
    const long long cap = static_cast<long long>((total + threads - 1) / threads);
    if (blocks > cap) blocks = cap;
    if (blocks < 1)   blocks = 1;

    copy_kernel<<<static_cast<int>(blocks), threads>>>(c.d_A, c.d_B, total);
}
