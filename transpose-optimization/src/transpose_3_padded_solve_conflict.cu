// =====================================================================
// STEP 3 -- PADDED TILE: THE BANK CONFLICT REMOVED BY ARITHMETIC
// =====================================================================
// MOTIVATION
// Step 2 fixed global memory and left a 32-way shared-memory conflict on
// the one access that does the transposing. One extra column removes it
// completely: tile[TILE_DIM][TILE_DIM + 1]. Nothing else changes -- same
// loops, same global accesses, same barrier, same work. The padding column
// is never read or written. Its entire purpose is to change the row stride
// from 32 words to 33, and that changes the bank mapping.
//
// THE ARITHMETIC, WHICH IS THE WHOLE TRICK.
// bank = word_index % 32, and the row stride is now 33:
//
//   Phase 2 load, tile[tx][ty+j]:
//       word_index = tx*33 + (ty+j)
//       bank       = (33*tx + (ty+j)) % 32
//   and since 33 = 32 + 1, we have 33 == 1 (mod 32), so
//       bank       = (tx + (ty+j)) % 32
//
// The tx coefficient is 1 instead of 0. As tx runs over 0..31 the bank
// index runs over all 32 residues exactly once -- a permutation of the
// banks, merely rotated by (ty+j). Every lane hits a different bank, the
// access completes in ONE cycle, and a 32-way conflict became none.
//
// Intuition worth carrying away: with stride 32 every tile row starts in
// the same bank, so a tile *column* lives entirely inside one bank. Stride
// 33 shifts each successive row one bank to the right, so the tile is
// stored skewed and a column cuts diagonally across all 32 banks. The
// padding is a shear transform on the storage layout.
//
// Check it without a GPU: `make banks` brute-forces every lane's bank for
// both strides and reports the worst-case conflict degree.
//
// WHY 33 AND NOT 40, OR 64.
// Any stride coprime with 32 works, because then tx -> (stride*tx mod 32)
// is a bijection. 33 is the smallest such stride above 32, so it is also
// the cheapest: shared usage goes from 32*32*4 = 4096 bytes to
// 32*33*4 = 4224 bytes, a 3.1% increase. That matters because shared
// memory is a per-SM resource limiting how many blocks can be resident,
// and resident blocks are what hide latency. Padding to 64 would double
// the footprint, potentially halve occupancy, and buy nothing extra.
// Note also what 33 is NOT: not about alignment, not about avoiding a
// cache line. It is modular arithmetic on bank indices, nothing more.
//
// EFFECT ON WARP SCHEDULING.
// A conflicting shared access keeps the LSU busy for 32 cycles and parks
// the issuing warp in a short-scoreboard stall throughout. With the
// conflict gone each shared read retires in one cycle, the warp reaches
// its global store sooner, and the memory pipeline receives its next
// request earlier. The kernel stops being limited by on-chip serialization
// and becomes limited by DRAM -- which, for a kernel of exactly zero
// arithmetic intensity, is precisely where it should be. That is the point
// of the whole ladder: arrive at a kernel whose only remaining bottleneck
// is the genuinely unavoidable one.
//
// WHAT IS LEFT AFTER THIS.
// Mostly partition camping: with a power-of-two M the concurrently active
// blocks can hash onto a subset of DRAM channels and queue behind each
// other while other channels idle. The classic fix reorders block indices
// diagonally so concurrent blocks spread across channels. Not implemented
// here: it is architecture-sensitive, has been largely mitigated by
// address swizzling in memory controllers since Fermi, and would obscure
// the single variable this step isolates. If step 3 lands well below the
// copy kernel's bandwidth on your GPU, that is the next thing to
// investigate -- and the copy row in the table is how you would know.
//
// COST: 24 registers, 0 spills, 4224 bytes shared. Identical register
// count to step 2; the only difference is 128 bytes of shared memory.
// =====================================================================

#include "transpose_common.cuh"

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_l3_padded(const float* __restrict__ A,
                                    float* __restrict__ B,
                                    int M, int N)
{
    // The +1 is the entire optimization. Row stride becomes 33 words, and
    // 33 == 1 (mod 32), so a tile column spreads across all 32 banks.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            // bank = (33*(ty+j) + tx) % 32 -- distinct per lane, as before.
            tile[threadIdx.y + j][threadIdx.x] =
                A[static_cast<size_t>(y + j) * N + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            // bank = (tx + ty + j) % 32 -- a rotation of all 32 banks.
            // Conflict-free: one cycle instead of thirty-two.
            B[static_cast<size_t>(y + j) * M + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launch_transpose_l3(const TransposeContext& c)
{
    transpose_l3_padded<kTileDim, kBlockRows>
        <<<tile_grid(c.M, c.N), tile_block()>>>(c.d_A, c.d_B, c.M, c.N);
}
