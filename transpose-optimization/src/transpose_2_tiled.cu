// =====================================================================
// STEP 2 -- TILED VIA SHARED MEMORY: BOTH GLOBAL SIDES COALESCED
// =====================================================================
// MOTIVATION
// Step 1 is stuck because global memory only rewards contiguous access
// while a transpose needs two contradictory layouts at once. Shared memory
// has no such constraint: it is on-chip SRAM with no cache lines and no
// sectors, addressed in 32 independent banks, and a scattered pattern
// costs nothing there provided the banks differ. So move the awkward part
// to where it is cheap.
//
// A block claims a 32x32 tile and works in two phases:
//
//   Phase 1  read the tile from A ROW-WISE        -> coalesced global load
//            store it into shared memory
//            __syncthreads()
//   Phase 2  read the tile from shared TRANSPOSED
//            write it to B ROW-WISE               -> coalesced global store
//
// The transposition now lives entirely in the shared-memory indexing.
// Both global accesses index with threadIdx.x along the fastest-varying
// axis, so both are full-width 128-byte transactions. Sectors per request
// on the store drops from ~32 to ~4 -- step 1's 8x write amplification is
// simply gone. This single change is usually worth most of the total
// speedup in the ladder.
//
// THE INDEX SUBTLETY THAT MAKES IT WORK.
// Phase 2 must not write the tile back where it came from; it writes to
// the *transposed block position*. Note how blockIdx.x and blockIdx.y swap
// roles below. Block (bx,by) reads A rows [by*32, by*32+32) and columns
// [bx*32, bx*32+32); those elements belong in B rows [bx*32, ...) and B
// columns [by*32, ...). Getting this backwards is the classic bug, and it
// produces a result that looks tantalizingly transpose-like on a square
// matrix and obviously broken on a rectangular one -- one good reason the
// driver defaults to M != N and warns when M == N.
//


#include "transpose_common.cuh"

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_l2_tiled(const float* __restrict__ A,
                                   float* __restrict__ B,
                                   int M, int N)
{

    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;   // column of A
    int y = blockIdx.y * TILE_DIM + threadIdx.y;   // row of A

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // Bounds-checked per element so a ragged edge tile needs no
        // separate kernel. Untouched shared entries are never read back,
        // because phase 2 applies the mirrored check.
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] =
                A[static_cast<size_t>(y + j) * N + x];
        }
    }

    __syncthreads();   // the tile must be complete before anyone reads it


    x = blockIdx.y * TILE_DIM + threadIdx.x;       // column of B, [0,M)
    y = blockIdx.x * TILE_DIM + threadIdx.y;       // row of B,    [0,N)

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            // The column read: bank = (ty+j) % 32 for EVERY lane.
            // 32-way conflict, 32 serialized cycles.
            B[static_cast<size_t>(y + j) * M + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launch_transpose_l2(const TransposeContext& c)
{
    transpose_l2_tiled<kTileDim, kBlockRows>
        <<<tile_grid(c.M, c.N), tile_block()>>>(c.d_A, c.d_B, c.M, c.N);
}
