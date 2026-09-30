// =====================================================================
// bank_sim.cpp -- brute-force shared-memory bank conflict simulator
//
//   g++ -O2 -std=c++14 tools/bank_sim.cpp -o bank_sim && ./bank_sim
//   (or: make banks)
//
// The central claim of the transpose ladder is that the phase-2 column
// read of the tile suffers a 32-way bank conflict at row stride 32 and
// none at stride 33. That claim is exact integer arithmetic, so it can be
// checked by exhaustive enumeration on a CPU -- no GPU required.
//
// This program does exactly that: for every (threadIdx.y + j) offset it
// computes the bank each of the 32 lanes would touch, and reports the
// worst-case number of lanes landing in a single bank. That number IS the
// conflict degree, because a shared-memory access serializes into as many
// cycles as the maximum number of distinct addresses in one bank.
//
// Hardware model (Volta through Hopper, and Kepler onward in practice):
//   * 32 banks, 4-byte words
//   * bank(word_index) = word_index % 32
//   * one cycle if all 32 lanes hit distinct banks; N cycles if some bank
//     is wanted by N lanes at N different addresses
//   * lanes reading the IDENTICAL address broadcast for free -- tracked
//     below so a broadcast is never miscounted as a conflict
// =====================================================================

#include <cstdio>
#include <set>
#include <algorithm>

static const int BANKS = 32;
static const int TILE  = 32;
static const int BROWS = 8;

// Worst-case conflict degree over all offsets, for one access pattern.
//   transposed = false : tile[ty+j][tx]  (phase 1 store)
//   transposed = true  : tile[tx][ty+j]  (phase 2 load)
int worst_conflict(int stride, bool transposed, int* worst_offset)
{
    int worst = 0;
    *worst_offset = -1;

    for (int off = 0; off < TILE; ++off) {          // the (ty + j) value
        // Per bank, collect the distinct word addresses the 32 lanes want.
        std::set<long long> addrs_in_bank[BANKS];

        for (int tx = 0; tx < BANKS; ++tx) {
            const long long word = transposed
                ? static_cast<long long>(tx) * stride + off   // tile[tx][off]
                : static_cast<long long>(off) * stride + tx;  // tile[off][tx]
            addrs_in_bank[static_cast<int>(word % BANKS)].insert(word);
        }

        // Conflict degree = max distinct addresses in any single bank.
        // Distinct, not count: identical addresses broadcast for free.
        int degree = 0;
        for (int b = 0; b < BANKS; ++b) {
            degree = std::max(degree, static_cast<int>(addrs_in_bank[b].size()));
        }
        if (degree > worst) { worst = degree; *worst_offset = off; }
    }
    return worst;
}

int main()
{
    std::printf("shared-memory bank conflict simulation\n");
    std::printf("  model: %d banks, 4-byte words, bank = word_index %% %d\n",
                BANKS, BANKS);
    std::printf("  tile %dx%d, block %dx%d\n\n", TILE, TILE, TILE, BROWS);

    std::printf("  stride  phase-1 store     phase-2 load (the transpose)\n");
    std::printf("  ------------------------------------------------------\n");

    for (int pad = 0; pad <= 1; ++pad) {
        const int stride = TILE + pad;
        int o1 = -1, o2 = -1;
        const int store = worst_conflict(stride, false, &o1);
        const int load  = worst_conflict(stride, true,  &o2);

        std::printf("  %2d      %2d-way%-11s %2d-way  %s\n",
                    stride, store, (store == 1 ? " (free)" : ""),
                    load, (load == 1 ? "CONFLICT-FREE" : "SERIALIZES"));
    }

    std::printf("\n  Why: bank = (stride*tx + off) %% 32 for the phase-2 load.\n");
    std::printf("    stride 32 -> 32 %% 32 == 0, the tx term vanishes, so every\n");
    std::printf("                 lane computes bank = off. One bank, 32 distinct\n");
    std::printf("                 addresses -> 32 cycles.\n");
    std::printf("    stride 33 -> 33 %% 32 == 1, so bank = (tx + off) %% 32, a\n");
    std::printf("                 rotation hitting all 32 banks once -> 1 cycle.\n");
    std::printf("\n  Any stride coprime with 32 works; 33 is the cheapest, costing\n");
    std::printf("  %d extra bytes of shared memory per block (%d -> %d).\n",
                TILE * 4, TILE * TILE * 4, TILE * (TILE + 1) * 4);

    // The claim under test, asserted so `make banks` fails loudly if the
    // arithmetic is ever broken by an edit to TILE or BROWS.
    int tmp;
    const bool ok = (worst_conflict(TILE,     true, &tmp) == BANKS) &&
                    (worst_conflict(TILE + 1, true, &tmp) == 1) &&
                    (worst_conflict(TILE,     false, &tmp) == 1) &&
                    (worst_conflict(TILE + 1, false, &tmp) == 1);
    std::printf("\n  [%s] documented claim: stride 32 load is %d-way, stride 33 load\n",
                ok ? "PASS" : "FAIL", BANKS);
    std::printf("         is conflict-free, and both stores are conflict-free.\n");
    return ok ? 0 : 1;
}
