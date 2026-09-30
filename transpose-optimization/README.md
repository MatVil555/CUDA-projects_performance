# transpose-optimization

A ladder of four out-of-place matrix transpose kernels for a **non-square** matrix
(default 4096 x 2048), from a naive baseline with scattered writes to a
bank-conflict-free tiled kernel, with cuBLAS as the reference point. Each step
lives in its own `.cu` file, fixes one hardware bottleneck, and opens with a
comment explaining what that bottleneck is and why the change addresses it.

This is the same ladder as the single-file `../Small_projects/transpose.cu`,
split into separately compiled units with a shared harness. The kernel bodies
are identical between the two; use whichever you prefer.

## Why transpose is the purest memory benchmark there is

`B[j][i] = A[i][j]` performs **exactly zero** floating-point operations — not
"few", zero. Arithmetic intensity is 0.0 flop/byte. There is no arithmetic to
optimize, no reuse to exploit, no blocking tradeoff to tune and no algorithmic
choice to make: every correct implementation moves exactly the same bytes.

That makes this an unusually honest teacher. Any performance difference between
the four steps is caused **purely** by memory access pattern. Nothing else is
left to explain it.

The awkwardness is inherent to the operation: element `(i,j)` sits at
`A[i*N+j]` and must land at `B[j*M+i]`. Walking A along a row is contiguous;
writing those same values into B walks a *column*, striding by M floats. You can
choose which side pays, but not whether anyone pays.

## The ladder

| File | Step | Bottleneck it fixes | Registers | Shared/block |
|---|---|---|---|---|
| `src/transpose_1_naive.cu` | Coalesced reads, scattered writes | — (baseline; ~8x write amplification) | 12 | — |
| `src/transpose_2_tiled.cu` | Tile via `__shared__`, both sides coalesced | uncoalesced global writes | 24 | 4096 B |
| `src/transpose_3_padded.cu` | `[TILE_DIM][TILE_DIM+1]` | 32-way shared bank conflict | 24 | 4224 B |
| `src/transpose_4_cublas.cu` | `cublasSgeam`, alpha=1 beta=0 | — (library baseline) | — | — |

Supporting files, which are not steps of the ladder:

| File | Role |
|---|---|
| `include/transpose_common.cuh` | error macros, tile geometry, grid helpers, input generation, CPU reference, bit-exact verification, CUDA-event timing |
| `src/copy_reference.cu` | plain copy — moves identical traffic without transposing, to bound the problem from above |
| `src/main.cu` | driver: generate, verify, time, report |
| `tools/bank_sim.cpp` | CPU bank-conflict simulator; checks step 3's central claim with no GPU |

Read each file's header comment before its code. The comment is the point; the
kernel is the answer.

## Build and run

```bash
cd transpose-optimization
make ARCH=sm_75            # set ARCH for your GPU (Colab T4 -> sm_75)
make run                   # default 4096 x 2048, 50 iterations
make run M=8192 N=4096 ITERS=20
```

Find your architecture with `nvidia-smi --query-gpu=compute_cap --format=csv`
(7.0 → `sm_70`, 7.5 → `sm_75`, 8.0 → `sm_80`, 8.6 → `sm_86`, 8.9 → `sm_89`,
9.0 → `sm_90`).

## Checking the claims yourself

```bash
make banks     # bank-conflict simulation -- NO GPU REQUIRED
make smem      # shared memory per kernel (4096 vs 4224 B)
make regs      # registers and, more importantly, spill traffic
```

`make banks` is the interesting one: the whole justification for step 3 is exact
integer arithmetic, so it can be checked by exhaustive enumeration on a CPU. It
prints:

```
  stride  phase-1 store     phase-2 load (the transpose)
  ------------------------------------------------------
  32       1-way (free)     32-way  SERIALIZES
  33       1-way (free)      1-way  CONFLICT-FREE
```

and asserts that result, exiting non-zero if the arithmetic ever stops holding.
Note what it also shows: the phase-1 *store* was never conflicting in either
case. The padding fixes the column read — the access that actually performs the
transposition — and nothing else.

`make smem` should report 4096 bytes for step 2 and 4224 for step 3: the 128-byte
price of removing a 32-way conflict.

For profiling:

```bash
make profile-step3       # full Nsight Compute report -> reports/step3.ncu-rep
make metrics-step3       # key metrics on stdout
```

The metric that tells the step 1 → 2 story is sectors-per-request on the **store**
(expect ~32 before, ~4 after); for step 2 → 3 it is the shared-memory bank
conflict counters. Profile one kernel per report and use one iteration: `ncu`
replays and serializes kernels, so timings taken inside a profile are not
benchmarks. Take those from `make run`.

## The bank-conflict arithmetic

Shared memory has 32 banks of 4-byte words, `bank = word_index % 32`. A warp's
access completes in one cycle only if its 32 lanes hit 32 distinct banks.

Step 2's phase-2 column read, `tile[tx][off]`, with row stride 32:

```
word = tx*32 + off
bank = (tx*32 + off) % 32 = off        <- the tx term VANISHES
```

Every lane computes the *same* bank while wanting a *different* address in it: a
32-way conflict, serialized into 32 cycles.

Step 3 pads the row to 33 words. Since `33 ≡ 1 (mod 32)`:

```
word = tx*33 + off
bank = (33*tx + off) % 32 = (tx + off) % 32   <- a rotation of all 32 banks
```

Distinct for every lane, so the access completes in one cycle.

Intuition: with stride 32 every tile row starts in the same bank, so a tile
column lives entirely inside one bank. Stride 33 shifts each row one bank to the
right, so the tile is stored skewed and a column cuts diagonally across all 32
banks. **The padding is a shear transform on the storage layout.**

Any stride coprime with 32 works; 33 is the smallest and therefore cheapest.
Padding to 64 would double the shared footprint, potentially halve occupancy, and
buy nothing. Note what 33 is *not* about: alignment, or avoiding a cache line.
It is modular arithmetic on bank indices, nothing more.

## Correctness: this kernel should be bit-exact

A transpose does no arithmetic, so it has no rounding, so the output must match
the reference in **every bit**. This is one of the few kernels where a tolerance
check is the wrong instinct — a nonzero tolerance quietly passing a transpose is
hiding a bug, not absorbing floating-point noise. The driver reports the
bit-exact mismatch count as the headline figure and keeps a tolerance only as a
safety net for the cuBLAS path.

The input reinforces this: `A[i][j] = (i*N + j) & 0xFFFFFF`, so every element is
a distinct exactly-representable integer carrying its own linear index. Any index
error — swapped axis, off-by-one tile offset, wrong leading dimension, dropped
edge tile — lands a provably wrong, identifiable number in a known place instead
of something plausible. The driver also warns if you pass `M == N`, since a
square matrix cannot catch a swapped-axis bug: the wrong answer has the right
shape.

## Why there is a copy row in the table

`% of theoretical peak` is a slightly dishonest yardstick, because no real kernel
reaches the datasheet number — DRAM refresh, read/write bus turnaround and
imperfect access scheduling all take a cut before your code is involved. The copy
kernel pays all of those and nothing else, so:

- **copy vs theoretical peak** = what the memory system costs anyway
- **best transpose vs copy** = what transposition actually costs

If step 3 reaches 85% of the copy, the remaining 15% is the price of the scatter
and closing it means attacking partition camping rather than the kernel body. If
step 3 reaches 99% of the copy while both sit at 60% of peak, the kernel is
finished and the rest is the hardware's. A single `% of peak` column cannot tell
those apart, which is the whole argument for an empirical ceiling.

## Expected shape of the result

`3 <= 2 < 1` in time, with step 4 near step 3. A transpose gives a library almost
nothing to exploit — no arithmetic to schedule, no reuse to block for — so expect
a near-tie rather than a rout. Compare the sibling ladders here:

| Problem | Who wins | Why |
|---|---|---|
| GEMM | library, decisively | deep blocking hierarchy, huge reuse, per-arch tuning |
| softmax | library **loses** | no fused primitive, so BLAS calls force extra DRAM passes |
| reduction | roughly a tie | one streaming pass, nothing to exploit |
| transpose | roughly a tie | pure data movement, nothing to exploit |

Recognising which of those regimes a problem is in is the skill worth taking
away. Absolute numbers depend entirely on your GPU; the *shape* of the
progression is the point. If a step is slower on your hardware, that is a result,
not a failure — find out why in the profiler.

## Verification status

Written and compile-verified in an environment with **no GPU**, so it has never
been executed on real silicon. What was verified:

- Builds warning-free (`-Wall -Wextra`) for `sm_70/75/80/86/89/90` and links
  against cuBLAS.
- `ptxas -v`: zero register spills in every kernel, and the shared-memory
  footprints match the documentation exactly (4096 B for step 2, 4224 B for
  step 3) — reproduce with `make smem` / `make regs`.
- `make banks` passes: worst-case 32-way conflict at stride 32, conflict-free at
  stride 33, both stores conflict-free.
- Every kernel's index math was re-implemented on the CPU and checked
  **bit-exactly** at 7 shapes — the default, aligned non-square, ragged tiles in
  both dimensions, tall-thin, 1xN, Nx1 and square — with the `cublasSgeam`
  column-major mapping emulated from the documented formula rather than from the
  derivation. Four deliberately injected bugs (missing block-index swap,
  non-transposed tile read, wrong `lda`, off-by-one bound) were all caught,
  confirming the checks have real power.
- The kernel bodies here are byte-identical to the validated single-file version,
  so that validation carries over unchanged.

Not verified: real timings, achieved bandwidth, occupancy, hardware conflict
counters, or cuBLAS kernel selection. Read the PASS/FAIL column before the
timings.
