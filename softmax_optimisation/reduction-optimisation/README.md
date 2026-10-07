# reduction-optimisation

A ladder of four sum-reduction kernels for a large 1D float array, from a
deliberately divergent baseline to a cascaded warp-shuffle kernel, with cuBLAS
as the reference point. Each step lives in its own `.cu` file, fixes one
hardware bottleneck, and opens with a comment explaining what that bottleneck
is and why the change addresses it.

This is the same ladder as the single-file `../Small_projects/reduction.cu`,
split into separately compiled units with a shared harness. The kernel bodies
are identical between the two; use whichever you prefer.

## The ladder

| File | Step | Bottleneck it fixes | Barriers |
|---|---|---|---|
| `reduce_1_interleaved.cu` | Interleaved addressing, `tid % (2*stride)` | — (baseline; ~19% lane utilization by design) | 9 |
| `reduce_2_sequential.cu` | Sequential addressing, `tid < stride` | warp divergence | 9 |
| `reduce_3_shuffle.cu` | Cascaded grid-stride `float4` + `__shfl_down_sync` | instruction overhead starving the memory bus | **1** |
| `reduce_4_cublas.cu` | `cublasSasum`, plus `cublasSdot` vs ones | — (library baseline) | — |

Supporting files, which are not steps of the ladder:

| File | Role |
|---|---|
| `reduction_common.h` | error macros, tunables, grid geometry, `warp_reduce_sum`, input generation + exact reference, host baselines, CUDA-event timing |
| `reduce_final.cu` | shared stage-2 kernel collapsing per-block partials; explains why it is a second launch rather than an `atomicAdd` |
| `main.cpp` | driver: generate, verify, time, report |

Read each file's header comment before its code. The comment is the point; the
kernel is the answer.

## Layout

Everything sits in one flat directory, and the file extensions carry real
build information rather than decoration:

```
reduction-optimisation/
├── Makefile
├── main.cpp                  <- host only: compiled by g++
├── reduction_common.h        <- shared by both compilers
├── reduce_1_interleaved.cu   <- device code: compiled by nvcc
├── reduce_2_sequential.cu
├── reduce_3_shuffle.cu
├── reduce_4_cublas.cu
└── reduce_final.cu
```

`main.cpp` is `.cpp`, not `.cu`, because it contains no device code and no
`<<<>>>` launches — it reaches the GPU only through the CUDA runtime and cuBLAS
C APIs, which any host compiler can call given `-I$(CUDA_PATH)/include`. Each
kernel file owns its own launcher, so the `<<<>>>` syntax stays entirely inside
the `.cu` files. `reduction_common.h` is therefore written to be parseable by
both compilers: the one `__device__` helper, `warp_reduce_sum`, is fenced behind
`#ifdef __CUDACC__`. The Makefile compiles each group with the right compiler
and lets `nvcc` do the final link.

## Why this problem is all about memory

One add per 4-byte element is 0.25 flop/byte. A modern GPU needs 10–40
flop/byte to be compute bound, so a correct reduction kernel is purely a
bandwidth exercise and the target is simple: **read the array once at full bus
speed.** Effective bandwidth is `N*4 / elapsed`, and the report prints it as a
percentage of theoretical peak.

The corollary is worth internalising: steps 1 and 2 are not slow because their
trees are badly shaped. The trees are fine — same `N-1` additions, same
`log2(N)` depth. They are slow because they burn instruction issue slots on
masked-off lanes and barriers while the DRAM bus sits idle.

## Build and run

```bash
cd reduction-optimisation
make ARCH=sm_75            # set ARCH for your GPU (T4 -> sm_75)
make run                   # default N = 32,768,000 (125 MiB), 50 iterations
make run N=8000000 ITERS=20
```

Find your architecture with `nvidia-smi --query-gpu=compute_cap --format=csv`
(7.0 → `sm_70`, 7.5 → `sm_75`, 8.0 → `sm_80`, 8.6 → `sm_86`, 8.9 → `sm_89`,
9.0 → `sm_90`).

## Checking the claims yourself

The performance claims in the comments are measurable, and the Makefile has
targets for the two that matter most:

```bash
make instr      # rem / and.b32 / bar.sync / shfl counts per kernel
make regs       # registers and, more importantly, spill traffic
```

`make instr` should print:

```
kernel                          rem  and.b32  bar.sync  shfl
reduce_1_interleaved              0        8         9     0
reduce_2_sequential               0        0         9     0
reduce_3_shuffle                  0        1         1    10
reduce_final                      0        2         1    10
```

That table is also how one piece of standard folklore gets refuted — see below.

For profiling:

```bash
make profile-step3       # full Nsight Compute report -> reports/step3.ncu-rep
make metrics-step3       # key metrics on stdout
```

Profile one kernel per report, and use one iteration: `ncu` replays and
serializes kernels, so timings taken inside a profile are not benchmarks. Take
those from `make run`.

## Two textbook claims this code contradicts

Both were written the conventional way first, then checked against the
generated code and corrected.

**1. "The `%` operator is very slow" — not here.** The standard explanation for
step 1 blames `tid % (2*stride)` for being a synthesized integer remainder.
Because the block size is a compile-time constant, nvcc fully unrolls the tree,
each copy gets a literal stride, and `%` is strength-reduced to a single
`and.b32`. Verified: step 1 emits **0** `rem.u32` and 8 `and.b32`. Write the
same loop against a runtime `blockDim.x` and a real `rem.u32` appears. So step
1 → 2 is a pure *divergence* win; the modulo costs 8 extra ANDs, which is
minor.

**2. Step 1 does not suffer shared-memory bank conflicts.** Its active lanes
land on distinct banks (at stride 1: banks 0, 2, … 30). Conflicts appear in the
*intermediate* fix people reach for when removing the modulo
(`index = 2*stride*tid`), which is 2-way conflicting at stride 1 and 8-way by
stride 4. That near-miss is documented in `reduce_2_sequential.cu`, since
"pad the shared array to fix bank conflicts" would fix nothing on step 1.

## Correctness: an exactly-known answer

Validating a reduction against a floating-point CPU sum is circular — you end
up comparing two differently-wrong numbers. Instead every input is an integer
multiple of 1/256 (exactly representable in FP32), so the true sum is computed
in **uint64 integer arithmetic with zero rounding**. The per-step errors the
program prints are therefore meaningful in absolute terms.

That also exposes a result worth seeing: the parallel tree is both faster *and*
substantially more accurate than a sequential host loop.

| Summation | Result | Relative error |
|---|---|---|
| naive sequential FP32 | 32703606.0 | 1.23e-05 |
| GPU tree (steps 1–3) | 32704008.0 | **1.30e-08** |
| sequential `double` | 32704007.574219 | 0.0 (bit-exact) |

The FP32 loop degrades because once its running total passes 2^24 the
representable spacing is 2.0, so adding elements of ~1.0 contributes 0 or 2.
Sequential error grows `O(N)`; a balanced tree grows `O(log N)`, because it only
ever adds operands of similar magnitude.

Stated carefully, because it is easy to overstate: the FP32 damage is ~1e-5,
not the tens of percent a "half the additions are wrong" analysis predicts.
Round-to-nearest is unbiased, so the errors largely cancel in a random walk.

The `double` loop is *provably* exact at this size, not merely accurate: every
value is a multiple of 2^-8 and totals stay under 2^25, needing 33 of its 53
mantissa bits, so no addition rounds at all.

## The `cublasSasum` trap

`Sasum` computes `sum(|x|)` — the L1 norm, **not** a sum. It returns the right
answer here only because the input is non-negative by construction; the program
demonstrates the discrepancy on signed data at startup rather than asking you to
take it on faith. BLAS has no plain sum at all, because summation is neither a
norm nor a product.

`Sdot` against a ones vector is the signed-safe alternative, but it reads two
N-element arrays, a hard 2× bandwidth ceiling for identical mathematical work.
The report charges it for that traffic so the comparison stays honest.

cuBLAS is also put in `CUBLAS_POINTER_MODE_DEVICE`. Without it, every call
returns its scalar to host memory and forces an implicit device-to-host sync, so
a timing loop would measure round-trip latency rather than throughput — one of
the easiest ways to accidentally slander a library in a benchmark.

## Expected shape of the result

`3 ≈ 4 < 2 < 1` in time. Unlike a GEMM ladder, the library is **not** expected
to win by much here: a reduction is one streaming pass with no data reuse and no
blocking decisions to get right, so there is little for a vendor to be cleverer
about. Compare `../Small_projects/softmax.cu`, where the library loses badly for
structural reasons, and a GEMM ladder, where it wins decisively. Knowing which
of those three regimes you are in is the actual skill.

Absolute numbers depend entirely on your GPU; the *shape* of the progression is
the point. If a step makes things slower on your hardware, that is a result, not
a failure — find out why in the profiler.

## Verification status

Written and compile-verified in an environment with **no GPU**, so it has never
been executed on real silicon. What was verified:

- Builds warning-free (`-Wall -Wextra`) for `sm_70/75/80/86/89/90` and links
  against cuBLAS.
- `ptxas -v`: zero register spills in every kernel (step 3: 18 registers,
  1 barrier, 32 bytes shared).
- PTX instruction-counted per kernel — the `make instr` table above.
- Every kernel's arithmetic and index math was re-implemented on the CPU
  (barrier-separated tree steps, exact `__shfl_down_sync` out-of-range
  semantics, `float4` grouping, the grid-stride cascade, and the two-stage
  partials pipeline) and checked against the exact integer reference for
  N = 32,768,000 / 1,000,000 / 32,771 / 255 / 1. The kernel bodies here are
  byte-identical to the validated single-file version.

Not verified: real timings, occupancy, achieved bandwidth, divergence cost, or
cuBLAS kernel selection. Read the PASS/FAIL column before the timings.
