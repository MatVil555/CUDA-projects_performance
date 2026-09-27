# CUDA profiling exercises

A ladder of eight matrix-multiplication kernels, from a deliberately naive
version to a software-pipelined one, plus cuBLAS as the reference. Each step
lives in its own `.cu` file, fixes exactly one problem that a profiler can
show you, and explains in its header comment which metric should move and why.

The subject is not really matrix multiplication. It is the loop:
**measure → find the actual bottleneck → change one thing → measure again.**
Matmul is just the ideal specimen, because it passes through almost every
bottleneck a CUDA kernel can have on the way from naive to fast.

## The ladder

| File | Step | What it fixes | Metric that should move |
|---|---|---|---|
| `matmul_naive.cu` | Naive baseline | nothing — one thread per output, uncoalesced on purpose | sectors/request ≈ 32 (worst case) |
| `matmul_coalesced.cu` | Coalescing | `threadIdx.x` mapped to the column, not the row (two lines) | sectors/request 32 → ~4, DRAM bytes collapse |
| `matmul_shared_memory_tiled.cu` | Shared-memory tiling | re-reading A and B from global memory N times | `dram__bytes_read` ÷ ~32 |
| `matmul_4.cu` | 1D register tiling | 2 shared loads per FMA | shared loads per FMA ≈ 1.1; occupancy drops (fine) |
| `matmul_5.cu` | 2D register tiling | reuse only in one dimension | 16 shared loads per 64 FMAs; FMA pipe utilization |
| `matmul_6.cu` | `float4` + transposed A tile | scalar memory instructions, A-side bank conflicts | 4× fewer memory instructions; load-side conflicts |
| `matmul_7.cu` | Double buffering | load and compute strictly serialized, 2 barriers/tile | barrier + long-scoreboard stalls |
| `matmul_8.cu` | cuBLAS reference | — | the number that makes yours interpretable |

Read the header comment of each file before its code. The comment is the
exercise; the kernel is the answer.

## Build and run

```bash
cd "CUDA profiling exercises"

make ARCH=sm_86            # build all 8 — set ARCH for your GPU
make run                   # run the whole ladder at N=1024
make run N=2048 ITERS=20   # bigger problem, more timed iterations

make DEBUG=1 matmul_3      # build with error checking on (see below)
./matmul_3 1024 10         # or run one exercise directly: <N> <iterations>
```

Find your architecture with `nvidia-smi --query-gpu=compute_cap --format=csv`
(8.0 → `sm_80`, 8.6 → `sm_86`, 8.9 → `sm_89`, 9.0 → `sm_90`).

Exercises 3–7 require `N` to be a multiple of their tile size (32, 64 or 128).
They refuse to run otherwise and suggest a valid size, rather than silently
computing a wrong answer. The default `N = 1024` works for all of them.

## Error checking: `CUDA_CALL` and `CUDA_CHECK`

Every `.cu` file starts with the same block, and it is duplicated on purpose —
reproducing it and remembering to wrap every CUDA call with it is part of the
habit being trained:

```c
#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif
```

* `CUDA_CALL(F)` wraps any call returning `cudaError_t` — `cudaMalloc`,
  `cudaMemcpy`, `cudaEventRecord`, `cudaFree`, and so on. Every such call in
  these exercises goes through it.
* `CUDA_CHECK()` takes no argument and is for kernel launches, which return
  `void`. It reports `__LINE__-1`, so **it must sit on the line immediately
  after the launch** — that is how it names the offending kernel. Every launch
  here is followed by it on the next line.
* With `-DDEBUG` absent, both expand to nothing (or to the bare call), so
  timing runs carry no checking overhead at all. That is the reason for the
  `#ifdef`: develop with `DEBUG=1`, measure without it.
* Asynchronous caveat: a kernel launch only reports *launch* errors
  synchronously. A fault inside the kernel surfaces at the next synchronizing
  call, which is why the harness also wraps `cudaDeviceSynchronize` and
  `cudaMemcpy` in `CUDA_CALL`. `matmul_8.cu` adds a `CUBLAS_CALL` in the same
  style, because cuBLAS returns `cublasStatus_t` and `CUDA_CALL` cannot check it.
* One sharp edge: in a `DEBUG` build `CUDA_CALL` expands to an `if` statement,
  so never use it as the unbraced body of another `if`/`else`.

## The two known matrices and the correctness check

`matmul_common.cuh` holds everything that is identical across the ladder, so
each exercise file contains only the idea it teaches.

**`create_known_matrices(A, B, N)`** fills two fixed, reproducible matrices:

```
A[i][k] = (i + 2k + 1) / N
B[k][j] = (3k + j + 1) / N
```

**`check_result(C, N)`** verifies the output against the closed form of that
product. With `a = i+1`, `b = j+1`, `S1 = Σk = N(N-1)/2` and
`S2 = Σk² = (N-1)N(2N-1)/6`:

```
C[i][j] = (1/N²) · Σₖ (a + 2k)(3k + b)
        = (1/N²) · [ 3a·S1 + 2b·S1 + ab·N + 6·S2 ]
```

Why this pattern rather than random numbers:

* **Verification is O(N²), not O(N³).** No CPU GEMM is needed, so checking a
  4096² result is instant and every run can afford to be verified. A kernel
  that is never checked is not fast, it is just wrong quickly.
* **Reproducible.** Same inputs on every step, every run, every GPU, so a
  difference in output is never the input's fault.
* **It catches the bugs this code actually produces.** The pattern is
  non-symmetric (coefficient 2 on `i`, 3 on `k`) and not rank-1, so a
  transposed index, swapped operands, or a tile written to the wrong place
  gives a wrong number instead of accidentally landing on the right one. The
  routines were checked against a brute-force double-precision GEMM, and
  against deliberately broken outputs (transposed, operand-swapped, one
  element perturbed by 1%, one output left unwritten) to confirm each is
  rejected.
* **Values stay small (≤ ~3) and strictly positive**, so FP32 accumulation
  suffers no cancellation and a tight tolerance stays meaningful. For
  power-of-two `N` the division by `N` is exact in binary floating point, so
  the only error in `C` is the kernel's own summation order.

The check uses a relative tolerance (default `1e-3`) because every kernel here
sums the k dimension in a different order, and FP32 addition is not
associative — bit-exact agreement is not a legitimate expectation. The
observed worst-case relative error is printed on every run (typically ~1e-6),
so the real margin is always visible rather than hidden behind the threshold.
`cpu_matmul_reference()` is included as well, for cross-checking the algebra
at small `N`.

## Profiling workflow

```bash
make metrics-matmul_3          # key metrics straight to the terminal
make profile-matmul_3          # full report -> reports/matmul_3.ncu-rep
ncu-ui reports/matmul_3.ncu-rep
```

Binaries are built with `-lineinfo`, so Nsight Compute can attribute stalls
and memory transactions to individual source lines. For cuBLAS, filter by
kernel name pattern, since the name is chosen at run time:

```bash
ncu --set full --kernel-name regex:".*gemm.*" -o reports/matmul_8 ./matmul_8 1024 1
```

A few habits worth keeping:

* **Profile one kernel per report.** Mixed reports are hard to read.
* **Use one iteration when profiling.** `ncu` replays and serializes kernels;
  timings taken inside a profile are not benchmarks. Take timings from the
  program's own CUDA-event measurement (what `make run` prints).
* **Look at Memory Workload Analysis and Warp State Statistics before
  Occupancy.** Occupancy is a means, not an end — exercises 4–7 deliberately
  *reduce* it and get faster. If you tune for the occupancy number you will
  undo them.
* **Change one thing per measurement.** That is why the ladder is eight files
  and not one file with eight flags.

## Record your numbers

The absolute values depend entirely on your GPU; the *shape* of the
progression is the point. Fill this in as you work through the ladder:

| Step | ms | GFLOP/s | × over step 1 | % of cuBLAS | Bottleneck the profiler showed |
|---|---|---|---|---|---|
| 1 naive | | | 1.00 | | uncoalesced global access |
| 2 coalesced | | | | | |
| 3 shared tiling | | | | | |
| 4 1D tiling | | | | | |
| 5 2D tiling | | | | | |
| 6 vectorized | | | | | |
| 7 double buffered | | | | | |
| 8 cuBLAS | | | | 100% | — |

If a step makes things *slower* on your hardware, that is a result, not a
failure — find out why in the profiler. Register spilling in step 5 (check
`launch__registers_per_thread` and local-memory traffic) and the extra shared
memory in step 7 costing you a resident block are the two most likely
culprits, and both are visible in the metrics listed in each file.
