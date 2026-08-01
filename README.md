# SIMD Matrix Multiplication Kernel

A single-precision dense matrix multiply optimized in five stages — naive
triple-loop, cache-blocked tiling, AVX2 FMA vectorization, 32-byte aligned loads,
and OpenMP multithreading — benchmarked against OpenBLAS as a reference ceiling.
Every kernel is verified against a naive reference before timing, and each stage
is meant to isolate one effect: cache locality, vector throughput, load
alignment, or core scaling.

Measured on this machine, the progression at N=1024 runs from 0.75 GFLOP/s
(naive) to 92.7 GFLOP/s (8-thread), with OpenBLAS at 124.5. One stage — aligned
loads — produced **no** measurable gain, and the writeup keeps it as a negative
result instead of inflating it. See [`results/benchmarks.md`](results/benchmarks.md).

---

## Build

**Prerequisites**
- g++ 13+ with AVX2/FMA support (`sudo apt install g++`)
- OpenMP (ships with GCC)
- OpenBLAS for the reference kernel (`sudo apt install libopenblas-dev`)

```bash
make all        # builds build/benchmark
make clean
```

All kernels link into one `build/benchmark`. Each kernel object is compiled with
its own stage flags (baselines with `-fno-tree-vectorize`, SIMD stages with
`-O3 -march=native -mavx2 -mfma`, and only the threaded object with `-fopenmp`),
then linked together — so every kernel is measured under the codegen it actually
represents.

## Run

```bash
# ./build/benchmark [N] [kernel] [block_size]
./build/benchmark 1024 naive
./build/benchmark 1024 blocked 64
./build/benchmark 1024 avx2
./build/benchmark 1024 aligned
OMP_NUM_THREADS=8 ./build/benchmark 1024 threaded
./build/benchmark 1024 openblas

./scripts/run_bench.sh 1024 8   # full sweep -> markdown table
```

Each run verifies correctness against a naive reference, then reports per-run
time and GFLOP/s over 5 timed runs after a warmup. `block_size` tunes only the
`blocked` kernel; the SIMD kernels use a fixed 64-wide tile. Set `VERIFY=0` to
skip the O(N³) reference check when timing very large matrices.

---

## Results (N = 1024, best of 5)

| Kernel | GFLOP/s | Speedup vs naive | Effect |
|--------|---------|------------------|--------|
| Naive triple-loop | 0.75 | 1.0× | column-stride B access thrashes cache |
| Cache-blocked | 4.15 | 5.5× | i-k-j tiling keeps a B tile L1-resident |
| AVX2 vectorized | 22.0 | 29× | 8-wide FMA across columns |
| 32-byte aligned | 19.8 | 26× | no gain over AVX2 (within noise) |
| OpenMP threaded (8T) | 92.7 | 124× | disjoint C-row stripes across cores |
| OpenBLAS (reference) | 124.5 | 166× | tuned micro-kernel ceiling |

Threaded scaling at N=1024: 23 → 45 → 68 → 103 GFLOP/s on 1/2/4/8 threads
(sub-linear past 4 — partly bandwidth bound). Full data, the N=512 sweep, and the
aligned negative-result analysis are in [`results/benchmarks.md`](results/benchmarks.md).

---

## The design decision worth asking about: why block instead of transpose B?

The naive kernel is slow because of one line — the inner loop reads
`B[k*N + j]`, striding N floats (a full row) down a column of B every step. A
cache line holds 16 floats; this uses one of them before jumping away, so B is
re-fetched from memory over and over.

There are two standard fixes:

- **Transpose B once** into `B_T`, so `B_T[j*N + k]` is contiguous in the inner
  loop. Simple, and it makes the access sequential — but it costs an O(N²) copy
  and doubles B's memory footprint.
- **Block/tile** the loops (what this project does) so a `block_size²` tile of B
  is loaded once and reused across a whole tile of A rows. Zero copy, in place,
  and it also improves A and C reuse, not just B's access pattern.

Blocking wins when B changes between multiplies or you multiply once, because you
never pay the transpose. Transpose wins when you reuse the same B across many
multiplies and want dead-simple inner code. They also compose — a tuned kernel
(and OpenBLAS) *packs* tiles into a transposed, aligned scratch buffer, getting
both contiguous access and reuse. That packing is most of the remaining gap
between the threaded kernel here and OpenBLAS.

---

## Honest scope

- **Numbers are wall-clock GFLOP/s**, verified correct and reproducible, but not
  backed by hardware counters. WSL2 does not expose the PMU — `perf` returns
  `<not supported>` for hardware events — so per-stage counter attribution
  (L1 miss rate, FP-pipe utilization, IPC) is deferred to a bare-metal run. The
  intended `perf` commands are listed in [`results/benchmarks.md`](results/benchmarks.md).
- **The aligned stage is a negative result**, kept for honesty.
- **Next step to close the OpenBLAS gap**: a register-blocked micro-kernel with
  panel packing and software prefetch.

---

## System

| Component | Details |
|-----------|---------|
| CPU | Intel Core Ultra 7 155H (Meteor Lake, 16C/22T) |
| ISA | AVX2, FMA3 |
| Cache | L1d 48 KB/core · L2 2 MB/core · L3 24 MB shared |
| Memory | 32 GB LPDDR5 |
| OS | WSL2 Ubuntu 24.04 |
| Compiler | g++ 13.3 |
