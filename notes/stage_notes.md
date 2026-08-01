# Stage Notes

Working notes on each kernel: what changed, and what the measurement actually
showed (see results/benchmarks.md for numbers).

## Stage 1 — Naive triple-loop
i-j-k dot-product order. `B[k*N+j]` strides by N floats down a column each step,
so one useful float per 64-byte line fetched. Fine while the matrix fits in cache
(2.46 GFLOP/s at N=512), collapses once it does not (0.75 at N=1024). This is the
baseline the rest is measured against, compiled `-fno-tree-vectorize` so it stays
honestly scalar.

## Stage 2 — Cache-blocked tiling
Two changes: i-k-j order (inner loop now streams B and C contiguously) and
tiling into block_size chunks so a B tile is reused across the A tile's rows.
Still scalar. Win is size-dependent: 1.6× at N=512, 5.5× at N=1024 — blocking
only pays once the naive working set stops fitting in cache. Worth remembering
that the loop-reorder alone (i-k-j) already recovers a lot; the tiling adds
reuse on top.

## Stage 3 — AVX2 vectorization
Inner j loop becomes `_mm256_fmadd_ps`: broadcast one A element, load 8 contiguous
B elements, accumulate 8 columns of C. ~5× over blocked (4.15 → 22 GFLOP/s at
N=1024). Scalar tail handles the last <8 columns so arbitrary N stays correct.
The core is templated on load alignment and shared with stages 4 and 5.

## Stage 4 — 32-byte alignment (negative result)
Hypothesis: aligned loads beat unaligned. Measured: no difference beyond noise
(see the five-trial comparison in results). On this microarchitecture `vmovups`
on aligned data is as fast as `vmovaps`; alignment only matters on accesses that
split a cache line, which these don't. Kept as an honest negative rather than
inflated into a fake 5-15% win. The real lever left here is register blocking
(compute a 4x2 tile of C per k step to reuse loaded B across multiple C rows) —
that would move the needle where alignment did not.

## Stage 5 — OpenMP multithreading
`#pragma omp parallel for schedule(static)` on the outer ii loop. Each thread owns
a disjoint stripe of C rows, so no race and no false sharing (stripes are
block_size rows apart). Same source as stage 4 — only the -fopenmp compile flag
differs. Scaling at N=1024: 23 → 45 → 68 → 103 GFLOP/s on 1/2/4/8 threads. Sub-
linear past 4 threads because the kernel is partly memory-bandwidth bound, not
compute bound — worth profiling to confirm.

## Stage 6 — OpenBLAS reference
`cblas_sgemm`. The ceiling: 124 GFLOP/s at N=1024, ~1.3× over the threaded kernel.
The gap is register-blocked micro-kernels, software prefetch, and panel packing.
Closing it would mean writing a packed micro-kernel — the obvious next step if
this project continues.

## Stage 7 — Hardware counter attribution (blocked on environment)
The intent was to back each speedup with a counter (L1 miss rate for blocking,
256-bit FP ops for AVX2, IPC for threading). WSL2 does not expose the PMU — perf
returns `<not supported>` for hardware events. This is deferred to a bare-metal
run; until then the attributions are the expected mechanism, not measured fact.
