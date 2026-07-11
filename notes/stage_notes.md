# Stage Notes

## Stage 1 — Naive triple-loop
The baseline teaches you where the ceiling is before any optimization — the inner loop's column-stride access to B thrashes the cache on every iteration, so this kernel is almost entirely bottlenecked on memory latency rather than compute.

## Stage 2 — Cache-blocked tiling
Blocking teaches that the cache hierarchy is the real bottleneck for memory-bound kernels — by keeping a tile of B resident in L1, you go from one memory access per FLOP to one memory access per tile worth of FLOPs.

## Stage 3 — AVX2 vectorization
AVX2 teaches that the CPU's vector units are doing almost nothing in the naive and blocked kernels — switching to 256-bit FMA lets you retire 8 FMAs per instruction instead of one, turning a memory-bottleneck problem into a compute-throughput problem.

## Stage 4 — 32-byte alignment
Alignment teaches that unaligned 256-bit loads have a hidden penalty on Zen 2 when they cross cache line boundaries — posix_memalign with aligned loads is a small but measurable win that shows up clearly in the load latency counters.

## Stage 5 — OpenMP multithreading
Threading teaches that once you've extracted per-core throughput, you scale to all 8 cores essentially for free with a single pragma — and that the scheduling and false-sharing story matters as much as the thread count.

## Stage 6 — OpenBLAS reference ceiling
OpenBLAS teaches humility and context — seeing how far a professionally tuned BLAS is above your kernel tells you exactly how much headroom is left and what techniques (micro-kernel register blocking, loop unrolling, prefetch intrinsics) would close the gap.

## Stage 7 — Profiling and attribution
This stage teaches that "X% faster" is a weak claim without a counter to back it — every speedup in this project is assigned a specific hardware event, which is the difference between an engineer who benchmarks and one who understands the machine.
