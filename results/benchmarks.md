# Benchmark Results

Hardware: Intel Core Ultra 7 155H (Meteor Lake, 16C/22T) | L1d 48 KB/core, L2 2 MB/core, L3 24 MB shared | 32 GB LPDDR5
Environment: WSL2 Ubuntu 24.04, g++ 13.3 | single precision | best of 5 timed runs after a warmup
Reproduce: `make all && ./scripts/run_bench.sh 1024 8`

All kernels are verified against a naive reference (relative tolerance 1e-3) before timing; the harness aborts if any kernel disagrees.

---

## Results — N = 1024

| Kernel | GFLOP/s | Speedup vs naive | Effect |
|--------|---------|------------------|--------|
| Naive triple-loop | 0.75 | 1.0× | column-stride B access thrashes cache |
| Cache-blocked | 4.15 | 5.5× | i-k-j tiling keeps a B tile L1-resident |
| AVX2 vectorized | 22.0 | 29× | 8-wide FMA across columns |
| 32-byte aligned | 19.8 | 26× | within noise of AVX2 — see note |
| OpenMP threaded (8T) | 92.7 | 124× | disjoint C-row stripes across cores |
| OpenBLAS (reference) | 124.5 | 166× | tuned micro-kernel ceiling |

## Results — N = 512

| Kernel | GFLOP/s | Speedup vs naive |
|--------|---------|------------------|
| Naive triple-loop | 2.46 | 1.0× |
| Cache-blocked | 3.98 | 1.6× |
| AVX2 vectorized | 22.6 | 9.2× |
| 32-byte aligned | 18.9 | 7.7× |
| OpenMP threaded (8T) | 84.7 | 34× |
| OpenBLAS (reference) | 66.5 | 27× |

Note how naive collapses from 2.46 GFLOP/s at N=512 to 0.75 at N=1024: at 512 each
matrix is 1 MB and partially cache-resident, but at 1024 (4 MB/matrix) the
column-stride walk of B misses on nearly every inner iteration. That size
dependence is exactly why blocking's win grows from 1.6× to 5.5× — blocking only
matters once the working set stops fitting in cache. OpenBLAS at N=512 is
comparatively low (27×) because sgemm's packing/threading overhead is not
amortized by such a small problem; by N=1024 it pulls clearly ahead.

---

## The aligned stage did not help — and that is the finding

The hypothesis for stage 4 was that swapping `_mm256_loadu_ps` for
`_mm256_load_ps` on 32-byte-aligned buffers would improve load throughput. It
does not, measurably. Five trials each at N=1024:

```
avx2:    18.1, 17.6, 21.1, 22.4, 22.4 GFLOP/s
aligned: 18.7, 19.2, 17.4, 18.1, 18.6 GFLOP/s
```

The distributions overlap; the difference is noise. The reason is
microarchitectural: since Nehalem/Sandy Bridge, `vmovups` on an address that
happens to be aligned executes identically to `vmovaps` — the penalty only
appears when an access actually splits a cache line, which never happens here
because both kernels touch the same aligned addresses. `objdump` confirms the
instructions really differ (`aligned.o` emits `vmovaps`, `avx2.o` emits
`vmovups`); the hardware simply treats them the same on aligned data. The
kernel is kept as an honest negative result rather than deleted.

---

## Hardware counter attribution — not available under WSL2

The original plan was to attribute each stage's speedup to a specific hardware
event (L1 miss rate, FP pipe utilization, IPC). That requires PMU access, and
WSL2 does not virtualize the PMU: `perf stat -e cache-misses,instructions`
returns `<not supported>` for every hardware event on this setup (only the
software `task-clock`/`page-faults` events work). AMD uProf is likewise a
non-starter here — wrong vendor and same PMU limitation.

So the numbers above are wall-clock GFLOP/s, which is a real and reproducible
result but a weaker one than counter-backed attribution. To do this properly the
project needs to run on bare-metal Linux, where the intended commands are:

```
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./build/benchmark 1024 naive
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./build/benchmark 1024 blocked   # expect miss rate to drop
perf stat -e fp_arith_inst_retired.256b_packed_single ./build/benchmark 1024 avx2    # expect 256-bit ops to appear
perf stat -e instructions,cycles ./build/benchmark 1024 threaded                     # expect IPC ~flat, cores up
```

Until that bare-metal run happens, treating the cache/vector/IPC attributions as
*measured* would be dishonest — they are the expected mechanism, not verified here.
