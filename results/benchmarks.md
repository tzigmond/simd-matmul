# Benchmark Results

Hardware: Intel Core Ultra 7 155H (Meteor Lake), 22 logical CPUs | per-core L1d 48 KB, L1i 64 KB, L2 2 MB (64-byte lines), shared L3 24 MB | 16 GB visible to the WSL2 VM | AVX2/FMA3, no AVX-512
Environment: Ubuntu 24.04.4 LTS on WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2), g++ 13.3.0 | single precision | best of 5 timed runs after a warmup
Reproduce: `make all && ./scripts/run_bench.sh 1024 8`

All kernels are verified against a naive reference (relative tolerance 1e-3) before
timing; the harness aborts if any kernel disagrees. Numbers below were measured on
this laptop and **carry large run-to-run variance** — read the variance section
before trusting any single figure.

---

## Results — N = 1024 (one same-session sweep)

| Kernel | GFLOP/s | Speedup vs naive | Effect |
|--------|---------|------------------|--------|
| Naive triple-loop | 0.83 | 1.0× | column-stride B access thrashes cache |
| Cache-blocked | 5.28 | 6.4× | i-k-j tiling keeps a B tile L1-resident |
| AVX2 vectorized | 23.8 | 29× | 8-wide FMA across columns |
| 32-byte aligned | ~20 † | ~24× | no gain over AVX2 — see head-to-head below |
| OpenMP threaded (8T) | 73 | 88× ‡ | disjoint C-row stripes across cores |
| OpenBLAS (reference) | 41–308 ‡ | — | tuned, self-threaded ceiling |

† The aligned figure in this particular sweep landed at 11.4, but that was avx2
catching a good moment and aligned a bad one within the sequential sweep. A
back-to-back head-to-head (below) puts aligned at ~0.85× of avx2, so ~20 is the
honest number. This is exactly why sequential single-sweep comparisons are weak
here.

‡ The threaded and OpenBLAS speedup-vs-naive numbers are **not meaningful to
quote precisely** — see variance.

## Results — N = 512

| Kernel | GFLOP/s | Speedup vs naive |
|--------|---------|------------------|
| Naive triple-loop | 2.77 | 1.0× |
| Cache-blocked | 5.31 | 1.9× |
| AVX2 vectorized | 26.5 | 9.6× |
| 32-byte aligned | 23.0 | 8.3× |
| OpenMP threaded (8T) | 84–132 | — |
| OpenBLAS (reference) | 34–67 | — |

Naive collapses from 2.77 GFLOP/s at N=512 to ~0.8 at N=1024: at 512 each matrix
is 1 MB and partly cache-resident; at 1024 (4 MB/matrix) the column-stride walk of
B misses on nearly every inner iteration. That size dependence is why blocking's
win grows from 1.9× to 6.4×.

## Results — N = 2048 (fast kernels, VERIFY=0)

| Kernel | GFLOP/s |
|--------|---------|
| AVX2 vectorized | 12.6 |
| 32-byte aligned | 13.1 |
| OpenMP threaded (8T) | 89.8 |
| OpenBLAS (reference) | 232.8 |

At N=2048 single-thread AVX2 drops from ~13–22 to ~13 as the 16 MB working set
overflows L2. OpenBLAS pulls further ahead (2.6× the threaded kernel here) because
its panel packing and register-blocked micro-kernel manage the memory hierarchy
far better than a flat tiled loop — this is the gap that would take real work to
close.

---

## Run-to-run variance is the headline finding

This is a thermally-throttled laptop under WSL2 with shared cores, not a pinned
benchmark box. The same binary, same N, produces very different absolute
throughput depending on machine state. Observed ranges at N=1024 across sessions:

| Kernel | Low | High | Spread |
|--------|-----|------|--------|
| naive | 0.41 | 0.83 | 2.0× |
| blocked | 2.7 | 5.3 | 2.0× |
| avx2 | 13.1 | 23.8 | 1.8× |
| aligned | 11.1 | ~20 | ~1.8× |
| threaded | 70 | 132 | 1.9× |
| openblas | 41 | 308 | **7.5×** |

Two conclusions follow, and they matter more than any single number:

1. **The single-core chain throttles together, so its *ratios* are stable.**
   Whether naive is 0.42 or 0.83, blocked is ~6.5× it and avx2 is ~5× blocked in
   every session. Those speedups are real and reproducible.

2. **Multicore results (threaded, OpenBLAS) decouple from the single-core
   baseline**, because core count, contention, and per-core boost interact.
   OpenBLAS swings 7.5× (41→308) depending on whether it gets the machine to
   itself; it manages its own threading (up to all 16 threads by default). So
   quoting "threaded is 88× naive" or "OpenBLAS is 166× naive" is meaningless —
   the honest statement is a *range*, and that OpenBLAS is the ceiling.

In a tight back-to-back set (4 trials each, same minute) the numbers are stable
to a few percent — the variance is between sessions, driven by thermal/frequency
state, not measurement noise within a session.

---

## The aligned stage did not help — head-to-head confirms it

Measuring avx2 and aligned back-to-back at N=1024 (4 trials each), which controls
for session drift:

```
avx2:    14.62  13.28  13.09  13.37   GFLOP/s
aligned: 11.92  11.28  11.14  12.30   GFLOP/s
```

Aligned is reproducibly ~13% *slower* — the opposite of the "aligned loads are
faster" hypothesis. Two reasons: (1) since Nehalem, `vmovups` on aligned data
runs identically to `vmovaps`, so there is no upside; (2) the aligned wrapper
adds a runtime `N % 8` branch to pick the path, a small downside with no
compensating gain. `objdump` confirms the instructions really differ
(`aligned.o` emits `vmovaps`, `avx2.o` emits `vmovups`) — the hardware just
treats them the same on aligned data. Kept as an honest negative result.

---

## Hardware counter attribution — not available under WSL2

Per-stage attribution to hardware events (L1 miss rate, FP-pipe utilization, IPC)
requires PMU access, and WSL2 does not virtualize the PMU: `perf stat -e
cache-misses,instructions` returns `<not supported>` for every hardware event
here (only software `task-clock`/`page-faults` work). AMD uProf is a non-starter
too — wrong vendor and same PMU limitation.

So every number here is wall-clock GFLOP/s: real and reproducible-in-trend, but
not counter-backed. On bare-metal Linux the intended commands are:

```
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./build/benchmark 1024 naive
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./build/benchmark 1024 blocked   # expect miss rate to drop
perf stat -e fp_arith_inst_retired.256b_packed_single ./build/benchmark 1024 avx2    # expect 256-bit ops to appear
perf stat -e instructions,cycles ./build/benchmark 1024 threaded                     # IPC ~flat, cores up
```

Until that bare-metal run happens, the cache/vector/IPC attributions are the
expected mechanism, not verified fact.
