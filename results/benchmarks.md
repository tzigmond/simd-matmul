# Benchmark Results

Hardware: AMD Ryzen 7 3700X (Zen 2, 8c/16t) | 32 GB DDR4 | WSL2 Ubuntu 24.04
Compiler: g++ 13 | Matrix size: N=1024 | Each result: best of 5 runs

---

## Results Table

| Kernel | GFLOP/s | Speedup vs naive | Key hardware effect |
|--------|---------|-----------------|-------------------|
| Naive triple-loop | TBD | 1× (baseline) | — |
| Cache-blocked | TBD | TBD | L1 miss rate ↓ |
| AVX2 vectorized | TBD | TBD | Vector throughput ↑ |
| 32-byte aligned | TBD | TBD | Load efficiency ↑ |
| OpenMP threaded | TBD | TBD | IPC × core count |
| OpenBLAS (reference) | TBD | TBD | — |

---

## AMD uProf Counter Readings

### Stage 1 — Naive
- Wall time:
- GFLOP/s:
- `dc_miss_l2` (L1 miss rate):
- `l2_cache_miss` (LLC miss rate):
- Vector instruction count:
- Notes:

---

### Stage 2 — Cache-blocked
- Wall time:
- GFLOP/s:
- `dc_miss_l2` (L1 miss rate):
- `l2_cache_miss` (LLC miss rate):
- Block size used:
- Notes:

---

### Stage 3 — AVX2 vectorized
- Wall time:
- GFLOP/s:
- `FPUPipeUtil` / vector instruction count:
- Notes:

---

### Stage 4 — 32-byte aligned
- Wall time:
- GFLOP/s:
- Load efficiency delta vs AVX2:
- Notes:

---

### Stage 5 — OpenMP threaded
- Wall time:
- GFLOP/s:
- IPC:
- Core utilization:
- `OMP_NUM_THREADS` used:
- Notes:

---

### OpenBLAS reference
- Wall time:
- GFLOP/s:
- Notes:
