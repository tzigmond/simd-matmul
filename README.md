# SIMD Matrix Multiplication Kernel

A C++ matrix multiplication kernel optimized in five stages — naive triple-loop, cache-blocked tiling, AVX2 SIMD intrinsics, 32-byte aligned loads, and OpenMP multithreading — with hardware performance counter attribution at every step using AMD uProf. The goal is not just to show that each stage is faster, but to identify the specific hardware event responsible: L1 miss rate, vector throughput, load efficiency, or IPC scaling. OpenBLAS is included as a reference ceiling to contextualize how far hand-optimization can take a naive kernel. This project demonstrates the systems programming depth expected at low-latency trading firms — cache hierarchy awareness, SIMD vectorization, and the ability to explain *why* code is fast at the hardware level.

---

## Build

**Prerequisites**
- g++ 13+ (`sudo apt install g++`)
- OpenMP (`sudo apt install libomp-dev`)
- OpenBLAS for reference comparison (`sudo apt install libopenblas-dev`)

**Compile individual stages**

```bash
make naive      # -O2 -fno-tree-vectorize
make blocked    # -O2 -fno-tree-vectorize
make avx2       # -O3 -march=native -mavx2 -mfma
make aligned    # -O3 -march=native -mavx2 -mfma
make threaded   # -O3 -march=native -mavx2 -mfma -fopenmp
make all        # builds all five
make clean
```

**Run benchmarks**

```bash
# Usage: ./build/<kernel> [N] [kernel_name] [block_size]
./build/naive   1024 naive
./build/blocked 1024 blocked 64
./build/avx2    1024 avx2
./build/aligned 1024 aligned
OMP_NUM_THREADS=8 ./build/threaded 1024 threaded
```

Each run executes one warmup pass followed by 5 timed runs, reporting per-run time, GFLOP/s, and the best result.

---

## Results

| Kernel | GFLOP/s | Speedup vs naive | Key hardware effect |
|--------|---------|-----------------|-------------------|
| Naive triple-loop | TBD | 1× (baseline) | — |
| Cache-blocked | TBD | TBD | L1 miss rate ↓ |
| AVX2 vectorized | TBD | TBD | Vector throughput ↑ |
| 32-byte aligned | TBD | TBD | Load efficiency ↑ |
| OpenMP threaded | TBD | TBD | IPC × core count |
| OpenBLAS (reference) | TBD | TBD | — |

Results will be updated with real numbers as each stage is benchmarked. Full counter readings are in [`results/benchmarks.md`](results/benchmarks.md).

---

## Hardware Counter Methodology

Wall-clock speedup alone is a weak result. Each stage in this project is accompanied by specific hardware event counters measured with **AMD uProf**, attributing the speedup to a single identifiable effect:

| Stage | Counter of interest | Expected change |
|-------|-------------------|----------------|
| Naive → blocked | `dc_miss_l2` (L1 miss rate) | Sharp decrease |
| Blocked → AVX2 | `FPUPipeUtil`, vector instruction mix | Vector utilization ↑ |
| AVX2 → aligned | Load latency / `MemBandwidth` | Aligned load throughput ↑ |
| Aligned → threaded | IPC, core utilization | Near-linear core scaling |

This separates genuine hardware-level understanding from black-box benchmarking.

---

## System Specs

| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen 7 3700X (Zen 2, 8c/16t, 3.6 GHz base / 4.4 GHz boost) |
| ISA extensions | AVX2, FMA3 |
| Memory | 32 GB DDR4 |
| Motherboard | ASRock B450M Pro4-F |
| OS | Windows 10, benchmarks run from WSL2 Ubuntu 24.04 |
| Compiler | g++ 13 (GCC 13) |
| Profiler | AMD uProf |
