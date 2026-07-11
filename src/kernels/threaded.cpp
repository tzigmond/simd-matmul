#include "matmul.h"
#include <cstdio>

// TODO: implement OpenMP multithreaded AVX2 kernel
//
// Add #pragma omp parallel for to the outer loop of the aligned AVX2 kernel.
// Each thread handles a contiguous stripe of rows of C, keeping private
// accumulator registers and avoiding false sharing on output rows.
//
// Expected hardware effect: near-linear scaling across the 8 physical cores
// of the Ryzen 7 3700X (verify with AMD uProf IPC and core utilization).
// Compile with -fopenmp; set OMP_NUM_THREADS=8 at runtime.
//
// Reference: OpenMP 5.1 spec — omp parallel for schedule(static)
// Reference: MIL-HDBK-1211 — not applicable here, wrong domain :)

void matmul_threaded(const float* A, const float* B, float* C, int N) {
    fprintf(stderr, "WARNING: %s not yet implemented — falling back to naive\n", __func__);
    matmul_naive(A, B, C, N);
}
