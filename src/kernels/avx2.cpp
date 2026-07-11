#include "matmul.h"
#include <cstdio>

// TODO: implement AVX2 vectorized kernel
//
// Use 256-bit AVX2 intrinsics (_mm256_fmadd_ps) to process 8 floats per
// instruction. The inner loop over k accumulates into a __m256 register,
// broadcasting A[i][k] and loading 8 consecutive B[k][j..j+7] per step.
//
// Expected hardware effect: vector throughput increases ~8x over scalar
// (verify with AMD uProf FPUPipeUtil / VectorInstructions counters).
// Requires -mavx2 -mfma compile flags.
//
// Reference: Intel Intrinsics Guide — _mm256_fmadd_ps
// Reference: Fog — "Optimizing software in C++" (2023), Chapter 12

void matmul_avx2(const float* A, const float* B, float* C, int N) {
    fprintf(stderr, "WARNING: %s not yet implemented — falling back to naive\n", __func__);
    matmul_naive(A, B, C, N);
}
