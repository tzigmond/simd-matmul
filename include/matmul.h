#pragma once

// All kernels compute C = A * B for row-major N x N single-precision matrices.
// C must be zero-initialized by the caller for every kernel except
// matmul_openblas (sgemm writes C directly with beta = 0).

void matmul_naive(const float* A, const float* B, float* C, int N);
void matmul_blocked(const float* A, const float* B, float* C, int N, int block_size);
void matmul_avx2(const float* A, const float* B, float* C, int N);
void matmul_aligned(const float* A, const float* B, float* C, int N);
void matmul_threaded(const float* A, const float* B, float* C, int N);
void matmul_openblas(const float* A, const float* B, float* C, int N);
