#pragma once

void matmul_naive(const float* A, const float* B, float* C, int N);
void matmul_blocked(const float* A, const float* B, float* C, int N, int block_size);
void matmul_avx2(const float* A, const float* B, float* C, int N);
void matmul_aligned(const float* A, const float* B, float* C, int N);
void matmul_threaded(const float* A, const float* B, float* C, int N);
