#include "matmul.h"
#include <cstdio>

// TODO: implement cache-blocked tiling
//
// Partition A, B, C into block_size x block_size tiles. Process one tile
// at a time so the working set fits in L1/L2 cache, eliminating the column
// stride misses that kill the naive kernel on large N.
//
// Expected hardware effect: L1 miss rate drops sharply (verify with AMD uProf
// dc_miss_l2 counter). Speedup is typically 2-4x over naive at N=1024.
//
// Reference: Lam, Rothberg, Wolf — "The Cache Performance and Optimizations
// of Blocked Algorithms" (ASPLOS 1991)

void matmul_blocked(const float* A, const float* B, float* C, int N, int block_size) {
    fprintf(stderr, "WARNING: %s not yet implemented — falling back to naive\n", __func__);
    matmul_naive(A, B, C, N);
}
