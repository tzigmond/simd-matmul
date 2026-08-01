#include "matmul.h"
#include <algorithm>

// Cache-blocked scalar kernel. Two things change relative to naive:
//
//   1. Loop order goes from i-j-k (dot product, which strides B by N floats on
//      every step) to i-k-j, so the innermost loop walks B[k][*] and C[i][*]
//      contiguously. This alone removes the column-stride misses.
//   2. The i/k/j ranges are tiled into block_size chunks so a block_size^2 tile
//      of B stays resident in L1/L2 and is reused across all rows of the A tile,
//      turning O(1) FLOPs per memory access into O(block_size).
//
// Compiled with -fno-tree-vectorize so this stage isolates the *cache* win;
// vectorization is introduced separately in the AVX2 stage. Relies on C being
// pre-zeroed (the harness memsets it) because it accumulates with +=.
void matmul_blocked(const float* A, const float* B, float* C, int N, int block_size) {
    const int BS = block_size > 0 ? block_size : 64;

    for (int ii = 0; ii < N; ii += BS) {
        const int imax = std::min(ii + BS, N);
        for (int kk = 0; kk < N; kk += BS) {
            const int kmax = std::min(kk + BS, N);
            for (int jj = 0; jj < N; jj += BS) {
                const int jmax = std::min(jj + BS, N);
                for (int i = ii; i < imax; ++i) {
                    for (int k = kk; k < kmax; ++k) {
                        const float a = A[i*N + k];
                        for (int j = jj; j < jmax; ++j)
                            C[i*N + j] += a * B[k*N + j];
                    }
                }
            }
        }
    }
}
