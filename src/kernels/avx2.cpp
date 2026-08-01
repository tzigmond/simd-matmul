#include "matmul.h"
#include "avx2_kernel.h"

// Tiled AVX2 kernel using unaligned 256-bit loads. Same tiling as the blocked
// stage, but the inner j loop is replaced by explicit _mm256_fmadd_ps over 8
// columns at a time. The aligned stage reuses this core with aligned loads; see
// avx2_kernel.h.
void matmul_avx2(const float* A, const float* B, float* C, int N) {
    avx2_matmul_impl</*Aligned=*/false>(A, B, C, N, 64);
}
