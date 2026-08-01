#include "matmul.h"
#include "avx2_kernel.h"

// Same tiled AVX2 core, but with _mm256_load_ps / _mm256_store_ps instead of the
// unaligned variants. Aligned 256-bit loads avoid the split-load penalty when an
// access straddles a cache line. This is only safe when every row start B[k*N]
// and C[i*N] is 32-byte aligned: the harness allocates the buffers with
// posix_memalign(32), so it comes down to N being a multiple of 8. If it is not,
// fall back to the unaligned path to stay correct rather than fault.
void matmul_aligned(const float* A, const float* B, float* C, int N) {
    if (N % 8 == 0) avx2_matmul_impl</*Aligned=*/true >(A, B, C, N, 64);
    else            avx2_matmul_impl</*Aligned=*/false>(A, B, C, N, 64);
}
