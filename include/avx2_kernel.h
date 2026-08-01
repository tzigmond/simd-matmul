#pragma once
#include <immintrin.h>
#include <algorithm>

// Shared tiled + vectorized matmul core for the AVX2, aligned, and threaded
// stages. Those three stages differ only in two axes, both resolved at compile
// time so there is zero runtime branching in the hot loop:
//
//   * Aligned (template param): false -> _mm256_loadu/storeu_ps (unaligned),
//     true -> _mm256_load/store_ps. The aligned wrapper only instantiates
//     <true> when N % 8 == 0, which guarantees every row start is 32-byte
//     aligned given the harness's posix_memalign(32) buffers.
//   * OpenMP: the outer ii loop is parallelized only in the translation unit
//     compiled with -fopenmp (threaded.o). Each thread owns a disjoint stripe
//     of C rows, so there is no write sharing and no reduction needed.
//
// The vectorization runs across j: for each output row we hold 8 columns of C
// in a __m256 accumulator, broadcast a single A[i][k], and FMA against 8
// contiguous B[k][j..j+7]. The scalar tail handles j counts not divisible by 8
// so arbitrary N stays correct. Accumulates into C (+= via load/store), so the
// caller must zero C first.
template <bool Aligned>
inline void avx2_matmul_impl(const float* A, const float* B, float* C,
                             int N, int BS) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int ii = 0; ii < N; ii += BS) {
        const int imax = std::min(ii + BS, N);
        for (int kk = 0; kk < N; kk += BS) {
            const int kmax = std::min(kk + BS, N);
            for (int jj = 0; jj < N; jj += BS) {
                const int jmax = std::min(jj + BS, N);
                for (int i = ii; i < imax; ++i) {
                    int j = jj;
                    for (; j + 8 <= jmax; j += 8) {
                        __m256 acc;
                        if constexpr (Aligned) acc = _mm256_load_ps(&C[i*N + j]);
                        else                    acc = _mm256_loadu_ps(&C[i*N + j]);

                        for (int k = kk; k < kmax; ++k) {
                            const __m256 a = _mm256_set1_ps(A[i*N + k]);
                            __m256 b;
                            if constexpr (Aligned) b = _mm256_load_ps(&B[k*N + j]);
                            else                    b = _mm256_loadu_ps(&B[k*N + j]);
                            acc = _mm256_fmadd_ps(a, b, acc);
                        }

                        if constexpr (Aligned) _mm256_store_ps(&C[i*N + j], acc);
                        else                    _mm256_storeu_ps(&C[i*N + j], acc);
                    }
                    // Scalar remainder for the final < 8 columns of the tile.
                    for (; j < jmax; ++j) {
                        float s = C[i*N + j];
                        for (int k = kk; k < kmax; ++k)
                            s += A[i*N + k] * B[k*N + j];
                        C[i*N + j] = s;
                    }
                }
            }
        }
    }
}
