#include "matmul.h"
#include <cstdio>

// TODO: implement 32-byte aligned AVX2 kernel
//
// Same as avx2.cpp but allocate A, B, C with posix_memalign (32-byte
// alignment) and use _mm256_load_ps instead of _mm256_loadu_ps.
// Aligned loads avoid the cross-cache-line penalty on misaligned 256-bit
// accesses, improving load throughput on Zen 2.
//
// Expected hardware effect: load efficiency improves (verify with AMD uProf
// MemBandwidth / L1 load latency counters). Speedup over avx2 is modest
// (~5-15%) but measurable and worth understanding.
//
// Note: benchmark.cpp must be updated to use aligned allocation for this
// kernel — document that change when implementing.

void matmul_aligned(const float* A, const float* B, float* C, int N) {
    fprintf(stderr, "WARNING: %s not yet implemented — falling back to naive\n", __func__);
    matmul_naive(A, B, C, N);
}
