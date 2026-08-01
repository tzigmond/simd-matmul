#include "matmul.h"
#include "avx2_kernel.h"

// Identical source to the aligned kernel, the difference is entirely in the
// build: this translation unit is the only one compiled with -fopenmp, so the
// `#pragma omp parallel for` guarding the outer ii loop in avx2_kernel.h becomes
// active here and is a no-op everywhere else. Threads split the ii row-tiles
// with schedule(static); each thread writes a disjoint set of C rows, so there
// is no data race and no false sharing (stripes are block_size rows apart, far
// wider than a cache line). Set OMP_NUM_THREADS to control the thread count.
void matmul_threaded(const float* A, const float* B, float* C, int N) {
    if (N % 8 == 0) avx2_matmul_impl</*Aligned=*/true >(A, B, C, N, 64);
    else            avx2_matmul_impl</*Aligned=*/false>(A, B, C, N, 64);
}
