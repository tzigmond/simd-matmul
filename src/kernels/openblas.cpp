#include "matmul.h"
#include <cblas.h>

// Reference ceiling. OpenBLAS dispatches a hand-tuned, architecture-specific
// sgemm micro-kernel (register blocking + software prefetch + packed panels),
// which is the target these hand-written stages are measured against.
//
// beta = 0 means C is overwritten, so the caller does not need to zero it.
void matmul_openblas(const float* A, const float* B, float* C, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A, N,
                      B, N,
                0.0f, C, N);
}
