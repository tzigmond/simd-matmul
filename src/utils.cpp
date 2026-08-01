#include "utils.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

void fill_random(float* M, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N * N; i++)
        M[i] = (float)rand() / RAND_MAX;
}

bool verify_correct(const float* ref, const float* test, int N, float tol) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float r = ref[i*N + j];
            float t = test[i*N + j];
            float rel = fabsf(r - t) / (fabsf(r) + 1e-6f);
            if (rel > tol) {
                fprintf(stderr, "MISMATCH at [%d][%d]: ref=%.6f got=%.6f rel=%.2e\n",
                        i, j, r, t, rel);
                return false;
            }
        }
    }
    return true;
}
