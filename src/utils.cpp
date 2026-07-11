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
            float diff = fabsf(ref[i*N + j] - test[i*N + j]);
            if (diff > tol) {
                fprintf(stderr, "MISMATCH at [%d][%d]: ref=%.6f  got=%.6f  diff=%.6f\n",
                        i, j, ref[i*N + j], test[i*N + j], diff);
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const float* M, int N, int max_print) {
    int lim = (max_print < N) ? max_print : N;
    for (int i = 0; i < lim; i++) {
        for (int j = 0; j < lim; j++)
            printf("%8.4f ", M[i*N + j]);
        printf("\n");
    }
}
