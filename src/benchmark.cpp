#include "matmul.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static double gflops(int N, double seconds) {
    return 2.0 * N * N * N / (seconds * 1e9);
}

int main(int argc, char* argv[]) {
    int N = 1024;
    const char* kernel = "naive";
    int block_size = 64;

    if (argc >= 2) N       = atoi(argv[1]);
    if (argc >= 3) kernel  = argv[2];
    if (argc >= 4) block_size = atoi(argv[3]);

    printf("kernel=%-10s  N=%d\n", kernel, N);

    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    // Warmup
    memset(C, 0, N * N * sizeof(float));
    if      (strcmp(kernel, "naive")    == 0) matmul_naive(A, B, C, N);
    else if (strcmp(kernel, "blocked")  == 0) matmul_blocked(A, B, C, N, block_size);
    else if (strcmp(kernel, "avx2")     == 0) matmul_avx2(A, B, C, N);
    else if (strcmp(kernel, "aligned")  == 0) matmul_aligned(A, B, C, N);
    else if (strcmp(kernel, "threaded") == 0) matmul_threaded(A, B, C, N);
    else { fprintf(stderr, "unknown kernel: %s\n", kernel); return 1; }

    // Timed runs
    const int RUNS = 5;
    double best = 0.0;

    for (int r = 0; r < RUNS; r++) {
        memset(C, 0, N * N * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();

        if      (strcmp(kernel, "naive")    == 0) matmul_naive(A, B, C, N);
        else if (strcmp(kernel, "blocked")  == 0) matmul_blocked(A, B, C, N, block_size);
        else if (strcmp(kernel, "avx2")     == 0) matmul_avx2(A, B, C, N);
        else if (strcmp(kernel, "aligned")  == 0) matmul_aligned(A, B, C, N);
        else if (strcmp(kernel, "threaded") == 0) matmul_threaded(A, B, C, N);

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double gf   = gflops(N, secs);

        printf("  run %d: %.4f s  %.2f GFLOP/s\n", r + 1, secs, gf);
        if (gf > best) best = gf;
    }

    printf("best: %.2f GFLOP/s\n", best);
    printf("C[0] = %f\n", C[0]);  // prevent dead-code elimination

    free(A); free(B); free(C);
    return 0;
}
