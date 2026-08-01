#include "matmul.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static double gflops(int N, double seconds) {
    // A dense N x N multiply performs N^3 multiplies + N^3 adds = 2 N^3 flops.
    // Cast to double before multiplying: 2 * 1024^3 overflows a 32-bit int.
    return 2.0 * (double)N * N * N / (seconds * 1e9);
}

// 32-byte aligned so matmul_aligned / matmul_threaded can issue _mm256_load_ps
// on the base pointer and on every row start. A row begins at offset i*N; for
// that offset to stay 32-byte (8-float) aligned, N must be a multiple of 8,
// which the AVX2 kernels rely on before taking their aligned path.
static float* alloc_aligned(int N) {
    void* p = nullptr;
    if (posix_memalign(&p, 32, (size_t)N * N * sizeof(float)) != 0) {
        fprintf(stderr, "posix_memalign failed for N=%d\n", N);
        exit(1);
    }
    return (float*)p;
}

static void run_kernel(const char* kernel, const float* A, const float* B,
                       float* C, int N, int block_size) {
    if      (strcmp(kernel, "naive")    == 0) matmul_naive(A, B, C, N);
    else if (strcmp(kernel, "blocked")  == 0) matmul_blocked(A, B, C, N, block_size);
    else if (strcmp(kernel, "avx2")     == 0) matmul_avx2(A, B, C, N);
    else if (strcmp(kernel, "aligned")  == 0) matmul_aligned(A, B, C, N);
    else if (strcmp(kernel, "threaded") == 0) matmul_threaded(A, B, C, N);
    else if (strcmp(kernel, "openblas") == 0) matmul_openblas(A, B, C, N);
    else { fprintf(stderr, "unknown kernel: %s\n", kernel); exit(1); }
}

int main(int argc, char* argv[]) {
    int N = 1024;
    const char* kernel = "naive";
    int block_size = 64;

    if (argc >= 2) N          = atoi(argv[1]);
    if (argc >= 3) kernel     = argv[2];
    if (argc >= 4) block_size = atoi(argv[3]);

    printf("kernel=%-9s N=%d block=%d\n", kernel, N, block_size);

    float* A   = alloc_aligned(N);
    float* B   = alloc_aligned(N);
    float* C   = alloc_aligned(N);
    float* ref = alloc_aligned(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    // Independent reference via the naive kernel, then verify the kernel under
    // test against it. This doubles as the warmup run.
    memset(ref, 0, (size_t)N * N * sizeof(float));
    matmul_naive(A, B, ref, N);

    memset(C, 0, (size_t)N * N * sizeof(float));
    run_kernel(kernel, A, B, C, N, block_size);
    if (!verify_correct(ref, C, N)) {
        fprintf(stderr, "correctness: FAIL\n");
        free(A); free(B); free(C); free(ref);
        return 1;
    }
    printf("correctness: PASS\n");

    const int RUNS = 5;
    double best = 0.0;
    for (int r = 0; r < RUNS; r++) {
        memset(C, 0, (size_t)N * N * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        run_kernel(kernel, A, B, C, N, block_size);
        auto t1 = std::chrono::high_resolution_clock::now();

        double secs = std::chrono::duration<double>(t1 - t0).count();
        double gf   = gflops(N, secs);
        printf("  run %d: %.4f s  %.2f GFLOP/s\n", r + 1, secs, gf);
        if (gf > best) best = gf;
    }

    printf("best: %.2f GFLOP/s\n", best);
    printf("C[0]=%f (anti-DCE)\n", C[0]);

    free(A); free(B); free(C); free(ref);
    return 0;
}
