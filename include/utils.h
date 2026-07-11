#pragma once

void fill_random(float* M, int N, unsigned int seed);
bool verify_correct(const float* ref, const float* test, int N, float tol = 1e-3f);
void print_matrix(const float* M, int N, int max_print = 4);
