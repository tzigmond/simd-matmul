#pragma once

void fill_random(float* M, int N, unsigned int seed);

// Compares test against ref using a relative tolerance. Reassociation (tiling,
// FMA, and OpenBLAS all sum the k dimension in a different order than the naive
// reference) perturbs results at the 1e-5 level for N ~ 1000, so an absolute
// tolerance would produce false mismatches on large accumulations. Returns true
// when every element is within tol of the reference relative to its magnitude.
bool verify_correct(const float* ref, const float* test, int N, float tol = 1e-3f);
