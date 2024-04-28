#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <time.h>
#include <iostream>
#include <windows.h>

void gaussian_elimination(float *A, int n) {
    for (int k = 0; k < n-1; k++) {
        __m128 vt = _mm_set_ps1(A[k * n + k]);
        for (int j = k + 1; j + 4 <= n; j += 4) {
            __m128 va = _mm_loadu_ps(&A[k * n + j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&A[k * n + j], va);
        }
        for (int j = k + 1; j < n; j++) {
            A[k * n + j] /= A[k * n + k];
        }
        A[k * n + k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m128 vaik = _mm_set_ps1(A[i * n + k]);
            for (int j = k + 1; j + 4 <= n; j += 4) {
                __m128 vakj = _mm_loadu_ps(&A[k * n + j]);
                __m128 vaij = _mm_loadu_ps(&A[i * n + j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i * n + j], vaij);
            }
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0.0;
        }
    }
}

void gaussian_elimination1(float *A, int n) {
    for (int k = 0; k < n-1; k++) {
        __m128 vt = _mm_set_ps1(A[k * n + k]);
        int j;
        for (j = k + 1; j < n && ((uintptr_t)&A[k * n + j] % 16 != 0); j++) {
            A[k * n + j] /= A[k * n + k];
        }
        for (; j + 4 <= n; j += 4) {
            __m128 va = _mm_load_ps(&A[k * n + j]);
            va = _mm_div_ps(va, vt);
            _mm_store_ps(&A[k * n + j], va);
        }
        for (; j < n; j++) {
            A[k * n + j] /= A[k * n + k];
        }
        A[k * n + k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m128 vaik = _mm_set_ps1(A[i * n + k]);
            for (j = k + 1; j < n && ((uintptr_t)&A[k * n + j] % 16 != 0); j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            for (; j + 4 <= n; j += 4) {
                __m128 vakj = _mm_load_ps(&A[k * n + j]);
                __m128 vaij = _mm_load_ps(&A[i * n + j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_store_ps(&A[i * n + j], vaij);
            }
            for (; j < n; j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0.0;
        }
    }
}

int main() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double elapsed_time;

    int n;
    printf("Enter the size of the matrix: ");
    scanf("%d", &n);
    
    // Allocate memory for the matrix
    float *A = (float*)malloc(n * n * sizeof(float));
    
    // Initialize the matrix with random values
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX * 10.0; // Random values between 0 and 10
    }
    
    // Print the original matrix
    /*
    printf("Original matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }
    */
    // Start timing for non-aligned Gaussian elimination
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    // Perform Gaussian elimination
    gaussian_elimination(A, n);
    
    // Stop timing for non-aligned Gaussian elimination
    QueryPerformanceCounter(&end);
    elapsed_time = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart; // Convert to milliseconds
    printf("\nElapsed time for non-aligned Gaussian elimination: %.2f milliseconds\n", elapsed_time);
    
    // Start timing for aligned Gaussian elimination
    QueryPerformanceCounter(&start);
    
    // Perform aligned Gaussian elimination
    gaussian_elimination1(A, n);
    
    // Stop timing for aligned Gaussian elimination
    QueryPerformanceCounter(&end);
    elapsed_time = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart; // Convert to milliseconds
    printf("\nElapsed time for aligned Gaussian elimination: %.2f milliseconds\n", elapsed_time);
    
    // Free allocated memory
    free(A);
    
    return 0;
}
