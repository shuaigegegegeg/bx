#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h>

#define N 8 // AVX向量化每次处理8个元素

// 定义 AVX 操作需要的数据类型和宏
typedef float v8sf __attribute__((vector_size(32)));
#define load8FloatFrom(ptr) _mm256_loadu_ps(ptr)
#define store8FloatTo(ptr, val) _mm256_storeu_ps(ptr, val)
#define dupToVector8(val) _mm256_broadcast_ss(&val)

// 高斯消元函数
void gaussianElimination(float *A, int n) {
    for (int k = 0; k < n - 1; k++) {
        float vt = A[k * n + k];
        for (int j = k + 1; j + N <= n; j += N) {
            v8sf va = load8FloatFrom(&A[k * n + j]);
            va = va / vt;
            store8FloatTo(&A[k * n + j], va);
        }
        for (int j = k + 1; j < n; j++) {
            A[k * n + j] = A[k * n + j] / A[k * n + k];
        }
        A[k * n + k] = 1.0;

        for (int i = k + 1; i < n; i++) {
            v8sf vaik = dupToVector8(A[i * n + k]);
            for (int j = k + 1; j + N <= n; j += N) {
                v8sf vakj = load8FloatFrom(&A[k * n + j]);
                v8sf vaij = load8FloatFrom(&A[i * n + j]);
                v8sf vx = vakj * vaik;
                vaij = vaij - vx;
                store8FloatTo(&A[i * n + j], vaij);
            }
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] = A[i * n + j] - A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0.0;
        }
    }
}
void gaussianEliminationAligned(float *A, int n) {
    const int alignment = 4; // 对齐边界为4个元素
    for (int k = 0; k < n - 1; k++) {
        float vt = A[k * n + k];
        
        // 串行处理到对齐边界
        int j = k + 1;
        for (; j < n && j % alignment != 0; j++) {
            A[k * n + j] /= vt;
        }

        // 对齐 SIMD 计算
        for (; j + N <= n; j += N) {
            v8sf va = load8FloatFrom(&A[k * n + j]);
            va = va / vt;
            store8FloatTo(&A[k * n + j], va);
        }

        // 处理剩余的元素
        for (; j < n; j++) {
            A[k * n + j] /= vt;
        }
        
        A[k * n + k] = 1.0;

        for (int i = k + 1; i < n; i++) {
            v8sf vaik = dupToVector8(A[i * n + k]);
            for (j = k + 1; j + N <= n; j += N) {
                v8sf vakj = load8FloatFrom(&A[k * n + j]);
                v8sf vaij = load8FloatFrom(&A[i * n + j]);
                v8sf vx = vakj * vaik;
                vaij = vaij - vx;
                store8FloatTo(&A[i * n + j], vaij);
            }
            for (; j < n; j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0.0;
        }
    }
}



// 生成随机矩阵
void generateRandomMatrix(float *A, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int n;
    std::cout << "Enter the size of the matrix: ";
    std::cin >> n;

    float *matrix = new float[n * n];
    generateRandomMatrix(matrix, n);

    // 测试不对齐的 AVX(-512) 指令集优化的高斯消元函数
    auto start1 = std::chrono::high_resolution_clock::now();
    gaussianElimination(matrix, n);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms1 = end1 - start1;
    std::cout << "Elapsed time for AVX(-512) unaligned: " << elapsed_ms1.count() << "ms\n";

    // 测试对齐的 AVX 指令集优化的高斯消元函数
    auto start2 = std::chrono::high_resolution_clock::now();
    gaussianEliminationAligned(matrix, n);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms2 = end2 - start2;
    std::cout << "Elapsed time for AVX aligned: " << elapsed_ms2.count() << "ms\n";

    delete[] matrix;
    return 0;
}

