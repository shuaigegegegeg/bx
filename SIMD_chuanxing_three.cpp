#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <Windows.h>

// SSE头文件
#include <xmmintrin.h>

using namespace std;

// 生成随机矩阵
void generateRandomMatrix(float *A, int n) {
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < n*n; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// 优化除法阶段的二重循环
void divideOptimized(float *A, int n) {
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

        // 更新剩余部分
        for (int i = k + 1; i < n; i++) {
            if (i % 4 == 0) {
                A[i * n + k] = 0.0; // 对齐处理
            } else {
                A[i * n + k] /= A[k * n + k];
            }
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

// 优化消去阶段的三重循环
void eliminateOptimized(float *A, int n) {
    for (int k = 0; k < n-1; k++) {
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
        }
    }
}

int main() {
    int n;
    cout << "Enter the size of the matrix: ";
    cin >> n;

    // 生成随机矩阵
    float *matrix = new float[n*n];
    generateRandomMatrix(matrix, n);

    // 测量除法阶段优化算法耗时
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    divideOptimized(matrix, n);
    QueryPerformanceCounter(&end);
    double timeDivide = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // 生成新的随机矩阵
    generateRandomMatrix(matrix, n);

    // 测量消去阶段优化算法耗时
    QueryPerformanceCounter(&start);
    eliminateOptimized(matrix, n);
    QueryPerformanceCounter(&end);
    double timeEliminate = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Time taken for divide stage: " << timeDivide << " ms" << endl;
    cout << "Time taken for elimination stage: " << timeEliminate << " ms" << endl;

    delete[] matrix;
    return 0;
}
