#include <iostream>
#include <vector>
#include <random>
#include <windows.h>
using namespace std;

vector<double> Gaussian_Elimination(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    
    // 消去过程
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    // 回代过程
    vector<double> x(n);
    /*
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    */
    return x;
}

int main() {
    int n;
    cout << "请输入矩阵的大小 n：";
    cin >> n;
    
    // 生成随机测试样例
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }
    /*
    cout << "原始矩阵 A：" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "原始向量 b：" << endl;
    for (int i = 0; i < n; i++) {
        cout << b[i] << " ";
    }
    cout << endl;
    */
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double elapsed_time;
     // Start timing
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    // 解方程
    vector<double> x = Gaussian_Elimination(A, b);
    
    // Stop timing
    QueryPerformanceCounter(&end);
    elapsed_time = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart; // Convert to milliseconds
    /*
    cout << "解向量 x：" << endl;
    for (int i = 0; i < n; i++) {
        cout << x[i] << " ";
    }
    cout << endl;
    
    cout << "上三角矩阵 U：" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) cout << "0 ";
            else cout << A[i][j] << " ";
        }
        cout << endl;
    }
    */
     // Print the elapsed time
    printf("\nElapsed time: %.2f milliseconds\n", elapsed_time);

    return 0;
}
