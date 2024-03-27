#include <iostream>
#include <vector>
#include <random>
#include <windows.h>
using namespace std;

// 生成一个随机整数
int generateRandomNumber(int min, int max) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> distribution(min, max);
    return distribution(gen);
}
// 生成随机矩阵和数组
void generateRandomMatrixAndVector(int n, vector<vector<int>>& matrix, vector<int>& vector1) {
    // 清空原有数据
    matrix.clear();
    vector1.clear();

    // 生成随机矩阵
    for (int i = 0; i < n; i++) {
        vector<int> row;
        for (int j = 0; j < n; j++) {
            row.push_back(generateRandomNumber(1, 100));  // 假设随机数范围为1到100
        }
        matrix.push_back(row);
    }
    // 生成随机数组
    for (int i = 0; i < n; i++) {
        vector1.push_back(generateRandomNumber(1, 100));
    }
}
vector<int> rowVectorProductCacheOptimized(vector<vector<int>>& matrix, vector<int>& vector1) {
    int n = matrix.size();
    vector<int> result(n, 0);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            result[col] += matrix[row][col] * vector1[row];
        }
    }
    return result;
}
int main() {
    int n;
    cout << "Enter the size n for the matrix and vector: ";
    cin >> n;
    vector<vector<int>> matrix;
    vector<int> vector1;
    generateRandomMatrixAndVector(n, matrix, vector1);
    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    //vector<int> result = rowVectorProductCacheOptimized(matrix, vector1);
    for (int i = 0; i < 20; i++) {
        rowVectorProductCacheOptimized(matrix, vector1);
    }
    QueryPerformanceCounter(&t2);
    cout << "TimeConsume: " << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;
/*
    cout << "Resulting row vector products:" << endl;
    for (int i = 0; i < result.size(); i++) {
        cout << result[i] << " ";
    }
    cout << endl;
    */
    return 0;
}
