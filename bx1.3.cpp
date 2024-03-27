#include <iostream>
#include <vector>
using namespace std;
int partialSum(vector<int> numbers, int n,int sum1,int sum2) {
    for (int i = 0;i<n; i += 2) {
        sum1 += numbers[i];
        sum2 += numbers[i + 1];
       
    }
    int sum = sum1 + sum2;
    return sum;
}
int main() {
    int m; cout << "enter n:"; cin >> m;
    vector<int> numbers;
    for (int i = 0; i < m; i++) {
        numbers.push_back(i + 1);
    }
    int n = numbers.size();
    int mid = n / 2;
    int sum1 = 0;
    int sum2 = 0;
    clock_t begin, end;
    double cost;
    begin = clock();
    int result = partialSum(numbers, n, sum1, sum2);
    for (int i = 0; i < 99; i++) {
         partialSum(numbers, n,sum1,sum2);
    }
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << cost << endl;
    cout << "The sum of the numbers is: " << result <<endl;
    return 0;
}
