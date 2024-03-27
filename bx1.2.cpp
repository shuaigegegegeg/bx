#include <iostream>
#include <vector>
using namespace std;
int sumOfNumbers(vector<int>numbers, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += numbers[i];
    }
    return sum;
}
int main() {
    int n;
    cout << "Enter a number n: ";
    cin >> n;
    vector<int> numbers;
    for (int i = 1; i <= n; ++i) {
        numbers.push_back(i);
    }
    clock_t begin, end;
    double cost;
    //开始记录
    begin = clock();
    int result = sumOfNumbers(numbers, n);
    
    for (int i = 0; i < 99; i++) {
        sumOfNumbers(numbers, n);
    }
    end = clock();
    cost = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << cost << endl;
    cout << "The sum of the numbers from 1 to " << n << " is: " << result <<endl;
    return 0;
}
