//ARM平台上进行的特殊高斯消去
#include <iostream>
#include <fstream>
#include <map>
#include <cstring>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <arm_neon.h>
using namespace std;

const int maxsize = 3000;
const int maxrow = 60000;
const int numBasis = 40000;

map<int, int*> iToBasis;
map<int, int> iToFirst;
map<int, int*> ans;

int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

fstream RowFile;
fstream BasisFile;

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    
    RowFile.close();
    BasisFile.close();

    RowFile.open("xh0.txt", ios::in | ios::out);
    BasisFile.open("xz0.txt", ios::in | ios::out);

    iToBasis.clear();
    iToFirst.clear();
    ans.clear();
}

void readBasis() {
    BasisFile.clear();
    BasisFile.seekg(0, ios::beg);

    for (int i = 0; i < numBasis; i++) {
        if (BasisFile.eof()) {
            cout << "读取消元子" << i - 1 << "行" << endl;
            return;
        }

        string tmp;
        getline(BasisFile, tmp);
        if (tmp.empty()) continue;

        stringstream ss(tmp);
        bool isFirst = true;
        int pos;

        while (ss >> pos) {
            int index = pos / 32;
            int offset = pos % 32;

            if (isFirst) {
                iToBasis[pos] = gBasis[pos];
                isFirst = false;
            }

            gBasis[pos][index] |= (1 << offset);
        }
    }
}

int readRowsFrom(int pos) {
    iToFirst.clear();
    RowFile.close();
    RowFile.open("xh0.txt", ios::in | ios::out);

    memset(gRows, 0, sizeof(gRows));

    RowFile.seekg(0, ios::beg);
    string line;

    for (int i = 0; i < pos; i++) {
        getline(RowFile, line);
    }

    for (int i = pos; i < pos + maxrow; i++) {
        if (!getline(RowFile, line) || line.empty()) {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i;
        }

        stringstream ss(line);
        int tmp;
        bool isFirst = true;

        while (ss >> tmp) {
            int index = tmp / 32;
            int offset = tmp % 32;

            gRows[i - pos][index] |= (1 << offset);

            if (isFirst) {
                iToFirst[i - pos] = tmp;
                isFirst = false;
            }
        }
    }

    cout << "Read max rows" << endl;
    return -1;
}

void update(int row) {
    bool found = false;

    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] == 0) continue;

        int index = i * 32;
        for (int k = 31; k >= 0; k--) {
            if (gRows[row][i] & (1 << k)) {
                iToFirst[row] = index + k;
                found = true;
                break;
            }
        }

        if (found) break;
    }

    if (!found) {
        iToFirst.erase(row);
    }
}

void writeResult(ofstream& out) {
    for (auto it = ans.rbegin(); it != ans.rend(); ++it) {
        int* result = it->second;

        for (int i = (it->first / 32); i >= 0; i--) {
            if (result[i] == 0) continue;

            int pos = i * 32;
            for (int k = 31; k >= 0; k--) {
                if (result[i] & (1 << k)) {
                    out << pos + k << " ";
                }
            }
        }

        out << endl;
    }
}
long long head0 = 0;
void GE() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;
	while (true) {
		 auto start_time = std::chrono::high_resolution_clock::now();
        flag = readRowsFrom(begin);
        auto end_time = std::chrono::high_resolution_clock::now();
        head0 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

		int num = (flag == -1) ? maxrow : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;      //first是首项
				if (iToBasis.find(first) != iToBasis.end()) {  //存在首项为first消元子
					int* basis = iToBasis.find(first)->second;  //找到该消元子的数组
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];     //进行异或消元

					}
					update(i);   //更新map
				}
				else {   //升级为消元子
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxrow;
		else
			break;
	}

}

long long head = 0;
void NEON_GE() {
    long long readBegin, readEnd;
    int begin = 0;
    int flag;

    while (true) {
        auto start_time = std::chrono::high_resolution_clock::now();
        flag = readRowsFrom(begin);
        auto end_time = std::chrono::high_resolution_clock::now();
        head += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

        int num = (flag == -1) ? maxrow : flag;
        for (int i = 0; i < num; i++) {
            while (iToFirst.find(i) != iToFirst.end()) {
                int first = iToFirst.find(i)->second;
                if (iToBasis.find(first) != iToBasis.end()) {
                    int* basis = iToBasis.find(first)->second;
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        int32x4_t vij = vld1q_s32(&gRows[i][j]);
                        int32x4_t vj = vld1q_s32(&basis[j]);
                        int32x4_t vx = veorq_s32(vij, vj);
                        vst1q_s32(&gRows[i][j], vx);
                    }
                    for (; j < maxsize; j++) {
                        gRows[i][j] ^= basis[j];
                    }
                    update(i);
                } else {
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        int32x4_t vij = vld1q_s32(&gRows[i][j]);
                        vst1q_s32(&gBasis[first][j], vij);
                    }
                    for (; j < maxsize; j++) {
                        gBasis[first][j] = gRows[i][j];
                    }
                    iToBasis.insert({first, gBasis[first]});
                    ans.insert({first, gBasis[first]});
                    iToFirst.erase(i);
                }
            }
        }

        if (flag == -1) {
            begin += maxrow;
        } else {
            break;
        }
    }
}


int main() {
    ofstream resultFile("result.txt");
    if (!resultFile.is_open()) {
        cerr << "Could not open result file." << endl;
        return 1;
    }

    reset();
    readBasis();
    int readCount = readRowsFrom(0);
    auto start_time = std::chrono::high_resolution_clock::now();
    NEON_GE();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    std::cout << "NEON_GE function time: " << duration/1000000 << " ms" << std::endl;
/*
    reset();
    readBasis();
    int readCount0 = readRowsFrom(0);
    auto start_time0 = std::chrono::high_resolution_clock::now();
    GE();
    auto end_time0 = std::chrono::high_resolution_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time0 - start_time0).count();
    std::cout << "GE function time: " << duration0/1000000 << " ms" << std::endl;
*/
    /*
    if (readCount > 0) {
        cout << "Successfully read rows from 0 to " << readCount << endl;
    }
*/
    // Example usage
    update(0);
    writeResult(resultFile);

    resultFile.close();
    return 0;
}
