#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
//#include <arm_neon.h>
using namespace std;

const int maxsize = 4000;
const int maxrow = 80000; 
const int numBasis = 40000;

//long long read = 0;
long long head, tail, freq;
int gRows[maxrow][maxsize];   
int gBasis[numBasis][maxsize]; 
map<int, int*>iToBasis;   
map<int, int>iToFirst;    
map<int, int*>ans;			
fstream RowFile("xh4.txt", ios::in | ios::out);
fstream BasisFile("xz4.txt", ios::in | ios::out);
void reset() {
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("xh4.txt", ios::in | ios::out);
	BasisFile.open("xz4.txt", ios::in | ios::out);
	iToBasis.clear();
	iToFirst.clear();
	ans.clear();
}

void readBasis() {        
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			cout << "消元子为" << i-1 << "行" << endl;
			return;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			if (!flag) {
				row = pos;
				flag = true;
				iToBasis.insert(pair<int, int*>(row, gBasis[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {  
	iToFirst.clear();
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("xh4.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows)); 
	string line;
	for (int i = 0; i < pos; i++) { 
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "被消元行为 "<<i<<" 行" << endl;
			return i;  
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			if (!flag) {
				iToFirst.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "读到最大row" << endl;
	return -1;

}

void update(int row) {
	bool flag = 0;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			if (!flag)
				flag = true;
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			int newfirst = pos + offset;
			iToFirst.erase(row);
			iToFirst.insert(pair<int, int>(row, newfirst));
			break;
		}
	}
	if (!flag) {
		iToFirst.erase(row);
	}
	return;
}

void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;
	while (true) {
		QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
		flag = readRowsFrom(begin);    
		QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
		head += (readEnd - readBegin);   

		int num = (flag == -1) ? maxrow : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;     
				if (iToBasis.find(first) != iToBasis.end()) {  
					int* basis = iToBasis.find(first)->second; 
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];    

					}
					update(i);  
				}
				else {   
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

void CACHE_GE() {
    long long readBegin, readEnd;
    int begin = 0;
    int flag;
    const int CACHE_LINE_SIZE = 64; // 假设缓存行大小为64字节

    while (true) {
        QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
        flag = readRowsFrom(begin);     // 读取被消元行
        QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
        head += (readEnd - readBegin);   // 除去读取数据的时间

        int num = (flag == -1) ? maxrow : flag;
        for (int i = 0; i < num; i++) {
            // 预取下一次迭代可能会访问到的数据
            _mm_prefetch(reinterpret_cast<const char*>(&gRows[i + 1]), _MM_HINT_T0);

            while (iToFirst.find(i) != iToFirst.end()) {
                int first = iToFirst.find(i)->second;      // first是首项
                if (iToBasis.find(first) != iToBasis.end()) {  // 存在首项为first消元子
                    int* basis = iToBasis.find(first)->second;  // 找到该消元子的数组
                    for (int j = 0; j < maxsize; j++) {
                        gRows[i][j] = gRows[i][j] ^ basis[j];     // 进行异或消元
                    }
                    update(i);   // 更新map
                } else {   // 升级为消元子
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

void AVX_GE() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;
	while (true) {
		QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
		flag = readRowsFrom(begin);     //读取被消元行
		QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
		head += (readEnd - readBegin);              //除去读取数据的时间
		int num = (flag == -1) ? maxrow : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //存在该消元子
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for (; j + 8 < maxsize; j += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					int j = 0;
					for (; j + 8 < maxsize; j += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
						_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
					}
					for (; j < maxsize; j++) {
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
void SSE_GE() {
    long long readBegin, readEnd;
    int begin = 0;
    int flag;
    while (true) {
        QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
        flag = readRowsFrom(begin); //读取被消元行
        QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
        head += (readEnd - readBegin); //除去读取数据的时间
        int num = (flag == -1) ? maxrow : flag;
        for (int i = 0; i < num; i++) {
            while (iToFirst.find(i) != iToFirst.end()) {
                int first = iToFirst.find(i)->second;
                if (iToBasis.find(first) != iToBasis.end()) { //存在该消元子
                    int* basis = iToBasis.find(first)->second;
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        __m128i vij = _mm_loadu_si128((__m128i*)&gRows[i][j]);
                        __m128i vj = _mm_loadu_si128((__m128i*)&basis[j]);
                        __m128i vx = _mm_xor_si128(vij, vj);
                        _mm_storeu_si128((__m128i*)&gRows[i][j], vx);
                    }
                    for (; j < maxsize; j++) {
                        gRows[i][j] = gRows[i][j] ^ basis[j];
                    }
                    update(i);
                } else {
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        __m128i vij = _mm_loadu_si128((__m128i*)&gRows[i][j]);
                        _mm_storeu_si128((__m128i*)&gBasis[first][j], vij);
                    }
                    for (; j < maxsize; j++) {
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

/*
void NEON_GE() {
    uint64_t readBegin, readEnd;
    int begin = 0;
    int flag;
    while (true) {
        QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
        flag = readRowsFrom(begin);     // 读取被消元行
        QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
        head += (readEnd - readBegin);   // 除去读取数据的时间
        int num = (flag == -1) ? maxrow : flag;
        for (int i = 0; i < num; i++) {
            while (iToFirst.find(i) != iToFirst.end()) {
                int first = iToFirst.find(i)->second;
                if (iToBasis.find(first) != iToBasis.end()) {  // 存在该消元子
                    float32x4_t* basis = reinterpret_cast<float32x4_t*>(iToBasis.find(first)->second);
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        float32x4_t vij = vld1q_f32(&gRows[i][j]);
                        float32x4_t vj = vld1q_f32(reinterpret_cast<float*>(&basis[j]));
                        float32x4_t vx = vsubq_f32(vij, vj); // 使用vsubq_f32进行向量减法操作
                        vst1q_f32(&gRows[i][j], vx);
                    }
                    for (; j < maxsize; j++) {
                        gRows[i][j] = gRows[i][j] ^ basis[j];
                    }
                    update(i);
                }
                else {
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4) {
                        float32x4_t vij = vld1q_f32(&gRows[i][j]);
                        vst1q_f32(reinterpret_cast<float*>(&gBasis[first][j]), vij);
                    }
                    for (; j < maxsize; j++) {
                        gBasis[first][j] = gRows[i][j];
                    }
                    iToBasis.insert(pair<int, int*>(first, reinterpret_cast<int*>(gBasis[first])));
                    ans.insert(pair<int, int*>(first, reinterpret_cast<int*>(gBasis[first])));
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
*/

int main() {
	double time1 = 0;
	double time2 = 0;
    double time3 = 0;
	//double time4 = 0;
		/*
		ofstream out("xj0.txt");
		ofstream out1("xj0(AVX).txt");
        ofstream out2("xj0(SSE).txt");
		//ofstream out3("xj0(NEON).txt");
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		readBasis();
		//writeResult();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time1 += (tail - head) * 1000 / freq;
		//writeResult(out);
		reset();
/*
		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time2 += (tail - head) * 1000 / freq;
		writeResult(out1);
		reset();
        readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		SSE_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "SSE time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time3 += (tail - head) * 1000 / freq;
		writeResult(out2);
*/
		reset();
		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		CACHE_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "CACHE_GE time:" << (tail - head) * 1000 / freq << "ms" << endl;
		reset();
/*
		 readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		NEON_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "NEON time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time4 += (tail - head) * 1000 / freq;
		writeResult(out2);
		reset();
*/
/*
		out.close();
		out1.close();
        out2.close();
	*/	
	//out3.close();
	//cout << "time1:" << time1 << endl << "time2:" << time2  <<endl << "time3:"<<time3;
	//cout<<"time4:" << time4 << endl;
}

