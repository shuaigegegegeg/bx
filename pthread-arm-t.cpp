#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <chrono>
#include <pthread.h>
#include <omp.h>
#include <cstring>
#include <arm_neon.h>
using namespace std;

#define NUM_THREADS 7

struct threadParam_t
{ // 参数数据结构
    int t_id;
    int num;
};

const int maxsize = 3000;
const int maxrow = 60000;    // 3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 100000; // 最多存储90000*100000的消元子

pthread_mutex_t lock; // 写入消元子时需要加锁

// long long read = 0;
long long head, tail, freq;

map<int, int *> iToBasis; // 首项为i的消元子的映射
map<int, int *> ans;      // 答案

fstream RowFile("xh1.txt", ios::in | ios::out);
fstream BasisFile("xz1.txt", ios::in | ios::out);

int gRows[maxrow][maxsize];    // 被消元行最多60000行，3000列
int gBasis[numBasis][maxsize]; // 消元子最多40000行，3000列

void reset()
{
    //	read = 0;
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    RowFile.close();
    BasisFile.close();
    RowFile.open("xh1.txt", ios::in | ios::out);
    BasisFile.open("xz1.txt", ios::in | ios::out);
    iToBasis.clear();

    ans.clear();
}

int readBasis()
{ // 读取消元子
    for (int i = 0; i < numBasis; i++)
    {
        if (BasisFile.eof())
        {
            cout << "读取消元子" << i - 1 << "行" << endl;
            return i - 1;
        }
        string tmp;
        bool flag = false;
        int row = 0;
        getline(BasisFile, tmp);
        stringstream s(tmp);
        int pos;
        while (s >> pos)
        {
            // cout << pos << " ";
            if (!flag)
            {
                row = pos;
                flag = true;
                iToBasis.insert(pair<int, int *>(row, gBasis[row]));
            }
            int index = pos / 32;
            int offset = pos % 32;
            gBasis[row][index] = gBasis[row][index] | (1 << offset);
        }
        flag = false;
        row = 0;
    }
}

int readRowsFrom(int pos)
{ // 读取被消元行
    if (RowFile.is_open())
        RowFile.close();
    RowFile.open("xh1.txt", ios::in | ios::out);
    memset(gRows, 0, sizeof(gRows)); // 重置为0
    string line;
    for (int i = 0; i < pos; i++)
    { // 读取pos前的无关行
        getline(RowFile, line);
    }
    for (int i = pos; i < pos + maxrow; i++)
    {
        int tmp;
        getline(RowFile, line);
        if (line.empty())
        {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i; // 返回读取的行数
        }
        bool flag = false;
        stringstream s(line);
        while (s >> tmp)
        {
            int index = tmp / 32;
            int offset = tmp % 32;
            gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
            flag = true;
        }
    }
    cout << "read max rows" << endl;
    return -1; // 成功读取maxrow行
}

int findfirst(int row)
{ // 寻找第row行被消元行的首项
    int first;
    for (int i = maxsize - 1; i >= 0; i--)
    {
        if (gRows[row][i] == 0)
            continue;
        else
        {
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--)
            {
                if (gRows[row][i] & (1 << k))
                {
                    offset = k;
                    break;
                }
            }
            first = pos + offset;
            return first;
        }
    }
    return -1;
}

void writeResult(ofstream &out)
{
    for (auto it = ans.rbegin(); it != ans.rend(); it++)
    {
        int *result = it->second;
        int max = it->first / 32 + 1;
        for (int i = max; i >= 0; i--)
        {
            if (result[i] == 0)
                continue;
            int pos = i * 32;
            // int offset = 0;
            for (int k = 31; k >= 0; k--)
            {
                if (result[i] & (1 << k))
                {
                    out << k + pos << " ";
                }
            }
        }
        out << endl;
    }
}

void GE()
{
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin); // 读取被消元行

    int num = (flag == -1) ? maxrow : flag;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num; i++)
    {
        while (findfirst(i) != -1)
        {                             // 存在首项
            int first = findfirst(i); // first是首项
            if (iToBasis.find(first) != iToBasis.end())
            {                                              // 存在首项为first消元子
                int *basis = iToBasis.find(first)->second; // 找到该消元子的数组
                for (int j = 0; j < maxsize; j++)
                {
                    gRows[i][j] = gRows[i][j] ^ basis[j]; // 进行异或消元
                }
            }
            else
            { // 升级为消元子
                for (int j = 0; j < maxsize; j++)
                {
                    gBasis[first][j] = gRows[i][j];
                }
                iToBasis.insert(pair<int, int *>(first, gBasis[first]));
                ans.insert(pair<int, int *>(first, gBasis[first]));
                break;
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "Ordinary time: " << elapsed.count() * 1000 << " ms" << endl;
}

void NEON_GE()
{
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin); // 读取被消元行
    int num = (flag == -1) ? maxrow : flag;

    // 使用chrono库进行时间测量
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num; i++)
    {
        while (findfirst(i) != -1)
        {
            int first = findfirst(i);
            if (iToBasis.find(first) != iToBasis.end())
            { // 存在该消元子
                int *basis = iToBasis.find(first)->second;
                int j = 0;
                for (; j + 4 < maxsize; j += 4)
                {
                    uint32x4_t vij = vld1q_u32((const uint32_t *)&gRows[i][j]);
                    uint32x4_t vj = vld1q_u32((const uint32_t *)&basis[j]);
                    uint32x4_t vx = veorq_u32(vij, vj);
                    vst1q_u32((uint32_t *)&gRows[i][j], vx);
                }
                for (; j < maxsize; j++)
                {
                    gRows[i][j] = gRows[i][j] ^ basis[j];
                }
            }
            else
            {
                int j = 0;
                for (; j + 4 < maxsize; j += 4)
                {
                    uint32x4_t vij = vld1q_u32((const uint32_t *)&gRows[i][j]);
                    vst1q_u32((uint32_t *)&gBasis[first][j], vij);
                }
                for (; j < maxsize; j++)
                {
                    gBasis[first][j] = gRows[i][j];
                }
                iToBasis.insert(pair<int, int *>(first, gBasis[first]));
                ans.insert(pair<int, int *>(first, gBasis[first]));
                break;
            }
        }
    }

    // 结束时间测量
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "NEON time:" << elapsed.count() * 1000 << "ms" << endl;
}
void *GE_lock_thread(void *param)
{

    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    int num = p->num;

    for (int i = t_id; i + NUM_THREADS < num; i += NUM_THREADS)
    {
        while (findfirst(i) != -1)
        {
            int first = findfirst(i); // first是首项
            if (iToBasis.find(first) != iToBasis.end())
            {                                              // 存在首项为first消元子
                int *basis = iToBasis.find(first)->second; // 找到该消元子的数组
                for (int j = 0; j < maxsize; j++)
                {
                    gRows[i][j] = gRows[i][j] ^ basis[j]; // 进行异或消元
                }
            }
            else
            {                              // 升级为消元子
                pthread_mutex_lock(&lock); // 如果第first行消元子没有被占用，则加锁
                if (iToBasis.find(first) != iToBasis.end())
                {
                    pthread_mutex_unlock(&lock);
                    continue;
                }
                for (int j = 0; j < maxsize; j++)
                {
                    gBasis[first][j] = gRows[i][j]; // 消元子的写入
                }
                iToBasis.insert(pair<int, int *>(first, gBasis[first]));
                ans.insert(pair<int, int *>(first, gBasis[first]));
                pthread_mutex_unlock(&lock); // 解锁
                break;
            }
        }
    }
    // cout << t_id << "线程完毕" << endl;
    pthread_exit(NULL);
    return NULL;
}

void GE_pthread()
{
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin); // 读取被消元行

    int num = (flag == -1) ? maxrow : flag;

    pthread_mutex_init(&lock, NULL); // 初始化锁

    pthread_t *handle = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));

    // 使用chrono库进行跨平台的高精度时间测量
    auto start = std::chrono::high_resolution_clock::now();

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        // 分配任务
        param[t_id].t_id = t_id;
        param[t_id].num = num;
        pthread_create(&handle[t_id], NULL, GE_lock_thread, &param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    // 结束时间测量
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "GE_pthread time: " << elapsed.count() * 1000 << " ms" << endl;

    free(handle);
    free(param);
    pthread_mutex_destroy(&lock);
}
void *NEON_lock_thread(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    int num = p->num;

    for (int i = t_id; i < num; i += NUM_THREADS)
    {
        while (findfirst(i) != -1)
        {
            int first = findfirst(i);
            if (iToBasis.find(first) != iToBasis.end())
            {
                int *basis = iToBasis.find(first)->second;
                int j = 0;
                for (; j + 4 < maxsize; j += 4)
                {
                    uint32x4_t vij = vld1q_u32(reinterpret_cast<uint32_t *>(&gRows[i][j]));
                    uint32x4_t vj = vld1q_u32(reinterpret_cast<uint32_t *>(&basis[j]));
                    uint32x4_t vx = veorq_u32(vij, vj);
                    vst1q_u32(reinterpret_cast<uint32_t *>(&gRows[i][j]), vx);
                }
                for (; j < maxsize; j++)
                {
                    gRows[i][j] = gRows[i][j] ^ basis[j];
                }
            }
            else
            {
                pthread_mutex_lock(&lock);
                if (iToBasis.find(first) == iToBasis.end())
                {
                    int j = 0;
                    for (; j + 4 < maxsize; j += 4)
                    {
                        uint32x4_t vij = vld1q_u32(reinterpret_cast<uint32_t *>(&gRows[i][j]));
                        vst1q_u32(reinterpret_cast<uint32_t *>(&gBasis[first][j]), vij);
                    }
                    for (; j < maxsize; j++)
                    {
                        gBasis[first][j] = gRows[i][j];
                    }
                    iToBasis[first] = gBasis[first];
                    ans[first] = gBasis[first];
                    pthread_mutex_unlock(&lock);
                    break;
                }
                pthread_mutex_unlock(&lock);
            }
        }
    }
    pthread_exit(NULL);
    return NULL;
}

void NEON_pthread()
{
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin); // 读取被消元行

    int num = (flag == -1) ? maxrow : flag;

    pthread_mutex_init(&lock, NULL); // 初始化锁

    pthread_t *handle = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));

    // 使用chrono库进行跨平台的高精度时间测量
    auto start = chrono::high_resolution_clock::now();

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    { // 分配任务
        param[t_id].t_id = t_id;
        param[t_id].num = num;
        pthread_create(&handle[t_id], NULL, NEON_lock_thread, &param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    // 结束时间测量
    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = finish - start;
    cout << "NEON_pthread time:" << elapsed.count() * 1000 << " ms" << endl;

    free(handle);
    free(param);
    pthread_mutex_destroy(&lock);
}
int main()
{
    for (int i = 0; i < 1; i++)
    {
        // 先前使用的QueryPerformanceFrequency在这里不再需要
        /*
                readBasis();
                GE(); // 不带NEON优化的高斯消元

                reset();

        readBasis();
        GE_pthread(); // 使用pthread的高斯消元

        reset();

                        readBasis();
                        NEON_GE(); // 使用NEON优化的高斯消元

                        reset();
*/
        readBasis();
        NEON_pthread(); // 结合NEON和pthread的高斯消元

        reset();
    }
}