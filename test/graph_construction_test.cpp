//
// Created by Jeffery_Xeom on 2025/6/20.
//

#include <iostream>
#include <vector>
#include <cmath>    // For std::sqrt, std::abs
#include <random>   // For modern C++ random number generation
#include <chrono>   // For timing
#include <omp.h> // 必须包含 OpenMP 头文件

// OpenBLAS a C library, so we use extern "C" in C++
extern "C" {
#include <cblas.h>
}

// 定义矩阵的维度
constexpr int N_POINTS = 6000;
constexpr int DIMS = 3;

/**
 * @brief 使用 OpenBLAS 高效实现 MATLAB 的 pdist 功能.
 * @param points_matrix 指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @return              一个 std::vector<double>，包含 n*(n-1)/2 个成对距离.
 */

extern "C" {
#include <cblas.h>
    // 声明 OpenBLAS 的线程控制函数


}

/**
 * @brief 获取 OpenMP 当前将使用的最大线程数.
 * @return 线程数.
*/
int get_omp_threads() {
    // omp_get_max_threads() 返回并行区域将使用的线程数
    return omp_get_max_threads();
}

/**
 * @brief 设置 OpenMP 将要使用的线程数.
 * @param num_threads 您希望设置的线程数量.
 */
void set_omp_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}


std::vector<double> pdist_blas(const double* points_matrix, const int n, const int dims) {
    // 计算输出向量的大小
    const long long result_size = (long long)n * (n - 1) / 2;
    if (result_size <= 0) {
        return {};
    }
    std::vector<double> pdist_vector(result_size);

    // --- 步骤 1: 计算格拉姆矩阵 G = points * points^T ---
    std::vector<double> gram_matrix(n * n);
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);

    // --- 步骤 2: 提取对角线元素 (范数的平方) ---
    std::vector<double> norms_sq(n);
    for (int i = 0; i < n; ++i) {
        norms_sq[i] = gram_matrix[i * n + i];
    }

    // --- 步骤 3: 装配最终的距离向量 (关键修改) ---
    long long k = 0; // pdist_vector 的索引
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // 从上三角部分读取 G[i,j]
            double dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
            // 避免因浮点数误差导致开方负数
            pdist_vector[k] = std::sqrt(std::max(0.0, dist_sq));
            k++;
        }
    }
    return pdist_vector;
}

/**
 * @brief 使用纯 C++ 循环计算 MATLAB 的 pdist 功能，不依赖任何外部库.
 * @param points_matrix 指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @return              一个 std::vector<double>，包含 n*(n-1)/2 个成对距离.
 */

std::vector<double> pdist_naive(const double* points_matrix, const int n, const int dims) {
    const long long result_size = (long long)n * (n - 1) / 2;
    if (result_size <= 0) {
        return {};
    }
    std::vector<double> pdist_vector(result_size);

    long long k = 0; // pdist_vector 的索引
    // 遍历所有唯一的点对 (i, j) where j > i
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {

            double dist_sq = 0.0;
            // 计算这对点之间距离的平方
            // (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
            for (int d = 0; d < dims; ++d) {
                double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
                dist_sq += diff * diff;
            }

            pdist_vector[k] = std::sqrt(dist_sq);
            k++;
        }
    }
    return pdist_vector;
}


std::vector<double> pdist_naive_parallel(const double* points_matrix, const int n, const int dims) {
    const long long result_size = (long long)n * (n - 1) / 2;
    if (result_size <= 0) {
        return {};
    }
    std::vector<double> pdist_vector(result_size);

    // 将 i 循环并行化。每个线程处理不同的 i 值。
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // 直接计算出当前行(i)的配对在结果向量中的起始偏移量
        // 这是前 i 行所有配对的总数，是一个等差数列求和
        long long offset = (long long)i * n - (long long)i * (i + 1) / 2;

        for (int j = i + 1; j < n; ++j) {
            double dist_sq = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
                dist_sq += diff * diff;
            }

            // 根据偏移量和 j 的位置计算出确切的索引 k
            // 不再需要共享的 k++
            long long k = offset + (j - (i + 1));

            pdist_vector[k] = std::sqrt(dist_sq);
        }
    }
    return pdist_vector;
}

int main() {

    int default_threads = openblas_get_num_threads();
    std::cout << "OpenBLAS default threads: " << default_threads << std::endl;

    // 设置为4个线程
    openblas_set_num_threads(1);
    std::cout << "OpenBLAS now set to use " << openblas_get_num_threads() << " threads." << std::endl;

    // 1. 查看默认情况下的最大线程数
    //    这通常等于您机器的逻辑核心数
    default_threads = get_omp_threads();
    std::cout << "Default max OpenMP threads: " << default_threads << std::endl;

    // 2. 设置希望使用的线程数，例如设置为 4
    int desired_threads = 1;
    std::cout << "\nSetting OpenMP threads to: " << desired_threads << std::endl;
    set_omp_threads(desired_threads);

    // 3. 验证设置是否生效
    int current_threads = get_omp_threads();
    std::cout << "Current max OpenMP threads: " << current_threads << std::endl;

    // 使用 C++ 的 vector 来自动管理内存，更安全
    std::vector<double> a(N_POINTS * DIMS);
    std::vector<double> b(N_POINTS * DIMS);

    // --- 数据初始化 ---
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    for (int i = 0; i < N_POINTS * DIMS; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // --- 计时开始 ---
    auto start = std::chrono::high_resolution_clock::now();

    // --- 步骤 1: da = pdist(a) ---
    std::cout << "Calculating pdist(a)..." << std::endl;
    std::vector<double> da = pdist_naive_parallel(a.data(), N_POINTS, DIMS);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nda finished in " << elapsed.count() << " seconds." << std::endl;

    // --- 步骤 2: db = pdist(b) ---
    std::cout << "Calculating pdist(b)..." << std::endl;
    std::vector<double> db = pdist_naive_parallel(b.data(), N_POINTS, DIMS);

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "\ndb finished in " << elapsed.count() << " seconds." << std::endl;

    // --- 步骤 3: d = abs(da - db) ---
    std::cout << "Calculating d = abs(da - db)..." << std::endl;
    const long long pdist_size = (long long)N_POINTS * (N_POINTS - 1) / 2;
    std::vector<double> d(pdist_size);
    for (long long k = 0; k < pdist_size; ++k) {
        d[k] = std::abs(da[k] - db[k]);
    }

    // --- 计时结束 ---
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    std::cout << "\nCalculations finished in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Size of pdist vector: " << d.size() << std::endl;

    // 打印前10个结果以供验证
    std::cout << "\nFirst 10 elements of the final result vector 'd':" << std::endl;
    for (int i = 0; i < 10 && i < d.size(); ++i) {
        std::cout << "d[" << i << "] = " << d[i] << std::endl;
    }

    return 0;
}