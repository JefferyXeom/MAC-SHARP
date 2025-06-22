//// For input/output operations and system call wrappers
#include <iostream>
#include <filesystem>
//// For string operations
#include <string> // For string operations
//// For exit function
#include <cstdlib> // For exit function
//// For timing
#include <chrono>

// for PCL
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>

// 
#include "MAC_SHARP.hpp"

#include <__msvc_filebuf.hpp>
#include <cblas.h>

// Terminal color codes for output
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

// const std::string RED = "\x1b[91m";
// const std::string GREEN = "\x1b[92m";
// const std::string YELLOW = "\x1b[93m";
// const std::string BLUE = "\x1b[94m";
// const std::string RESET = "\x1b[0m"; // 恢复默认颜色

bool low_inlier_ratio = false; // Flag for low inlier ratio
bool add_overlap = false; // Flag for adding overlap, maybe deprecated in future versions
bool no_logs = false; // Flag for no logs

int total_correspondences_num = 0; // Total number of correspondences

// 使用枚举类型来清晰地表示要使用的公式
enum class ScoreFormula {
    GAUSSIAN_KERNEL,
    QUADRATIC_FALLOFF
};

// Timing
// Only consider one iteration of the registration process!
std::chrono::high_resolution_clock::time_point start_time, end_time;
std::chrono::duration<double> elapsed_time;
vector<double> time_vec; // Vector to store elapsed times for each iteration

void timing(const int time_flag) {
    if (time_flag == 0) {
        // Start timing
        start_time = std::chrono::high_resolution_clock::now();
    } else if (time_flag == 1) {
        // End timing and calculate elapsed time
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = end_time - start_time;
        std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
        time_vec.push_back(elapsed_time.count()); // Store elapsed time in vector
    }
}


float mesh_resolution_calculation(const PointCloudPtr &pointcloud) {
    // Calculate the resolution of the pointcloud. We use the default mean root metric.
    float mr = 0; // mean root
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    vector<int> point_idx; // point index
    vector<float> point_dis; // point distance
    kdtree.setInputCloud(pointcloud);
    // Iterate each points and find the distance between it and its nearest neighbor
    for (int i = 0; i < pointcloud->points.size(); i++) {
        // One could declare query_points, x, y, z, mr_temp to the outside of the loop (without const) if a major
        // performance issue occurs. However. generally the modern compiler will optimize this automatically.
        pcl::PointXYZ query_point = pointcloud->points[i];
        kdtree.nearestKSearch(query_point, 2, point_idx, point_dis);
        mr += point_dis[1]; // distance from query point to nearest neighbor ([0] is the query itself)
        // const float x = pointcloud->points[point_idx[0]].x - pointcloud->points[point_idx[1]].x;
        // const float y = pointcloud->points[point_idx[0]].y - pointcloud->points[point_idx[1]].y;
        // const float z = pointcloud->points[point_idx[0]].z - pointcloud->points[point_idx[1]].z;
        // const float mr_temp = sqrt(x * x + y * y + z * z);
        // mr += mr_temp;
    }
    mr /= static_cast<float>(pointcloud->points.size());
    return mr; //approximate calculation
}

// For now we do not exactly know why there is a need to find the index of correspondences, but it is used in the original code.
// NOTE: the keypoints are not in the original point cloud, therefore a nearest search is required.
// Find the nearest point in the source and target point clouds for each correspondence, and assign the indices to the correspondences.
void find_index_for_correspondences(PointCloudPtr &src, PointCloudPtr &tgt, vector<Corre_3DMatch> &correspondences) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_tgt;
    kdtree_src.setInputCloud(src);
    kdtree_tgt.setInputCloud(tgt);
    vector<int> src_ind(1), des_ind(1);
    vector<float> src_dis(1), des_dis(1);
    for (auto &corr: correspondences) {
        pcl::PointXYZ src_pt, des_pt;
        src_pt = corr.src;
        des_pt = corr.tgt;
        kdtree_src.nearestKSearch(src_pt, 1, src_ind, src_dis);
        kdtree_tgt.nearestKSearch(des_pt, 1, des_ind, des_dis);
        corr.src_index = src_ind[0];
        corr.tgt_index = des_ind[0];
    }
}


std::vector<double> pdist_blas(const double *points_matrix, const int n, const int dims) {
    // 计算输出向量的大小
    const long long result_size = (long long) n * (n - 1) / 2;
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
std::vector<double> pdist_naive(const double *points_matrix, const int n, const int dims) {
    const long long result_size = (long long) n * (n - 1) / 2;
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

// Modified variable definition in for loop for acceleration
/**
 * @brief 使用 OpenMP 并行化循环计算 MATLAB 的 pdist 功能，返回一个 std::vector<double> 结果.
 * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @return              一个 std::vector<double>，包含 n*(n-1)/2 个成对距离.
 */
std::vector<double> pdist_naive_parallel(const double *points_matrix, const int n, const int dims) {
    const long long result_size = (long long) n * (n - 1) / 2;
    if (result_size <= 0) {
        return {};
    }
    std::vector<double> pdist_vector(result_size);

    // 将 i 循环并行化。每个线程处理不同的 i 值。
    // #pragma omp parallel for collapse(2) schedule(static)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // 直接计算出当前行(i)的配对在结果向量中的起始偏移量
        // 这是前 i 行所有配对的总数，是一个等差数列求和
        const long long offset = (long long) i * n - (long long) i * (i + 1) / 2;
        const int temp_i = i * dims;
        for (int j = i + 1; j < n; ++j) {
            double dist_sq = 0.0;
            const int temp_j = j * dims;
            for (int d = 0; d < dims; ++d) {
                const double diff = points_matrix[temp_i + d] - points_matrix[temp_j + d];
                dist_sq += diff * diff;
            }

            // 根据偏移量和 j 的位置计算出确切的索引 k
            // 不再需要共享的 k++
            const long long k = offset + (j - (i + 1));

            pdist_vector[k] = std::sqrt(dist_sq);
        }
    }
    return pdist_vector;
}

/**
 * @brief 使用 OpenBLAS 高效实现 MATLAB 的 pdist 功能，结果写入预分配的指针.
 * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
 */
void pdist_blas(const double *points_matrix, const int n, const int dims, double *result_buffer) {
    const long long result_size = (long long) n * (n - 1) / 2;
    if (result_size <= 0) {
        return; // 如果没有要计算的距离，则直接返回
    }

    // 内部的临时缓冲区仍然可以使用 std::vector，方便管理
    std::vector<double> gram_matrix(n * n);
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);

    std::vector<double> norms_sq(n);
    for (int i = 0; i < n; ++i) {
        norms_sq[i] = gram_matrix[i * n + i];
    }

    long long k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
            result_buffer[k] = std::sqrt(std::max(0.0, dist_sq));
            k++;
        }
    }
}

/**
 * @brief 使用纯 C++ 循环计算 MATLAB 的 pdist 功能，结果写入预分配的指针.
 * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
 */
void pdist_naive(const double *points_matrix, const int n, const int dims, double *result_buffer) {
    const long long result_size = (long long) n * (n - 1) / 2;
    if (result_size <= 0) {
        return;
    }

    long long k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dist_sq = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
                dist_sq += diff * diff;
            }
            result_buffer[k] = std::sqrt(dist_sq);
            k++;
        }
    }
}

/**
 * @brief 使用 OpenMP 并行化循环计算 pdist 功能，结果写入预分配的指针.
 * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
 * @param n             点的数量.
 * @param dims          点的维度.
 * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
 */
void pdist_naive_parallel(const double *points_matrix, const int n, const int dims, double *result_buffer) {
    const long long result_size = (long long) n * (n - 1) / 2;
    if (result_size <= 0) {
        return;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        long long offset = (long long) i * n - (long long) i * (i + 1) / 2;
        for (int j = i + 1; j < n; ++j) {
            double dist_sq = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
                dist_sq += diff * diff;
            }
            long long k = offset + (j - (i + 1));
            result_buffer[k] = std::sqrt(dist_sq);
        }
    }
}

/**
 * @brief 使用高斯核函数将距离平方矩阵转换为相似度得分矩阵 (并行化).
 * @param dist_sq_matrix 输入，指向 n*n 距离平方矩阵的指针.
 * @param score_matrix   输出，指向 n*n 得分矩阵的预分配内存.
 * @param size           矩阵的总元素数量 (n*n).
 * @param alpha_dis      高斯核函数的带宽参数 alpha.
 */
void gaussian_kernel_omp(const double *dist_sq_matrix, double *score_matrix, long long size, double alpha_dis) {
    // 预先计算出不变的系数部分，避免在循环中重复计算
    // 公式是 -1 / (2 * alpha * alpha)
    const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);

    // 使用 OpenMP 将这个巨大的循环并行化
#pragma omp parallel for schedule(static) default(none) shared(gamma, size, score_matrix, dist_sq_matrix)
    for (long long i = 0; i < size; ++i) {
        score_matrix[i] = std::exp(dist_sq_matrix[i] * gamma);
    }
}

// 这是一个计算完整 n*n 距离平方矩阵的函数 (之前pdist是向量，这里需要方阵)
void pdist_sq_matrix_omp(const double *points, int n, int dims, double *dist_sq_matrix) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double dist_sq = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = points[i * dims + d] - points[j * dims + d];
                dist_sq += diff * diff;
            }
            dist_sq_matrix[i * n + j] = dist_sq;
            dist_sq_matrix[j * n + i] = dist_sq; // 距离矩阵是对称的
        }
    }
}

/**
 * @brief 根据指定的公式，对距离向量逐元素计算得分 (OpenMP 并行化).
 * @param dist_vector    输入，指向距离向量的指针.
 * @param score_mat      输出，指向分数矩阵的预分配内存.
 * @param size           向量的长度.
 * @param formula        要使用的公式类型 (枚举).
 * @param alpha_dis      高斯核函数的带宽参数 (仅在 GAUSSIAN_KERNEL 时使用).
 * @param inlier_thresh  二次衰减的阈值参数 (仅在 QUADRATIC_FALLOFF 时使用).
 */
// paralle version of calculate_scores
void calculate_scores_omp(const vector<double> &dist_vector, vector<double> score_mat, long long size,
                          const ScoreFormula formula, const double alpha_dis, const double inlier_thresh) {
    const int total_correspondences_num_2 = 2 * total_correspondences_num - 2;
    switch (formula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
            // 预计算高斯核的 gamma 系数
            const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
#pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_mat, total_correspondences_num, total_correspondences_num_2)
            for (int i = 0; i < total_correspondences_num; ++i) {
                const long long temp_i = i * total_correspondences_num;
                const long long temp_i_2 = total_correspondences_num * i - i * (i + 1) / 2; // 计算当前行的偏移量
                for (int j = i + 1; j < total_correspondences_num; ++j) {
                    const long long temp_k = temp_i_2 + j - i - 1;
                    const double dist_sq = dist_vector.at(temp_k) * dist_vector[temp_k];
                    score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
                    score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
                    score_mat[j * total_correspondences_num + i] = score_mat[temp_i + j]; // 确保矩阵对称
                }
            }
            break;
        }
        case ScoreFormula::QUADRATIC_FALLOFF: {
            // 预计算二次衰减的系数，用乘法代替除法以提高效率
            const double inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
#pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_mat, total_correspondences_num, total_correspondences_num_2)
            for (int i = 0; i < total_correspondences_num; ++i) {
                const long long temp_i = i * total_correspondences_num;
                for (int j = i; j < total_correspondences_num; ++j) {
                    const long long temp_k = (total_correspondences_num_2 - i) * i / 2 + j;
                    const double dist_sq = dist_vector[temp_k] * dist_vector[temp_k];
                    // 使用 std::max 确保分数不会小于0，这通常是期望的行为
                    score_mat[temp_i + j] = std::max(0.0, 1.0 - dist_sq * inv_thresh_sq);
                    score_mat[j * total_correspondences_num + i] = score_mat[temp_i + j]; // 确保矩阵对称
                }
            }
        }
        break;
    }
}

/**
 * @brief 根据指定的公式，对距离向量逐元素计算得分 (OpenMP 并行化).
 * @param dist_vector    输入，指向距离向量的指针.
 * @param score_vector   输出，指向得分向量的预分配内存.
 * @param size           向量的长度.
 * @param formula        要使用的公式类型 (枚举).
 * @param alpha_dis      高斯核函数的带宽参数 (仅在 GAUSSIAN_KERNEL 时使用).
 * @param inlier_thresh  二次衰减的阈值参数 (仅在 QUADRATIC_FALLOFF 时使用).
 */
void calculate_scores_omp(const double* dist_vector, double* score_vector, long long size,
                          ScoreFormula formula, double alpha_dis, double inlier_thresh) {
    const long long dis_size = total_correspondences_num * (total_correspondences_num - 1) / 2;
    switch (formula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
            // 预计算高斯核的 gamma 系数
            const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
#pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_vector, dis_size)
            for (long long i = 0; i < dis_size ; ++i) {
                double dist_sq = dist_vector[i] * dist_vector[i];
                score_vector[i] = std::exp(dist_sq * gamma);
            }
            break;
        }

        case ScoreFormula::QUADRATIC_FALLOFF: {
            // 预计算二次衰减的系数，用乘法代替除法以提高效率
            const double inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
#pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_vector, dis_size)
            for (long long i = 0; i < dis_size; ++i) {
                double dist_sq = dist_vector[i] * dist_vector[i];
                // 使用 std::max 确保分数不会小于0，这通常是期望的行为
                score_vector[i] = std::max(0.0, 1.0 - dist_sq * inv_thresh_sq);
            }
            break;
        }
    }
}


/**
 * @brief 将存储上三角数据的压缩向量解包到一个完整的 Eigen 稠密矩阵中.
 * @param packed_upper_triangle 输入，只包含上三角元素的一维向量.
 * @param full_matrix           输出，将被填充的 n x n Eigen 矩阵.
 * @param diagonal_value        对角线元素应该被设置成什么值 (例如，距离矩阵为0，相似度矩阵为1).
 */
void unpack_upper_triangle(const std::vector<double>& packed_upper_triangle,
                           Eigen::MatrixXd& full_matrix,
                           double diagonal_value = 0.0)
{
    const int n = full_matrix.rows();
    if (full_matrix.cols() != n) {
        std::cerr << "Error: Output matrix must be square." << std::endl;
        return;
    }

    const long long expected_size = (long long)n * (n - 1) / 2;
    if (packed_upper_triangle.size() != expected_size) {
        std::cout << "Packed vector size: " << packed_upper_triangle.size() << std::endl;
        std::cout << "Expected size: " << expected_size << std::endl;
        std::cerr << "Error: Packed vector size does not match matrix dimensions." << std::endl;
        return;
    }

#pragma omp parallel for schedule(static) default(none) shared(full_matrix, diagonal_value, packed_upper_triangle, n)
    for (int i = 0; i < n; ++i) {
        // 1. 设置对角线元素
        full_matrix(i, i) = diagonal_value;
        // 2. 填充上三角和下三角部分
        for (int j = i + 1; j < n; ++j) {
            long long k = i * n - i * (i + 1) / 2 + j - i - 1; // 计算压缩向量的索引
            double value = packed_upper_triangle[k];
            full_matrix(i, j) = value; // 填充上三角
            full_matrix(j, i) = value; // 利用对称性，同时填充下三角
        }
    }
}


// Eigen::MatrixXd graph_construction(vector<Corre_3DMatch> &correspondences, float resolution,
//                                    bool second_order_graph_flag, const std::string &dataset_name,
//                                    const std::string &descriptor, float inlier_thresh) {
//     // Construct a graph from the correspondences. The graph is represented as an adjacency matrix
//     // TODO: Is there a more efficient way to construct or represent the graph?
//     // Total_correspondences_num is the size of the correspondences, which is also a global variable
//     Eigen::MatrixXd graph = Eigen::MatrixXd::Zero(total_correspondences_num, total_correspondences_num);
//     const float alpha_dis = 10 * resolution;
//     const long long distance_size = static_cast<long long>(total_correspondences_num) * (total_correspondences_num - 1) / 2;
//     // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
//     std::vector<double> src_mat(total_correspondences_num * 3, 0.0), tgt_mat(total_correspondences_num * 3, 0.0),
//             score_mat(total_correspondences_num * total_correspondences_num, 0.0), score_vec(distance_size);
//
//     // Construct the two points vectors (dimension n x 3)
//     std::cout << "Constructing the two points vectors..." << std::endl;
//     timing(0);
// #pragma omp parallel for schedule(static) default(none) shared(correspondences, src_mat, tgt_mat, total_correspondences_num)
//     for (int i = 0; i < total_correspondences_num; ++i) {
//         int temp_i = i * 3; // 3 for x, y, z
//         src_mat[temp_i] = correspondences[i].src.x;
//         tgt_mat[temp_i] = correspondences[i].tgt.x;
//         temp_i++; // Move to the next dimension
//         src_mat[temp_i] = correspondences[i].src.y;
//         tgt_mat[temp_i] = correspondences[i].tgt.y;
//         temp_i++; // Move to the next dimension
//         src_mat[temp_i] = correspondences[i].src.z;
//         tgt_mat[temp_i] = correspondences[i].tgt.z;
//     }
//     timing(1);
//
//     //
//     // TODO: Consider change to pointer version.
//     //
//     // std::vector<double> da(pdist_size);
//     // std::vector<double> db(pdist_size);
//     std::vector<double> d(distance_size, 0.0); // Initialize the distance vector with zeros
//
//     std::cout << "Calculating the distance vector..." << std::endl;
//     timing(0);
//     const std::vector<double> da = pdist_naive_parallel(src_mat.data(), total_correspondences_num, 3);
//     const std::vector<double> db = pdist_naive_parallel(tgt_mat.data(), total_correspondences_num, 3);
//     timing(1);
//
//     std::cout << "Calculating the scores vector... (abs)" << std::endl;
//     timing(0);
//     for (long long k = 0; k < distance_size; ++k) {
//         d[k] = std::abs(da[k] - db[k]);
//     }
//     timing(1);
//
//     std::cout << "Calculating the scores matrix... (score)" << std::endl;
//     timing(0);
//     // calculate_scores_omp(d, score_mat, distance_size, ScoreFormula::GAUSSIAN_KERNEL, alpha_dis,
//     //                      inlier_thresh);
//     calculate_scores_omp(d.data(), score_vec.data(), distance_size,
//                          ScoreFormula::GAUSSIAN_KERNEL, alpha_dis, inlier_thresh);
//     timing(1);
//
//     std::cout << "Unpacking the upper triangle of the score matrix into the graph..." << std::endl;
//     timing(0);
//     unpack_upper_triangle(score_vec, graph, 0.0); // Unpack the upper triangle of the score matrix into the graph
//     timing(1);
//     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map_for_copy(
//     //     score_mat.data(), total_correspondences_num, total_correspondences_num);
//     // graph = map_for_copy; // Eigen 的赋值操作符会处理好复制
//
//     if (second_order_graph_flag) {
//         // Second order graph construction
//         std::cout << "Constructing the second order graph..." << std::endl;
//         timing(0);
//         graph = graph.cwiseProduct(graph * graph);
//         timing(1);
//     }
//     return graph;
// }


inline float distance(pcl::PointXYZ& A, pcl::PointXYZ& B) {
    float distance = 0;
    float d_x = A.x - B.x;
    float d_y = A.y - B.y;
    float d_z = A.z - B.z;
    distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if(!isfinite(distance)){
        cout << distance <<"\t" <<A.x << " " <<A.y << " " << A.z << "\t" << B.x << " " <<B.y << " " << B.z <<endl;
    }
    return distance;
}

Eigen::MatrixXd graph_construction(vector<Corre_3DMatch> &correspondences, float resolution,
                                   bool second_order_graph_flag, const std::string &dataset_name,
                                   const std::string &descriptor, float inlier_thresh) {
    // Construct a graph from the correspondences. The graph is represented as an adjacency matrix
    // TODO: Is there a more efficient way to construct or represent the graph?
    // Total_correspondences_num is the size of the correspondences, which is also a global variable
    Eigen::MatrixXd graph = Eigen::MatrixXd::Zero(total_correspondences_num, total_correspondences_num);
    const float alpha_dis = 10 * resolution;
    const long long distance_size = static_cast<long long>(total_correspondences_num) * (total_correspondences_num - 1) / 2;
    // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
    std::vector<double> src_mat(total_correspondences_num * 3, 0.0), tgt_mat(total_correspondences_num * 3, 0.0),
            score_mat(total_correspondences_num * total_correspondences_num, 0.0), score_vec(distance_size);

    const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
    timing(0);
    if (second_order_graph_flag) {
#pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, correspondences, graph, gamma)
        for (int i = 0; i < total_correspondences_num; ++i) {
            Corre_3DMatch c1 = correspondences[i];
            for (int j = 0; j < total_correspondences_num; ++j) {
                Corre_3DMatch c2 = correspondences[j];
                float src_dis = distance (c1.src, c2.src);
                float tgt_dis = distance (c1.tgt, c2.tgt);
                float dis = src_dis - tgt_dis;
                double score = exp(dis * dis * gamma);
                score = (score < 0.7) ? 0 : score;
                graph(i, j) = score;
                graph(j, i) = score;
            }
        }
    } else {
        #pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, correspondences, graph, gamma)
        for (int i = 0; i < total_correspondences_num; ++i) {
            Corre_3DMatch c1 = correspondences[i];
            for (int j = 0; j < total_correspondences_num; ++j) {
                Corre_3DMatch c2 = correspondences[j];
                float src_dis = distance (c1.src, c2.src);
                float tgt_dis = distance (c1.tgt, c2.tgt);
                float dis = src_dis - tgt_dis;
                double score = exp(dis * dis * gamma);
                score = (score < 0.7) ? 0 : score;
                graph(i, j) = score;
                graph(j, i) = score;
            }
        }
    }
    std::cout << "Graph has been constructed" << std::endl;
    timing(1);
    timing(0);
    if (second_order_graph_flag) {
        // Eigen::setNbThreads(16);
        graph = graph.cwiseProduct(graph * graph);
    }
    return graph;
}

bool registration(const std::string &src_pointcloud_path, const std::string &tgt_pointcloud_path,
                  const std::string &corr_path, const std::string &gt_label_path, const std::string &gt_tf_path,
                  const std::string &output_path, const std::string &descriptor, double &RE, double &TE,
                  int &correct_est_num, int &gt_inlier_num, double &time_epoch) {
    bool second_order_graph_flag = true;
    bool use_icp_flag = true;
    bool instance_equal_flag = true;
    bool cluster_internal_evaluation_flag = true;
    bool use_top_k_flag = false;
    int max_estimate_num = INT_MAX; // ?
    low_inlier_ratio = false;
    add_overlap = false;
    no_logs = false;
    std::string metric = "MAC_SHARP";

    // Configure OpenBLAS threads (May not be used)
    int default_threads = openblas_get_num_threads();
    std::cout << "OpenBLAS default threads: " << default_threads << std::endl;
    int desired_threads = 1;
    openblas_set_num_threads(16);
    std::cout << "OpenBLAS now set to use " << openblas_get_num_threads() << " threads." << std::endl;

    // Configure OpenMP threads
    // 1. 查看默认情况下的最大线程数
    //    这通常等于您机器的逻辑核心数
    default_threads = omp_get_max_threads();
    std::cout << "Default max OpenMP threads: " << default_threads << std::endl;

    // 2. 设置希望使用的线程数
    desired_threads = 16;
    std::cout << "\nSetting OpenMP threads to: " << desired_threads << std::endl;
    omp_set_num_threads(desired_threads);

    // Set the number of threads for OpenMP, minus 2 to avoid overloading the system
    // omp_set_num_threads(omp_get_max_threads() - 2);

    int success_num = 0; // Number of successful registrations

    std::cout << BLUE << "Output path: " << output_path << RESET << std::endl;
    std::string input_data_path = corr_path.substr(0, corr_path.rfind('/'));
    std::string item_name = output_path.substr(output_path.rfind('/'), output_path.length());

    std::vector<std::pair<int, std::vector<int> > > matches; // one2k_match

    FILE *corr_file, *gt;
    corr_file = fopen(corr_path.c_str(), "r");
    gt = fopen(gt_label_path.c_str(), "r");

    if (corr_file == NULL) {
        std::cout << " error in loading correspondence data. " << std::endl;
        cout << corr_path << endl;
        exit(-1);
    }
    if (gt == NULL) {
        std::cout << " error in loading ground truth label data. " << std::endl;
        cout << gt_label_path << endl;
        exit(-1);
    }


    // overlap is deprecated, but kept for compatibility

    // FILE* ov;
    // std::vector<float>ov_corr_label;
    // float max_corr_weight = 0;
    // if (add_overlap && ov_label != "NULL")
    // {
    //     ov = fopen(ov_label.c_str(), "r");
    //     if (ov == NULL) {
    //         std::cout << " error in loading overlap data. " << std::endl;
    //         exit(-1);
    //     }
    //     cout << ov_label << endl;
    //     while (!feof(ov))
    //     {
    //         float value;
    //         fscanf(ov, "%f\n", &value);
    //         if(value > max_corr_weight){
    //             max_corr_weight = value;
    //         }
    //         ov_corr_label.push_back(value);
    //     }
    //     fclose(ov);
    //     cout << "load overlap data finished." << endl;
    // }

    // Load source and target point clouds
    PointCloudPtr raw_src(new pcl::PointCloud<pcl::PointXYZ>); // may not be used
    PointCloudPtr raw_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    float raw_src_resolution = 0.0f;
    float raw_tgt_resolution = 0.0f;

    PointCloudPtr pointcloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr pointcloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr pointcloud_src_kpts(new pcl::PointCloud<pcl::PointXYZ>); // source point cloud keypoints
    PointCloudPtr pointcloud_tgt_kpts(new pcl::PointCloud<pcl::PointXYZ>); // target point cloud keypoints

    pcl::PointCloud<pcl::Normal>::Ptr normal_src(new pcl::PointCloud<pcl::Normal>); // normal vector
    pcl::PointCloud<pcl::Normal>::Ptr normal_tgt(new pcl::PointCloud<pcl::Normal>); // normal vector

    std::vector<Corre_3DMatch> correspondences; // vector to store correspondences
    std::vector<int> gt_correspondences; // ground truth correspondences
    int inlier_num = 0; // Initialize inlier number
    float resolution = 0.0f; // Initialize resolution
    Eigen::Matrix4f gt_mat; // Ground truth transformation matrix

    FILE *gt_tf_file = fopen(gt_tf_path.c_str(), "r");
    if (gt_tf_file == NULL) {
        std::cerr << RED << "Error: Unable to open ground truth transformation file: " << gt_tf_path << RESET <<
                std::endl;
        return false;
    }
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(0, 0), &gt_mat(0, 1), &gt_mat(0, 2), &gt_mat(0, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(1, 0), &gt_mat(1, 1), &gt_mat(1, 2), &gt_mat(1, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(2, 0), &gt_mat(2, 1), &gt_mat(2, 2), &gt_mat(2, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(3, 0), &gt_mat(3, 1), &gt_mat(3, 2), &gt_mat(3, 3));
    fclose(gt_tf_file);

    if (pcl::io::loadPLYFile(src_pointcloud_path, *pointcloud_src) < 0) {
        std::cout << RED << "Error: Unable to load source point cloud file: " << src_pointcloud_path << RESET <<
                std::endl;
        return false;
    }
    if (pcl::io::loadPLYFile(tgt_pointcloud_path, *pointcloud_tgt) < 0) {
        std::cout << RED << "Error: Unable to load target point cloud file: " << tgt_pointcloud_path << RESET <<
                std::endl;
        return false;
    }

    // Load correspondences
    // TODO: After integrate the keypoints detection and description, the index and xyz coordinates are located in a
    // TODO: single file, so the correspondence loading function should be modified accordingly.
    while (!feof(corr_file)) {
        Corre_3DMatch match;
        pcl::PointXYZ src_point, tgt_point; // source point and target point in each match
        fscanf(corr_file, "%f %f %f %f %f %f\n",
               &src_point.x, &src_point.y, &src_point.z,
               &tgt_point.x, &tgt_point.y, &tgt_point.z);
        match.src = src_point;
        match.tgt = tgt_point;
        match.inlier_weight = 0; // Initialize inlier weight to 0
        correspondences.push_back(match);
    }
    fclose(corr_file);
    find_index_for_correspondences(pointcloud_src, pointcloud_tgt, correspondences);
    resolution = (mesh_resolution_calculation(pointcloud_src) + mesh_resolution_calculation(pointcloud_tgt)) / 2;

    // if (low_inlier_ratio) {
    //     if )
    //
    // }

    total_correspondences_num = static_cast<int>(correspondences.size());
    int value = 0;
    while (!feof(gt)) {
        fscanf(gt, "%d\n", &value);
        gt_correspondences.push_back(value);
        if (value == 1) {
            inlier_num++;
        }
    }
    fclose(gt);

    if (inlier_num == 0) {
        std::cout << YELLOW << "Warning: No inliers found in the ground truth correspondences." << RESET << std::endl;
        return false;
    }
    float inlier_ratio = static_cast<float>(inlier_num) / static_cast<float>(total_correspondences_num);
    std::cout << "Inlier ratio: " << inlier_ratio << std::endl;

    ////////////////////////////////
    /// Setting up evaluation thresholds.
    /// Dataset_name is not passed into this function
    /// therefore we set it manually.
    /// TODO: Make dataset_name a parameter of this function

    std::string dataset_name = "3dmatch";
    float RE_thresh, TE_thresh, inlier_thresh;
    if (dataset_name == "KITTI") {
        RE_thresh = 5;
        TE_thresh = 180;
        inlier_thresh = 1.8;
    } else if (dataset_name == "3dmatch" || dataset_name == "3dlomatch") {
        RE_thresh = 15;
        TE_thresh = 30;
        inlier_thresh = 0.1;
    } else if (dataset_name == "U3M") {
        inlier_thresh = 5 * resolution;
    }

    // NOTE: we do not consider the outer loop of the registration process, which is used to
    // repeat the registration process for multiple iterations.
    timing(0); // Start timing
    Eigen::Matrix graph = graph_construction(correspondences, resolution, second_order_graph_flag, dataset_name,
                                             descriptor, inlier_thresh);
    timing(1); // End timing

    return false;
}


int main(int argc, char **argv) {
    // Check if the required arguments are provided
    if (argc < 9) {
        std::cerr << RED << "Error: Not enough arguments provided. " << RESET << std::endl;
        std::cout << "Usage: " << argv[0] <<
                " <dataset_name> <descriptor> <src_pointcloud_path> <tgt_pointcloud_path> <corr_path> <gt_label_path> <gt_tf_path> <output_path>"
                << std::endl;
        return -1;
    }

    std::string dataset_name(argv[1]);
    // dataset name, previously used for different parameter settings. Evaluation metrics
    std::string descriptor(argv[2]); // descriptor name, e.g., "SHOT", "FPFH", etc.
    std::string src_pointcloud_path(argv[3]); // source point cloud file path
    std::string tgt_pointcloud_path(argv[4]); // target point cloud file path
    std::string corr_path(argv[5]); // correspondence file path
    std::string gt_label_path(argv[6]); // ground truth label file path, indicating which correspondences are inliers
    std::string gt_tf_path(argv[7]); // ground truth transformation file path
    std::string output_path(argv[8]); // output path for results

    // Check if the output directory exists, if not, create it
    std::error_code ec;
    if (std::filesystem::exists(output_path.c_str(), ec)) {
        if (std::filesystem::create_directory(output_path.c_str())) {
            std::cerr << "Error creating output directory: " << output_path << std::endl;
            return -1;
        }
    } else {
        std::cout << YELLOW << "Warning: Output directory already exists: " << output_path
                << ". Existing files may be overwritten." << std::endl
                << "Press anything to continue, or ctrl + c to exit." << RESET << std::endl;
        std::cin.get();
    }

    // Start execution
    int iterations = 1; // Number of iterations for ICP
    for (int i = 0; i < iterations; ++i) {
        double time_epoch = 0.0; // ?

        double RE, TE; // Rotation and translation errors
        int correct_est_num = 0; // Number of correct estimated correspondences
        int gt_inlier_num = 0; // Number of inliers in the ground truth correspondences
        bool estimate_success = registration(src_pointcloud_path, tgt_pointcloud_path, corr_path, gt_label_path,
                                             gt_tf_path,
                                             output_path,
                                             descriptor, RE, TE, correct_est_num, gt_inlier_num, time_epoch);

        std::ofstream results_out;
        // Output the evaluation results
        if (estimate_success) {
            std::string eva_result_path = output_path + "/evaluation_result.txt";
            results_out.open(eva_result_path.c_str(), std::ios::out);
            results_out.setf(std::ios::fixed, std::ios::floatfield);
            results_out << std::setprecision(6) << "RE: " << RE << std::endl
                    << "TE: " << TE << std::endl
                    << "Correct estimated correspondences: " << correct_est_num << std::endl
                    << "Inliers in ground truth correspondences: " << gt_inlier_num << std::endl
                    << "Total correspondences: " << total_correspondences_num << std::endl
                    << "Time taken for registration: " << time_epoch << " seconds" << std::endl;
            results_out.close();
        }

        // Output the status of the registration process
        std::string status_path = output_path + "/status.txt";
        results_out.open(status_path.c_str(), std::ios::out);
        results_out.setf(std::ios::fixed, std::ios::floatfield);
        results_out << std::setprecision(6) << "Time in one iteration: " << time_epoch <<
                " seconds, memory used in one iteration: " << std::endl;
        results_out.close();
    }


    return 0;
}
