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
#include <pcl/filters/filter.h>
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

int clique_num = 0; // Number of cliques found

// 使用枚举类型来清晰地表示要使用的公式
enum class ScoreFormula {
    GAUSSIAN_KERNEL,
    QUADRATIC_FALLOFF
};

// Timing
// Only consider one iteration of the registration process!
std::chrono::high_resolution_clock::time_point start_time, end_time;
std::chrono::duration<double> elapsed_time;
std::vector<double> time_vec; // Vector to store elapsed times for each iteration

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
    std::vector<int> point_idx; // point index
    std::vector<float> point_dis; // point distance
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
// Find the nearest point in the source and target key point clouds for each correspondence, and assign the indices to the correspondences.
// Another note is that, the original MAC++ does not use the corr_ind file for correspondences indexing. Therefore this function is necessary.
void find_index_for_correspondences(PointCloudPtr &src, PointCloudPtr &tgt, std::vector<Corre_3DMatch> &correspondences) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_tgt;
    kdtree_src.setInputCloud(src);
    kdtree_tgt.setInputCloud(tgt);
    std::vector<int> src_ind(1), des_ind(1);
    std::vector<float> src_dis(1), des_dis(1);
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
void calculate_scores_omp(const std::vector<double> &dist_vector, std::vector<double> score_mat, long long size,
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
void calculate_scores_omp(const double *dist_vector, double *score_vector, long long size,
                          ScoreFormula formula, double alpha_dis, double inlier_thresh) {
    const long long dis_size = total_correspondences_num * (total_correspondences_num - 1) / 2;
    switch (formula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
            // 预计算高斯核的 gamma 系数
            const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
#pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_vector, dis_size)
            for (long long i = 0; i < dis_size; ++i) {
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
void unpack_upper_triangle(const std::vector<double> &packed_upper_triangle,
                           Eigen::MatrixXd &full_matrix,
                           double diagonal_value = 0.0) {
    const int n = full_matrix.rows();
    if (full_matrix.cols() != n) {
        std::cerr << "Error: Output matrix must be square." << std::endl;
        return;
    }

    const long long expected_size = (long long) n * (n - 1) / 2;
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


inline float get_distance(pcl::PointXYZ &A, pcl::PointXYZ &B) {
    float distance = 0;
    float d_x = A.x - B.x;
    float d_y = A.y - B.y;
    float d_z = A.z - B.z;
    distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if (!isfinite(distance)) {
        cout << distance << "\t" << A.x << " " << A.y << " " << A.z << "\t" << B.x << " " << B.y << " " << B.z << endl;
    }
    return distance;
}

// igraph need eigen matrix be double type
Eigen::MatrixXd graph_construction(vector<Corre_3DMatch> &correspondences, float resolution,
                                   bool second_order_graph_flag, const std::string &dataset_name,
                                   const std::string &descriptor, float inlier_thresh) {
    // Construct a graph from the correspondences. The graph is represented as an adjacency matrix
    // TODO: Is there a more efficient way to construct or represent the graph?
    // Total_correspondences_num is the size of the correspondences, which is also a global variable
    Eigen::MatrixXd graph = Eigen::MatrixXd::Zero(total_correspondences_num, total_correspondences_num);
    const float alpha_dis = 10 * resolution;
    const long long distance_size = static_cast<long long>(total_correspondences_num) * (total_correspondences_num - 1)
                                    / 2;
    // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
    std::vector<float> src_mat(total_correspondences_num * 3, 0.0), tgt_mat(total_correspondences_num * 3, 0.0),
            score_mat(total_correspondences_num * total_correspondences_num, 0.0), score_vec(distance_size);

    const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
    timing(0);
    ScoreFormula formula = ScoreFormula::GAUSSIAN_KERNEL;
    switch (formula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
#pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, correspondences, graph, gamma)
            for (int i = 0; i < total_correspondences_num; ++i) {
                Corre_3DMatch c1 = correspondences[i];
                for (int j = 0; j < total_correspondences_num; ++j) {
                    Corre_3DMatch c2 = correspondences[j];
                    float src_dis = get_distance(c1.src, c2.src);
                    float tgt_dis = get_distance(c1.tgt, c2.tgt);
                    float dis = src_dis - tgt_dis;
                    double score = exp(dis * dis * gamma);
                    score = (score < 0.7) ? 0 : score;
                    graph(i, j) = score;
                    graph(j, i) = score;
                }
            }
            break;
        }
        case ScoreFormula::QUADRATIC_FALLOFF: {
#pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, correspondences, graph, gamma)
            for (int i = 0; i < total_correspondences_num; ++i) {
                Corre_3DMatch c1 = correspondences[i];
                for (int j = 0; j < total_correspondences_num; ++j) {
                    Corre_3DMatch c2 = correspondences[j];
                    float src_dis = get_distance(c1.src, c2.src);
                    float tgt_dis = get_distance(c1.tgt, c2.tgt);
                    float dis = src_dis - tgt_dis;
                    double score = exp(dis * dis * gamma);
                    score = (score < 0.7) ? 0 : score;
                    graph(i, j) = score;
                    graph(j, i) = score;
                }
            }
            break;
        }
    }
    std::cout << "Graph has been constructed" << std::endl;
    timing(1);
    timing(0);
    // Second order graphing is time-consuming, size 6000 will use up to 2s
    if (second_order_graph_flag) {
        // Eigen::setNbThreads(16);
        graph = graph.cwiseProduct(graph * graph);
    }
    return graph;
}


// TODO: This function needs optimization
float otsu_thresh(vector<float> all_scores)
{
    int i;
    int Quant_num = 100;
    float score_sum = 0.0;
    float fore_score_sum = 0.0;
    std::vector<int> score_Hist(Quant_num, 0);
    std::vector<float> score_sum_Hist(Quant_num, 0.0);
    float max_score_value, min_score_value;
    for (i = 0; i < all_scores.size(); i++)
    {
        score_sum += all_scores[i];
    }
    sort(all_scores.begin(), all_scores.end());
    max_score_value = all_scores[all_scores.size() - 1];
    min_score_value = all_scores[0];
    float Quant_step = (max_score_value - min_score_value) / Quant_num;
    for (i = 0; i < all_scores.size(); i++)
    {
        int ID = all_scores[i] / Quant_step;
        if (ID >= Quant_num) ID = Quant_num - 1;
        score_Hist[ID]++;
        score_sum_Hist[ID] += all_scores[i];
    }
    float fmax = -1000;
    int n1 = 0, n2;
    float m1, m2, sb;
    float thresh = (max_score_value - min_score_value) / 2;//default value
    for (i = 0; i < Quant_num; i++)
    {
        float Thresh_temp = i * (max_score_value - min_score_value) / float (Quant_num);
        n1 += score_Hist[i];
        if (n1 == 0) continue;
        n2 = all_scores.size() - n1;
        if (n2 == 0) break;
        fore_score_sum += score_sum_Hist[i];
        m1 = fore_score_sum / n1;
        m2 = (score_sum - fore_score_sum) / n2;
        sb = (float )n1 * (float )n2 * pow(m1 - m2, 2);
        if (sb > fmax)
        {
            fmax = sb;
            thresh = Thresh_temp;
        }
    }
    return thresh;
}



// Comparison functions
bool compare_vote_score(const Vote& v1, const Vote& v2) {
    return v1.score > v2.score;
}
bool compare_local_score(const local &l1, const local &l2){
    return l1.score > l2.score;
}
bool compare_corres_ind(const Corre_3DMatch& c1, const Corre_3DMatch& c2){
    return c1.tgt_index < c2.tgt_index;
}

// Find the vertex score based on clique edge weight.
// Select the correspondences who have high scores
// sampled_ind is the order of the correspondences that are selected which score is higher than average
// remain is the index of the neighbor of sampled_ind that also locate in the high score clique
void clique_sampling(Eigen::MatrixXd &graph, const igraph_vector_int_list_t *cliques, std::vector<int> &sampled_ind, std::vector<int> &remain){
    // the clear process may be rebundant
    // remain.clear();
    // sampled_ind.clear();
    unordered_set<int> visited;
    std::vector<local> result(total_correspondences_num);
    // Assign current index
#pragma omp parallel for
    for(int i = 0; i < total_correspondences_num; i++){
        result[i].current_ind = i;
    }
    // compute the weight of each clique
    // Weight of each clique is the sum of the weights of all edges in the clique
#pragma omp parallel for
    for(int i = 0; i < clique_num; i++){
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(cliques, i);
        double weight = 0.0;
        int length = igraph_vector_int_size(v); // size of the clique

        for (int j = 0; j < length; j++)
        {
            int a = static_cast<int>(VECTOR(*v)[j]);
            for (int k = j + 1; k < length; k++)
            {
                int b = static_cast<int>(VECTOR(*v)[k]);
                weight += graph(a, b);
            }
        }
        // assign the weight to each correspondence in the clique
        for (int j = 0; j < length; j++)
        {
            int k = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            result[k].clique_ind_score.emplace_back(i, weight, false); // Weight of k-th correspondecnce in i-th clique
        }
    }

    float avg_score = 0;
    // sum the scores and assign it to the score member variable
#pragma omp parallel for
    for(int i = 0; i < total_correspondences_num; i++){
        result[i].score = 0;
        // compute the score of each correspondence, clique_ind_score.size() is the number of cliques that the correspondence belongs to
        for(int j = 0; j < result[i].clique_ind_score.size(); j ++){
            result[i].score += result[i].clique_ind_score[j].score;
        }
#pragma omp critical
        {
            avg_score += result[i].score;
        }
    }

    //
    sort(result.begin(), result.end(), compare_local_score); //所有节点从大到小排序

    if( clique_num <= total_correspondences_num ){ // 如果clique数目小于等于correspondence数目
        for(int i = 0; i < clique_num; i++){ // Assign all cliques indexes to the remain in order.
            remain.push_back(i);
        }
        for(int i = 0; i < total_correspondences_num; i++){ // sampled_ind 中存放的是被选中的correspondence的index
            if(!result[i].score){ // skip if the score of correspondence is 0
                continue;
            }
            sampled_ind.push_back(result[i].current_ind); // only keep index whose correspondence has a non-zero score
        }
        return;
    }

    //
    avg_score /= static_cast<float>(total_correspondences_num);
    int max_cnt = 10;  //default 10
    for(int i = 0; i < total_correspondences_num; i++){
        // We only consider the correspondences whose score is greater than the average score
        // This can filter low score vertex (vertex and correspondence are the same thing)
        if(result[i].score < avg_score) break;
        sampled_ind.push_back(result[i].current_ind); // Only keep index of correspondence whose score is higher than the average score, ordered
        // sort the clique_ind_score of each correspondence from large to small
        sort(result[i].clique_ind_score.begin(), result[i].clique_ind_score.end(), compare_vote_score); //局部从大到小排序
        int selected_cnt = 1;
        // Check top 10 neighbors of each correspondence in high score clique
        for(int j = 0; j < result[i].clique_ind_score.size(); j++){
            if(selected_cnt > max_cnt) break;
            int ind = result[i].clique_ind_score[j].current_index;
            if(visited.find(ind) == visited.end()){
                visited.insert(ind);
            }
            else{
                continue;
            }
            selected_cnt ++;
        }
    }
    // Keep the correspondences that have high neighboring score.
    // Its neighbor has high score, and it is in its neighbor's high score clique
    remain.assign(visited.begin(), visited.end()); // no order
}

// TODO: Chech this function
// TODO: This function is not optimized
// Our source target pair is a normal but non-invertible function (surjective, narrowly), which means a source can only have a single target,
// but a target may have many sources. This function is used to find target source pair, where target paired with various sources.
void make_tgt_src_pair(const std::vector<Corre_3DMatch>& correspondence, std::vector<pair<int, std::vector<int>>>& tgt_src){ //需要读取保存的kpts, 匹配数据按照索引形式保存
    assert(correspondence.size() > 1); // 保留一个就行
    if (correspondence.size() < 2) {
        std::cerr << "The correspondence vector is empty." << std::endl;
    }
    tgt_src.clear();
    std::vector<Corre_3DMatch> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    sort(corr.begin(), corr.end(), compare_corres_ind); // sort by target index increasing order
    int tgt = corr[0].tgt_index;
    std::vector<int>src;
    src.push_back(corr[0].src_index);
    for(int i = 1; i < corr.size(); i++){
        if(corr[i].tgt_index != tgt){
            tgt_src.emplace_back(tgt, src);
            src.clear();
            tgt = corr[i].tgt_index;
        }
        src.push_back(corr[i].src_index);
    }
    corr.clear();
    corr.shrink_to_fit();
}

// TODO: This function is not optimized
// TODO: We only get the logic check
void weight_svd(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXf& weights, float weight_threshold, Eigen::Matrix4f& trans_Mat) {
    for (int i = 0; i < weights.size(); i++)
    {
        weights(i) = (weights(i) < weight_threshold) ? 0 : weights(i);
    }
    //weights升维度
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> weight;
    Eigen::VectorXf ones = weights;
    ones.setOnes();
    weight = (weights * ones.transpose());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;
    //构建对角阵
    Identity.setIdentity();
    weight = (weights * ones.transpose()).cwiseProduct(Identity);
    pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_pts);
    pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_pts);
    //获取点云质心
    src_it.reset(); des_it.reset();
    Eigen::Matrix<float, 4, 1> centroid_src, centroid_des;
    pcl::compute3DCentroid(src_it, centroid_src);
    pcl::compute3DCentroid(des_it, centroid_des);

    //去除点云质心
    src_it.reset(); des_it.reset();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
    pcl::demeanPointCloud(src_it, centroid_src, src_demean);
    pcl::demeanPointCloud(des_it, centroid_des, des_demean);

    //计算加权协方差矩阵
    Eigen::Matrix<float, 3, 3> H = (src_demean * weight * des_demean.transpose()).topLeftCorner(3, 3);
    //cout << H << endl;

    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<float, 3, 3> u = svd.matrixU();
    Eigen::Matrix<float, 3, 3> v = svd.matrixV();

    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0)
    {
        for (int x = 0; x < 3; ++x)
            v(x, 2) *= -1;
    }

    Eigen::Matrix<float, 3, 3> R = v * u.transpose(); //正交矩阵的乘积还是正交矩阵，因此R的逆等于R的转置

    // Return the correct transformation
    Eigen::Matrix<float, 4, 4> Trans;
    Trans.setIdentity();
    Trans.topLeftCorner(3, 3) = R;
    const Eigen::Matrix<float, 3, 1> Rc(R * centroid_src.head(3));
    Trans.block(0, 3, 3, 1) = centroid_des.head(3) - Rc;
    trans_Mat = Trans;
}



// TODO: This function is not optimized
// TODO: We only get the logic check
// Overall Average Mean Absolute Error（OAMAE）
// | 指标               | 说明                     | 是否对异常敏感    |
// | ---------------- | ---------------------- | ---------- |
// | **OAMAE**        | 总体平均的绝对误差              | 否（比RMSE稳健） |
// | RMSE             | 平方误差平均，强调大误差           | 是          |
// | Chamfer Distance | 最近邻距离之和或平均             | 常用于点云任务    |
// | EMD              | Earth Mover's Distance | 更严谨但计算开销大  |

float OAMAE(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &des_src, float thresh){
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for(auto & i : des_src){
        int des_ind = i.first;
        vector<int> src_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        for(auto & e : src_ind){
            if(!pcl::isFinite(src_trans->points[e])) continue;
            //计算距离
            float distance = get_distance(src_trans->points[e], raw_des->points[des_ind]);
            if (distance < thresh)
            {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}

float calculate_rotation_error(Eigen::Matrix3f& est, Eigen::Matrix3f& gt) {
    float tr = (est.transpose() * gt).trace();
    return acos(min(max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

float calculate_translation_error(Eigen::Vector3f& est, Eigen::Vector3f& gt) {
    Eigen::Vector3f t = est - gt;
    return sqrt(t.dot(t)) * 100;
}

float evaluation_trans(vector<Corre_3DMatch>& correspondnece, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& trans, float metric_thresh, const string &metric, float resolution) {
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src_corr_pts, *src_trans, trans);
    src_trans->is_dense = false;
    vector<int>mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if(!src_trans->size()) return 0;
    float score = 0.0;
    int inlier = 0;
    int corr_num = src_corr_pts->points.size();
    for (int i = 0; i < corr_num; i++)
    {
        float dist = get_distance(src_trans->points[i], des_corr_pts->points[i]);
        float w = 1;
        if (add_overlap)
        {
            w = correspondnece[i].score;
        }
        if (dist < metric_thresh)
        {
            inlier++;
            if (metric == "inlier")
            {
                score += 1*w;//correspondence[i].inlier_weight; <- commented by the MAC++ author
            }
            else if (metric == "MAE")
            {
                score += (metric_thresh - dist)*w / metric_thresh;
            }
            else if (metric == "MSE")
            {
                score += pow((metric_thresh - dist), 2)*w / pow(metric_thresh, 2);
            }
        }
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}


bool evaluation_est(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float re_thresh, float te_thresh, float& RE, float& TE) {
    Eigen::Matrix3f rotation_est, rotation_gt;
    Eigen::Vector3f translation_est, translation_gt;
    rotation_est = est.topLeftCorner(3, 3);
    rotation_gt = gt.topLeftCorner(3, 3);
    translation_est = est.block(0, 3, 3, 1);
    translation_gt = gt.block(0, 3, 3, 1);

    RE = calculate_rotation_error(rotation_est, rotation_gt);
    TE = calculate_translation_error(translation_est, translation_gt);
    if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh)
    {
        return true;
    }
    return false;
}




bool registration(const std::string &src_pointcloud_path, const std::string &tgt_pointcloud_path,
                  const std::string &corr_path, const std::string &gt_label_path, const std::string &gt_tf_path,
                  const std::string &output_path, const std::string &descriptor, double &RE, double &TE,
                  int &correct_est_num, int &gt_inlier_num, double &time_epoch) {
    // temporary variables. Delete these after unifying the data load
    std::string src_pointcloud_kpts_path = "./test_data/src.pcd";
    std::string tgt_pointcloud_kpts_path = "./test_data/tgt.pcd";

    bool second_order_graph_flag = true;
    bool use_icp_flag = true;
    bool instance_equal_flag = true;
    bool cluster_internal_evaluation_flag = true;
    bool use_top_k_flag = false;
    int max_estimate_num = INT_MAX; // ?
    low_inlier_ratio = false;
    add_overlap = false;
    no_logs = false;
    std::string metric = "MAE";

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
    if (pcl::io::loadPCDFile(src_pointcloud_kpts_path, *pointcloud_src_kpts) < 0) {
        std::cout << RED << "Error: Unable to load source point cloud keypoints file: " << src_pointcloud_kpts_path
                  << RESET << std::endl;
        return false;
    }
    if (pcl::io::loadPCDFile(tgt_pointcloud_kpts_path, *pointcloud_tgt_kpts) < 0) {
        std::cout << RED << "Error: Unable to load target point cloud keypoints file: " << tgt_pointcloud_kpts_path
                  << RESET << std::endl;
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
    find_index_for_correspondences(pointcloud_src_kpts, pointcloud_tgt_kpts, correspondences);
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
    float RE_eva_thresh, TE_eva_thresh, inlier_eva_thresh;
    if (dataset_name == "KITTI") {
        RE_eva_thresh = 5;
        TE_eva_thresh = 180;
        inlier_eva_thresh = 1.8;
    } else if (dataset_name == "3dmatch" || dataset_name == "3dlomatch") {
        RE_eva_thresh = 15;
        TE_eva_thresh = 30;
        inlier_eva_thresh = 0.1;
    } else if (dataset_name == "U3M") {
        inlier_eva_thresh = 5 * resolution;
    }

    // NOTE: we do not consider the outer loop of the registration process, which is used to
    // repeat the registration process for multiple iterations.
    timing(0); // Start timing
    Eigen::Matrix graph_eigen = graph_construction(correspondences, resolution, second_order_graph_flag, dataset_name,
                                                   descriptor, inlier_eva_thresh);
    timing(1); // End timing
    std::cout << "Graph has been constructed, time elapsed: " << std::endl; // TODO: complete the timing log logics

    // Check whether the graph is all 0
    if (graph_eigen.norm() == 0) {
        cout << "Graph is disconnected. You may need to check the compatibility threshold!" << endl;
        return false;
    }


    timing(0);
    // Prepaer for filtering

    // Calculate degree of the vertexes

    // std::vector<int> graph_degree(total_correspondences_num, 0);
    std::vector<Vote_exp> points_degree(total_correspondences_num);
    // points_degree.reserve(total_correspondences_num); // used for single thread
#pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, points_degree, gt_correspondences)
    for (int i = 0; i < total_correspondences_num; ++i) {
        // Construct variables
        int current_index = 0;
        int degree = 0;
        float score = 0;
        std::vector<int> correspondences_index;
        correspondences_index.reserve(total_correspondences_num);
        int true_num = 0;
        for (int j = 0; j < total_correspondences_num; ++j) {
            if (i != j && graph_eigen(i, j)) {
                degree++;
                correspondences_index.push_back(j);
                if (gt_correspondences[j]) {
                    true_num++;
                }
            }
        }
        points_degree[i].current_index = current_index;
        points_degree[i].degree = degree;
        points_degree[i].correspondences_index = correspondences_index;
        points_degree[i].true_num = true_num;
    }

    // // igraph version, should be carefully tested. I did not try igraph libs
    // igraph_t graph_igraph;
    // igraph_matrix_t graph_igraph_matrix;
    // igraph_matrix_view(&graph_igraph_matrix, graph_eigen.data(), total_correspondences_num, total_correspondences_num);
    //
    // igraph_adjacency(&graph_igraph, &graph_igraph_matrix, IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS);
    // std::cout << "\nigraph graph object created successfully." << std::endl;
    // igraph_vector_int_t degrees_igraph;
    //
    // igraph_error_t error_code_igraph = igraph_degree(&graph_igraph, &degrees_igraph, igraph_vss_all(), IGRAPH_ALL,
    //                                                  IGRAPH_NO_LOOPS);
    // if (error_code_igraph != IGRAPH_SUCCESS) {
    //     std::cerr << "Error calculating degree: " << igraph_strerror(error_code_igraph) << std::endl;
    //     return false;
    // }


    timing(1);

    // Calculate the vertex clustering factor to determine the density of the graph.
    // Delete some of the vertexes and edges if the graph is dense

    timing(0);
    std::vector<Vote> cluster_factor;
    float sum_numerator = 0;
    float sum_denominator = 0;
    for (int i = 0; i < total_correspondences_num; ++i) {
        double weight_sum_i = 0.0;
        int neighbor_size = points_degree[i].degree; // degree = correspondences_index.size()
        if (neighbor_size > 1) {
            int current_index = 0;
            int score = 0;
#pragma omp parallel
            {
#pragma omp for
                for (int j = 0; j < neighbor_size; ++j) {
                    int neighbor_index_1 = points_degree[i].correspondences_index[j];
                    for (int k = j + 1; k < neighbor_size; ++k) {
                        int neighbor_index_2 = points_degree[i].correspondences_index[k];
                        if (graph_eigen(neighbor_index_1, neighbor_index_2)) {
#pragma omp critical
                            weight_sum_i += graph_eigen(i, neighbor_index_1);
                            weight_sum_i += graph_eigen(i, neighbor_index_2);
                            // weight_sum_i += graph_eigen(neighbor_index_1, neighbor_index_2);
                            // weight_sum_i += pow(
                            //     graph_eigen(i, neighbor_index_1) * graph_eigen(i, neighbor_index_2) * graph_eigen(
                            //         neighbor_index_1, neighbor_index_2), 1.0 / 3);
                        }
                    }
                }
            }
            float vertex_numerator = weight_sum_i;
            float vertex_denominator = static_cast<float>(neighbor_size * (neighbor_size - 1)) / 2.0f;
            sum_numerator += vertex_numerator;
            sum_denominator += vertex_denominator;
            float vertex_factor = vertex_numerator / vertex_denominator;
            cluster_factor.emplace_back(i, vertex_factor, false);
        } else {
            cluster_factor.emplace_back(i, 0.0f, false); // If the vertex has no neighbors, set the factor to 0
        }
    }

    timing(1);
    std::cout << "cluster factors calculation completed. Time elapsed: " << std::endl; // Need to complete the timing logics

    // average factor for clusters
    float average_factor_cluster = 0;
    for (auto & i : cluster_factor) {
        average_factor_cluster += i.score;
    }
    average_factor_cluster /= static_cast<float>(cluster_factor.size());

    // average factor for vertexes
    float average_factor_vertex = sum_numerator / sum_denominator;

    std::vector<Vote>cluster_factor_sorted;
    cluster_factor_sorted.assign(cluster_factor.begin(), cluster_factor.end()); // copy of cluster_factor
    sort(cluster_factor_sorted.begin(), cluster_factor_sorted.end(), compare_vote_score);

    // Prepaer data for OTSU thresholding
    std::vector<float> cluster_factor_scores;
    cluster_factor_scores.resize(cluster_factor.size());
    for (int i = 0; i < cluster_factor.size(); ++i) {
        cluster_factor_scores[i] = cluster_factor[i].score;
    }

    float otsu = 0;
    if (cluster_factor_sorted[0].score != 0) {
        otsu = otsu_thresh(cluster_factor_scores);
    }
    float cluster_threshold = min (otsu, min(average_factor_cluster, average_factor_vertex));


    cout << cluster_threshold << "->min(" << average_factor_cluster << " " << average_factor_vertex << " " << otsu << ")" << endl;
    cout << " inliers: " << inlier_num << "\ttotal num: " << total_correspondences_num << "\tinlier ratio: " << inlier_ratio*100 << "%" << endl;
    //OTSU计算权重的阈值
    float weight_thresh = 0; //OTSU_thresh(sorted); // no overlap, thus weigth thresh is 0.

    // assign score member variable. Note that we need to align the indexes
    if (instance_equal_flag)
    {
        for (size_t i = 0; i < total_correspondences_num; i++)
        {
            correspondences[i].score = cluster_factor[i].score;
        }
    }

    // Maximal clique searching

    // Create igraph graph from the Eigen matrix
    igraph_t graph_igraph;
    igraph_matrix_t graph_igraph_matrix;
    // igraph_matrix_init(&graph_igraph_matrix, graph_eigen.rows(), graph_eigen.cols());

    // Filtering, reduce the graph size
    // Note that the original mac++ use this to filter the graph on kitti dataset. We ignore that for now

    // for (int i = 0; i < graph_eigen.rows(); ++i) {
    //     for (int j = 0; j < graph_eigen.cols(); ++j) {
    //         if (graph_eigen(i, j)) {
    //             igraph_matrix_set(&graph_igraph_matrix, i, j, graph_eigen(i, j));
    //         } else {
    //             igraph_matrix_set(&graph_igraph_matrix, i, j, 0);
    //         }
    //     }
    // }

    // TODO: We can use igraph_adjlist to construct the igraph graph. This may reduce the graph construction time.
    // TODO: igraph can also use BLAS to speed up processing.
    // Need to be checked!!! I do not know how to use igraph!!
    igraph_matrix_view(&graph_igraph_matrix, graph_eigen.data(), total_correspondences_num, total_correspondences_num);
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&graph_igraph, &graph_igraph_matrix, IGRAPH_ADJ_UNDIRECTED, &weight, IGRAPH_NO_LOOPS);

    // Find the maximal cliques in the graph
    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);
    timing(0);

    int min_clique_size = 3; // Minimum size of the clique to be considered, 3 is the minimum number to creat a triangle
    int max_clique_size = 0; // Maximum size of the clique, 0 is no limit.
    bool recalculate_flag = true; // Flag to indicate whether to recalculate the cliques
    int iter_num = 1;

    while (recalculate_flag) {
        igraph_maximal_cliques(&graph_igraph, &cliques, min_clique_size, max_clique_size);
        clique_num = static_cast<int>(igraph_vector_int_list_size(&cliques));
        // For now, we do not know in what case this will happen
        if (clique_num > 10000000 && iter_num <= 5) {
            max_clique_size = 15;
            min_clique_size += iter_num;
            iter_num++;
            igraph_vector_int_list_destroy(&cliques);
            igraph_vector_int_list_init(&cliques, 0);
            std::cout << "clique number " << clique_num << " is too large, recalculate with min_clique_size = "
                    << min_clique_size << " and max_clique_size = " << max_clique_size << std::endl;
        } else {
            recalculate_flag = false;
        }
    }

    timing(1);

    if (clique_num == 0) {
        std::cout << YELLOW << "Error: No cliques found in the graph." << RESET << std::endl;
        return false;
    }
    std::cout << "Number of cliques found: " << clique_num << ". Time for maximal clique search: "<< std::endl; // timing logic should be completed

    // Data cleaning
    igraph_destroy(&graph_igraph);
    igraph_matrix_destroy(&graph_igraph_matrix);


    // Correspondence seed generation and clique pre filtering
    std::vector<int> sampled_ind; // sampled correspondences index
    std::vector<int> remain; // remaining correspondences index after filtering

    clique_sampling(graph_eigen, &cliques, sampled_ind, remain);

    std::vector<Corre_3DMatch> sampled_corr; // sampled correspondences
    PointCloudPtr sampled_corr_src(new pcl::PointCloud<pcl::PointXYZ>); // sampled source point cloud
    PointCloudPtr sampled_corr_tgt(new pcl::PointCloud<pcl::PointXYZ>); // sampled target point cloud
    int inlier_num_af_clique_sampling = 0;
    for(auto &ind : sampled_ind){
        sampled_corr.push_back(correspondences[ind]);
        sampled_corr_src->push_back(correspondences[ind].src);
        sampled_corr_tgt->push_back(correspondences[ind].tgt);
        if(gt_correspondences[ind]){
            inlier_num_af_clique_sampling++;
        }
    }

    // Save log
    string sampled_corr_txt = output_path + "/sampled_corr.txt";
    ofstream outFile1;
    outFile1.open(sampled_corr_txt.c_str(), ios::out);
    for(int i = 0;i <(int)sampled_corr.size(); i++){
        outFile1 << sampled_corr[i].src_index << " " << sampled_corr[i].tgt_index <<endl;
    }
    outFile1.close();

    string sampled_corr_label = output_path + "/sampled_corr_label.txt";
    ofstream outFile2;
    outFile2.open(sampled_corr_label.c_str(), ios::out);
    for(auto &ind : sampled_ind){
        if(gt_correspondences[ind]){
            outFile2 << "1" << endl;
        }
        else{
            outFile2 << "0" << endl;
        }
    }
    outFile2.close();

    // The inlier ratio should be higher than the original inlier ratio
    std::cout << "Inlier ratio after clique sampling: "
              << static_cast<float>(inlier_num_af_clique_sampling) / static_cast<float>(sampled_ind.size()) * 100
              << "%" << std::endl;

    std::cout << "Number of sampled correspondences: " << sampled_ind.size() << std::endl;
    std::cout << "Number of remaining correspondences: " << remain.size() << std::endl;
    std::cout << "Number of cliques: " << clique_num << std::endl;
    std::cout << "Time for clique sampling: " << std::endl; // timing logic should be completed
    timing(1);

    // construct the correspondence points index list for sampled correspondences
    PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < total_correspondences_num; i++)
    {
        src_corr_pts->push_back(correspondences[i].src);
        des_corr_pts->push_back(correspondences[i].tgt);
    }


    // Registration

    Eigen::Matrix4f best_est1, best_est2; // TODO: change the name

    bool found_flag = false; // Flag to indicate whether a valid registration was found
    float best_score = 0.0f; // Best score for the registration

    timing(0);
    int total_estimate_num = remain.size(); // Total number of estimated correspondences

    std::vector<Eigen::Matrix3f> Rs;
    std::vector<Eigen::Vector3f> Ts;
    std::vector<float> scores;
    std::vector<std::vector<int>>group_corr_ind;
    int max_size = 0;
    int min_size = 666;
    int selected_size = 0;

    std::vector<Vote>est_vector;
    std::vector<pair<int, std::vector<int>>> tgt_src;
    make_tgt_src_pair(correspondences, tgt_src); //将初始匹配形成点到点集的对应

    // Get each clique and estimate the transformation matrix by the points in the clique
#pragma omp parallel for
    for (int i = 0; i < total_estimate_num; ++i) {
        std::vector<Corre_3DMatch>group, group1;
        std::vector<int> selected_index;
        igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int group_size = igraph_vector_int_size(v); // size of the current clique
        for (int j = 0; j < group_size; j++) {
            int ind = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            group.push_back(correspondences[ind]);
            selected_index.push_back(ind);
        }
        sort(selected_index.begin(), selected_index.end()); // sort before get intersection

        Eigen::Matrix4f est_trans_mat;
        PointCloudPtr src_pts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr tgt_pts(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<float> weights;
        for (auto &k : group) {
            if (k.score >= weight_thresh) { // 0 by default
                group1.push_back(k);
                src_pts->push_back(k.src);
                tgt_pts->push_back(k.tgt);
                weights.push_back(k.score); // score is calculated by cluster factor
            }
        }
        if (weights.size() < 3) { //
            continue;
        }
        Eigen::VectorXf weight_vec = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(weights.data(), weights.size());
        weights.clear();
        weights.shrink_to_fit();
        weight_vec /= weight_vec.maxCoeff();
        // This can be done before weight assignments
        if (instance_equal_flag) {
            weight_vec.setOnes();
        }
        weight_svd(src_pts, tgt_pts, weight_vec, weight_thresh, est_trans_mat); // weight_thresh is 0 in original MAC++
        // When weight thresh is 0, the two group is identical
        group.assign(group1.begin(), group1.end()); // assign the filtered group to the original group
        group1.clear();

        // pre evaluate the transformation matrix generated by each clique (group, in MAC++)
        float score = 0.0f, score_local = 0.0f;
        // These evaluation is important
        // Global
        score = OAMAE(pointcloud_src_kpts, pointcloud_tgt_kpts, est_trans_mat, tgt_src, inlier_eva_thresh);
        // Local
        score_local = evaluation_trans(group, src_pts, tgt_pts, est_trans_mat, inlier_eva_thresh, metric, resolution);

        src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        tgt_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        group.clear();
        group.shrink_to_fit();

        //GT未知 <- commented by the MAC++ author
        if (score > 0)
        {
#pragma omp critical
            {
                Eigen::Matrix4f trans_f = est_trans_mat;
                Eigen::Matrix3f R = trans_f.topLeftCorner(3, 3);
                Eigen::Vector3f T = trans_f.block(0, 3, 3, 1);
                Rs.push_back(R);
                Ts.push_back(T);
                scores.push_back(score_local); // local score add to scores
                group_corr_ind.push_back(selected_index);
                selected_size = selected_index.size();
                Vote t;
                t.current_index = i;
                t.score = score;
                float re, te;
                // This part use the gt mat, only for method evaluation
                t.flag = evaluation_est(est_trans_mat, gt_mat, 15, 30, re, te);
                if(t.flag){
                    success_num ++;
                }
                //
                est_vector.push_back(t);
                if (best_score < score)
                {
                    best_score = score; // score is the global evaluation score
                    best_est1 = est_trans_mat; // best_est1 is the one generated from each clique weighted svd
                    //selected = Group;
                    //corre_index = selected_index;
                }
            }
        }
        selected_index.clear();
        selected_index.shrink_to_fit();
    }

    //释放内存空间
    // Clique searching is done, we can destroy the cliques
    igraph_vector_int_list_destroy(&cliques);


    bool clique_reduce = false;
    vector<int>indices(est_vector.size());
    for (int i = 0; i < (int )est_vector.size(); ++i) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(), [&est_vector](int a, int b){return est_vector[a].score > est_vector[b].score;});
    vector<Vote>est_vector1(est_vector.size()); // sorted est_vector
    for(int i = 0; i < (int )est_vector.size(); i++){
        est_vector1[i] = est_vector[indices[i]];
    }
    est_vector.assign(est_vector1.begin(), est_vector1.end()); // est_vector is sorted
    est_vector1.clear();


    // TODO: Check all groud true evaluations, and unify the naming. Also pay attension to the method evaluation. Unify the comment expression.
    // GT Evaluation first then filter
    int max_num = min(min(total_correspondences_num,total_estimate_num), max_estimate_num);
    success_num = 0; // note the last sucess_num is not used, check that whether can be used
    vector<int>remained_est_ind;
    vector<Eigen::Matrix3f> Rs_new;
    vector<Eigen::Vector3f> Ts_new;
    if((int )est_vector.size() > max_num) { //选出排名靠前的假设
        cout << "too many cliques" << endl;
    }
    for(int i = 0; i < min(max_num, (int )est_vector.size()); i++){
        remained_est_ind.push_back(indices[i]);
        Rs_new.push_back(Rs[indices[i]]);
        Ts_new.push_back(Ts[indices[i]]);
        success_num += est_vector[i].flag ? 1 : 0;
    }
    Rs.clear();
    Ts.clear();
    Rs.assign(Rs_new.begin(), Rs_new.end());
    Ts.assign(Ts_new.begin(), Ts_new.end());
    Rs_new.clear();
    Ts_new.clear();

    if(success_num > 0){
        if(!no_logs){
            string est_info = output_path + "/est_info.txt";
            ofstream est_info_file(est_info, ios::trunc);
            est_info_file.setf(ios::fixed, ios::floatfield);
            for(auto &i : est_vector){
                est_info_file << setprecision(10) << i.score << " " << i.flag << endl;
            }
            est_info_file.close();
        }
    }
    else{
        cout<< "NO CORRECT ESTIMATION!!!" << endl;
    }

    //cout << success_num << " : " << max_num << " : " << total_estimate << " : " << clique_num << endl;
    //cout << min_size << " : " << max_size << " : " << selected_size << endl;
    correct_est_num = success_num;

    // Clustering
    // Set parameters according to datasets
    float angle_thresh;
    float dis_thresh;
    if(dataset_name == "3dmatch" || dataset_name == "3dlomatch"){
        angle_thresh = 5.0 * M_PI / 180.0;
        dis_thresh = inlier_eva_thresh;
    }
    else if(dataset_name == "U3M"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = 5*resolution;
    }
    else if(dataset_name == "KITTI"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = inlier_eva_thresh;
    }
    else{
        cout << "not implement" << endl;
        exit(-1);
    }


    //




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
