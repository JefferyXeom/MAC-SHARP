#include <chrono>
#include <vector>
#include <iostream>

// For pcl
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>


#include "MAC_utils.hpp"

#include "MAC_SHARP.hpp"


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

// We do not exactly know why there is a need to find the index of correspondences, but it is used in the original code.
// NOTE: the keypoints are not in the original point cloud, therefore nearest search is required.
// Find the nearest point in the source and target key point clouds for each correspondence, and assign the indices to the correspondences.
// Another note is that, the original MAC++ does not use the corr_ind file for correspondences indexing. Therefore, this function is necessary.
void find_index_for_correspondences(PointCloudPtr &src, PointCloudPtr &tgt, std::vector<Correspondence_Struct> &correspondences) {
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



///////////////////////////////////////////////////////////////
// These function are used for seperative matrix formed graph construction
// After performance analysis, we found that the joint graph construction seems more efficient.
// These functions are deprecated, but kept for reference.

// std::vector<double> pdist_blas(const double *points_matrix, const int n, const int dims) {
//     // 计算输出向量的大小
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<double> pdist_vector(result_size);
//
//     // --- 步骤 1: 计算格拉姆矩阵 G = points * points^T ---
//     std::vector<double> gram_matrix(n * n);
//     cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
//                 n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);
//
//     // --- 步骤 2: 提取对角线元素 (范数的平方) ---
//     std::vector<double> norms_sq(n);
//     for (int i = 0; i < n; ++i) {
//         norms_sq[i] = gram_matrix[i * n + i];
//     }
//
//     // --- 步骤 3: 装配最终的距离向量 (关键修改) ---
//     long long k = 0; // pdist_vector 的索引
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             // 从上三角部分读取 G[i,j]
//             double dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
//             // 避免因浮点数误差导致开方负数
//             pdist_vector[k] = std::sqrt(std::max(0.0, dist_sq));
//             k++;
//         }
//     }
//     return pdist_vector;
// }
//
// /**
//  * @brief 使用纯 C++ 循环计算 MATLAB 的 pdist 功能，不依赖任何外部库.
//  * @param points_matrix 指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @return              一个 std::vector<double>，包含 n*(n-1)/2 个成对距离.
//  */
// std::vector<double> pdist_naive(const double *points_matrix, const int n, const int dims) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<double> pdist_vector(result_size);
//
//     long long k = 0; // pdist_vector 的索引
//     // 遍历所有唯一的点对 (i, j) where j > i
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             double dist_sq = 0.0;
//             // 计算这对点之间距离的平方
//             // (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
//             for (int d = 0; d < dims; ++d) {
//                 double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
//                 dist_sq += diff * diff;
//             }
//
//             pdist_vector[k] = std::sqrt(dist_sq);
//             k++;
//         }
//     }
//     return pdist_vector;
// }
//
// // Modified variable definition in for loop for acceleration
// /**
//  * @brief 使用 OpenMP 并行化循环计算 MATLAB 的 pdist 功能，返回一个 std::vector<double> 结果.
//  * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @return              一个 std::vector<double>，包含 n*(n-1)/2 个成对距离.
//  */
// std::vector<double> pdist_naive_parallel(const double *points_matrix, const int n, const int dims) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<double> pdist_vector(result_size);
//
//     // 将 i 循环并行化。每个线程处理不同的 i 值。
//     // #pragma omp parallel for collapse(2) schedule(static)
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         // 直接计算出当前行(i)的配对在结果向量中的起始偏移量
//         // 这是前 i 行所有配对的总数，是一个等差数列求和
//         const long long offset = (long long) i * n - (long long) i * (i + 1) / 2;
//         const int temp_i = i * dims;
//         for (int j = i + 1; j < n; ++j) {
//             double dist_sq = 0.0;
//             const int temp_j = j * dims;
//             for (int d = 0; d < dims; ++d) {
//                 const double diff = points_matrix[temp_i + d] - points_matrix[temp_j + d];
//                 dist_sq += diff * diff;
//             }
//
//             // 根据偏移量和 j 的位置计算出确切的索引 k
//             // 不再需要共享的 k++
//             const long long k = offset + (j - (i + 1));
//
//             pdist_vector[k] = std::sqrt(dist_sq);
//         }
//     }
//     return pdist_vector;
// }
//
// /**
//  * @brief 使用 OpenBLAS 高效实现 MATLAB 的 pdist 功能，结果写入预分配的指针.
//  * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
//  */
// void pdist_blas(const double *points_matrix, const int n, const int dims, double *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return; // 如果没有要计算的距离，则直接返回
//     }
//
//     // 内部的临时缓冲区仍然可以使用 std::vector，方便管理
//     std::vector<double> gram_matrix(n * n);
//     cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
//                 n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);
//
//     std::vector<double> norms_sq(n);
//     for (int i = 0; i < n; ++i) {
//         norms_sq[i] = gram_matrix[i * n + i];
//     }
//
//     long long k = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             double dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
//             result_buffer[k] = std::sqrt(std::max(0.0, dist_sq));
//             k++;
//         }
//     }
// }
//
// /**
//  * @brief 使用纯 C++ 循环计算 MATLAB 的 pdist 功能，结果写入预分配的指针.
//  * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
//  */
// void pdist_naive(const double *points_matrix, const int n, const int dims, double *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return;
//     }
//
//     long long k = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             double dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
//                 dist_sq += diff * diff;
//             }
//             result_buffer[k] = std::sqrt(dist_sq);
//             k++;
//         }
//     }
// }
//
// /**
//  * @brief 使用 OpenMP 并行化循环计算 pdist 功能，结果写入预分配的指针.
//  * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @param[out] result_buffer 输出，指向大小为 n*(n-1)/2 的预分配内存区域.
//  */
// void pdist_naive_parallel(const double *points_matrix, const int n, const int dims, double *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return;
//     }
//
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         long long offset = (long long) i * n - (long long) i * (i + 1) / 2;
//         for (int j = i + 1; j < n; ++j) {
//             double dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 double diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
//                 dist_sq += diff * diff;
//             }
//             long long k = offset + (j - (i + 1));
//             result_buffer[k] = std::sqrt(dist_sq);
//         }
//     }
// }
//
// /**
//  * @brief 使用高斯核函数将距离平方矩阵转换为相似度得分矩阵 (并行化).
//  * @param dist_sq_matrix 输入，指向 n*n 距离平方矩阵的指针.
//  * @param score_matrix   输出，指向 n*n 得分矩阵的预分配内存.
//  * @param size           矩阵的总元素数量 (n*n).
//  * @param alpha_dis      高斯核函数的带宽参数 alpha.
//  */
// void gaussian_kernel_omp(const double *dist_sq_matrix, double *score_matrix, long long size, double alpha_dis) {
//     // 预先计算出不变的系数部分，避免在循环中重复计算
//     // 公式是 -1 / (2 * alpha * alpha)
//     const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
//
//     // 使用 OpenMP 将这个巨大的循环并行化
// #pragma omp parallel for schedule(static) default(none) shared(gamma, size, score_matrix, dist_sq_matrix)
//     for (long long i = 0; i < size; ++i) {
//         score_matrix[i] = std::exp(dist_sq_matrix[i] * gamma);
//     }
// }
//
// // 这是一个计算完整 n*n 距离平方矩阵的函数 (之前pdist是向量，这里需要方阵)
// void pdist_sq_matrix_omp(const double *points, int n, int dims, double *dist_sq_matrix) {
// #pragma omp parallel for
//     for (int i = 0; i < n; ++i) {
//         for (int j = i; j < n; ++j) {
//             double dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 double diff = points[i * dims + d] - points[j * dims + d];
//                 dist_sq += diff * diff;
//             }
//             dist_sq_matrix[i * n + j] = dist_sq;
//             dist_sq_matrix[j * n + i] = dist_sq; // 距离矩阵是对称的
//         }
//     }
// }
//
//
// /**
//  * @brief 根据指定的公式，对距离向量逐元素计算得分 (OpenMP 并行化).
//  * @param dist_vector    输入，指向距离向量的指针.
//  * @param score_mat      输出，指向分数矩阵的预分配内存.
//  * @param size           向量的长度.
//  * @param formula        要使用的公式类型 (枚举).
//  * @param alpha_dis      高斯核函数的带宽参数 (仅在 GAUSSIAN_KERNEL 时使用).
//  * @param inlier_thresh  二次衰减的阈值参数 (仅在 QUADRATIC_FALLOFF 时使用).
//  */
// // paralle version of calculate_scores
// void calculate_scores_omp(const std::vector<double> &dist_vector, std::vector<double> score_mat, long long size,
//                           const ScoreFormula formula, const double alpha_dis, const double inlier_thresh) {
//     const int total_correspondences_num_2 = 2 * total_correspondences_num - 2;
//     switch (formula) {
//         case ScoreFormula::GAUSSIAN_KERNEL: {
//             // 预计算高斯核的 gamma 系数
//             const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
// #pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_mat, total_correspondences_num, total_correspondences_num_2)
//             for (int i = 0; i < total_correspondences_num; ++i) {
//                 const long long temp_i = i * total_correspondences_num;
//                 const long long temp_i_2 = total_correspondences_num * i - i * (i + 1) / 2; // 计算当前行的偏移量
//                 for (int j = i + 1; j < total_correspondences_num; ++j) {
//                     const long long temp_k = temp_i_2 + j - i - 1;
//                     const double dist_sq = dist_vector.at(temp_k) * dist_vector[temp_k];
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat[j * total_correspondences_num + i] = score_mat[temp_i + j]; // 确保矩阵对称
//                 }
//             }
//             break;
//         }
//         case ScoreFormula::QUADRATIC_FALLOFF: {
//             // 预计算二次衰减的系数，用乘法代替除法以提高效率
//             const double inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
// #pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_mat, total_correspondences_num, total_correspondences_num_2)
//             for (int i = 0; i < total_correspondences_num; ++i) {
//                 const long long temp_i = i * total_correspondences_num;
//                 for (int j = i; j < total_correspondences_num; ++j) {
//                     const long long temp_k = (total_correspondences_num_2 - i) * i / 2 + j;
//                     const double dist_sq = dist_vector[temp_k] * dist_vector[temp_k];
//                     // 使用 std::max 确保分数不会小于0，这通常是期望的行为
//                     score_mat[temp_i + j] = std::max(0.0, 1.0 - dist_sq * inv_thresh_sq);
//                     score_mat[j * total_correspondences_num + i] = score_mat[temp_i + j]; // 确保矩阵对称
//                 }
//             }
//         }
//         break;
//     }
// }
//
// /**
//  * @brief 根据指定的公式，对距离向量逐元素计算得分 (OpenMP 并行化).
//  * @param dist_vector    输入，指向距离向量的指针.
//  * @param score_vector   输出，指向得分向量的预分配内存.
//  * @param size           向量的长度.
//  * @param formula        要使用的公式类型 (枚举).
//  * @param alpha_dis      高斯核函数的带宽参数 (仅在 GAUSSIAN_KERNEL 时使用).
//  * @param inlier_thresh  二次衰减的阈值参数 (仅在 QUADRATIC_FALLOFF 时使用).
//  */
// void calculate_scores_omp(const double *dist_vector, double *score_vector, long long size,
//                           ScoreFormula formula, double alpha_dis, double inlier_thresh) {
//     const long long dis_size = total_correspondences_num * (total_correspondences_num - 1) / 2;
//     switch (formula) {
//         case ScoreFormula::GAUSSIAN_KERNEL: {
//             // 预计算高斯核的 gamma 系数
//             const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
// #pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_vector, dis_size)
//             for (long long i = 0; i < dis_size; ++i) {
//                 double dist_sq = dist_vector[i] * dist_vector[i];
//                 score_vector[i] = std::exp(dist_sq * gamma);
//             }
//             break;
//         }
//
//         case ScoreFormula::QUADRATIC_FALLOFF: {
//             // 预计算二次衰减的系数，用乘法代替除法以提高效率
//             const double inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
// #pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_vector, dis_size)
//             for (long long i = 0; i < dis_size; ++i) {
//                 double dist_sq = dist_vector[i] * dist_vector[i];
//                 // 使用 std::max 确保分数不会小于0，这通常是期望的行为
//                 score_vector[i] = std::max(0.0, 1.0 - dist_sq * inv_thresh_sq);
//             }
//             break;
//         }
//     }
// }
//
//
// /**
//  * @brief 将存储上三角数据的压缩向量解包到一个完整的 Eigen 稠密矩阵中.
//  * @param packed_upper_triangle 输入，只包含上三角元素的一维向量.
//  * @param full_matrix           输出，将被填充的 n x n Eigen 矩阵.
//  * @param diagonal_value        对角线元素应该被设置成什么值 (例如，距离矩阵为0，相似度矩阵为1).
//  */
// void unpack_upper_triangle(const std::vector<double> &packed_upper_triangle,
//                            Eigen::MatrixXd &full_matrix,
//                            double diagonal_value = 0.0) {
//     const int n = full_matrix.rows();
//     if (full_matrix.cols() != n) {
//         std::cerr << "Error: Output matrix must be square." << std::endl;
//         return;
//     }
//
//     const long long expected_size = (long long) n * (n - 1) / 2;
//     if (packed_upper_triangle.size() != expected_size) {
//         std::cout << "Packed vector size: " << packed_upper_triangle.size() << std::endl;
//         std::cout << "Expected size: " << expected_size << std::endl;
//         std::cerr << "Error: Packed vector size does not match matrix dimensions." << std::endl;
//         return;
//     }
//
// #pragma omp parallel for schedule(static) default(none) shared(full_matrix, diagonal_value, packed_upper_triangle, n)
//     for (int i = 0; i < n; ++i) {
//         // 1. 设置对角线元素
//         full_matrix(i, i) = diagonal_value;
//         // 2. 填充上三角和下三角部分
//         for (int j = i + 1; j < n; ++j) {
//             long long k = i * n - i * (i + 1) / 2 + j - i - 1; // 计算压缩向量的索引
//             double value = packed_upper_triangle[k];
//             full_matrix(i, j) = value; // 填充上三角
//             full_matrix(j, i) = value; // 利用对称性，同时填充下三角
//         }
//     }
// }


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

///////////////////////////////////////////////////////////////




inline float get_distance(const pcl::PointXYZ &A, const pcl::PointXYZ &B) {
    float distance = 0;
    const float d_x = A.x - B.x;
    const float d_y = A.y - B.y;
    const float d_z = A.z - B.z;
    distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if (!isfinite(distance)) {
        std::cout << YELLOW << "Warning, infinite distance occurred: " << distance << "\t" << A.x << " " << A.y << " " << A.z << "\t" << B.x << " " << B.y << " " << B.z << std::endl;
    }
    return distance;
}

// igraph need eigen matrix be double type
Eigen::MatrixXd graph_construction(std::vector<Correspondence_Struct> &correspondences, float resolution,
                                   bool second_order_graph_flag) {
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
                Correspondence_Struct c1 = correspondences[i];
                for (int j = 0; j < total_correspondences_num; ++j) {
                    Correspondence_Struct c2 = correspondences[j];
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
                Correspondence_Struct c1 = correspondences[i];
                for (int j = 0; j < total_correspondences_num; ++j) {
                    Correspondence_Struct c2 = correspondences[j];
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
