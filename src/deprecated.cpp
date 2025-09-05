//
// Created by Jeffery_Xeom on 2025/8/24.
// Project: MAC_SHARP
// File: deprecated.cpp
//





///////////////////////////////////////////////////////////////
// These function are used for separative matrix formed graph construction
// After performance analysis, we found that the joint graph construction seems more efficient.
// These functions are deprecated, but kept for reference.

// std::vector<float> pdist_blas(const float *points_matrix, const int n, const int dims) {
//     // 计算输出向量的大小
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<float> pdist_vector(result_size);
//
//     // --- 步骤 1: 计算格拉姆矩阵 G = points * points^T ---
//     std::vector<float> gram_matrix(n * n);
//     cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
//                 n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);
//
//     // --- 步骤 2: 提取对角线元素 (范数的平方) ---
//     std::vector<float> norms_sq(n);
//     for (int i = 0; i < n; ++i) {
//         norms_sq[i] = gram_matrix[i * n + i];
//     }
//
//     // --- 步骤 3: 装配最终的距离向量 (关键修改) ---
//     long long k = 0; // pdist_vector 的索引
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             // 从上三角部分读取 G[i,j]
//             float dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
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
//  * @return              一个 std::vector<float>，包含 n*(n-1)/2 个成对距离.
//  */
// std::vector<float> pdist_naive(const float *points_matrix, const int n, const int dims) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<float> pdist_vector(result_size);
//
//     long long k = 0; // pdist_vector 的索引
//     // 遍历所有唯一的点对 (i, j) where j > i
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             float dist_sq = 0.0;
//             // 计算这对点之间距离的平方
//             // (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
//             for (int d = 0; d < dims; ++d) {
//                 float diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
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
//  * @brief 使用 OpenMP 并行化循环计算 MATLAB 的 pdist 功能，返回一个 std::vector<float> 结果.
//  * @param points_matrix 输入，指向 nxdims 点矩阵数据的指针.
//  * @param n             点的数量.
//  * @param dims          点的维度.
//  * @return              一个 std::vector<float>，包含 n*(n-1)/2 个成对距离.
//  */
// std::vector<float> pdist_naive_parallel(const float *points_matrix, const int n, const int dims) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return {};
//     }
//     std::vector<float> pdist_vector(result_size);
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
//             float dist_sq = 0.0;
//             const int temp_j = j * dims;
//             for (int d = 0; d < dims; ++d) {
//                 const float diff = points_matrix[temp_i + d] - points_matrix[temp_j + d];
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
// void pdist_blas(const float *points_matrix, const int n, const int dims, float *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return; // 如果没有要计算的距离，则直接返回
//     }
//
//     // 内部的临时缓冲区仍然可以使用 std::vector，方便管理
//     std::vector<float> gram_matrix(n * n);
//     cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
//                 n, dims, 1.0, points_matrix, dims, 0.0, gram_matrix.data(), n);
//
//     std::vector<float> norms_sq(n);
//     for (int i = 0; i < n; ++i) {
//         norms_sq[i] = gram_matrix[i * n + i];
//     }
//
//     long long k = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             float dist_sq = norms_sq[i] + norms_sq[j] - 2 * gram_matrix[i * n + j];
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
// void pdist_naive(const float *points_matrix, const int n, const int dims, float *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return;
//     }
//
//     long long k = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             float dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 float diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
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
// void pdist_naive_parallel(const float *points_matrix, const int n, const int dims, float *result_buffer) {
//     const long long result_size = (long long) n * (n - 1) / 2;
//     if (result_size <= 0) {
//         return;
//     }
//
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         long long offset = (long long) i * n - (long long) i * (i + 1) / 2;
//         for (int j = i + 1; j < n; ++j) {
//             float dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 float diff = points_matrix[i * dims + d] - points_matrix[j * dims + d];
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
// void gaussian_kernel_omp(const float *dist_sq_matrix, float *score_matrix, long long size, float alpha_dis) {
//     // 预先计算出不变的系数部分，避免在循环中重复计算
//     // 公式是 -1 / (2 * alpha * alpha)
//     const float gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
//
//     // 使用 OpenMP 将这个巨大的循环并行化
// #pragma omp parallel for schedule(static) default(none) shared(gamma, size, score_matrix, dist_sq_matrix)
//     for (long long i = 0; i < size; ++i) {
//         score_matrix[i] = std::exp(dist_sq_matrix[i] * gamma);
//     }
// }
//
// // 这是一个计算完整 n*n 距离平方矩阵的函数 (之前pdist是向量，这里需要方阵)
// void pdist_sq_matrix_omp(const float *points, int n, int dims, float *dist_sq_matrix) {
// #pragma omp parallel for
//     for (int i = 0; i < n; ++i) {
//         for (int j = i; j < n; ++j) {
//             float dist_sq = 0.0;
//             for (int d = 0; d < dims; ++d) {
//                 float diff = points[i * dims + d] - points[j * dims + d];
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
// void calculate_scores_omp(const std::vector<float> &dist_vector, std::vector<float> score_mat, long long size,
//                           const ScoreFormula formula, const float alpha_dis, const float inlier_thresh) {
//     const int totalCorresNum_2 = 2 * totalCorresNum - 2;
//     switch (formula) {
//         case ScoreFormula::GAUSSIAN_KERNEL: {
//             // 预计算高斯核的 gamma 系数
//             const float gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
// #pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_mat, totalCorresNum, totalCorresNum_2)
//             for (int i = 0; i < totalCorresNum; ++i) {
//                 const long long temp_i = i * totalCorresNum;
//                 const long long temp_i_2 = totalCorresNum * i - i * (i + 1) / 2; // 计算当前行的偏移量
//                 for (int j = i + 1; j < totalCorresNum; ++j) {
//                     const long long temp_k = temp_i_2 + j - i - 1;
//                     const float dist_sq = dist_vector.at(temp_k) * dist_vector[temp_k];
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat[j * totalCorresNum + i] = score_mat[temp_i + j]; // 确保矩阵对称
//                 }
//             }
//             break;
//         }
//         case ScoreFormula::QUADRATIC_FALLOFF: {
//             // 预计算二次衰减的系数，用乘法代替除法以提高效率
//             const float inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
// #pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_mat, totalCorresNum, totalCorresNum_2)
//             for (int i = 0; i < totalCorresNum; ++i) {
//                 const long long temp_i = i * totalCorresNum;
//                 for (int j = i; j < totalCorresNum; ++j) {
//                     const long long temp_k = (totalCorresNum_2 - i) * i / 2 + j;
//                     const float dist_sq = dist_vector[temp_k] * dist_vector[temp_k];
//                     // 使用 std::max 确保分数不会小于0，这通常是期望的行为
//                     score_mat[temp_i + j] = std::max(0.0, 1.0 - dist_sq * inv_thresh_sq);
//                     score_mat[j * totalCorresNum + i] = score_mat[temp_i + j]; // 确保矩阵对称
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
// void calculate_scores_omp(const float *dist_vector, float *score_vector, long long size,
//                           ScoreFormula formula, float alpha_dis, float inlier_thresh) {
//     const long long dis_size = totalCorresNum * (totalCorresNum - 1) / 2;
//     switch (formula) {
//         case ScoreFormula::GAUSSIAN_KERNEL: {
//             // 预计算高斯核的 gamma 系数
//             const float gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
// #pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_vector, dis_size)
//             for (long long i = 0; i < dis_size; ++i) {
//                 float dist_sq = dist_vector[i] * dist_vector[i];
//                 score_vector[i] = std::exp(dist_sq * gamma);
//             }
//             break;
//         }
//
//         case ScoreFormula::QUADRATIC_FALLOFF: {
//             // 预计算二次衰减的系数，用乘法代替除法以提高效率
//             const float inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
// #pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_vector, dis_size)
//             for (long long i = 0; i < dis_size; ++i) {
//                 float dist_sq = dist_vector[i] * dist_vector[i];
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
// void unpack_upper_triangle(const std::vector<float> &packed_upper_triangle,
//                            Eigen::MatrixXf &full_matrix,
//                            float diagonal_value = 0.0) {
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
//             float value = packed_upper_triangle[k];
//             full_matrix(i, j) = value; // 填充上三角
//             full_matrix(j, i) = value; // 利用对称性，同时填充下三角
//         }
//     }
// }

// Eigen::MatrixXf graph_construction(vector<Corre_3DMatch> &correspondences, float resolution,
//                                    bool second_order_graph_flag, const std::string &dataset_name,
//                                    const std::string &descriptor, float inlier_thresh) {
//     // Construct a graph from the correspondences. The graph is represented as an adjacency matrix
//     // TODO: Is there a more efficient way to construct or represent the graph?
//     // totalCorresNum is the size of the correspondences, which is also a global variable
//     Eigen::MatrixXf graph = Eigen::MatrixXf::Zero(totalCorresNum, totalCorresNum);
//     const float alpha_dis = 10 * resolution;
//     const long long distance_size = static_cast<long long>(totalCorresNum) * (totalCorresNum - 1) / 2;
//     // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
//     std::vector<float> src_mat(totalCorresNum * 3, 0.0), tgt_mat(totalCorresNum * 3, 0.0),
//             score_mat(totalCorresNum * totalCorresNum, 0.0), score_vec(distance_size);
//
//     // Construct the two points vectors (dimension n x 3)
//     std::cout << "Constructing the two points vectors..." << std::endl;
//     timing(0);
// #pragma omp parallel for schedule(static) default(none) shared(correspondences, src_mat, tgt_mat, totalCorresNum)
//     for (int i = 0; i < totalCorresNum; ++i) {
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
//     // std::vector<float> da(pdist_size);
//     // std::vector<float> db(pdist_size);
//     std::vector<float> d(distance_size, 0.0); // Initialize the distance vector with zeros
//
//     std::cout << "Calculating the distance vector..." << std::endl;
//     timing(0);
//     const std::vector<float> da = pdist_naive_parallel(src_mat.data(), totalCorresNum, 3);
//     const std::vector<float> db = pdist_naive_parallel(tgt_mat.data(), totalCorresNum, 3);
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
//     // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map_for_copy(
//     //     score_mat.data(), totalCorresNum, totalCorresNum);
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