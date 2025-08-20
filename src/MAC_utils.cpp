#include <chrono>
#include <vector>
#include <iostream>
#include <unordered_set>

// pcl
#include <pcl/cloud_iterator.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/impl/conditional_euclidean_clustering.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>


#include <cblas.h>
#include <boost/lexical_cast.hpp>

#include "MAC_utils.hpp"


#include "config_loader.hpp"


// Timing, temporary function for time recording.
// This function will be replaced by Timer class.
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
        std::cout << BLUE << "Elapsed time: " << elapsed_time.count() << " seconds" << RESET << std::endl;
        time_vec.push_back(elapsed_time.count()); // Store elapsed time in vector
    }
}

void settingThreads(const int desired_threads) {
    // Configure OpenBLAS threads
    const int open_blas_max = openblas_get_num_threads();
    const int omp_max = omp_get_max_threads();
    std::cout << "OpenBLAS default threads: " << open_blas_max << ", OMP default threads: " << omp_max << std::endl;
    if (desired_threads == -1) {
        openblas_set_num_threads(open_blas_max);
        omp_set_num_threads(omp_max);
        std::cout << BLUE << "Use maximum threads for computation. OpenBLAS now set to use " <<
                openblas_get_num_threads() << " threads, OMP now set to use " << omp_get_num_threads() << RESET <<
                std::endl;
    } else {
        if (desired_threads > open_blas_max) {
            openblas_set_num_threads(open_blas_max);
            omp_set_num_threads(omp_max);
            std::cout << YELLOW << "Desired thread number exceeds device capacity: " << desired_threads << " > " <<
                    open_blas_max << RESET << std::endl;
            std::cout << "Set OpenBLAS threads to maximum available: " << open_blas_max << std::endl;
        } else {
            openblas_set_num_threads(desired_threads);
            omp_set_num_threads(desired_threads);
            std::cout << BLUE << "Set OpenBLAS and OMP threads to: " << desired_threads << RESET << std::endl;
        }
    }
}

/**
 * @brief 从不同格式的文件中加载点云
 * * @tparam PointT 点的类型 (例如, pcl::PointXYZ, pcl::PointXYZI)
 * @param file_path 点云文件的路径
 * @param cloud 用于存储加载后点云的 PCL 点云对象
 * @return true 如果加载成功
 * @return false 如果加载失败
 */
template<typename PointT>
bool loadPointCloud(const std::string &file_path, pcl::PointCloud<PointT> &cloud) {
    // 1. 获取文件扩展名并转为小写
    std::string extension;
    size_t dot_pos = file_path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        std::cerr << "Error: No file extension found in " << file_path << std::endl;
        return false;
    }
    extension = file_path.substr(dot_pos);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // 2. 根据扩展名选择加载方法
    if (extension == ".pcd") {
        if (pcl::io::loadPCDFile<PointT>(file_path, cloud) == -1) {
            return false;
        }
    } else if (extension == ".ply") {
        if (pcl::io::loadPLYFile<PointT>(file_path, cloud) == -1) {
            return false;
        }
    } else if (extension == ".bin") {
        // 假设是 KITTI 数据集格式: float x, y, z, intensity
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }
        cloud.clear();
        while (in.good() && !in.eof()) {
            PointT point;
            // 读取4个 float
            in.read(reinterpret_cast<char *>(&point.x), sizeof(float));
            in.read(reinterpret_cast<char *>(&point.y), sizeof(float));
            in.read(reinterpret_cast<char *>(&point.z), sizeof(float));

            // 注意：只有当 PointT 有 intensity 字段时才应读取第四个值
            // 为简单起见，我们先跳过它，或者你可以使用更复杂的模板技巧
            float intensity_val;
            in.read(reinterpret_cast<char *>(&intensity_val), sizeof(float));

            // 如果点类型是 PointXYZI，可以取消下面的注释
            // if constexpr (std::is_same_v<PointT, pcl::PointXYZI>) {
            //     point.intensity = intensity_val;
            // }

            if (in.gcount() == 4 * sizeof(float)) {
                // 确保读取了完整的一个点
                cloud.push_back(point);
            }
        }
    } else if (extension == ".txt") {
        // 假设是 TXT 格式: 每行 x y z ...
        std::ifstream in(file_path);
        if (!in.is_open()) {
            return false;
        }
        cloud.clear();
        std::string line;
        while (std::getline(in, line)) {
            std::stringstream ss(line);
            PointT point;
            // 至少需要读取 x, y, z
            if (!(ss >> point.x >> point.y >> point.z)) {
                continue; // 跳过空行或格式错误的行
            }
            // 你可以在这里继续读取更多字段，例如 rgb, intensity 等
            cloud.push_back(point);
        }
    } else {
        std::cerr << "Error: Unsupported file format '" << extension << "' for file: " << file_path << std::endl;
        return false;
    }

    // 3. 确保点云有效
    if (cloud.empty()) {
        return false;
    }

    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;

    return true;
}

bool loadData(const MACConfig &macConfig, PointCloudPtr &cloudSrc, PointCloudPtr &cloudTgt, PointCloudPtr &cloudSrcKpts,
              PointCloudPtr &cloudTgtKpts, std::vector<CorresStruct> &corresOriginal, std::vector<int> &gtCorres,
              Eigen::Matrix4f &gtMat, int &gtInlierNum, float &cloudResolution) {
    // Log and files configuration
    // std::cout << BLUE << "Output path: " << macConfig.outputPath << RESET << std::endl;

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

    // 定义一个辅助加载函数，用于检查、加载并打印错误信息
    auto load_and_check = [&](const std::string &path, auto &cloud_ptr, const std::string &description) {
        if (!loadPointCloud(path, *cloud_ptr)) {
            std::cerr << RED << "Error: Unable to load " << description << " from: " << path << RESET << std::endl;
            return false;
        }
        std::cout << GREEN << "Successfully loaded " << description << " with " << cloud_ptr->size() << " points." <<
                RESET << std::endl;
        return true;
    };

    // 依次加载所有点云和关键点
    if (!load_and_check(macConfig.cloudSrcPath, cloudSrc, "source point cloud")) return false;
    if (!load_and_check(macConfig.cloudTgtPath, cloudTgt, "target point cloud")) return false;
    if (!load_and_check(macConfig.cloudSrcKptPath, cloudSrcKpts, "source keypoints")) return false;
    if (!load_and_check(macConfig.cloudTgtKptPath, cloudTgtKpts, "target keypoints")) return false;

    // Check correspondences and ground truth data
    // FILE *corresFile = fopen(macConfig.corresPath.c_str(), "r");
    // FILE *corresIndexFile = fopen(macConfig.corresPath.c_str(), "r");
    // FILE *GTLabelFile = fopen(macConfig.gtLabelPath.c_str(), "r");
    // FILE *GTTFFile = fopen(macConfig.gtTfPath.c_str(), "r");

    std::ifstream corresFile(macConfig.corresPath);
    std::ifstream corresIndexFile(macConfig.corresIndexPath);

    if (!corresFile.is_open()) {
        std::cout << " error in loading correspondence data. " << std::endl;
        std::cout << macConfig.corresPath << std::endl;
        return false;
    }


    // Note that in our version of test data, the source and target matched kpts clouds are already corresponded.
    // But for the original MAC paper, the source and target matched kpts clouds are not corresponded.
    // Load correspondences, xyz
    CorresStruct match;
    pcl::PointXYZ srcPoint, tgtPoint; // source point and target point in each match
    while (corresFile >> srcPoint.x >> srcPoint.y >> srcPoint.z >>
           tgtPoint.x >> tgtPoint.y >> tgtPoint.z) {
        match.src = srcPoint;
        match.tgt = tgtPoint;
        match.inlierWeight = 0; // Initialize inlier weight to 0
        corresOriginal.push_back(match);
    }
    totalCorresNum = static_cast<int>(corresOriginal.size());
    // Load correspondence indices
    // If the correspondence index file are not provided, we will find the index for correspondences.
    if (!corresIndexFile.is_open()) {
        findIndexForCorrespondences(cloudSrcKpts, cloudTgtKpts, corresOriginal);
    } else {
        int i = 0;
        while (i != corresOriginal.size() && corresIndexFile >> corresOriginal[i].srcIndex >> corresOriginal[i].tgtIndex) {
            i++;
        }
        if (i > totalCorresNum) {
            std::cout << YELLOW << "Warning: too many correspondences in the index file. This could probably "
                    "be caused by mistakenly input the wrong correspondence index file. "
                    "Ignoring the rest." << RESET << std::endl;
        } else if (i < totalCorresNum) {
            std::cout << YELLOW << "Warning: Not enough correspondences in the index file. This could probably "
                    "be caused by mistakenly input the wrong correspondence index file. Ignoring the rest." <<
                    RESET << std::endl;
        }
    }
    // Calculate cloud resolution
    cloudResolution = (meshResolutionCalculation(cloudSrc) + meshResolutionCalculation(cloudTgt)) / 2;
    std::cout << "Cloud resolution: " << cloudResolution << std::endl;


    // ---------------------------- Evaluation part ----------------------------
    if (std::ifstream gtTfFile(macConfig.gtTfPath); !gtTfFile.is_open()) {
        if (macConfig.flagVerbose) {
            std::cout << YELLOW << "No Ground truth transformation data: " << macConfig.gtTfPath << std::endl;
            std::cout << "System working without evaluation" << RESET << std::endl;
        }
    } else {
        // Load ground truth labels
        std::cout << "We are currently in gt reading" << std::endl;
        gtTfFile >> gtMat(0, 0) >> gtMat(0, 1) >> gtMat(0, 2) >> gtMat(0, 3);
        gtTfFile >> gtMat(1, 0) >> gtMat(1, 1) >> gtMat(1, 2) >> gtMat(1, 3);
        gtTfFile >> gtMat(2, 0) >> gtMat(2, 1) >> gtMat(2, 2) >> gtMat(2, 3);
        gtTfFile >> gtMat(3, 0) >> gtMat(3, 1) >> gtMat(3, 2) >> gtMat(3, 3);
        std::cout << "Ground truth transformation matrix: \n" << gtMat << std::endl;
    }
    if (std::ifstream gtLabelFile(macConfig.gtLabelPath); !gtLabelFile.is_open()) {
        if (macConfig.flagVerbose) {
            std::cout << YELLOW << "No Ground truth correspondence data: " << macConfig.gtLabelPath << std::endl;
            std::cout << "System working without evaluation" << RESET << std::endl;
        }
    } else {
        // if (low_inlier_ratio) {
        //     if )
        //
        // }

        // Orignal MAC++ version
        int value = 0;
        while (gtLabelFile >> value) {
            gtCorres.push_back(value);
            if (value == 1) {
                gtInlierNum++;
            }
        }
        // Our version, list graph
        // while (gtLabelFile >> value) {
        //     gtCorres.push_back(value);
        // }
        // gtInlierNum = static_cast<int>(gtCorres.size());

        if (gtInlierNum == 0) {
            std::cout << YELLOW << "Warning: No inliers found in the ground truth correspondences." << RESET << std::endl;
        }
        const float inlier_ratio = static_cast<float>(gtInlierNum) / static_cast<float>(totalCorresNum);
        std::cout << "Inlier ratio: " << inlier_ratio * 100 << std::endl;
    }
    // -------------------------------------------------------------------------

    return true;
}

float meshResolutionCalculation(const PointCloudPtr &pointcloud) {
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
        mr += std::sqrt(point_dis[1]); // distance from query point to nearest neighbor ([0] is the query itself)
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
void findIndexForCorrespondences(PointCloudPtr &cloudSrcKpts, PointCloudPtr &cloudTgtKpts,
                                 std::vector<CorresStruct> &corres) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeSrcKpts, kdtreeTgtKpts;
    kdtreeSrcKpts.setInputCloud(cloudSrcKpts);
    kdtreeTgtKpts.setInputCloud(cloudTgtKpts);
    std::vector<int> kdtreeSrcIndex(1), kdtreeTgtIndex(1);
    std::vector<float> kdtreeSrcDistance(1), kdtreeTgtDistance(1);
    for (auto &corr: corres) {
        pcl::PointXYZ srcPoint, tgtPoint;
        srcPoint = corr.src;
        tgtPoint = corr.tgt;
        kdtreeSrcKpts.nearestKSearch(srcPoint, 1, kdtreeSrcIndex, kdtreeSrcDistance);
        kdtreeTgtKpts.nearestKSearch(tgtPoint, 1, kdtreeTgtIndex, kdtreeTgtDistance);
        corr.srcIndex = kdtreeSrcIndex[0];
        corr.tgtIndex = kdtreeTgtIndex[0];
    }
}


///////////////////////////////////////////////////////////////
// These function are used for separative matrix formed graph construction
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
//     const int totalCorresNum_2 = 2 * totalCorresNum - 2;
//     switch (formula) {
//         case ScoreFormula::GAUSSIAN_KERNEL: {
//             // 预计算高斯核的 gamma 系数
//             const double gamma = -1.0 / (2.0 * alpha_dis * alpha_dis);
// #pragma omp parallel for schedule(static) default(none) shared(gamma, dist_vector, score_mat, totalCorresNum, totalCorresNum_2)
//             for (int i = 0; i < totalCorresNum; ++i) {
//                 const long long temp_i = i * totalCorresNum;
//                 const long long temp_i_2 = totalCorresNum * i - i * (i + 1) / 2; // 计算当前行的偏移量
//                 for (int j = i + 1; j < totalCorresNum; ++j) {
//                     const long long temp_k = temp_i_2 + j - i - 1;
//                     const double dist_sq = dist_vector.at(temp_k) * dist_vector[temp_k];
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat.at(temp_i + j) = std::exp(dist_sq * gamma);
//                     score_mat[j * totalCorresNum + i] = score_mat[temp_i + j]; // 确保矩阵对称
//                 }
//             }
//             break;
//         }
//         case ScoreFormula::QUADRATIC_FALLOFF: {
//             // 预计算二次衰减的系数，用乘法代替除法以提高效率
//             const double inv_thresh_sq = 1.0 / (inlier_thresh * inlier_thresh);
// #pragma omp parallel for schedule(static) default(none) shared(inv_thresh_sq, dist_vector, score_mat, totalCorresNum, totalCorresNum_2)
//             for (int i = 0; i < totalCorresNum; ++i) {
//                 const long long temp_i = i * totalCorresNum;
//                 for (int j = i; j < totalCorresNum; ++j) {
//                     const long long temp_k = (totalCorresNum_2 - i) * i / 2 + j;
//                     const double dist_sq = dist_vector[temp_k] * dist_vector[temp_k];
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
// void calculate_scores_omp(const double *dist_vector, double *score_vector, long long size,
//                           ScoreFormula formula, double alpha_dis, double inlier_thresh) {
//     const long long dis_size = totalCorresNum * (totalCorresNum - 1) / 2;
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
//     // totalCorresNum is the size of the correspondences, which is also a global variable
//     Eigen::MatrixXd graph = Eigen::MatrixXd::Zero(totalCorresNum, totalCorresNum);
//     const float alpha_dis = 10 * resolution;
//     const long long distance_size = static_cast<long long>(totalCorresNum) * (totalCorresNum - 1) / 2;
//     // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
//     std::vector<double> src_mat(totalCorresNum * 3, 0.0), tgt_mat(totalCorresNum * 3, 0.0),
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
//     // std::vector<double> da(pdist_size);
//     // std::vector<double> db(pdist_size);
//     std::vector<double> d(distance_size, 0.0); // Initialize the distance vector with zeros
//
//     std::cout << "Calculating the distance vector..." << std::endl;
//     timing(0);
//     const std::vector<double> da = pdist_naive_parallel(src_mat.data(), totalCorresNum, 3);
//     const std::vector<double> db = pdist_naive_parallel(tgt_mat.data(), totalCorresNum, 3);
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


inline float getDistance(const pcl::PointXYZ &A, const pcl::PointXYZ &B) {
    float distance = 0;
    const float d_x = A.x - B.x;
    const float d_y = A.y - B.y;
    const float d_z = A.z - B.z;
    distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if (!isfinite(distance)) {
        std::cout << YELLOW << "Warning, infinite distance occurred: " << distance << "\t" << A.x << " " << A.y << " "
                << A.z << "\t" << B.x << " " << B.y << " " << B.z << std::endl;
    }
    return distance;
}

inline float dynamicThreshold(float dis) {
    // Calculate a dynamic threshold based on the model of Lidar and depth camera (under development)
    float sigma = 0.0; // Under development
    return 0.0f;
}


void save_matrix(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << YELLOW << "Cannot open file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j < mat.cols() - 1)
                file << " ";
        }
        file << "\n";
    }
    file.close();
}


// igraph need eigen matrix be double type
// Optimization is still needed
// TODO: list version graph can be applied
Eigen::MatrixXd graphConstruction(std::vector<CorresStruct> &correspondences, float resolution,
                                  bool secondOrderGraphFlag, ScoreFormula formula) {
    // Construct a graph from the correspondences. The graph is represented as an adjacency matrix
    // TODO: Is there a more efficient way to construct or represent the graph?
    // totalCorresNum is the size of the correspondences, which is also a global variable
    Eigen::MatrixXd graph = Eigen::MatrixXd::Zero(totalCorresNum, totalCorresNum);
    const float alphaDis = 10 * resolution;
    const double gamma = -1.0 / (2.0 * alphaDis * alphaDis);
    const long long distance_size = static_cast<long long>(totalCorresNum) * (totalCorresNum - 1) / 2;
    // note that src_mat and tgt_mat are actually point vectors (nx3 for x y z)
    std::vector<float> src_mat(totalCorresNum * 3, 0.0), tgt_mat(totalCorresNum * 3, 0.0),
            score_mat(totalCorresNum * totalCorresNum, 0.0), score_vec(distance_size);

    timing(0);
    int localTotalEdges = 0;
    switch (formula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
#pragma omp parallel for schedule(static) default(none) shared(totalCorresNum, correspondences, graph, gamma) reduction(+:localTotalEdges)
            for (int i = 0; i < totalCorresNum; ++i) {
                CorresStruct c1 = correspondences[i];
                for (int j = i + 1; j < totalCorresNum; ++j) {
                    CorresStruct c2 = correspondences[j];
                    float src_dis = getDistance(c1.src, c2.src);
                    float tgt_dis = getDistance(c1.tgt, c2.tgt);
                    float dis = src_dis - tgt_dis;
                    double score = exp(dis * dis * gamma);
                    // score = (score < dynamicThreshold(dis)) ? 0 : score;
                    score = (score < 0.8) ? 0.0 : score;
                    graph(i, j) = score;
                    graph(j, i) = score;
                    localTotalEdges++; // This is not correct!
                }
            }
            break;
        }
        case ScoreFormula::QUADRATIC_FALLOFF: {
#pragma omp parallel for schedule(static) default(none) shared(totalCorresNum, correspondences, graph, alphaDis)reduction(+:localTotalEdges)
            for (int i = 0; i < totalCorresNum; ++i) {
                CorresStruct c1 = correspondences[i];
                for (int j = i + 1; j < totalCorresNum; ++j) {
                    CorresStruct c2 = correspondences[j];
                    float src_dis = getDistance(c1.src, c2.src);
                    float tgt_dis = getDistance(c1.tgt, c2.tgt);
                    float dis = src_dis - tgt_dis;
                    double score = 1 - dis * dis / alphaDis * alphaDis;
                    score = (score < dynamicThreshold(dis)) ? 0 : score;
                    graph(i, j) = score;
                    graph(j, i) = score;
                    localTotalEdges++;
                }
            }
            break;
        }
    }
    int totalEdges = localTotalEdges;
    std::cout << "First order graph has been constructed" << std::endl;
    std::cout << "Total edges for first order graph: " << totalEdges << std::endl;  // 输出边的总数
    timing(1);

    if (Eigen::MatrixXd graphEigenTemp = graph - graph.transpose(); graphEigenTemp.norm() == 0) {
        std::cout << GREEN << "graphEigenTemp is zero! first order graph is symmetric" << RESET << std::endl;
    } else {
        std::cout << RED << "graphEigenTemp is not zero! first order graph is not symmetric" << RESET << std::endl;
    }
    // save_matrix(graph, "../temp_test/cmp_score.txt");
    timing(0);
    // Second order graphing is time-consuming, size 6000 will use up to 2s
    if (secondOrderGraphFlag) {
        // Eigen::setNbThreads(16);
        graph = graph.cwiseProduct(graph * graph);
        int nonZeroCount = 0;
#pragma omp parallel for reduction(+:nonZeroCount)
        for (int i = 0; i < graph.rows(); ++i) {
            for (int j = 0; j < graph.cols(); ++j) {
                if (graph(i, j) != 0.0) {
                    nonZeroCount++;
                }
            }
        }
        graph = (graph + graph.transpose()) / 2.0; // Ensure symmetry
        std::cout << "Second order graph has been constructed" << std::endl;
        std::cout << "Total edges for second order graph: " << nonZeroCount << std::endl;  // 输出边的总数
        if (Eigen::MatrixXd graphEigenTemp = graph - graph.transpose(); graphEigenTemp.norm() == 0) {
            std::cout << GREEN << "graphEigenTemp is zero! second order graph is symmetric" << RESET << std::endl;
        } else {
            std::cout << RED << "graphEigenTemp is not zero! second order graph is not symmetric" << RESET << std::endl;
        }
    }
    return graph;
}


// // TODO: This function needs optimization
float otsuThresh(std::vector<float> all_scores)
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
// Decremental
bool compareLocalScore(const VertexStruct &v1, const VertexStruct &v2) {
    return v1.score > v2.score;
}
// Decremental
bool compareVertexCliqueScore(const LocalClique &l1, const LocalClique &l2) {
    return l1.score > l2.score;
}
// Incremental
bool compareCorrespondenceIndex(const CorresStruct &c1, const CorresStruct &c2) {
    return c1.tgtIndex < c2.tgtIndex;
}
// Decremental
bool compareCliqueSize(const igraph_vector_int_t *v1, const igraph_vector_int_t *v2) {
    return igraph_vector_int_size(v1) > igraph_vector_int_size(v2);
}
bool compareClusterScore(const ClusterStruct &v1, const ClusterStruct &v2) {
    return v1.clusterSize > v2.clusterSize;
}

// Find the vertex score based on clique edge weight.
// Select the correspondences who have high scores
// sampledCorresIndex is the order of the correspondences that are selected which score is higher than average
// sampledCliqueIndex is the index of the neighbor of sampledCorresIndex that also locate in the high score clique
void cliqueSampling(const MACConfig &macConfig, Eigen::MatrixXd &graph, const igraph_vector_int_list_t *cliques, std::vector<int> &sampledCorresIndex,
                    std::vector<int> &sampledCliqueIndex) {
    std::vector<LocalClique> vertexCliqueScores(totalCorresNum);
    // Clear the outputs if they are mistakenly not empty
    sampledCorresIndex.clear();
    sampledCliqueIndex.clear();

    // std::vector<igraph_vector_int_t> cliques_vec_ptr;
    // cliques_vec_ptr.reserve(totalCliqueNum);

    // Assign current index
// #pragma omp parallel for
    for (int i = 0; i < totalCorresNum; i++) {
        vertexCliqueScores[i].currentInd = i;
    }

    // compute the weight of each clique
    // Weight of each clique is the sum of the weights of all edges in the clique
// #pragma omp parallel for
    for (int i = 0; i < totalCliqueNum; i++) {
        const igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(cliques, i);
        float weight = 0.0;
        const int length = igraph_vector_int_size(v); // size of the clique

        for (int j = 0; j < length; j++) {
            const int a = static_cast<int>(VECTOR(*v)[j]);
            for (int k = j + 1; k < length; k++) {
                const int b = static_cast<int>(VECTOR(*v)[k]);
                weight += graph(a, b);
            }
        }
        // cliques_vec_ptr.push_back(*v);
        // assign the weight to each correspondence in the clique
        for (int j = 0; j < length; j++) {
            const int k = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            vertexCliqueScores[k].cliqueIndScore.emplace_back(i, weight); // Weight of k-th correspondence in i-th clique
        }
    }

    float avg_score = 0;
    // sum the scores and assign it to the score member variable
// #pragma omp parallel for
    for (int i = 0; i < totalCorresNum; i++) {
        vertexCliqueScores[i].score = 0;
        // compute the score of each correspondence, clique_ind_score.size() is the number of cliques that the correspondence belongs to
        for (int j = 0; j < vertexCliqueScores[i].cliqueIndScore.size(); j++) {
            vertexCliqueScores[i].score += vertexCliqueScores[i].cliqueIndScore[j].score;
        }
// #pragma omp critical
        {
            avg_score += vertexCliqueScores[i].score;
        }
    }

    //
    sort(vertexCliqueScores.begin(), vertexCliqueScores.end(), compareVertexCliqueScore); //所有节点从大到小排序

    // 如果clique数目小于等于correspondence数目, clique number is small enough
    if (totalCliqueNum <= totalCorresNum) {
        for (int i = 0; i < totalCliqueNum; i++) {
            // Assign all cliques indexes to the sampledCliqueIndex in order.
            sampledCliqueIndex.push_back(i);
        }
        for (int i = 0; i < totalCorresNum; i++) {
            // sampledInd 中存放的是被选中的correspondence的index
            if (!vertexCliqueScores[i].score) {
                // skip if the score of correspondence is 0
                continue;
            }
            sampledCorresIndex.push_back(vertexCliqueScores[i].currentInd); // only keep index whose correspondence has a non-zero score
        }
        return;
    }

    std::unordered_set<int> visitedCliqueIndex;
    // Otherwise we only keep the correspondences whose score is greater than the average score
    avg_score /= static_cast<float>(totalCorresNum);
    for (int i = 0; i < totalCorresNum; i++) {
        // We only consider the correspondences whose score is greater than the average score
        // This can filter low score vertex (vertex and correspondence are the same thing)
        if (vertexCliqueScores[i].score < avg_score) break;
        sampledCorresIndex.push_back(vertexCliqueScores[i].currentInd);
        // Only keep index of correspondence whose score is higher than the average score, ordered
        // sort the clique_ind_score of each correspondence from large to small
        sort(vertexCliqueScores[i].cliqueIndScore.begin(), vertexCliqueScores[i].cliqueIndScore.end(), compareLocalScore); //局部从大到小排序
        int selectedCnt = 1;
        // Check top 10 neighbors of each correspondence in high score clique
        for (int j = 0; j < vertexCliqueScores[i].cliqueIndScore.size(); j++) {
            if (selectedCnt > macConfig.maxLocalCliqueNum) break;
            if (int ind = vertexCliqueScores[i].cliqueIndScore[j].currentIndex; visitedCliqueIndex.find(ind) == visitedCliqueIndex.end()) {
                visitedCliqueIndex.insert(ind);
            } else {
                continue;
            }
            selectedCnt++;
        }
    }
    // Keep the correspondences that have high neighboring score.
    // Its neighbor has high score, and it is in its neighbor's high score clique
    sampledCliqueIndex.assign(visitedCliqueIndex.begin(), visitedCliqueIndex.end()); // no order
    std::sort(sampledCliqueIndex.begin(), sampledCliqueIndex.end()); // 增加这一行

    // ---------------------------- Evaluation part ----------------------------
    // TODO: 这里加上检查最大若干团是否包含真实团
    // sort(cliques_vec_ptr.begin(), cliques_vec_ptr.end(), compareCliqueSize); // sort the cliques from large to small
    // std::cout << "Top 5 largest cliques: " << igraph_vector_int_size(&cliques_vec_ptr[0]) << " " << cliques_vec_ptr[0]<< std::endl;


    // -------------------------------------------------------------------------
}

// TODO: Check this function
// TODO: This function is not optimized
// Our source target pair is a normal but non-invertible function (surjective, narrowly), which means a source can only have a single target,
// but a target may have many sources. This function is used to find target source pair, where target paired with various sources.
// Only happen if we use one way matching method
void makeTgtSrcPair(const std::vector<CorresStruct> &correspondence,
                       std::vector<std::pair<int, std::vector<int> > > &tgtSrc) {
    //需要读取保存的kpts, 匹配数据按照索引形式保存
    assert(correspondence.size() > 1); // 保留一个就行
    if (correspondence.size() < 2) {
        std::cout << "The correspondence vector is empty." << std::endl;
    }
    tgtSrc.clear();
    std::vector<CorresStruct> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    std::sort(corr.begin(), corr.end(), compareCorrespondenceIndex); // sort by target index increasing order
    int tgt = corr[0].tgtIndex;
    std::vector<int> src;
    src.push_back(corr[0].srcIndex);
    for (int i = 1; i < corr.size(); i++) {
        if (corr[i].tgtIndex != tgt) {
            tgtSrc.emplace_back(tgt, src);
            src.clear();
            tgt = corr[i].tgtIndex;
        }
        src.push_back(corr[i].srcIndex);
    }
    corr.clear();
    corr.shrink_to_fit();
}

// TODO: This function is not optimized
// TODO: We only get the logic check
void weightSvd(PointCloudPtr &srcPts, PointCloudPtr &tgtPts, Eigen::VectorXf &weights, float weightThreshold,
                Eigen::Matrix4f &transMat) {
    for (int i = 0; i < weights.size(); i++) {
        weights(i) = (weights(i) < weightThreshold) ? 0 : weights(i);
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
    pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*srcPts);
    pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*tgtPts);
    //获取点云质心
    src_it.reset();
    des_it.reset();
    Eigen::Matrix<float, 4, 1> centroid_src, centroid_des;
    pcl::compute3DCentroid(src_it, centroid_src);
    pcl::compute3DCentroid(des_it, centroid_des);

    //去除点云质心
    src_it.reset();
    des_it.reset();
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
    if (u.determinant() * v.determinant() < 0) {
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
    transMat = Trans;
}


// TODO: This function is not optimized
// TODO: We only get the logic check
// Overall Average Mean Absolute Error (OAMAE)
// | Metrix           | Intro                                             | Robust         |
// | ---------------- | ------------------------------------------------- | -------------- |
// | **OAMAE**        | Overall Average Mean Absolute Error               | More than RMSE |
// | RMSE             | Squared error averaging, emphasis on large errors | No             |
// | Chamfer Distance | Sum or average of nearest neighbor distances      | For pointcloud |
// | EMD              | Earth Mover's Distance                            | rigorous but computationally expensive |
float OAMAE(PointCloudPtr &raw_src, PointCloudPtr &raw_des, Eigen::Matrix4f &est,
            std::vector<std::pair<int, std::vector<int> > > &des_src, float thresh) {
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for (auto &i: des_src) {
        int des_ind = i.first;
        std::vector<int> src_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        for (auto &e: src_ind) {
            if (!pcl::isFinite(src_trans->points[e])) continue;
            //计算距离
            float distance = getDistance(src_trans->points[e], raw_des->points[des_ind]);
            if (distance < thresh) {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}

float calculateRotationError(Eigen::Matrix3f &est, Eigen::Matrix3f &gt) {
    float tr = (est.transpose() * gt).trace();
    return acos(std::min(std::max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

float calculateTranslationError(Eigen::Vector3f &est, Eigen::Vector3f &gt) {
    Eigen::Vector3f t = est - gt;
    return sqrt(t.dot(t)) * 100;
}

float evaluateTransByLocalClique(const PointCloudPtr &srcCorrPts, const PointCloudPtr &tgtCorrPts, const Eigen::Matrix4f &trans,
    const float metricThresh, const std::string &metric) {
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*srcCorrPts, *src_trans, trans);
    src_trans->is_dense = false;
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if (!src_trans->size()) return 0;
    float score = 0.0;
    int inlier = 0;
    const int corr_num = srcCorrPts->points.size();
    for (int i = 0; i < corr_num; i++) {
        if (const float dist = getDistance(src_trans->points[i], tgtCorrPts->points[i]); dist < metricThresh) {
            constexpr float w = 1;
            inlier++;
            if (metric == "inlier") {
                score += 1 * w; //correspondence[i].inlier_weight; <- commented by the MAC++ author
            } else if (metric == "MAE") {
                score += (metricThresh - dist) * w / metricThresh;
            } else if (metric == "MSE") {
                score += pow((metricThresh - dist), 2) * w / pow(metricThresh, 2);
            }
        }
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}


bool evaluationEst(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float re_thresh, float te_thresh, double &RE,
                    double &TE) {
    Eigen::Matrix3f rotation_est = est.topLeftCorner(3, 3);
    Eigen::Matrix3f rotation_gt = gt.topLeftCorner(3, 3);
    Eigen::Vector3f translation_est = est.block(0, 3, 3, 1);
    Eigen::Vector3f translation_gt = gt.block(0, 3, 3, 1);

    RE = calculateRotationError(rotation_est, rotation_gt);
    TE = calculateTranslationError(translation_est, translation_gt);
    if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh) {
        return true;
    }
    return false;
}


// ################################################################
float g_angleThreshold = 5.0 * M_PI / 180; //5 degree
float g_distanceThreshold = 0.1;
#ifndef M_PIf32
#define M_PIf32 3.1415927f
#endif

bool EnforceSimilarity1(const pcl::PointXYZINormal &point_a, const pcl::PointXYZINormal &point_b,
                        float squared_distance) {
    if (point_a.normal_x == 666 || point_b.normal_x == 666 || point_a.normal_y == 666 || point_b.normal_y == 666 ||
        point_a.normal_z == 666 || point_b.normal_z == 666) {
        return false;
    }
    Eigen::VectorXf temp(3);
    temp[0] = point_a.normal_x - point_b.normal_x;
    temp[1] = point_a.normal_y - point_b.normal_y;
    temp[2] = point_a.normal_z - point_b.normal_z;
    if (temp.norm() < g_distanceThreshold) {
        return true;
    }
    return false;
}

// Check if the Euler angles are within the valid range
bool checkEulerAngles(const float angle) {
    if (isfinite(angle) && angle >= -M_PIf32 && angle <= M_PIf32) {
        return true;
    }
    return false;
}

int clusterTransformationByRotation(const std::vector<Eigen::Matrix3f> &Rs, const std::vector<Eigen::Vector3f> &Ts,
                                    const float angleThresh, const float disThresh, pcl::IndicesClusters &clusters,
                                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans) {
    if (Rs.empty() || Ts.empty() || Rs.size() != Ts.size()) {
        std::cout << YELLOW << "Rs and Ts are empty or not the same size!" << RESET << std::endl;
        return -1;
    }
    const int num = Rs.size();
    g_distanceThreshold = disThresh;
    trans->resize(num);
    for (size_t i = 0; i < num; i++) {
        Eigen::Transform<float, 3, Eigen::Affine> R(Rs[i]);
        pcl::getEulerAngles<float>(R, (*trans)[i].x, (*trans)[i].y, (*trans)[i].z); // R -> trans
        // 去除无效解
        if (!checkEulerAngles((*trans)[i].x) || !checkEulerAngles((*trans)[i].y) || !checkEulerAngles((*trans)[i].z)) {
            std::cout << "INVALID POINT" << std::endl;
            (*trans)[i].x = 666;
            (*trans)[i].y = 666;
            (*trans)[i].z = 666;
            (*trans)[i].normal_x = 666;
            (*trans)[i].normal_y = 666;
            (*trans)[i].normal_z = 666;
        } else {
            // 需要解决同一个角度的正负问题 6.14   平面 y=PI 右侧的解（需要验证） 6.20
            // -pi - pi -> 0 - 2pi
            (*trans)[i].x = ((*trans)[i].x < 0 && (*trans)[i].x >= -M_PIf32)
                                ? (*trans)[i].x + 2 * M_PIf32
                                : (*trans)[i].x;
            (*trans)[i].y = ((*trans)[i].y < 0 && (*trans)[i].y >= -M_PIf32)
                                ? (*trans)[i].y + 2 * M_PIf32
                                : (*trans)[i].y;
            (*trans)[i].z = ((*trans)[i].z < 0 && (*trans)[i].z >= -M_PIf32)
                                ? (*trans)[i].z + 2 * M_PIf32
                                : (*trans)[i].z;
            (*trans)[i].normal_x = static_cast<float>(Ts[i][0]);
            (*trans)[i].normal_y = static_cast<float>(Ts[i][1]);
            (*trans)[i].normal_z = static_cast<float>(Ts[i][2]);
        }
    }

    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true); // true for using dense mode, no NaN points
    cec.setInputCloud(trans);
    cec.setConditionFunction(&EnforceSimilarity1);
    cec.setClusterTolerance(angleThresh);
    cec.setMinClusterSize(2); // cluster size
    cec.setMaxClusterSize(static_cast<int>(num)); // nearly impossible to reach the maximum?
    cec.segment(clusters);
    for (int i = 0; i < clusters.size(); ++i) {
        for (int j = 0; j < clusters[i].indices.size(); ++j) {
            // Set intensity of each cluster point to their cluster number
            trans->points[clusters[i].indices[j]].intensity = i;
        }
    }
    return 0;
}


float rmseCompute(const PointCloudPtr &cloud_source, const PointCloudPtr &cloud_target, Eigen::Matrix4f &Mat_est,
                   Eigen::Matrix4f &Mat_GT, float mr) {
    float RMSE_temp = 0.0f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
    std::vector<int> overlap_idx;
    float overlap_thresh = 4 * mr;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
    pcl::PointXYZ query_point;
    std::vector<int> pointIdx;
    std::vector<float> pointDst;
    kdtree1.setInputCloud(cloud_target);
    for (int i = 0; i < cloud_source_trans_GT->points.size(); i++) {
        query_point = cloud_source_trans_GT->points[i];
        kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
        if (sqrt(pointDst[0]) <= overlap_thresh)
            overlap_idx.push_back(i);
    }
    //
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
    kdtree2.setInputCloud(cloud_source_trans_GT);
    for (int i = 0; i < overlap_idx.size(); i++) {
        //query_point = cloud_source_trans_EST->points[overlap_idx[i]];
        //kdtree2.nearestKSearch(query_point,1,pointIdx,pointDst); RMSE_temp+=sqrt(pointDst[0]);
        float dist_x = pow(
            cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
        float dist_y = pow(
            cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
        float dist_z = pow(
            cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
        float dist = sqrt(dist_x + dist_y + dist_z);
        RMSE_temp += dist;
    }
    RMSE_temp /= overlap_idx.size();
    RMSE_temp /= mr;
    //
    return RMSE_temp;
}


void postRefinement(std::vector<CorresStruct> &correspondence, PointCloudPtr &src_corr_pts,
                     PointCloudPtr &des_corr_pts, Eigen::Matrix4f &initial/* 由最大团生成的变换 */, float &best_score,
                     float inlier_thresh, int iterations, const std::string &metric) {
    int pointNum = src_corr_pts->points.size();
    float pre_score = best_score;
    for (int i = 0; i < iterations; i++) {
        float score = 0;
        Eigen::VectorXf weights, weight_pred;
        weights.resize(pointNum);
        weights.setZero();
        std::vector<int> pred_inlier_index;
        PointCloudPtr trans(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*src_corr_pts, *trans, initial);
        //remove nan points
        trans->is_dense = false;
        std::vector<int> mapping;
        pcl::removeNaNFromPointCloud(*trans, *trans, mapping);
        if (!trans->size()) return;
        for (int j = 0; j < pointNum; j++) {
            float dist = getDistance(trans->points[j], des_corr_pts->points[j]);
            float w = 1;
            // if (flagAddOverlap)
            // {
            // 	w = correspondence[j].score;
            // }
            if (dist < inlier_thresh) {
                pred_inlier_index.push_back(j);
                weights[j] = 1 / (1 + pow(dist / inlier_thresh, 2));
                if (metric == "inlier") {
                    score += 1 * w;
                } else if (metric == "MAE") {
                    score += (inlier_thresh - dist) * w / inlier_thresh;
                } else if (metric == "MSE") {
                    score += pow((inlier_thresh - dist), 2) * w / pow(inlier_thresh, 2);
                }
            }
        }
        if (score < pre_score) {
            break;
        } else {
            pre_score = score;
            //估计pred_inlier
            PointCloudPtr pred_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
            PointCloudPtr pred_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*src_corr_pts, pred_inlier_index, *pred_src_pts);
            pcl::copyPointCloud(*des_corr_pts, pred_inlier_index, *pred_des_pts);
            weight_pred.resize(pred_inlier_index.size());
            for (int k = 0; k < pred_inlier_index.size(); k++) {
                weight_pred[k] = weights[pred_inlier_index[k]];
            }
            //weighted_svd
            weightSvd(pred_src_pts, pred_des_pts, weight_pred, 0, initial);
            pred_src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
            pred_des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        }
        pred_inlier_index.clear();
        trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    best_score = pre_score;
}

std::vector<int> vectorsUnion(const std::vector<int> &v1, const std::vector<int> &v2) {
    std::vector<int> v;
    set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}

void getCorrPatch(std::vector<CorresStruct> &sampled_corr, PointCloudPtr &src, PointCloudPtr &des,
                  PointCloudPtr &src_batch, PointCloudPtr &des_batch, float radius) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_des;
    kdtree_src.setInputCloud(src);
    kdtree_des.setInputCloud(des);
    std::vector<int> src_ind, des_ind;
    std::vector<float> src_dis, des_dis;
    std::vector<int> src_batch_ind, des_batch_ind;
    for (int i = 0; i < sampled_corr.size(); i++) {
        kdtree_src.radiusSearch(sampled_corr[i].srcIndex, radius, src_ind, src_dis);
        kdtree_des.radiusSearch(sampled_corr[i].tgtIndex, radius, des_ind, des_dis);
        sort(src_ind.begin(), src_ind.end());
        sort(des_ind.begin(), des_ind.end());
        src_batch_ind = vectorsUnion(src_ind, src_batch_ind);
        des_batch_ind = vectorsUnion(des_ind, des_batch_ind);
    }
    pcl::copyPointCloud(*src, src_batch_ind, *src_batch);
    pcl::copyPointCloud(*des, des_batch_ind, *des_batch);
    return;
}

float truncatedChamferDistance(PointCloudPtr &src, PointCloudPtr &des, Eigen::Matrix4f &est, float thresh) {
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, est);
    //remove nan points
    src_trans->is_dense = false;
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if (!src_trans->size()) return 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(des);
    std::vector<int> src_ind(1), des_ind(1);
    std::vector<float> src_dis(1), des_dis(1);
    float score1 = 0, score2 = 0;
    int cnt1 = 0, cnt2 = 0;
    for (int i = 0; i < src_trans->size(); i++) {
        pcl::PointXYZ src_trans_query = (*src_trans)[i];
        if (!pcl::isFinite(src_trans_query)) continue;
        kdtree_des.nearestKSearch(src_trans_query, 1, des_ind, des_dis);
        if (des_dis[0] > pow(thresh, 2)) {
            continue;
        }
        score1 += (thresh - sqrt(des_dis[0])) / thresh;
        cnt1++;
    }
    score1 /= cnt1;
    for (int i = 0; i < des->size(); i++) {
        pcl::PointXYZ des_query = (*des)[i];
        if (!pcl::isFinite(des_query)) continue;
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, src_dis);
        if (src_dis[0] > pow(thresh, 2)) {
            continue;
        }
        score2 += (thresh - sqrt(src_dis[0])) / thresh;
        cnt2++;
    }
    score2 /= cnt2;
    return (score1 + score2) / 2;
}

std::vector<int> vectorsIntersection(const std::vector<int> &v1, const std::vector<int> &v2) {
    std::vector<int> v;
    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}


float OAMAE1tok(PointCloudPtr &raw_src, PointCloudPtr &raw_des, Eigen::Matrix4f &est,
                 std::vector<std::pair<int, std::vector<int> > > &src_des, float thresh) {
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for (auto &i: src_des) {
        int src_ind = i.first;
        std::vector<int> des_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        if (!pcl::isFinite(src_trans->points[src_ind])) continue;
        for (auto &e: des_ind) {
            //计算距离
            float distance = getDistance(src_trans->points[src_ind], raw_des->points[e]);
            if (distance < thresh) {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}


Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial,
                                        std::vector<Eigen::Matrix3f> &Rs, std::vector<Eigen::Vector3f> &Ts,
                                        PointCloudPtr &srcKpts, PointCloudPtr &des_kpts,
                                        std::vector<std::pair<int, std::vector<int> > > &desSrc, float thresh,
                                        Eigen::Matrix4f &gtMat, std::string folderpath) {
    //std::string cluster_eva = folderpath + "/cluster_eva.txt";
    //std::ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    double RE, TE;
    bool suc = evaluationEst(initial, gtMat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3, 3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = OAMAE(srcKpts, des_kpts, initial, desSrc, thresh);
    std::cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << std::endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    std::vector<std::pair<float, float> > RTdifference;
    float avg_Rdiff = 0, avg_Tdiff = 0;
    int n = 0;
    for (int i = 0; i < clusterTrans[best_index].indices.size(); i++) {
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix3f R = Rs[ind];
        Eigen::Vector3f T = Ts[ind];
        float R_diff = calculateRotationError(R, R_initial);
        float T_diff = calculateTranslationError(T, T_initial);
        if (isfinite(R_diff) && isfinite(T_diff)) {
            avg_Rdiff += R_diff;
            avg_Tdiff += T_diff;
            n++;
        }
        RTdifference.emplace_back(R_diff, T_diff);
    }
    avg_Tdiff /= n;
    avg_Rdiff /= n;

    for (int i = 0; i < clusterTrans[best_index].indices.size(); i++) {
        //继续缩小解空间
        if (!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second) || RTdifference[i].first > avg_Rdiff
            || RTdifference[i].second > avg_Tdiff) continue;
        //if(RTdifference[i].first > 5 || RTdifference[i].second > 10) continue;
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[ind];
        mat.topLeftCorner(3, 3) = Rs[ind];
        suc = evaluationEst(mat, gtMat, 15, 30, RE, TE);
        float score = OAMAE(srcKpts, des_kpts, mat, desSrc, thresh);
        //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
        if (score > max_score) {
            max_score = score;
            est = mat;
            std::cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << score <<
                    std::endl;
        }
    }
    //outfile.close();
    return est;
}

// 1tok version
Eigen::Matrix4f clusterInternalTransEva1(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial,
                                         std::vector<Eigen::Matrix3f> &Rs, std::vector<Eigen::Vector3f> &Ts,
                                         PointCloudPtr &src_kpts, PointCloudPtr &des_kpts,
                                         std::vector<std::pair<int, std::vector<int> > > &des_src, float thresh,
                                         Eigen::Matrix4f &GTmat, bool _1tok, std::string folderpath) {
    //std::string cluster_eva = folderpath + "/cluster_eva.txt";
    //std::ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    double RE, TE;
    bool suc = evaluationEst(initial, GTmat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3, 3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = 0.0;
    if (_1tok) {
        max_score = OAMAE1tok(src_kpts, des_kpts, initial, des_src, thresh);
    } else {
        max_score = OAMAE(src_kpts, des_kpts, initial, des_src, thresh);
    }
    std::cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << std::endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    std::vector<std::pair<float, float> > RTdifference;
    int n = 0;
    for (int i = 0; i < clusterTrans[best_index].indices.size(); i++) {
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix3f R = Rs[ind];
        Eigen::Vector3f T = Ts[ind];
        float R_diff = calculateRotationError(R, R_initial);
        float T_diff = calculateTranslationError(T, T_initial);
        RTdifference.emplace_back(R_diff, T_diff);
    }
    ///TODO RTdifference排序
    sort(RTdifference.begin(), RTdifference.end());
    int i = 0, cnt = 10;
    while (i < std::min(100, (int) clusterTrans[best_index].indices.size()) && cnt > 0) {
        ///TODO 第一个mat可能与initial一样
        //继续缩小解空间
        if (!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second)) {
            i++;
            continue;
        }
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[ind];
        mat.topLeftCorner(3, 3) = Rs[ind];
        if (i > 0 && (est.inverse() * mat - Eigen::Matrix4f::Identity(4, 4)).norm() < 0.01) {
            break;
        }
        suc = evaluationEst(mat, GTmat, 15, 30, RE, TE);
        float score = 0.0;
        if (_1tok) {
            score = OAMAE1tok(src_kpts, des_kpts, mat, des_src, thresh);
        } else {
            score = OAMAE(src_kpts, des_kpts, mat, des_src, thresh);
        }

        //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
        if (score > max_score) {
            max_score = score;
            est = mat;
            std::cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << score <<
                    std::endl;
            cnt--;
        }
        i++;
    }
    //outfile.close();
    return est;
}
