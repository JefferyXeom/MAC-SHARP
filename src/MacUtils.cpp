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

#include "MacUtils.hpp"
#include "MacConfig.hpp"
#include "MacData.hpp"


// Function to set the number of threads for OpenBLAS and OpenMP
// Note that there are used for eigen speed up and parallel for
// get_num_threads may provide wrong threads!
void settingThreads(const int desiredThreads) {
    // Configure OpenBLAS threads
    const int open_blas_max = openblas_get_num_threads();
    const int omp_max = omp_get_max_threads();
    LOG_INFO("OpenBLAS default threads: " << open_blas_max << ", OMP default threads: " << omp_max);
    if (desiredThreads == -1) {
        openblas_set_num_threads(open_blas_max);
        omp_set_num_threads(omp_max);
        LOG_INFO("Using maximum available threads for computation. OpenBLAS now set to use "
            << openblas_get_num_threads() << " threads, OMP now set to use " << omp_get_num_threads() << " threads.");
    } else {
        if (desiredThreads > open_blas_max) {
            openblas_set_num_threads(open_blas_max);
            omp_set_num_threads(omp_max);
            LOG_WARNING("Desired thread number exceeds device capacity: " << desiredThreads << " > " << open_blas_max
                << "Set both to maximum available: " << open_blas_max);
        } else if (desiredThreads < -1) {
            openblas_set_num_threads(open_blas_max);
            omp_set_num_threads(omp_max);
            LOG_WARNING("Desired thread number is invalid: " << desiredThreads << " < -1"
                << "Set both to maximum available: " << open_blas_max);
        } else {
            openblas_set_num_threads(desiredThreads);
            omp_set_num_threads(desiredThreads);
            LOG_INFO("Set OpenBLAS and OMP threads to: " << desiredThreads);
        }
    }
}

// Comparison functions
// Decremental
// Compare clique score for each vertex
bool compareLocalScore(const CliqueInfo &v1, const CliqueInfo &v2) {
    return v1.cliqueScore > v2.cliqueScore;
}

// Decremental
// Compare vertex score (sum of local clique scores)
bool compareVertexCliqueScore(const vertexCliqueSupport &l1, const vertexCliqueSupport &l2) {
    return l1.score > l2.score;
}

// Incremental
bool compareCorresTgtIndex(const CorresStruct &c1, const CorresStruct &c2) {
    return c1.tgtIndex < c2.tgtIndex;
}

// Decremental
bool compareCliqueSize(const igraph_vector_int_t *v1, const igraph_vector_int_t *v2) {
    return igraph_vector_int_size(v1) > igraph_vector_int_size(v2);
}

// bool compareClusterScore(const ClusterStruct &v1, const ClusterStruct &v2) {
//     return v1.clusterSize > v2.clusterSize;
// }

/**
 * @brief Euclidean distance between two points
 */
float getDistance(const pcl::PointXYZ &A, const pcl::PointXYZ &B) {
    const float d_x = A.x - B.x;
    const float d_y = A.y - B.y;
    const float d_z = A.z - B.z;
    const float distance = std::sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if (!std::isfinite(distance)) {
        std::cout << YELLOW << "Warning, infinite distance occurred: " << distance << "\t"
                << A.x << " " << A.y << " " << A.z << "\t"
                << B.x << " " << B.y << " " << B.z << std::endl;
    }
    return distance;
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
float oamae(const PointCloudPtr &src, const PointCloudPtr &tgt, const Eigen::Matrix4f &est,
            const std::vector<std::pair<int, std::vector<int> > > &tgtSrc, const float thresh) {
    float score = 0.0;
    const PointCloudPtr srcTrans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *srcTrans, est);
    for (const auto &[tgtIndex, srcIndices]: tgtSrc) {
        float counter = 0.0;
        float scoreSum = 0.0;
        for (auto &srcIndex: srcIndices) {
            if (!pcl::isFinite(srcTrans->points[srcIndex])) continue;
            //计算距离
            if (const float distance = getDistance(srcTrans->points[srcIndex], tgt->points[tgtIndex]); distance < thresh) {
                counter++;
                scoreSum += (thresh - distance) / thresh;
            }
        }
        score += counter > 0 ? (scoreSum / counter) : 0;
    }
    return score;
}

float calculateRotationError(Eigen::Matrix3f &est, const Eigen::Matrix3f &gt) {
    const float tr = (est.transpose() * gt).trace();
    return acos(std::min(std::max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI; // degree
}

float calculateTranslationError(const Eigen::Vector3f &est, const Eigen::Vector3f &gt) {
    const Eigen::Vector3f t = est - gt;
    return sqrt(t.dot(t));
}

// 这个函数的很多评估阈值还是写死的，请注意！！！!!!
bool evaluationEst(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float reThresh, float teThresh, float &RE,
                   float &TE) {
    Eigen::Matrix3f rotation_est = est.topLeftCorner(3, 3);
    Eigen::Matrix3f rotation_gt = gt.topLeftCorner(3, 3);
    Eigen::Vector3f translation_est = est.block(0, 3, 3, 1);
    Eigen::Vector3f translation_gt = gt.block(0, 3, 3, 1);
    RE = calculateRotationError(rotation_est, rotation_gt);
    TE = calculateTranslationError(translation_est, translation_gt);
    if (0 <= RE && RE <= reThresh && 0 <= TE && TE <= teThresh) {
        return true;
    }
    return false;
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


// TODO: Check this function
// TODO: This function is not optimized
// Our source target pair is a normal but non-invertible function (surjective, narrowly), which means a source can only have a single target,
// but a target may have many sources. This function is used to find target source pair, where target paired with various sources.
// Only happen if we use one way matching method
///////////////////////////////// temporary added here!!!
void makeTgtSrcPair(const std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &correspondence,
                       std::vector<std::pair<int, std::vector<int> > > &tgtSrc) {
    //需要读取保存的kpts, 匹配数据按照索引形式保存
    assert(correspondence.size() > 1); // 保留一个就行
    if (correspondence.size() < 2) {
        std::cout << "The correspondence vector is empty." << std::endl;
    }
    tgtSrc.clear();
    std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    std::sort(corr.begin(), corr.end(), compareCorresTgtIndex); // sort by target index increasing order
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
///////////////////////////////////// Temporary added here!!!
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

// TODO 还差这一步以及ICP！！！！！！！！！!!!!!!!!!!!!!!!!!
void postRefinement(std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &correspondence, PointCloudPtr &src_corr_pts,
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
        }
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
        pred_inlier_index.clear();
        trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    best_score = pre_score;
}

// std::vector<int> vectorsUnion(const std::vector<int> &v1, const std::vector<int> &v2) {
//     std::vector<int> v;
//     std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
//     return v;
// }

/**
 * @brief 计算两个（无需预先排序的）vector<int>的并集。
 * @param v1 第一个vector
 * @param v2 第二个vector
 * @return 包含v1和v2所有不重复元素的、已排序的新vector。
 */
std::vector<int> vectorsUnion(const std::vector<int>& v1, const std::vector<int>& v2) {
    // 1. 创建一个足够大的新vector，先将v1的所有内容复制进去。
    std::vector<int> result;
    result.reserve(v1.size() + v2.size()); // 预分配内存，提高效率
    result.insert(result.end(), v1.begin(), v1.end());

    // 2. 再将v2的所有内容追加进去。
    //    现在 result 中包含了 v1 和 v2 的所有元素（可能有重复）。
    result.insert(result.end(), v2.begin(), v2.end());

    // 3. 对合并后的长vector进行排序。
    //    这是得到不重复并集的关键前置步骤。
    std::stable_sort(result.begin(), result.end());

    // 4. 使用 std::unique 将所有相邻的重复元素移动到vector的末尾，
    //    并返回一个指向第一个重复元素的迭代器。
    const auto last = std::unique(result.begin(), result.end());

    // 5. 使用 erase 方法，将从第一个重复元素开始到末尾的所有元素删除。
    result.erase(last, result.end());

    // 6. 返回最终的、已排序且不含重复元素的并集。
    return result;
}

void getCorrPatch(std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &sampledCorr, PointCloudPtr &src, PointCloudPtr &tgt,
                  PointCloudPtr &patchSrc, PointCloudPtr &patchTgt, float radius) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeSrc, kdtreeTgt;
    kdtreeSrc.setInputCloud(src);
    kdtreeTgt.setInputCloud(tgt);
    std::vector<int> srcInd, tgtInd;
    std::vector<float> srcDis, tgtDis;
    std::vector<int> patchSrcIndices, patchTgtIndices;
    for (auto & i : sampledCorr) {
        kdtreeSrc.radiusSearch(i.srcIndex, radius, srcInd, srcDis);
        kdtreeTgt.radiusSearch(i.tgtIndex, radius, tgtInd, tgtDis);
        sort(srcInd.begin(), srcInd.end());
        sort(tgtInd.begin(), tgtInd.end());
        patchSrcIndices = vectorsUnion(srcInd, patchSrcIndices);
        patchTgtIndices = vectorsUnion(tgtInd, patchTgtIndices);
    }
    pcl::copyPointCloud(*src, patchSrcIndices, *patchSrc);
    pcl::copyPointCloud(*tgt, patchTgtIndices, *patchTgt);
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

// std::vector<int> vectorsIntersection(const std::vector<int> &v1, const std::vector<int> &v2) {
//     std::vector<int> v;
//     set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
//     return v;
// }

/**
 * @brief Calculate the intersection of two unsorted vector<int>.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return A sorted vector containing the intersection of v1 and v2 (elements that are in both vectors).
 */
std::vector<int> vectorsIntersection(const std::vector<int>& v1, const std::vector<int>& v2) {
    // 1. Create a vector to store the intersection of v1 and v2.
    std::vector<int> result;

    // 2. Sort both input vectors.
    std::vector<int> sortedV1 = v1;
    std::vector<int> sortedV2 = v2;
    std::sort(sortedV1.begin(), sortedV1.end());
    std::sort(sortedV2.begin(), sortedV2.end());

    set_intersection(sortedV1.begin(), sortedV1.end(), sortedV2.begin(), sortedV2.end(), back_inserter(result));

    // 3. Use two iterators to traverse both sorted vectors.
    // auto it1 = sortedV1.begin();
    // auto it2 = sortedV2.begin();

    // while (it1 != sortedV1.end() && it2 != sortedV2.end()) {
    //     if (*it1 == *it2) {
    //         // If elements are equal, add it to the result.
    //         result.push_back(*it1);
    //         ++it1;
    //         ++it2;
    //     } else if (*it1 < *it2) {
    //         // Move the iterator for the first vector forward if the element is smaller.
    //         ++it1;
    //     } else {
    //         // Move the iterator for the second vector forward if the element is smaller.
    //         ++it2;
    //     }
    // }

    // 4. Return the result vector containing the intersection.
    return result;
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
    return score;
}


// Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial,
//                                         std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> &Rs, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &Ts,
//                                         PointCloudPtr &srcKpts, PointCloudPtr &des_kpts,
//                                         std::vector<std::pair<int, std::vector<int> > > &desSrc, float thresh,
//                                         Eigen::Matrix4f &gtMat, std::string folderpath) {
//     //std::string cluster_eva = folderpath + "/cluster_eva.txt";
//     //std::ofstream outfile(cluster_eva, ios::trunc);
//     //outfile.setf(ios::fixed, ios::floatfield);
//
//     float RE, TE;
//     bool suc = evaluationEst(initial, gtMat, 15, 30, RE, TE);
//
//
//     Eigen::Matrix3f R_initial = initial.topLeftCorner(3, 3);
//     Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
//     float max_score = oamae(srcKpts, des_kpts, initial, desSrc, thresh);
//     std::cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << std::endl;
//     //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
//     Eigen::Matrix4f est = initial;
//
//     //统计类内R T差异情况
//     std::vector<std::pair<float, float> > RTdifference;
//     float avg_Rdiff = 0, avg_Tdiff = 0;
//     int n = 0;
//     for (int i = 0; i < clusterTrans[best_index].indices.size(); i++) {
//         int ind = clusterTrans[best_index].indices[i];
//         Eigen::Matrix3f R = Rs[ind];
//         Eigen::Vector3f T = Ts[ind];
//         float R_diff = calculateRotationError(R, R_initial);
//         float T_diff = calculateTranslationError(T, T_initial);
//         if (isfinite(R_diff) && isfinite(T_diff)) {
//             avg_Rdiff += R_diff;
//             avg_Tdiff += T_diff;
//             n++;
//         }
//         RTdifference.emplace_back(R_diff, T_diff);
//     }
//     avg_Tdiff /= n;
//     avg_Rdiff /= n;
//
//     for (int i = 0; i < clusterTrans[best_index].indices.size(); i++) {
//         //继续缩小解空间
//         if (!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second) || RTdifference[i].first > avg_Rdiff
//             || RTdifference[i].second > avg_Tdiff)
//             continue;
//         //if(RTdifference[i].first > 5 || RTdifference[i].second > 10) continue;
//         int ind = clusterTrans[best_index].indices[i];
//         Eigen::Matrix4f mat;
//         mat.setIdentity();
//         mat.block(0, 3, 3, 1) = Ts[ind];
//         mat.topLeftCorner(3, 3) = Rs[ind];
//         suc = evaluationEst(mat, gtMat, 15, 30, RE, TE);
//         float score = oamae(srcKpts, des_kpts, mat, desSrc, thresh);
//         //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
//         if (score > max_score) {
//             max_score = score;
//             est = mat;
//             std::cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << score <<
//                     std::endl;
//         }
//     }
//     //outfile.close();
//     return est;
// }

// 1tok version
Eigen::Matrix4f clusterInternalTransEva1(const pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial,
                                         std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> &Rs, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &Ts,
                                         PointCloudPtr &src_kpts, PointCloudPtr &des_kpts,
                                         std::vector<std::pair<int, std::vector<int> > > &tgtSrc, const float thresh,
                                         Eigen::Matrix4f &GTmat, bool _1tok, std::string folderpath) {
    //std::string cluster_eva = folderpath + "/cluster_eva.txt";
    //std::ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    float RE, TE;
    bool suc = evaluationEst(initial, GTmat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3, 3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = 0.0;
    if (_1tok) {
        max_score = OAMAE1tok(src_kpts, des_kpts, initial, tgtSrc, thresh);
    } else {
        max_score = oamae(src_kpts, des_kpts, initial, tgtSrc, thresh);
    }
    std::cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << std::endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    std::vector<std::pair<float, float> > RTdifference;
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
    while (i < std::min(100, static_cast<int>(clusterTrans[best_index].indices.size())) && cnt > 0) {
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
            score = OAMAE1tok(src_kpts, des_kpts, mat, tgtSrc, thresh);
        } else {
            score = oamae(src_kpts, des_kpts, mat, tgtSrc, thresh);
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
    return est;
}
