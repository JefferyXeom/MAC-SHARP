//
// Created by Jeffery_Xeom on 2025/8/25.
// Project: MAC_SHARP
// File: MacRtHypothesis.cpp
//


#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <unordered_set>
#include <fstream>
#include <iomanip>

#include "CommonTypes.hpp"
#include "MacTimer.hpp"
#include "MacRtHypothesis.hpp"

#include <pcl/filters/filter.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/impl/conditional_euclidean_clustering.hpp>

#include "MacUtils.hpp"
// [Added] Monitor inline wrappers for internal sub-stage timing/metrics.
#include "MacMonitor.hpp"  // MON_ENTER / MON_RECORD / MON_SET_KV / MON_SET_NOTE


MacRtHypothesis::MacRtHypothesis(MacData &data, const MacConfig &config, const MacGraph &graph, MacResult &result)
    : data_(data),
      config_(config),
      graph_(graph),
      result_(result) {
    sampledCorrSrc_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    sampledCorrTgt_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    transPoints_.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
}

// ==============================================================================
// ==                       核心函数的实现                       ==
// ==============================================================================

/**
 * @brief 步骤1: Find the vertex score based on clique edge weight.
 * Select the correspondences who have high scores
 * sampledCorresIndices_ is the index of the correspondences that are selected which score is higher than average
 * sampledCliqueIndices_ is the index of the cliques that are selected with high score
 */
void MacRtHypothesis::sampleCandidates(const MacGraph &graph) {
    LOG_INFO("--- 1. Sampling candidates from cliques ---");
    // This stores the clique scores (sum of edge weights in the clique) for each correspondence
    // One correspondence may belong to multiple cliques, so we need a vector (CliqueInfo) to store them
    std::vector<vertexCliqueSupport> vertexCliqueSupports(data_.totalCorresNum);
    // Clear the outputs if they are mistakenly not empty
    sampledCorresIndices_.clear();
    sampledCliqueIndices_.clear();

    // Assign current index
    // Later we will sort the vertexCliqueSupports based on score, so we need to keep track of original index
    // ---- [MON] hypo/sample/init_index ----
    MON_ENTER("hypo/sample/init_index");
    // #pragma omp parallel for
    for (int i = 0; i < data_.totalCorresNum; i++) {
        vertexCliqueSupports[i].vertexIndex = i;
    }
    MON_RECORD(); // close hypo/sample/init_index

    // compute the weight of each clique
    // Weight of each clique is the sum of the weights of all edges in the clique
    // ---- [MON] hypo/sample/score_aggregation ----
    MON_ENTER("hypo/sample/score_aggregation");
    const auto &graphMatrix = graph.getGraphEigen();
    // #pragma omp parallel for
    for (int i = 0; i < data_.totalCliqueNum; i++) {
        const igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(graph.getCliques(), i);
        float weight = 0.0;
        const int length = igraph_vector_int_size(v); // size of the clique

        for (int j = 0; j < length; ++j) {
            const int a = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            for (int k = j + 1; k < length; ++k) {
                const int b = static_cast<int>(VECTOR(*v)[k]); // Global index for k-th vertex in i-th clique
                weight += graphMatrix(a, b);
            }
        }
        // assign the weight to each vertex in the clique
        for (int j = 0; j < length; ++j) {
            const int k = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            vertexCliqueSupports[k].participatingCliques.emplace_back(i, weight);
            // Weight of i-th clique added to correspondence k
        }
    }

    float avg_score = 0;
    // For each vertex, iterate through all cliques it belongs to and sum up the clique scores
    // Ths sum of the score is the score of the vertex
    // #pragma omp parallel for
    for (int i = 0; i < data_.totalCorresNum; ++i) {
        // compute the score of each correspondence, clique_ind_score.size() is the number of cliques that the correspondence belongs to
        for (int j = 0; j < vertexCliqueSupports[i].participatingCliques.size(); ++j) {
            vertexCliqueSupports[i].score += vertexCliqueSupports[i].participatingCliques[j].cliqueScore;
        }
        // 写法能够优化，让omp自己去管理累加
        // #pragma omp critical
        {
            avg_score += vertexCliqueSupports[i].score;
        }
    }
    // If you maintain any summary like mean/median, you can also record it here.
    // e.g., MON_SET_KV("hypo/score_mean", scoreMean);
    MON_RECORD(); // close hypo/sample/score_aggregation

    // Attention! From now on, vertexCliqueSupports is sorted based on score
    std::stable_sort(vertexCliqueSupports.begin(), vertexCliqueSupports.end(), compareVertexCliqueScore); //所有节点从大到小排序

    // ---- [MON] hypo/sample/select_vertices ----
    MON_ENTER("hypo/sample/select_vertices");
    // 如果clique数目小于等于correspondence数目, clique number is small enough
    if (data_.totalCliqueNum <= data_.totalCorresNum) {
        for (int i = 0; i < data_.totalCliqueNum; i++) {
            // Assign all cliques indexes to the sampledCliqueIndices_ in order.
            sampledCliqueIndices_.push_back(i);
        }
        for (int i = 0; i < data_.totalCorresNum; i++) {
            if (vertexCliqueSupports[i].score == 0.0f) {
                // skip if the score of correspondence is 0
                continue;
            }
            sampledCorresIndices_.push_back(vertexCliqueSupports[i].vertexIndex);
            // only keep index whose correspondence has a non-zero score
        }
        MON_CANCEL();
        return;
    }
    MON_SET_KV("hypo/sampled_correspondences",
               static_cast<int>(sampledCorresIndices_.size())); // replace with your actual container if different
    MON_RECORD(); // close hypo/sample/select_vertices

    // ---- [MON] hypo/sample/select_cliques ----
    MON_ENTER("hypo/sample/select_cliques");
    std::unordered_set<int> visitedCliqueIndex;
    // Otherwise we only keep the correspondences whose score is greater than the average score
    avg_score /= static_cast<float>(data_.totalCorresNum);
    for (int i = 0; i < data_.totalCorresNum; ++i) {
        // We only consider the correspondences whose score is greater than the average score
        // This can filter low score vertex (vertex and correspondence are the same thing)
        if (vertexCliqueSupports[i].score < avg_score) break;
        // This is also in order of score from high to low
        sampledCorresIndices_.push_back(vertexCliqueSupports[i].vertexIndex);
        // Only keep index of correspondence whose score is higher than the average score, ordered
        // sort the clique_ind_score of each correspondence from large to small
        std::stable_sort(vertexCliqueSupports[i].participatingCliques.begin(),
                         vertexCliqueSupports[i].participatingCliques.end(), compareLocalScore); //局部从大到小排序
        int selectedCnt = 1;
        // Check top 10 neighbors of each correspondence in high score clique
        for (int j = 0; j < vertexCliqueSupports[i].participatingCliques.size(); ++j) {
            if (selectedCnt == config_.maxLocalCliqueNum) break;
            if (int ind = vertexCliqueSupports[i].participatingCliques[j].cliqueIndex;
                visitedCliqueIndex.find(ind) == visitedCliqueIndex.end()) {
                visitedCliqueIndex.insert(ind);
            } else {
                continue;
            }
            selectedCnt++;
        }
    }
    // Keep the correspondences that have high neighboring score.
    // Its neighbor has high score, and it is in its neighbor's high score clique
    sampledCliqueIndices_.assign(visitedCliqueIndex.begin(), visitedCliqueIndex.end()); // no order
    std::stable_sort(sampledCliqueIndices_.begin(), sampledCliqueIndices_.end()); // ordered
    MON_SET_KV("hypo/sampled_cliques",
               static_cast<int>(sampledCliqueIndices_.size())); // replace with your actual container if different
    MON_RECORD(); // close hypo/sample/select_cliques

    // ---------------------------- Evaluation part ----------------------------
    // TODO: 这里加上检查最大若干团是否包含真实团
    // LOG_DEBUG("----------------Evaluation part----------------");

    LOG_INFO("Sampled " << sampledCliqueIndices_.size() << " cliques and "
        << sampledCorresIndices_.size() << " correspondences.");
}

/**
 * @brief [私有] 步骤2：根据采样索引，准备好后续步骤所需的点云和对应关系数据
 */
void MacRtHypothesis::prepareSampledData() {
    LOG_INFO("--- 2. Preparing data from sampled correspondences ---");
    // ---- [MON] hypo/prepare/gather_pairs ----
    MON_ENTER("hypo/prepare/gather_pairs");

    // 清空旧数据，确保每次运行都是干净的状态
    sampledCorr_.clear();
    sampledCorrSrc_->clear();
    sampledCorrTgt_->clear();

    // 调整容器大小以提高效率
    sampledCorr_.reserve(sampledCorresIndices_.size());
    sampledCorrSrc_->reserve(sampledCorresIndices_.size());
    sampledCorrTgt_->reserve(sampledCorresIndices_.size());

    int inlierNumAfterSampling = 0;
    for (const auto &index: sampledCorresIndices_) {
        const auto &originalCorr = data_.corres[index];
        sampledCorr_.push_back(originalCorr);
        sampledCorrSrc_->push_back(originalCorr.src);
        sampledCorrTgt_->push_back(originalCorr.tgt);
        // ---------------------------- Evaluation part ----------------------------
        if (!data_.gtInlierLabels.empty() && data_.gtInlierLabels[index] == 1) {
            inlierNumAfterSampling++;
        }
        // -------------------------------------------------------------------------
    }

    // ---------------------------- Evaluation part ----------------------------
    LOG_DEBUG("----------------Evaluation part----------------");
    // Save log
    // std::string sampledCorrTxt = config_.outputPath + "/sampled_corr.txt";
    // std::ofstream outFile1;
    // outFile1.open(sampledCorrTxt.c_str(), std::ios::out);
    // for (auto &i: sampledCorr_) {
    //     outFile1 << i.srcIndex << " " << i.tgtIndex << std::endl;
    // }
    // outFile1.close();
    //
    // std::string sampledCorrLabel = config_.outputPath + "/sampled_corr_label.txt";
    // std::ofstream outFile2;
    // outFile2.open(sampledCorrLabel.c_str(), std::ios::out);
    // for (auto &ind: sampledCorresIndices_) {
    //     if (data_.gtInlierLabels[ind]) {
    //         outFile2 << "1" << std::endl;
    //     } else {
    //         outFile2 << "0" << std::endl;
    //     }
    // }
    // outFile2.close();

    // 这部分统计代码可以保留在这里，因为它紧跟着数据的创建
    float inlierRatioAfterSampling = 0.0f;
    if (!sampledCorresIndices_.empty()) {
        inlierRatioAfterSampling = static_cast<float>(inlierNumAfterSampling) / static_cast<float>(sampledCorresIndices_
                                       .size()) * 100;
        LOG_INFO("Inlier ratio after sampling: " << inlierRatioAfterSampling << "%");
        // -------------------------------------------------------------------------
    }
    MON_SET_KV("hypo/sampled_count",
               static_cast<int>(sampledCorresIndices_.size()));
    // If you compute inlier ratio after sampling, record it here:
    MON_SET_KV("hypo/inlier_num_after_sampling", inlierNumAfterSampling);
    MON_SET_KV("hypo/inlier_ratio_after_sampling", inlierRatioAfterSampling); // replace with your actual variable
    MON_RECORD(); // close hypo/prepare/gather_pairs
}


void MacRtHypothesis::generateHypotheses() {
    LOG_INFO("--- 3. Generating initial hypotheses from sampled cliques ---");
    // ---- [MON] hypo/generate/loop ----
    MON_ENTER("hypo/generate/loop");

    // 1. Estimate the transformation matrix by the points in the clique (SVD)
    // 遍历所有采样的团，生成假设
    const int totalEstimateNum = static_cast<int>(sampledCliqueIndices_.size());

    // reserve memory for faster push_back
    hypotheses_.reserve(totalEstimateNum);
    int hypothesisCounter = -1;
    int bestIndividualHypoIndex = -1; // 用于记录最佳假设在 hypotheses_ 列表中的索引
    float bestIndividualScore = -1.0f; // 用于快速比较当前最高分

    PointCloudPtr srcPts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr tgtPts(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> currentCliqueVertexesIndex; // selected vertexes index in the current clique
    std::vector<float> triangularScoresInFilteredClique; // triangular scores in the current clique

    // #pragma omp parallel for
    for (int i = 0; i < totalEstimateNum; ++i) {
        const int clique_idx = sampledCliqueIndices_[i];
        const igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(graph_.getCliques(), clique_idx);
        const int cliqueSize = static_cast<int>(igraph_vector_int_size(v)); // size of the current clique
        std::vector<int> currentCliqueVertexIndices; // selected vertexes index in the current clique
        currentCliqueVertexIndices.reserve(cliqueSize);
        for (int j = 0; j < cliqueSize; j++) {
            int ind = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            // The vertex in each clique
            CorresStruct localCliqueVertexes = data_.corres[ind];
            currentCliqueVertexesIndex.push_back(ind);
            if (localCliqueVertexes.corresScore >= config_.triangularScoreThresh) {
                // This score is the correspondence triangular score, 0 by default
                srcPts->push_back(localCliqueVertexes.src);
                tgtPts->push_back(localCliqueVertexes.tgt);
                // The size of the weights to svd is different with the size of the srcPts and tgtPts.
                // This will be extended in weightSvd function
                triangularScoresInFilteredClique.push_back(localCliqueVertexes.corresScore);
                // The correspondence triangular score.
                // Use this as the weight for the
                currentCliqueVertexIndices.push_back(ind);
            }
        }

        std::stable_sort(currentCliqueVertexesIndex.begin(), currentCliqueVertexesIndex.end());
        // sort before get intersection

        // If the clique is too small, skip it
        if (triangularScoresInFilteredClique.size() < 3) {
            // reset for next iteration
            triangularScoresInFilteredClique.clear();
            continue;
        }

        Eigen::VectorXf scoreVec = Eigen::Map<Eigen::VectorXf>(triangularScoresInFilteredClique.data(),
                                                               triangularScoresInFilteredClique.size());;
        if (!config_.flagInstanceEqual) {
            // Use the triangle score as the SVD weight
            scoreVec = Eigen::Map<Eigen::VectorXf>(triangularScoresInFilteredClique.data(),
                                                   triangularScoresInFilteredClique.size());
            scoreVec /= scoreVec.maxCoeff(); // normalize to [0, 1]
        } else {
            scoreVec.setOnes();
        }
        Eigen::Matrix4f estTransMat;
        weightedSvd(srcPts, tgtPts, scoreVec, config_.triangularScoreThresh, estTransMat);

        // The threshold logic must be handled in config stage
        const float globalScore = oamae(data_.cloudSrcKpts, data_.cloudTgtKpts, estTransMat, data_.tgtSrc,
                                        config_.currentDatasetConfig.inlierEvaThresh);
        const float localScore = transScoreByLocalClique(srcPts, tgtPts, estTransMat,
                                                         config_.currentDatasetConfig.inlierEvaThresh, config_.metric);

        if (globalScore > 0) {
            // #pragma omp critical
            TransformHypothesis currentHypo;
            currentHypo.originalIndex_ = i;
            currentHypo.transform_ = estTransMat;
            currentHypo.globalScore_ = globalScore;
            currentHypo.localScore_ = localScore;
            currentHypo.sourceCorrespondenceIndices_ = std::move(currentCliqueVertexIndices);

            hypothesisCounter++;
            // temporary variables for debug
            float re;
            float te;
            // ---------------------------- Evaluation part ----------------------------
            currentHypo.isGtCorrect_ = evaluationEst(estTransMat, data_.gtTransform, 15, 30, re, te);
            // -------------------------------------------------------------------------
            // betsIndividualHypothesis is generated from each single clique weighted SVD
            if (currentHypo.globalScore_ > bestIndividualScore) {
                bestIndividualHypoIndex = hypothesisCounter;
                bestIndividualScore = currentHypo.globalScore_;
            }
            hypotheses_.push_back(currentHypo);
        }

        // reset for next iteration
        srcPts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        tgtPts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        currentCliqueVertexesIndex.clear();
        triangularScoresInFilteredClique.clear();
    }
    LOG_INFO("Generated " << hypotheses_.size() << " initial hypotheses.");
    // 循环结束后，我们就可以通过 bestIndividualHypoIndex_ 找到那个最佳的假设了
    if (bestIndividualHypoIndex != -1) {
        LOG_INFO("Best individual hypothesis found at index " << bestIndividualHypoIndex
            << " with score " << bestIndividualScore);
        bestIndividualHypothesis_ = hypotheses_[bestIndividualHypoIndex];
    } else {
        LOG_WARNING("No valid individual hypothesis was found.");
    }

    MON_SET_KV("hypo/hypotheses_generated",
               static_cast<int>(hypotheses_.size())); // replace with vector name if different
    // Optional (if available):
    MON_SET_KV("hypo/best_individual_score", bestIndividualScore);
    MON_RECORD(); // close hypo/generate/loop
}

/**
 * @brief [私有] 步骤4：对已生成的假设进行排序和筛选 (Top-K)
 */
void MacRtHypothesis::sortAndFilterHypotheses() {
    LOG_INFO("--- 4. Sorting and filtering hypotheses ---");
    // ---- [MON] hypo/sort_filter/sort ----
    MON_ENTER("hypo/sort_filter/sort");

    if (hypotheses_.empty()) {
        LOG_WARNING("Hypotheses list is empty, nothing to sort or filter.");
        return;
    }

    // ==============================================================================
    // ==         核心步骤 1: 直接对 hypotheses_ 列表进行排序         ==
    // ==============================================================================
    // 我们不再需要复杂的间接排序，因为每个 TransformHypothesis 对象都包含了自己的分数
    std::stable_sort(hypotheses_.begin(), hypotheses_.end(),
                     [](const TransformHypothesis &a, const TransformHypothesis &b) {
                         return a.globalScore_ > b.globalScore_;
                     });
    // From now on, hypotheses_ is sorted by globalScore_ in descending order
    MON_RECORD(); // close hypo/sort_filter/sort

    // ==============================================================================
    // ==         核心步骤 2: 筛选出 Top-K 个结果                  ==
    // ==============================================================================
    // ---- [MON] hypo/sort_filter/filter ----
    MON_ENTER("hypo/sort_filter/filter");
    const int totalEstimateNum = static_cast<int>(hypotheses_.size());
    // Force reduce the number of hypotheses!!!!!!!!!!!!!
    const int selectedNum = std::min(data_.totalCorresNum, std::min(totalEstimateNum, config_.maxEstimateNum));
    // k is selectedNum
    if (totalEstimateNum > selectedNum) {
        LOG_INFO("Too many hypotheses (" << totalEstimateNum << "), choosing top "
            << selectedNum << " candidates.");
        // 直接使用 resize，高效地截断 vector，只保留前 selectedNum 个元素
        hypotheses_.resize(selectedNum);
    }

    // ---------------------------- Evaluation part ----------------------------
    // ==============================================================================
    // ==         核心步骤 3: 统计评估结果 (成功数量)              ==
    // ==============================================================================
    // 遍历最终筛选出的假设列表，统计评估正确的数量
    this->numCorrectHypotheses_ = 0;
    for (const auto &hypo: hypotheses_) {
        if (hypo.isGtCorrect_) {
            this->numCorrectHypotheses_++;
        }
    }
    if (numCorrectHypotheses_ > 0) {
        if (!config_.flagNoLogs) {
            const std::string estInfo = config_.outputPath + "/est_info.txt";
            std::ofstream estInfoFile(estInfo, std::ios::trunc);
            estInfoFile.setf(std::ios::fixed, std::ios::floatfield);
            for (const auto &hypo: hypotheses_) {
                estInfoFile << std::setprecision(10) << hypo.globalScore_ << " " << hypo.isGtCorrect_ << std::endl;
            }
            estInfoFile.close();
        }
    } else {
        LOG_CRITICAL("NO CORRECT ESTIMATION!!!");
    }
    result_.correctEstNum = this->numCorrectHypotheses_;
    // -------------------------------------------------------------------------

    LOG_INFO("Sorting and filtering complete. " << hypotheses_.size()
        << " hypotheses remain. Correct estimations: " << numCorrectHypotheses_);
    MON_SET_KV("hypo/selected_topk",
               static_cast<int>(hypotheses_.size())); // after truncation
    // Optional (if maintained in this function):
    MON_SET_KV("hypo/correct_estimations", numCorrectHypotheses_);
    MON_RECORD(); // close hypo/sort_filter/filter
}

// ################################################################
// This two variable should be moved to MacConfig later
float g_angleThreshold = 5.0 * M_PI / 180; //5 degree
float g_distanceThreshold = 0.1;


bool MacRtHypothesis::EnforceSimilarity1(const pcl::PointXYZINormal &pointA, const pcl::PointXYZINormal &pointB,
                                         float squared_distance) {
    if (std::isnan(pointA.normal_x) || std::isnan(pointB.normal_x)) {
        return false;
    }
    const Eigen::Vector3f transA(pointA.normal_x, pointA.normal_y, pointA.normal_z);
    const Eigen::Vector3f transB(pointB.normal_x, pointB.normal_y, pointB.normal_z);
    if ((transA - transB).norm() < g_distanceThreshold) {
        return true;
    }
    return false;
}

int MacRtHypothesis::clusterTransformationByRotation(
    const std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > &Rs,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &Ts,
    const float angleThresh, const float disThresh, pcl::IndicesClusters &clusters,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans) {
    if (Rs.empty() || Ts.empty() || Rs.size() != Ts.size()) {
        LOG_ERROR("Rs and Ts are empty or not the same size!");
        return -1;
    }
    const int num = Rs.size();
    g_distanceThreshold = disThresh;
    trans->resize(num);
    for (size_t i = 0; i < num; i++) {
        auto &point = (*trans)[i];
        Eigen::Transform<float, 3, Eigen::Affine> R(Rs[i]);
        pcl::getEulerAngles<float>(R, point.x, point.y, point.z); // R -> trans
        // 去除无效解
        if (!checkEulerAngles(point.x) || !checkEulerAngles(point.y) || !checkEulerAngles(point.z)) {
            LOG_WARNING("INVALID POINT: " << i << " th point, set to NaN");
            point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
            point.normal_x = point.normal_y = point.normal_z = std::numeric_limits<float>::quiet_NaN();
        } else {
            // 需要解决同一个角度的正负问题 6.14   平面 y=PI 右侧的解（需要验证） 6.20
            // -pi - pi -> 0 - 2pi
            // 将欧拉角范围从 [-PI, PI] 归一化到 [0, 2*PI]
            if (point.x < 0) point.x += 2 * M_PIf32;
            if (point.y < 0) point.y += 2 * M_PIf32;
            if (point.z < 0) point.z += 2 * M_PIf32;
            point.normal_x = static_cast<float>(Ts[i][0]);
            point.normal_y = static_cast<float>(Ts[i][1]);
            point.normal_z = static_cast<float>(Ts[i][2]);
        }
    }

    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true); // true for using dense mode, no NaN points
    cec.setInputCloud(trans);
    cec.setConditionFunction(&EnforceSimilarity1);
    cec.setClusterTolerance(angleThresh);
    cec.setMinClusterSize(2); // cluster size
    cec.setMaxClusterSize(static_cast<int>(num)); // nearly impossible to reach the maximum?
    cec.segment(clusters);

    // 为每个聚类的点设置强度值（聚类编号）
    for (int i = 0; i < clusters.size(); ++i) {
        for (int j = 0; j < clusters[i].indices.size(); ++j) {
            // Set intensity of each cluster point to their cluster number
            trans->points[clusters[i].indices[j]].intensity = i;
        }
    }
    return 0;
}

void MacRtHypothesis::clusterHypotheses() {
    LOG_INFO("--- 5. Clustering hypotheses ---");
    // ---- [MON] hypo/cluster/build ----
    MON_ENTER("hypo/cluster/build");

    // 从 hypotheses_ 成员中提取出旋转(R)和平移(T)矩阵列表
    // 这是聚类函数需要的输入
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Rs;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Ts;
    Rs.reserve(hypotheses_.size());
    Ts.reserve(hypotheses_.size());
    for (const auto &hypo: hypotheses_) {
        Rs.push_back(hypo.transform_.topLeftCorner<3, 3>());
        Ts.push_back(hypo.transform_.block<3, 1>(0, 3));
    }

    // 使用配置化的聚类参数
    const auto &datasetConf = config_.getCurrentDatasetConfig();
    const float angleThresh = datasetConf.clusterAngThresh * M_PI / 180.0f;
    // Modify the logic below

    // const float disThresh = datasetConf.getActualClusteringDistanceThreshold(data_.cloudResolution, config_.isU3MDataset());
    const float disThresh = datasetConf.clusterDistThresh;

    LOG_INFO("Angle threshold: " << angleThresh << ", Distance threshold: " << disThresh);

    // 调用聚类函数，并将结果存储到成员变量中
    clusterTransformationByRotation(Rs, Ts, angleThresh, disThresh, this->clusters_, this->transPoints_);

    LOG_INFO("Found " << clusters_.size() << " clusters from " << hypotheses_.size() << " hypotheses.");
    // ---- [MON] hypo/cluster/metrics: compute cluster stats ----
    // Compute largest cluster size and its index.
    int largestClusterSize = 0;
    int largestClusterIndex = -1;
    /**
     * PCL's IndicesClusters = std::vector<pcl::PointIndices>.
     * Each cluster has 'indices' storing member point indices.
     */
    for (int ci = 0; ci < static_cast<int>(clusters_.size()); ++ci) {
        const int sz = static_cast<int>(clusters_[ci].indices.size());
        if (sz > largestClusterSize) {
            largestClusterSize = sz;
            largestClusterIndex = ci;
        }
    }
    MON_SET_KV("hypo/num_clusters", static_cast<int>(clusters_.size()));
    MON_SET_KV("hypo/largest_cluster_size", largestClusterSize);
    MON_SET_KV("hypo/largest_cluster_index", largestClusterIndex);
    MON_RECORD(); // close hypo/cluster/build
}

// Fallback function when no clusters are found
// Temporary not used
bool MacRtHypothesis::handleClusteringFailure() {
    LOG_WARNING("No clusters found, using the single best individual hypothesis as fallback.");

    // 这个方法直接使用已经计算好的 bestIndividualHypothesis_
    // 并对其进行精炼，然后设置最终结果。
    // 这部分逻辑暂时可以先直接迁移，后续可以再与 performFinalSelectionAndRefinement 复用

    this->winningTransform_ = bestIndividualHypothesis_.transform_;

    // ---------------------------- Evaluation part ----------------------------
    // if (config_.datasetName == "U3M") {
    //     result_.RE = rmseCompute(cloudSrc, cloudTgt, bestIndividualHypothesis_, gtMat, cloudResolution);
    //     result_.TE = 0;
    // } else {
    if (!flagFound_) {
        flagFound_ = evaluationEst(bestIndividualHypothesis_.transform_, data_.gtTransform,
                                   config_.getCurrentDatasetConfig().reThresh,
                                   config_.getCurrentDatasetConfig().teThresh, result_.RE,
                                   result_.TE);
    }
    finalBestTransform_ = bestIndividualHypothesis_.transform_;
    float bestGlobalScore = 0;
    postRefinement(sampledCorr_, sampledCorrSrc_, sampledCorrTgt_, bestIndividualHypothesis_.transform_,
                   bestGlobalScore,
                   config_.getCurrentDatasetConfig().inlierEvaThresh, 20,
                   config_.metric);
    // }
    // if (config_.datasetName == "U3M") {
    //     if (result_.RE <= 5) {
    //         std::cout << result_.RE << std::endl;
    //         std::cout << bestIndividualHypothesis_ << std::endl;
    //         return true;
    //     }
    //     return false;
    // }
    float rmse = rmseCompute(data_.cloudSrc, data_.cloudTgt, finalBestTransform_, data_.gtTransform,
                             config_.getCurrentDatasetConfig().effectiveResolution());
    std::cout << "RMSE: " << rmse << std::endl;
    if (flagFound_) {
        float newRe, newTe;
        evaluationEst(bestIndividualHypothesis_.transform_, data_.gtTransform,
                      config_.getCurrentDatasetConfig().reThresh, config_.getCurrentDatasetConfig().teThresh, newRe,
                      newTe);

        if (newRe < result_.RE && newTe < result_.TE) {
            std::cout << "est_trans updated!!!" << std::endl;
            std::cout << "RE=" << newRe << " " << "TE=" << newTe << std::endl;
            std::cout << bestIndividualHypothesis_.transform_ << std::endl;
        } else {
            std::cout << "RE=" << result_.RE << " " << "TE=" << result_.TE << std::endl;
            std::cout << finalBestTransform_ << std::endl;
        }
        result_.RE = newRe;
        result_.TE = newTe;
        //                if(rmse > 0.2) return false;
        //                else return true;
        return true;
    }
    float newRe, newTe;
    flagFound_ = evaluationEst(finalBestTransform_, data_.gtTransform,
                               config_.getCurrentDatasetConfig().reThresh, config_.getCurrentDatasetConfig().teThresh,
                               newRe, newTe);
    if (flagFound_) {
        result_.RE = newRe;
        result_.TE = newTe;
        std::cout << "est_trans corrected!!!" << std::endl;
        std::cout << "RE=" << result_.RE << " " << "TE=" << result_.TE << std::endl;
        std::cout << finalBestTransform_ << std::endl;
        return true;
    }
    std::cout << "RE=" << result_.RE << " " << "TE=" << result_.TE << std::endl;
    return false;
    // -------------------------------------------------------------------------
    //                if(rmse > 0.2) return false;
    //                else return true;

}

/**
 * @brief [私有] 从聚类中选择与最佳单个假设最相似的成员
 */
void MacRtHypothesis::selectBestConsensus() {
    LOG_INFO("--- 6. Selecting best consensus from clusters ---");

    // 1. 对聚类按大小排序 (原 clusterSorted 逻辑)
    // Attention: sorted cluster is not used. Later we will consider filter the
    // clusters by size if they are too small.
    sortedClusters_.clear();
    sortedClusters_.reserve(clusters_.size());
    int goodClusterCount = 0;
    for (size_t i = 0; i < clusters_.size(); ++i) {
        const int clusterSize = static_cast<int>(clusters_[i].indices.size());
        // Initialize flagGtCorrect to false
        sortedClusters_.push_back({static_cast<int>(i), clusterSize, false});
        // clusterSize > 1
        if (clusterSize != 1) {
            goodClusterCount++;
        }
    }
    if (goodClusterCount == 0) {
        LOG_CRITICAL("No good clusters found. The result is probably unreliable.");
    }
    // For the trace back of the clusters are not implemented.
    // And sort does not increase performance
    // Do not sort the clusters for now!!!
    // std::stable_sort(sortedClusters_.begin(), sortedClusters_.end(), compareClusterScore);

    // 2. 寻找与 bestIndividualHypothesis_ 最相似的聚类内成员

    const Eigen::Matrix4f bestIndividualInv = bestIndividualHypothesis_.transform_.inverse();

    // #pragma omp parallel for schedule(static) default(none) shared(sortedClusters_, hypotheses_, bestIndividualInv, bestSimilarity_, bestSimClusterOriginalIndex_, bestSimHypoInClusterIndex_)
    for (const auto &clusterInfo: sortedClusters_) {
        const int clusterIdx = clusterInfo.clusterIndex;
        for (size_t i = 0; i < clusters_[clusterIdx].indices.size(); ++i) {
            const int hypoIdx = clusters_[clusterIdx].indices[i];
            const auto &hypo = hypotheses_[hypoIdx];
            const float similarity = (bestIndividualInv * hypo.transform_ - Eigen::Matrix4f::Identity()).norm();
            // #pragma omp critical
            if (similarity < bestSimilarity_) {
                bestSimilarity_ = similarity;
                bestSimClusterOriginalIndex_ = clusterIdx;
                bestSimHypoInClusterIndex_ = i;
            }
        }
    }

    // similarity is smaller the better
    if (bestSimClusterOriginalIndex_ != -1) {
        LOG_INFO("Best individual hypothesis is most similar to the transformation " << bestSimHypoInClusterIndex_
            << " in cluster " << bestSimClusterOriginalIndex_ << " (similarity score: " << bestSimilarity_ << ")");
    }
}

/**
 * @brief [私有] 从聚类中寻找最佳共识假设
 */
void MacRtHypothesis::findConsensusHypotheses() {
    LOG_INFO("--- 7. Finding best consensus hypothesis from clusters ---");
    // ---- [MON] hypo/consensus/select ----
    MON_ENTER("hypo/consensus/select");

    // 临时变量，用于存储每个聚类的代表(中心)假设，以及所有聚类包含的对应关系索引
    std::vector<TransformHypothesis, Eigen::aligned_allocator<TransformHypothesis> > clusterCenterHypotheses;


    // Reserve space to avoid multiple allocations
    clusterCenterHypotheses.reserve(sortedClusters_.size());
    subClusterCorrIndices_.reserve(sortedClusters_.size());
    // temporary setup variable clusterIndices for trace back cluster index
    // There are two ways to improve, either use the cluster_ itself for iteration
    // either change the struct hypotheses to include the cluster index
    std::vector<int> clusterIndices;
    clusterIndices.reserve(sortedClusters_.size());

    // --- 步骤 1: 遍历所有聚类，在一个循环内完成“找中心”和“收集索引” ---
    // 这个“收集”阶段的循环是线性的，不适合简单地用 omp for 并行化，
    // 因为并发地向 vector push_back 会导致竞争条件，加锁的开销可能比并行收益还大。
    for (const auto &clusterInfo: sortedClusters_) {
        const auto &clusterHypoIndices = clusters_[clusterInfo.clusterIndex].indices;
        // Redundant check, cluster must have at least one member
        if (clusterHypoIndices.empty()) {
            continue;
        }

        // a) 寻找该聚类中 localScore_ 最高的假设作为“中心”
        int centerHypoIndex = clusterHypoIndices[0];
        float maxLocalScore = hypotheses_[centerHypoIndex].localScore_;
        for (size_t i = 1; i < clusterHypoIndices.size(); ++i) {
            if (const int currentHypoIndex = clusterHypoIndices[i];
                hypotheses_[currentHypoIndex].localScore_ > maxLocalScore) {
                maxLocalScore = hypotheses_[currentHypoIndex].localScore_;
                centerHypoIndex = currentHypoIndex;
            }
        }
        clusterCenterHypotheses.push_back(hypotheses_[centerHypoIndex]);
        clusterIndices.push_back(clusterInfo.clusterIndex); // 记录该中心假设所属的聚类索引

        // b) 在同一个循环中，计算该聚类所有成员的索引并集
        std::vector<int> currentClusterUnionIndices;
        // 这个版本的排序在交并函数内实现
        for (const int hypoIdx: clusterHypoIndices) {
            currentClusterUnionIndices = vectorsUnion(currentClusterUnionIndices,
                                                      hypotheses_[hypoIdx].sourceCorrespondenceIndices_);
        }
        subClusterCorrIndices_.push_back(currentClusterUnionIndices);
    }

    if (clusterCenterHypotheses.empty()) {
        LOG_WARNING("Could not determine any cluster centers. Using best individual as fallback.");
        bestConsensusHypothesis_ = bestIndividualHypothesis_;
        return;
    }

    // --- 步骤 2: 准备用于 OAMAE 精确评估的“精英子集”，这部分是线性的 ---
    std::vector<int> globalUnionCorrIndices;
    for (const auto &subUnion: subClusterCorrIndices_) {
        globalUnionCorrIndices = vectorsUnion(globalUnionCorrIndices, subUnion);
    }

    std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct> > globalUnionCorr;
    globalUnionCorr.reserve(globalUnionCorrIndices.size());
    for (const int index: globalUnionCorrIndices) {
        globalUnionCorr.push_back(data_.corres[index]);
    }

    std::vector<std::pair<int, std::vector<int> > > tgtSrcFromClusters;
    makeTgtSrcPair(globalUnionCorr, tgtSrcFromClusters);

    // --- 步骤 3: 并行地对所有“中心”假设，使用“精英子集”进行采样后的 OAMAE 评分 ---

    // 这个循环计算量大，且每次迭代独立，非常适合并行化。
    float bestConsensusScore = -1.0f;
    // bestConsensusHypothesis_ 在 clusterCenterHypotheses 中的索引
#pragma omp parallel for
    for (int clusterCenterIdx = 0; clusterCenterIdx < static_cast<int>(clusterCenterHypotheses.size()); ++
         clusterCenterIdx) {
        const auto &datasetConf = config_.getCurrentDatasetConfig();
        // 完善数据集阈值逻辑
        const float inlierEvaThresh = datasetConf.getActualInlierThreshold();

        const float clusterEvaScore = oamae(data_.cloudSrcKpts, data_.cloudTgtKpts,
                                            clusterCenterHypotheses[clusterCenterIdx].transform_,
                                            tgtSrcFromClusters, inlierEvaThresh);

        // 使用临界区保护对共享变量 bestConsensusScore 和 bestConsensusIndex 的写入
#pragma omp critical
        {
            if (clusterEvaScore > bestConsensusScore) {
                bestConsensusScore = clusterEvaScore;
                bestConsensusIndex_ = clusterCenterIdx;
            }
        }
    }

    // --- 步骤 4: 将最终找到的最佳共识假设存储到成员变量中 ---
    if (bestConsensusIndex_ != -1) {
        bestConsensusHypothesis_ = clusterCenterHypotheses[bestConsensusIndex_];
        bestConsensusHypothesis_.globalScore_ = bestConsensusScore;
        LOG_INFO(
            "Best consensus hypothesis found with refined OAMAE (cluster sampled pairs) score " << bestConsensusScore
            << " from cluster " << clusterIndices[bestConsensusIndex_] << " with hypothesis index " <<
            bestConsensusHypothesis_.originalIndex_);
    } else {
        LOG_WARNING(
            "Could not determine a best consensus hypothesis via refined OAMAE. Using best individual as fallback.");
        bestConsensusHypothesis_ = bestIndividualHypothesis_;
    }
    MON_SET_KV("hypo/best_consensus_score", bestConsensusScore);
    MON_SET_KV("hypo/best_consensus_idx", bestConsensusIndex_);
    MON_RECORD(); // close hypo/consensus/select
}

/**
 * @brief [私有] 在“最佳独立解”和“最佳共识解”之间进行最终选择
 */
void MacRtHypothesis::performFinalSelection() {
    LOG_INFO("--- 8. Performing final selection between individual and consensus ---");
    // ---- [MON] hypo/final_selection ----
    MON_ENTER("hypo/final_selection");

    // 使用 TCD (Truncated Chamfer Distance) 作为最终的评判标准
    // 准备用于 TCD 评估的点云 patch
    const auto &datasetConf = config_.getCurrentDatasetConfig();
    const float inlierEvaThresh = datasetConf.getActualInlierThreshold();

    PointCloudPtr patchSrc(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudPtr patchTgt(new pcl::PointCloud<pcl::PointXYZ>());

    // 根据采样后的对应关系，提取出源点云和目标点云的关键点子点云
    getCorrPatch(sampledCorr_, data_.cloudSrcKpts, data_.cloudTgtKpts, patchSrc, patchTgt, 2 * inlierEvaThresh);

    // 计算两个候选者的分数
    individualScore_ = truncatedChamferDistance(patchSrc, patchTgt, bestIndividualHypothesis_.transform_,
                                                inlierEvaThresh);
    consensusScore_ = truncatedChamferDistance(patchSrc, patchTgt, bestConsensusHypothesis_.transform_,
                                               inlierEvaThresh);

    LOG_INFO("Finalist TCD Scores -> Individual: " << individualScore_ << ", Consensus: " << consensusScore_);

    // 根据分数选出胜出者，并将其变换存入 winningTransform_ 成员，准备进行精炼
    if (individualScore_ > consensusScore_) {
        LOG_INFO("Winner: Best Individual Hypothesis.");
        winningTransform_ = bestIndividualHypothesis_.transform_;
    } else {
        LOG_INFO("Winner: Best Consensus Hypothesis.");
        winningTransform_ = bestConsensusHypothesis_.transform_;
    }
    MON_SET_KV("hypo/score_individual", individualScore_);
    MON_SET_KV("hypo/score_consensus", consensusScore_);
    MON_SET_KV("hypo/winner", std::string(individualScore_ > consensusScore_ ? "individual" : "consensus"));
    MON_RECORD(); // close hypo/final_selection
}

/**
 * @brief [私有] 对胜出的变换进行最终精炼
 */
void MacRtHypothesis::refineBestTransform() {
    LOG_INFO("--- 9. Refining the winning transform ---");

    // 将胜出的变换作为精炼的起点
    finalBestTransform_ = winningTransform_;

    // 1. cluster_internal_evaluation 逻辑 (如果启用)
    // 注意：这是一个非常复杂的过程，它本身也应该被拆解成更小的辅助函数
    // 这里为了忠实于原始代码，暂时将其放在一起
    const PointCloudPtr clusterEvaCorrSrc(new pcl::PointCloud<pcl::PointXYZ>);
    const PointCloudPtr clusterEvaCorrTgt(new pcl::PointCloud<pcl::PointXYZ>);
    // 临时解决方案
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Rs;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Ts;
    Rs.reserve(hypotheses_.size());
    Ts.reserve(hypotheses_.size());
    for (const auto &hypo: hypotheses_) {
        Rs.emplace_back(hypo.transform_.topLeftCorner<3, 3>());
        Ts.emplace_back(hypo.transform_.block<3, 1>(0, 3));
    }
    // cluster_internal_evaluation
    int bestClusterIndex = -1;
    if (config_.flagClusterInternalEvaluation) {
        int inlierCounter = 0;
        std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct> > clusterEvaCorr;
        // if best similarity is small enough, indicating bestIndividualHypothesis_ is in the cluster
        if (bestSimilarity_ < 0.1) {
            LOG_INFO("bestEstIndividual is in a cluster, bestSimilarity: " << bestSimilarity_);
            // The individual is better than the cluster center estimate
            if (individualScore_ > consensusScore_) {
                bestClusterIndex = bestSimClusterOriginalIndex_;
                finalBestTransform_ = bestIndividualHypothesis_.transform_;
                LOG_INFO("bestEstIndividual (global individual) is better");
            } else {
                bestClusterIndex = bestConsensusIndex_;
                finalBestTransform_ = bestConsensusHypothesis_.transform_;
                LOG_INFO("bestEstConsensus (cluster center estimate) is better");
            }
            // Get the intersection of the sampled correspondences and the correspondences in the best cluster
            std::vector<int> finalSelectedCorresIndices;
            std::vector<int> bestClusterIndices = subClusterCorrIndices_[bestSimClusterOriginalIndex_];
            std::sort(bestClusterIndices.begin(), bestClusterIndices.end());
            std::sort(sampledCorresIndices_.begin(), sampledCorresIndices_.end());

            finalSelectedCorresIndices.assign(bestClusterIndices.begin(), bestClusterIndices.end());
            finalSelectedCorresIndices = vectorsIntersection(finalSelectedCorresIndices, sampledCorresIndices_);
            // for (auto &ind: finalSelectedCorresIndices) {
            //     std::cout << ind << " ";
            // }
            // std::cout << std::endl << std::endl << std::endl ;
            // for (auto &ind: sampledCorresIndices_) {
            //     std::cout << ind << " ";
            // }
            // std::cout << std::endl << std::endl << std::endl ;
            // for (auto &ind: bestClusterIndices) {
            //     std::cout << ind << " ";
            // }
            // std::cout << std::endl << std::endl << std::endl ;
            // std::cout << bestSimClusterOriginalIndex_ << std::endl;
            // ---------------------------- Evaluation part ----------------------------
            if (finalSelectedCorresIndices.empty()) {
                LOG_ERROR("No intersection correspondences found in the best cluster. Aborting refinement.");
                return;
            }
            for (const auto &ind: finalSelectedCorresIndices) {
                clusterEvaCorr.push_back(data_.corres[ind]);
                clusterEvaCorrSrc->push_back(data_.corres[ind].src);
                clusterEvaCorrTgt->push_back(data_.corres[ind].tgt);
                if (data_.gtInlierLabels[ind]) {
                    inlierCounter++;
                }
            }
            std::cout << finalSelectedCorresIndices.size() << " intersection correspondences have " << inlierCounter <<
                    " inliers: " << inlierCounter / (static_cast<int>(finalSelectedCorresIndices.size()) / 1.0) * 100 <<
                    "%" <<
                    std::endl;
            // -------------------------------------------------------------------------
            std::vector<std::pair<int, std::vector<int> > > tgtSrc3;
            makeTgtSrcPair(clusterEvaCorr, tgtSrc3);

            finalBestTransform_ = clusterInternalTransEva1(clusters_, bestClusterIndex, finalBestTransform_, Rs, Ts,
                                                           data_.cloudSrcKpts, data_.cloudTgtKpts,
                                                           tgtSrc3, config_.threshold, data_.gtTransform, false,
                                                           config_.outputPath);
        } else {
            // bestEstIndividual is not in a cluster
            LOG_INFO("bestEstIndividual is not in a cluster, bestSimilarity: " << bestSimilarity_);
            if (consensusScore_ > individualScore_) {
                bestClusterIndex = bestConsensusIndex_;
                // bestEstConsensusScore is better, then use this for final evaluation
                finalBestTransform_ = bestIndividualHypothesis_.transform_;
                LOG_INFO("bestEstConsensus is better");
                std::vector<int> finalSelectedCorresIndices;
                std::vector<int> bestClusterIndices = subClusterCorrIndices_[bestSimClusterOriginalIndex_];
                finalSelectedCorresIndices.assign(bestClusterIndices.begin(), bestClusterIndices.end());
                finalSelectedCorresIndices = vectorsIntersection(finalSelectedCorresIndices, sampledCorresIndices_);
                if (finalSelectedCorresIndices.empty()) {
                    return;
                }
                inlierCounter = 0;

                for (const auto &ind: finalSelectedCorresIndices) {
                    clusterEvaCorr.push_back(data_.corres[ind]);
                    clusterEvaCorrSrc->push_back(data_.corres[ind].src);
                    clusterEvaCorrTgt->push_back(data_.corres[ind].tgt);
                    if (data_.gtInlierLabels[ind]) {
                        inlierCounter++;
                    }
                }
                std::cout << finalSelectedCorresIndices.size() << " intersection correspondences have " << inlierCounter
                        << " inliers: " << inlierCounter / (static_cast<int>(finalSelectedCorresIndices.size()) / 1.0) *
                        100 <<
                        "%" << std::endl;
                std::vector<std::pair<int, std::vector<int> > > tgtSrc3;
                makeTgtSrcPair(clusterEvaCorr, tgtSrc3);
                finalBestTransform_ = clusterInternalTransEva1(clusters_, bestClusterIndex, finalBestTransform_, Rs, Ts,
                                                               data_.cloudSrcKpts, data_.cloudTgtKpts,
                                                               tgtSrc3, config_.threshold, data_.gtTransform, false,
                                                               config_.outputPath);
            } else {
                finalBestTransform_ = bestIndividualHypothesis_.transform_;
                LOG_INFO("bestEstIndividual is better but not in cluster! Refine it");
            }
        }
    }

    // 2. 最终的 postRefinement
    LOG_INFO("Performing final post-refinement...");
    // 假设 postRefinement 已经迁移为私有方法
    // postRefinement(finalBestTransform_, sampledCorr_);


    LOG_INFO("Refinement complete.");
}


void MacRtHypothesis::processGraphResultAndFindBest(const MacGraph &graph) {
    LOG_INFO("================ Starting Hypothesis Stage ================");
    Timer timerProcessClique;
    timerProcessClique.startTiming("sample candidates");
    // 步骤 1: 候选采样
    MON_ENTER("hypo/sample");
    sampleCandidates(graph);
    if (sampledCliqueIndices_.empty()) {
        LOG_ERROR("Clique sampling resulted in zero candidates. Aborting.");
        winningTransform_.setIdentity(); // 返回一个默认值
        return;
    }
    timerProcessClique.endTiming();
    MON_RECORD();

    // ---- [MON] hypo/prepare ----
    MON_ENTER("hypo/prepare");
    timerProcessClique.startTiming("prepare sampled data");
    // 步骤 2: 准备采样数据
    prepareSampledData();
    timerProcessClique.endTiming();
    MON_RECORD();

    LOG_INFO("Number of sampled correspondences: " << sampledCorresIndices_.size());
    LOG_INFO("Number of sampled cliques: " << sampledCliqueIndices_.size());

    MON_ENTER("hypo/generate");
    timerProcessClique.startTiming("generate initial hypotheses");
    // 3. 生成假设
    generateHypotheses();
    timerProcessClique.endTiming();
    MON_RECORD();

    MON_ENTER("hypo/process_clique");
    timerProcessClique.startTiming("sort and filter hypotheses");
    // 4. 排序与过滤假设
    sortAndFilterHypotheses();
    timerProcessClique.endTiming();
    MON_RECORD();

    // Later move the store logic out of the function sortAndFilterHypotheses
    // // --- 日志记录和最终结果赋值 (职责属于 Aligner) ---
    // const auto& finalHypotheses = getFinalHypotheses();
    // const int successNum = getNumCorrectHypotheses();
    //
    // // 对应你原始代码中的文件写入部分
    // if (successNum > 0) {
    //     if (!config_.flagNoLogs) {
    //         std::string estInfo = config_.outputPath + "/est_info.txt";
    //         std::ofstream estInfoFile(estInfo);
    //         estInfoFile.setf(std::ios::fixed, std::ios::floatfield);
    //
    //         // 使用 hypoManager 提供的数据写入日志
    //         for (const auto& hypo : finalHypotheses) {
    //             estInfoFile << std::setprecision(10) << hypo.globalScore_ << " " << hypo.isGtCorrect_ << std::endl;
    //         }
    //         estInfoFile.close();
    //     }
    // } else {
    //     LOG_WARNING("NO CORRECT ESTIMATION!!!");
    // }
    // // 对应你原始代码中的结果赋值部分
    // result_.correctEstNum = successNum; // macResult 是 run() 方法的参数

    // Cluster

    // --- 聚类与共识阶段 ---
    MON_ENTER("hypo/cluster_and_consensus");
    timerProcessClique.startTiming("cluster");
    // 5.
    clusterHypotheses();
    timerProcessClique.endTiming();
    MON_RECORD();

    if (clusters_.empty()) {
        MON_ENTER("Basic MAC");
        handleClusteringFailure();
        MON_RECORD();
        return; // 备用逻辑已处理完毕，直接返回
    }

    // 如果聚类成功，则继续
    // 6. 寻找最佳共识假设
    MON_ENTER("hypo/cluster");
    timerProcessClique.startTiming("find consensus");
    selectBestConsensus();
    timerProcessClique.endTiming();
    MON_RECORD();


    // 7. Find the best consensus hypothesis
    MON_ENTER("hypo/findConsensus");
    timerProcessClique.startTiming("find Consensus Hypotheses");
    findConsensusHypotheses();
    timerProcessClique.endTiming();
    MON_RECORD();

    if (bestIndividualHypothesis_.originalIndex_ == bestConsensusHypothesis_.originalIndex_) {
        LOG_INFO(
            "Best individual and best consensus hypotheses are the same: " << bestIndividualHypothesis_.originalIndex_
            << ".Skipping final selection."); // skip implementation is not done yet
    } else {
        LOG_INFO("Best individual hypothesis index: " << bestIndividualHypothesis_.originalIndex_
            << ", Best consensus hypothesis index: " << bestConsensusHypothesis_.originalIndex_);
    }

    // 8. 在“最佳独立解”和“最佳共识解”之间进行最终选择
    MON_ENTER("hypo/final_selection");
    timerProcessClique.startTiming("final selection");
    performFinalSelection();
    timerProcessClique.endTiming();
    MON_RECORD();


    // 9. 最终精炼
    MON_ENTER("hypo/refineBestTransform");
    timerProcessClique.startTiming("refineBestTransform");
    refineBestTransform();
    timerProcessClique.endTiming();
    MON_RECORD();

    // ==============================================================================
    // ==         在这里添加您的打印代码，这是最佳位置         ==
    // ==============================================================================
    LOG_INFO("================ Final Registration Result ================");
    std::cout << "Final Estimated Transformation Matrix:\n" << winningTransform_ << std::endl;
    // ==============================================================================
}

// TODO: This function is not optimized
// TODO: We only get the logic check
void MacRtHypothesis::weightedSvd(const PointCloudPtr &srcPts, const PointCloudPtr &tgtPts, Eigen::VectorXf &weights,
                                  float weightThreshold, Eigen::Matrix4f &transMat) {
    for (int i = 0; i < weights.size(); i++) {
        weights(i) = (weights(i) < weightThreshold) ? 0 : weights(i);
    }
    //weights升维度
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> weight;
    Eigen::VectorXf ones = weights;
    ones.setOnes();
    weight = weights * ones.transpose(); // 扩展为矩阵
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;
    //构建对角阵
    Identity.setIdentity();
    weight = weight.cwiseProduct(Identity);
    pcl::ConstCloudIterator src_it(*srcPts);
    pcl::ConstCloudIterator des_it(*tgtPts);
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


float MacRtHypothesis::transScoreByLocalClique(const PointCloudPtr &srcCorrPts, const PointCloudPtr &tgtCorrPts,
                                               const Eigen::Matrix4f &trans,
                                               const float metricThresh, const std::string &metric) {
    const PointCloudPtr srcTrans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*srcCorrPts, *srcTrans, trans);
    srcTrans->is_dense = false;
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*srcTrans, *srcTrans, mapping);
    if (srcTrans->empty()) return 0;
    float score = 0.0;
    const int corr_num = srcCorrPts->points.size();
    for (int i = 0; i < corr_num; i++) {
        if (const float dist = getDistance(srcTrans->points[i], tgtCorrPts->points[i]); dist < metricThresh) {
            constexpr float w = 1;
            if (metric == "inlier") {
                score += 1 * w; //correspondence[i].inlier_weight; <- commented by the MAC++ author
            } else if (metric == "MAE") {
                score += (metricThresh - dist) * w / metricThresh;
            } else if (metric == "MSE") {
                score += pow((metricThresh - dist), 2) * w / pow(metricThresh, 2);
            }
        }
    }
    return score;
}
