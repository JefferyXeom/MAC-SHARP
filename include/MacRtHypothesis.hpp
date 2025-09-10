//
// Created by Jeffery_Xeom on 2025/8/25.
// Project: MAC_SHARP
// File: MacRtHypothesis.hpp
//

#pragma once


#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "CommonTypes.hpp"
#include "MacConfig.hpp"
#include "MacData.hpp"
#include "MacGraph.hpp"

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif

class MacRtHypothesis {
// private:
    // --- 成员变量 ---
    MacData& data_;
    const MacConfig& config_;
    const MacGraph& graph_;
    MacResult& result_;

    // --- 流程中的中间状态 ---
    // --- 采样后的数据，作为后续步骤共享的资源 ---
    std::vector<int> sampledCorresIndices_;
    std::vector<int> sampledCliqueIndices_;
    std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> sampledCorr_;
    PointCloudPtr sampledCorrSrc_;
    PointCloudPtr sampledCorrTgt_;

    // Sorted by clique score, decremental
    // Alraedy sampled and filtered before clustering
    std::vector<TransformHypothesis, Eigen::aligned_allocator<TransformHypothesis>> hypotheses_;

    // Best individual
    TransformHypothesis bestIndividualHypothesis_;
    float bestSimilarity_ = std::numeric_limits<float>::max();
    int bestSimClusterOriginalIndex_ = -1; // 聚类在 clusters_ 中的原始索引
    int bestSimHypoInClusterIndex_ = -1; // 假设在聚类内部的索引
    int bestConsensusIndex_ = -1;
    // get hypoIdx by clusters_[bestSimClusterOriginalIndex_].indices[bestSimHypoInClusterIndex_]
    // also, the index can be accessed in TransformHypothesis struct
    float individualScore_ = -1.0f;

    // Best consensus
    TransformHypothesis bestConsensusHypothesis_;  // 对应原来的 bestEstConsensus
    float consensusScore_ = -1.0f;

    // 用于存储评估结果
    int numCorrectHypotheses_ = 0;

    // ---  用于聚类的成员变量 ---
    pcl::IndicesClusters clusters_;                  // 存储聚类的结果
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr transPoints_; // 存储用于聚类的变换点云
    // Attention!!! This member variable sortedClusters_ is not sorted (but we want implement the sorted version later)
    // Do not get this sorted!!!
    std::vector<ClusterStruct> sortedClusters_;     // 存储按大小排序后的聚类信息
    // sortedClusters_ only keep the cluster index and cluster size

    std::vector<std::vector<int> > subClusterCorrIndices_;

    Eigen::Matrix4f winningTransform_;          // 存储在精炼之前的“胜出”变换
    Eigen::Matrix4f finalBestTransform_;        // 存储精炼后的最终结果

    // 临时为handleClusteringFailure设计的变量
    bool flagFound_ = false;

    // --- 内部核心流程 ---
    void sampleCandidates(const MacGraph &graph);
    void prepareSampledData();

    void generateHypotheses();
    void sortAndFilterHypotheses();


    // 聚类和变换相关函数
static bool EnforceSimilarity1(const pcl::PointXYZINormal &pointA, const pcl::PointXYZINormal &pointB, float squared_distance);

// Check if the Euler angles are within the valid range
// Use inline function
static bool checkEulerAngles(const float angle) {
        return std::isfinite(angle) && angle >= -M_PIf32 && angle <= M_PIf32;
    }
static int clusterTransformationByRotation(const std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> &Rs, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &Ts, float angleThresh, float disThresh, pcl::IndicesClusters &clusters, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans);

    void clusterHypotheses();



    bool handleClusteringFailure(); // 返回 bool 表示是否应提前终止
    void selectBestConsensus();

    // 从聚类中寻找最佳共识假设
    void findConsensusHypotheses();


    void performFinalSelection();
    void refineBestTransform();




// float oamae(const PointCloudPtr& src, const PointCloudPtr& tgt, const Eigen::Matrix4f& est,
//             const std::vector<std::pair<int, std::vector<int>>>& tgtToSrc, float thresh) const;

static void weightedSvd(const PointCloudPtr& srcPts, const PointCloudPtr& tgtPts, Eigen::VectorXf &weights, float weightThreshold, Eigen::Matrix4f& transMat);
    static float transScoreByLocalClique(const PointCloudPtr& srcCorrPts, const PointCloudPtr& tgtCorrPts, const Eigen::Matrix4f& trans, float metricThresh, const std::string &metric) ;
    //     void postRefinement(Eigen::Matrix4f& transform, const std::vector<CorresStruct, Eigen::aligned_allocater<CorresStruct>>& corrs) const;
//     float truncatedChamferDistance(const PointCloudPtr& src, const PointCloudPtr& tgt, const Eigen::Matrix4f& est) const;

public:
    /**
     * @brief 构造函数，注入所需的依赖
     */
    MacRtHypothesis(MacData& data, const MacConfig& config, const MacGraph& graph, MacResult& result);

    // 禁止拷贝，因为它管理着复杂的状态
    MacRtHypothesis(const MacRtHypothesis&) = delete;
    MacRtHypothesis& operator=(const MacRtHypothesis&) = delete;

    /**
     * @brief 主流程函数：接收图分析结果，处理并找到最佳变换
     * @param graph 一个已经完成最大团搜索的 MacGraph 对象
     */
    void processGraphResultAndFindBest(const MacGraph& graph);


    // Getter
    [[nodiscard]] const std::vector<TransformHypothesis, Eigen::aligned_allocator<TransformHypothesis>> &getFinalHypotheses() const { return hypotheses_; }
    [[nodiscard]] int getNumCorrectHypotheses() const { return numCorrectHypotheses_; }
    [[nodiscard]] Eigen::Matrix4f getBestTransform() const { return finalBestTransform_; }
    [[nodiscard]] const std::vector<TransformHypothesis, Eigen::aligned_allocator<TransformHypothesis>>& getHypotheses() const { return hypotheses_; }

};