//
// Created by Jeffery_Xeom on 2025/8/24.
//

#pragma once

#include <pcl/point_cloud.h>
#include "CommonTypes.hpp"  // 新增：提供 PointCloudPtr、CorresStruct、日志宏、Eigen 等常用公共类型
#include "MacConfig.hpp" // 为了 MacConfig

class MacData {
    // --- 私有辅助方法 (静态) ---
    // 将它们设为静态(static)是因为它们不依赖于某个特定的RegistrationData实例的状态，
    // 它们更像是独立的工具函数，只是逻辑上属于这个类。

    /**
     * @brief 为对应关系查找在关键点云中的索引
     */
    static void findIndexForCorrespondences(PointCloudPtr& cloudSrcKpts, PointCloudPtr& cloudTgtKpts,
                                              std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>>& corres, std::ofstream& corresIndexFileOut);

    /**
     * @brief 计算点云的分辨率
     */
    static float meshResolutionCalculation(const PointCloudPtr& pointcloud);

    /**
     * @brief 模板化的点云加载函数 (原loadPointCloud)
     */
    template<typename PointT>
    static bool loadPointCloud(const std::string &filePath, pcl::PointCloud<PointT> &cloud);

    static void makeTgtSrcPair(const std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &correspondence,
                               std::vector<std::pair<int, std::vector<int> > > &tgtSrc);

public:
    // Input Data
    PointCloudPtr cloudSrc; // Complete point clouds
    PointCloudPtr cloudTgt;
    // If keypoints contain only correspondences and are ordered with correspondence,
    // then originalCorr is the same with cloudKpts.
    PointCloudPtr cloudSrcKpts; // Keypoints clouds, considered as unordered
    PointCloudPtr cloudTgtKpts;
    std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> corres;

    // Ground Truth (for evaluation)
    Eigen::Matrix4f gtTransform;
    std::vector<int> gtInlierLabels; // No default initialization, calculated by us

    // Calculated Properties
    float cloudResolution; // 由数据加载模块计算后赋值
    int gtInlierCount;  // 由数据加载模块计算后赋值
    int totalCorresNum; // 由数据加载模块计算后赋值
    int totalCliqueNum; // 由图模块计算后赋值
    std::vector<std::pair<int, std::vector<int>>> tgtSrc; // 目标点到源点索引的映射

    // --- correspondences ---
    // Correspondence point clouds.
    // If keypoints contain only correspondences and are ordered with correspondence,
    // then originalCorr is the same with cloudKpts.
    PointCloudPtr originalCorrSrc;
    PointCloudPtr originalCorrTgt;

    // --- 公有成员方法 ---
    /**
     * @brief 构造函数：初始化所有智能指针
     */
    MacData();

    /**
     * @brief 从配置文件加载所有相关数据
     * @param config 配置对象
     * @return 如果加载成功返回 true，否则返回 false
     */
    bool loadData(const MacConfig &config);

    // geters
    [[nodiscard]] const PointCloudPtr& getSrcCorrPts() const { return originalCorrSrc; }
    [[nodiscard]] const PointCloudPtr& getTgtCorrPts() const { return originalCorrTgt; }

};

/**
 * @struct MacResult
 * @brief MAC算法的结果结构体，用于统一管理所有输出参数
 *
 * 该结构体包含了MAC点云配准算法执行后的所有关键结果信息：
 * - 旋转和平移误差：用于评估配准精度
 * - 正确估计数量：算法找到的正确对应关系数量
 * - 真实内点数量：ground truth中的内点数量
 * - 执行时间：算法运行耗时
 * - 预测内点比率：算法预测的内点统计信息
 */
typedef struct MacResult {
    float RE;                          // 旋转误差 (Rotation Error)，单位：度
    float TE;                          // 平移误差 (Translation Error)，单位：毫米或米
    int correctEstNum;                  // 算法估计对应关系中的正确对应关系数量
    int gtInlierNum;                    // Ground Truth中的内点数量
    float timeEpoch;                   // 算法执行时间，单位：秒
    std::vector<float> predictedInlier; // 预测内点比率向量，包含精确率、召回率、F1分数等
    bool evalPass{};
    float reThreshDegUsed{};
    float teThreshMUsed{};
    float inlierThreshMUsed{};
    int inlierPredictedCount{};
    int inlierCorrectCount{};
    int inlierGtTotalCount{};

    /**
     * @brief 默认构造函数
     *
     * 初始化所有数值成员为0，向量为空
     */
    MacResult() : RE(0.0), TE(0.0), correctEstNum(0), gtInlierNum(0), timeEpoch(0.0) {
        predictedInlier.clear();
    }

    /**
     * @brief 重置所有结果数据
     *
     * 将所有成员变量重置为初始状态，用于多次运行时清理之前的结果
     */
    void reset() {
        RE = 0.0;
        TE = 0.0;
        correctEstNum = 0;
        gtInlierNum = 0;
        timeEpoch = 0.0;
        predictedInlier.clear();
    }
} MacResult;