//
// Created by Jeffery_Xeom on 2025/8/24.
//

#include <fstream>
#include <memory>
#include <sstream>    // 新增：TXT 加载需要解析每行
#include <algorithm>  // 新增：扩展名小写转换需要
#include <cmath>      // 新增：sqrt 计算
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "CommonTypes.hpp"
#include "MacTimer.hpp"
#include "MacMonitor.hpp"
#include "MacData.hpp"
#include "MacUtils.hpp"


/**
 * @brief 构造函数：初始化所有点云指针与评估相关成员
 *  - 指针指向新的空点云，避免 nullptr
 *  - gtTransform 置为单位阵
 *  - 计数与统计清零
 */
MacData::MacData()
    : cloudSrc(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >()),
      cloudTgt(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >()),
      cloudSrcKpts(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >()),
      cloudTgtKpts(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >()),
      cloudResolution(0.0f),
      gtInlierCount(0),
      totalCorresNum(0),
      totalCliqueNum(0){
    gtTransform = Eigen::Matrix4f::Identity();
    originalCorrSrc = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(); // 在构造时初始化智能指针
    originalCorrTgt = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

/**
 * @brief 从不同格式的文件中加载点云（类静态方法）
 *  - 支持 .pcd/.ply/.bin/.txt
 *  - .bin 按 KITTI 风格读取每点4个float(x,y,z,intensity)，默认忽略强度
 *  - .txt 默认按行 "x y z [其他]"，忽略多余列和空/注释行
 * @tparam PointT 点的类型 (例如, pcl::PointXYZ, pcl::PointXYZI)
 * @param filePath 点云文件的路径
 * @param cloud 用于存储加载后点云的 PCL 点云对象
 * @return true 如果加载成功
 * @return false 如果加载失败
 */
template<typename PointT>
bool MacData::loadPointCloud(const std::string &filePath, pcl::PointCloud<PointT> &cloud) {
    // 1. 获取文件扩展名并转为小写
    std::string extension;
    size_t dot_pos = filePath.find_last_of('.');
    if (dot_pos == std::string::npos) {
        LOG_ERROR("Error: No file extension found in " << filePath);
        return false;
    }
    extension = filePath.substr(dot_pos);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // 2. 根据扩展名选择加载方法
    if (extension == ".pcd") {
        if (pcl::io::loadPCDFile<PointT>(filePath, cloud) == -1) {
            LOG_ERROR("Can not open PCD file: " << filePath);
            return false;
        }
    } else if (extension == ".ply") {
        if (pcl::io::loadPLYFile<PointT>(filePath, cloud) == -1) {
            LOG_ERROR("Can not open PLY file: " << filePath);
            return false;
        }
    } else if (extension == ".bin") {
        // 假设是 KITTI 数据集格式: float x, y, z, intensity
        std::ifstream in(filePath, std::ios::binary);
        if (!in.is_open()) {
            LOG_ERROR("Can not open BIN file: " << filePath);
            return false;
        }
        cloud.clear();
        while (true) {
            PointT point;
            float x, y, z, intensity_val;
            // 连续读取 4 个 float；若任一读取失败，跳出循环
            if (!in.read(reinterpret_cast<char *>(&x), sizeof(float))) break;
            if (!in.read(reinterpret_cast<char *>(&y), sizeof(float))) break;
            if (!in.read(reinterpret_cast<char *>(&z), sizeof(float))) break;
            if (!in.read(reinterpret_cast<char *>(&intensity_val), sizeof(float))) break;

            // 赋值几何坐标；强度字段可按需要写入（此处默认忽略）
            point.x = x;
            point.y = y;
            point.z = z;
            // 如果点类型是 PointXYZI，可取消下面的注释写入强度
            if constexpr (std::is_same_v<PointT, pcl::PointXYZI>) {
                point.intensity = intensity_val;
            }
            cloud.push_back(point);
        }
        if (!in.eof()) {
            // 若非正常 EOF 结束，说明文件可能损坏
            LOG_ERROR("The BIN file format is incorrect or corrupted: " << filePath);
            return false;
        }
    } else if (extension == ".txt") {
        // 假设是 TXT 格式: 每行 x y z ...
        std::ifstream in(filePath);
        if (!in.is_open()) {
            LOG_ERROR("Can not open TXT file: " << filePath);
            return false;
        }
        cloud.clear();
        std::string line;
        while (std::getline(in, line)) {
            // 跳过空行与注释行
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            PointT point;
            // 至少需要读取 x, y, z
            if (!(ss >> point.x >> point.y >> point.z)) {
                continue; // 跳过格式错误的行
            }
            // 可在此继续读取更多字段，例如 rgb, intensity 等
            cloud.push_back(point);
        }
    } else {
        LOG_ERROR("Unsupported file format '" << extension << "' for file: " << filePath);
        return false;
    }

    // 3. 确保点云有效
    if (cloud.empty()) {
        LOG_ERROR("Empty cloud input: " << filePath);
        return false;
    }

    // 4. 统一设置点云元数据：非组织点云，稠密
    cloud.width = cloud.size(); // 设置为点的总数
    cloud.height = 1; // 设置为1，表示无组织点云
    cloud.is_dense = true; // 假设所有点都有效

    return true;
}

// The correspondences file contains the xyz coordinates of the source and target keypoints. But the order is the same
//  with the original keypoints clouds. To associate it with kpts, we need to find the index of each correspondence in the original kpts clouds.
// NOTE: the keypoints are not in the original point cloud, therefore nearest search is required.
// Find the nearest point in the source and target key point clouds for each correspondence, and assign the indices to the correspondences.
void MacData::findIndexForCorrespondences(PointCloudPtr &cloudSrcKpts, PointCloudPtr &cloudTgtKpts,
                                                   std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &corres, std::ofstream& corresIndexFileOut) {
    // 使用 KD-Tree 在关键点云中查找每个对应关系的最近索引
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

        // -------- optional --------
        if (corresIndexFileOut.good()) { // 检查文件流是否处于良好状态
            corresIndexFileOut << corr.srcIndex << " " << corr.tgtIndex << "\n";
        }
    }
}

float MacData::meshResolutionCalculation(const PointCloudPtr &pointcloud) {
    // 使用均值最近邻距离来近似点云分辨率（mean root）
    float mr = 0.0f; // mean root
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> point_idx; // 最近邻索引
    std::vector<float> point_dis; // 最近邻距离平方
    kdtree.setInputCloud(pointcloud);

    // 遍历每个点，查找最近邻（k=2，第一个是自身，第二个是最近邻）
    for (auto query_point: pointcloud->points) {
        kdtree.nearestKSearch(query_point, 2, point_idx, point_dis);
        mr += std::sqrt(point_dis[1]); // 距离从平方还原
    }
    mr /= static_cast<float>(pointcloud->points.size());
    return mr; // 近似分辨率
}


// TODO: Check this function
// TODO: This function is not optimized
// Our source target pair is a normal but non-invertible function (surjective, narrowly), which means a source can only have a single target,
// but a target may have many sources. This function is used to find target source pair, where target paired with various sources.
// Only happen if we use one way matching method
void MacData::makeTgtSrcPair(const std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> &correspondence,
                       std::vector<std::pair<int, std::vector<int> > > &tgtSrc) {
    //需要读取保存的kpts, 匹配数据按照索引形式保存
    if (correspondence.size() < 2) {
        LOG_ERROR("The correspondence vector is empty.");
        tgtSrc.clear();
        return;
    }
    tgtSrc.clear();

    // Copy & sort by target index
    std::vector<CorresStruct, Eigen::aligned_allocator<CorresStruct>> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    std::sort(corr.begin(), corr.end(), compareCorresTgtIndex); // sort by target index increasing order

    // Bucketize by tgtIndex: one target to multiple sources (surjective mapping)
    int tgt = corr[0].tgtIndex;
    std::vector<int> src;
    src.reserve(10); // 预分配内存
    src.push_back(corr[0].srcIndex);

    for (size_t i = 1; i < corr.size(); ++i) {
        if (corr[i].tgtIndex != tgt) {
            // flush previous bucket
            tgtSrc.emplace_back(tgt, src);
            src.clear();
            tgt = corr[i].tgtIndex;
        }
        src.push_back(corr[i].srcIndex);
    }
    // Flush the last bucket
    if (!src.empty()) {
        tgtSrc.emplace_back(tgt, src);
    }
    corr.clear();
    corr.shrink_to_fit();
}


/**
 * @brief 从配置中加载所有数据到 MacData 的成员
 *  - 点云与关键点：支持 pcd/ply/bin/txt
 *  - 对应关系：必需，有则读取；索引文件可选，缺失时自动基于关键点查找最近邻索引
 *  - GT 变换/标签：可选，无则以无评估模式继续
 *  - 自动计算点云分辨率
 */
bool MacData::loadData(const MacConfig &config) {
    // 清理旧数据，确保幂等
    // Input Data
    cloudSrc->clear();
    cloudTgt->clear();
    cloudSrcKpts->clear();
    cloudTgtKpts->clear();
    corres.clear();
    // Ground Truth (for evaluation)
    gtTransform.setIdentity();
    gtInlierLabels.clear();

    // Calculated Properties
    cloudResolution = 0.0f;
    gtInlierCount = 0;
    totalCorresNum = 0;
    totalCliqueNum = 0;

    // Correspondence point clouds.
    originalCorrSrc->clear();
    originalCorrTgt->clear();

    // Local timer
    MON_ENTER("loadData");
    Timer timerLoadData;
    LOG_INFO("Output path: " << config.outputPath);

    // --------------------- 读取点云与关键点 ---------------------
    // Stage 1: load clouds (src/tgt & keypoints)
    MON_ENTER("loadClouds");
    timerLoadData.startTiming("load data: load cloud points");
    // 定义一个辅助加载函数，用于检查、加载并打印错误信息
    auto loadAndCheck = [&](const std::string &path, auto &cloudPtr, const std::string &description) -> bool {
        // 使用类静态方法，根据扩展名自动选择加载器
        if (!MacData::loadPointCloud(path, *cloudPtr)) {
            LOG_ERROR("Unable to load " << description << " from: " << path);
            return false;
        }
        if (cloudPtr->empty()) {
            LOG_ERROR(description << " is empty after loading from: " << path);
            return false;
        }
        LOG_INFO("Successfully loaded " << description << " with " << cloudPtr->size()
            << " points from: " << path);
        return true;
    };

    // 依次加载所有点云和关键点
    if (!loadAndCheck(config.cloudSrcPath, cloudSrc, "source point cloud")) return false;
    if (!loadAndCheck(config.cloudTgtPath, cloudTgt, "target point cloud")) return false;
    if (!loadAndCheck(config.cloudSrcKptPath, cloudSrcKpts, "source keypoints")) return false;
    if (!loadAndCheck(config.cloudTgtKptPath, cloudTgtKpts, "target keypoints")) return false;
    timerLoadData.endTiming();
    // Record key counts as metrics for this stage
    MON_SET_KV("N_src", static_cast<int>(cloudSrc->size()));
    MON_SET_KV("N_tgt", static_cast<int>(cloudTgt->size()));
    MON_SET_KV("N_src_kpt", static_cast<int>(cloudSrcKpts->size()));
    MON_SET_KV("N_tgt_kpt", static_cast<int>(cloudTgtKpts->size()));
    MON_RECORD(); // close "loadClouds"

    // --------------------- 读取对应关系与索引 ---------------------
    // Stage 2: load correspondences (+ optional indices)
    MON_ENTER("loadCorrespondences");
    timerLoadData.startTiming("load data: load and process correspondences");
    std::ifstream corresFile(config.corresPath);
    std::ifstream corresIndexFile(config.corresIndexPath);

    // Note that in our version of test data, the source and target matched kpts clouds are already corresponded.
    // But for the original MAC paper, the source and target matched kpts clouds are not corresponded.
    // Load correspondences, xyz
    if (!corresFile.is_open()) {
        LOG_ERROR("Unable to open correspondence file: " << config.corresPath);
        return false;
    }
    CorresStruct match;
    pcl::PointXYZ srcPoint, tgtPoint; // source point and target point in each match
    corres.reserve(10000); // 预分配内存提高效率，一般来说帧对匹配的对应关系不会超过10k
    while (corresFile >> srcPoint.x >> srcPoint.y >> srcPoint.z >>
           tgtPoint.x >> tgtPoint.y >> tgtPoint.z) {
        match.src = srcPoint;
        match.tgt = tgtPoint;
        match.inlierWeight = 0; // 初始化内点权重
        corres.push_back(match);
    }
    corres.shrink_to_fit(); // 释放多余内存
    totalCorresNum = static_cast<int>(corres.size());
    LOG_INFO("Successfully loaded " << totalCorresNum << " correspondences from: " << config.corresPath);

    // 对应关系索引（可选）：缺失则基于关键点查找最近邻索引
    if (!corresIndexFile.is_open()) {
        LOG_INFO("No correspondence index file provided: " << config.corresIndexPath
            << ". Finding and saving indices for correspondences.");
        std::ofstream corresIndexFileOut(config.corresIndexPath);
        findIndexForCorrespondences(cloudSrcKpts, cloudTgtKpts, corres, corresIndexFileOut);
        LOG_INFO("correspondence index file saved: " << config.corresIndexPath);
    } else {
        int i = 0;
        while (i != static_cast<int>(corres.size()) && (corresIndexFile >> corres[i].srcIndex >> corres[i].tgtIndex)) {
            i++;
        }
        if (i > totalCorresNum) {
            LOG_WARNING("Too many correspondences in the index file: " << config.corresIndexPath
                << ". This is probably a wrong index file. Ignoring the rest.");
        } else if (i < totalCorresNum) {
            LOG_WARNING("Not enough correspondences in the index file: " << config.corresIndexPath
                << ". This is probably a wrong index file. Some correspondences will be missing indices.");
        }
    }
    timerLoadData.endTiming();
    MON_SET_KV("corres_total", totalCorresNum);
    MON_RECORD(); // close "loadCorrespondences"


    // --------------------- Compute cloud resolution ---------------------
    // Stage 3: compute or use mesh resolution
    MON_ENTER("calcMeshResolution");
    // Prefer dataset-level override; otherwise estimate from NN distances.
    const DatasetConfig& ds = config.getCurrentDatasetConfig();
    const bool userReso = (ds.meshResolution > 0.0f);
    if (ds.meshResolution != 0.0f) {
        cloudResolution = ds.meshResolution; // user-specified override from dataset config
        LOG_INFO("Using user-defined mesh resolution: " << cloudResolution);
    } else {
        timerLoadData.startTiming("load data: calculate mesh resolution");
        // Estimate resolution as the mean NN distance averaged between src and tgt clouds
        cloudResolution = (meshResolutionCalculation(cloudSrc)
                           + meshResolutionCalculation(cloudTgt)) / 2.0f;
        LOG_INFO("Cloud resolution: " << cloudResolution);
        timerLoadData.endTiming();
    }
    MON_SET_KV("mesh_reso", cloudResolution);
    MON_SET_KV("mesh_reso_source", userReso ? std::string("user") : std::string("estimated"));
    MON_RECORD(); // close "calcMeshResolution"

    // --------------------- 准备完整对应关系点云 ---------------------
    // Stage 4: build full correspondence clouds
    MON_ENTER("prepareCorrClouds");
    timerLoadData.startTiming("load data: prepare full correspondence point clouds");
    LOG_INFO("Preparing full correspondence point clouds...");
    originalCorrSrc->reserve(totalCorresNum); // 预分配内存提高效率
    originalCorrTgt->reserve(totalCorresNum);
    for (const auto& corr : corres) {
        originalCorrSrc->push_back(corr.src);
        originalCorrTgt->push_back(corr.tgt);
    }
    LOG_INFO("Full correspondence point clouds prepared.");
    timerLoadData.endTiming();
    MON_SET_KV("corr_clouds_size", totalCorresNum);
    MON_RECORD(); // close "prepareCorrClouds"

    // 准备评估所需要的数据结构 (makeTgtSrcPair)
    // Stage 5: build tgt->src buckets (surjective mapping)
    MON_ENTER("buildTgtSrcMap");
    timerLoadData.startTiming("load data: prepare target-source pairs");
    LOG_INFO("Preparing target-source pairs...");
    makeTgtSrcPair(corres, tgtSrc);
    LOG_INFO("Total target points with multiple source points: " << tgtSrc.size());
    timerLoadData.endTiming();
    MON_SET_KV("tgt_with_multi_src", static_cast<int>(tgtSrc.size()));
    MON_RECORD(); // close "buildTgtSrcMap"

    // --------------------- 评估部分（可选） ---------------------
    // Stage 6: load/process ground truth transform and labels
    MON_ENTER("loadGroundTruth");
    bool hasGtTf = false;
    bool hasGtLabels = false;
    timerLoadData.startTiming("load data: load and process ground truth data");
    LOG_DEBUG("----------------Evaluation part----------------");
    // Ground Truth 变换矩阵（可选）
    if (std::ifstream gtTfFile(config.gtTfPath); !gtTfFile.is_open()) {
        LOG_DEBUG("No ground truth transformation data: " << config.gtTfPath
            << ". System working without evaluation.");
        // SETTING the no evaluation flag. It is still not implemented.
    } else {
        // 读取 4x4 变换矩阵
        gtTfFile >> gtTransform(0, 0) >> gtTransform(0, 1) >> gtTransform(0, 2) >> gtTransform(0, 3);
        gtTfFile >> gtTransform(1, 0) >> gtTransform(1, 1) >> gtTransform(1, 2) >> gtTransform(1, 3);
        gtTfFile >> gtTransform(2, 0) >> gtTransform(2, 1) >> gtTransform(2, 2) >> gtTransform(2, 3);
        gtTfFile >> gtTransform(3, 0) >> gtTransform(3, 1) >> gtTransform(3, 2) >> gtTransform(3, 3);
        LOG_DEBUG("Ground truth transformation matrix: \n" << gtTransform);
        hasGtTf = true;
    }
    // Ground Truth 内点标签（可选）
    if (std::ifstream gtLabelFile(config.gtLabelPath); !gtLabelFile.is_open()) {
        LOG_DEBUG("No Ground truth correspondence data: " << config.gtLabelPath
            << ". System working without evaluation.");
    } else {
        // 原 MAC++ 版本：逐值读取内/外点标记（1 表示内点）
        int value = 0;
        while (gtLabelFile >> value) {
            gtInlierLabels.push_back(value);
            if (value == 1) {
                gtInlierCount++;
            }
        }

        if (gtInlierCount == 0) {
            LOG_DEBUG("No inliers found in the ground truth correspondences.");
        }
        const float inlier_ratio = (totalCorresNum > 0)
                                       ? static_cast<float>(gtInlierCount) / static_cast<float>(totalCorresNum)
                                       : 0.0f;
        LOG_DEBUG("Inlier ratio: " << inlier_ratio * 100 << "%, GT inliers: " << gtInlierCount
            << ", total correspondences: " << totalCorresNum);
        hasGtLabels = true;
    }
    LOG_DEBUG("-----------------------------------------------");
    timerLoadData.endTiming();
    MON_SET_KV("has_gt_tf", hasGtTf);
    MON_SET_KV("has_gt_labels", hasGtLabels);
    MON_SET_KV("gt_inliers", gtInlierCount);
    MON_RECORD(); // close "loadGroundTruth"

    MON_RECORD(); // close "loadData"

    return true;
}
