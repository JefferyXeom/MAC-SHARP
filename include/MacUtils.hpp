//
// Created by Jeffery_Xeom on 2025/6/19.
//

#ifndef MAC_UTILS_
#define MAC_UTILS_

// For pcl
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <pcl/kdtree/kdtree_flann.h>

//igraph
#include <igraph/igraph.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "MacConfig.hpp"

// Functions declaration
void settingThreads(int desiredThreads);

// Comparison functions
bool compareLocalScore(const CliqueInfo& v1, const CliqueInfo& v2);
bool compareVertexCliqueScore(const vertexCliqueSupport &l1, const vertexCliqueSupport &l2);
bool compareCorresTgtIndex(const CorresStruct& c1, const CorresStruct& c2);
bool compareClusterScore(const ClusterStruct &v1, const ClusterStruct &v2);

// Euclidean distance between two points
inline float getDistance(const pcl::PointXYZ &A, const pcl::PointXYZ &B);



void makeTgtSrcPair(const std::vector<CorresStruct>& correspondence, std::vector<std::pair<int, std::vector<int>>>& tgtSrc);



void weightSvd(PointCloudPtr& srcPts, PointCloudPtr& tgtPts, Eigen::VectorXf& weights, float weightThreshold, Eigen::Matrix4f& transMat);



float oamae(const PointCloudPtr& rawSrc, const PointCloudPtr& rawDes,const Eigen::Matrix4f &est, const std::vector<std::pair<int, std::vector<int>>> &desSrc,const float thresh);

// 添加缺失的函数声明
float calculateRotationError(const Eigen::Matrix3f& est, const Eigen::Matrix3f& gt);
float calculateTranslationError(const Eigen::Vector3f& est,const  Eigen::Vector3f& gt);
float evaluateTransByLocalClique(const PointCloudPtr& srcCorrPts, const PointCloudPtr& desCorrPts, const Eigen::Matrix4f& trans, float metricThresh, const std::string &metric);
bool evaluationEst(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float reThresh, float teThresh, float& RE, float& TE);

// RMSE计算
float rmseCompute(const PointCloudPtr& cloudSource, const PointCloudPtr& cloudTarget, Eigen::Matrix4f& matEst, Eigen::Matrix4f& matGt, float mr);

// 后处理优化
void postRefinement(std::vector<CorresStruct>&correspondence, PointCloudPtr& srcCorrPts, PointCloudPtr& desCorrPts, Eigen::Matrix4f& initial, float& bestScore, float inlierThresh, int iterations, const std::string &metric);

// 向量操作函数
std::vector<int> vectorsUnion(const std::vector<int>& v1, const std::vector<int>& v2);
std::vector<int> vectorsIntersection(const std::vector<int>& v1, const std::vector<int>& v2);

// 点云处理函数
void getCorrPatch(std::vector<CorresStruct>&sampledCorr, PointCloudPtr &src, PointCloudPtr &tgt, PointCloudPtr &patchSrc, PointCloudPtr &patchTgt, float radius);
float truncatedChamferDistance(PointCloudPtr& src, PointCloudPtr& des, Eigen::Matrix4f &est, float thresh);
float oamae1ToK(PointCloudPtr& rawSrc, PointCloudPtr& rawDes, Eigen::Matrix4f &est, std::vector<std::pair<int, std::vector<int>>> &srcDes, float thresh);

// 聚类内部变换评估
Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int bestIndex, Eigen::Matrix4f &initial, std::vector<Eigen::Matrix3f> &Rs, std::vector<Eigen::Vector3f> &Ts, PointCloudPtr& srcKpts, PointCloudPtr& desKpts, std::vector<std::pair<int, std::vector<int>>> &desSrc, float thresh, Eigen::Matrix4f& gtMat, std::string folderPath);
Eigen::Matrix4f clusterInternalTransEva1(const pcl::IndicesClusters &clusterTrans, int bestIndex, Eigen::Matrix4f &initial, std::vector<Eigen::Matrix3f> &Rs, const std::vector<Eigen::Vector3f> &Ts, PointCloudPtr& srcKpts, PointCloudPtr& desKpts, std::vector<std::pair<int, std::vector<int>>> &desSrc, float thresh, Eigen::Matrix4f& gtMat, bool oneToK, std::string folderPath);


#endif // MAC_UTILS_

