//
// Created by Jeffery_Xeom on 2025/6/19.
//

#ifndef _MAC_UTILS_
#define _MAC_UTILS_

// For pcl
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

//igraph
#include <igraph/igraph.h>

#include "config_loader.hpp"


// Terminal color codes for output
// Various platform terminal supported
// --------
// const std::string RED = "\x1b[91m";
// const std::string GREEN = "\x1b[92m";
// const std::string YELLOW = "\x1b[93m";
// const std::string BLUE = "\x1b[94m";
// const std::string RESET = "\x1b[0m"; // 恢复默认颜色
// --------
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"



// General type define
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

// Point cloud correspondences structure
// For variable correspondences
/**
 * @struct CorresStruct
 * @brief Correspondence structure to store the correspondence between two point clouds
 */
typedef struct CorresStruct {
    int srcIndex;
    int tgtIndex;
    pcl::PointXYZ src;
    pcl::PointXYZ tgt;
    Eigen::Vector3f srcNorm;
    Eigen::Vector3f TgtNorm;
    float score;
    int inlierWeight;
} CorresStruct;


/**
 * @struct VertexDgree
 * @brief Vertex Dgree structure for degree calculation
 */
typedef struct VertexDgree
{
    int currentIndex;
    int degree;
    float score;
    std::vector<int> corresIndex;
    int localCorrectMatchNum;
} VertexDgree; // for degree calculation

/**
 * @struct VertexStruct
 * @brief Vertex structure for cluster factor and evaluation_est
 */
typedef struct VertexStruct
{
    int currentIndex;
    float score;
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    VertexStruct() : currentIndex(0), score(0.0f) {}
    // 这正是 emplace_back 需要的构造函数！
    VertexStruct(const int idx, const float scr)
        : currentIndex(idx), score(scr) {}
} VertexStruct; //


/**
 * @struct CliqueStruct
 * @brief Clique structure for cluster factor and evaluation_est
 */
typedef struct CliqueStruct
{
    int currentIndex;
    float score;
    bool flagGtCorrect; // Only used in gt evaluation
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    CliqueStruct() : currentIndex(0), score(0.0f), flagGtCorrect(false) {}
    // 这正是 emplace_back 需要的构造函数！
    CliqueStruct(const int idx, const float scr, const bool flg)
        : currentIndex(idx), score(scr), flagGtCorrect(flg) {}
} CliqueStruct;


/**
 * @struct ClusterStruct
 * @brief Cluster structure for cluster factor and evaluation_est
 */
typedef struct ClusterStruct
{
    int currentIndex;
    float clusterSize;
    bool flagGtCorrect; // Only used in gt evaluation
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    ClusterStruct() : currentIndex(0), clusterSize(0.0f), flagGtCorrect(false) {}
    // 这正是 emplace_back 需要的构造函数！
    ClusterStruct(const int idx, const float scr, const bool flg)
        : currentIndex(idx), clusterSize(scr), flagGtCorrect(flg) {}
} ClusterStruct;

/**
 * @struct LocalClique
 * @brief Local clique structure
 */
typedef struct LocalClique{
    int currentInd = 0; // clique index
    std::vector<VertexStruct>cliqueIndScore;
    float score = 0.0f;

    // default constructor
    LocalClique() {
        cliqueIndScore.clear();
    }
    // constructor
    explicit LocalClique(int ind, float scr) : currentInd(ind), score(scr) {
        cliqueIndScore.clear();
    }
} LocalClique; // for



//
// C++17 Standard specialized, ^ 17+, v 98+
// --------
// Global variables
inline int totalCorresNum = 0; // Total number of correspondences
inline int totalCliqueNum = 0; // Number of cliques found

// --------
// extern int totalCorrespondencesNum = 0; // Total number of correspondences
// extern int cliqueNum = 0; // Number of cliques found
// --------
// However, the conventional one need to be explict defined in the cpp files.
// So if you want to use C++ standard below 17, you need to define it in the cpp file.


// Timer class should be carefully checked
/**
 * @class Timer
 * @brief A timer class to measure the elapsed time
 */
class Timer {
public:
    Timer() = default;

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(endTime - startTime).count();
        elapsedTimes.push_back(elapsed);
        std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;
    }

    const std::vector<double>& getElapsedTimes() const {
        return elapsedTimes;
    }

    void reset() {
        elapsedTimes.clear();
    }

private:
    std::chrono::high_resolution_clock::time_point startTime, endTime;
    std::vector<double> elapsedTimes;
};

// Temporary time functions
void timing(const int timeFlag);

/**
 * @struct MACResult
 * @brief MAC算法的结果结构体，用于统一管理所有输出参数
 *
 * 该结构体包含了MAC点云配准算法执行后的所有关键结果信息：
 * - 旋转和平移误差：用于评估配准精度
 * - 正确估计数量：算法找到的正确对应关系数量
 * - 真实内点数量：ground truth中的内点数量
 * - 执行时间：算法运行耗时
 * - 预测内点比率：算法预测的内点统计信息
 */
typedef struct MACResult {
    double RE;                          // 旋转误差 (Rotation Error)，单位：度
    double TE;                          // 平移误差 (Translation Error)，单位：毫米或米
    int correctEstNum;                  // 正确估计的对应关系数量
    int gtInlierNum;                    // Ground Truth中的内点数量
    double timeEpoch;                   // 算法执行时间，单位：秒
    std::vector<double> predicatedInlier; // 预测内点比率向量，包含精确率、召回率、F1分数等

    /**
     * @brief 默认构造函数
     *
     * 初始化所有数值成员为0，向量为空
     */
    MACResult() : RE(0.0), TE(0.0), correctEstNum(0), gtInlierNum(0), timeEpoch(0.0) {
        predicatedInlier.clear();
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
        predicatedInlier.clear();
    }
} MACResult;

// Functions declaration
void settingThreads(int desiredThreads);
bool loadData(const MACConfig &macConfig, PointCloudPtr &cloudSrc, PointCloudPtr &cloudTgt, PointCloudPtr &cloudSrcKpts,
    PointCloudPtr &cloudTgtKpts, std::vector<CorresStruct> &corresOriginal, std::vector<int> &gtCorres, Eigen::Matrix4f &gtMat,
    int &gtInlierNum, float &cloudResolution);
float meshResolutionCalculation(const PointCloudPtr &pointcloud);
void findIndexForCorrespondences(PointCloudPtr &cloudSrcKpts, PointCloudPtr &cloudTgtKpts, std::vector<CorresStruct> &corres);
inline float getDistance(const pcl::PointXYZ &A, const pcl::PointXYZ &B);
Eigen::MatrixXd graphConstruction(std::vector<CorresStruct> &correspondences, float resolution, bool secondOrderGraphFlag, ScoreFormula formula);
float otsuThresh(std::vector<float> allScores);

// Comparison functions
bool compareLocalScore(const VertexStruct& v1, const VertexStruct& v2);
bool compareVertexCliqueScore(const LocalClique &l1, const LocalClique &l2);
bool compareCorrespondenceIndex(const CorresStruct& c1, const CorresStruct& c2);
bool compareClusterScore(const ClusterStruct &v1, const ClusterStruct &v2);

void cliqueSampling(const MACConfig &macConfig, Eigen::MatrixXd &graph, const igraph_vector_int_list_t *cliques, std::vector<int> &sampledCorresIndex, std::vector<int> &sampledCliqueIndex);
void makeTgtSrcPair(const std::vector<CorresStruct>& correspondence, std::vector<std::pair<int, std::vector<int>>>& tgtSrc);
void weightSvd(PointCloudPtr& srcPts, PointCloudPtr& tgtPts, Eigen::VectorXf& weights, float weightThreshold, Eigen::Matrix4f& transMat);

float OAMAE(PointCloudPtr& rawSrc, PointCloudPtr& rawDes, Eigen::Matrix4f &est, std::vector<std::pair<int, std::vector<int>>> &desSrc, float thresh);

// 添加缺失的函数声明
float calculateRotationError(Eigen::Matrix3f& est, Eigen::Matrix3f& gt);
float calculateTranslationError(Eigen::Vector3f& est, Eigen::Vector3f& gt);
float evaluateTransByLocalClique(const PointCloudPtr& srcCorrPts, const PointCloudPtr& desCorrPts, const Eigen::Matrix4f& trans, float metricThresh, const std::string &metric);
bool evaluationEst(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float reThresh, float teThresh, double& RE, double& TE);

// 聚类和变换相关函数
bool EnforceSimilarity1(const pcl::PointXYZINormal &pointA, const pcl::PointXYZINormal &pointB, float squaredDistance);
bool checkEulerAngles(float angle);
int clusterTransformationByRotation(const std::vector<Eigen::Matrix3f> &Rs, const std::vector<Eigen::Vector3f> &Ts, float angleThresh, float disThresh, pcl::IndicesClusters &clusters, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans);

// RMSE计算
float rmseCompute(const PointCloudPtr& cloudSource, const PointCloudPtr& cloudTarget, Eigen::Matrix4f& matEst, Eigen::Matrix4f& matGt, float mr);

// 后处理优化
void postRefinement(std::vector<CorresStruct>&correspondence, PointCloudPtr& srcCorrPts, PointCloudPtr& desCorrPts, Eigen::Matrix4f& initial, float& bestScore, float inlierThresh, int iterations, const std::string &metric);

// 向量操作函数
std::vector<int> vectorsUnion(const std::vector<int>& v1, const std::vector<int>& v2);
std::vector<int> vectorsIntersection(const std::vector<int>& v1, const std::vector<int>& v2);

// 点云处理函数
void getCorrPatch(std::vector<CorresStruct>&sampledCorr, PointCloudPtr &src, PointCloudPtr &des, PointCloudPtr &srcBatch, PointCloudPtr &desBatch, float radius);
float truncatedChamferDistance(PointCloudPtr& src, PointCloudPtr& des, Eigen::Matrix4f &est, float thresh);
float oamae1ToK(PointCloudPtr& rawSrc, PointCloudPtr& rawDes, Eigen::Matrix4f &est, std::vector<std::pair<int, std::vector<int>>> &srcDes, float thresh);

// 聚类内部变换评估
Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int bestIndex, Eigen::Matrix4f &initial, std::vector<Eigen::Matrix3f> &Rs, std::vector<Eigen::Vector3f> &Ts, PointCloudPtr& srcKpts, PointCloudPtr& desKpts, std::vector<std::pair<int, std::vector<int>>> &desSrc, float thresh, Eigen::Matrix4f& gtMat, std::string folderPath);
Eigen::Matrix4f clusterInternalTransEva1(pcl::IndicesClusters &clusterTrans, int bestIndex, Eigen::Matrix4f &initial, std::vector<Eigen::Matrix3f> &Rs, std::vector<Eigen::Vector3f> &Ts, PointCloudPtr& srcKpts, PointCloudPtr& desKpts, std::vector<std::pair<int, std::vector<int>>> &desSrc, float thresh, Eigen::Matrix4f& gtMat, bool oneToK, std::string folderPath);


#endif //_MAC_UTILS_

