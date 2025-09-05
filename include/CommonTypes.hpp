//
// Created by Jeffery_Xeom on 2025/8/24.
//

#pragma once

// system
#include <ostream>
#include <string>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>

////============================================================
///
/// System
///
////============================================================

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
// #define ORANGE "\033[38;5;208m" // 橙色
#define BRIGHT_YELLOW "\033[93m"  // 亮黄色，比普通黄色更显眼
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

// 日志级别枚举
enum class MacLogLevel {
    MAC_SILENT = 0,   // 完全静默
    MAC_ERROR = 1,    // 仅错误
    MAC_CRITICAL = 2, // 严重警告及以上
    MAC_WARNING = 3,  // 警告及以上
    MAC_INFO = 4,     // 信息及以上
    MAC_DEBUG = 5     // 所有输出
};

// 全局日志控制类
class MacLogger {
    // private
    inline static MacLogLevel currentLevel;
public:
    static void setLevel(const MacLogLevel level) {
        currentLevel = level;
    }
    static MacLogLevel getLevel() {
        return currentLevel;
    }
    static bool shouldLog(MacLogLevel level) {
        return static_cast<int>(level) <= static_cast<int>(currentLevel);
    }
};
// 日志宏定义
#define LOG_ERROR(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_ERROR)) \
std::cout << RED << "[ERROR] " << msg << RESET << std::endl
#define LOG_CRITICAL(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_CRITICAL)) \
std::cout << BRIGHT_YELLOW << "[CRITICAL] " << msg << RESET << std::endl
#define LOG_WARNING(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_WARNING)) \
std::cout << YELLOW << "[WARNING] " << msg << RESET << std::endl
#define LOG_INFO(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_INFO)) \
std::cout << "[INFO] " << msg << std::endl
#define LOG_DEBUG(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_DEBUG)) \
std::cout << BLUE << "[DEBUG] " << msg << RESET << std::endl
#define LOG_TIMER(msg) \
if (MacLogger::shouldLog(MacLogLevel::MAC_INFO)) \
std::cout << BLUE << "[Timer] " << msg << RESET << std::endl



////============================================================
///
/// Registration related
///
////============================================================

// Use an enumeration type to clearly represent the formula to be used
enum class ScoreFormula {
    GAUSSIAN_KERNEL,
    QUADRATIC_FALLOFF
};
/**
 * @brief Parses a string to its corresponding ScoreFormula enum value.
 * @param str The string from the YAML file.
 * @return The ScoreFormula enum value.
 * @throws std::invalid_argument if the string is not a valid formula name.
 */
inline ScoreFormula parseScoreFormula(const std::string& str) {
    if (str == "GAUSSIAN_KERNEL") return ScoreFormula::GAUSSIAN_KERNEL;
    if (str == "QUADRATIC_FALLOFF") return ScoreFormula::QUADRATIC_FALLOFF;
    // For robustness, throw an exception for unknown values
    LOG_WARNING("Unknown ScoreFormula: " + str + ". Defaulting to GAUSSIAN_KERNEL.");
    return ScoreFormula::GAUSSIAN_KERNEL;
}

/**
 * @struct CorresStruct
 * @brief Correspondence structure to store the correspondence between two point clouds
 */
typedef struct CorresStruct {
    int srcIndex = -1; // Index in source keypoints cloud for current correspondence
    int tgtIndex = -1;
    pcl::PointXYZ src; // point in source keypoints cloud for current correspondence
    pcl::PointXYZ tgt;
    Eigen::Vector3f srcNorm; // not initialized, not used
    Eigen::Vector3f TgtNorm; // not initialized, not used
    float corresScore = 1.0f; // equal to triangle score in graph
    int inlierWeight = 0; // initialized with 0, not used
} CorresStruct;

/**
 * @struct VertexStruct
 * @brief Vertex structure for graph and degree calculation: graphVertex_
 *
 */
typedef struct VertexStruct
{
    int vertexIndex = -1;
    int degree = -1;
    float triWeight = 0.0f; // triangle weight must be in initialized to 0.0f
    float vertexScore = -1.0f;
    int neighborCorrectMatchNum = -1;
    std::vector<int> neighborIndices{};
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    VertexStruct() {}
    // 这正是 emplace_back 需要的构造函数！
    VertexStruct(const int idx, const float scr)
        : vertexIndex(idx), vertexScore(scr) {}
} VertexStruct;

/**
 * @struct CliqueInfo
 * @brief 轻量级结构体，用于存储一个团的索引及其分数。
 */
typedef struct CliqueInfo {
    int cliqueIndex = -1;
    float cliqueScore = -1.0f;
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    CliqueInfo() {}
    // 这正是 emplace_back 需要的构造函数！
    CliqueInfo(const int idx, const float scr)
        : cliqueIndex(idx), cliqueScore(scr) {}
} CliqueInfo;

/**
 * @struct vertexCliqueSupport
 * @brief 存储每个顶点从其所属的所有最大团中获得的支持度信息。
 * @details
 *  该结构体为图中的每一个顶点（即一个对应关系）创建一个实例。它用于累加和存储一个顶点
 *  因其参与构成的所有最大团（Maximal Cliques）而获得的分数。这个分数可以被看作是
 *  该顶点在图中最密集区域的“中心性”或“重要性”的度量。
 *
 *  - `vertexIndex`: 顶点的原始索引，与 `MacData::corres` 中的索引对应。
 *  - `cliqueSupportScore`: 累加分数。该分数是此顶点所属的所有最大团的权重之和。
 *  - `participatingCliques`: 存储了所有包含该顶点的最大团的索引和分数。
 *  - `flagGtCorrect`: (仅用于评估) 标记该顶点是否为真值内点（Ground Truth Inlier）。
 */
typedef struct vertexCliqueSupport
{
    int vertexIndex = -1; // clique index
    float score = 0.0f; // clique score must be initialized with 0.0f
    std::vector<CliqueInfo>participatingCliques{};
    bool flagGtCorrect = false; // Only used in gt evaluation
    // --- 构造函数 ---
    // default constructor
    vertexCliqueSupport() {}
    // for emplace_back
    vertexCliqueSupport(const int idx, const float scr, const bool flg)
        : vertexIndex(idx), score(scr), flagGtCorrect(flg) {}
} vertexCliqueSupport;

/**
 * @struct ClusterStruct
 * @brief Cluster structure
 */
typedef struct ClusterStruct
{
    int clusterIndex;
    int clusterSize;
    bool flagGtCorrect; // Only used in gt evaluation
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    ClusterStruct() : clusterIndex(0), clusterSize(0), flagGtCorrect(false) {}
    // 这正是 emplace_back 需要的构造函数！
    ClusterStruct(const int idx, const int size, const bool flg)
        : clusterIndex(idx), clusterSize(size), flagGtCorrect(flg) {}
} ClusterStruct;

// 这里的每一个成员变量的_都去掉，并重命名
// 用一个清晰的结构体来表示一个变换假设，取代原来零散的 std::vector
struct TransformHypothesis {
    int originalIndex_ = -1; // 在初始假设列表中的索引
    Eigen::Matrix4f transform_ = Eigen::Matrix4f::Identity();
    float localScore_ = 0.0f;  // 基于团内部点的分数
    float globalScore_ = 0.0f; // 基于所有关键点的分数 (OAMAE)
    bool isGtCorrect_ = false; // (仅用于评估)
    std::vector<int> sourceCorrespondenceIndices_; // 这个假设来源于哪些对应关系的索引
};

// 定义 PointCloudPtr 以便复用
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

