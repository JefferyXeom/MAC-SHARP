//
// Created by Jeffery_Xeom on 2025/8/24.
//

#pragma once

// system
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
    MAC_SILENT = 0, // 完全静默
    MAC_ERROR = 1, // 仅错误
    MAC_CRITICAL = 2, // 严重警告及以上
    MAC_WARNING = 3, // 警告及以上
    MAC_INFO = 4, // 信息及以上
    MAC_DEBUG = 5 // 所有输出
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
///
// 定义一个枚举来表示方差的计算模式
enum class VarianceMode {
    FIXED, // 传统方案：使用固定的alphaDis
    DYNAMIC // 动态方案：使用我们推导的sigma_ij^2
};

// Use an enumeration type to clearly represent the formula to be used
enum class ScoreFormula {
    GAUSSIAN_KERNEL,
    QUADRATIC_FALLOFF
};

// // 包含一个点所有必需信息（包括预计算）的最终结构体
// struct PointInfo {
//     // 原始数据
//     float x, y, z;
//     float rho, theta, phi;
//
//     // 预计算的几何属性
//     Eigen::Vector3f cartesianPos;
//     Eigen::Vector3f eHatRho;
//     Eigen::Vector3f eHatTheta;
//     Eigen::Vector3f eHatPhi;
//     float sinTheta;
//
//     // 构造函数：从最基础的笛卡尔坐标初始化，并完成所有预计算
//     // 在数据加载阶段调用一次即可
//     explicit PointInfo(const float pX = 0.f, const float pY = 0.f, const float pZ = 0.f) : x(pX), y(pY), z(pZ) {
//         // --- 预计算阶段 ---
//         // 1. 计算球坐标 (mu 值)
//         rho = std::sqrt(x*x + y*y + z*z);
//         if (rho > 1e-6f) {
//             theta = std::acos(z / rho);
//             phi = std::atan2(y, x);
//         } else {
//             theta = 0.0f;
//             phi = 0.0f;
//         }
//
//         // 2. 缓存笛卡尔向量
//         cartesianPos = Eigen::Vector3f(x, y, z);
//
//         // 3. 缓存三角函数值和局部基向量
//         sinTheta = std::sin(theta);
//         const float cosTheta = std::cos(theta);
//         const float sinPhi = std::sin(phi);
//         const float cosPhi = std::cos(phi);
//
//         eHatRho = Eigen::Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
//         eHatTheta = Eigen::Vector3f(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
//         eHatPhi = Eigen::Vector3f(-sinPhi, cosPhi, 0.0f);
//     }
// };

/**
 * @struct PrecomputedInfo
 * @brief 存储一个点的预计算几何信息，避免在O(N^2)循环中重复计算。
 * @details 该结构体在初始化时，会一次性计算并缓存点的球坐标、
 * 笛卡尔向量表示、局部基向量以及后续方差计算所需的三角函数值。
 */
struct PrecomputedInfo {
    // Overload new operator to ensure proper memory alignment for Eigen types in heaps
    // This prevents potential segmentation faults due to misaligned memory access
    // when using Eigen types in STL containers like std::vector.
    // Reference: https://eigen.tuxfamily.org/dox/group__TopicStructHaving
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // <-- 确保Eigen向量的内存对齐以避免索引错误

    // --- 预计算的球坐标 (对应我们公式中的 mu 值) ---
    /// @brief 径向距离 rho
    float rho;
    /// @brief 极角/天顶角 theta (弧度)
    float theta;
    /// @brief 方位角 phi (弧度)
    float phi;

    // --- 预计算的几何属性 ---
    /// @brief 笛卡尔坐标，存储为Eigen向量以提高后续运算效率
    Eigen::Vector3f cartesianPos;
    /// @brief 局部的径向单位基向量 e_hat_rho
    Eigen::Vector3f eHatRho;
    /// @brief 局部的极角单位基向量 e_hat_theta
    Eigen::Vector3f eHatTheta;
    /// @brief 局部的方位角单位基向量 e_hat_phi
    Eigen::Vector3f eHatPhi;
    /// @brief 预计算的 sin(theta)，用于方位角方差的计算，避免重复调用sin函数
    float sinTheta;

    /// @brief 默认构造函数
    PrecomputedInfo() = default;

    /**
     * @brief 核心初始化函数，从一个PCL点对象完成所有预计算
     * @param point 输入的pcl::PointXYZ点
     */
    void computeFrom(const pcl::PointXYZ &point) {
        // 将PCL点转换为Eigen向量，便于后续的向量运算
        cartesianPos = Eigen::Vector3f(point.x, point.y, point.z);

        // 步骤 1: 从笛卡尔坐标计算球坐标 (mu 值)
        rho = cartesianPos.norm(); // rho 是向量的模长
        if (rho > 1e-6f) {
            // acos的参数范围必须在[-1, 1]，这里通过归一化保证
            theta = std::acos(cartesianPos.z() / rho);
            // 使用atan2确保phi落在正确的象限
            phi = std::atan2(cartesianPos.y(), cartesianPos.x());
        } else {
            // 如果点在原点，球坐标未定义，设为0
            theta = 0.0f;
            phi = 0.0f;
        }

        // 步骤 2: 缓存三角函数值和局部基向量
        sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        const float sinPhi = std::sin(phi);
        const float cosPhi = std::cos(phi);

        eHatRho = Eigen::Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        eHatTheta = Eigen::Vector3f(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
        eHatPhi = Eigen::Vector3f(-sinPhi, cosPhi, 0.0f);
    }
};

/**
 * @struct CorresStruct
 * @brief Correspondence structure to store the correspondence between two point clouds
 */
typedef struct CorresStruct {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // <-- 确保Eigen向量的内存对齐以避免索引错误
    int srcIndex = -1; // Index in source keypoints cloud for current correspondence
    int tgtIndex = -1;
    pcl::PointXYZ src; // point in source keypoints cloud for current correspondence
    pcl::PointXYZ tgt;
    Eigen::Vector3f srcNorm; // not initialized, not used
    Eigen::Vector3f TgtNorm; // not initialized, not used
    float corresScore = 1.0f; // equal to triangle score in graph
    int inlierWeight = 0; // initialized with 0, not used

    // --- 新增的成员 ---
    PrecomputedInfo srcPrecomputed;
    PrecomputedInfo tgtPrecomputed;
} CorresStruct;

/**
 * @struct VertexStruct
 * @brief Vertex structure for graph and degree calculation: graphVertex_
 *
 */
typedef struct VertexStruct {
    int vertexIndex = -1;
    int degree = -1;
    float triWeight = 0.0f; // triangle weight must be in initialized to 0.0f
    float vertexScore = -1.0f;
    int neighborCorrectMatchNum = -1;
    std::vector<int> neighborIndices{};
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    VertexStruct() {
    }

    // 这正是 emplace_back 需要的构造函数！
    VertexStruct(const int idx, const float scr)
        : vertexIndex(idx), vertexScore(scr) {
    }
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
    CliqueInfo() {
    }

    // 这正是 emplace_back 需要的构造函数！
    CliqueInfo(const int idx, const float scr)
        : cliqueIndex(idx), cliqueScore(scr) {
    }
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
typedef struct vertexCliqueSupport {
    int vertexIndex = -1; // clique index
    float score = 0.0f; // clique score must be initialized with 0.0f
    std::vector<CliqueInfo> participatingCliques{};
    bool flagGtCorrect = false; // Only used in gt evaluation
    // --- 构造函数 ---
    // default constructor
    vertexCliqueSupport() {
    }

    // for emplace_back
    vertexCliqueSupport(const int idx, const float scr, const bool flg)
        : vertexIndex(idx), score(scr), flagGtCorrect(flg) {
    }
} vertexCliqueSupport;

/**
 * @struct ClusterStruct
 * @brief Cluster structure
 */
typedef struct ClusterStruct {
    int clusterIndex;
    int clusterSize;
    bool flagGtCorrect; // Only used in gt evaluation
    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    ClusterStruct() : clusterIndex(0), clusterSize(0), flagGtCorrect(false) {
    }

    // 这正是 emplace_back 需要的构造函数！
    ClusterStruct(const int idx, const int size, const bool flg)
        : clusterIndex(idx), clusterSize(size), flagGtCorrect(flg) {
    }
} ClusterStruct;

// 这里的每一个成员变量的_都去掉，并重命名
// 用一个清晰的结构体来表示一个变换假设，取代原来零散的 std::vector
struct TransformHypothesis {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // <-- 确保Eigen向量的内存对齐以避免索引错误
    int originalIndex_ = -1; // 在初始假设列表中的索引
    Eigen::Matrix4f transform_ = Eigen::Matrix4f::Identity();
    float localScore_ = 0.0f; // 基于团内部点的分数
    float globalScore_ = 0.0f; // 基于所有关键点的分数 (OAMAE)
    bool isGtCorrect_ = false; // (仅用于评估)
    std::vector<int> sourceCorrespondenceIndices_; // 这个假设来源于哪些对应关系的索引
};

// 定义 PointCloudPtr 以便复用
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
