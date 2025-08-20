//
// Created by Jeffery_Xeom on 2025/6/30.
//
#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>
#include <climits>
#include <map>
#include <vector>

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
    throw std::invalid_argument("Unknown ScoreFormula: " + str);
}

/**
 * @struct DatasetConfig
 * @brief 数据集特定配置结构体
 *
 * 该结构体包含了针对不同数据集的特定参数配置，这些参数之前在代码中是硬编码的。
 * 现在通过配置文件统一管理，提高了代码的灵活性和可维护性。
 */
struct DatasetConfig {
    // 评估相关阈值
    float rotationErrorThreshold;              // 旋转误差评估阈值，单位：度
    float translationErrorThreshold;           // 平移误差评估阈值，单位：厘米
    float inlierEvaluationThreshold;           // 内点评估阈值，单位：米

    // U3M数据集特殊参数：使用分辨率的倍数
    float inlierEvaluationThresholdMultiplier; // 内点评估阈值倍数（针对U3M）

    // 聚类相关阈值
    float clusteringAngleThreshold;            // 变换矩阵聚类的角度阈值，单位：度
    float clusteringDistanceThreshold;         // 变换矩阵聚类的距离阈值，单位：米
    float clusteringDistanceThresholdMultiplier; // 聚类距离阈值倍数（针对U3M）

    // 成功判断阈值
    float successRMSEThreshold;                // 配准成功的RMSE阈值

    /**
     * @brief 默认构造函数
     *
     * 初始化所有参数为默认值，避免未初始化的内存访问
     */
    DatasetConfig() :
        rotationErrorThreshold(15.0f),
        translationErrorThreshold(30.0f),
        inlierEvaluationThreshold(0.1f),
        inlierEvaluationThresholdMultiplier(5.0f),
        clusteringAngleThreshold(5.0f),
        clusteringDistanceThreshold(0.1f),
        clusteringDistanceThresholdMultiplier(5.0f),
        successRMSEThreshold(0.2f) {}

    /**
     * @brief 从YAML节点加载数据集配置
     * @param node YAML配置节点
     */
    void loadFromYAML(const YAML::Node& node) {
        // 加载基本阈值参数，如果YAML中没有对应key则使用默认值
        rotationErrorThreshold = node["rotationErrorThreshold"].as<float>(rotationErrorThreshold);
        translationErrorThreshold = node["translationErrorThreshold"].as<float>(translationErrorThreshold);
        inlierEvaluationThreshold = node["inlierEvaluationThreshold"].as<float>(inlierEvaluationThreshold);
        inlierEvaluationThresholdMultiplier = node["inlierEvaluationThresholdMultiplier"].as<float>(inlierEvaluationThresholdMultiplier);
        clusteringAngleThreshold = node["clusteringAngleThreshold"].as<float>(clusteringAngleThreshold);
        clusteringDistanceThreshold = node["clusteringDistanceThreshold"].as<float>(clusteringDistanceThreshold);
        clusteringDistanceThresholdMultiplier = node["clusteringDistanceThresholdMultiplier"].as<float>(clusteringDistanceThresholdMultiplier);
        successRMSEThreshold = node["successRMSEThreshold"].as<float>(successRMSEThreshold);
    }

    /**
     * @brief 获取实际的内点评估阈值
     * @param cloudResolution 点云分辨率
     * @param isU3M 是否为U3M数据集
     * @return 实际使用的内点评估阈值
     */
    float getActualInlierThreshold(float cloudResolution, bool isU3M) const {
        if (isU3M) {
            return inlierEvaluationThresholdMultiplier * cloudResolution;
        }
        return inlierEvaluationThreshold;
    }

    /**
     * @brief 获取实际的聚类距离阈值
     * @param cloudResolution 点云分辨率
     * @param isU3M 是否为U3M数据集
     * @return 实际使用的聚类距离阈值
     */
    float getActualClusteringDistanceThreshold(float cloudResolution, bool isU3M) const {
        if (isU3M) {
            return clusteringDistanceThresholdMultiplier * cloudResolution;
        }
        return clusteringDistanceThreshold;
    }
};

/**
 * @struct MACConfig
 * @brief Holds all configuration parameters loaded from the YAML file.
 * [STYLE] Struct name is UpperCamelCase, member variables are lowerCamelCase.
 */
struct MACConfig {
    // --- Path Configurations ---
    std::string datasetName;        // Dataset name, used for selecting different parameter presets.
    std::string descriptor;         // Descriptor name, e.g., "FPFH".
    std::string cloudSrcPath;       // Source point cloud file path.
    std::string cloudTgtPath;       // Target point cloud file path.
    std::string cloudSrcKptPath;    // Source keypoint cloud file path.
    std::string cloudTgtKptPath;    // Target keypoint cloud file path.
    std::string corresPath;         // Correspondence file path (coordinates).
    std::string corresIndexPath;    // Correspondence index file path, used for one-to-many matching.
    std::string gtLabelPath;        // Ground truth label file path (inliers/outliers).
    std::string gtTfPath;           // Ground truth transformation file path.
    std::string outputPath;         // Output directory for results and logs.

    // --- Dataset Structure ---
    // Stores the list of scenes for each dataset defined in the YAML file.
    std::map<std::string, std::vector<std::string>> datasets;

    // --- 数据集特定配置 ---
    // 存储所有数据集的特定配置参数，用于替代代码中的硬编码值
    std::map<std::string, DatasetConfig> datasetConfigs;

    // --- Global Configuration Variables ---
    // These variables are defined as inline to allow definition in the header file (C++17 feature).
    // They will be loaded from the YAML file and can be accessed from anywhere in the program.
    // [STYLE] All variable names are unified to lowerCamelCase for consistency.

    // Global configuration flags, with default values
    bool flagLowInlierRatio = false;              // Flag for low inlier ratio scenarios, may trigger different strategies.
    bool flagAddOverlap = false;                  // Flag for adding overlap info, maybe deprecated in future versions.
    bool flagNoLogs = false;                      // Flag to disable logging, for cleaner execution.
    bool flagSecondOrderGraph = true;             // Flag to enable second-order graph construction (SC^2).
    bool flagUseIcp = true;                       // Flag to enable the final ICP refinement step.
    bool flagInstanceEqual = true;                // Flag related to instance weighting.
    bool flagClusterInternalEvaluation = true;    // Flag to enable the final refinement within the best cluster.
    bool flagUseTopK = false;                     // Flag for a developmental feature related to top-k selection.
    bool flagVerbose = false;                     // Verbose flag for detailed output, default is false.

    // Global configuration parameters
    int maxEstimateNum = INT_MAX;                 // Maximum number of hypotheses to process before clustering. Default is no limit.
    std::string metric = "MAE";                   // The metric used for local evaluation (evaluation_trans).
    int desiredThreads = -1;                      // Number of threads to use. -1 means using all available threads.
    int totalIterations = 1;                      // Number of iterations for the registration process, default is 1.
    ScoreFormula scoreFormula = ScoreFormula::GAUSSIAN_KERNEL; // The formula used in Graph_construction. Default is GAUSSIAN_KERNEL.
    // Maximal clique search parameters
    int maxTotalCliqueNum = INT_MAX;
    int maxCliqueIterations = INT_MAX;
    int maxCliqueSize = INT_MAX;
    // Clique filtering parameters
    int maxLocalCliqueNum = INT_MAX;

    /**
     * @brief 获取当前数据集的配置
     * @return 当前数据集的DatasetConfig对象引用
     * @throws std::runtime_error 如果当前数据集没有对应的配置
     */
    const DatasetConfig& getCurrentDatasetConfig() const {
        const auto it = datasetConfigs.find(datasetName);
        if (it == datasetConfigs.end()) {
            throw std::runtime_error("No configuration found for dataset: " + datasetName);
        }
        return it->second;
    }

    /**
     * @brief 检查数据集名称是否为U3M
     * @return 如果是U3M数据集返回true，否则返回false
     */
    bool isU3MDataset() const {
        return datasetName == "U3M";
    }

    /**
     * @brief Loads all configuration parameters from a YAML file into the struct and global variables.
     * @param filename The path to the config.yaml file.
     */
    void load(const std::string& filename) {
        YAML::Node config = YAML::LoadFile(filename);

        // --- Loading Inputs and Outputs ---
        // [NOTE] The keys used here (e.g., "datasetName") must match the keys in the YAML file.
        datasetName = config["datasetName"].as<std::string>();
        descriptor = config["descriptor"].as<std::string>();
        cloudSrcPath = config["cloudSrcPath"].as<std::string>();
        cloudTgtPath = config["cloudTgtPath"].as<std::string>();
        cloudSrcKptPath = config["cloudSrcKptPath"].as<std::string>();
        cloudTgtKptPath = config["cloudTgtKptPath"].as<std::string>();
        corresPath = config["corresPath"].as<std::string>();
        corresIndexPath = config["corresIndexPath"].as<std::string>();
        gtLabelPath = config["gtLabelPath"].as<std::string>();
        gtTfPath = config["gtTfPath"].as<std::string>();
        outputPath = config["outputPath"].as<std::string>();

        // --- Loading Flags into global variables ---
        flagLowInlierRatio = config["flags"]["lowInlierRatio"].as<bool>();
        flagAddOverlap = config["flags"]["addOverlap"].as<bool>();
        flagNoLogs = config["flags"]["noLogs"].as<bool>();
        flagSecondOrderGraph = config["flags"]["secondOrderGraph"].as<bool>();
        flagUseIcp = config["flags"]["useIcp"].as<bool>();
        flagInstanceEqual = config["flags"]["instanceEqual"].as<bool>();
        flagClusterInternalEvaluation = config["flags"]["clusterInternalEvaluation"].as<bool>();
        flagUseTopK = config["flags"]["useTopK"].as<bool>();
        flagVerbose = config["flags"]["verbose"].as<bool>(); // Verbose flag, default is false

        // --- Loading Global Variables ---
        maxEstimateNum = config["globalVariables"]["maxEstimateNum"].as<int>();
        metric = config["globalVariables"]["metric"].as<std::string>();
        scoreFormula = parseScoreFormula(config["globalVariables"]["scoreFormula"].as<std::string>());
        desiredThreads = config["globalVariables"]["desiredThreads"].as<int>();
        totalIterations = config["globalVariables"]["totalIterations"].as<int>();

        maxTotalCliqueNum = config["globalVariables"]["maxTotalCliqueNum"].as<int>();
        maxCliqueIterations = config["globalVariables"]["maxCliqueIterations"].as<int>();
        maxCliqueSize = config["globalVariables"]["maxCliqueSize"].as<int>();

        maxLocalCliqueNum = config["globalVariables"]["maxLocalCliqueNum"].as<int>();

        // --- 加载数据集特定配置 ---
        // 这里加载所有数据集的特定配置参数，替代之前的硬编码方式
        if (config["datasetConfigs"]) {
            YAML::Node datasetConfigsNode = config["datasetConfigs"];
            for (const auto& dsConfig : datasetConfigsNode) {
                std::string dsName = dsConfig.first.as<std::string>();
                DatasetConfig dsConf;
                dsConf.loadFromYAML(dsConfig.second);
                datasetConfigs[dsName] = dsConf;
            }
        }

        // --- Loading Datasets Structure ---
        YAML::Node dsNode = config["datasets"];
        for (const auto& category : dsNode) {
            auto key = category.first.as<std::string>();
            std::vector<std::string> sceneList;
            for (const auto& name : category.second) {
                sceneList.push_back(name.as<std::string>());
            }
            datasets[key] = sceneList;
        }
    }
};