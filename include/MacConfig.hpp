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

#include "CommonTypes.hpp"

/**
 * @struct DatasetConfig
 * @brief 数据集特定配置结构体
 *
 * 该结构体包含了针对不同数据集的特定参数配置，这些参数之前在代码中是硬编码的。
 * 现在通过配置文件统一管理，提高了代码的灵活性和可维护性。
 */
struct DatasetConfig {
    // Dataset identification, only for evaluation
    std::string datasetName;

    // 评估相关阈值
    float reThresh = 15.0f; // 旋转误差评估阈值，单位：度
    float teThresh = 30.0f; // 平移误差评估阈值，单位：厘米
    float inlierEvaThresh = 0.1f; // 内点评估阈值，单位：米

    // 聚类相关阈值
    float clusterAngThresh = 5.0f; // 变换矩阵聚类的角度阈值，单位：度
    float clusterDistThresh = 0.1f; // 变换矩阵聚类的距离阈值，单位：米

    // 成功判断阈值
    float successRmseThresh = 0.2f; // 配准成功的RMSE阈值

    /**
     * @brief 从YAML节点加载数据集特定配置
     * @param node YAML配置节点
     * @param name 数据集名称
     */
    void loadFromYAML(const YAML::Node &node, const std::string &name) {
        datasetName = name;

        // 加载基本阈值参数，如果YAML中没有对应key则使用默认值
        reThresh = node["rotationErrorThreshold"].as<float>(reThresh);
        teThresh = node["translationErrorThreshold"].as<float>(teThresh);
        inlierEvaThresh = node["inlierEvaluationThreshold"].as<float>(inlierEvaThresh);
        clusterAngThresh = node["clusteringAngleThreshold"].as<float>(clusterAngThresh);
        clusterDistThresh = node["clusteringDistanceThreshold"].as<float>(clusterDistThresh);
        successRmseThresh = node["successRMSEThreshold"].as<float>(successRmseThresh);
    }

    /**
     * @brief 获取实际的内点评估阈值
     * @return 实际使用的内点评估阈值
     */
    [[nodiscard]] float getActualInlierThreshold() const {
        return inlierEvaThresh;
    }

    /**
     * @brief 获取实际的聚类距离阈值
     * @return 实际使用的聚类距离阈值
     */
    [[nodiscard]] float getActualClusteringDistanceThreshold() const {
        return clusterDistThresh;
    }
};

/**
 * @struct MacConfig
 * @brief Holds all configuration parameters loaded from the YAML file.
 * [STYLE] Struct name is UpperCamelCase, member variables are lowerCamelCase.
 */
struct MacConfig {
    // --- Path Configurations ---
    std::string descriptor; // Descriptor name, e.g., "FPFH".
    std::string cloudSrcPath; // Source point cloud file path.
    std::string cloudTgtPath; // Target point cloud file path.
    std::string cloudSrcKptPath; // Source keypoint cloud file path.
    std::string cloudTgtKptPath; // Target keypoint cloud file path.
    std::string corresPath; // Correspondence file path (coordinates).
    std::string corresIndexPath; // Correspondence index file path, used for one-to-many matching.
    std::string gtLabelPath; // Ground truth label file path (inliers/outliers).
    std::string gtTfPath; // Ground truth transformation file path.
    std::string outputPath; // Output directory for results and logs.

    // --- Global Configuration Flags ---
    bool flagLowInlierRatio = false; // Flag for low inlier ratio scenarios
    bool flagAddOverlap = false; // Flag for adding overlap info
    bool flagNoLogs = false; // Flag to disable logging
    bool flagSecondOrderGraph = true; // Flag to enable second-order graph construction (SC^2)
    bool flagUseIcp = true; // Flag to enable the final ICP refinement step
    bool flagInstanceEqual = true; // Flag related to instance weighting
    bool flagClusterInternalEvaluation = true; // Flag to enable final refinement within best cluster
    bool flagUseTopK = false; // Flag for developmental feature related to top-k selection
    bool flagVerbose = false; // Verbose flag for detailed output

    // --- Global Configuration Parameters ---
    int maxEstimateNum = INT_MAX; // Maximum number of hypotheses to process before clustering
    std::string metric = "MAE"; // Metric used for local evaluation
    int desiredThreads = -1; // Number of threads (-1 = all available)
    int totalIterations = 1; // Number of iterations for registration process
    ScoreFormula scoreFormula = ScoreFormula::GAUSSIAN_KERNEL; // Formula used in Graph_construction

    // Thresholds
    float triangularScoreThresh = 0.0f; // Threshold for triangle score in generate hypotheses
    float threshold = 1.8f; // kitti, for trans score by local clique

    // Maximal clique search parameters
    int maxTotalCliqueNum = INT_MAX;
    int maxCliqueIterations = INT_MAX;
    int maxCliqueSize = INT_MAX;

    // Clique filtering parameters
    int maxLocalCliqueNum = INT_MAX;



    // temporary mesh resolution
    float meshResolution = 0.00605396; // for 3DLoMatch

    // --- Logging Configuration ---
    MacLogLevel logLevel = MacLogLevel::MAC_DEBUG;

    // --- Current Dataset Configuration ---
    std::string currentDatasetName;
    DatasetConfig currentDatasetConfig;

    // --- Dataset Structure ---
    std::map<std::string, std::vector<std::string> > datasets;

private:
    // 存储所有可用的数据集配置
    std::map<std::string, DatasetConfig> availableDatasetConfigs;

    /**
     * @brief 解析评分公式字符串
     */
    static ScoreFormula parseScoreFormula(const std::string &formulaStr) {
        if (formulaStr == "GAUSSIAN_KERNEL") return ScoreFormula::GAUSSIAN_KERNEL;
        if (formulaStr == "QUADRATIC_FALLOFF") return ScoreFormula::QUADRATIC_FALLOFF;
        return ScoreFormula::GAUSSIAN_KERNEL; // default
    }

    /**
     * @brief 解析日志级别字符串
     */
    static MacLogLevel parseLogLevel(const std::string &levelStr) {
        if (levelStr == "DEBUG") return MacLogLevel::MAC_DEBUG;
        if (levelStr == "INFO") return MacLogLevel::MAC_INFO;
        if (levelStr == "WARNING") return MacLogLevel::MAC_WARNING;
        if (levelStr == "CRITICAL") return MacLogLevel::MAC_CRITICAL;
        if (levelStr == "ERROR") return MacLogLevel::MAC_ERROR;
        return MacLogLevel::MAC_INFO; // default
    }

public:
    /**
     * @brief 获取当前数据集的配置
     * @return 当前数据集的DatasetConfig对象引用
     */
    [[nodiscard]] const DatasetConfig &getCurrentDatasetConfig() const {
        return currentDatasetConfig;
    }

    /**
     * @brief 切换到指定数据集
     * @param datasetName 数据集名称
     * @throws std::runtime_error 如果指定数据集不存在
     */
    void switchToDataset(const std::string &datasetName) {
        const auto it = availableDatasetConfigs.find(datasetName);
        if (it == availableDatasetConfigs.end()) {
            throw std::runtime_error("No configuration found for dataset: " + datasetName);
        }
        currentDatasetName = datasetName;
        currentDatasetConfig = it->second;
    }

    /**
     * @brief 获取所有可用的数据集名称
     * @return 数据集名称列表
     */
    [[nodiscard]] std::vector<std::string> getAvailableDatasets() const {
        std::vector<std::string> names;
        names.reserve(availableDatasetConfigs.size());
        for (const auto &[name, config]: availableDatasetConfigs) {
            names.push_back(name);
        }
        return names;
    }

    /**
     * @brief Loads all configuration parameters from a YAML file into the struct.
     * @param filename The path to the config.yaml file.
     * @throws std::runtime_error if file cannot be loaded or required fields are missing
     */
    void load(const std::string &filename) {
        try {
            YAML::Node config = YAML::LoadFile(filename);

            // --- Loading Inputs and Outputs ---
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
            // --- Loading Flags ---
            if (config["flags"]) {
                const auto &flags = config["flags"];
                flagLowInlierRatio = flags["lowInlierRatio"].as<bool>(flagLowInlierRatio);
                flagAddOverlap = flags["addOverlap"].as<bool>(flagAddOverlap);
                flagNoLogs = flags["noLogs"].as<bool>(flagNoLogs);
                flagSecondOrderGraph = flags["secondOrderGraph"].as<bool>(flagSecondOrderGraph);
                flagUseIcp = flags["useIcp"].as<bool>(flagUseIcp);
                flagInstanceEqual = flags["instanceEqual"].as<bool>(flagInstanceEqual);
                flagClusterInternalEvaluation = flags["clusterInternalEvaluation"].as<bool>(
                    flagClusterInternalEvaluation);
                flagUseTopK = flags["useTopK"].as<bool>(flagUseTopK);
                flagVerbose = flags["verbose"].as<bool>(flagVerbose);
            }

            // --- Loading Global Variables ---
            if (config["globalVariables"]) {
                const auto &globals = config["globalVariables"];
                maxEstimateNum = globals["maxEstimateNum"].as<int>(maxEstimateNum);
                metric = globals["metric"].as<std::string>(metric);
                scoreFormula = parseScoreFormula(globals["scoreFormula"].as<std::string>("GAUSSIAN_KERNEL"));
                desiredThreads = globals["desiredThreads"].as<int>(desiredThreads);
                totalIterations = globals["totalIterations"].as<int>(totalIterations);
                maxTotalCliqueNum = globals["maxTotalCliqueNum"].as<int>(maxTotalCliqueNum);
                maxCliqueIterations = globals["maxCliqueIterations"].as<int>(maxCliqueIterations);
                maxCliqueSize = globals["maxCliqueSize"].as<int>(maxCliqueSize);
                maxLocalCliqueNum = globals["maxLocalCliqueNum"].as<int>(maxLocalCliqueNum);
            }

            // --- Setting Log Level ---
            if (config["systemControls"] && config["systemControls"]["logLevel"]) {
                logLevel = parseLogLevel(config["systemControls"]["logLevel"].as<std::string>("INFO"));
                MacLogger::setLevel(logLevel);
            }

            // --- 加载所有数据集配置 ---
            if (config["datasetConfigs"]) {
                for (const auto &datasetNode: config["datasetConfigs"]) {
                    const auto datasetName = datasetNode.first.as<std::string>();
                    DatasetConfig datasetConfig;
                    datasetConfig.loadFromYAML(datasetNode.second, datasetName);
                    availableDatasetConfigs[datasetName] = datasetConfig;
                }
            }

            // --- 设置当前数据集 ---
            // if (config["currentDataset"]) {
            //     const auto requestedDataset = config["currentDataset"].as<std::string>();
            //     switchToDataset(requestedDataset);
            // } else if (!availableDatasetConfigs.empty()) {
            //     // 如果没有指定当前数据集，使用第一个可用的
            //     switchToDataset(availableDatasetConfigs.begin()->first);
            // }

            // --- 加载数据集结构 ---
            if (config["datasets"]) {
                for (const auto &datasetNode: config["datasets"]) {
                    const auto datasetName = datasetNode.first.as<std::string>();
                    datasets[datasetName] = datasetNode.second.as<std::vector<std::string> >();
                }
            }
        } catch (const YAML::Exception &e) {
            throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
        } catch (const std::exception &e) {
            throw std::runtime_error("Configuration loading error: " + std::string(e.what()));
        }
    }

    /**
     * @brief 验证配置的完整性
     * @return true if configuration is valid
     */
    [[nodiscard]] bool validate() const {
        // 检查必要的路径是否为空
        if (descriptor.empty() || cloudSrcPath.empty() || cloudTgtPath.empty()) {
            return false;
        }

        // 检查当前数据集配置是否有效
        if (currentDatasetName.empty()) {
            return false;
        }

        return true;
    }
};
