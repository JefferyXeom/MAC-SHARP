//
// Created by Jeffery_Xeom on 2025/6/30.
//
// Refactored MacConfig.hpp to align with the new config.yaml layout:
// Sections: execution / algorithm / evaluation
// - Preserves backward-compatible field names used across the codebase.
// - C++17, header-only implementation for convenience.
// - Thorough English comments explaining responsibilities and units.
//
// Notes:
//  * All member names use lowerCamelCase.
//  * Threshold units in YAML are already SI (meters/degrees). No unit conversion is applied here.
//  * NSigma must be strictly positive (> 0) — validated at load time.
//  * U3M-style multiplier thresholds are normalized to absolute values on demand.
//    If resolution is required but not available, a warning is printed and a reasonable fallback is used.
//
#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <limits>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <cctype>

#include "CommonTypes.hpp"  // enums: VarianceMode, ScoreFormula, MacLogLevel; MacLogger; LOG_* macros

/**
 * @brief Per-dataset evaluation configuration loaded from YAML.
 *
 * The structure holds both absolute thresholds (meters/degrees) and optional
 * multiplier-based definitions (e.g., U3M uses multipliers w.r.t. mesh resolution).
 * To keep the rest of the pipeline simple, we expose helpers that return the
 * *effective* absolute thresholds:
 *   - getActualInlierThreshold()
 *   - getActualClusteringDistanceThreshold()
 *
 * Resolution priority when multipliers are used:
 *   1) meshResolution (explicitly provided in YAML for this dataset), if > 0
 *   2) runtimeMeshResolution (set by algorithm after loading point clouds), if > 0
 * If both are missing and only multipliers are provided, a warning is logged and a
 * conservative fallback is used (defaults kept from initialization).
 */
struct DatasetConfig {
    // Dataset identification, only for evaluation
    std::string datasetName;

    // --- Absolute thresholds (preferred when present) ---
    // Units:
    //   * rotationErrorThreshold: degrees
    //   * translationErrorThreshold: meters
    //   * inlierThreshold: meters
    //   * clusteringAngleThreshold: degrees
    //   * clusteringDistanceThreshold: meters
    //   * successRMSEThreshold: meters
    float reThresh = 15.0f;
    float teThresh = 0.30f;
    float inlierEvaThresh = 0.10f;
    float clusterAngThresh = 5.0f;
    float clusterDistThresh = 0.10f;
    float successRmseThresh = 0.20f; // 配准成功的RMSE阈值

    // --- Optional multiplier-based definitions (U3M style) ---
    // If > 0, the corresponding absolute value should be resolved using resolution.
    float inlierThreshMultiplier = 0.0f;
    float clusterDistThreshMultiplier = 0.0f;

    // --- Resolution hints (meters) ---
    // meshResolution: dataset-scope, provided in YAML; takes precedence when > 0
    // runtimeMeshResolution: provided by the algorithm after loading the clouds
    float meshResolution = 0.0f;
    float runtimeMeshResolution = 0.0f;

    /**
     * @brief Load dataset config from the YAML evaluation.datasets.<name> node.
     * @param node YAML node containing the dataset section.
     * @param name Dataset name key from YAML (e.g., "3DMatch", "U3M").
     */
    void loadFromYAML(const YAML::Node &node, const std::string &name) {
        datasetName = name;

        // Absolute paths: all are optional with sensible defaults.
        if (node["rotationErrorThreshold"]) reThresh = node["rotationErrorThreshold"].as<float>(reThresh);
        if (node["translationErrorThreshold"]) teThresh = node["translationErrorThreshold"].as<float>(teThresh);
        if (node["inlierThreshold"]) inlierEvaThresh = node["inlierThreshold"].as<float>(inlierEvaThresh);
        if (node["clusteringAngleThreshold"])
            clusterAngThresh = node["clusteringAngleThreshold"].as<float>(
                clusterAngThresh);
        if (node["clusteringDistanceThreshold"])
            clusterDistThresh = node["clusteringDistanceThreshold"].as<float>(
                clusterDistThresh);
        if (node["successRMSEThreshold"]) successRmseThresh = node["successRMSEThreshold"].as<float>(successRmseThresh);

        // Multiplier-based fields (U3M style)
        if (node["inlierThresholdMultiplier"])
            inlierThreshMultiplier = node["inlierThresholdMultiplier"].as<float>(
                inlierThreshMultiplier);
        if (node["clusteringDistanceThresholdMultiplier"])
            clusterDistThreshMultiplier = node["clusteringDistanceThresholdMultiplier"].as<float>(
                clusterDistThreshMultiplier);

        // Optional per-dataset mesh resolution (meters)
        if (node["meshResolution"]) meshResolution = node["meshResolution"].as<float>(meshResolution);
    }

    /**
     * @brief Update runtime mesh resolution (meters), typically computed from data.
     */
    void setRuntimeMeshResolution(const float resM) {
        runtimeMeshResolution = resM;
    }

    /**
     * @brief Return the resolution to use for multiplier-based thresholds.
     * Priority: YAML meshResolution > runtimeMeshResolution. 0.0f means "unknown".
     */
    [[nodiscard]] float effectiveResolution() const {
        if (meshResolution > 0.0f) return meshResolution;
        if (runtimeMeshResolution > 0.0f) return runtimeMeshResolution;
        return 0.0f;
    }

    /**
     * @brief Compute the absolute inlier distance threshold in meters.
     * If the dataset provides a multiplier and a resolution is available, use them.
     * Else, fall back to the absolute value (inlierEvaThresh).
     */
    [[nodiscard]] float getActualInlierThreshold() const {
        if (inlierThreshMultiplier > 0.0f) {
            if (const float res = effectiveResolution(); res > 0.0f) return inlierThreshMultiplier * res;
            // Fall back with a warning: keep absolute value if present
            LOG_WARNING("inlierThresholdMultiplier is provided for dataset \"" << datasetName
                << "\" but mesh resolution is unknown. Falling back to inlierThreshold=" << inlierEvaThresh << " m.");
        }
        return inlierEvaThresh;
    }

    /**
     * @brief Compute the absolute clustering distance threshold in meters.
     * If the dataset provides a multiplier and a resolution is available, use them.
     * Else, fall back to the absolute value (clusterDistThresh).
     */
    [[nodiscard]] float getActualClusteringDistanceThreshold() const {
        if (clusterDistThreshMultiplier > 0.0f) {
            if (const float res = effectiveResolution(); res > 0.0f) return clusterDistThreshMultiplier * res;
            // Fall back with a warning: keep absolute value if present
            LOG_WARNING("clusteringDistanceThresholdMultiplier is provided for dataset \"" << datasetName
                << "\" but mesh resolution is unknown. Falling back to clusteringDistanceThreshold="
                << clusterDistThresh << " m.");
        }
        return clusterDistThresh;
    }
};

// ==============================
// Monitor configuration
// ==============================

/** Granularity levels for performance logging.
 *  CORE: only key/top-level stages;
 *  ALL : every stage recorded (fine-grained).
 */
enum class MonitorGranularity { CORE, ALL };

/** Metric-writing strategy for stage notes (CSV "note" column).
 *  OFF : never write metrics (time/memory only).
 *  AUTO: write metrics only when evaluation.enabled==true.
 *  ALL : always write metrics regardless of evaluation flag.
 */
enum class MonitorMetrics { OFF, AUTO, ALL };

/** Charset for the ASCII tree printed at the end. */
enum class MonitorTreeCharset { UNICODE, ASCII };

/** Sub-config for the time-tree (hierarchical timing summary). */
struct MonitorTimeTreeConfig {
    bool enabled = true; // print time-tree summary at the end
    MonitorTreeCharset charset = MonitorTreeCharset::UNICODE;
    bool showMemory = true; // show RSS/PeakRSS on each node
    bool showNotes = true; // append a short note snippet per node
    int maxDepth = -1; // -1 means no limit
};

// [Added] Summary printing control (start-of-run config dump)
struct MonitorSummaryConfig {
    bool enabled = true; // when false, skip printing the summary banner
};

/** Global monitor configuration. */
struct MonitorConfig {
    bool enabled = true; // master switch for monitoring
    MonitorGranularity granularity = MonitorGranularity::ALL;
    MonitorMetrics metrics = MonitorMetrics::AUTO;
    int coreDepth = 2; // only used when granularity==CORE
    MonitorTimeTreeConfig timeTree; // nested config
    MonitorSummaryConfig summary; // [Added]
};

/**
 * @struct MacConfig
 * @brief The central configuration structure used by the whole pipeline.
 *
 * Compatibility goals:
 *  - Keep the public member names previously used by the code (e.g., sigmaRho, flagSecondOrderGraph, outputPath, etc.).
 *  - Internally align to your new YAML schema: execution / algorithm / evaluation.
 *  - Provide helpers for dataset switching and U3M-style threshold resolution.
 */
struct MacConfig {
    // public:
    // =========================== Execution/Paths ===========================
    // (Backward-compatible member names expected by data loading code.)
    std::string cloudSrcPath; // Source point cloud file path.
    std::string cloudTgtPath; // Target point cloud file path.
    std::string cloudSrcKptPath; // Source keypoint cloud file path.
    std::string cloudTgtKptPath; // Target keypoint cloud file path.
    std::string corresPath; // Correspondence file path (coordinates).
    std::string corresIndexPath; // Correspondence index file path, used for one-to-many matching.
    std::string gtLabelPath; // Ground truth label file path (inliers/outliers).
    std::string gtTfPath; // Ground truth transformation file path.
    std::string outputPath; // Output directory for results and logs.

    // Which dataset to run/evaluate.
    std::string currentDatasetName;

    // System controls
    int desiredThreads = -1; // Number of threads (-1 = all available)
    MacLogLevel logLevel = MacLogLevel::MAC_DEBUG; // console verbosity
    bool flagNoLogs = false; // if true, do not write log file (maps from execution.system.noLogFile)

    // ============================ Algorithm ================================
    // Lidar noise model
    VarianceMode varianceMode = VarianceMode::DYNAMIC;
    float sigmaRho = 0.05f; // meters
    float sigmaTheta = 0.2304f; // degrees (zenith)
    float sigmaPhi = 0.09f; // degrees (azimuth)
    float nSigma = 1.0f; // n sigma for outlier rejection, must be > 0

    // Graph construction
    ScoreFormula scoreFormula = ScoreFormula::GAUSSIAN_KERNEL; // Formula used in Graph_construction
    bool flagSecondOrderGraph = false; // Flag to enable second-order graph construction (SC^2)

    // Clique search
    int maxTotalCliqueNum = 10000000;
    int maxCliqueIterations = 5;
    int maxCliqueSize = 15;
    // Clique filtering parameters
    int maxLocalCliqueNum = 11;

    // Hypothesis
    int maxEstimateNum = std::numeric_limits<int>::max(); // Maximum number of hypotheses to process before clustering
    bool flagInstanceEqual = true; // Flag related to instance weighting, used for icp
    std::string metric = "MAE"; // local evaluation metric method

    // Refinement / selection
    bool flagUseIcp = false; // Flag to enable the final ICP refinement step
    bool flagClusterInternalEvaluation = true; // Flag to enable final refinement within best cluster

    // Additional algorithm parameters kept for compatibility
    // TODO: Check this two threshold! They are algorithm related
    float triangularScoreThresh = 0.0f; // used in hypothesis generation
    float threshold = 0.0f; // distance threshold used by some evaluation/refinement helpers

    // ============================= Evaluation ==============================
    // Current dataset resolved config
    DatasetConfig currentDatasetConfig;
    // When false, the entire evaluation pass is skipped:
    // - MacEvaluator::validateFinalTransform is not executed
    // - No evaluation-specific monitor stages are recorded
    // - No metrics (RE/TE/inliers/PRF1) are written
    bool evaluationEnabled = true;
    MonitorConfig monitor;

private:
    // All available datasets parsed from evaluation.datasets
    std::map<std::string, DatasetConfig> availableDatasetConfigs_;

    // --- Dataset Structure ---
    // pending how to use
    std::map<std::string, std::vector<std::string> > datasets;

    // Helper: parse "FIXED"/"DYNAMIC"
    static VarianceMode parseVarianceMode(const std::string &modeStr) {
        if (modeStr == "FIXED") return VarianceMode::FIXED;
        if (modeStr == "DYNAMIC") return VarianceMode::DYNAMIC;
        LOG_WARNING("Unknown varianceMode: " << modeStr << ", defaulting to DYNAMIC.");
        return VarianceMode::DYNAMIC;
    }

    // Helper: parse score formula
    static ScoreFormula parseScoreFormula(const std::string &formulaStr) {
        if (formulaStr == "GAUSSIAN_KERNEL") return ScoreFormula::GAUSSIAN_KERNEL;
        if (formulaStr == "QUADRATIC_FALLOFF") return ScoreFormula::QUADRATIC_FALLOFF;
        LOG_WARNING("Unknown scoreFormula: " << formulaStr << ", defaulting to GAUSSIAN_KERNEL.");
        return ScoreFormula::GAUSSIAN_KERNEL;
    }

    // Helper: parse log level string -> MacLogLevel
    static MacLogLevel parseLogLevel(const std::string &levelStr) {
        if (levelStr == "DEBUG") return MacLogLevel::MAC_DEBUG;
        if (levelStr == "INFO") return MacLogLevel::MAC_INFO;
        if (levelStr == "WARNING") return MacLogLevel::MAC_WARNING;
        if (levelStr == "CRITICAL") return MacLogLevel::MAC_CRITICAL;
        if (levelStr == "ERROR") return MacLogLevel::MAC_ERROR;
        LOG_WARNING("Unknown logLevel: " << levelStr << ", defaulting to DEBUG.");
        return MacLogLevel::MAC_DEBUG;
    }

    // ==============================
    // Inline helpers for parsing (header-only)
    // ==============================
    static inline std::string toLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    static inline MonitorGranularity parseGranularity(const std::string &v, const MonitorGranularity defVal) {
        const auto s = toLower(v);
        if (s == "core") return MonitorGranularity::CORE;
        if (s == "all") return MonitorGranularity::ALL;
        return defVal;
    }

    static inline MonitorMetrics parseMetrics(const std::string &v, MonitorMetrics defVal) {
        const auto s = toLower(v);
        if (s == "off") return MonitorMetrics::OFF;
        if (s == "auto") return MonitorMetrics::AUTO;
        if (s == "all") return MonitorMetrics::ALL;
        return defVal;
    }

    static inline MonitorTreeCharset parseCharset(const std::string &v, MonitorTreeCharset defVal) {
        const auto s = toLower(v);
        if (s == "unicode") return MonitorTreeCharset::UNICODE;
        if (s == "ascii") return MonitorTreeCharset::ASCII;
        return defVal;
    }

public:
    MacConfig() = default;

    // [Added] Optional: print a concise summary for monitor/evaluation configs.
    void printMonitorAndEvalSummary() const;

    /**
     * @brief Load configuration from YAML file following the new schema.
     *
     * Schema (abridged):
     *   execution:
     *     datasetToRun: "3DMatch"
     *     paths: { cloudSrc, cloudTgt, cloudSrcKpt, cloudTgtKpt, correspondences, corresIndex, output }
     *     system: { desiredThreads, logLevel, noLogFile }
     *   algorithm:
     *     lidarNoiseModel: { varianceMode, sigmaRho, sigmaTheta, sigmaPhi, NSigma, ... }
     *     graph: { scoreFormula, secondOrderGraph }
     *     clique: { maxTotalNum, maxIterations, maxSize, maxLocalNum }
     *     hypothesis: { maxEstimateNum, localEvaluationMetric }
     *     refinement: { useIcp, instanceEqual, clusterInternalEvaluation }
     *   evaluation:
     *     paths: { groundTruthLabels, groundTruthTransform }
     *     datasets:
     *       "<Name>": { thresholds..., [meshResolution], [*Multiplier] }
     */
    void load(const std::string &filename) {
        YAML::Node root;
        try {
            root = YAML::LoadFile(filename);
        } catch (const YAML::Exception &e) {
            LOG_ERROR("YAML parsing error: " << e.what());
            return;
        }

        // -------------------------- execution --------------------------
        if (root["execution"]) {
            const YAML::Node exec = root["execution"];

            // datasetToRun
            if (exec["datasetToRun"]) currentDatasetName = exec["datasetToRun"].as<std::string>();

            // paths
            if (exec["paths"]) {
                const YAML::Node p = exec["paths"];
                if (p["cloudSrc"]) cloudSrcPath = p["cloudSrc"].as<std::string>();
                if (p["cloudTgt"]) cloudTgtPath = p["cloudTgt"].as<std::string>();
                if (p["cloudSrcKpt"]) cloudSrcKptPath = p["cloudSrcKpt"].as<std::string>();
                if (p["cloudTgtKpt"]) cloudTgtKptPath = p["cloudTgtKpt"].as<std::string>();
                if (p["correspondences"]) corresPath = p["correspondences"].as<std::string>();
                if (p["corresIndex"]) corresIndexPath = p["corresIndex"].as<std::string>();
                if (p["output"]) outputPath = p["output"].as<std::string>();
            }

            // system
            if (exec["system"]) {
                const YAML::Node sys = exec["system"];
                desiredThreads = sys["desiredThreads"] ? sys["desiredThreads"].as<int>(desiredThreads) : desiredThreads;
                if (sys["logLevel"]) logLevel = parseLogLevel(sys["logLevel"].as<std::string>());
                flagNoLogs = sys["noLogFile"] ? sys["noLogFile"].as<bool>(flagNoLogs) : flagNoLogs;

                // Apply console log level
                MacLogger::setLevel(logLevel);
            }
        } else {
            LOG_ERROR("Config file has no execution field!");
            return;
        }

        // -------------------------- algorithm --------------------------
        if (root["algorithm"]) {
            const YAML::Node alg = root["algorithm"];

            // lidarNoiseModel
            if (alg["lidarNoiseModel"]) {
                const YAML::Node ln = alg["lidarNoiseModel"];
                if (ln["varianceMode"]) varianceMode = parseVarianceMode(ln["varianceMode"].as<std::string>());
                sigmaRho = ln["sigmaRho"] ? ln["sigmaRho"].as<float>(sigmaRho) : sigmaRho;
                sigmaTheta = ln["sigmaTheta"] ? ln["sigmaTheta"].as<float>(sigmaTheta) : sigmaTheta;
                sigmaPhi = ln["sigmaPhi"] ? ln["sigmaPhi"].as<float>(sigmaPhi) : sigmaPhi;
                nSigma = ln["NSigma"] ? ln["NSigma"].as<float>(nSigma) : nSigma;

                if (!(nSigma > 0.0f)) {
                    LOG_WARNING(
                        "NSigma must be strictly positive; NSigma=0 would reject all points. Setting to default 1");
                    nSigma = 1.0f;
                }
            }

            // graph
            if (alg["graph"]) {
                const YAML::Node g = alg["graph"];
                if (g["scoreFormula"]) scoreFormula = parseScoreFormula(g["scoreFormula"].as<std::string>());
                flagSecondOrderGraph = g["secondOrderGraph"]
                                           ? g["secondOrderGraph"].as<bool>(flagSecondOrderGraph)
                                           : flagSecondOrderGraph;
            }

            // clique
            if (alg["clique"]) {
                const YAML::Node c = alg["clique"];
                maxTotalCliqueNum = c["maxTotalNum"] ? c["maxTotalNum"].as<int>(maxTotalCliqueNum) : maxTotalCliqueNum;
                maxCliqueIterations = c["maxIterations"]
                                          ? c["maxIterations"].as<int>(maxCliqueIterations)
                                          : maxCliqueIterations;
                maxCliqueSize = c["maxSize"] ? c["maxSize"].as<int>(maxCliqueSize) : maxCliqueSize;
                maxLocalCliqueNum = c["maxLocalNum"] ? c["maxLocalNum"].as<int>(maxLocalCliqueNum) : maxLocalCliqueNum;
            }

            // hypothesis
            if (alg["hypothesis"]) {
                const YAML::Node h = alg["hypothesis"];
                maxEstimateNum = h["maxEstimateNum"] ? h["maxEstimateNum"].as<int>(maxEstimateNum) : maxEstimateNum;
                if (h["localEvaluationMetric"]) metric = h["localEvaluationMetric"].as<std::string>();
            }

            // refinement
            if (alg["refinement"]) {
                const YAML::Node r = alg["refinement"];
                flagUseIcp = r["useIcp"] ? r["useIcp"].as<bool>(flagUseIcp) : flagUseIcp;
                flagInstanceEqual = r["instanceEqual"]
                                        ? r["instanceEqual"].as<bool>(flagInstanceEqual)
                                        : flagInstanceEqual;
                flagClusterInternalEvaluation = r["clusterInternalEvaluation"]
                                                    ? r["clusterInternalEvaluation"].as<bool>(
                                                        flagClusterInternalEvaluation)
                                                    : flagClusterInternalEvaluation;
            }
        } else {
            LOG_WARNING("Config file has no algorithm field, program runs with default parameters");
        }

        // -------------------------- evaluation --------------------------
        if (root["evaluation"]) {
            const YAML::Node eva = root["evaluation"];

            // paths (optional)
            if (eva["paths"]) {
                const YAML::Node ep = eva["paths"];
                if (ep["groundTruthLabels"]) gtLabelPath = ep["groundTruthLabels"].as<std::string>();
                if (ep["groundTruthTransform"]) gtTfPath = ep["groundTruthTransform"].as<std::string>();
            }

            // datasets
            if (eva["datasets"]) {
                for (const auto &kv: eva["datasets"]) {
                    const std::string name = kv.first.as<std::string>();
                    DatasetConfig ds;
                    ds.loadFromYAML(kv.second, name);
                    availableDatasetConfigs_[name] = ds;
                }
            }
        } else {
            LOG_INFO("Config has no evaluation field, program runs without evaluation");
        }

        // If a dataset was requested, activate it now.
        if (!currentDatasetName.empty()) {
            switchToDataset(currentDatasetName);
        }

        // ==============================
        // [Added] evaluation.enabled
        // ==============================
        try {
            if (root["evaluation"] && root["evaluation"]["enabled"]) {
                this->evaluationEnabled = root["evaluation"]["enabled"].as<bool>(true);
            } else {
                this->evaluationEnabled = true; // default
            }
        } catch (const std::exception &e) {
            LOG_WARNING("Failed to parse evaluation.enabled, fallback to true. Error: " << e.what());
            this->evaluationEnabled = true;
        }

        // ==============================
        // [Added] monitor.* (enabled/granularity/metrics/coreDepth/timeTree/summary)
        // ==============================
        try {
            if (root["monitor"]) {
                const auto &m = root["monitor"];

                // monitor.enabled
                if (m["enabled"]) {
                    this->monitor.enabled = m["enabled"].as<bool>(true);
                } else {
                    this->monitor.enabled = true;
                }

                // monitor.granularity
                if (m["granularity"]) {
                    auto gStr = m["granularity"].as<std::string>("CORE");
                    auto gNew = parseGranularity(gStr, MonitorGranularity::CORE);
                    this->monitor.granularity = gNew;
                    // We have quite strong case check, those the redundant check can be ignored.
                    // Same for the other fields
                    // if (!(gNew == MonitorGranularity::CORE || gNew == MonitorGranularity::ALL)) {
                    //     LOG_WARNING("Invalid monitor.granularity=" << gStr.c_str() << ", fallback to CORE.");
                    //     this->monitor.granularity = MonitorGranularity::CORE;
                    // }
                } else {
                    this->monitor.granularity = MonitorGranularity::CORE;
                }

                // monitor.metrics
                if (m["metrics"]) {
                    auto ms = m["metrics"].as<std::string>("AUTO");
                    auto mv = parseMetrics(ms, MonitorMetrics::AUTO);
                    this->monitor.metrics = mv;
                    // if (!(mv == MonitorMetrics::OFF || mv == MonitorMetrics::AUTO || mv == MonitorMetrics::ALL)) {
                    //     LOG_WARNING("Invalid monitor.metrics=" << ms.c_str() <<", fallback to AUTO.");
                    //     this->monitor.metrics = MonitorMetrics::AUTO;
                    // }
                } else {
                    this->monitor.metrics = MonitorMetrics::AUTO;
                }

                // monitor.coreDepth (only used when granularity==CORE)
                if (m["coreDepth"]) {
                    this->monitor.coreDepth = std::max(0, m["coreDepth"].as<int>(2));
                } else {
                    this->monitor.coreDepth = 2;
                }

                // monitor.timeTree
                if (m["timeTree"]) {
                    const auto &t = m["timeTree"];

                    if (t["enabled"]) {
                        this->monitor.timeTree.enabled = t["enabled"].as<bool>(true);
                    } else {
                        this->monitor.timeTree.enabled = true;
                    }
                    if (t["charset"]) {
                        auto cs = t["charset"].as<std::string>("unicode");
                        this->monitor.timeTree.charset = parseCharset(cs, MonitorTreeCharset::UNICODE);
                        // if (!(this->monitor.timeTree.charset == MonitorTreeCharset::UNICODE ||
                        //       this->monitor.timeTree.charset == MonitorTreeCharset::ASCII)) {
                        //     LOG_WARNING("Invalid monitor.timeTree.charset, fallback to 'unicode'.");
                        //     this->monitor.timeTree.charset = MonitorTreeCharset::UNICODE;
                        // }
                    } else {
                        this->monitor.timeTree.charset = MonitorTreeCharset::UNICODE;
                    }
                    if (t["showMemory"]) {
                        this->monitor.timeTree.showMemory = t["showMemory"].as<bool>(true);
                    } else {
                        this->monitor.timeTree.showMemory = true;
                    }
                    if (t["showNotes"]) {
                        this->monitor.timeTree.showNotes = t["showNotes"].as<bool>(true);
                    } else {
                        this->monitor.timeTree.showNotes = true;
                    }
                    if (t["maxDepth"]) {
                        this->monitor.timeTree.maxDepth = t["maxDepth"].as<int>(-1);
                    } else {
                        this->monitor.timeTree.maxDepth = -1;
                    }
                } else {
                    this->monitor.timeTree.enabled = true;
                    this->monitor.timeTree.charset = MonitorTreeCharset::UNICODE;
                    this->monitor.timeTree.showMemory = true;
                    this->monitor.timeTree.showNotes = true;
                    this->monitor.timeTree.maxDepth = -1;
                }

                // [Added] monitor.summary.enabled
                if (m["summary"] && m["summary"]["enabled"]) {
                    this->monitor.summary.enabled = m["summary"]["enabled"].as<bool>(true);
                } else {
                    this->monitor.summary.enabled = true; // default
                }
            } else {
                // Entire monitor section missing: set defaults
                this->monitor.enabled = true;
                this->monitor.granularity = MonitorGranularity::CORE;
                this->monitor.metrics = MonitorMetrics::AUTO;
                this->monitor.coreDepth = 2;
                this->monitor.timeTree.enabled = true;
                this->monitor.timeTree.charset = MonitorTreeCharset::UNICODE;
                this->monitor.timeTree.showMemory = true;
                this->monitor.timeTree.showNotes = true;
                this->monitor.timeTree.maxDepth = -1;
                this->monitor.summary.enabled = true;
            }
        } catch (const std::exception &e) {
            LOG_WARNING("Failed to parse 'monitor' config, fall back to defaults. Error: " << e.what());
            this->monitor.enabled = true;
            this->monitor.granularity = MonitorGranularity::CORE;
            this->monitor.metrics = MonitorMetrics::AUTO;
            this->monitor.coreDepth = 2;
            this->monitor.timeTree.enabled = true;
            this->monitor.timeTree.charset = MonitorTreeCharset::UNICODE;
            this->monitor.timeTree.showMemory = true;
            this->monitor.timeTree.showNotes = true;
            this->monitor.timeTree.maxDepth = -1;
            this->monitor.summary.enabled = true;
        }
    }

    /**
     * @brief Called by the pipeline after cloud resolution is computed.
     * This allows U3M-style multiplier thresholds to be resolved without special
     * case logic downstream.
     */
    void setRuntimeMeshResolutionForCurrentDataset(const float resM) {
        if (const auto it = availableDatasetConfigs_.find(currentDatasetName); it != availableDatasetConfigs_.end()) {
            it->second.setRuntimeMeshResolution(resM);
            currentDatasetConfig = it->second;
            // Keep "threshold" aligned with the dataset's effective clustering distance.
            threshold = currentDatasetConfig.getActualClusteringDistanceThreshold();
        }
    }

    /**
     * @brief Switch to a given dataset; throws if not present in YAML.
     * Also refreshes "threshold" to the current dataset's effective clustering distance.
     */
    void switchToDataset(const std::string &datasetName) {
        const auto it = availableDatasetConfigs_.find(datasetName);
        if (it == availableDatasetConfigs_.end()) {
            LOG_ERROR("No configuration found for dataset: " + datasetName);
            return;
        }
        currentDatasetName = datasetName;
        currentDatasetConfig = it->second;

        // Keep compatibility: "threshold" is used as a generic distance cutoff by some routines.
        threshold = currentDatasetConfig.getActualClusteringDistanceThreshold();
    }

    /**
     * @brief Quick helper used by a few places to check U3M-specific paths.
     */
    [[nodiscard]] bool isU3MDataset() const {
        // Compare case-insensitively with "U3M"
        if (currentDatasetName.size() != 3) return false;
        std::string s = currentDatasetName;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
        return s == "U3M";
    }

    /**
     * @brief Get active dataset config (const).
     */
    [[nodiscard]] const DatasetConfig &getCurrentDatasetConfig() const {
        return currentDatasetConfig;
    }

    /**
     * @brief Validate a few essential fields; returns false if any is missing.
     */
    [[nodiscard]] bool validate() const {
        if (cloudSrcPath.empty() || cloudTgtPath.empty() ||
            cloudSrcKptPath.empty() || cloudTgtKptPath.empty() ||
            corresPath.empty() || corresIndexPath.empty() ||
            outputPath.empty()) {
            return false;
        }
        if (currentDatasetName.empty()) return false;
        return true;
    }

    /**
     * @brief List dataset names found in YAML.
     */
    [[nodiscard]] std::vector<std::string> getAvailableDatasets() const {
        std::vector<std::string> names;
        names.reserve(availableDatasetConfigs_.size());
        for (const auto &[fst, snd]: availableDatasetConfigs_) names.push_back(fst);
        return names;
    }
};
