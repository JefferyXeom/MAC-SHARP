//
// Created by Jeffery_Xeom on 2025/8/24.
//

#include <cmath>       // sqrt, exp
#include <fstream>     // save matrix / debug
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "CommonTypes.hpp"
#include "MacTimer.hpp"
#include "MacUtils.hpp"  // Logs, Timer, ScoreFormula
#include "MacGraph.hpp"


// ---- MacGraph build private static helpers ----


/**
 * @brief A lightweight dynamic threshold heuristic
 * @details
 *  - For QUADRATIC_FALLOFF, we optionally drop very small weights.
 *  - Heuristic: threshold increases mildly with |dis| relative to alpha.
 *  - base is an additive floor; set to 0 by default to be conservative.
 */
// TODO: Complete this critical function!!
float MacGraph::dynamicThreshold(const float dis, const float alpha, const float base) {
    // Normalize deviation by alpha (avoid div-by-zero)
    const float a = (alpha > 1e-6f) ? (std::fabs(dis) / alpha) : std::fabs(dis);
    // Map to [0, 0.2] approximately; you can tune these constants
    const float t = std::min(0.2f, 0.05f + 0.05f * a);
    return base + t;
}


// ---- MacGraph calculateGraphThreshold private static helpers ----
/**
 * @brief 使用大津算法(Otsu's Method)自动计算一组分数的最佳分割阈值。
 * * @param scores 包含所有分数的 std::vector<float>。注意：此函数会为了效率而修改输入vector的内容。
 * @return float 计算出的最佳阈值。
 */
float MacGraph::otsuThresh(std::vector<float> scores) {
    // 如果分数数量过少，无法进行有意义的分割，直接返回0
    const int scoreSize = static_cast<int>(scores.size());
    if (scoreSize < 2) {
        return 0.0f;
    }
    // --- 步骤 1: 数据准备与统计 ---
    sort(scores.begin(), scores.end());
    const float maxScoreValue = scores[scoreSize - 1];
    const float minScoreValue = scores[0];
    // 如果所有分数都一样，阈值设置在中间即可
    if (maxScoreValue == minScoreValue) {
        return minScoreValue;
    }
    const double scoreSum = std::accumulate(scores.begin(), scores.end(), 0.0);

    // --- 步骤 2: 构建直方图 ---
    constexpr int numBins = 100; // 将分数范围均匀划分为100个“桶”
    std::vector scoreHist(numBins, 0); // 每个桶里有多少个分数
    std::vector<float> scoreSumHist(numBins, 0.0); // 每个桶里分数的总和

    // 遍历所有分数，填充直方图
    const float quantStep = (maxScoreValue - minScoreValue) / numBins;
    for (const float score: scores) {
        int binIdx = static_cast<int>((score - minScoreValue) / quantStep);
        // 边界检查，确保索引不会越界
        if (binIdx >= numBins) binIdx = numBins - 1;
        scoreHist[binIdx]++;
        scoreSumHist[binIdx] += score;
    }
    // --- 步骤 3: 遍历所有可能的阈值，寻找最大类间方差 ---
    float maxVariance = -1.0;
    float optimalThresh = (maxScoreValue + minScoreValue) / 2; //default value
    // 变量用于迭代计算
    int backgroundTotalCount = 0; // 背景组（低分区）的总数量
    float backgroundScoreSum = 0.0f; // 背景组的总分数

    // 遍历 numBins-1 个可能的分割点
    for (int i = 0; i < numBins; i++) {
        // 更新背景组的统计数据
        backgroundTotalCount += scoreHist[i];
        backgroundScoreSum += scoreSumHist[i];

        // 如果背景组为空，则无法分割，继续
        if (backgroundTotalCount == 0) continue;

        // 前景组（高分区）的总数量
        const int foregroundTotalCount = scoreSize - backgroundTotalCount;
        // 如果前景组为空，说明所有点都在背景组，结束遍历
        if (foregroundTotalCount == 0) break;

        // 计算背景组和前景组的平均分数
        const float backgroundMean = backgroundScoreSum / backgroundTotalCount;
        const float foregroundMean = (scoreSum - backgroundScoreSum) / foregroundTotalCount;
        // 计算类间方差 (Between-class variance)
        // 这是大津算法的核心公式
        if (const double betweenClassVariance = static_cast<double>(backgroundTotalCount) * foregroundTotalCount *
                                                std::pow(backgroundMean - foregroundMean, 2);
            betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            // 更新最佳阈值。阈值是当前“桶”的右边界
            optimalThresh = minScoreValue + (i + 1) * quantStep;
        }
    }
    return optimalThresh;
}


// ---- MacGraph findMaximalClique private core methods ----

/**
 * @brief [私有] 步骤1：初始化 igraph 矩阵并应用滤波逻辑
 */
void MacGraph::initializeIgraphMatrixWithFilter(igraph_matrix_t &outMatrix) {
    igraph_matrix_init(&outMatrix, graphEigen_.rows(), graphEigen_.cols());

    // Filtering, reduce the graph size
    // Note that the original mac++ use this to filter the graph on kitti dataset. We ignore it here
    if (graphThreshold_ > 2.9 && data_.totalCorresNum > 50) {
        LOG_INFO("Graph filtered due to high threshold and correspondence count.");
    } else {
        // 复制数据
        for (int i = 0; i < graphEigen_.rows(); ++i) {
            for (int j = 0; j < graphEigen_.cols(); ++j) {
                igraph_matrix_set(&outMatrix, i, j, graphEigen_(i, j));
            }
        }
    }
}

/**
 * @brief [私有] 步骤2：从 igraph_matrix_t 创建 igraph_t 对象
 */
void MacGraph::buildIgraphObjectFromMatrix(const igraph_matrix_t &igraphMatrix) {
    // 如果之前已存在 igraph 对象，先销毁，确保状态干净
    if (igraphInitialized_) {
        igraph_destroy(&graphIgraph_);
        igraphInitialized_ = false;
    }

    igraph_set_attribute_table(&igraph_cattribute_table);
    // 从矩阵创建 igraph 对象

    // temporary weight vector, not used but required by the function signature
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    if (const igraph_error_t status = igraph_weighted_adjacency(&graphIgraph_, &igraphMatrix, IGRAPH_ADJ_UNDIRECTED,
                                                                &weight, IGRAPH_NO_LOOPS); status == IGRAPH_SUCCESS) {
        this->igraphInitialized_ = true; // 标记成功
    } else {
        LOG_ERROR("Failed to build igraph_t from matrix. Error: " << igraph_strerror(status));
    }
}

/**
 * @brief [私有] 步骤3：运行核心的循环来查找最大团
 */
void MacGraph::runCliqueFindingLoop() {
    LOG_INFO("Find maximal clique");
    // 清空上一次的团结果
    igraph_vector_int_list_clear(&cliques_);

    int minCliqueSize = 3; // Minimum size of the clique to be considered, 3 is the minimum number to creat a triangle
    int maxCliqueSize = 0; // Maximum size of the clique, 0 is no limit.
    bool recalculateFlag = true; // Flag to indicate whether to recalculate the cliques
    int iterNum = 1;

    while (recalculateFlag) {
        const igraph_error_t error_code =
                igraph_maximal_cliques(&graphIgraph_, &cliques_, minCliqueSize, maxCliqueSize);
        totalCliqueNum_ = static_cast<int>(igraph_vector_int_list_size(&cliques_));

        if (totalCliqueNum_ > config_.maxTotalCliqueNum && iterNum <= config_.maxCliqueIterations) {
            maxCliqueSize = config_.maxCliqueSize;
            minCliqueSize += iterNum;
            iterNum++;
            igraph_vector_int_list_destroy(&cliques_);
            igraph_vector_int_list_init(&cliques_, 0);
            LOG_INFO("Number of cliques (" << totalCliqueNum_ << ") is too large, recalculating with min="
                << minCliqueSize << " and max=" << maxCliqueSize);
        } else {
            recalculateFlag = false;
        }
        // 3. **必须**检查返回值
        if (error_code != IGRAPH_SUCCESS) {
            LOG_ERROR("Error finding maximal cliques: " << igraph_strerror(error_code));
            break;
        }
    }

    // --- 可靠性检查 3：报告最终结果 ---
    if (totalCliqueNum_ == 0) {
        LOG_WARNING("No cliques found in the graph. The graph might be too sparse.");
    } else {
        LOG_INFO("Found " << totalCliqueNum_ << " maximal cliques.");
    }
}

// ---- Free igraph ----

void MacGraph::freeIgraph() {
    if (igraphInitialized_) {
        igraph_destroy(&graphIgraph_);
        igraphInitialized_ = false;
    }
}

// ---- Ctor / Dtor & resource management ----

MacGraph::MacGraph(MacData &data, const MacConfig &config)
    : data_(data),
      config_(config),
      graphIgraph_(),
      cliques_(),
      igraphInitialized_(false),
      totalCliqueNum_(0) {
    // Initialize clique list container (0 capacity, it grows on demand)
    igraph_vector_int_list_init(&cliques_, 0);
}

MacGraph::~MacGraph() {
    freeIgraph(); // Release igraph first
    igraph_vector_int_list_destroy(&cliques_); // Then clique container
}

// Keep your save_matrix function (enhanced to dump the matrix for debugging)
void save_matrix(const Eigen::MatrixXf &mat, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << YELLOW << "Cannot open file: " << filename << std::endl;
        return;
    }
    // Write a simple ASCII matrix: rows cols then data row by row
    file << mat.rows() << " " << mat.cols() << "\n";
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j + 1 < mat.cols()) file << " ";
        }
        file << "\n";
    }
    file.close();
}

// ---- Build graph ----

/**
 * @brief Build the first-order (and optional second-order) compatibility graph
 * @details
 *  - Edge weight between correspondences i,j uses distance consistency:
 *    Δd = | d(src_i,src_j) - d(tgt_i, tgt_j) |
 *    - GAUSSIAN_KERNEL: w = exp( - (Δd^2) / (2 alpha^2) ), with 0.8 hard threshold
 *    - QUADRATIC_FALLOFF: w = 1 - (Δd/alpha)^2, with dynamicThreshold cutoff
 */
void MacGraph::build() {
    Timer timerConstGraph;
    const int n = static_cast<int>(data_.corres.size());
    if (n <= 0) {
        LOG_WARNING("No correspondences, graph is empty.");
        graphEigen_.resize(0, 0);
        return;
    }

    timerConstGraph.startTiming("construct graph: initialize");
    graphEigen_ = Eigen::MatrixXf::Zero(n, n);

    // alpha = 10 * resolution (same as MAC++)
    const float alphaDis = std::max(1e-6f, 10.0f * data_.cloudResolution);
    const float gamma = -1.0f / (2.0f * alphaDis * alphaDis);
    int localTotalEdges = 0; // Count edges after thresholding
    timerConstGraph.endTiming();

    // -------- First order graph --------
    timerConstGraph.startTiming("construct graph: first order graph");
    switch (config_.scoreFormula) {
        case ScoreFormula::GAUSSIAN_KERNEL: {
            // Parallelize outer loop if OpenMP is available
#pragma omp parallel for schedule(static) default(none) shared(n, gamma) reduction(+:localTotalEdges)
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    const CorresStruct &c1 = data_.corres[i];
                    const CorresStruct &c2 = data_.corres[j];
                    const float src_dis = getDistance(c1.src, c2.src);
                    const float tgt_dis = getDistance(c1.tgt, c2.tgt);
                    const float diff = src_dis - tgt_dis;
                    float w = std::exp(diff * diff * gamma);
                    // Hard cutoff at 0.8 (consistent with original implementation)
                    // kitti is 0.9 in the original code.
                    if (w < 0.8f) w = 0.0f;
                    graphEigen_(i, j) = w;
                    graphEigen_(j, i) = w;
                    if (w >= 0.8f) localTotalEdges++;
                }
            }
            break;
        }
        case ScoreFormula::QUADRATIC_FALLOFF: {
#pragma omp parallel for schedule(static) default(none) shared(n, alphaDis) reduction(+:localTotalEdges)
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    // Before the dynamic threshold function is ready, do NOT use this option!
                    const CorresStruct &c1 = data_.corres[i];
                    const CorresStruct &c2 = data_.corres[j];
                    const float src_dis = getDistance(c1.src, c2.src);
                    const float tgt_dis = getDistance(c1.tgt, c2.tgt);
                    const float diff = src_dis - tgt_dis;
                    float w = 1.0f - (diff * diff) / (alphaDis * alphaDis);
                    // Drop tiny/negative weights using a mild dynamic threshold
                    if (w < dynamicThreshold(diff, alphaDis, 0.0f)) w = 0.0f;
                    graphEigen_(i, j) = w;
                    graphEigen_(j, i) = w;
                    if (w > 0.0f) localTotalEdges++;
                }
            }
            break;
        }
        default:
            LOG_WARNING("Unknown score formula. Graph will remain zero.");
            break;
    }
    timerConstGraph.endTiming();
    // Symmetry check
    if (const Eigen::MatrixXf tmp = graphEigen_ - graphEigen_.transpose(); tmp.norm() != 0) {
        LOG_ERROR("First order graph is not symmetric! Please check parallel reduction or float precision.");
    }
    LOG_INFO("First order graph has been constructed, total edges: " << localTotalEdges);

    // -------- Optional second-order graph --------
    // Save the graph for debug, comment for normal use
    // save_matrix(graphEigen_, config_.outputPath + "/graph_matrix.txt");
    // Second order graphing is time-consuming, size 6000 will use up to 2s
    if (config_.flagSecondOrderGraph) {
        timerConstGraph.startTiming("construct graph: second order graph");
        // A ∘ (A * A)
        graphEigen_ = graphEigen_.cwiseProduct(graphEigen_ * graphEigen_);
        // Ensure symmetry explicitly
#pragma omp parallel for schedule(static)
        for (int i = 0; i < graphEigen_.rows(); ++i) {
            for (int j = i + 1; j < graphEigen_.cols(); ++j) {
                graphEigen_(j, i) = graphEigen_(i, j);
            }
        }
        timerConstGraph.endTiming();

        if (const Eigen::MatrixXf tmp2 = graphEigen_ - graphEigen_.transpose(); tmp2.norm() != 0) {
            LOG_ERROR("Second order graph is not symmetric! Please check operations.");
        }

        // Count non-zero edges (upper triangle)
        int nonZeroCount = 0;
#pragma omp parallel for schedule(static) default(none) shared(nonZeroCount)
        for (int i = 0; i < graphEigen_.rows(); ++i) {
            for (int j = i + 1; j < graphEigen_.cols(); ++j) {
                if (graphEigen_(i, j) != 0.0f) {
                    nonZeroCount++;
                }
            }
        }
        LOG_INFO("Second order graph has been constructed, total edges: " << nonZeroCount);
    }

    // Check whether the graph is all 0.
    // If using dynamic threshold, it generally will not happen.
    if (graphEigen_.norm() == 0) {
        LOG_ERROR("Graph is disconnected. You may need to check the compatibility threshold!");
    }

    // Optional: dump matrix for debugging
    // save_matrix(graphEigen_, config_.outputPath + "/graph_matrix.txt");
}

// ---- Degree & node triangle weights ----

/**
 * @brief Compute degree for each vertex.
 * @note O(N^2). Requires a valid graphEigen_.
 */
void MacGraph::computeGraphDegree() {
    const int n = static_cast<int>(graphEigen_.rows());
    if (n == 0 || graphEigen_.cols() != n) {
        LOG_ERROR("Invalid graph matrix for degree computation.");
        graphVertex_.clear();
        return;
    }
    graphVertex_.resize(n);

#pragma omp parallel for schedule(static) default(none) shared(graphVertex_, n)
    for (int i = 0; i < n; ++i) {
        int degree = 0;
        std::vector<int> neighborIndices;
        int neighborCorrectMatchNum = 0;
        const float *row = graphEigen_.row(i).data();
        for (int j = 0; j < n; ++j) {
            if (i != j && row[j] != 0.0f) {
                degree++;
                neighborIndices.push_back(j);
                // ---------------------------- Evaluation part ----------------------------
                if (!data_.gtInlierLabels.empty() && data_.gtInlierLabels[j]) {
                    neighborCorrectMatchNum++;
                }
                // -------------------------------------------------------------------------
            }
        }
        graphVertex_[i].degree = degree;
        graphVertex_[i].neighborIndices = std::move(neighborIndices);
        graphVertex_[i].neighborCorrectMatchNum = neighborCorrectMatchNum;
    }
}

// ---- Triangle weights ----

/**
 * @brief Compute per-vertex triangle weight: sum of w(i,j)*w(j,k)*w(k,i) over all j<k and i!=j!=k
 * @note O(N^3); consider pruning or sparse representation for very large graphs.
 */
void MacGraph::calculateTriangularWeights() {
    Timer timerTriWeights;
    timerTriWeights.startTiming("calculate triangular weights");
    const int n = static_cast<int>(data_.corres.size());
    if (graphEigen_.rows() != n || graphEigen_.cols() != n) {
        LOG_ERROR("Graph matrix dimension mismatch. Call build() first.");
        return;
    }
    totalTriangleWeightSum_ = 0.0f;
    totalPossibleTriangleNum_ = 0;

#pragma omp parallel for schedule(static) default(none) shared(n, graphVertex_, graphEigen_) reduction(+: totalTriangleWeightSum_, totalPossibleTriangleNum_)
    for (int i = 0; i < n; ++i) {
        if (const int neighborSize = graphVertex_[i].degree; neighborSize > 1) {
            float acc = 0.0f; // index i vertex neighbor triangle weight summation
            for (int j = 0; j < neighborSize; ++j) {
                const int neighborIndex1 = graphVertex_[i].neighborIndices[j];
                for (int k = j + 1; k < neighborSize; ++k) {
                    if (const int neighborIndex2 = graphVertex_[i].neighborIndices[k]; graphEigen_(neighborIndex1,
                        neighborIndex2) != 0.0f) {
                        acc += std::pow(
                            graphEigen_(i, neighborIndex2) * graphEigen_(i, neighborIndex2) * graphEigen_(
                                neighborIndex1, neighborIndex2), 1.0f / 3.0f);
                    }
                }
            }
            const float currentPossibleTriangleNum =
                    static_cast<int>(neighborSize) * (static_cast<float>(neighborSize) - 1) / 2.0f;
            totalTriangleWeightSum_ += acc;
            totalPossibleTriangleNum_ += currentPossibleTriangleNum;
            graphVertex_[i].triWeight = acc / currentPossibleTriangleNum;
        }
    }

    // Write back to data_.corres for hypothesis and weighted SVD later
    for (int i = 0; i < n; ++i) {
        data_.corres[i].corresScore = graphVertex_[i].triWeight;
    }
    timerTriWeights.endTiming();
}

/**
 * @brief Calculates and returns a threshold for graph pruning.
 * @details This threshold is determined by the minimum of the OTSU threshold of triangle weights,
 *          the average vertex weight, and the average triangle weight.
 * @return The calculated graph threshold.
 */
void MacGraph::calculateGraphThreshold() {
    Timer timerGraphThreshold;
    timerGraphThreshold.startTiming("calculate graph threshold");
    graphThreshold_ = 0.0f;

    // 1. 计算每个顶点的平均权重
    float averageVertexWeight = 0.0f;
    for (const VertexStruct &vertex: graphVertex_) {
        averageVertexWeight += vertex.triWeight;
    }
    averageVertexWeight /= static_cast<float>(data_.corres.size());

    // 2. 计算每个三角形的平均权重
    float averageTriangleWeight = 0.0f;
    if (totalPossibleTriangleNum_ > 0) {
        averageTriangleWeight = totalTriangleWeightSum_ / static_cast<float>(totalPossibleTriangleNum_);
    }

    // 3. 使用OTSU方法计算阈值
    float otsu = 0.0f;
    std::vector<float> triangularWeightScores;
    triangularWeightScores.reserve(data_.corres.size());
    for (const VertexStruct &vertex: graphVertex_) {
        triangularWeightScores.push_back(vertex.triWeight);
    }
    otsu = otsuThresh(triangularWeightScores);

    // 4. 取三者中的最小值作为最终阈值，这样图会更稀疏
    graphThreshold_ = std::min({otsu, averageVertexWeight, averageTriangleWeight});

    LOG_INFO("Graph threshold calculation: " << graphThreshold_ << " -> min(otsu: " << otsu
        << ", avg_vertex: " << averageVertexWeight << ", avg_triangle: " << averageTriangleWeight << ")");
    timerGraphThreshold.endTiming();
}

// ---- Maximal cliques ----

/**
 * @brief [主入口] 查找最大团的协调者方法
 */
void MacGraph::findMaximalCliques() {
    LOG_INFO("Clique finding stage started.");

    Timer timerFindMaximalCliques;
    // --- 可靠性检查 1：确保图已构建 ---
    if (graphEigen_.size() == 0) {
        LOG_ERROR("Graph matrix is empty. Please call build() first.");
        return;
    }
    timerFindMaximalCliques.startTiming("find maximal cliques: initialize igraph");
    // --- 步骤 1: 初始化 igraph 矩阵并应用滤波 ---
    igraph_matrix_t igraphMatrix; // Create a local matrix for igraph
    initializeIgraphMatrixWithFilter(igraphMatrix);
    timerFindMaximalCliques.endTiming();

    // --- 步骤 2: 从矩阵构建 igraph 对象 ---
    timerFindMaximalCliques.startTiming("find maximal cliques: build igraph");
    buildIgraphObjectFromMatrix(igraphMatrix);
    // igraphMatrix 在上一步函数结束后已被销毁，资源得到管理
    timerFindMaximalCliques.endTiming();

    // --- 可靠性检查 2：确保 igraph 对象已成功创建 ---
    if (!igraphInitialized_) {
        LOG_ERROR("igraph_t object failed to initialize. Aborting clique search.");
        return;
    }

    // --- 步骤 3: 运行核心的循环来查找团 ---
    timerFindMaximalCliques.startTiming("find maximal cliques: find maximal cliques");
    runCliqueFindingLoop();
    timerFindMaximalCliques.endTiming();

    LOG_INFO("Number of total cliques: " << totalCliqueNum_);
    data_.totalCliqueNum = totalCliqueNum_;
}
