//
// Created by Jeffery_Xeom on 2025/8/24.
//
#pragma once

#include <Eigen/Dense>
#include <igraph.h>
#include <vector>  // Added: for std::vector
#include "MacData.hpp" // 依赖于我们第一阶段的成果

class MacGraph {
// private:
    // --- Data and config ---
    MacData& data_;                   // Reference to data (no copy)
    const MacConfig& config_;         // Reference to config (no copy)

    // --- Graph storage ---
    Eigen::MatrixXf graphEigen_;       // Symmetric adjacency matrix (float)
    igraph_t graphIgraph_;             // igraph graph object
    igraph_vector_int_list_t cliques_; // Maximal cliques result container

    bool igraphInitialized_;        // Whether igraph_ has been created
    int totalCliqueNum_;            // Number of maximal cliques found

    float totalTriangleWeightSum_ = 0.0f; // Sum of all triangle weights
    int totalPossibleTriangleNum_ = 0; // Total number of possible triangles

    // --- graph vertex information (including degree) ---
    // In this vector variable, index are strictly aligned with data_.corres
    // Therefore the field vertexIndex in VertexStruct is not used
    std::vector<VertexStruct> graphVertex_;

    // threshold
    float graphThreshold_ = 0.0f;

    // --- Helpers ---
    // Lightweight dynamic threshold heuristic for QUADRATIC_FALLOFF
    static float dynamicThreshold(float dis, float alpha, float base = 0.0f);

static float calculateVariance(const PrecomputedInfo& p1,
                               const PrecomputedInfo& p2,
                               const float varRho,
                               const float varTheta,
                               const float varPhi);
    static float otsuThresh(std::vector<float> scores);

    // --- findMaximalClique Core methods ---
    void initializeIgraphMatrixWithFilter(igraph_matrix_t& outMatrix);
    void buildIgraphObjectFromMatrix(const igraph_matrix_t& igraphMatrix);
    void runCliqueFindingLoop();

    // Safe destroy igraph_ if created
    void freeIgraph();


public:
    /**
     * @brief 构造函数，传入数据和配置
     * @details Borrow references; do not copy heavy data.
     */
    MacGraph(MacData& data, const MacConfig& config);

    /**
     * @brief 析构函数，安全释放 igraph 内存
     */
    ~MacGraph();

// Heavy data copy is not allowed
    MacGraph(const MacGraph&) = delete;
    MacGraph& operator=(const MacGraph&) = delete;

    /**
     * @brief 构建兼容性图的邻接矩阵
     * @details
     *  - 边权采用匹配间几何一致性的核函数：
     *    GAUSSIAN_KERNEL: exp(-(Δd)^2 / (2 alpha^2))
     *    QUADRATIC_FALLOFF: 1 - (Δd/alpha)^2（启用 dynamicThreshold 前请谨慎）
     *  - alpha = 10 * data_.cloudResolution
     *  - 当 config_.flagSecondOrderGraph 为 true 时，执行二阶图 graph ∘ (graph * graph)
     */
    void build();

    /**
     * @brief Compute graph degree for each correspondence (vertex).
     * @note Result cached in degrees_.
     */
    void computeGraphDegree();

    /**
     * @brief Compute triangle weights per correspondence and cache them
     * @note If your CorresStruct has a score-like member, you can write back.
     */
    void calculateTriangularWeights();

    void calculateGraphThreshold();

    /**
     * @brief Find all maximal cliques in the current graph using igraph
     */
    void findMaximalCliques();


    // --- Getters ---
    Eigen::MatrixXf &getNSetGraphMatrix() { return graphEigen_; }
    const igraph_vector_int_list_t* getCliques() const { return &cliques_; }
    int getCliqueCount() const { return totalCliqueNum_; }

    const Eigen::MatrixXf &getGraphEigen() const { return graphEigen_; }
    const std::vector<VertexStruct>& getVertex() const { return graphVertex_; }

};