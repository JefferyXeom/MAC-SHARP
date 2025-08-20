//// For input/output operations and system call wrappers
#include <iostream>
#include <filesystem>

//// For string operations
#include <string>

//// For exit function
#include <cstdlib>

//// For timing
#include <chrono>

// for PCL
#include <pcl/point_types.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
// #include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

// Windows system api
// #include <__msvc_filebuf.hpp>
#include <cblas.h>
#include <process.h>

#include <yaml-cpp/yaml.h>

//
#include "MAC_SHARP.hpp"
#include "MAC_utils.hpp"
#include "config_loader.hpp"


// TODO: 之后把整个项目包装成一个类
/**
 * @brief 主要的点云配准函数
 *
 * 该函数实现了基于最大团搜索的点云配准算法(MAC-SHARP)。
 * 通过构建兼容性图、搜索最大团、聚类变换矩阵等步骤实现鲁棒的点云配准。
 *
 * @param macConfig 配置对象，包含算法执行所需的所有参数设置
 * @param result MAC算法结果结构体的引用，用于存储所有输出参数：
 *               - RE: 旋转误差
 *               - TE: 平移误差
 *               - correctEstNum: 正确估计的对应关系数量
 *               - gtInlierNum: Ground Truth中的内点数量
 *               - timeEpoch: 算法执行时间
 *               - predicatedInlier: 预测内点比率向量
 * @return bool 配准是否成功，true表示成功，false表示失败
 */
bool registration(const MACConfig &macConfig, MACResult &result) {
    //
    // pcl::PointCloud<pcl::Normal>::Ptr normalSrc(new pcl::PointCloud<pcl::Normal>); // normal vector
    // pcl::PointCloud<pcl::Normal>::Ptr normalTgt(new pcl::PointCloud<pcl::Normal>); // normal vector

    // PointCloudPtr rawSrc(new pcl::PointCloud<pcl::PointXYZ>); // may not be used
    // PointCloudPtr rawTgt(new pcl::PointCloud<pcl::PointXYZ>);
    // float rawSrcResolution = 0.0f;
    // float rawTgtResolution = 0.0f;


    ////////////////////////////////
    /// System setting up
    ////////////////////////////////
    // Setting up threads number for OpenBLAS and OpenMP
    settingThreads(macConfig.desiredThreads); // Set up the threads for OpenBLAS and OpenMP

    ////////////////////////////////////////////////////////////////
    ///
    /// Prepare data
    ///
    ////////////////////////////////////////////////////////////////

    // Load source and target point clouds
    PointCloudPtr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>); // source point cloud
    PointCloudPtr cloudTgt(new pcl::PointCloud<pcl::PointXYZ>); // target point cloud
    PointCloudPtr cloudSrcKpts(new pcl::PointCloud<pcl::PointXYZ>); // source point cloud keypoints
    PointCloudPtr cloudTgtKpts(new pcl::PointCloud<pcl::PointXYZ>); // target point cloud keypoints

    std::vector<CorresStruct> corresOriginal; // vector to store the original correspondences
    float cloudResolution = 0.0f; // Initialize resolution


    // ---------------------------- Evaluation part ----------------------------
    std::vector<int> gtCorres; // ground truth correspondences
    Eigen::Matrix4f gtMat; // Ground truth transformation matrix
    result.gtInlierNum = 0; // Initialize inlier number
    int successNum = 0; // Number of successful registrations

    ////////////////////////////////
    /// 使用配置化的数据集参数替代硬编码值
    /// 从配置文件中获取数据集特定的评估阈值
    ////////////////////////////////
    // Still under development
    // 获取当前数据集的配置信息
    const DatasetConfig &currentDatasetConfig = macConfig.getCurrentDatasetConfig();
    bool isU3M = macConfig.isU3MDataset();

    // 根据数据集配置设置评估阈值，替代之前的硬编码方式
    float REEvaThresh = currentDatasetConfig.rotationErrorThreshold;
    float TEEvaThresh = currentDatasetConfig.translationErrorThreshold;

    // 内点评估阈值：对于U3M数据集使用分辨率倍数，其他数据集使用固定值
    float inlierEvaThresh;
    if (isU3M) {
        inlierEvaThresh = currentDatasetConfig.getActualInlierThreshold(cloudResolution, true);
    } else {
        inlierEvaThresh = currentDatasetConfig.inlierEvaluationThreshold;
    }
    // -------------------------------------------------------------------------

    // std::vector<std::pair<int, std::vector<int>>> matches; // one2k_match

    // Load point clouds, correspondences, and ground truth data
    loadData(macConfig, cloudSrc, cloudTgt, cloudSrcKpts, cloudTgtKpts, corresOriginal, gtCorres, gtMat,
             result.gtInlierNum, cloudResolution);

    ////////////////////////////////////////////////////////////////
    ///
    /// Graph construction
    ///
    ////////////////////////////////////////////////////////////////

    // NOTE: we do not consider the outer loop of the registration process, which is used to
    // repeat the registration process for multiple iterations.
    timing(0); // Start timing
    Eigen::Matrix graphEigen = graphConstruction(corresOriginal, cloudResolution, macConfig.flagSecondOrderGraph,
                                                 macConfig.scoreFormula);
    timing(1); // End timing
    std::cout << "Graph has been constructed, time elapsed: " << std::endl; // TODO: complete the timing log logics

    // Check whether the graph is all 0
    if (graphEigen.norm() == 0) {
        std::cout << "Graph is disconnected. You may need to check the compatibility threshold!" << std::endl;
        return false;
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// Graph filtering
    ///
    ////////////////////////////////////////////////////////////////

    // Prepare for filtering
    // 1. Compute degree of the vertexes
    timing(0);
    std::vector<VertexDgree> graphDegree(totalCorresNum);
#pragma omp parallel for schedule(static) default(none) shared(totalCorresNum, graphDegree, gtCorres, graphEigen)
    for (int i = 0; i < totalCorresNum; ++i) {
        // Construct variables
        int currentIndex = 0;
        int degree = 0;
        // float score = 0;
        std::vector<int> corresIndex;
        corresIndex.reserve(totalCorresNum);
        int localCorrectMatchNum = 0;
        for (int j = 0; j < totalCorresNum; ++j) {
            if (i != j && graphEigen(i, j)) {
                degree++;
                corresIndex.push_back(j);
                if (!gtCorres.empty() && gtCorres[j]) {
                    localCorrectMatchNum++;
                }
            }
        }
        graphDegree[i].currentIndex = currentIndex;
        graphDegree[i].degree = degree;
        graphDegree[i].corresIndex = corresIndex;
        graphDegree[i].localCorrectMatchNum = localCorrectMatchNum;
    }

    // // igraph version, should be carefully tested. I did not try igraph libs
    // igraph_t graphIgraph;
    // igraph_matrix_t graphIgraphMatrix;
    // igraph_matrix_view(&graphIgraphMatrix, graphEigen.data(), totalCorrespondencesNum, totalCorrespondencesNum);
    //
    // igraph_adjacency(&graphIgraph, &graphIgraphMatrix, IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS);
    // std::cout << "\nigraph graph object created successfully." << std::endl;
    // igraph_vector_int_t degreesIgraph;
    //
    // igraph_error_t errorCodeIgraph = igraph_degree(&graphIgraph, &degreesIgraph, igraph_vss_all(), IGRAPH_ALL,
    //                                                  IGRAPH_NO_LOOPS);
    // if (errorCodeIgraph != IGRAPH_SUCCESS) {
    //     std::cerr << "Error calculating degree: " << igraph_strerror(errorCodeIgraph) << std::endl;
    //     return false;
    // }
    timing(1);

    // 2. Calculate the triangular weight to determine the density of the graph.
    // Pruning if the graph is dense

    timing(0);
    std::vector<VertexStruct> triangularWeights;
    triangularWeights.reserve(totalCorresNum);
    float weightSumNumerator = 0;
    float weightSumDenominator = 0;
    for (int i = 0; i < totalCorresNum; ++i) {
        if (int neighborSize = graphDegree[i].degree; neighborSize > 1) {
            double weightSumI = 0.0;
#pragma omp parallel
            {
#pragma omp for
                for (int j = 0; j < neighborSize; ++j) {
                    int neighborIndex1 = graphDegree[i].corresIndex[j];
                    for (int k = j + 1; k < neighborSize; ++k) {
                        if (int neighborIndex2 = graphDegree[i].corresIndex[k]; graphEigen(
                            neighborIndex1, neighborIndex2)) {
#pragma omp critical
                            // The three vertexes construct a triangle
                            weightSumI += pow(
                                graphEigen(i, neighborIndex1) * graphEigen(i, neighborIndex2) * graphEigen(
                                    neighborIndex1, neighborIndex2), 1.0 / 3);
                        }
                    }
                }
            }
            // Total possible triangles in the neighborhood of vertex i
            float vertexDenominator = static_cast<float>(neighborSize * (neighborSize - 1)) / 2.0f;
            weightSumNumerator += weightSumI;
            weightSumDenominator += vertexDenominator;
            float vertexFactor = weightSumI / vertexDenominator;
            triangularWeights.emplace_back(i, vertexFactor); // optimization needed?
        } else {
            triangularWeights.emplace_back(i, 0.0f); // If the vertex has no neighbors, set the weight to 0
        }
    }

    std::cout << "Triangular weight computation completed. Time elapsed: " << std::endl;
    timing(1);
    // Need to complete the timing logics


    // 3. Threshold calculation
    // average weight among all vertex
    float averageVertexWeight = 0;
    for (auto &i: triangularWeights) {
        averageVertexWeight += i.score;
    }
    averageVertexWeight /= static_cast<float>(triangularWeights.size());

    // average weight among all triangle
    float averageTrianguleWeight = weightSumNumerator / weightSumDenominator;

    std::vector<VertexStruct> triangularWeightsSorted;
    triangularWeightsSorted.assign(triangularWeights.begin(), triangularWeights.end()); // copy of clusterFactor
    std::sort(triangularWeightsSorted.begin(), triangularWeightsSorted.end(), compareLocalScore);

    // Prepare data for OTSU thresholding
    std::vector<float> triangularWeightSores;
    triangularWeightSores.resize(triangularWeightsSorted.size());
    for (int i = 0; i < triangularWeightsSorted.size(); ++i) {
        triangularWeightSores[i] = triangularWeightsSorted[i].score;
    }
    float otsu = 0;
    if (triangularWeightSores[0] != 0) {
        otsu = otsuThresh(triangularWeightSores);
    }
    float graphThreshold = std::min (otsu, std::min(averageVertexWeight, averageTrianguleWeight));

    std::cout << graphThreshold << "->min(" << otsu << " " << averageVertexWeight << " " << averageTrianguleWeight << ")" << std::endl;

    // ---------------------------- Evaluation part ----------------------------
    float gtInlierRatio = result.gtInlierNum / static_cast<float>(totalCorresNum);
    std::cout << "Ground truth inliers: " << result.gtInlierNum << "\ttotal num: " << totalCorresNum << std::endl;
    std::cout << "Ground truth inlier ratio: " << gtInlierRatio*100 << "%" << std::endl;
    // -------------------------------------------------------------------------
    // OTSU计算权重的阈值

    // assign score member variable. Note that we need to align the indexes
    if (macConfig.flagInstanceEqual) {
        for (size_t i = 0; i < totalCorresNum; i++) {
            corresOriginal[i].score = triangularWeights[i].score;
        }
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// Maximal clique
    ///
    ////////////////////////////////////////////////////////////////

    timing(0);
    // Create igraph graph from the Eigen matrix
    igraph_t graphIgraph;
    igraph_matrix_t graphIgraphMatrix;
    igraph_matrix_init(&graphIgraphMatrix, graphEigen.rows(), graphEigen.cols());

    // Filtering, reduce the graph size
    // Note that the original mac++ use this to filter the graph on kitti dataset. We ignore it here
    if (graphThreshold > 2.9 && totalCorresNum > 50) {

    }
    else{
        for (int i = 0; i < graphEigen.rows(); ++i) {
            for (int j = 0; j < graphEigen.cols(); ++j) {
                if (graphEigen(i, j)) {
                    igraph_matrix_set(&graphIgraphMatrix, i, j, graphEigen(i, j));
                } else {
                    igraph_matrix_set(&graphIgraphMatrix, i, j, 0);
                }
            }
        }
    }

    // TODO: We can use igraph_adjlist to construct the igraph graph. This may reduce the graph construction time.
    // TODO: igraph can also use BLAS to speed up processing.
    // Need to be checked!!! I do not know how to use igraph!!
    // igraph_matrix_view(&graphIgraphMatrix, graphEigen.data(), totalCorresNum, totalCorresNum);
    igraph_set_attribute_table(&igraph_cattribute_table);
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&graphIgraph, &graphIgraphMatrix, IGRAPH_ADJ_UNDIRECTED, &weight, IGRAPH_NO_LOOPS);

    // Find the maximal cliques in the graph
    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);

    int minCliqueSize = 3; // Minimum size of the clique to be considered, 3 is the minimum number to creat a triangle
    int maxCliqueSize = 0; // Maximum size of the clique, 0 is no limit.
    bool recalculateFlag = true; // Flag to indicate whether to recalculate the cliques
    int iterNum = 1;

    while (recalculateFlag) {
        igraph_error_t error_code = igraph_maximal_cliques(&graphIgraph, &cliques, minCliqueSize, maxCliqueSize);
        totalCliqueNum = static_cast<int>(igraph_vector_int_list_size(&cliques));
        // For now, we do not know in what case this will happen
        if (totalCliqueNum > macConfig.maxTotalCliqueNum && iterNum <= macConfig.maxCliqueIterations) {
            maxCliqueSize = macConfig.maxCliqueSize;
            minCliqueSize += iterNum;
            iterNum++;
            igraph_vector_int_list_destroy(&cliques);
            igraph_vector_int_list_init(&cliques, 0);
            std::cout << BLUE << "Number of clique(" << totalCliqueNum << ") is too large, recalculate with minCliqueSize = "
                    << minCliqueSize << " and maxCliqueSize = " << maxCliqueSize << RESET << std::endl;
        } else {
            recalculateFlag = false;
        }
        // 3. **必须**检查返回值
        if (error_code != IGRAPH_SUCCESS) {
            // 如果失败，打印错误信息并退出，不要继续使用 cliques
            std::cout << RED << "Error finding maximal cliques: " << igraph_strerror(error_code) << RESET << std::endl;
            // 在这里处理错误，比如 return -1;
        }
    }


    timing(1);

    if (totalCliqueNum == 0) {
        std::cout << RED << "Error: No cliques found in the graph." << RESET << std::endl;
        std::cout << "Exiting..." << std::endl;
        return false;
    }
    std::cout << "Number of cliques found: " << totalCliqueNum << ". Time for maximal clique search: " << "Not implemented yet" << std::endl;
    // timing logic should be completed

    // Data cleaning
    // igraph_destroy(&graphIgraph);
    // igraph_matrix_destroy(&graphIgraphMatrix);

    // Correspondence seed generation and clique pre filtering
    std::vector<int> sampledCorresIndex; // sampled correspondences index
    std::vector<int> sampledCliqueIndex; // sampled clique index after filtering

    // Here is the problem!!! Check this
    cliqueSampling(macConfig, graphEigen, &cliques, sampledCorresIndex, sampledCliqueIndex);

    std::vector<CorresStruct> sampledCorr; // sampled correspondences
    PointCloudPtr sampledCorrSrc(new pcl::PointCloud<pcl::PointXYZ>); // sampled source point cloud
    PointCloudPtr sampledCorrTgt(new pcl::PointCloud<pcl::PointXYZ>); // sampled target point cloud
    int inlierNumAfCliqueSampling = 0;
    for (auto &ind: sampledCorresIndex) {
        sampledCorr.push_back(corresOriginal[ind]);
        sampledCorrSrc->push_back(corresOriginal[ind].src);
        sampledCorrTgt->push_back(corresOriginal[ind].tgt);
        if (gtCorres[ind]) {
            inlierNumAfCliqueSampling++;
        }
    }

    // ---------------------------- Evaluation part ----------------------------
    // Save log
    std::string sampledCorrTxt = macConfig.outputPath + "/sampled_corr.txt";
    std::ofstream outFile1;
    outFile1.open(sampledCorrTxt.c_str(), ios::out);
    for (int i = 0; i < static_cast<int>(sampledCorr.size()); i++) {
        outFile1 << sampledCorr[i].srcIndex << " " << sampledCorr[i].tgtIndex << std::endl;
    }
    outFile1.close();

    std::string sampledCorrLabel = macConfig.outputPath + "/sampled_corr_label.txt";
    std::ofstream outFile2;
    outFile2.open(sampledCorrLabel.c_str(), ios::out);
    for (auto &ind: sampledCorresIndex) {
        if (gtCorres[ind]) {
            outFile2 << "1" << std::endl;
        } else {
            outFile2 << "0" << std::endl;
        }
    }
    outFile2.close();

    // TODO: 统一说法，clique sampling、clique filtering、clique pruning
    // The inlier ratio should be higher than the original inlier ratio
    std::cout << "Inlier ratio after clique sampling: "
            << static_cast<float>(inlierNumAfCliqueSampling) / static_cast<float>(sampledCorresIndex.size()) * 100
            << "%" << std::endl;
    // -------------------------------------------------------------------------

    std::cout << "Number of total cliques: " << totalCliqueNum << std::endl;
    std::cout << "Number of sampled correspondences: " << sampledCorresIndex.size() << std::endl;
    std::cout << "Number of sampled cliques: " << sampledCliqueIndex.size() << std::endl;
    std::cout << BLUE << "Time for clique sampling: " << RESET << std::endl; // timing logic should be completed
    timing(1);

    // Construct the correspondence points index list for sampled correspondences
    PointCloudPtr srcCorrPts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr desCorrPts(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < totalCorresNum; i++) {
        srcCorrPts->push_back(corresOriginal[i].src);
        desCorrPts->push_back(corresOriginal[i].tgt);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// Registration
    ///
    ////////////////////////////////////////////////////////////////

    Eigen::Matrix4f bestEstIndividual, bestEstConsensus; //

    bool flagFound = false; // Flag to indicate whether a valid registration was found
    float bestGlobalScore = 0.0f; // Best score for the registration

    timing(0);
    int totalEstimateNum = sampledCliqueIndex.size(); // Total number of estimated correspondences

    std::vector<Eigen::Matrix3f> Rs;
    std::vector<Eigen::Vector3f> Ts;
    std::vector<float> localScores;
    std::vector<std::vector<int> > cliqueVertexesIndices; // store the global indices of the vertexes in each sampled clique

    std::vector<CliqueStruct> cliquesEvaluated;
    std::vector<std::pair<int, std::vector<int> > > tgtSrc; // Used for one way matching
    makeTgtSrcPair(corresOriginal, tgtSrc); // 将初始匹配形成点到点集的对应

    // 1. Estimate the transformation matrix by the points in the clique (SVD)
// #pragma omp parallel for
    for (int i = 0; i < totalEstimateNum; ++i) {
        float triangularScoreThresh = 0;
        std::vector<CorresStruct> localCliqueVertexes; // , localCliqueVertexesFiltered; // Seemed not used
        std::vector<int> currentCliqueVertexesIndex;  // selected vertexes index in the current clique
        igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(&cliques, sampledCliqueIndex[i]);
        int cliqueSize = igraph_vector_int_size(v); // size of the current clique
        for (int j = 0; j < cliqueSize; j++) {
            int ind = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            localCliqueVertexes.push_back(corresOriginal[ind]);
            currentCliqueVertexesIndex.push_back(ind);
        }
        if (cliqueSize==13 && currentCliqueVertexesIndex[0] == 2435 && currentCliqueVertexesIndex[1] == 2850) {
            std::cout << "Clique found" << std::endl;
        }
        std::sort(currentCliqueVertexesIndex.begin(), currentCliqueVertexesIndex.end()); // sort before get intersection

        //
        Eigen::Matrix4f estTransMat;
        PointCloudPtr srcPts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr tgtPts(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<float> triangularScoresInFilteredClique; // triangular scores in the current clique
        for (auto &k: localCliqueVertexes) {
            if (k.score >= triangularScoreThresh) { // This score is the correspondence triangular score
                // 0 by default
                // localCliqueVertexesFiltered.push_back(k);
                srcPts->push_back(k.src);
                tgtPts->push_back(k.tgt);
                // !!! Caution, there is critical logic error.
                // The size of the weights to svd is different with the size of the srcPts and tgtPts.
                triangularScoresInFilteredClique.push_back(k.score); // The correspondence triangular score.
                // Use this as the weight for the SVD
            }
        }
        // If the clique is too small, skip it
        if (triangularScoresInFilteredClique.size() < 3) {
            continue;
        }
        // Optimization needed
        Eigen::VectorXf scoreVec = Eigen::Map<Eigen::VectorXf>(triangularScoresInFilteredClique.data(), triangularScoresInFilteredClique.size());
        triangularScoresInFilteredClique.clear();
        triangularScoresInFilteredClique.shrink_to_fit();
        // This can be done before weight assignments
        // if (macConfig.flagInstanceEqual) {
        scoreVec.setOnes();
        // } else {
        //     scoreVec /= scoreVec.maxCoeff();
        // }
        weightSvd(srcPts, tgtPts, scoreVec, triangularScoreThresh, estTransMat); // scoreThresh is 0 in original MAC++

        // 2. pre evaluate the transformation matrix generated by each clique (group, in MAC++)
        float globalScore = 0.0f, localScore = 0.0f;
        // These evaluation is important
        // Use the whole kpts point cloud to evaluate
        globalScore = OAMAE(cloudSrcKpts, cloudTgtKpts, estTransMat, tgtSrc, inlierEvaThresh);
        // Use the clique points to evaluate
        localScore = evaluateTransByLocalClique(srcPts, tgtPts, estTransMat, inlierEvaThresh, macConfig.metric);

        // srcPts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        // tgtPts.reset(new pcl::PointCloud<pcl::PointXYZ>);

        if (globalScore > 0) {
// #pragma omp critical
            {
                Eigen::Matrix4f transF = estTransMat;
                Eigen::Matrix3f R = transF.topLeftCorner(3, 3);
                Eigen::Vector3f T = transF.block(0, 3, 3, 1);
                Rs.push_back(R);
                Ts.push_back(T);
                localScores.push_back(localScore); // local score add to scores
                cliqueVertexesIndices.push_back(currentCliqueVertexesIndex);
                CliqueStruct currentClique;
                currentClique.currentIndex = i;
                currentClique.score = globalScore;
                double re, te;
                // ---------------------------- Evaluation part ----------------------------
                currentClique.flagGtCorrect = evaluationEst(estTransMat, gtMat, 15, 30, re, te);
                // -------------------------------------------------------------------------
                cliquesEvaluated.push_back(currentClique);
                if (bestGlobalScore < globalScore) {
                    bestGlobalScore = globalScore; // score is the global evaluation score
                    bestEstIndividual = estTransMat; // bestEstIndividual is the one generated from each clique weighted svd
                }
            }
        }
        currentCliqueVertexesIndex.clear();
        currentCliqueVertexesIndex.shrink_to_fit();
    }

    std::cout << BLUE << "TEST" << RESET << std::endl;

    //释放内存空间
    // Clique searching is done, we can destroy the cliques
    igraph_vector_int_list_destroy(&cliques);

    std::vector<int> cliqueIndicesEvaluated(cliquesEvaluated.size());
    for (int i = 0; i < static_cast<int>(cliquesEvaluated.size()); ++i) {
        cliqueIndicesEvaluated[i] = i;
    }
    std::sort(cliqueIndicesEvaluated.begin(), cliqueIndicesEvaluated.end(), [&cliquesEvaluated](const int a, const int b) {
        return cliquesEvaluated[a].score > cliquesEvaluated[b].score;
    });
    std::vector<CliqueStruct> cliqueEvaluatedSorted(cliquesEvaluated.size()); // sorted estVector
    for (int i = 0; i < static_cast<int>(cliquesEvaluated.size()); i++) {
        cliqueEvaluatedSorted[i] = cliquesEvaluated[cliqueIndicesEvaluated[i]];
    }
    // TODO: Check all groud true evaluations, and unify the naming. Also pay attention to the method evaluation. Unify the comment expression.
    int selectedCliqueNum = std::min(std::min(totalCorresNum, totalEstimateNum), macConfig.maxEstimateNum);
    std::vector<Eigen::Matrix3f> RsNew;
    std::vector<Eigen::Vector3f> TsNew;
    if (static_cast<int>(cliqueEvaluatedSorted.size()) > selectedCliqueNum) {
        //选出排名靠前的假设
        std::cout << "Too many cliques (" << cliqueEvaluatedSorted.size() << "), we choose top " << selectedCliqueNum << " candidates." << std::endl;
    }
    // Update the Rs and Ts
    for (int i = 0; i < std::min(selectedCliqueNum, static_cast<int>(cliqueEvaluatedSorted.size())); i++) {
        RsNew.push_back(Rs[cliqueIndicesEvaluated[i]]);
        TsNew.push_back(Ts[cliqueIndicesEvaluated[i]]);
        // ---------------------------- Evaluation part ----------------------------
        successNum += cliqueEvaluatedSorted[i].flagGtCorrect ? 1 : 0;
        // -------------------------------------------------------------------------
    }
    Rs.clear();
    Rs.shrink_to_fit();
    Ts.clear();
    Ts.shrink_to_fit();

    // ---------------------------- Evaluation part ----------------------------
    if (successNum > 0) {
        if (!macConfig.flagNoLogs) {
            std::string estInfo = macConfig.outputPath + "/est_info.txt";
            std::ofstream estInfoFile(estInfo, ios::trunc);
            estInfoFile.setf(ios::fixed, ios::floatfield);
            for (auto &i: cliqueEvaluatedSorted) {
                estInfoFile << setprecision(10) << i.score << " " << i.flagGtCorrect << std::endl;
            }
            estInfoFile.close();
        }
    } else {
        std::cout << YELLOW << "NO CORRECT ESTIMATION!!!" << RESET << std::endl;
    }
    // -------------------------------------------------------------------------

    result.correctEstNum = successNum;

    // 3. Clustering
    // 使用配置化的聚类参数替代硬编码值
    float angleThresh = currentDatasetConfig.clusteringAngleThreshold * M_PI / 180.0f; // 转换为弧度
    float disThresh;
    // 根据数据集类型设置聚类距离阈值
    if (isU3M) {
        disThresh = currentDatasetConfig.getActualClusteringDistanceThreshold(cloudResolution, true);
    } else {
        disThresh = currentDatasetConfig.clusteringDistanceThreshold;
    }

    // Clustering the estimated transformations
    pcl::IndicesClusters clusterTrans;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr trans(new pcl::PointCloud<pcl::PointXYZINormal>);
    float bestSimilarity = std::numeric_limits<float>::max();
    int bestSimCliqueIndex2bestEstIndividual; // Cluster index
    int bestSimInCliqueIndex2bestEstIndividual; // Vertex index in the clique
    int bestCliqueIndexWithBestEstConsensus;

    std::cout<< BLUE << "Start clustering the transformations..." << RESET <<std::endl;
    std::cout << "angle threshold : " << angleThresh << std::endl;
    std::cout << "distance threshold : " << disThresh << std::endl;
    clusterTransformationByRotation(RsNew, TsNew, angleThresh, disThresh, clusterTrans, trans);
    std::cout << "Total " << selectedCliqueNum << " cliques(transformations) found, " << clusterTrans.size() << " clusters found."
            << std::endl;
    // If the clustering failed, then we use the standard MAC
    // TODO: Revise the code below
    if (clusterTrans.size() == 0) {
        std::cout << YELLOW << "Warning: No clusters found, using the standard MAC from the cliques." << RESET <<
                std::endl;
        Eigen::MatrixXf tmpBest;
        // ---------------------------- Evaluation part ----------------------------
        if (macConfig.datasetName == "U3M") {
            result.RE = rmseCompute(cloudSrc, cloudTgt, bestEstIndividual, gtMat, cloudResolution);
            result.TE = 0;
        } else {
            if (!flagFound) {
                flagFound = evaluationEst(bestEstIndividual, gtMat, REEvaThresh, TEEvaThresh, result.RE, result.TE);
            }
            tmpBest = bestEstIndividual;
            bestGlobalScore = 0;
            postRefinement(sampledCorr, sampledCorrSrc, sampledCorrTgt, bestEstIndividual, bestGlobalScore, inlierEvaThresh, 20,
                           macConfig.metric);
        }
        if (macConfig.datasetName == "U3M") {
            if (result.RE <= 5) {
                std::cout << result.RE << std::endl;
                std::cout << bestEstIndividual << std::endl;
                return true;
            }
            return false;
        }
        //            float rmse = RMSE_compute_scene(cloudSrc, cloudTgt, bestEst1, GTmat, 0.0375);
        //            std::cout << "RMSE: " << rmse <<endl;
        if (flagFound) {
            double newRe, newTe;
            evaluationEst(bestEstIndividual, gtMat, REEvaThresh, TEEvaThresh, newRe, newTe);

            if (newRe < result.RE && newTe < result.TE) {
                std::cout << "est_trans updated!!!" << std::endl;
                std::cout << "RE=" << newRe << " " << "TE=" << newTe << std::endl;
                std::cout << bestEstIndividual << std::endl;
            } else {
                bestEstIndividual = tmpBest;
                std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
                std::cout << bestEstIndividual << std::endl;
            }
            result.RE = newRe;
            result.TE = newTe;
            //                if(rmse > 0.2) return false;
            //                else return true;
            return true;
        }
        double newRe, newTe;
        flagFound = evaluationEst(bestEstIndividual, gtMat, REEvaThresh, TEEvaThresh, newRe, newTe);
        if (flagFound) {
            result.RE = newRe;
            result.TE = newTe;
            std::cout << "est_trans corrected!!!" << std::endl;
            std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
            std::cout << bestEstIndividual << std::endl;
            return true;
        }
        std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
        return false;
        // -------------------------------------------------------------------------
        //                if(rmse > 0.2) return false;
        //                else return true;
    }

    // Sort the clusters by size
    int goodClusterNum = 0;
    std::vector<ClusterStruct> clusterSorted(clusterTrans.size());
    for (size_t i = 0; i < clusterTrans.size(); ++i) {
        clusterSorted[i].currentIndex = i;
        clusterSorted[i].clusterSize = static_cast<float>(clusterTrans[i].indices.size());
        if (clusterSorted[i].clusterSize >= 1) {
            goodClusterNum++;
        }
    }
    if (goodClusterNum <= 0) {
        std::cout << YELLOW << "Warning: No good clusters found. The result is probably unreliable." << RESET << std::endl;
    }
    std::sort(clusterSorted.begin(), clusterSorted.end(), compareClusterScore);

    // Find where the bestEst1 locates
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > estTransFromClusterCenter;
    // align the memory (do not know why)
    std::vector<int> clusterIndexOfClusterCenterEst;
    std::vector<int> globalUnionInd;

    // Find the most similar transformation to the bestEstIndividual
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(clusterSorted.size()); i++) {
        int index = clusterSorted[i].currentIndex;
        for (int j = 0; j < static_cast<int>(clusterTrans[index].indices.size()); j++) {
            int k = clusterTrans[index].indices[j];
            Eigen::Matrix3f R = RsNew[k];
            Eigen::Vector3f T = TsNew[k];
            Eigen::Matrix4f mat;
            mat.setIdentity();
            mat.block(0, 3, 3, 1) = T;
            mat.topLeftCorner(3, 3) = R;
            float similarity = (bestEstIndividual.inverse() * mat - Eigen::Matrix4f::Identity(4, 4)).norm();
#pragma omp critical
            {
                if (similarity < bestSimilarity) {
                    bestSimilarity = similarity;
                    bestSimInCliqueIndex2bestEstIndividual = j;
                    bestSimCliqueIndex2bestEstIndividual = index;
                }
            }
        }
    }
    std::cout << "Transformation " << bestSimInCliqueIndex2bestEstIndividual << " in cluster "
            << bestSimCliqueIndex2bestEstIndividual << " (" << clusterSorted[bestSimCliqueIndex2bestEstIndividual].clusterSize
            << ") is similar to bestEstIndividual with score " << bestSimilarity << std::endl;

    // 4. Find the best cluster center and cluster vertexes
    // Get centers
    std::vector<std::vector<int> > subClusterIndexes;
    std::cout << "START TEST" << std::endl;
// #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(clusterSorted.size()); i++) {

        //考察同一聚类的匹配
        // std::vector<CorresStruct> subClusterCorr;
        PointCloudPtr clusterSrcPts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr clusterDesPts(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> subUnionInd;
        int index = clusterSorted[i].currentIndex;
        int centerCliqueIndexGlobal = clusterTrans[index].indices[0]; // Set a random vertex as initial cluster center
        float clusterCenterScore = localScores[cliqueIndicesEvaluated[centerCliqueIndexGlobal]]; //
        // subUnionInd contain global index for corresOriginal
        // Find the union of the vertexes in the clique
        subUnionInd.assign(cliqueVertexesIndices[cliqueIndicesEvaluated[centerCliqueIndexGlobal]].begin(),
            cliqueVertexesIndices[cliqueIndicesEvaluated[centerCliqueIndexGlobal]].end());

        for (int j = 1; j < static_cast<int>(clusterTrans[index].indices.size()); j++) {
            int m = clusterTrans[index].indices[j];
            // Set the vertex with the highest local score as the new cluster center
            if (float currentScore = localScores[cliqueIndicesEvaluated[m]]; currentScore > clusterCenterScore) {
                centerCliqueIndexGlobal = m;
                clusterCenterScore = currentScore;
            }
            subUnionInd = vectorsUnion(subUnionInd, cliqueVertexesIndices[cliqueIndicesEvaluated[m]]);
        }

        for (int l = 0; l < static_cast<int>(subUnionInd.size()); ++l) {
            // subClusterCorr.push_back(corresOriginal[subUnionInd[l]]);
            clusterSrcPts->push_back(corresOriginal[subUnionInd[l]].src);
            clusterDesPts->push_back(corresOriginal[subUnionInd[l]].tgt);
        }
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = TsNew[centerCliqueIndexGlobal];
        mat.topLeftCorner(3, 3) = RsNew[centerCliqueIndexGlobal];

// #pragma omp critical
        {
            // 将subUnionInd中的索引与globalUnionInd合并，这是所有聚类所对应的节点的集合
            globalUnionInd = vectorsUnion(globalUnionInd, subUnionInd);
            estTransFromClusterCenter.push_back(mat);
            subClusterIndexes.push_back(subUnionInd);
            clusterIndexOfClusterCenterEst.push_back(index);
        }
        // subClusterCorr.clear();
        subUnionInd.clear();
    }

    std::vector<CorresStruct> globalUnionCorr;
    PointCloudPtr globalUnionCorrSrc(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr globalUnionCorrTgt(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < static_cast<int>(globalUnionInd.size()); ++i) {
        globalUnionCorr.push_back(corresOriginal[globalUnionInd[i]]);
    }
    std::vector<std::pair<int, std::vector<int> > > tgtSrcFromCluster;
    makeTgtSrcPair(globalUnionCorr, tgtSrcFromCluster); //将初始匹配形成点到点集的对应

    // Find the best cluster center, bestEstConsensus
    bestGlobalScore = 0;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(estTransFromClusterCenter.size()); i++) {
        double clusterEvaScore;
        // _1tok is not used in this project
        // if(_1tok){
        //     clusterEvaScore = OAMAE_1tok(cloudSrcKpts, cloudTgtKpts, estTrans2[i], one2k_match, inlierThresh);
        // }
        // else{
        // Use kpts to evaluate the transformation matrix
        clusterEvaScore = OAMAE(cloudSrcKpts, cloudTgtKpts, estTransFromClusterCenter[i], tgtSrcFromCluster, inlierEvaThresh);
        // }
#pragma omp critical
        {
            if (bestGlobalScore < clusterEvaScore) {
                bestGlobalScore = clusterEvaScore;
                bestEstConsensus = estTransFromClusterCenter[i];
                bestCliqueIndexWithBestEstConsensus = clusterIndexOfClusterCenterEst[i];
            }
        }
    }

    // clusterIndexOfClusterCenterEst 排序 sub cluster indices
    // update the cliqueIndicesEvaluated by the newly got clusterIndexOfClusterCenterEst, those get the cluster cliques
    cliqueIndicesEvaluated.clear();
    for (int i = 0; i < static_cast<int>(clusterIndexOfClusterCenterEst.size()); i++) {
        cliqueIndicesEvaluated.push_back(i);
    }
    sort(cliqueIndicesEvaluated.begin(), cliqueIndicesEvaluated.end(), [&clusterIndexOfClusterCenterEst](const int a, const int b) {
        return clusterIndexOfClusterCenterEst[a] < clusterIndexOfClusterCenterEst[b];
    });
    std::vector<std::vector<int> > subClusterIndexesSorted;
    for (auto &ind: cliqueIndicesEvaluated) {
        subClusterIndexesSorted.push_back(subClusterIndexes[ind]);
    }
    subClusterIndexes.clear();

    //输出每个best_est分别在哪个聚类
    if (bestCliqueIndexWithBestEstConsensus == bestSimCliqueIndex2bestEstIndividual) {
        std::cout << "Both choose clique: " << bestCliqueIndexWithBestEstConsensus << std::endl;
    } else {
        // Best consensus is the center clique using global OAMAE
        // Best sim is the clustered clique that is similar to the best estimated individual (global estimate)
        // These two cliques
        std::cout << "Best clique index with best consensus estimation: " << bestCliqueIndexWithBestEstConsensus
        << ", best similarity clique index to best estimated individual: " << bestSimCliqueIndex2bestEstIndividual << std::endl;
    }
    //sampled corr -> overlap prior batch -> TCD 确定bestEst1和bestEst2中最好的
    Eigen::Matrix4f bestEst;
    PointCloudPtr sampledSrc(new pcl::PointCloud<pcl::PointXYZ>); // dense point cloud
    PointCloudPtr sampledDes(new pcl::PointCloud<pcl::PointXYZ>);

    getCorrPatch(sampledCorr, cloudSrcKpts, cloudTgtKpts, sampledSrc, sampledDes, 2 * inlierEvaThresh);
    //点云patch后校验两个best_est
    float bestEstIndividualScore = truncatedChamferDistance(sampledSrc, sampledDes, bestEstIndividual, inlierEvaThresh);
    float bestEstConsensusScore = truncatedChamferDistance(sampledSrc, sampledDes, bestEstConsensus, inlierEvaThresh);
    PointCloudPtr clusterEvaCorrSrc(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr clusterEvaCorrDes(new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "bestEstIndividualScore: " << bestEstIndividualScore << ", bestEstConsensusScore: " << bestEstConsensusScore << std::endl;

    // cluster_internal_evaluation
    if (macConfig.flagClusterInternalEvaluation) {
        std::vector<CorresStruct> clusterEvaCorr;
        if (bestSimilarity < 0.1) {
            // bestEstIndividual is in the cluster
            std::cout << "bestEstIndividual is in a cluster, bestSimilarity: " << bestSimilarity << std::endl;
            if (bestEstIndividualScore > bestEstConsensusScore) {
                // bestEstIndividualScore is better, then the individual is better than the cluster center estimate
                bestCliqueIndexWithBestEstConsensus = bestSimCliqueIndex2bestEstIndividual;
                bestEst = bestEstIndividual;
                std::cout << "bestEstIndividual (global individual) is better" << std::endl;
            } else {
                // bestEstConsensusScore is better, then use this for final evaluation
                bestEst = bestEstConsensus;
                std::cout << "bestEstConsensus (cluster center estimate) is better" << std::endl;
            }
            // Get the intersection of the sampled correspondences and the correspondences in the best cluster
            std::vector<int> clusterEvaCorrInd;
            clusterEvaCorrInd.assign(subClusterIndexesSorted[bestCliqueIndexWithBestEstConsensus].begin(), subClusterIndexesSorted[bestCliqueIndexWithBestEstConsensus].end());
            std::sort(clusterEvaCorrInd.begin(), clusterEvaCorrInd.end());
            std::sort(sampledCorresIndex.begin(), sampledCorresIndex.end());
            clusterEvaCorrInd = vectorsIntersection(clusterEvaCorrInd, sampledCorresIndex);
            // ---------------------------- Evaluation part ----------------------------
            if (!clusterEvaCorrInd.size()) {
                return false;
            }
            inlierNumAfCliqueSampling = 0;
            for (auto &ind: clusterEvaCorrInd) {
                clusterEvaCorr.push_back(corresOriginal[ind]);
                clusterEvaCorrSrc->push_back(corresOriginal[ind].src);
                clusterEvaCorrDes->push_back(corresOriginal[ind].tgt);
                if (gtCorres[ind]) {
                    inlierNumAfCliqueSampling++;
                }
            }
            std::cout << clusterEvaCorrInd.size() << " intersection correspondences have " << inlierNumAfCliqueSampling <<
                    " inliers: " << inlierNumAfCliqueSampling / (static_cast<int>(clusterEvaCorrInd.size()) / 1.0) * 100 << "%" <<
                    std::endl;
            // -------------------------------------------------------------------------
            std::vector<std::pair<int, std::vector<int> > > tgtSrc3;
            makeTgtSrcPair(clusterEvaCorr, tgtSrc3);
            // if(_1tok){
            //     bestEst = clusterInternalTransEva1(clusterTrans, bestIndex, bestEst, RsNew, TsNew, cloudSrcKpts, cloudTgtKpts, one2k_match, inlierThresh, GTmat, true, folderPath);
            // }
            // else{
            bestEst = clusterInternalTransEva1(clusterTrans, bestCliqueIndexWithBestEstConsensus, bestEst, RsNew, TsNew, cloudSrcKpts, cloudTgtKpts,
                                               tgtSrc3, inlierEvaThresh, gtMat, false, macConfig.outputPath);
            // }
        } else {
            // bestEstIndividual is not in a cluster
            std::cout << "bestEstIndividual is not in a cluster, bestSimilarity: " << bestSimilarity << std::endl;
            if (bestEstConsensusScore > bestEstIndividualScore) {
                // bestEstConsensusScore is better, then use this for final evaluation
                bestEst = bestEstConsensus;
                std::cout << "bestEstConsensus is better" << std::endl;
                std::vector<int> clusterEvaCorrInd;
                clusterEvaCorrInd.assign(subClusterIndexesSorted[bestCliqueIndexWithBestEstConsensus].begin(), subClusterIndexesSorted[bestCliqueIndexWithBestEstConsensus].end());
                std::sort(clusterEvaCorrInd.begin(), clusterEvaCorrInd.end());
                std::sort(sampledCorresIndex.begin(), sampledCorresIndex.end());
                clusterEvaCorrInd = vectorsIntersection(clusterEvaCorrInd, sampledCorresIndex);
                if (!clusterEvaCorrInd.size()) {
                    return false;
                }
                inlierNumAfCliqueSampling = 0;

                for (auto &ind: clusterEvaCorrInd) {
                    clusterEvaCorr.push_back(corresOriginal[ind]);
                    clusterEvaCorrSrc->push_back(corresOriginal[ind].src);
                    clusterEvaCorrDes->push_back(corresOriginal[ind].tgt);
                    if (gtCorres[ind]) {
                        inlierNumAfCliqueSampling++;
                    }
                }
                std::cout << clusterEvaCorrInd.size() << " intersection correspondences have " << inlierNumAfCliqueSampling
                        << " inliers: " << inlierNumAfCliqueSampling / (static_cast<int>(clusterEvaCorrInd.size()) / 1.0) * 100 <<
                        "%" << std::endl;
                std::vector<std::pair<int, std::vector<int> > > desSrc3;
                makeTgtSrcPair(clusterEvaCorr, desSrc3);
                bestEst = clusterInternalTransEva1(clusterTrans, bestCliqueIndexWithBestEstConsensus, bestEst, RsNew, TsNew, cloudSrcKpts, cloudTgtKpts,
                                                   desSrc3, inlierEvaThresh, gtMat, false, macConfig.outputPath); // Check the outputPath
            } else {
                //
                bestCliqueIndexWithBestEstConsensus = -1; //不存在类中
                bestEst = bestEstIndividual;
                std::cout << "bestEstIndividual is better but not in cluster! Refine it" << std::endl;
            }
        }
    } else {
        bestEst = bestEstIndividualScore > bestEstConsensusScore ? bestEstIndividual : bestEstConsensus;
    }

    timing(1);
    std::cout << " post evaluation: " << std::endl; //timing logic should be implemented

    Eigen::Matrix4f tmpBest;
    if (macConfig.datasetName == "U3M") {
        result.RE = rmseCompute(cloudSrc, cloudTgt, bestEst, gtMat, cloudResolution);
        result.TE = 0;
    } else {
        if (!flagFound) {
            flagFound = evaluationEst(bestEst, gtMat, REEvaThresh, TEEvaThresh, result.RE, result.TE);
        }
        tmpBest = bestEst;
        bestGlobalScore = 0;
        postRefinement(sampledCorr, sampledCorrSrc, sampledCorrTgt, bestEst, bestGlobalScore, inlierEvaThresh, 20,
                       macConfig.metric);

        std::vector<int> predInlierIndex;
        PointCloudPtr srcTrans(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*srcCorrPts, *srcTrans, bestEst);
        int cnt = 0;
        int t = 0;
        for (int j = 0; j < corresOriginal.size(); j++) {
            double dist = getDistance(srcTrans->points[j], desCorrPts->points[j]);
            if (dist < inlierEvaThresh) {
                cnt++;
                if (gtCorres[j]) {
                    t++;
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        ///
        /// ICP
        ///
        ////////////////////////////////////////////////////////////////
        //ICP
        if (macConfig.flagUseIcp) {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(cloudSrcKpts); //稀疏一些耗时小
            icp.setInputTarget(cloudTgt);
            icp.setMaxCorrespondenceDistance(0.05);
            icp.setTransformationEpsilon(1e-10);
            icp.setMaximumIterations(50);
            icp.setEuclideanFitnessEpsilon(0.2);
            PointCloudPtr final(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*final, bestEst);
            if (icp.hasConverged()) {
                bestEst = icp.getFinalTransformation();
                std::cout << "ICP fitness score: " << icp.getFitnessScore() << std::endl;
            } else {
                std::cout << "ICP cannot converge!!!" << std::endl;
            }
        }

        ////////////////////////////////////////////////////////////////
        ///
        /// Show and save results
        ///
        ////////////////////////////////////////////////////////////////

        // 计算精确率(Precision)、召回率(Recall)和F1分数
        double IP = 0, IR = 0, F1 = 0;
        if (cnt > 0) IP = t / (cnt / 1.0); // 精确率 = 正确预测的内点 / 总预测内点
        if (result.gtInlierNum > 0) IR = t / (result.gtInlierNum / 1.0); // 召回率 = 正确预测的内点 / 实际内点
        if (IP && IR) {
            F1 = 2.0 / (1.0 / IP + 1.0 / IR); // F1分数 = 2 * 精确率 * 召回率 / (精确率 + 召回率)
        }
        std::cout << IP << " " << IR << " " << F1 << std::endl;

        // 将统计信息存储到结果结构体中
        result.predicatedInlier.clear();
        result.predicatedInlier.push_back(IP);
        result.predicatedInlier.push_back(IR);
        result.predicatedInlier.push_back(F1);

    }

    if (!macConfig.flagNoLogs) {
        //保存匹配到txt
        //savetxt(correspondence, folderPath + "/corr.txt");
        //savetxt(selected, folderPath + "/selected.txt");
        std::string saveEst = macConfig.outputPath + "/est.txt";
        //std::string saveGt = folderPath + "/GTmat.txt";
        std::ofstream outfile(saveEst, ios::trunc);
        outfile.setf(ios::fixed, ios::floatfield);
        outfile << setprecision(10) << bestEst;
        outfile.close();
        //CopyFile(gt_mat.c_str(), saveGt.c_str(), false);
        //std::string saveLabel = folderPath + "/label.txt";
        //CopyFile(label_path.c_str(), saveLabel.c_str(), false);

        //保存ply
        //std::string saveSrcCloud = folderPath + "/source.ply";
        //std::string saveTgtCloud = folderPath + "/target.ply";
        //CopyFile(src_pointcloud.c_str(), saveSrcCloud.c_str(), false);
        //CopyFile(des_pointcloud.c_str(), saveTgtCloud.c_str(), false);
    }

    // memory cost evaluation is pending
    // int pid = getpid();
    // mem_epoch = getPidMemory(pid);

    //保存聚类信息
    std::string analyseCsv = macConfig.outputPath + "/cluster.csv";
    std::string correctCsv = macConfig.outputPath + "/cluster_correct.csv";
    std::string selectedCsv = macConfig.outputPath + "/cluster_selected.csv";
    std::ofstream outFile, outFileCorrect, outFileSelected;
    outFile.open(analyseCsv.c_str(), ios::out);
    outFileCorrect.open(correctCsv.c_str(), ios::out);
    outFileSelected.open(selectedCsv.c_str(), ios::out);
    outFile.setf(ios::fixed, ios::floatfield);
    outFileCorrect.setf(ios::fixed, ios::floatfield);
    outFileSelected.setf(ios::fixed, ios::floatfield);
    outFile << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << std::endl;
    outFileCorrect << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << std::endl;
    outFileSelected << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << std::endl;
    for (int i = 0; i < static_cast<int>(clusterSorted.size()); i++) {
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        int clusterId = clusterSorted[i].currentIndex;
        for (int j = 0; j < static_cast<int>(clusterTrans[clusterId].indices.size()); j++) {
            int id = clusterTrans[clusterId].indices[j];
            if (cliquesEvaluated[id].flagGtCorrect) {
                outFileCorrect << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->
                        points[id].z << ',' << r << ',' << g << ',' << b << std::endl;
                //cout << "Correct est in cluster " << clusterId << " (" << sortCluster[i].score << ")" << std::endl;
            }
            if (clusterId == bestCliqueIndexWithBestEstConsensus) outFileSelected << setprecision(4) << trans->points[id].x << ',' << trans->
                                        points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b
                                        << std::endl;
            outFile << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].
                    z << ',' << r << ',' << g << ',' << b << std::endl;
        }
    }
    outFile.close();
    outFileCorrect.close();




    corresOriginal.clear();
    corresOriginal.shrink_to_fit();
    // ovCorrLabel.clear();
    // ovCorrLabel.shrink_to_fit();
    gtCorres.clear();
    gtCorres.shrink_to_fit();
    // degree.clear();
    // degree.shrink_to_fit();
    graphDegree.clear();
    graphDegree.shrink_to_fit();
    triangularWeights.clear();
    triangularWeights.shrink_to_fit();
    triangularWeightSores.clear(); // clusterFactorBac
    triangularWeightSores.shrink_to_fit();
    sampledCliqueIndex.clear();
    sampledCliqueIndex.shrink_to_fit();
    sampledCorresIndex.clear();
    sampledCorresIndex.shrink_to_fit();
    RsNew.clear();
    RsNew.shrink_to_fit();
    TsNew.clear();
    TsNew.shrink_to_fit();
    srcCorrPts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    desCorrPts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloudSrc.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloudTgt.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloudSrcKpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloudTgtKpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    // normalSrc.reset(new pcl::PointCloud<pcl::Normal>);
    // normalTgt.reset(new pcl::PointCloud<pcl::Normal>);
    // rawSrc.reset(new pcl::PointCloud<pcl::PointXYZ>);
    // rawTgt.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (macConfig.datasetName == "U3M") {
        // 对于U3M数据集，使用配置的成功阈值进行判断
        if (result.RE <= 5) {
            std::cout << bestEst << std::endl;
            return true;
        }
        return false;
    }
    
    //float rmse = RMSE_compute_scene(cloudSrc, cloudTgt, bestEst1, GTmat, 0.0375);
    //cout << "RMSE: " << rmse <<endl;
    if (flagFound) {
        double newRe, newTe;
        evaluationEst(bestEst, gtMat, REEvaThresh, TEEvaThresh, newRe, newTe);
        if (newRe < result.RE && newTe < result.TE) {
            std::cout << "est_trans updated!!!" << std::endl;
            std::cout << "RE=" << newRe << " " << "TE=" << newTe << std::endl;
            std::cout << bestEst << std::endl;
            std::cout << GREEN << "Registration success!" << RESET << std::endl;
        } else {
            bestEst = tmpBest;
            std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
            std::cout << bestEst << std::endl;
        }
        result.RE = newRe;
        result.TE = newTe;
        //            if(rmse > 0.2){
        //                return false;
        //            }
        //            else{
        //                return true;
        //            }
        return true;
    }
    double newRe, newTe;
    flagFound = evaluationEst(bestEst, gtMat, REEvaThresh, TEEvaThresh, newRe, newTe);
    if (flagFound) {
        result.RE = newRe;
        result.TE = newTe;
        std::cout << GREEN << "Registration success!" << RESET << std::endl;
        std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
        std::cout << bestEst << std::endl;
        return true;
    }
    std::cout << "RE=" << result.RE << " " << "TE=" << result.TE << std::endl;
    return false;
    //            if(rmse > 0.2){
    //                return false;
    //            }
    //            else{
    //                return true;
    //            }
    //Corres_selected_visual(rawSrc, rawTgt, correspondence, resolution, 0.1, GTmat);
    //Corres_selected_visual(rawSrc, rawTgt, selected, resolution, 0.1, GTmat);

}


/**
 * @brief 主函数
 * @param argc 参数数量
 * @param argv 参数值数组
 * @return 成功返回0，失败返回-1
 */
int main(const int argc, char **argv) {
    // Check if the required arguments are provided
    if (argc == 1) {
        std::cout << RED << "Error: Not enough arguments provided. " << RESET << std::endl;
        std::cout << "Usage: " << argv[0] <<
                " <config_file> "
                << std::endl;
        return -1;
    }
    if (argc > 2) {
        std::cout << YELLOW << "Warning: Too many arguments provided. Ignoring the reset arguments" << RESET <<
                std::endl;
    }

    MACConfig macConfig;
    macConfig.load(argv[1]); // We do not validate the config file.

    // Check if the output directory exists, if not, create it
    if (std::error_code ec; std::filesystem::exists(macConfig.outputPath, ec)) {
        if (std::filesystem::create_directory(macConfig.outputPath)) {
            std::cout << "Error creating output directory: " << macConfig.outputPath << std::endl;
            return -1;
        }
    } else {
        std::cout << YELLOW << "Warning: Output directory already exists: " << macConfig.outputPath
                << ". Existing files may be overwritten." << std::endl << RESET
                << "Press anything to continue, or ctrl + c to exit." << std::endl;
        std::cin.get();
    }

    // Start execution
    for (int i = 0; i < macConfig.totalIterations; ++i) {
        // 创建MAC算法结果结构体实例
        MACResult macResult;

        // 执行配准算法，所有结果参数现在统一存储在macResult中
        const bool flagEstimateSuccess = registration(macConfig, macResult);

        std::ofstream resultsOut;
        // Output the evaluation results
        if (flagEstimateSuccess) {
            std::string evaResultPath = macConfig.outputPath + "/evaluation_result.txt";
            resultsOut.open(evaResultPath.c_str(), std::ios::out);
            resultsOut.setf(std::ios::fixed, std::ios::floatfield);
            resultsOut << std::setprecision(6) << "RE: " << macResult.RE << std::endl
                    << "TE: " << macResult.TE << std::endl
                    << "Correct estimated correspondences: " << macResult.correctEstNum << std::endl
                    << "Inliers in ground truth correspondences: " << macResult.gtInlierNum << std::endl
                    << "Total correspondences: " << totalCorresNum << std::endl
                    << "Time taken for registration: " << macResult.timeEpoch << " seconds" << std::endl;
            resultsOut.close();
            std::cout << GREEN << "Registration successful" << RESET << std::endl;
        } else {
            std::cout << YELLOW << "Registration failed" << RESET << std::endl;
        }

        // Output the status of the registration process
        std::string statusPath = macConfig.outputPath + "/status.txt";
        resultsOut.open(statusPath.c_str(), std::ios::out);
        resultsOut.setf(std::ios::fixed, std::ios::floatfield);
        resultsOut << std::setprecision(6) << "Time in one iteration: " << macResult.timeEpoch <<
                " seconds, memory used in one iteration: " << std::endl;
        resultsOut.close();
    }


    return 0;
}
