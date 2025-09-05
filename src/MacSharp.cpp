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
// #include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

// Windows system api
#include <cblas.h>
#include <process.h>

#include <yaml-cpp/yaml.h>

#include <random>

//
#include "MacTimer.hpp"
#include "MacConfig.hpp"
#include "MacData.hpp"
#include "MacGraph.hpp"
#include "MacSharp.hpp"

#include "MacRtHypothesis.hpp"
#include "MacUtils.hpp"

/**
 * @brief 主要的点云配准函数
 *
 * 该函数实现了基于最大团搜索的点云配准算法(MAC-SHARP)。
 * 通过构建兼容性图、搜索最大团、聚类变换矩阵等步骤实现鲁棒的点云配准。
 *
 * @param macConfig 配置对象，包含算法执行所需的所有参数设置
 * @param macData
 * @param macResult MAC算法结果结构体的引用，用于存储所有输出参数：
 *               - RE: 旋转误差
 *               - TE: 平移误差
 *               - correctEstNum: 正确估计的对应关系数量
 *               - gtInlierNum: Ground Truth中的内点数量
 *               - timeEpoch: 算法执行时间
 *               - predicatedInlier: 预测内点比率向量
 * @return bool 配准是否成功，true表示成功，false表示失败
 */
bool registration(const MacConfig &macConfig, MacData &macData, MacResult &macResult) {
    // Setting software timer
    Timer timer; // Timer class instance
    timer.clearHistory();

    ////////////////////////////////////////////////////////////////
    ///
    /// Prepare data
    ///
    ////////////////////////////////////////////////////////////////

    macResult.reset();

    // Load point clouds, correspondences, and ground truth data
    LOG_INFO("================Loading data...================");
    timer.startTiming("load data");
    if (!macData.loadData(macConfig)) {
        LOG_ERROR("Failed to load all essential data. Exiting registration.");
        return false;
    };
    timer.endTiming();

    ////////////////////////////////////////////////////////////////
    ///
    /// Graph construction
    ///
    ////////////////////////////////////////////////////////////////

    // NOTE: we do not consider the outer loop of the registration process, which is used to
    // repeat the registration process for multiple iterations.
    LOG_INFO("================Constructing graph...================");
    timer.startTiming("construct graph");
    MacGraph macGraph(macData, macConfig);
    macGraph.build();
    timer.endTiming();

    ////////////////////////////////////////////////////////////////
    ///
    /// Graph filtering
    ///
    ////////////////////////////////////////////////////////////////
    LOG_INFO("================Compute graph weights...================");
    timer.startTiming("compute graph weights");
    //// 初步剪枝的作用竟然几乎没有。。。真的吗？仔细检查这一部分
    // Prepare for filtering
    // 1. Compute degree of the vertexes
    macGraph.computeGraphDegree();

    // 2. Calculate the triangular weight to determine the density of the graph.
    // Pruning if the graph is dense
    macGraph.calculateTriangularWeights();

    // 3. Threshold calculation
    // average weight among all vertex
    // macGraph.calculateGraphThreshold();
    timer.endTiming();

    // redundant data
    macResult.gtInlierNum = macData.gtInlierCount;

    // ---------------------------- Evaluation part ----------------------------
    const float gtInlierRatio = static_cast<float>(macData.gtInlierCount) / static_cast<float>(macData.totalCorresNum);
    std::cout << "Ground truth inliers: " << macData.gtInlierCount << "\t total num: " << macData.totalCorresNum << std::endl;
    std::cout << "Ground truth inlier ratio: " << gtInlierRatio * 100 << "%" << std::endl;
    // const float gtInlierRatio = static_cast<float>(macResult.gtInlierNum) / static_cast<float>(macData.totalCorresNum);
    // std::cout << "Ground truth inliers: " << macResult.gtInlierNum << "\t total num: " << macData.totalCorresNum << std::endl;
    // std::cout << "Ground truth inlier ratio: " << gtInlierRatio * 100 << "%" << std::endl;
    // -------------------------------------------------------------------------

    ////////////////////////////////////////////////////////////////
    ///
    /// Maximal clique
    ///
    ////////////////////////////////////////////////////////////////
    LOG_INFO("================ Finding maximal cliques... ================");
    timer.startTiming("find maximal cliques");
    macGraph.findMaximalCliques();
    timer.endTiming();

    if (macData.totalCliqueNum == 0) {
        LOG_ERROR("No cliques found in the graph." << RESET);
        return false;
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// hypotheses generation
    ///
    ////////////////////////////////////////////////////////////////

    LOG_INFO("================ Generate RT hypotheses... ================");
    timer.startTiming("generate RT hypotheses");
    MacRtHypothesis macRtHypothesis(macData, macConfig, macGraph, macResult);
    macRtHypothesis.processGraphResultAndFindBest(macGraph);




return true;

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
        LOG_ERROR("Not enough arguments provided. " << RESET);
        LOG_INFO("Usage: " << argv[0] << " <config_file> ");
        return -1;
    }
    if (argc > 2) {
        LOG_ERROR( "Warning: Too many arguments provided. Ignoring the reset arguments" << RESET);
    }

    MacConfig macConfig;
    macConfig.load(argv[1]); // We do not validate the config file.

    LOG_INFO("MAC configuration file is loaded from " << argv[1]);

    // Check if the output directory exists, if not, create it
    if (std::error_code ec; std::filesystem::exists(macConfig.outputPath, ec)) {
        LOG_WARNING(
            "Output directory already exists: " << macConfig.outputPath << ". Existing files may be overwritten."
            << RESET << std::endl << "Press anything to continue, or ctrl + c to exit.");
        std::cin.get();
    } else {
        if (!std::filesystem::create_directory(macConfig.outputPath)) {
            LOG_ERROR("Error creating output directory: " << macConfig.outputPath);
            return -1;
        }
        LOG_INFO("Output directory created: " << macConfig.outputPath);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// Set system
    ///
    ////////////////////////////////////////////////////////////////
    // Setting up threads number for OpenBLAS and OpenMP
    settingThreads(macConfig.desiredThreads); // Set up the threads for OpenBLAS and OpenMP

    // Start registration
    // Create data and result structures
    MacData macData;
    MacResult macResult;

    // 执行配准算法，所有结果现在统一存储在macResult中
    const bool flagEstimateSuccess = registration(macConfig, macData, macResult);


    std::ofstream resultsOut;
    // Output the evaluation results
    if (flagEstimateSuccess) {

        // std::string evaResultPath = macConfig.outputPath + "/evaluation_result.txt";
        // resultsOut.open(evaResultPath.c_str(), std::ios::out);
        // resultsOut.setf(std::ios::fixed, std::ios::floatfield);
        // resultsOut << std::setprecision(6) << "RE: " << macResult.RE << std::endl
        //         << "TE: " << macResult.TE << std::endl
        //         << "Correct estimated correspondences: " << macResult.correctEstNum << std::endl
        //         << "Inliers in ground truth correspondences: " << macResult.gtInlierNum << std::endl
        //         << "Total correspondences: " << macData.totalCorresNum << std::endl
        //         << "Time taken for registration: " << macResult.timeEpoch << " seconds" << std::endl;
        // resultsOut.close();
        std::cout << GREEN << "Registration successful" << RESET << std::endl;
    } else {
        std::cout << YELLOW << "Registration failed" << RESET << std::endl;
    }

    // Output the status of the registration process
    // std::string statusPath = macConfig.outputPath + "/status.txt";
    // resultsOut.open(statusPath.c_str(), std::ios::out);
    // resultsOut.setf(std::ios::fixed, std::ios::floatfield);
    // resultsOut << std::setprecision(6) << "Time in one iteration: " << macResult.timeEpoch <<
    //         " seconds, memory used in one iteration: " << std::endl;
    // resultsOut.close();


    return 0;
}
