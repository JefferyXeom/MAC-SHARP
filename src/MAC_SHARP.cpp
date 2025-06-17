//// For input/output operations and system call wrappers
#include <iostream>
#include <filesystem>
//// For string operations
#include <string> // For string operations
//// For exit function
#include <cstdlib> // For exit function
//// For timing
#include <chrono>

// for PCL
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>

// 
#include "MAC_SHARP.hpp"

#include <__msvc_filebuf.hpp>


// Terminal color codes for output
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

// const std::string RED = "\x1b[91m";
// const std::string GREEN = "\x1b[92m";
// const std::string YELLOW = "\x1b[93m";
// const std::string BLUE = "\x1b[94m";
// const std::string RESET = "\x1b[0m"; // 恢复默认颜色

bool low_inlier_ratio = false; // Flag for low inlier ratio
bool add_overlap = false; // Flag for adding overlap, maybe deprecated in future versions
bool no_logs = false; // Flag for no logs



bool registration(const std::string &src_pointcloud_path, const std::string &tgt_pointcloud_path,
                  const std::string &corr_path, const std::string &gt_label_path, const std::string &gt_tf_path,
                  const std::string &output_path, const std::string &descriptor, double &RE, double &TE,
                  int &correct_est_num, int &gt_inlier_num, int &total_num, double &time_epoch) {
    bool second_order_graph = true;
    bool use_icp = true;
    bool instance_equal = true;
    bool cluster_internal_evaluation = true;
    bool use_top_k = false;
    int max_estimate_num = INT_MAX; // ?
    low_inlier_ratio = false;
    add_overlap = false;
    no_logs = false;
    std::string metric = "MAC_SHARP";
    omp_set_num_threads(omp_get_max_threads() - 2); // Set the number of threads for OpenMP, minus 2 to avoid overloading the system
    int success_num = 0; // Number of successful registrations

    std::cout << BLUE << "Output path: " << output_path << RESET << std::endl;
    std::string input_data_path = corr_path.substr(0, corr_path.rfind('/'));
    std::string item_name = output_path.substr(output_path.rfind('/'), output_path.length());

    std::vector<std::pair<int, std::vector<int>>> matches; // one2k_match

    FILE* corr_file, * gt;
    corr_file = fopen(corr_path.c_str(), "r");
    gt = fopen(gt_label_path.c_str(), "r");

    if (corr_file == NULL) {
        std::cout << " error in loading correspondence data. " << std::endl;
        cout << corr_path << endl;
        exit(-1);
    }
    if (gt == NULL) {
        std::cout << " error in loading ground truth label data. " << std::endl;
        cout << gt_label_path << endl;
        exit(-1);
    }


    // overlap is deprecated, but kept for compatibility

    // FILE* ov;
    // std::vector<float>ov_corr_label;
    // float max_corr_weight = 0;
    // if (add_overlap && ov_label != "NULL")
    // {
    //     ov = fopen(ov_label.c_str(), "r");
    //     if (ov == NULL) {
    //         std::cout << " error in loading overlap data. " << std::endl;
    //         exit(-1);
    //     }
    //     cout << ov_label << endl;
    //     while (!feof(ov))
    //     {
    //         float value;
    //         fscanf(ov, "%f\n", &value);
    //         if(value > max_corr_weight){
    //             max_corr_weight = value;
    //         }
    //         ov_corr_label.push_back(value);
    //     }
    //     fclose(ov);
    //     cout << "load overlap data finished." << endl;
    // }

    // Load source and target point clouds
    PointCloudPtr raw_src(new pcl::PointCloud<pcl::PointXYZ>); // may not be used
    PointCloudPtr raw_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    float raw_src_resolution = 0.0f;
    float raw_tgt_resolution = 0.0f;

    PointCloudPtr pointcloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr pointcloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr pointcloud_src_kpts(new pcl::PointCloud<pcl::PointXYZ>); // source point cloud keypoints
    PointCloudPtr pointcloud_tgt_kpts(new pcl::PointCloud<pcl::PointXYZ>); // target point cloud keypoints

    pcl::PointCloud<pcl::Normal>::Ptr normal_src(new pcl::PointCloud<pcl::Normal>); // normal vector
    pcl::PointCloud<pcl::Normal>::Ptr normal_tgt(new pcl::PointCloud<pcl::Normal>); // normal vector

    std::vector<Corre_3DMatch> correspondence; // vector to store correspondences
    std::vector<int> gt_score; // ground truth scores
    int inlier_num = 0; // Initialize inlier number
    float resolution = 0.0f; // Initialize resolution
    Eigen::Matrix4f gt_mat; // Ground truth transformation matrix

    FILE *gt_tf_file = fopen(gt_tf_path.c_str(), "r");
    if (gt_tf_file == NULL) {
        std::cerr << RED << "Error: Unable to open ground truth transformation file: " << gt_tf_path << RESET << std::endl;
        return false;
    }
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(0, 0), &gt_mat(0, 1), &gt_mat(0, 2), &gt_mat(0, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(1, 0), &gt_mat(1, 1), &gt_mat(1, 2), &gt_mat(1, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(2, 0), &gt_mat(2, 1), &gt_mat(2, 2), &gt_mat(2, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(3, 0), &gt_mat(3, 1), &gt_mat(3, 2), &gt_mat(3, 3));
    fclose(gt_tf_file);

    if (pcl::io::loadPCDFile(src_pointcloud_path.c_str(),  *pointcloud_src) < 0) {
        std::cout << RED << "Error: Unable to load source point cloud file: " << src_pointcloud_path << RESET << std::endl;
        return false;
    }
    if (pcl::io::loadPCDFile(tgt_pointcloud_path.c_str(),  *pointcloud_tgt) < 0) {
        std::cout << RED << "Error: Unable to load target point cloud file: " << tgt_pointcloud_path << RESET << std::endl;
        return false;
    }

    while (!feof(corr_file)) {
        Corre_3DMatch match;
        p
    }


    // if (low_inlier_ratio) {
    //     if )
    //
    // }









    return false;
}


int main(int argc, char **argv) {
    // Check if the required arguments are provided
    if (argc < 9) {
        std::cerr << RED << "Error: Not enough arguments provided. " << RESET << std::endl;
        std::cout << "Usage: " << argv[0] <<
                " <dataset_name> <descriptor> <src_pointcloud_path> <tgt_pointcloud_path> <corr_path> <gt_label_path> <gt_tf_path> <output_path>"
                << std::endl;
        return -1;
    }

    std::string dataset_name(argv[1]); // dataset name, previously used for different parameter settings. Evaluation metrics
    std::string descriptor(argv[2]); // descriptor name, e.g., "SHOT", "FPFH", etc.
    std::string src_pointcloud_path(argv[3]); // source point cloud file path
    std::string tgt_pointcloud_path(argv[4]); // target point cloud file path
    std::string corr_path(argv[5]); // correspondence file path
    std::string gt_label_path(argv[6]); // ground truth label file path, indicating which correspondences are inliers
    std::string gt_tf_path(argv[7]); // ground truth transformation file path
    std::string output_path(argv[8]); // output path for results

    // Check if the output directory exists, if not, create it
    std::error_code ec;
    if (std::filesystem::exists(output_path.c_str(), ec)) {
        if (std::filesystem::create_directory(output_path.c_str())) {
            std::cerr << "Error creating output directory: " << output_path << std::endl;
            return -1;
        }
    } else {
        std::cout << YELLOW << "Warning: Output directory already exists: " << output_path
                << ". Existing files may be overwritten." << std::endl
                << "Press anything to continue, or ctrl + c to exit." << RESET << std::endl;
        std::cin.get();
    }

    // Start execution
    int iterations = 1; // Number of iterations for ICP
    for (int i = 0; i < iterations; ++i) {
        double time_epoch = 0.0; // ?

        double RE, TE; // Rotation and translation errors
        int correct_est_num = 0; // Number of correct estimated correspondences
        int gt_inlier_num = 0; // Number of inliers in the ground truth correspondences
        int total_num = 0; // Total number of correspondences
        bool estimate_success = registration(src_pointcloud_path, tgt_pointcloud_path, corr_path, gt_label_path, gt_tf_path,
                                             output_path,
                                             descriptor, RE, TE, correct_est_num, gt_inlier_num, total_num, time_epoch);

        std::ofstream results_out;
        // Output the evaluation results
        if (estimate_success) {
            std::string eva_result_path = output_path + "/evaluation_result.txt";
            results_out.open(eva_result_path.c_str(), std::ios::out);
            results_out.setf(std::ios::fixed, std::ios::floatfield);
            results_out << std::setprecision(6) << "RE: " << RE << std::endl
                    << "TE: " << TE << std::endl
                    << "Correct estimated correspondences: " << correct_est_num << std::endl
                    << "Inliers in ground truth correspondences: " << gt_inlier_num << std::endl
                    << "Total correspondences: " << total_num << std::endl
                    << "Time taken for registration: " << time_epoch << " seconds" << std::endl;
            results_out.close();
        }

        // Output the status of the registration process
        std::string status_path = output_path + "/status.txt";
        results_out.open(status_path.c_str(), std::ios::out);
        results_out.setf(std::ios::fixed, std::ios::floatfield);
        results_out << std::setprecision(6) << "Time in one iteration: " << time_epoch << " seconds, memory used in one iteration: " << std::endl;
        results_out.close();
    }



    return 0;
}
