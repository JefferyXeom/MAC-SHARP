//// For input/output operations and system call wrappers
#include <iostream>
#include <unistd.h> // For POSIX system calls (System dependent)
//// For string operations
#include <string> // For string operations
//// For exit function
#include <cstdlib> // For exit functio
//// For timing
#include <chrono>

// for PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

// 
#include "MAC_SHARP.hpp"



// Terminal color codes for output
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"



int main(int argc, char** argv) {
    // Check if the required arguments are provided
    if (argc < 9) {
        std::cerr << RED << "Error: Not enough arguments provided. " << RESET << std::endl <<  "Usage: " << argv[0] << " <dataset_name> <descriptor> <src_cloud_path> <tgt_cloud_path> <corr_path> <gt_label_path> <gt_tf_path> <output_path>" << std::endl;
        return -1;
    }

    std::string dataset_name(argv[1]); // 
    std::string descriptor(argv[2]); // descriptor name, e.g., "SHOT", "FPFH", etc.
    std::string src_cloud_path(argv[3]); // source point cloud file path
    std::string tgt_cloud_path(argv[4]); // target point cloud file path
    std::string corr_path(argv[5]); // correspondence file path
    std::string gt_label_path(argv[6]); // ground truth label file path, indicating which correspondences are inliers
    std::string gt_tf_path(argv[7]); // ground truth transformation file path
    std::string output_path(argv[8]); // output path for results
    
    // Check if the output directory exists, if not, create it
    if (access(output_path.c_str(), F_OK)){
        if (mkdir(output_path.c_str(), S_IRWXU)) {
            std::cerr << "Error creating output directory: " << output_path << std::endl;
            return -1;
        }
    } else{
        std::cout << YELLOW << "Warning: Output directory already exists: " << output_path << ". Existing files may be overwritten." << std::endl << "Press anything to continue, or ctrl + c to exit." << RESET << std::endl;
        std::cin.get();
    }


    int iterations = 1; // Number of iterations for ICP
    for  (int i = 0; i < iterations; ++i) {
        double time_epoch = 0.0; // ?

        float RE, TE; // Rotation and translation errors
        int correct_est_num = 0; // Number of correct estimated correspondences
    }


    // Load source and target point clouds

    return 0;
}
