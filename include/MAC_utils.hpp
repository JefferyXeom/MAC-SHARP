//
// Created by Jeffery_Xeom on 2025/6/19.
//

#ifndef _MAC_UTILS_
#define _MAC_UTILS_



// Use an enumeration type to clearly represent the formula to be used
enum class ScoreFormula {
    GAUSSIAN_KERNEL,
    QUADRATIC_FALLOFF
};


// General type define
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

// Point cloud correspondences structure
// For variable correspondences
typedef struct {
    int src_index;
    int tgt_index;
    pcl::PointXYZ src;
    pcl::PointXYZ tgt;
    Eigen::Vector3f src_norm;
    Eigen::Vector3f tgt_norm;
    float score;
    int inlier_weight;
}Correspondence_Struct;

//
// C++17 Standard flag , ^ 17+, v 98+
inline extern int total_correspondences_num = 0; // Total number of correspondences
// extern int total_correspondences_num = 0; // Total number of correspondences



// Timer class should be carefully checked
class Timer {
public:
    Timer() = default;

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(end_time_ - start_time_).count();
        elapsed_times_.push_back(elapsed);
        std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;
    }

    const std::vector<double>& getElapsedTimes() const {
        return elapsed_times_;
    }

    void reset() {
        elapsed_times_.clear();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_, end_time_;
    std::vector<double> elapsed_times_;
};


// Functions declaration
float mesh_resolution_calculation(const PointCloudPtr &pointcloud);
void find_index_for_correspondences(PointCloudPtr &src, PointCloudPtr &tgt, std::vector<Correspondence_Struct> &correspondences);
inline float get_distance(const pcl::PointXYZ &A, const pcl::PointXYZ &B);
Eigen::MatrixXd graph_construction(std::vector<Correspondence_Struct> &correspondences, float resolution, bool second_order_graph_flag);



#endif //_MAC_UTILS_
