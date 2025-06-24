//

#ifndef _MAC_SHARP_
#define _MAC_SHARP_


#define constE 2.718282
#define NULL_POINTID -1
#define NULL_Saliency -1000
#define Random(x) (rand()%x)
#define Corres_view_gap -200
#define Align_precision_threshold 0.1
#define tR 116//30
#define tG 205//144
#define tB 211//255
#define sR 253//209//220
#define sG 224//26//20
#define sB 2//32//60
#define L2_thresh 0.5
#define Ratio_thresh 0.2
#define GC_dist_thresh 3
#define Hough_bin_num 15
#define SI_GC_thresh 0.8
#define RANSAC_Iter_Num 5000
#define GTM_Iter_Num 100
#define CV_voting_size 20
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
using namespace std;

extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;
//
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <unordered_set>
#include <Eigen/Eigen>
#include <igraph/igraph.h>
#include <sys/stat.h>
// #include <unistd.h>
#include <pcl/segmentation/impl/conditional_euclidean_clustering.hpp>
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;




typedef struct {
    int src_index;
    int tgt_index;
    pcl::PointXYZ src;
    pcl::PointXYZ tgt;
    Eigen::Vector3f src_norm;
    Eigen::Vector3f tgt_norm;
    float score;
    int inlier_weight;
}Corre_3DMatch;

typedef struct
{
    int current_index;
    int degree;
    float score;
    vector<int> correspondences_index;
    int true_num;
}Vote_exp; // for degree calculation

typedef struct Vote
{
    int current_index;
    float score;
    bool flag;

    // --- 添加构造函数 ---
    // 提供一个默认构造函数，以防需要创建空对象
    Vote() : current_index(0), score(0.0f), flag(false) {}

    // 这正是 emplace_back 需要的构造函数！
    Vote(const int idx, const float scr, const bool flg)
        : current_index(idx), score(scr), flag(flg) {}
}Vote; // for cluster factor, for evaluation_est



typedef struct local{
    int current_ind; // clique index
    vector<Vote>clique_ind_score;
    float score;

    // default constructor
    local() : current_ind(0), score(0.0f) {
        clique_ind_score.clear();
    }
    // constructor
    local(int ind = 0, float scr = 0.0f) : current_ind(ind), score(scr) {
        clique_ind_score.clear();
    }
}local; // for

#endif