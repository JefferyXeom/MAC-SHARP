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
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

// Windows system api
#include <__msvc_filebuf.hpp>
#include <cblas.h>
#include <process.h>


//
#include "MAC_SHARP.hpp"

#include "MAC_utils.hpp"


// const std::string RED = "\x1b[91m";
// const std::string GREEN = "\x1b[92m";
// const std::string YELLOW = "\x1b[93m";
// const std::string BLUE = "\x1b[94m";
// const std::string RESET = "\x1b[0m"; // 恢复默认颜色

bool low_inlier_ratio = false; // Flag for low inlier ratio
bool add_overlap = false; // Flag for adding overlap, maybe deprecated in future versions
bool no_logs = false; // Flag for no logs


int clique_num = 0; // Number of cliques found


// Timing
// Only consider one iteration of the registration process!
std::chrono::high_resolution_clock::time_point start_time, end_time;
std::chrono::duration<double> elapsed_time;
std::vector<double> time_vec; // Vector to store elapsed times for each iteration

void timing(const int time_flag) {
    if (time_flag == 0) {
        // Start timing
        start_time = std::chrono::high_resolution_clock::now();
    } else if (time_flag == 1) {
        // End timing and calculate elapsed time
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = end_time - start_time;
        std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
        time_vec.push_back(elapsed_time.count()); // Store elapsed time in vector
    }
}




// TODO: This function needs optimization
float otsu_thresh(vector<float> all_scores)
{
    int i;
    int Quant_num = 100;
    float score_sum = 0.0;
    float fore_score_sum = 0.0;
    std::vector<int> score_Hist(Quant_num, 0);
    std::vector<float> score_sum_Hist(Quant_num, 0.0);
    float max_score_value, min_score_value;
    for (i = 0; i < all_scores.size(); i++)
    {
        score_sum += all_scores[i];
    }
    sort(all_scores.begin(), all_scores.end());
    max_score_value = all_scores[all_scores.size() - 1];
    min_score_value = all_scores[0];
    float Quant_step = (max_score_value - min_score_value) / Quant_num;
    for (i = 0; i < all_scores.size(); i++)
    {
        int ID = all_scores[i] / Quant_step;
        if (ID >= Quant_num) ID = Quant_num - 1;
        score_Hist[ID]++;
        score_sum_Hist[ID] += all_scores[i];
    }
    float fmax = -1000;
    int n1 = 0, n2;
    float m1, m2, sb;
    float thresh = (max_score_value - min_score_value) / 2;//default value
    for (i = 0; i < Quant_num; i++)
    {
        float Thresh_temp = i * (max_score_value - min_score_value) / float (Quant_num);
        n1 += score_Hist[i];
        if (n1 == 0) continue;
        n2 = all_scores.size() - n1;
        if (n2 == 0) break;
        fore_score_sum += score_sum_Hist[i];
        m1 = fore_score_sum / n1;
        m2 = (score_sum - fore_score_sum) / n2;
        sb = (float )n1 * (float )n2 * pow(m1 - m2, 2);
        if (sb > fmax)
        {
            fmax = sb;
            thresh = Thresh_temp;
        }
    }
    return thresh;
}



// Comparison functions
bool compare_vote_score(const Vote& v1, const Vote& v2) {
    return v1.score > v2.score;
}
bool compare_local_score(const local_clique &l1, const local_clique &l2){
    return l1.score > l2.score;
}
bool compare_corres_ind(const Correspondence_Struct& c1, const Correspondence_Struct& c2){
    return c1.tgt_index < c2.tgt_index;
}

// Find the vertex score based on clique edge weight.
// Select the correspondences who have high scores
// sampled_ind is the order of the correspondences that are selected which score is higher than average
// remain is the index of the neighbor of sampled_ind that also locate in the high score clique
void clique_sampling(Eigen::MatrixXd &graph, const igraph_vector_int_list_t *cliques, std::vector<int> &sampled_ind, std::vector<int> &remain){
    // the clear process may be rebundant
    // remain.clear();
    // sampled_ind.clear();
    unordered_set<int> visited;
    std::vector<local_clique> result(total_correspondences_num);
    // Assign current index
#pragma omp parallel for
    for(int i = 0; i < total_correspondences_num; i++){
        result[i].current_ind = i;
    }
    // compute the weight of each clique
    // Weight of each clique is the sum of the weights of all edges in the clique
#pragma omp parallel for
    for(int i = 0; i < clique_num; i++){
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(cliques, i);
        double weight = 0.0;
        int length = igraph_vector_int_size(v); // size of the clique

        for (int j = 0; j < length; j++)
        {
            int a = static_cast<int>(VECTOR(*v)[j]);
            for (int k = j + 1; k < length; k++)
            {
                int b = static_cast<int>(VECTOR(*v)[k]);
                weight += graph(a, b);
            }
        }
        // assign the weight to each correspondence in the clique
        for (int j = 0; j < length; j++)
        {
            int k = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            result[k].clique_ind_score.emplace_back(i, weight, false); // Weight of k-th correspondecnce in i-th clique
        }
    }

    float avg_score = 0;
    // sum the scores and assign it to the score member variable
#pragma omp parallel for
    for(int i = 0; i < total_correspondences_num; i++){
        result[i].score = 0;
        // compute the score of each correspondence, clique_ind_score.size() is the number of cliques that the correspondence belongs to
        for(int j = 0; j < result[i].clique_ind_score.size(); j ++){
            result[i].score += result[i].clique_ind_score[j].score;
        }
#pragma omp critical
        {
            avg_score += result[i].score;
        }
    }

    //
    sort(result.begin(), result.end(), compare_local_score); //所有节点从大到小排序

    if( clique_num <= total_correspondences_num ){ // 如果clique数目小于等于correspondence数目
        for(int i = 0; i < clique_num; i++){ // Assign all cliques indexes to the remain in order.
            remain.push_back(i);
        }
        for(int i = 0; i < total_correspondences_num; i++){ // sampled_ind 中存放的是被选中的correspondence的index
            if(!result[i].score){ // skip if the score of correspondence is 0
                continue;
            }
            sampled_ind.push_back(result[i].current_ind); // only keep index whose correspondence has a non-zero score
        }
        return;
    }

    //
    avg_score /= static_cast<float>(total_correspondences_num);
    int max_cnt = 10;  //default 10
    for(int i = 0; i < total_correspondences_num; i++){
        // We only consider the correspondences whose score is greater than the average score
        // This can filter low score vertex (vertex and correspondence are the same thing)
        if(result[i].score < avg_score) break;
        sampled_ind.push_back(result[i].current_ind); // Only keep index of correspondence whose score is higher than the average score, ordered
        // sort the clique_ind_score of each correspondence from large to small
        sort(result[i].clique_ind_score.begin(), result[i].clique_ind_score.end(), compare_vote_score); //局部从大到小排序
        int selected_cnt = 1;
        // Check top 10 neighbors of each correspondence in high score clique
        for(int j = 0; j < result[i].clique_ind_score.size(); j++){
            if(selected_cnt > max_cnt) break;
            int ind = result[i].clique_ind_score[j].current_index;
            if(visited.find(ind) == visited.end()){
                visited.insert(ind);
            }
            else{
                continue;
            }
            selected_cnt ++;
        }
    }
    // Keep the correspondences that have high neighboring score.
    // Its neighbor has high score, and it is in its neighbor's high score clique
    remain.assign(visited.begin(), visited.end()); // no order
}

// TODO: Chech this function
// TODO: This function is not optimized
// Our source target pair is a normal but non-invertible function (surjective, narrowly), which means a source can only have a single target,
// but a target may have many sources. This function is used to find target source pair, where target paired with various sources.
void make_tgt_src_pair(const std::vector<Correspondence_Struct>& correspondence, std::vector<pair<int, std::vector<int>>>& tgt_src){ //需要读取保存的kpts, 匹配数据按照索引形式保存
    assert(correspondence.size() > 1); // 保留一个就行
    if (correspondence.size() < 2) {
        std::cerr << "The correspondence vector is empty." << std::endl;
    }
    tgt_src.clear();
    std::vector<Correspondence_Struct> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    sort(corr.begin(), corr.end(), compare_corres_ind); // sort by target index increasing order
    int tgt = corr[0].tgt_index;
    std::vector<int>src;
    src.push_back(corr[0].src_index);
    for(int i = 1; i < corr.size(); i++){
        if(corr[i].tgt_index != tgt){
            tgt_src.emplace_back(tgt, src);
            src.clear();
            tgt = corr[i].tgt_index;
        }
        src.push_back(corr[i].src_index);
    }
    corr.clear();
    corr.shrink_to_fit();
}

// TODO: This function is not optimized
// TODO: We only get the logic check
void weight_svd(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXf& weights, float weight_threshold, Eigen::Matrix4f& trans_Mat) {
    for (int i = 0; i < weights.size(); i++)
    {
        weights(i) = (weights(i) < weight_threshold) ? 0 : weights(i);
    }
    //weights升维度
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> weight;
    Eigen::VectorXf ones = weights;
    ones.setOnes();
    weight = (weights * ones.transpose());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;
    //构建对角阵
    Identity.setIdentity();
    weight = (weights * ones.transpose()).cwiseProduct(Identity);
    pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_pts);
    pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_pts);
    //获取点云质心
    src_it.reset(); des_it.reset();
    Eigen::Matrix<float, 4, 1> centroid_src, centroid_des;
    pcl::compute3DCentroid(src_it, centroid_src);
    pcl::compute3DCentroid(des_it, centroid_des);

    //去除点云质心
    src_it.reset(); des_it.reset();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
    pcl::demeanPointCloud(src_it, centroid_src, src_demean);
    pcl::demeanPointCloud(des_it, centroid_des, des_demean);

    //计算加权协方差矩阵
    Eigen::Matrix<float, 3, 3> H = (src_demean * weight * des_demean.transpose()).topLeftCorner(3, 3);
    //cout << H << endl;

    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<float, 3, 3> u = svd.matrixU();
    Eigen::Matrix<float, 3, 3> v = svd.matrixV();

    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0)
    {
        for (int x = 0; x < 3; ++x)
            v(x, 2) *= -1;
    }

    Eigen::Matrix<float, 3, 3> R = v * u.transpose(); //正交矩阵的乘积还是正交矩阵，因此R的逆等于R的转置

    // Return the correct transformation
    Eigen::Matrix<float, 4, 4> Trans;
    Trans.setIdentity();
    Trans.topLeftCorner(3, 3) = R;
    const Eigen::Matrix<float, 3, 1> Rc(R * centroid_src.head(3));
    Trans.block(0, 3, 3, 1) = centroid_des.head(3) - Rc;
    trans_Mat = Trans;
}



// TODO: This function is not optimized
// TODO: We only get the logic check
// Overall Average Mean Absolute Error（OAMAE）
// | 指标               | 说明                     | 是否对异常敏感    |
// | ---------------- | ---------------------- | ---------- |
// | **OAMAE**        | 总体平均的绝对误差              | 否（比RMSE稳健） |
// | RMSE             | 平方误差平均，强调大误差           | 是          |
// | Chamfer Distance | 最近邻距离之和或平均             | 常用于点云任务    |
// | EMD              | Earth Mover's Distance | 更严谨但计算开销大  |

float OAMAE(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &des_src, float thresh){
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for(auto & i : des_src){
        int des_ind = i.first;
        vector<int> src_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        for(auto & e : src_ind){
            if(!pcl::isFinite(src_trans->points[e])) continue;
            //计算距离
            float distance = get_distance(src_trans->points[e], raw_des->points[des_ind]);
            if (distance < thresh)
            {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}

float calculate_rotation_error(Eigen::Matrix3f& est, Eigen::Matrix3f& gt) {
    float tr = (est.transpose() * gt).trace();
    return acos(min(max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

float calculate_translation_error(Eigen::Vector3f& est, Eigen::Vector3f& gt) {
    Eigen::Vector3f t = est - gt;
    return sqrt(t.dot(t)) * 100;
}

float evaluation_trans(vector<Correspondence_Struct>& correspondnece, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& trans, float metric_thresh, const string &metric, float resolution) {
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src_corr_pts, *src_trans, trans);
    src_trans->is_dense = false;
    vector<int>mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if(!src_trans->size()) return 0;
    float score = 0.0;
    int inlier = 0;
    int corr_num = src_corr_pts->points.size();
    for (int i = 0; i < corr_num; i++)
    {
        float dist = get_distance(src_trans->points[i], des_corr_pts->points[i]);
        float w = 1;
        if (add_overlap)
        {
            w = correspondnece[i].score;
        }
        if (dist < metric_thresh)
        {
            inlier++;
            if (metric == "inlier")
            {
                score += 1*w;//correspondence[i].inlier_weight; <- commented by the MAC++ author
            }
            else if (metric == "MAE")
            {
                score += (metric_thresh - dist)*w / metric_thresh;
            }
            else if (metric == "MSE")
            {
                score += pow((metric_thresh - dist), 2)*w / pow(metric_thresh, 2);
            }
        }
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}


bool evaluation_est(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float re_thresh, float te_thresh, double& RE, double& TE) {
    Eigen::Matrix3f rotation_est, rotation_gt;
    Eigen::Vector3f translation_est, translation_gt;
    rotation_est = est.topLeftCorner(3, 3);
    rotation_gt = gt.topLeftCorner(3, 3);
    translation_est = est.block(0, 3, 3, 1);
    translation_gt = gt.block(0, 3, 3, 1);

    RE = calculate_rotation_error(rotation_est, rotation_gt);
    TE = calculate_translation_error(translation_est, translation_gt);
    if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh)
    {
        return true;
    }
    return false;
}


// ################################################################
float g_angleThreshold = 5.0 * M_PI / 180;//5 degree
float g_distanceThreshold = 0.1;
#ifndef M_PIf32
#define M_PIf32 3.1415927f
#endif

bool EnforceSimilarity1(const pcl::PointXYZINormal &point_a, const pcl::PointXYZINormal &point_b, float squared_distance){
    if(point_a.normal_x == 666 || point_b.normal_x == 666 || point_a.normal_y == 666 || point_b.normal_y == 666 || point_a.normal_z == 666 || point_b.normal_z == 666){
        return false;
    }
    Eigen::VectorXf temp(3);
    temp[0] = point_a.normal_x - point_b.normal_x;
    temp[1] = point_a.normal_y - point_b.normal_y;
    temp[2] = point_a.normal_z - point_b.normal_z;
    if(temp.norm() < g_distanceThreshold){
        return true;
    }
    return false;
}

// Check if the Euler angles are within the valid range
bool checkEulerAngles(float angle){
    if(isfinite(angle) && angle >= -M_PIf32 && angle <= M_PIf32){
        return true;
    }
    return false;
}

int clusterTransformationByRotation(vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts, float angle_thresh,float dis_thresh,  pcl::IndicesClusters &clusters, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans){
    if(Rs.empty() || Ts.empty() || Rs.size() != Ts.size()){
        std::cout << YELLOW << "Rs and Ts are empty or not the same size!" << RESET << std::endl;
        return -1;
    }
    int num = Rs.size();
    g_distanceThreshold = dis_thresh;
    trans->resize(num);
    for (size_t i = 0; i < num; i++) {
        Eigen::Transform<float, 3, Eigen::Affine> R(Rs[i]);
        pcl::getEulerAngles<float>(R, (*trans)[i].x, (*trans)[i].y, (*trans)[i].z); // R -> trans
        // 去除无效解
        if(!checkEulerAngles((*trans)[i].x) || !checkEulerAngles((*trans)[i].y) || !checkEulerAngles((*trans)[i].z)){
            cout << "INVALID POINT" << endl;
            (*trans)[i].x = 666;
            (*trans)[i].y = 666;
            (*trans)[i].z = 666;
            (*trans)[i].normal_x = 666;
            (*trans)[i].normal_y = 666;
            (*trans)[i].normal_z = 666;
        }
        else{ // 需要解决同一个角度的正负问题 6.14   平面 y=PI 右侧的解（需要验证） 6.20
            // -pi - pi -> 0 - 2pi
            (*trans)[i].x = ((*trans)[i].x < 0 && (*trans)[i].x >= -M_PIf32) ? (*trans)[i].x + 2*M_PIf32 : (*trans)[i].x;
            (*trans)[i].y = ((*trans)[i].y < 0 && (*trans)[i].y >= -M_PIf32) ? (*trans)[i].y + 2*M_PIf32 : (*trans)[i].y;
            (*trans)[i].z = ((*trans)[i].z < 0 && (*trans)[i].z >= -M_PIf32) ? (*trans)[i].z + 2*M_PIf32 : (*trans)[i].z;
            (*trans)[i].normal_x = (float)Ts[i][0];
            (*trans)[i].normal_y = (float)Ts[i][1];
            (*trans)[i].normal_z = (float)Ts[i][2];
        }
    }

    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true); // true for using dense mode, no NaN points
    cec.setInputCloud(trans);
    cec.setConditionFunction(&EnforceSimilarity1);
    cec.setClusterTolerance(angle_thresh);
    cec.setMinClusterSize(2); // cluster size
    cec.setMaxClusterSize(static_cast<int>(num)); // nearlly impossible to reach the maximum?
    cec.segment(clusters);
    for (int i = 0; i < clusters.size (); ++i)
    {
        for (int j = 0; j < clusters[i].indices.size (); ++j) { // Set intensity of each cluster point to their cluster number
        }
    }
    return 0;
}


float RMSE_compute(const PointCloudPtr& cloud_source, const PointCloudPtr& cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float mr)
{
    float RMSE_temp = 0.0f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
    vector<int>overlap_idx; float overlap_thresh = 4 * mr;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
    pcl::PointXYZ query_point;
    vector<int>pointIdx;
    vector<float>pointDst;
    kdtree1.setInputCloud(cloud_target);
    for (int i = 0; i < cloud_source_trans_GT->points.size(); i++)
    {
        query_point = cloud_source_trans_GT->points[i];
        kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
        if (sqrt(pointDst[0]) <= overlap_thresh)
            overlap_idx.push_back(i);
    }
    //
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
    kdtree2.setInputCloud(cloud_source_trans_GT);
    for (int i = 0; i < overlap_idx.size(); i++)
    {
        //query_point = cloud_source_trans_EST->points[overlap_idx[i]];
        //kdtree2.nearestKSearch(query_point,1,pointIdx,pointDst); RMSE_temp+=sqrt(pointDst[0]);
        float dist_x = pow(cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
        float dist_y = pow(cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
        float dist_z = pow(cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
        float dist = sqrt(dist_x + dist_y + dist_z);
        RMSE_temp += dist;
    }
    RMSE_temp /= overlap_idx.size();
    RMSE_temp /= mr;
    //
    return RMSE_temp;
}


void post_refinement(vector<Correspondence_Struct>&correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& initial/* 由最大团生成的变换 */, float& best_score, float inlier_thresh, int iterations, const string &metric) {
    int pointNum = src_corr_pts->points.size();
	float pre_score = best_score;
	for (int i = 0; i < iterations; i++)
	{
		float score = 0;
		Eigen::VectorXf weights, weight_pred;
		weights.resize(pointNum);
		weights.setZero();
		vector<int> pred_inlier_index;
		PointCloudPtr trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*src_corr_pts, *trans, initial);
        //remove nan points
        trans->is_dense = false;
        vector<int>mapping;
        pcl::removeNaNFromPointCloud(*trans, *trans, mapping);
        if(!trans->size()) return;
		for (int j = 0; j < pointNum; j++)
		{
			float dist = get_distance(trans->points[j], des_corr_pts->points[j]);
			float w = 1;
			if (add_overlap)
			{
				w = correspondence[j].score;
			}
			if (dist < inlier_thresh)
			{
				pred_inlier_index.push_back(j);
				weights[j] = 1 / (1 + pow(dist / inlier_thresh, 2));
				if (metric == "inlier")
				{
					score+=1*w;
				}
				else if (metric == "MAE")
				{
					score += (inlier_thresh - dist)*w / inlier_thresh;
				}
				else if (metric == "MSE")
				{
					score += pow((inlier_thresh - dist), 2)*w / pow(inlier_thresh, 2);
				}
			}
		}
		if (score < pre_score) {
			break;
		}
		else {
			pre_score = score;
			//估计pred_inlier
			PointCloudPtr pred_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
			PointCloudPtr pred_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::copyPointCloud(*src_corr_pts, pred_inlier_index, *pred_src_pts);
			pcl::copyPointCloud(*des_corr_pts, pred_inlier_index, *pred_des_pts);
			weight_pred.resize(pred_inlier_index.size());
			for (int k = 0; k < pred_inlier_index.size(); k++)
			{
				weight_pred[k] = weights[pred_inlier_index[k]];
			}
			//weighted_svd
			weight_svd(pred_src_pts, pred_des_pts, weight_pred, 0, initial);
			pred_src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
			pred_des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
		}
		pred_inlier_index.clear();
		trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	}
	best_score = pre_score;
}

vector<int> vectors_union(const vector<int>& v1, const vector<int>& v2){
    vector<int> v;
    set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}

void getCorrPatch(vector<Correspondence_Struct>&sampled_corr, PointCloudPtr &src, PointCloudPtr &des, PointCloudPtr &src_batch, PointCloudPtr &des_batch, float radius){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_des;
    kdtree_src.setInputCloud(src);
    kdtree_des.setInputCloud(des);
    vector<int>src_ind, des_ind;
    vector<float>src_dis, des_dis;
    vector<int>src_batch_ind, des_batch_ind;
    for(int i = 0; i < sampled_corr.size(); i++){
        kdtree_src.radiusSearch(sampled_corr[i].src_index, radius, src_ind, src_dis);
        kdtree_des.radiusSearch(sampled_corr[i].tgt_index, radius, des_ind, des_dis);
        sort(src_ind.begin(), src_ind.end());
        sort(des_ind.begin(), des_ind.end());
        src_batch_ind = vectors_union(src_ind, src_batch_ind);
        des_batch_ind = vectors_union(des_ind, des_batch_ind);
    }
    pcl::copyPointCloud(*src, src_batch_ind, *src_batch);
    pcl::copyPointCloud(*des, des_batch_ind, *des_batch);
    return;
}

float trancatedChamferDistance(PointCloudPtr& src, PointCloudPtr& des, Eigen::Matrix4f &est, float thresh){
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, est);
    //remove nan points
    src_trans->is_dense = false;
    vector<int>mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if(!src_trans->size()) return 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(des);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    float score1 = 0, score2 = 0;
    int cnt1 = 0, cnt2 = 0;
    for(int i = 0; i < src_trans->size(); i++){
        pcl::PointXYZ src_trans_query = (*src_trans)[i];
        if(!pcl::isFinite(src_trans_query)) continue;
        kdtree_des.nearestKSearch(src_trans_query, 1, des_ind, des_dis);
        if(des_dis[0] > pow(thresh, 2)){
            continue;
        }
        score1 += (thresh - sqrt(des_dis[0])) / thresh;
        cnt1 ++;
    }
    score1 /= cnt1;
    for(int i = 0; i < des->size(); i++){
        pcl::PointXYZ des_query = (*des)[i];
        if(!pcl::isFinite(des_query)) continue;
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, src_dis);
        if(src_dis[0] > pow(thresh, 2)){
            continue;
        }
        score2 += (thresh - sqrt(src_dis[0])) / thresh;
        cnt2++;
    }
    score2 /= cnt2;
    return (score1 + score2) / 2;
}

vector<int> vectors_intersection(const vector<int>& v1, const vector<int>& v2) {
    vector<int> v;
    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}


float OAMAE_1tok(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &src_des, float thresh){
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for(auto & i : src_des){
        int src_ind = i.first;
        vector<int> des_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        if(!pcl::isFinite(src_trans->points[src_ind])) continue;
        for(auto & e : des_ind){
            //计算距离
            float distance = get_distance(src_trans->points[src_ind], raw_des->points[e]);
            if (distance < thresh)
            {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}


Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial, vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts,
                                        PointCloudPtr& src_kpts, PointCloudPtr& des_kpts, vector<pair<int, vector<int>>> &des_src, float thresh, Eigen::Matrix4f& GTmat, string folderpath){

    //string cluster_eva = folderpath + "/cluster_eva.txt";
    //ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    double RE, TE;
    bool suc = evaluation_est(initial, GTmat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3,3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = OAMAE(src_kpts, des_kpts, initial, des_src, thresh);
    cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    vector<pair<float, float>> RTdifference;
    float avg_Rdiff =0, avg_Tdiff =0;
    int n = 0;
    for(int i = 0; i < clusterTrans[best_index].indices.size(); i++){
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix3f R = Rs[ind];
        Eigen::Vector3f T = Ts[ind];
        float R_diff = calculate_rotation_error(R, R_initial);
        float T_diff = calculate_translation_error(T, T_initial);
        if(isfinite(R_diff) && isfinite(T_diff)){
            avg_Rdiff += R_diff;
            avg_Tdiff += T_diff;
            n++;
        }
        RTdifference.emplace_back(R_diff, T_diff);
    }
    avg_Tdiff /= n;
    avg_Rdiff /= n;

    for(int i = 0; i < clusterTrans[best_index].indices.size(); i++){
        //继续缩小解空间
        if(!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second) || RTdifference[i].first > avg_Rdiff || RTdifference[i].second > avg_Tdiff) continue;
        //if(RTdifference[i].first > 5 || RTdifference[i].second > 10) continue;
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[ind];
        mat.topLeftCorner(3,3) = Rs[ind];
        suc = evaluation_est(mat, GTmat, 15, 30, RE, TE);
        float score = OAMAE(src_kpts, des_kpts, mat, des_src, thresh);
        //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
        if(score > max_score){
            max_score = score;
            est = mat;
            cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE  << ", score = " << score  <<endl;
        }
    }
    //outfile.close();
    return est;
}
// 1tok version
Eigen::Matrix4f clusterInternalTransEva1(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial, vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts,
                                         PointCloudPtr& src_kpts, PointCloudPtr& des_kpts, vector<pair<int, vector<int>>> &des_src, float thresh, Eigen::Matrix4f& GTmat, bool _1tok ,string folderpath){

    //string cluster_eva = folderpath + "/cluster_eva.txt";
    //ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    double RE, TE;
    bool suc = evaluation_est(initial, GTmat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3,3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = 0.0;
    if(_1tok){
        max_score = OAMAE_1tok(src_kpts, des_kpts, initial, des_src, thresh);
    }
    else{
        max_score = OAMAE(src_kpts, des_kpts, initial, des_src, thresh);
    }
    cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    vector<pair<float, float>> RTdifference;
    int n = 0;
    for(int i = 0; i < clusterTrans[best_index].indices.size(); i++){
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix3f R = Rs[ind];
        Eigen::Vector3f T = Ts[ind];
        float R_diff = calculate_rotation_error(R, R_initial);
        float T_diff = calculate_translation_error(T, T_initial);
        RTdifference.emplace_back(R_diff, T_diff);
    }
    ///TODO RTdifference排序
    sort(RTdifference.begin(), RTdifference.end());
    int i = 0, cnt = 10;
    while(i < min(100, (int)clusterTrans[best_index].indices.size()) && cnt > 0){ ///TODO 第一个mat可能与initial一样
        //继续缩小解空间
        if(!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second)) {
            i++;
            continue;
        }
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[ind];
        mat.topLeftCorner(3,3) = Rs[ind];
        if(i > 0 && (est.inverse() * mat - Eigen::Matrix4f::Identity(4, 4)).norm() < 0.01){
            break;
        }
        suc = evaluation_est(mat, GTmat, 15, 30, RE, TE);
        float score = 0.0;
        if (_1tok) {
            score = OAMAE_1tok(src_kpts, des_kpts, mat, des_src, thresh);
        }
        else{
            score = OAMAE(src_kpts, des_kpts, mat, des_src, thresh);
        }

        //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
        if(score > max_score){
            max_score = score;
            est = mat;
            cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE  << ", score = " << score  <<endl;
            cnt--;
        }
        i++;
    }
    //outfile.close();
    return est;
}

bool registration(const std::string &src_pointcloud_path, const std::string &tgt_pointcloud_path,
                  const std::string &corr_path, const std::string &gt_label_path, const std::string &gt_tf_path,
                  const std::string &output_path, const std::string &descriptor, double &RE, double &TE,
                  int &correct_est_num, int &gt_inlier_num, double &time_epoch, vector<double> &pred_inlier) {
    // temporary variables. Delete these after unifying the data load
    std::string src_pointcloud_kpts_path = "./test_data/src.pcd";
    std::string tgt_pointcloud_kpts_path = "./test_data/tgt.pcd";

    bool second_order_graph_flag = true;
    bool use_icp_flag = true;
    bool instance_equal_flag = true;
    bool cluster_internal_evaluation_flag = true;
    bool use_top_k_flag = false;
    int max_estimate_num = INT_MAX; // ?
    low_inlier_ratio = false;
    add_overlap = false;
    no_logs = false;
    std::string metric = "MAE";

    // Configure OpenBLAS threads (May not be used)
    int default_threads = openblas_get_num_threads();
    std::cout << "OpenBLAS default threads: " << default_threads << std::endl;
    int desired_threads = 1;
    openblas_set_num_threads(16);
    std::cout << "OpenBLAS now set to use " << openblas_get_num_threads() << " threads." << std::endl;

    // Configure OpenMP threads
    // 1. 查看默认情况下的最大线程数
    //    这通常等于您机器的逻辑核心数
    default_threads = omp_get_max_threads();
    std::cout << "Default max OpenMP threads: " << default_threads << std::endl;

    // 2. 设置希望使用的线程数
    desired_threads = 16;
    std::cout << "\nSetting OpenMP threads to: " << desired_threads << std::endl;
    omp_set_num_threads(desired_threads);

    // Set the number of threads for OpenMP, minus 2 to avoid overloading the system
    // omp_set_num_threads(omp_get_max_threads() - 2);

    int success_num = 0; // Number of successful registrations

    std::cout << BLUE << "Output path: " << output_path << RESET << std::endl;
    std::string input_data_path = corr_path.substr(0, corr_path.rfind('/'));
    std::string item_name = output_path.substr(output_path.rfind('/'), output_path.length());

    std::vector<std::pair<int, std::vector<int> > > matches; // one2k_match

    FILE *corr_file, *gt;
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

    std::vector<Correspondence_Struct> correspondences; // vector to store correspondences
    std::vector<int> gt_correspondences; // ground truth correspondences
    int inlier_num = 0; // Initialize inlier number
    float resolution = 0.0f; // Initialize resolution
    Eigen::Matrix4f gt_mat; // Ground truth transformation matrix

    FILE *gt_tf_file = fopen(gt_tf_path.c_str(), "r");
    if (gt_tf_file == NULL) {
        std::cerr << RED << "Error: Unable to open ground truth transformation file: " << gt_tf_path << RESET <<
                std::endl;
        return false;
    }
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(0, 0), &gt_mat(0, 1), &gt_mat(0, 2), &gt_mat(0, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(1, 0), &gt_mat(1, 1), &gt_mat(1, 2), &gt_mat(1, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(2, 0), &gt_mat(2, 1), &gt_mat(2, 2), &gt_mat(2, 3));
    fscanf(gt_tf_file, "%f %f %f %f\n", &gt_mat(3, 0), &gt_mat(3, 1), &gt_mat(3, 2), &gt_mat(3, 3));
    fclose(gt_tf_file);

    if (pcl::io::loadPLYFile(src_pointcloud_path, *pointcloud_src) < 0) {
        std::cout << RED << "Error: Unable to load source point cloud file: " << src_pointcloud_path << RESET <<
                std::endl;
        return false;
    }
    if (pcl::io::loadPLYFile(tgt_pointcloud_path, *pointcloud_tgt) < 0) {
        std::cout << RED << "Error: Unable to load target point cloud file: " << tgt_pointcloud_path << RESET <<
                std::endl;
        return false;
    }
    if (pcl::io::loadPCDFile(src_pointcloud_kpts_path, *pointcloud_src_kpts) < 0) {
        std::cout << RED << "Error: Unable to load source point cloud keypoints file: " << src_pointcloud_kpts_path
                  << RESET << std::endl;
        return false;
    }
    if (pcl::io::loadPCDFile(tgt_pointcloud_kpts_path, *pointcloud_tgt_kpts) < 0) {
        std::cout << RED << "Error: Unable to load target point cloud keypoints file: " << tgt_pointcloud_kpts_path
                  << RESET << std::endl;
        return false;
    }

    // Load correspondences
    // TODO: After integrate the keypoints detection and description, the index and xyz coordinates are located in a
    // TODO: single file, so the correspondence loading function should be modified accordingly.
    while (!feof(corr_file)) {
        Correspondence_Struct match;
        pcl::PointXYZ src_point, tgt_point; // source point and target point in each match
        fscanf(corr_file, "%f %f %f %f %f %f\n",
               &src_point.x, &src_point.y, &src_point.z,
               &tgt_point.x, &tgt_point.y, &tgt_point.z);
        match.src = src_point;
        match.tgt = tgt_point;
        match.inlier_weight = 0; // Initialize inlier weight to 0
        correspondences.push_back(match);
    }
    fclose(corr_file);
    find_index_for_correspondences(pointcloud_src_kpts, pointcloud_tgt_kpts, correspondences);
    resolution = (mesh_resolution_calculation(pointcloud_src) + mesh_resolution_calculation(pointcloud_tgt)) / 2;

    // if (low_inlier_ratio) {
    //     if )
    //
    // }

    total_correspondences_num = static_cast<int>(correspondences.size());
    int value = 0;
    while (!feof(gt)) {
        fscanf(gt, "%d\n", &value);
        gt_correspondences.push_back(value);
        if (value == 1) {
            inlier_num++;
        }
    }
    fclose(gt);

    if (inlier_num == 0) {
        std::cout << YELLOW << "Warning: No inliers found in the ground truth correspondences." << RESET << std::endl;
        return false;
    }
    float inlier_ratio = static_cast<float>(inlier_num) / static_cast<float>(total_correspondences_num);
    std::cout << "Inlier ratio: " << inlier_ratio << std::endl;

    ////////////////////////////////
    /// Setting up evaluation thresholds.
    /// Dataset_name is not passed into this function
    /// therefore we set it manually.
    /// TODO: Make dataset_name a parameter of this function

    std::string dataset_name = "3dmatch";
    float RE_eva_thresh, TE_eva_thresh, inlier_eva_thresh;
    if (dataset_name == "KITTI") {
        RE_eva_thresh = 5;
        TE_eva_thresh = 180;
        inlier_eva_thresh = 1.8;
    } else if (dataset_name == "3dmatch" || dataset_name == "3dlomatch") {
        RE_eva_thresh = 15;
        TE_eva_thresh = 30;
        inlier_eva_thresh = 0.1;
    } else if (dataset_name == "U3M") {
        inlier_eva_thresh = 5 * resolution;
    }

    // NOTE: we do not consider the outer loop of the registration process, which is used to
    // repeat the registration process for multiple iterations.
    timing(0); // Start timing
    Eigen::Matrix graph_eigen = graph_construction(correspondences, resolution, second_order_graph_flag);
    timing(1); // End timing
    std::cout << "Graph has been constructed, time elapsed: " << std::endl; // TODO: complete the timing log logics

    // Check whether the graph is all 0
    if (graph_eigen.norm() == 0) {
        cout << "Graph is disconnected. You may need to check the compatibility threshold!" << endl;
        return false;
    }


    timing(0);
    // Prepaer for filtering

    // Calculate degree of the vertexes

    // std::vector<int> graph_degree(total_correspondences_num, 0);
    std::vector<Vote_exp> points_degree(total_correspondences_num);
    // points_degree.reserve(total_correspondences_num); // used for single thread
#pragma omp parallel for schedule(static) default(none) shared(total_correspondences_num, points_degree, gt_correspondences, graph_eigen)
    for (int i = 0; i < total_correspondences_num; ++i) {
        // Construct variables
        int current_index = 0;
        int degree = 0;
        float score = 0;
        std::vector<int> correspondences_index;
        correspondences_index.reserve(total_correspondences_num);
        int true_num = 0;
        for (int j = 0; j < total_correspondences_num; ++j) {
            if (i != j && graph_eigen(i, j)) {
                degree++;
                correspondences_index.push_back(j);
                if (gt_correspondences[j]) {
                    true_num++;
                }
            }
        }
        points_degree[i].current_index = current_index;
        points_degree[i].degree = degree;
        points_degree[i].correspondences_index = correspondences_index;
        points_degree[i].true_num = true_num;
    }

    // // igraph version, should be carefully tested. I did not try igraph libs
    // igraph_t graph_igraph;
    // igraph_matrix_t graph_igraph_matrix;
    // igraph_matrix_view(&graph_igraph_matrix, graph_eigen.data(), total_correspondences_num, total_correspondences_num);
    //
    // igraph_adjacency(&graph_igraph, &graph_igraph_matrix, IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS);
    // std::cout << "\nigraph graph object created successfully." << std::endl;
    // igraph_vector_int_t degrees_igraph;
    //
    // igraph_error_t error_code_igraph = igraph_degree(&graph_igraph, &degrees_igraph, igraph_vss_all(), IGRAPH_ALL,
    //                                                  IGRAPH_NO_LOOPS);
    // if (error_code_igraph != IGRAPH_SUCCESS) {
    //     std::cerr << "Error calculating degree: " << igraph_strerror(error_code_igraph) << std::endl;
    //     return false;
    // }


    timing(1);

    // Calculate the vertex clustering factor to determine the density of the graph.
    // Delete some of the vertexes and edges if the graph is dense

    timing(0);
    std::vector<Vote> cluster_factor;
    float sum_numerator = 0;
    float sum_denominator = 0;
    for (int i = 0; i < total_correspondences_num; ++i) {
        double weight_sum_i = 0.0;
        int neighbor_size = points_degree[i].degree; // degree = correspondences_index.size()
        if (neighbor_size > 1) {
            int current_index = 0;
            int score = 0;
#pragma omp parallel
            {
#pragma omp for
                for (int j = 0; j < neighbor_size; ++j) {
                    int neighbor_index_1 = points_degree[i].correspondences_index[j];
                    for (int k = j + 1; k < neighbor_size; ++k) {
                        int neighbor_index_2 = points_degree[i].correspondences_index[k];
                        if (graph_eigen(neighbor_index_1, neighbor_index_2)) {
#pragma omp critical
                            weight_sum_i += graph_eigen(i, neighbor_index_1);
                            weight_sum_i += graph_eigen(i, neighbor_index_2);
                            // weight_sum_i += graph_eigen(neighbor_index_1, neighbor_index_2);
                            // weight_sum_i += pow(
                            //     graph_eigen(i, neighbor_index_1) * graph_eigen(i, neighbor_index_2) * graph_eigen(
                            //         neighbor_index_1, neighbor_index_2), 1.0 / 3);
                        }
                    }
                }
            }
            float vertex_numerator = weight_sum_i;
            float vertex_denominator = static_cast<float>(neighbor_size * (neighbor_size - 1)) / 2.0f;
            sum_numerator += vertex_numerator;
            sum_denominator += vertex_denominator;
            float vertex_factor = vertex_numerator / vertex_denominator;
            cluster_factor.emplace_back(i, vertex_factor, false);
        } else {
            cluster_factor.emplace_back(i, 0.0f, false); // If the vertex has no neighbors, set the factor to 0
        }
    }

    timing(1);
    std::cout << "cluster factors calculation completed. Time elapsed: " << std::endl; // Need to complete the timing logics

    // average factor for clusters
    float average_factor_cluster = 0;
    for (auto & i : cluster_factor) {
        average_factor_cluster += i.score;
    }
    average_factor_cluster /= static_cast<float>(cluster_factor.size());

    // average factor for vertexes
    float average_factor_vertex = sum_numerator / sum_denominator;

    std::vector<Vote>cluster_factor_sorted;
    cluster_factor_sorted.assign(cluster_factor.begin(), cluster_factor.end()); // copy of cluster_factor
    sort(cluster_factor_sorted.begin(), cluster_factor_sorted.end(), compare_vote_score);

    // Prepaer data for OTSU thresholding
    std::vector<float> cluster_factor_scores;
    cluster_factor_scores.resize(cluster_factor.size());
    for (int i = 0; i < cluster_factor.size(); ++i) {
        cluster_factor_scores[i] = cluster_factor[i].score;
    }

    float otsu = 0;
    if (cluster_factor_sorted[0].score != 0) {
        otsu = otsu_thresh(cluster_factor_scores);
    }
    float cluster_threshold = min (otsu, min(average_factor_cluster, average_factor_vertex));


    cout << cluster_threshold << "->min(" << average_factor_cluster << " " << average_factor_vertex << " " << otsu << ")" << endl;
    cout << " inliers: " << inlier_num << "\ttotal num: " << total_correspondences_num << "\tinlier ratio: " << inlier_ratio*100 << "%" << endl;
    //OTSU计算权重的阈值
    float weight_thresh = 0; //OTSU_thresh(sorted); // no overlap, thus weigth thresh is 0.

    // assign score member variable. Note that we need to align the indexes
    if (instance_equal_flag)
    {
        for (size_t i = 0; i < total_correspondences_num; i++)
        {
            correspondences[i].score = cluster_factor[i].score;
        }
    }

    // Maximal clique searching

    // Create igraph graph from the Eigen matrix
    igraph_t graph_igraph;
    igraph_matrix_t graph_igraph_matrix;
    // igraph_matrix_init(&graph_igraph_matrix, graph_eigen.rows(), graph_eigen.cols());

    // Filtering, reduce the graph size
    // Note that the original mac++ use this to filter the graph on kitti dataset. We ignore that for now

    // for (int i = 0; i < graph_eigen.rows(); ++i) {
    //     for (int j = 0; j < graph_eigen.cols(); ++j) {
    //         if (graph_eigen(i, j)) {
    //             igraph_matrix_set(&graph_igraph_matrix, i, j, graph_eigen(i, j));
    //         } else {
    //             igraph_matrix_set(&graph_igraph_matrix, i, j, 0);
    //         }
    //     }
    // }

    // TODO: We can use igraph_adjlist to construct the igraph graph. This may reduce the graph construction time.
    // TODO: igraph can also use BLAS to speed up processing.
    // Need to be checked!!! I do not know how to use igraph!!
    igraph_matrix_view(&graph_igraph_matrix, graph_eigen.data(), total_correspondences_num, total_correspondences_num);
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&graph_igraph, &graph_igraph_matrix, IGRAPH_ADJ_UNDIRECTED, &weight, IGRAPH_NO_LOOPS);

    // Find the maximal cliques in the graph
    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);
    timing(0);

    int min_clique_size = 3; // Minimum size of the clique to be considered, 3 is the minimum number to creat a triangle
    int max_clique_size = 0; // Maximum size of the clique, 0 is no limit.
    bool recalculate_flag = true; // Flag to indicate whether to recalculate the cliques
    int iter_num = 1;

    while (recalculate_flag) {
        igraph_maximal_cliques(&graph_igraph, &cliques, min_clique_size, max_clique_size);
        clique_num = static_cast<int>(igraph_vector_int_list_size(&cliques));
        // For now, we do not know in what case this will happen
        if (clique_num > 10000000 && iter_num <= 5) {
            max_clique_size = 15;
            min_clique_size += iter_num;
            iter_num++;
            igraph_vector_int_list_destroy(&cliques);
            igraph_vector_int_list_init(&cliques, 0);
            std::cout << "clique number " << clique_num << " is too large, recalculate with min_clique_size = "
                    << min_clique_size << " and max_clique_size = " << max_clique_size << std::endl;
        } else {
            recalculate_flag = false;
        }
    }

    timing(1);

    if (clique_num == 0) {
        std::cout << YELLOW << "Error: No cliques found in the graph." << RESET << std::endl;
        return false;
    }
    std::cout << "Number of cliques found: " << clique_num << ". Time for maximal clique search: "<< std::endl; // timing logic should be completed

    // Data cleaning
    igraph_destroy(&graph_igraph);
    igraph_matrix_destroy(&graph_igraph_matrix);


    // Correspondence seed generation and clique pre filtering
    std::vector<int> sampled_ind; // sampled correspondences index
    std::vector<int> remain; // remaining correspondences index after filtering

    clique_sampling(graph_eigen, &cliques, sampled_ind, remain);

    std::vector<Correspondence_Struct> sampled_corr; // sampled correspondences
    PointCloudPtr sampled_corr_src(new pcl::PointCloud<pcl::PointXYZ>); // sampled source point cloud
    PointCloudPtr sampled_corr_tgt(new pcl::PointCloud<pcl::PointXYZ>); // sampled target point cloud
    int inlier_num_af_clique_sampling = 0;
    for(auto &ind : sampled_ind){
        sampled_corr.push_back(correspondences[ind]);
        sampled_corr_src->push_back(correspondences[ind].src);
        sampled_corr_tgt->push_back(correspondences[ind].tgt);
        if(gt_correspondences[ind]){
            inlier_num_af_clique_sampling++;
        }
    }

    // Save log
    string sampled_corr_txt = output_path + "/sampled_corr.txt";
    ofstream outFile1;
    outFile1.open(sampled_corr_txt.c_str(), ios::out);
    for(int i = 0;i <(int)sampled_corr.size(); i++){
        outFile1 << sampled_corr[i].src_index << " " << sampled_corr[i].tgt_index <<endl;
    }
    outFile1.close();

    string sampled_corr_label = output_path + "/sampled_corr_label.txt";
    ofstream outFile2;
    outFile2.open(sampled_corr_label.c_str(), ios::out);
    for(auto &ind : sampled_ind){
        if(gt_correspondences[ind]){
            outFile2 << "1" << endl;
        }
        else{
            outFile2 << "0" << endl;
        }
    }
    outFile2.close();

    // The inlier ratio should be higher than the original inlier ratio
    std::cout << "Inlier ratio after clique sampling: "
              << static_cast<float>(inlier_num_af_clique_sampling) / static_cast<float>(sampled_ind.size()) * 100
              << "%" << std::endl;

    std::cout << "Number of sampled correspondences: " << sampled_ind.size() << std::endl;
    std::cout << "Number of remaining correspondences: " << remain.size() << std::endl;
    std::cout << "Number of cliques: " << clique_num << std::endl;
    std::cout << "Time for clique sampling: " << std::endl; // timing logic should be completed
    timing(1);

    // construct the correspondence points index list for sampled correspondences
    PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < total_correspondences_num; i++)
    {
        src_corr_pts->push_back(correspondences[i].src);
        des_corr_pts->push_back(correspondences[i].tgt);
    }


    // Registration

    Eigen::Matrix4f best_est1, best_est2; // TODO: change the name

    bool found_flag = false; // Flag to indicate whether a valid registration was found
    float best_score = 0.0f; // Best score for the registration

    timing(0);
    int total_estimate_num = remain.size(); // Total number of estimated correspondences

    std::vector<Eigen::Matrix3f> Rs;
    std::vector<Eigen::Vector3f> Ts;
    std::vector<float> scores;
    std::vector<std::vector<int>>group_corr_ind;
    int max_size = 0;
    int min_size = 666;
    int selected_size = 0;

    std::vector<Vote>est_vector;
    std::vector<pair<int, std::vector<int>>> tgt_src;
    make_tgt_src_pair(correspondences, tgt_src); //将初始匹配形成点到点集的对应

    // Get each clique and estimate the transformation matrix by the points in the clique
#pragma omp parallel for
    for (int i = 0; i < total_estimate_num; ++i) {
        std::vector<Correspondence_Struct>group, group1;
        std::vector<int> selected_index;
        igraph_vector_int_t *v = igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int group_size = igraph_vector_int_size(v); // size of the current clique
        for (int j = 0; j < group_size; j++) {
            int ind = static_cast<int>(VECTOR(*v)[j]); // Global index for j-th vertex in i-th clique
            group.push_back(correspondences[ind]);
            selected_index.push_back(ind);
        }
        sort(selected_index.begin(), selected_index.end()); // sort before get intersection

        Eigen::Matrix4f est_trans_mat;
        PointCloudPtr src_pts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr tgt_pts(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<float> weights;
        for (auto &k : group) {
            if (k.score >= weight_thresh) { // 0 by default
                group1.push_back(k);
                src_pts->push_back(k.src);
                tgt_pts->push_back(k.tgt);
                weights.push_back(k.score); // score is calculated by cluster factor
            }
        }
        if (weights.size() < 3) { //
            continue;
        }
        Eigen::VectorXf weight_vec = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(weights.data(), weights.size());
        weights.clear();
        weights.shrink_to_fit();
        weight_vec /= weight_vec.maxCoeff();
        // This can be done before weight assignments
        if (instance_equal_flag) {
            weight_vec.setOnes();
        }
        weight_svd(src_pts, tgt_pts, weight_vec, weight_thresh, est_trans_mat); // weight_thresh is 0 in original MAC++
        // When weight thresh is 0, the two group is identical
        group.assign(group1.begin(), group1.end()); // assign the filtered group to the original group
        group1.clear();

        // pre evaluate the transformation matrix generated by each clique (group, in MAC++)
        float score = 0.0f, score_local = 0.0f;
        // These evaluation is important
        // Global
        score = OAMAE(pointcloud_src_kpts, pointcloud_tgt_kpts, est_trans_mat, tgt_src, inlier_eva_thresh);
        // Local
        score_local = evaluation_trans(group, src_pts, tgt_pts, est_trans_mat, inlier_eva_thresh, metric, resolution);

        src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        tgt_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        group.clear();
        group.shrink_to_fit();

        //GT未知 <- commented by the MAC++ author
        if (score > 0)
        {
#pragma omp critical
            {
                Eigen::Matrix4f trans_f = est_trans_mat;
                Eigen::Matrix3f R = trans_f.topLeftCorner(3, 3);
                Eigen::Vector3f T = trans_f.block(0, 3, 3, 1);
                Rs.push_back(R);
                Ts.push_back(T);
                scores.push_back(score_local); // local score add to scores
                group_corr_ind.push_back(selected_index);
                selected_size = selected_index.size();
                Vote t;
                t.current_index = i;
                t.score = score;
                double re, te;
                // This part use the gt mat, only for method evaluation
                t.flag = evaluation_est(est_trans_mat, gt_mat, 15, 30, re, te);
                if(t.flag){
                    success_num ++;
                }
                //
                est_vector.push_back(t);
                if (best_score < score)
                {
                    best_score = score; // score is the global evaluation score
                    best_est1 = est_trans_mat; // best_est1 is the one generated from each clique weighted svd
                    //selected = Group;
                    //corre_index = selected_index;
                }
            }
        }
        selected_index.clear();
        selected_index.shrink_to_fit();
    }

    //释放内存空间
    // Clique searching is done, we can destroy the cliques
    igraph_vector_int_list_destroy(&cliques);


    bool clique_reduce = false;
    vector<int>indices(est_vector.size());
    for (int i = 0; i < (int )est_vector.size(); ++i) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(), [&est_vector](int a, int b){return est_vector[a].score > est_vector[b].score;});
    vector<Vote>est_vector1(est_vector.size()); // sorted est_vector
    for(int i = 0; i < (int )est_vector.size(); i++){
        est_vector1[i] = est_vector[indices[i]];
    }
    est_vector.assign(est_vector1.begin(), est_vector1.end()); // est_vector is sorted
    est_vector1.clear();


    // TODO: Check all groud true evaluations, and unify the naming. Also pay attension to the method evaluation. Unify the comment expression.
    // GT Evaluation first then filter
    int max_num = min(min(total_correspondences_num,total_estimate_num), max_estimate_num);
    success_num = 0; // note the last sucess_num is not used, check that whether can be used
    vector<int>remained_est_ind;
    vector<Eigen::Matrix3f> Rs_new;
    vector<Eigen::Vector3f> Ts_new;
    if((int )est_vector.size() > max_num) { //选出排名靠前的假设
        cout << "too many cliques" << endl;
    }
    for(int i = 0; i < min(max_num, (int )est_vector.size()); i++){
        remained_est_ind.push_back(indices[i]);
        Rs_new.push_back(Rs[indices[i]]);
        Ts_new.push_back(Ts[indices[i]]);
        success_num += est_vector[i].flag ? 1 : 0;
    }
    Rs.clear();
    Ts.clear();
    Rs.assign(Rs_new.begin(), Rs_new.end());
    Ts.assign(Ts_new.begin(), Ts_new.end());
    Rs_new.clear();
    Ts_new.clear();

    if(success_num > 0){
        if(!no_logs){
            string est_info = output_path + "/est_info.txt";
            ofstream est_info_file(est_info, ios::trunc);
            est_info_file.setf(ios::fixed, ios::floatfield);
            for(auto &i : est_vector){
                est_info_file << setprecision(10) << i.score << " " << i.flag << endl;
            }
            est_info_file.close();
        }
    }
    else{
        cout<< "NO CORRECT ESTIMATION!!!" << endl;
    }

    //cout << success_num << " : " << max_num << " : " << total_estimate << " : " << clique_num << endl;
    //cout << min_size << " : " << max_size << " : " << selected_size << endl;
    correct_est_num = success_num;

    // Clustering
    // Set parameters according to datasets
    float angle_thresh;
    float dis_thresh;
    if(dataset_name == "3dmatch" || dataset_name == "3dlomatch"){
        angle_thresh = 5.0 * M_PI / 180.0;
        dis_thresh = inlier_eva_thresh;
    }
    else if(dataset_name == "U3M"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = 5*resolution;
    }
    else if(dataset_name == "KITTI"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = inlier_eva_thresh;
    }
    else{
        cout << "not implement" << endl;
        exit(-1);
    }


    // Clustering the estimated transformations
    pcl::IndicesClusters clusterTrans;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr trans(new pcl::PointCloud<pcl::PointXYZINormal>);
    float eigenSimilarityScore = numeric_limits<float>::max();
    int similar2est1_cluster; //类号
    int similar2est1_ind;//类内号
    int best_index;

    clusterTransformationByRotation(Rs, Ts, angle_thresh, dis_thresh,  clusterTrans, trans);
    std::cout << "Total " << max_num << " cliques(transformations) found, " << clusterTrans.size() << " clusters found." << std::endl;
    // If the clustering failted, then we use the standard MAC
    // TODO: Revise the code below
    if(clusterTrans.size() ==0){
        std::cout << YELLOW << "Warning: No clusters found, using the standard MAC from the cliques." << RESET << std::endl;
        Eigen::MatrixXf tmp_best;
        if (dataset_name == "U3M")
        {
            RE = RMSE_compute(pointcloud_src, pointcloud_tgt, best_est1, gt_mat, resolution);
            TE = 0;
        }
        else {
            if (!found_flag)
            {
                found_flag = evaluation_est(best_est1, gt_mat, RE_eva_thresh, TE_eva_thresh, RE, TE);
            }
            tmp_best = best_est1;
            best_score = 0;
            post_refinement(sampled_corr, sampled_corr_src, sampled_corr_tgt, best_est1, best_score, inlier_eva_thresh, 20, metric);
        }
        if (dataset_name == "U3M")
        {
            if (RE <= 5)
            {
                cout << RE << endl;
                cout << best_est1 << endl;
                return true;
            }
            else {
                return false;
            }
        }
        else {
//            float rmse = RMSE_compute_scene(cloud_src, cloud_des, best_est1, GTmat, 0.0375);
//            cout << "RMSE: " << rmse <<endl;
            if (found_flag) {
                double new_re, new_te;
                evaluation_est(best_est1, gt_mat, RE_eva_thresh, TE_eva_thresh, new_re, new_te);

                if (new_re < RE && new_te < TE) {
                    cout << "est_trans updated!!!" << endl;
                    cout << "RE=" << new_re << " " << "TE=" << new_te << endl;
                    cout << best_est1 << endl;
                } else {
                    best_est1 = tmp_best;
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    cout << best_est1 << endl;
                }
                RE = new_re;
                TE = new_te;
//                if(rmse > 0.2) return false;
//                else return true;
                return true;
            } else {
                double new_re, new_te;
                found_flag = evaluation_est(best_est1, gt_mat, RE_eva_thresh, TE_eva_thresh, new_re, new_te);
                if (found_flag) {
                    RE = new_re;
                    TE = new_te;
                    cout << "est_trans corrected!!!" << endl;
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    cout << best_est1 << endl;
                    return true;
                }
                else{
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    return false;
                }
//                if(rmse > 0.2) return false;
//                else return true;
            }
        }
    }

    // Sort the clusters by size
    int good_cluster_num = 0;
    std::vector<Vote> sort_cluster(clusterTrans.size());
    for (size_t i = 0; i < clusterTrans.size(); ++i) {
        sort_cluster[i].current_index = i;
        sort_cluster[i].score = static_cast<float>(clusterTrans[i].indices.size());
        if (sort_cluster[i].score >= 1) {
            good_cluster_num++;
        }
    }
    if (good_cluster_num <= 0) {
        std::cout << YELLOW << "Warning: No good clusters found." << RESET << std::endl;
    }
    std::sort(sort_cluster.begin(), sort_cluster.end(), compare_vote_score);

    // Find where the best_est1 locates
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> est_trans2; // allign the memory (do not know why)
    std::vector<int>clusterIndexOfest2;
    std::vector<int>globalUnionInd;

    // Find the most similar transformation to the best_est1
#pragma omp parallel for
    for(int i = 0; i  < (int )sort_cluster.size(); i++){
        int index = sort_cluster[i].current_index;
        for(int j = 0; j < (int )clusterTrans[index].indices.size(); j++){
            int k = clusterTrans[index].indices[j];
            Eigen::Matrix3f R = Rs[k];
            Eigen::Vector3f T = Ts[k];
            Eigen::Matrix4f mat;
            mat.setIdentity();
            mat.block(0, 3, 3, 1) = T;
            mat.topLeftCorner(3,3) = R;
            float similarity = (best_est1.inverse() * mat - Eigen::Matrix4f::Identity(4, 4)).norm();
#pragma omp critical
            {
                if(similarity < eigenSimilarityScore){
                    eigenSimilarityScore = similarity;
                    similar2est1_ind = j;
                    similar2est1_cluster = index;
                }
            }
        }
    }
    std::cout << "Mat " << similar2est1_ind <<" in cluster " << similar2est1_cluster << " ("
    << sort_cluster[similar2est1_cluster].score << ") is similar to best_est1 with score " << eigenSimilarityScore <<
    std::endl;

    // For each cluster, use the cluster matching (?)
    std::vector<std::vector<int>>sub_cluster_indexes;
#pragma omp parallel for
    for(int i = 0; i < (int )sort_cluster.size(); i ++){
        //考察同一聚类的匹配
        vector<Correspondence_Struct>subClusterCorr;
        PointCloudPtr cluster_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr cluster_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
        vector<int>subUnionInd;
        int index = sort_cluster[i].current_index; //clusterTrans中的序号
        int k = clusterTrans[index].indices[0]; //初始聚类中心
        float cluster_center_score = scores[remained_est_ind[k]]; //初始聚类中心分数
        subUnionInd.assign(group_corr_ind[remained_est_ind[k]].begin(), group_corr_ind[remained_est_ind[k]].end());

        for(int j = 1; j < (int )clusterTrans[index].indices.size(); j ++){
            int m = clusterTrans[index].indices[j];
            float current_score = scores[remained_est_ind[m]]; //local score
            if (current_score > cluster_center_score){ //分数最高的设为聚类中心
                k = m;
                cluster_center_score = current_score;
            }
            subUnionInd = vectors_union(subUnionInd, group_corr_ind[remained_est_ind[m]]);
        }

        for (int l = 0; l < (int )subUnionInd.size(); ++l) {
            subClusterCorr.push_back(correspondences[subUnionInd[l]]);
            cluster_src_pts->push_back(correspondences[subUnionInd[l]].src);
            cluster_des_pts->push_back(correspondences[subUnionInd[l]].tgt);
        }
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[k];
        mat.topLeftCorner(3,3) = Rs[k];

#pragma omp critical
        {
            globalUnionInd = vectors_union(globalUnionInd,subUnionInd);
            est_trans2.push_back(mat);
            sub_cluster_indexes.push_back(subUnionInd);
            clusterIndexOfest2.push_back(index);
        }
        subClusterCorr.clear();
        subUnionInd.clear();
    }

    vector<Correspondence_Struct> global_union_corr;
    PointCloudPtr global_union_corr_src (new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr global_union_corr_tgt (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < (int )globalUnionInd.size(); ++i) {
        global_union_corr.push_back(correspondences[globalUnionInd[i]]);
    }
    std::vector<std::pair<int, std::vector<int>>> tgt_src_2;
    make_tgt_src_pair(global_union_corr, tgt_src_2); //将初始匹配形成点到点集的对应

    // Find the best cluster center, best_est2
    best_score = 0;
#pragma omp parallel for
    for(int i = 0; i < (int )est_trans2.size(); i++){
        double cluster_eva_score;
        // _1tok is not used in this project
        // if(_1tok){
        //     cluster_eva_score = OAMAE_1tok(cloud_src_kpts, cloud_des_kpts, est_trans2[i], one2k_match, inlier_thresh);
        // }
        // else{
            cluster_eva_score = OAMAE(pointcloud_src_kpts, pointcloud_tgt_kpts, est_trans2[i], tgt_src_2, inlier_eva_thresh);
        // }
#pragma omp critical
        {
            if (best_score < cluster_eva_score) {
                best_score = cluster_eva_score;
                best_est2 = est_trans2[i];
                best_index = clusterIndexOfest2[i];
            }
        }
    }

    //按照clusterIndexOfest2 排序 subclusterinds
    indices.clear();
    for(int i = 0; i < (int )clusterIndexOfest2.size(); i++){
        indices.push_back(i);
    }
    sort(indices.begin(), indices.end(), [&clusterIndexOfest2](int a, int b){return clusterIndexOfest2[a] < clusterIndexOfest2[b];});
    vector<vector<int>> subclusterinds1;
    for(auto &ind : indices){
        subclusterinds1.push_back(sub_cluster_indexes[ind]);
    }
    sub_cluster_indexes.clear();
    sub_cluster_indexes.assign(subclusterinds1.begin(), subclusterinds1.end());
    subclusterinds1.clear();

    //输出每个best_est分别在哪个类
    if(best_index == similar2est1_cluster){
        cout << "Both choose cluster " << best_index << endl;
    }
    else{
        cout << "best_est1: " << similar2est1_cluster << ", best_est2: " << best_index << endl;
    }
 //sampled corr -> overlap prior batch -> TCD 确定best_est1和best_est2中最好的
    Eigen::Matrix4f best_est;
    PointCloudPtr sampled_src(new pcl::PointCloud<pcl::PointXYZ>); // dense point cloud
    PointCloudPtr sampled_des(new pcl::PointCloud<pcl::PointXYZ>);

    getCorrPatch(sampled_corr, pointcloud_src_kpts, pointcloud_tgt_kpts, sampled_src, sampled_des, 2*inlier_eva_thresh);
    //点云patch后校验两个best_est
    float score1 = trancatedChamferDistance(sampled_src, sampled_des, best_est1, inlier_eva_thresh);
    float score2 = trancatedChamferDistance(sampled_src, sampled_des, best_est2, inlier_eva_thresh);
    vector<Correspondence_Struct>cluster_eva_corr;
    PointCloudPtr cluster_eva_corr_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr cluster_eva_corr_des(new pcl::PointCloud<pcl::PointXYZ>);
    cout << "best_est1: " << score1 << ", best_est2: " << score2 << endl;

    // cluster_internal_evaluation
    if(cluster_internal_evaluation_flag){
        if(eigenSimilarityScore < 0.1){ //best_est1在聚类中
            if(score1 > score2) { //best_est1好的情况
                best_index = similar2est1_cluster;
                best_est = best_est1;
                cout << "prior is better" << endl;
            }
            else { //best_est2好的情况
                best_est = best_est2;
                cout << "post is better" << endl;
            }
            //取匹配交集
            vector<int>cluster_eva_corr_ind;
            cluster_eva_corr_ind.assign(sub_cluster_indexes[best_index].begin(), sub_cluster_indexes[best_index].end());
            sort(cluster_eva_corr_ind.begin(), cluster_eva_corr_ind.end());
            sort(sampled_ind.begin(), sampled_ind.end());
            cluster_eva_corr_ind = vectors_intersection(cluster_eva_corr_ind, sampled_ind);
            if(!cluster_eva_corr_ind.size()){
                exit(-1);
            }
            inlier_num_af_clique_sampling = 0;

            for(auto &ind : cluster_eva_corr_ind){
                cluster_eva_corr.push_back(correspondences[ind]);
                cluster_eva_corr_src->push_back(correspondences[ind].src);
                cluster_eva_corr_des->push_back(correspondences[ind].tgt);
                if(gt_correspondences[ind]){
                    inlier_num_af_clique_sampling++;
                }
            }
            //这里的内点率要比seed内点率高
            cout << cluster_eva_corr_ind.size() << " intersection correspondences have " << inlier_num_af_clique_sampling << " inlies: "<< inlier_num_af_clique_sampling / ((int)cluster_eva_corr_ind.size() / 1.0) * 100 << "%" << endl;
            vector<pair<int, vector<int>>> des_src3;
            make_tgt_src_pair(cluster_eva_corr, des_src3);
            // if(_1tok){
            //     best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, one2k_match, inlier_thresh, GTmat, true, folderPath);
            // }
            // else{
                best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, pointcloud_src_kpts, pointcloud_tgt_kpts, des_src3, inlier_eva_thresh, gt_mat, false, output_path);
            // }
        }
        else{ //best_est1不在聚类中
            if(score2 > score1){ //best_est2好的情况
                best_est = best_est2;
                cout << "post is better" << endl;
                vector<int>cluster_eva_corr_ind;
                cluster_eva_corr_ind.assign(sub_cluster_indexes[best_index].begin(), sub_cluster_indexes[best_index].end());
                sort(cluster_eva_corr_ind.begin(), cluster_eva_corr_ind.end());
                sort(sampled_ind.begin(), sampled_ind.end());
                cluster_eva_corr_ind = vectors_intersection(cluster_eva_corr_ind, sampled_ind);
                if(!cluster_eva_corr_ind.size()){
                    exit(-1);
                }
                inlier_num_af_clique_sampling = 0;

                for(auto &ind : cluster_eva_corr_ind){
                    cluster_eva_corr.push_back(correspondences[ind]);
                    cluster_eva_corr_src->push_back(correspondences[ind].src);
                    cluster_eva_corr_des->push_back(correspondences[ind].tgt);
                    if(gt_correspondences[ind]){
                        inlier_num_af_clique_sampling++;
                    }
                }
                cout << cluster_eva_corr_ind.size() << " intersection correspondences have " << inlier_num_af_clique_sampling << " inlies: "<< inlier_num_af_clique_sampling / ((int)cluster_eva_corr_ind.size() / 1.0) * 100 << "%" << endl;
                vector<pair<int, vector<int>>> des_src3;
                make_tgt_src_pair(cluster_eva_corr, des_src3);
                best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, pointcloud_src_kpts, pointcloud_tgt_kpts, des_src3, inlier_eva_thresh, gt_mat, false, output_path);
                //1tok
                //best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, des_src3, inlier_thresh, GTmat, folderPath);
            }
            else{ //仅优化best_est1
                best_index = -1; //不存在类中
                best_est = best_est1;
                cout << "prior is better but not in cluster! Refine est1" <<endl;
            }
        }
    }
    else{
        best_est = score1 > score2 ? best_est1 : best_est2;
    }

    timing(1);
    cout << " post evaluation: " << endl; //timing logic should be implemented

    Eigen::Matrix4f tmp_best;
    if (dataset_name == "U3M")
    {
        RE = RMSE_compute(pointcloud_src, pointcloud_tgt, best_est, gt_mat, resolution);
        TE = 0;
    }
    else {
        if (!found_flag)
        {
            found_flag = evaluation_est(best_est, gt_mat, RE_eva_thresh, TE_eva_thresh, RE, TE);
        }
        tmp_best = best_est;
        best_score = 0;
        post_refinement(sampled_corr, sampled_corr_src, sampled_corr_tgt, best_est, best_score, inlier_eva_thresh, 20, metric);

        vector<int> pred_inlier_index;
        PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*src_corr_pts, *src_trans, best_est);
        int cnt = 0;
        int t = 0;
        for (int j = 0; j < correspondences.size(); j++)
        {
            double dist = get_distance(src_trans->points[j], des_corr_pts->points[j]);
            if (dist < inlier_eva_thresh){
                cnt ++;
                if (gt_correspondences[j]){
                    t ++;
                }
            }
        }

        double IP = 0, IR = 0, F1 = 0;
        if(cnt > 0) IP = t / (cnt / 1.0);
        if(inlier_num > 0) IR = t / (inlier_num / 1.0);
        if( IP && IR){
            F1 = 2.0 / (1.0 /IP + 1.0 / IR);
        }
        cout << IP << " " << IR << " " << F1 << endl;
        pred_inlier.push_back(IP);
        pred_inlier.push_back(IR);
        pred_inlier.push_back(F1);

        //ICP
        if(use_icp_flag){
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(pointcloud_src_kpts); //稀疏一些耗时小
            icp.setInputTarget(pointcloud_tgt);
            icp.setMaxCorrespondenceDistance(0.05);
            icp.setTransformationEpsilon(1e-10);
            icp.setMaximumIterations(50);
            icp.setEuclideanFitnessEpsilon(0.2);
            PointCloudPtr final(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*final, best_est);
            if(icp.hasConverged()){
                best_est = icp.getFinalTransformation();
                cout << "ICP fitness score: " << icp.getFitnessScore() << endl;
            }
            else{
                cout << "ICP cannot converge!!!" << endl;
            }
        }
    }

    if(!no_logs){
        //保存匹配到txt
        //savetxt(correspondence, folderPath + "/corr.txt");
        //savetxt(selected, folderPath + "/selected.txt");
        string save_est = output_path + "/est.txt";
        //string save_gt = folderPath + "/GTmat.txt";
        ofstream outfile(save_est, ios::trunc);
        outfile.setf(ios::fixed, ios::floatfield);
        outfile << setprecision(10) << best_est;
        outfile.close();
        //CopyFile(gt_mat.c_str(), save_gt.c_str(), false);
        //string save_label = folderPath + "/label.txt";
        //CopyFile(label_path.c_str(), save_label.c_str(), false);

        //保存ply
        //string save_src_cloud = folderPath + "/source.ply";
        //string save_tgt_cloud = folderPath + "/target.ply";
        //CopyFile(src_pointcloud.c_str(), save_src_cloud.c_str(), false);
        //CopyFile(des_pointcloud.c_str(), save_tgt_cloud.c_str(), false);
    }

    // memory cost evaluation is pending
    // int pid = getpid();
    // mem_epoch = getPidMemory(pid);

    //保存聚类信息
    string analyse_csv = output_path + "/cluster.csv";
    string correct_csv = output_path + "/cluster_correct.csv";
    string selected_csv = output_path + "/cluster_selected.csv";
    ofstream outFile, outFile_correct, outFile_selected;
    outFile.open(analyse_csv.c_str(), ios::out);
    outFile_correct.open(correct_csv.c_str(), ios::out);
    outFile_selected.open(selected_csv.c_str(), ios::out);
    outFile.setf(ios::fixed, ios::floatfield);
    outFile_correct.setf(ios::fixed, ios::floatfield);
    outFile_selected.setf(ios::fixed, ios::floatfield);
    outFile << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    outFile_correct << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    outFile_selected << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    for(int i = 0;i <(int)sort_cluster.size(); i++){
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        int cluster_id = sort_cluster[i].current_index;
        for(int j = 0; j < (int)clusterTrans[cluster_id].indices.size(); j++){
            int id = clusterTrans[cluster_id].indices[j];
            if(est_vector[id].flag){
                outFile_correct << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
                //cout << "Correct est in cluster " << cluster_id << " (" << sortCluster[i].score << ")" << endl;
            }
            if(cluster_id == best_index) outFile_selected << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
            outFile << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
        }
    }
    outFile.close();
    outFile_correct.close();

    correspondences.clear();
    correspondences.shrink_to_fit();
    // ov_corr_label.clear();
    // ov_corr_label.shrink_to_fit();
    gt_correspondences.clear();
    gt_correspondences.shrink_to_fit();
    // degree.clear();
    // degree.shrink_to_fit();
    points_degree.clear();
    points_degree.shrink_to_fit();
    cluster_factor.clear();
    cluster_factor.shrink_to_fit();
    cluster_factor_scores.clear(); // cluster_factor_bac
    cluster_factor_scores.shrink_to_fit();
    remain.clear();
    remain.shrink_to_fit();
    sampled_ind.clear();
    sampled_ind.shrink_to_fit();
    Rs.clear();
    Rs.shrink_to_fit();
    Ts.clear();
    Ts.shrink_to_fit();
    src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pointcloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pointcloud_tgt.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pointcloud_src_kpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pointcloud_tgt_kpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    normal_src.reset(new pcl::PointCloud<pcl::Normal>);
    normal_tgt.reset(new pcl::PointCloud<pcl::Normal>);
    raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
    raw_tgt.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (dataset_name == "U3M")
    {
        if (RE <= 5)
        {
            cout << RE << endl;
            cout << best_est << endl;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        //float rmse = RMSE_compute_scene(cloud_src, cloud_des, best_est1, GTmat, 0.0375);
        //cout << "RMSE: " << rmse <<endl;
        if (found_flag)
        {
            double new_re, new_te;
            evaluation_est(best_est, gt_mat, RE_eva_thresh, TE_eva_thresh, new_re, new_te);
            if (new_re < RE && new_te < TE)
            {
                cout << "est_trans updated!!!" << endl;
                cout << "RE=" << new_re << " " << "TE=" << new_te << endl;
                cout << best_est << endl;
            }
            else {
                best_est = tmp_best;
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                cout << best_est << endl;
            }
            RE = new_re;
            TE = new_te;
//            if(rmse > 0.2){
//                return false;
//            }
//            else{
//                return true;
//            }
            return true;
        }
        else {
            double new_re, new_te;
            found_flag = evaluation_est(best_est, gt_mat, RE_eva_thresh, TE_eva_thresh, new_re, new_te);
            if (found_flag)
            {
                RE = new_re;
                TE = new_te;
                cout << "est_trans corrected!!!" << endl;
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                cout << best_est << endl;
                return true;
            }
            else{
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                return false;
            }
//            if(rmse > 0.2){
//                return false;
//            }
//            else{
//                return true;
//            }
            //Corres_selected_visual(Raw_src, Raw_des, correspondence, resolution, 0.1, GTmat);
            //Corres_selected_visual(Raw_src, Raw_des, selected, resolution, 0.1, GTmat);
        }
    }

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

    std::string dataset_name(argv[1]);
    // dataset name, previously used for different parameter settings. Evaluation metrics
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
        vector<double> pred_inlier; // Store predicted inliers
        double RE, TE; // Rotation and translation errors
        int correct_est_num = 0; // Number of correct estimated correspondences
        int gt_inlier_num = 0; // Number of inliers in the ground truth correspondences
        bool estimate_success = registration(src_pointcloud_path, tgt_pointcloud_path, corr_path, gt_label_path,
                                             gt_tf_path,
                                             output_path,
                                             descriptor, RE, TE, correct_est_num, gt_inlier_num, time_epoch, pred_inlier);

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
                    << "Total correspondences: " << total_correspondences_num << std::endl
                    << "Time taken for registration: " << time_epoch << " seconds" << std::endl;
            results_out.close();
        }

        // Output the status of the registration process
        std::string status_path = output_path + "/status.txt";
        results_out.open(status_path.c_str(), std::ios::out);
        results_out.setf(std::ios::fixed, std::ios::floatfield);
        results_out << std::setprecision(6) << "Time in one iteration: " << time_epoch <<
                " seconds, memory used in one iteration: " << std::endl;
        results_out.close();
    }


    return 0;
}
