#include <string> 
#include <cstdlib> // include exit function definition
#include <iostream>
#include <iomanip> // include setprecision function definition
#include <fstream> // include file stream definitions

string folder_path; // dataset folder path
bool add_overlap;  // 
bool low_inlier_ratio; // 
bool no_logs; 

string program_name = "";

// >>>> copied from mac++ >>>>

string three_d_match[8] = {
        "7-scenes-redkitchen",
        "sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30",
        "sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1",
        "sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2",
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
};

string three_d_lomatch[8] = {
        "7-scenes-redkitchen_3dlomatch",
        "sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
        "sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
        "sun3d-hotel_uc-scan3_3dlomatch",
        "sun3d-hotel_umd-maryland_hotel1_3dlomatch",
        "sun3d-hotel_umd-maryland_hotel3_3dlomatch",
        "sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
};

string ETH[4] = {
        "gazebo_summer",
        "gazebo_winter",
        "wood_autmn",
        "wood_summer",
};

// <<<< copied from mac++ <<<<

// RE: rotation error, TE: translation error, success_estimate_rate: success rate of the estimate
double RE, TE, success_estimate_rate;




int main(int argc, char** argv) {

}