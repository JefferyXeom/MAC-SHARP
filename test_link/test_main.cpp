//
// Created by Jeffery_Xeom on 2025/9/2.
// Project: MAC_SHARP
// File: pcl_test.cpp
//

// file: test_link/test_main.cpp
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/point_types.h>

int main() {
    // 我们甚至不需要调用 .segment() 方法。
    // 只要创建了这个类的对象，链接器就必须去寻找它的实现。
    // 这足以触发我们遇到的链接错误。
    pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec;

    (void)cec; // 避免 "unused variable" 警告

    return 0;
}