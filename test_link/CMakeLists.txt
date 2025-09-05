# file: test_link/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(PCL_LINK_TEST CXX)
set(CMAKE_CXX_STANDARD 17)

# 我们只需要 PCL 的 segmentation 组件
find_package(PCL REQUIRED COMPONENTS segmentation)

add_executable(test_pcl_link test_main.cpp)

# 这是最关键、最简单的链接命令
target_link_libraries(test_pcl_link PRIVATE ${PCL_LIBRARIES})