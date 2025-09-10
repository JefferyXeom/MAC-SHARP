# ==============================================================================
# ==                            项目基础设置                                  ==
# ==============================================================================
cmake_minimum_required(VERSION 3.22.1) # Not tested, our tested cmake version is 4.0.2
cmake_policy(SET CMP0144 NEW)
project(MAC_SHARP)
# Later will change add_definition to add_compile_options and add_compile_definitions
add_definitions(
    "-Wall"              # Basic warnings
    "-Wextra"           # Extra warnings
    "-Wpedantic"        # Strict ISO C++ compliance
    "-r"                # r: release, d: debug
    "-O3"               # No optimization (better for debugging)
#    "-march=native"    # Use native architecture optimizations
) # add all warnings and debug symbols
set(CMAKE_CXX_STANDARD 17) #
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ==============================================================================
# ==                            查找所有依赖包                                ==
# ==============================================================================
# Search necessary libraries
find_package(PCL REQUIRED COMPONENTS
        common
        io
        kdtree
        features
        registration
        segmentation)
#find_package(igraph 0.10.6 REQUIRED)
find_package(igraph REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenBLAS REQUIRED)
if(PCL_FOUND)
    message(STATUS "PCL FOUND")
    message(STATUS "PCL version: ${PCL_VERSION}")
    message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")
    message(STATUS "PCL library dirs: ${PCL_LIBRARY_DIRS}")
    message(STATUS "PCL libraries: ${PCL_LIBRARIES}")
    message(STATUS "PCL definitions: ${PCL_DEFINITIONS}")
    message(STATUS "PCL components: ${PCL_COMPONENTS}")
    message(STATUS "PCL version: ${PCL_VERSION}")
else()
    message(FATAL_ERROR "PCL not found")
endif()
if(igraph_FOUND)
    message(STATUS "igraph FOUND")
    message(STATUS "igraph version: ${igraph_VERSION}")
    message(STATUS "igraph include dirs: ${igraph_INCLUDE_DIRS}")
    message(STATUS "igraph library dirs: ${igraph_LIBRARY_DIRS}")
    message(STATUS "igraph libraries: ${igraph_LIBRARIES}")
    message(STATUS "igraph definitions: ${igraph_DEFINITIONS}")
    message(STATUS "igraph components: ${igraph_COMPONENTS}")
    message(STATUS "igraph version: ${igraph_VERSION}")
else()
    message(FATAL_ERROR "igraph not found")
endif()
if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp FOUND")
    message(STATUS "yaml-cpp version: ${yaml-cpp_VERSION}")
    message(STATUS "yaml-cpp include dirs: ${yaml-cpp_INCLUDE_DIRS}")
    message(STATUS "yaml-cpp library dirs: ${yaml-cpp_LIBRARY_DIRS}")
    message(STATUS "yaml-cpp libraries: ${yaml-cpp_LIBRARIES}")
    message(STATUS "yaml-cpp definitions: ${yaml-cpp_DEFINITIONS}")
else()
    message(FATAL_ERROR "yaml-cpp not found")
endif()
if(OpenBLAS_FOUND)
    message(STATUS "OpenBLAS FOUND")
    message(STATUS "OpenBLAS version: ${OpenBLAS_VERSION}")
    message(STATUS "OpenBLAS include dirs: ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "OpenBLAS library dirs: ${OpenBLAS_LIBRARY_DIRS}")
    message(STATUS "OpenBLAS libraries: ${OpenBLAS_LIBRARIES}")
    message(STATUS "OpenBLAS definitions: ${OpenBLAS_DEFINITIONS}")
else()
    message(FATAL_ERROR "OpenBLAS not found")
endif()
# ==============================================================================
# ==                            定义源文件                                    ==
# ==============================================================================

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
#include_directories(${OPENBLAS_INCLUDE_DIRS})
# Ones can use glob for source files, but it is not recommended
# Define source files
set(SOURCES
        src/MacSharp.cpp
        src/MacUtils.cpp
        src/MacData.cpp
        src/MacGraph.cpp
        src/MacRtHypothesis.cpp
        src/MacMonitor.cpp
        src/MacEvaluator.cpp
        include/MacEvaluator.hpp
        #    src/other_file2.cpp
)
# Define header files
set(HEADERS
        include/MacSharp.hpp
        include/MacUtils.hpp
        include/CommonTypes.hpp
        include/MacTimer.hpp
        include/MacConfig.hpp
        include/MacData.hpp
        include/MacGraph.hpp
        include/MacRtHypothesis.hpp
#    include/other_file1.h
#    include/other_file2.h
)
# Define test source files
set(TEST_SOURCES
    test/test.cpp
)
set(graph_construction_test_SOURCES
    test/graph_construction_test.cpp
)
#include_directories(${PROJECT_SOURCE_DIR}/include)
# ==============================================================================
# ==                      定义主程序目标 (MAC_SHARP)                         ==
# ==============================================================================
# Main functional executable
add_executable(MAC_SHARP ${SOURCES})
if(MSVC)
    target_compile_options(MAC_SHARP PRIVATE /arch:SSE4.1)
else()
    target_compile_options(MAC_SHARP PRIVATE -msse4.1)
endif()

# --- 为 MAC_SHARP 单独设置属性 (不再使用全局命令) ---

# 1. 设置头文件搜索路径 (现代方式)
target_include_directories(MAC_SHARP
        PRIVATE
#        ${PROJECT_INCLUDE_DIR}
#        ${HEADERS}
        ${PROJECT_SOURCE_DIR}/include
        ${PCL_INCLUDE_DIRS}
        ${igraph_INCLUDE_DIRS}
        ${yaml-cpp_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIRS}
)

# 2. 设置编译定义 (现代方式)
target_compile_definitions(MAC_SHARP
        PRIVATE
        ${PCL_DEFINITIONS}
)

# 3. 设置编译选项 (现代方式)
target_compile_options(MAC_SHARP
        PRIVATE
        $<$<CONFIG:DEBUG>:-g -O0>  # 只在Debug模式下使用-g -O0
        -Wall
        -Wextra
        -Wpedantic
        -msse
        -msse2
        # 不再使用 -mavx
        -D_ENABLE_EXTENDED_ALIGNED_STORAGE
        # --- 关键修改在这里：添加 -v ---
#        -v
)

# 4. 链接库 (现代方式)
target_link_libraries(MAC_SHARP
        PRIVATE
        ${PCL_LIBRARIES}
        ${PCL_COMMON_LIBRARIES}
        ${PCL_IO_LIBRARIES}
        ${PCL_KDTREE_LIBRARIES}
        ${PCL_FEATURES_LIBRARIES}
        ${PCL_REGISTRATION_LIBRARIES}
        ${PCL_SEGMENTATION_LIBRARIES} # <-- 最关键的一个，现在被显式链接
        igraph::igraph
        OpenBLAS::OpenBLAS
        yaml-cpp::yaml-cpp
        OpenMP::OpenMP_CXX
)


##add_compile_options(/arch:AVX)
## 检查编译器是否为 GCC 或 Clang
#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#    # 为 C++ 添加 SSE 和 SSE2 编译标志
#    # Eigen 经常使用 SSE2，加上更保险
#    add_compile_options(-msse -msse2)
#    message(STATUS "SSE/SSE2 flags enabled for GCC/Clang.")
#
#    # 检查编译器是否为 MSVC (Visual Studio)
#elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#    # MSVC 在 x64 架构下默认启用 SSE2，但在 x86 下需要 /arch:SSE2
#    # 为了保险起见，可以检查处理器架构
#    if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32-bit (x86)
#        add_compile_options(/arch:SSE2)
#        message(STATUS "SSE2 flag enabled for MSVC x86.")
#    else()
#        message(STATUS "SSE2 is enabled by default on MSVC x64.")
#    endif()
#endif()
#
##add_compile_options(-msse -msse2 -mavx)
#add_compile_options(-msse -msse2)
## Test executable
##add_executable(test ${TEST_SOURCES})
#target_link_libraries(MAC_SHARP PRIVATE ${PCL_LIBRARIES} igraph::igraph OpenBLAS::OpenBLAS yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX)
##target_link_libraries(test
##    ${PCL_LIBRARIES}
##    igraph::igraph
##    yaml-cpp::yaml-cpp
##)
#
#
#add_executable(graph_construction_test ${graph_construction_test_SOURCES})
#target_link_libraries(graph_construction_test PRIVATE OpenBLAS::OpenBLAS)

add_executable(temp_test test/temp_test.cpp)

add_subdirectory(test_link)