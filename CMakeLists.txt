cmake_minimum_required(VERSION 3.22.1) # Not tested, our tested cmake version is 4.0.2
cmake_policy(SET CMP0144 NEW)
project(MAC_SHARP)
# Later will change add_definition to add_compile_options and add_compile_definitions
add_definitions(
    "-Wall"              # Basic warnings
    "-Wextra"           # Extra warnings
    "-Wpedantic"        # Strict ISO C++ compliance
    "-g"                # Debug symbols
    "-O3"              # No optimization (better for debugging)
#    "-march=native"    # Use native architecture optimizations
) # add all warnings and debug symbols
set(CMAKE_CXX_STANDARD 17) #
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Search necessary libraries
find_package(PCL REQUIRED) 
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
###### 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${OPENBLAS_INCLUDE_DIRS})
# Ones can use glob for source files, but it is not recommended
# Define source files
set(SOURCES
    src/MAC_SHARP.cpp
#    src/other_file1.cpp
#    src/other_file2.cpp
)
# Define header files
set(HEADERS
    include/MAC_SHARP.hpp
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
include_directories(${PROJECT_SOURCE_DIR}/include)
# Main functional executable
add_executable(MAC_SHARP ${SOURCES})
# Test executable
add_executable(test ${TEST_SOURCES})
target_link_libraries(MAC_SHARP PRIVATE ${PCL_LIBRARIES} igraph::igraph OpenBLAS::OpenBLAS)
target_link_libraries(test
    ${PCL_LIBRARIES}
    igraph::igraph
    yaml-cpp::yaml-cpp
)

add_executable(graph_construction_test ${graph_construction_test_SOURCES})
target_link_libraries(graph_construction_test PRIVATE OpenBLAS::OpenBLAS)

add_executable(temp_test test/temp_test.cpp)
