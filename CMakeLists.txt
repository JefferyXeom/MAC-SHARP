cmake_minimum_required(VERSION 3.22.1) # Not tested, our tested cmake version is 4.0.2
project(MAC_SHARP)
# Later will change add_definition to add_compile_options and add_compile_definitions
add_definitions(
    "-Wall"              # Basic warnings
    "-Wextra"           # Extra warnings
    "-Wpedantic"        # Strict ISO C++ compliance
    "-g"                # Debug symbols
    "-O0"              # No optimization (better for debugging)
) # add all warnings and debug symbols
set(CMAKE_CXX_STANDARD 17) # 
# Search neccessary libraries
find_package(PCL REQUIRED) 
find_package(igraph 0.10.6 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
if(PCL_FOUND)
    message("PCL FOUND")
    message("PCL version: ${PCL_VERSION}")
    message("PCL include dirs: ${PCL_INCLUDE_DIRS}")
    message("PCL library dirs: ${PCL_LIBRARY_DIRS}")
    message("PCL libraries: ${PCL_LIBRARIES}")
    message("PCL definitions: ${PCL_DEFINITIONS}")
    message("PCL components: ${PCL_COMPONENTS}")
    message("PCL version: ${PCL_VERSION}")
else()
    message(FATAL_ERROR "PCL not found")
endif()
if(igraph_FOUND)
    message("igraph FOUND")
    message("igraph version: ${igraph_VERSION}")
    message("igraph include dirs: ${igraph_INCLUDE_DIRS}")
    message("igraph library dirs: ${igraph_LIBRARY_DIRS}")
    message("igraph libraries: ${igraph_LIBRARIES}")
    message("igraph definitions: ${igraph_DEFINITIONS}")
    message("igraph components: ${igraph_COMPONENTS}")
    message("igraph version: ${igraph_VERSION}")
else()
    message(FATAL_ERROR "igraph not found")
endif()
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if(yaml-cpp_FOUND)
    message("yaml-cpp FOUND")
    message("yaml-cpp version: ${yaml-cpp_VERSION}")
    message("yaml-cpp include dirs: ${yaml-cpp_INCLUDE_DIRS}")
    message("yaml-cpp library dirs: ${yaml-cpp_LIBRARY_DIRS}")
    message("yaml-cpp libraries: ${yaml-cpp_LIBRARIES}")
    message("yaml-cpp definitions: ${yaml-cpp_DEFINITIONS}")
else()
    message(FATAL_ERROR "yaml-cpp not found")
endif()
###### 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Main functional executable
add_executable(MAC_SHARP ${SOURCES})
# Test executable
add_executable(test ${TEST_SOURCES})
target_link_libraries(MAC_SHARP ${PCL_LIBRARIES} igraph::igraph)
target_link_libraries(test
    ${PCL_LIBRARIES}
    igraph::igraph
    yaml-cpp
)
