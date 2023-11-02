cmake_minimum_required(VERSION 3.10)
project(depth_estimation)

set(CMAKE_BUILD_TYPE release)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(depthExtraction  depth_extraction.cpp )
target_link_libraries( depthExtraction ${OpenCV_LIBS} )
add_executable(depth_extraction_without_slope depth_extraction_without_slope.cpp )
target_link_libraries( depth_extraction_without_slope ${OpenCV_LIBS} )
