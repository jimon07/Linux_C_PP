cmake_minimum_required(VERSION 2.8)
project(Diplo_c++)

include(CTest)
enable_testing()

set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(FilesToInclude 
main.cpp
Time.cpp
)
add_executable(${PROJECT_NAME} ${FilesToInclude})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
