cmake_minimum_required(VERSION 2.8)
project(Diplo_c++)

include(CTest)
enable_testing()

set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( includes )

set(FilesToInclude 
main.cpp
includes/Custom_functions.cpp
)
add_executable(${PROJECT_NAME} ${FilesToInclude})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(testCustom  test/testCustom.cpp includes/Custom_functions.cpp )
add_executable(calibration  calibration/calibration.cpp includes/Custom_functions.cpp )
add_executable(cameratest  cameratest/cameratest.cpp includes/Custom_functions.cpp )
target_link_libraries( testCustom ${OpenCV_LIBS} )
target_link_libraries( calibration ${OpenCV_LIBS} )
target_link_libraries( cameratest ${OpenCV_LIBS} )