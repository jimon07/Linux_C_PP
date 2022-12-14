#ifndef CAMERA_STAFF_HPP
#define CAMERA_STAFF_HPP

#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdio.h>

// defines declarations
using namespace cv;

// function declarations

double getTime();
void cameraInitiation(VideoCapture& cam, Mat& Purple);
static void floorThreshold(Mat inputMatrix,Mat& outputMatrix, float threshold);
void simulateObject(Mat i12, Mat i21, Mat objects_only);
#endif