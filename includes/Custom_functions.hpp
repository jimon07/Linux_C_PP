#ifndef CAMERA_STAFF_HPP
#define CAMERA_STAFF_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <sys/time.h>
#include <stdio.h>

// defines declarations
using namespace cv;
using namespace std;

// function declarations

void calibration(Mat image);
double getTime();
static void floorThreshold(Mat inputMatrix,Mat& outputMatrix, float threshold);
void simulateObject(Mat i12, Mat i21, Mat objects_only);
void simulateObjectv2(Mat blueToRed, Mat redToBlue);
void color_map(cv::Mat& input /*CV_32FC1*/, cv::Mat& dest, int color_map);
void findGoodContours(vector<vector<Point> >& contours,vector<vector<Point> >& GoodContours,int minObjectArea);
#endif