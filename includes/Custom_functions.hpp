#ifndef CAMERA_STAFF_HPP
#define CAMERA_STAFF_HPP

#include <opencv2/opencv.hpp>
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
void simulateObjectv2(Mat i12);
void findGoodContours(vector<vector<Point> >& contours,vector<vector<Point> >& GoodContours,int minObjectArea);
#endif