#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  // Load the original image
cv::Mat img = cv::imread("/home/jim/Desktop/Linux_C_PP/test/pexels-rahul-695644.jpg",IMREAD_GRAYSCALE);

// Define the size and position of the gradient rectangle
int rect_x = 50;
int rect_y = 50;
int rect_width = 200;
int rect_height = 100;

// Create a gradient image using the Jet colormap
cv::Mat gradient(rect_height, rect_width, CV_8UC1);
cv::applyColorMap(cv::Mat(cv::Size(rect_width, rect_height), CV_32FC1, cv::Scalar(0.0)), gradient, cv::COLORMAP_JET);

// Create a mask for the gradient rectangle
cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(0));
cv::Rect rect(rect_x, rect_y, rect_width, rect_height);
mask(rect) = 255;

// Copy the gradient rectangle into the main image
cv::Mat result;
img.copyTo(result);
gradient.copyTo(result, mask);

// Display the result
cv::imshow("Result", result);
cv::waitKey(0);
}