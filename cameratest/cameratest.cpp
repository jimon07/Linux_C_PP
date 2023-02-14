#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Create a test matrix
    Mat mat = (Mat_<float>(3,3) << -1000, 0, 1000,
                                    500, 250, 750,
                                    -500, -250, -750);
    cout << "Matrix: " << endl << mat << endl;

    // Create a named window to display the image
    namedWindow("Matrix", WINDOW_NORMAL);

    // Convert the matrix to a 8-bit image and apply a color map
    Mat color_mat;
    mat.convertTo(color_mat, CV_8UC1, 255.0 / 1000.0);
    applyColorMap(color_mat, color_mat, COLORMAP_JET);

    // Show the image in the window
    imshow("Matrix", color_mat);

    // Wait for a key press and return
    waitKey(0);
    return 0;
}