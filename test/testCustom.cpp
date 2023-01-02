#include "Custom_functions.hpp"


int main(int argc, char** argv)
{
    string path = "/home/jim/Desktop/Linux_C_PP/test/pexels-rahul-695644.jpg";

    Mat image = imread(path,IMREAD_GRAYSCALE);
     imshow("image",image);
     waitKey(0);
     simulateObjectv2(image);

}