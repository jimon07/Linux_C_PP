#include "Custom_functions.hpp"

// function definitions 

double getTime() {
    struct timeval ttime;
    gettimeofday(&ttime, NULL);
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

static void floorThreshold(Mat inputMatrix,Mat& outputMatrix, float threshold){
    Mat tmp(inputMatrix.cols,inputMatrix.rows , CV_32FC1);
    Mat not_tmp(inputMatrix.cols,inputMatrix.rows , CV_32FC1);
    Mat final(inputMatrix.cols,inputMatrix.rows , CV_32FC1);
    inRange(inputMatrix, 1-threshold , 1+threshold, tmp);
    bitwise_not(tmp,not_tmp);
    divide(tmp,255,tmp);
    divide(not_tmp,255,not_tmp);
    tmp.convertTo(tmp,inputMatrix.type());
    not_tmp.convertTo(not_tmp,inputMatrix.type());
    multiply(inputMatrix, not_tmp, final);
    add(final,tmp,outputMatrix);

}

void simulateObject(Mat i12, Mat i21, Mat objects_only){
    Mat tmp3 = i12.clone();
    Mat mask = objects_only.clone();
    Mat simul,sum;
    cvtColor(objects_only, mask, COLOR_BGR2GRAY );
    // mask.convertTo(mask,CV_8UC1);
    Mat outputMatrix;
    outputMatrix.convertTo(outputMatrix,CV_8UC1);

    distanceTransform(mask, outputMatrix, DIST_L2, DIST_MASK_PRECISE);
    normalize(outputMatrix, outputMatrix, 0, 1, cv::NORM_MINMAX);
    // outputMatrix.convertTo(outputMatrix,CV_8UC1,255,0);
    // imshow("Distance Image" , outputMatrix);
    add(i12,i21,sum);
    multiply(outputMatrix,sum,simul);
    normalize(simul, simul, 0, 1, cv::NORM_MINMAX);
    simul.convertTo(simul,CV_8UC1,255,0);
    applyColorMap(simul,simul,2);
    imshow("Object" , simul); 
}
