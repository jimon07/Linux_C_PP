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
    // Mat tmp3 = i12.clone();
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
    imshow("Object no color" , simul);
    applyColorMap(simul,simul,2);
    imshow("Object" , simul); 
}

void simulateObjectv2(Mat i12){
    

    Mat line,final,tmp;
    float eThreshold = 0.2;
    inRange(i12,1-eThreshold,1+eThreshold,tmp);
    i12.setTo(1,i12 > 1+eThreshold);
    i12.setTo(-1,i12 < 1-eThreshold);
    i12.setTo(0,tmp);
    
    for (int i = 0; i < i12.rows; i++)
    {
        Mat integral;
        line = i12.row(i);
        // i12.row(i).copyTo(line);
        cv::integral(line,integral,CV_32F);
        final.push_back(integral.row(1));
        // Or, for a deep copy:
        // line = i12.row(i).clone();
        imshow("Test", i12);
        imshow("line", line);
        imshow("Integral",integral);
        imshow("Final",final);
        // waitKey(0);
    }   
        cout << "I12 inside = " << endl << " "  << i12 << endl << endl;
        
        cout << "Final = " << endl << " "  << final << endl << endl;
        // waitKey(1);

}
void findGoodContours(vector<vector<Point> >& contours,vector<vector<Point> >& GoodContours,int minObjectArea){
    
    for (size_t idx = 0; idx < contours.size(); idx++) {
        int area = contourArea(contours[idx]);
        if (area > minObjectArea) {
            GoodContours.push_back(contours.at(idx));
        }
    }
}
