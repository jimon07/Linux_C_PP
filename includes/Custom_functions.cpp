#include "Custom_functions.hpp"

// function definitions 

double getTime() {
    struct timeval ttime;
    gettimeofday(&ttime, NULL);
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

void calibration(Mat image)
{
    vector<Mat> bgr;

    split(image, bgr);//split source
        int histSize = 256;

        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };

        bool uniform = true, accumulate = false;

        Mat b_hist, g_hist, r_hist;
        
        calcHist( &bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
        calcHist( &bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
        calcHist( &bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );


        int hist_w = 1024, hist_h = 600;
        int bin_w = cvRound( (double) hist_w/histSize );
        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        for( int i = 1; i < histSize; i++ )
            {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
                Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
                Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
                Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                Scalar( 0, 0, 255), 2, 8, 0  );
            }
        
        // imshow("Source image", image );
        imshow("calcHist Demo", histImage );
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

void simulateObject(Mat blueToRed, Mat i21, Mat objects_only){
    // Mat tmp3 = blueToRed.clone();
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
    add(blueToRed,i21,sum);
    multiply(outputMatrix,sum,simul);
    normalize(simul, simul, 0, 1, cv::NORM_MINMAX);
    simul.convertTo(simul,CV_8UC1,255,0);
    imshow("Object no color" , simul);
    applyColorMap(simul,simul,2);
    imshow("Object" , simul); 
}

void simulateObjectv2(Mat blueToRed){

    Mat line,final,tmp,obj;
    float eThreshold = 0.2;
    float objCols = (float)obj.cols;
    inRange(blueToRed,1-eThreshold,1+eThreshold,tmp);
    blueToRed.setTo(1,blueToRed > 1+eThreshold);
    blueToRed.setTo(-1,blueToRed < 1-eThreshold);
    blueToRed.setTo(0,tmp);
    
    for (int i = 0; i < blueToRed.rows; i++)
    {
        Mat integral;
        line = blueToRed.row(i);
        // blueToRed.row(i).copyTo(line);
        cv::integral(line,integral,CV_32FC1);
        final.push_back(integral.row(1));
        // Or, for a deep copy:
        // line = blueToRed.row(i).clone();
        // imshow("line", line);
        // imshow("Integral",integral);
        // waitKey(0);
    }   
        // imshow("Test", blueToRed);
        final = final.colRange(1,final.cols);
        // cout << "temp size = " << tmp.size()<< endl;
        // cout << "final size = " << final.size()<< endl;
        bitwise_not(tmp,tmp);
        tmp.convertTo(tmp,final.type());
        divide(tmp,255,tmp);
        // imshow("Temp",tmp);
        // cout << "TMP = " << endl << " "  << tmp << endl << endl;
        // cout << tmp.type() << endl;
        multiply(tmp,final,obj);
        // cout << "Object befor norm = " << endl << " "  << obj << endl << endl;
        obj.at<float>(0, 0, 0) = obj.cols;
        obj.at<float>(0, 1, 0) = -(obj.cols);
        // cout << "Object = " << endl << " "  << obj << endl << endl;
        normalize(obj, obj, 0, 255, cv::NORM_MINMAX);
        obj.convertTo(obj,CV_8UC1);
        imshow("Object before color map",obj);
        // cout << "ObjectNormalize = " << endl << " "  << obj << endl << endl;
        applyColorMap(obj,obj,2);
        // imshow("Final",final);
        imshow("Object",obj);
        // cout << "blueToRed inside = " << endl << " "  << blueToRed << endl << endl;
    
        // cout << "Final = " << endl << " "  << final << endl << endl;
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
