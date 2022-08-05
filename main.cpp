#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <sys/time.h>

#include <limits>



using namespace cv;

using namespace std;



//Calibration Variables

int lowThreshold = 65;

int maxThreshold = 200;

int minObjectArea = 7000;

const int max_lowThreshold = 400;

const int max_maxThreshold = 800;

// const int rat = 1;

const int kernel_size = 3;

int blurKernelSize = 3;

Mat canny_img,image,image_resized,image_blurred,dilated_img,objects_only;

Mat in1,in2,bgrFB,bgrFR,i12,i21,thres12,thres21,final12,final21,image_proccesing;

Mat elementKernel;

RNG rng(12345);

Scalar colorGreen = Scalar(0,255,0);

Scalar colorWhite = Scalar(255,255,255);



static double gettime() {

    struct timeval ttime;

    gettimeofday(&ttime, NULL);

    return ttime.tv_sec + ttime.tv_usec * 0.000001;

}



static void findObjects(int,void*)

{

    // blur( image_resized, image_blurred, Size(blurKernelSize, blurKernelSize));

    GaussianBlur( image_resized, image_blurred, Size(5, 5),5,5);

    

    Canny( image_blurred, canny_img, lowThreshold, maxThreshold, kernel_size );



    dilate(canny_img,dilated_img,elementKernel,Point(-1,-1),1);

   

    // imshow( "Edges Image", canny_img);

    

    

    // imshow( "Dilated Image", dilated_img);



    vector<vector<Point> > contours;

    vector<Vec4i> hierarchy;

    findContours( dilated_img, contours, hierarchy, RETR_EXTERNAL , CHAIN_APPROX_NONE );



    vector<vector<Point> > GoodContours;

    



  for (size_t idx = 0; idx < contours.size(); idx++) {

        int area = contourArea(contours[idx]);

        if (area > minObjectArea) {

              GoodContours.push_back(contours.at(idx));

        }

    }



    // get the moments

    vector<Moments> mu(GoodContours.size());

    for( int i = 0; i<GoodContours.size(); i++ ){

        mu[i] = moments( GoodContours[i], false );

    }



    // get the centroid of figures.

    vector<Point2f> mc(GoodContours.size());

    vector<RotatedRect> boundRect( GoodContours.size() );

    vector<Rect> boundBox( GoodContours.size() );

    vector<vector<Point> > contours_poly( GoodContours.size() );



    for( int i = 0; i<GoodContours.size(); i++){

        

        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

        boundRect[i] = minAreaRect( GoodContours[i] );

        approxPolyDP( Mat(GoodContours[i]), contours_poly[i], 3, true );

        boundBox[i] = boundingRect( Mat(contours_poly[i]) );

        

    }



    // for( int i = 0; i<mc.size(); i++){

    //     cout << mc[i] << endl;

    // }



    Mat objects_img = image_resized.clone();

    

    for( size_t i = 0; i< GoodContours.size(); i++ )

    {

        int area = contourArea(GoodContours[i]);

            

        // Draw the contours with thickness -1 in order to fill the contour

        drawContours( objects_img, GoodContours, (int)i, colorGreen, 2, LINE_8, hierarchy, 0 );

        drawContours( objects_only, GoodContours, (int)i, colorWhite, -1, LINE_8, hierarchy, 0 );

        circle( objects_img, mc[i], 4, colorGreen, -1, 8, 0 );

        rectangle( objects_img, boundBox[i].tl(), boundBox[i].br(), colorGreen, 2, 8, 0 );            // rectangle( objects_img, boundRect[i].tl(), boundRect[i].br(), colorGreen, 2 );

        // rotated rectangle

        Point2f rect_points[4];

        boundRect[i].points( rect_points );

        for ( int j = 0; j < 4; j++ )

        {

            line( objects_img, rect_points[j], rect_points[(j+1)%4], colorGreen, 2 );

        }

        putText(objects_img,"Area:" + to_string(area),rect_points[2],FONT_HERSHEY_PLAIN, 1 ,colorGreen,1.6);

        cv::Rect leftRoi(boundBox[i].x, boundBox[i].y, (mc[i].x-boundBox[i].x), boundBox[i].height);

        cv::Rect rightRoi(mc[i].x, boundBox[i].y, (boundBox[i].br().x-mc[i].x), boundBox[i].height);

        //Create the cv::Mat with the ROI you need, where "image" is the cv::Mat you want to extract the ROI from

        cv::Mat leftPartOfObject = objects_only(leftRoi);

        cv::Mat rightPartOfObject = objects_only(rightRoi);

        cv::Mat leftCrop = image_resized(leftRoi);

        cv::Mat rightCrop = image_resized(rightRoi);

        cv::Mat leftObjectOnly;

        cv::Mat rightObjectOnly;

        cv::bitwise_and(leftPartOfObject, leftCrop, leftObjectOnly);

        cv::bitwise_and(rightPartOfObject, rightCrop, rightObjectOnly);

        leftObjectOnly.setTo(1, leftObjectOnly == 0);

        rightObjectOnly.setTo(1, rightObjectOnly == 0);



        auto meanLeft(cv::mean(leftObjectOnly));

        auto meanRight(cv::mean(rightObjectOnly));

        



        // cout << "Mean: " << meanLeft << ", " << meanRight << endl;

        int errorFactor = 10;

        // Add text For identification of Holes and Objects

        if (meanLeft[0]-meanRight[0] > errorFactor && meanRight[2]-meanLeft[2] > errorFactor){

            putText(objects_img,"Hole",boundBox[i].br(),FONT_HERSHEY_PLAIN, 1 ,colorGreen,1.6);

        }else if(meanLeft[2]-meanRight[2] > errorFactor && meanRight[0]-meanLeft[0] > errorFactor){

            putText(objects_img,"Object",boundBox[i].br(),FONT_HERSHEY_PLAIN, 1 ,colorGreen,1.6);

        }else{

            putText(objects_img,"Undefined",boundBox[i].br(),FONT_HERSHEY_PLAIN, 1 ,colorGreen,1.6);

        }

        // imshow("Left",leftPartOfObject);

        // imshow("Left Final",leftObjectOnly);

        // imshow("Right Final",rightObjectOnly);

        // imshow("Right",rightPartOfObject);



    }

    imshow( "Contours", objects_img );



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



static void makeReconB(Mat inputMatrix1 ,Mat inputMatrix2 ,Mat& outputMatrix){

    Mat tmp(inputMatrix1.cols,inputMatrix1.rows , CV_32FC1);

    Mat tmp2(inputMatrix2.cols,inputMatrix2.rows , CV_32FC1);

    Mat final(inputMatrix1.cols,inputMatrix1.rows , CV_32FC1);

    Mat final_not(inputMatrix1.cols,inputMatrix1.rows , CV_32FC1);



    float inf = numeric_limits<float>::infinity();



    inRange(inputMatrix1, 0 , 0.9999999999999999999 , tmp);

    inRange(inputMatrix2, 1.000000000000000001 , inf , tmp2);

    // divide(tmp,255,tmp);

    // divide(tmp2,255,tmp2);

    // tmp.convertTo(tmp,inputMatrix1.type());

    // tmp2.convertTo(tmp2,inputMatrix2.type());

    bitwise_and(tmp,tmp2,final);

    bitwise_not(final,outputMatrix);



}



static void makeReconR(Mat inputMatrix1 ,Mat inputMatrix2,Mat& outputMatrix){

    Mat tmp(inputMatrix1.cols,inputMatrix1.rows , CV_32FC1);

    Mat tmp2(inputMatrix2.cols,inputMatrix2.rows , CV_32FC1);

    float inf = numeric_limits<float>::infinity();



    inRange(inputMatrix1, 1.000000000000000001 , inf , tmp);

    inRange(inputMatrix2, 0 , 0.9999999999999999999, tmp2);

    divide(tmp,255,tmp);

    divide(tmp2,255,tmp2);

    tmp.convertTo(tmp,inputMatrix1.type());

    tmp2.convertTo(tmp2,inputMatrix2.type());

    bitwise_and(tmp,tmp2,outputMatrix);



}



static void simulateObject(){

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



int main(int argc, char** argv)

{

    double startTime,stopTime;

    string path = "/home/jim/Desktop/Linux_C_PP/IMG_2167.mp4";

    Mat Purple,Purple_resized;

    Mat Purplebgr[3];   // Calibration destination array

    Mat bgr[3];   // Frame destination array





    VideoCapture cap(path);

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));      // More fps less resolution (at least for my setup)

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, IMAGE_W);

    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMAGE_H);

    // cap.set(cv::CAP_PROP_FPS, 60);

    int dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH); 

    int dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    int fps_counter = cap.get(cv::CAP_PROP_FPS);

    cap.read(Purple);

    resize(Purple,Purple_resized,Size(),0.5,0.5);

    split(Purple_resized, Purplebgr);



    Mat zeroes = Mat::zeros(Size(Purple.cols, Purple.rows), CV_32FC1);



    int width = Purple_resized.cols;

    int height = Purple_resized.rows;



    // Mat ones = Mat::ones(Size(Purple.cols, Purple.rows), 1);

    //Mat Irn, Ibn;

    // Mat Ibn = Mat(height, width, CV_8UC1);

    // Mat Irn = Mat(height, width, CV_8UC1);

    //Mat proc = Mat(height, width, CV_8UC1);

    //Irn = Mat(2, 2, CV_8UC3, Scalar(1, 1, 1));

    //cout << Irn;

    Mat irn(width, height, CV_32FC1);

    Mat ibn(width, height, CV_32FC1);

    // Mat PurpleF(width, height, CV_32FC3);

    Mat PurplebgrFB(width, height, CV_32FC1);

    Mat PurplebgrFR(width, height, CV_32FC1);

    



    Purplebgr[0].setTo(1, Purplebgr[0] == 0);

    Purplebgr[2].setTo(1, Purplebgr[2] == 0);

    Purplebgr[0].convertTo(PurplebgrFB,ibn.type());

    Purplebgr[2].convertTo(PurplebgrFR,irn.type());

    divide(1,PurplebgrFB,ibn);

    divide(1,PurplebgrFR,irn);

    // In.setTo(0, In == 1);



    // Mat img2;

    // normalize(In, dst, 0, 1, cv::NORM_MINMAX);

    // namedWindow("test", WINDOW_AUTOSIZE);



    // prospatheia na enosoume ksana tiw phtografies



    // for (int y = 0; y < In.rows; ++y) {

    //     for (int x = 0; x < In.cols; ++x){

    //         std::cout << 1 << "/" << Purple.at<> << " == " << In.at<__uint8_t>(y, x) << "\t";

    //     }

    //     std::cout << std::endl;

    // }

    // for (int i = 0; i < height; i++)

    // {

    //     for (int j = 0; j < width; j++)

    //     {

    //         if (Purple.at<Vec3b>(i, j)[0] != 0)

    //             Ibn.at<uchar>(i, j) = 1 / Purple.at<Vec3b>(i, j)[0];

    //         else

    //             Ibn.at<uchar>(i, j) = 1;



    //         if (Purple.at<Vec3b>(i, j)[2] != 0)

    //             Irn.at<uchar>(i, j) = 1 / Purple.at<Vec3b>(i, j)[2];

    //         else

    //             Irn.at<uchar>(i, j) = 1;

    //     }

    // }



    



    // namedWindow("Raw Image", WINDOW_AUTOSIZE);

    // namedWindow("Edges Image", WINDOW_AUTOSIZE);

    // namedWindow("Dilated Image", WINDOW_AUTOSIZE);

    namedWindow("TrackBars", WINDOW_AUTOSIZE);

    // namedWindow("Contours Objects", WINDOW_AUTOSIZE);



    // Create Task Bar In progress

    createTrackbar("Min Threshold:", "TrackBars", &lowThreshold, max_lowThreshold, findObjects );

    createTrackbar("Max Threshold:", "TrackBars", &maxThreshold, max_maxThreshold, findObjects );

    createTrackbar("Min Object Area:", "TrackBars", &minObjectArea, 30000, findObjects );



    bool playVideo = true;



    while (true)

    {

        if (playVideo)

        {

            startTime = gettime();

            cap.read(image);

        }

        if (!image.data) 

        {

            printf("No image data \n");

            return -1;

        }

        resize(image,image_resized,Size(),0.5,0.5);

        objects_only = Mat::zeros( image_resized.size(), CV_8UC3 );

        // imshow("Raw Image",image_resized);

        findObjects(lowThreshold,0);

        if (objects_only.data)

        {   

            //Proccess Only the Object pixels

            bitwise_and(image_resized,objects_only,image_proccesing);

            split(image_proccesing, bgr);//split source

            

            bgr[0].convertTo(bgrFB,ibn.type());

            bgr[2].convertTo(bgrFR,irn.type());

            multiply(bgrFB,ibn,in1);

            multiply(bgrFR,irn,in2);



            divide(in1,in2,i12);

            divide(in2,in1,i21);



            GaussianBlur(i12, i12, cv::Size(3, 3), 5, 5);

            GaussianBlur(i21, i21, cv::Size(3, 3), 5, 5);



            // imshow("I12", i12);

            // imshow("I21", i21);





            // Object Simulation Algorithm

            simulateObject();



            // floorThreshold(i12,thres12,0.2);

            // floorThreshold(i21,thres21,0.2);





            // makeReconR(i12,i21,final12);

            // makeReconB(i12,i21,final21);



            // imshow("ReconR", i12);

            // imshow("ReconB", i21);

            



            // moveWindow("Raw Image",100,100);

            // normalize(final21, final21, 0, 1, cv::NORM_MINMAX);

            // namedWindow("Recon R", WINDOW_AUTOSIZE);

            // imshow("Recon R", final12);

            // namedWindow("Recon B", WINDOW_AUTOSIZE);

            // imshow("Recon B", final21);

        }else{

            printf("No Objects \n");

        }

        

        stopTime = gettime();

        int fps = 1/(stopTime-startTime);

        cout << "Frame Time :" << (stopTime-startTime)*1000 << " | FPS : " << fps << endl;

        char key = waitKey(50);

        if(key == 'p')

            playVideo = !playVideo;

    }



    destroyAllWindows();

    return 0;

}
