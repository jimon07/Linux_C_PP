#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <limits>

using namespace cv;
using namespace std;

int lowThreshold = 180;
const int max_lowThreshold = 300;
const int rat = 3;
const int kernel_size = 3;
Mat canny_img,image,image_resized;

static double gettime() {
    struct timeval ttime;
    gettimeofday(&ttime, NULL);
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

static void CannyThreshold(int,void*)
{
    // blur( src_gray, detected_edges, Size(3,3) );
    Canny( image_resized, canny_img, lowThreshold, lowThreshold*rat, kernel_size );
    // dst = Scalar::all(0);
    // src.copyTo( dst, detected_edges);
    imshow( "Raw Image", canny_img );
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

// static void simulateObject(){
//     Mat tmp3 = final_not.clone();

//     Mat kernelDiER = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//     dilate(final_not, final_not, kernelDiER);
//     erode(final_not, final_not, kernelDiER);
    
//     namedWindow("No erosion", WINDOW_AUTOSIZE);
//     imshow("No erosion", tmp3);
//     namedWindow("Erosion", WINDOW_AUTOSIZE);
//     imshow("Erosion", final_not);

//     // cout << final << endl;

//     // imshow("Display Image", final_not);
    
//     // Mat dist;
//     // final.convertTo(final,CV_8UC1);
//     distanceTransform(final_not, outputMatrix, DIST_L2, DIST_MASK_PRECISE);
//     // cout << outputMatrix << endl;
// }

int main(int argc, char** argv)
{
    double startTime,stopTime;
    string path = "/home/jim/Desktop/Diplom_linux/IMG_2167.mp4";
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

    Mat in1,in2,bgrFB,bgrFR,i12,i21,thres12,thres21,final12,final21;

    bool playVideo = true;

    while (true)
    {
        if (playVideo)
        {
            startTime = gettime();
            cap.read(image);
        }

        resize(image,image_resized,Size(),0.5,0.5);
       
        split(image_resized, bgr);//split source
        // bgr[1] = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
        if (!image.data)
        {
            printf("No image data \n");
            return -1;
        }

        bgr[0].convertTo(bgrFB,ibn.type());
        bgr[2].convertTo(bgrFR,irn.type());
        multiply(bgrFB,ibn,in1);
        multiply(bgrFR,irn,in2);

        divide(in1,in2,i12);
        divide(in2,in1,i21);

        GaussianBlur(i12, i12, cv::Size(7, 7), 5, 5);
        GaussianBlur(i21, i21, cv::Size(7, 7), 5, 5);
 
        floorThreshold(i12,thres12,0.5);
        floorThreshold(i21,thres21,0.5);

        makeReconR(thres12,thres21,final12);
        makeReconB(thres12,thres21,final21);

       

        namedWindow("Raw Image", WINDOW_AUTOSIZE);

        // Create Task Bar In progress
        createTrackbar("Min Threshold:", "Raw Image", &lowThreshold, max_lowThreshold, CannyThreshold );
        CannyThreshold(lowThreshold,0);
        // Canny(image, canny_img, lowThreshold, lowThreshold*ratio, kernel_size );
        // imshow("Raw Image", image);

        
        // normalize(final21, final21, 0, 1, cv::NORM_MINMAX);
        namedWindow("Recon R", WINDOW_AUTOSIZE);
        imshow("Recon R", final12);
        namedWindow("Recon B", WINDOW_AUTOSIZE);
        imshow("Recon B", final21);
        stopTime = gettime();
        cout << (stopTime-startTime)*1000 << endl;
        char key = waitKey(50);
        if(key == 'p')
            playVideo = !playVideo;
    }


    return 0;
}