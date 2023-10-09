#include "Custom_functions.hpp"


int main(int argc, char** argv)
{

    // string path = "/home/jim/Desktop/Linux_C_PP/IMG_2167.mp4";
    VideoCapture cap(1, CAP_V4L2);

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));    // More fps less resolution (at least for my setup)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);  // Set Frame Width
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720); // Set Frame Height
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Enable Auto Exposure
    // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);  // Disable Auto Exposure
    cap.set(cv::CAP_PROP_AUTO_WB, 0.0);   // Disable automatic white balance 
    double capfps = cap.get(CAP_PROP_FPS);
    
    Mat image;
    vector<Mat> bgr;

    while (true)
    {
        cap.read(image);

        if (!image.data) 
        {
            printf("No image data \n");
            return -1;
        }
        
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
        
        imshow("Source image", image );
        imshow("calcHist Demo", histImage );
        waitKey(1);
    }

}