#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/viz.hpp>
#include <signal.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

volatile bool runLoop = true;

void signal_callback_handler(int signum) {
    runLoop = false;
}

int main() {  
    //First part
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return 1;
    }
    cv::Mat frame;
    capture >> frame;    
    //Define constants
    // Get the number of rows (lines) in the image
    int num_rows = frame.rows;
    int num_cols = frame.cols;
    // Get the current time
    auto now = std::chrono::system_clock::now();
    // Convert to local time
    auto localTime = std::chrono::system_clock::to_time_t(now);
    // Format the timestamp as a string
    std::stringstream ss;
    ss << std::put_time(std::localtime(&localTime), "%F_%H-%M-%S");
    std::string date = ss.str();
    // Define the codec and create VideoWriter object.The output is stored in file.

    int codec = cv::VideoWriter::fourcc('M','J','P','G'); //cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');  
    cv::VideoWriter video("../no-process-output-"+date+".avi", codec, 2, cv::Size(num_rows,num_cols));

    // while (true) {
    signal(SIGINT, signal_callback_handler);
    while(runLoop){
        capture >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Webcam frame is empty." << std::endl;
            break;
        }
                
        video.write(frame); // or video << frame;

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or Esc key
            break;
        }
    }

    // cv::waitKey(0);
    capture.release();  // Release the webcam
    video.release();    // Release the Video File
    cv::destroyAllWindows();  // Close any OpenCV windows

    return 0;
}
