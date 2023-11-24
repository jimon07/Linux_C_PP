#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/viz.hpp>
#include <signal.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

class ParallelProcess : public cv::ParallelLoopBody
{
private:
    const cv::Mat& inverse_lambda;
    const cv::Mat& slope_detected;
    cv::Mat& depth_map;
    const cv::Mat& depth_pixel;
    int num_cols;

public:
    ParallelProcess(const cv::Mat& _inverse_lambda, 
                    const cv::Mat& _slope_detected, 
                    cv::Mat& _depth_map, 
                    const cv::Mat& _depth_pixel, 
                    int _num_cols)
        : inverse_lambda(_inverse_lambda), 
          slope_detected(_slope_detected), 
          depth_map(_depth_map), 
          depth_pixel(_depth_pixel), 
          num_cols(_num_cols) {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        for (int row = range.start; row < range.end; ++row)
        {
            for (int col = num_cols - 2; col >= 0; --col)
            {
                if (inverse_lambda.at<double>(row, col) > 1.1 && slope_detected.at<uchar>(row, col) == 255)
                {
                    depth_map.at<double>(row, col) = depth_map.at<double>(row, col+1) + depth_pixel.at<double>(row, col);
                }
            }
        }
    }
};


class ParallelProcessLeftToRight : public cv::ParallelLoopBody
{
private:
    const cv::Mat& gradient_image;
    const cv::Mat& slope_detected;
    cv::Mat& depth_map;
    const cv::Mat& depth_pixel;
    int num_cols;

public:
    ParallelProcessLeftToRight(const cv::Mat& _gradient_image, 
                               const cv::Mat& _slope_detected, 
                               cv::Mat& _depth_map, 
                               const cv::Mat& _depth_pixel, 
                               int _num_cols)
        : gradient_image(_gradient_image), 
          slope_detected(_slope_detected), 
          depth_map(_depth_map), 
          depth_pixel(_depth_pixel), 
          num_cols(_num_cols) {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        for (int row = range.start; row < range.end; ++row)
        {
            for (int col = 1; col < num_cols; ++col)
            {
                if (gradient_image.at<double>(row, col) > 1.1 && slope_detected.at<uchar>(row, col) == 0)
                {
                    depth_map.at<double>(row, col) = depth_map.at<double>(row, col-1) + depth_pixel.at<double>(row, col);
                }
            }
        }
    }
};

class ParallelProcessScopeComputation : public cv::ParallelLoopBody
{
private:
    const cv::Mat& lambda;
    cv::Mat& slope_detected;
    double t1, L, e;

public:
    ParallelProcessScopeComputation (const cv::Mat& lambda_, double t1_, double L_, double e_, cv::Mat& slope_detected_)
        : lambda(lambda_), t1(t1_), L(L_), e(e_), slope_detected(slope_detected_)
    {}

    virtual void operator()(const cv::Range& range) const
    {
        for (int row = range.start; row < range.end; ++row)
        {
            cv::Mat line_lambda = lambda.row(row);

            double line_mean = cv::mean(line_lambda)[0];
            double lower_bound_1 = line_mean - t1;
            double upper_bound_1 = line_mean + t1;

            cv::Mat initial_mask = (line_lambda > lower_bound_1) & (line_lambda < upper_bound_1);
            cv::Mat initial_valid_values;
            line_lambda.copyTo(initial_valid_values, initial_mask);

            double initial_mean = initial_valid_values.empty() ? 0 : cv::mean(initial_valid_values)[0];
            double t3 = initial_mean - L;
            double t4 = initial_mean + L;

            cv::Mat final_mask = (line_lambda > t3) & (line_lambda < t4);
            cv::Mat final_valid_values;
            line_lambda.copyTo(final_valid_values, final_mask);

            double final_mean = final_valid_values.empty() ? 0 : cv::mean(final_valid_values)[0];

            double t_l = final_mean - e;
            double t_u = final_mean + e;

            cv::Mat current_slope = (line_lambda > t_l) & (line_lambda < t_u);
            current_slope.copyTo(slope_detected.row(row));
        }
    }
};

cv::Mat ProcessImage(const cv::Mat& lambda, double t1, double L, double e, cv::Mat& slope_detected)
{
    ParallelProcessScopeComputation parallelProcess(lambda, t1, L, e, slope_detected);
    cv::parallel_for_(cv::Range(0, lambda.rows), parallelProcess);

    return slope_detected;
}

cv::Mat createDepthScaleBar(cv::Mat& depth_map, double actual_min_val, double actual_max_val) {
    double fixed_min_val = -20.0; // Fixed minimum value for scale
    double fixed_max_val = 20.0;  // Fixed maximum value for scale

    // Clamp values in the depth map to the range of -20 to +20
    cv::Mat extended_depth_map;
    depth_map.convertTo(extended_depth_map, CV_64F); // Convert to double for operations
    cv::threshold(depth_map, extended_depth_map, fixed_max_val, fixed_max_val, cv::THRESH_TRUNC); // Upper clamp
    cv::threshold(depth_map, extended_depth_map, fixed_min_val, fixed_min_val, cv::THRESH_TOZERO); // Lower clamp

    // Extend the depth map by adding rows/columns with -20 and +20
    int rowsToAdd = 2; // Number of rows/columns to add
    extended_depth_map.create(depth_map.rows + rowsToAdd, depth_map.cols, depth_map.type());
    // Set the first and last added row/column to -20 and +20
    extended_depth_map.row(0).setTo(cv::Scalar(fixed_min_val));
    extended_depth_map.row(extended_depth_map.rows - 1).setTo(cv::Scalar(fixed_max_val));
    depth_map.copyTo(extended_depth_map.rowRange(1, extended_depth_map.rows - 1));

    // Normalize to the 0-255 range for coloring
    cv::Mat normalized_depth_map;
    cv::normalize(extended_depth_map, normalized_depth_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Create heatmap from the normalized depth map
    cv::Mat depth_heatmap;
    cv::applyColorMap(normalized_depth_map, depth_heatmap, cv::COLORMAP_JET);

    int scale_height = depth_heatmap.rows;
    int scale_width = 60; // Width of the scale bar
    cv::Mat scale_bar(scale_height, scale_width, CV_8UC3);

    // Create a gradient for the scale bar based on fixed range
    for (int i = 0; i < scale_height; ++i) {
        // Map the loop variable to -20 to 20
        double scale_value = fixed_min_val + (static_cast<double>(i) / (scale_height - 1)) * (fixed_max_val - fixed_min_val);
        
        // Normalize the value to the 0-255 range for the colormap
        // In this case, -20 should map to 0 and +20 to 255
        uchar normalized_value = static_cast<uchar>(255.0 * (scale_value - fixed_min_val) / (fixed_max_val - fixed_min_val));

        cv::Mat single_color_row(1, 1, CV_8U, cv::Scalar(normalized_value));
        cv::Mat colored_row;
        cv::applyColorMap(single_color_row, colored_row, cv::COLORMAP_JET);
        cv::Mat color_row;
        cv::repeat(colored_row.reshape(3,1), 1, scale_width, color_row);
        color_row.copyTo(scale_bar.row(i));
    }

    cv::flip(scale_bar, scale_bar, 0);

    // Draw scale labels for the fixed range
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 2;
    cv::Scalar colour = cv::Scalar(0, 0, 0);

    for (int i = 0; i <= 5; ++i) {
        double scale_val = fixed_min_val + (i / 5.0) * (fixed_max_val - fixed_min_val);
        std::ostringstream scaleStream;
        scaleStream << std::fixed << std::setprecision(2) << scale_val;
        std::string scaleText = scaleStream.str();

        int textPosVertical;
        if(i < 3)
            textPosVertical = scale_height - static_cast<int>((scale_val - fixed_min_val) / (fixed_max_val - fixed_min_val) * scale_height) - 20;
        else if(i >= 2)
            textPosVertical = scale_height - static_cast<int>((scale_val - fixed_min_val) / (fixed_max_val - fixed_min_val) * scale_height) + 20;

        cv::Point textPos(1, textPosVertical);
        cv::putText(scale_bar, scaleText, textPos, fontFace, fontScale, colour, thickness);
    }

    // Combine heatmap and scale bar
    cv::Mat combined_image;
    cv::namedWindow("Depth Heatmap with Scale Bar", cv::WINDOW_AUTOSIZE);
    cv::hconcat(depth_heatmap, scale_bar, combined_image);

    // Display the combined image
    // cv::imshow("Depth Heatmap with Scale Bar", combined_image);
    // return the final image
    return combined_image;
}

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
    // cv::Mat input_image = cv::imread("../dataset/Images_3/image-0001.png");
    cv::Mat input_image;
    capture >> input_image;

    // Split the image into its RGB channels
    std::vector<cv::Mat> channels(3);
    cv::split(input_image, channels);
    cv::Mat& R = channels[0]; // Remember OpenCV is BGR
    cv::Mat& B = channels[2];

    // Set I_rn and I_bn to 1 where R and B are zero, respectively
    cv::Mat I_rn = R.clone();
    I_rn.setTo(1, R == 0);
    cv::Mat I_bn = B.clone();
    I_bn.setTo(1, B == 0);

    // Convert to double and invert
    I_rn.convertTo(I_rn, CV_64F);
    I_bn.convertTo(I_bn, CV_64F);
    cv::divide(1.0, I_rn, I_rn);
    cv::divide(1.0, I_bn, I_bn);

    /*cv::Mat input_image_2 = cv::imread("../dataset/Images_3/image-0400.png");
    input_image_2.convertTo(input_image_2, CV_64FC3);*/

    cv::setNumThreads(4);
    // cv::VideoCapture capture("../dataset/Images_3/image-0400.png");

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    // int codec =  cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');  
    // Get the current time
    auto now = std::chrono::system_clock::now();
    // Convert to local time
    auto localTime = std::chrono::system_clock::to_time_t(now);
    // Format the timestamp as a string
    std::stringstream ss;
    ss << std::put_time(std::localtime(&localTime), "%F_%H-%M-%S");
    std::string date = ss.str();

    int codec = cv::VideoWriter::fourcc('M','J','P','G'); //cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');  
    cv::VideoWriter video("../output-"+date+".avi", codec, 2, cv::Size(1980,1282)); //Size is Capture Width + 60 & Capture Height + 2 


    cv::Mat frame;
    // while (true) {
    signal(SIGINT, signal_callback_handler);
    while(runLoop){
        capture >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Webcam frame is empty." << std::endl;
            break;
        }
        frame.convertTo(frame, CV_64FC3);

        //Define constants
        // Get the number of rows (lines) in the image
        int num_rows = input_image.rows;
        int num_cols = input_image.cols;
        // Define thresholds for detecting slopes
        static double L = 0.6;  //
        static double e = 0.6;  //
        static double t1 = 0.1; //
        // Initialize matrices for slope detection and line means
        cv::Mat slope_detected = cv::Mat::zeros(input_image.size(), CV_8U);  // false(size(lambda))
        cv::Mat temp_slope_detected = cv::Mat::zeros(input_image.size(), CV_8U);  // false(size(lambda))
        std::vector<double> mean_lambda_line(num_rows, 0);  // Initialize vector for storing line-by-line means

        // Given camera intrinsic parameters:
        double focal_length = 1.538756463579103e+03;//1.53903578e+03*2.5e-03;  // in mm
        double sensor_width_mm = 22;  // in mm
        int image_width_pixels = input_image.cols;

        // Calculate object size of one pixel using equation (3.7)
        double objectSize_pixel = sensor_width_mm / image_width_pixels; // Size of one pixel in mm
        // Initializing depth_map to zeros with the same size as lambda
        cv::Mat depth_map = cv::Mat::zeros(input_image.size(),CV_64F);

        int64 start = cv::getTickCount();  // Start the timer
        //Second part
        // Split the second image into its RGB channels
        std::vector<cv::Mat> channels2(3);
        cv::split(frame, channels2);
        cv::Mat& R2 = channels2[0];
        cv::Mat& B2 = channels2[2];

        // Calculate normalized images using the second image channels
        cv::Mat I_n1, I_n2;
        cv::divide(R2, I_bn, I_n1);
        cv::divide(B2, I_rn, I_n2);

        // Calculate the gradient of the image as the ratio of I_n2 to I_n1
        cv::Mat gradient_image;
        cv::divide(I_n1, I_n2, gradient_image);

        // Apply a 5x5 Gaussian filter to the gradient image
        //cv::GaussianBlur(gradient_image, gradient_image, cv::Size(5, 5), 2);

        // lambda is equivalent to gradient_image in this context
        cv::Mat& lambda = gradient_image;
        ProcessImage(lambda, t1, L, e, temp_slope_detected);

        // Calculate depth for each pixel in the lambda image
        cv::Mat depth_pixel;
        depth_pixel = lambda * objectSize_pixel;

        // Process the image in two directions, left to right and right to left

        // Depth accumulation - left to right
        cv::parallel_for_(cv::Range(0, num_rows), ParallelProcessLeftToRight(gradient_image, slope_detected, depth_map, depth_pixel, num_cols));
        /*for (int row = 0; row < num_rows; ++row) {
            for (int col = 1; col < num_cols; ++col) {
                if (gradient_image.at<double>(row, col) > 1.1 && slope_detected.at<uchar>(row, col) == 0) {
                    depth_map.at<double>(row, col) = depth_map.at<double>(row, col-1) + depth_pixel.at<double>(row, col);  // Adjust the data type 'float' if your depth_map or depth_pixel has a different type
                }
            }
        }*/

        // Inverse of lambda
        cv::Mat inverse_lambda = 1.0 / lambda;
        ProcessImage(inverse_lambda, t1, L, e, slope_detected);

        slope_detected = (255 - slope_detected) | (255 - temp_slope_detected);

        // Calculate depth for each pixel in the lambda image
        depth_pixel = inverse_lambda * objectSize_pixel;  // Ensure objectSize_pixel is of type float or double

        // Inverse of gradient_image
        cv::Mat& inverse_gradient_image = inverse_lambda;

        // Depth accumulation - right to left
        cv::parallel_for_(cv::Range(0, num_rows), ParallelProcess(inverse_lambda, slope_detected, depth_map, depth_pixel, num_cols));
        /*for (int row = 0; row < num_rows; ++row) {
            for (int col = num_cols - 2; col >= 0; --col) {
                if (inverse_lambda.at<double>(row, col) > 1.1 && slope_detected.at<uchar>(row, col) == 255) {
                    depth_map.at<double>(row, col) = depth_map.at<double>(row, col+1) + depth_pixel.at<double>(row, col);
                }
            }
        }*/

        double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "Time taken: " << 1/duration << " FPS" << std::endl;

        // Find the minimum and maximum values in the depth map
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(depth_map, &min_val, &max_val, &min_loc, &max_loc);

        // Print the maximum value
        std::cout << "Maximum depth value: " << max_val << std::endl;
        std::cout << "Minimum depth value: " << min_val << std::endl;

        // Example usage
        cv::Mat finalFrame = createDepthScaleBar(depth_map, min_val, max_val);

        cv::imshow("Depth Heatmap with Scale Bar", finalFrame);
        // std::cout << "Final Frame size: " << finalFrame.size() <<"Type" << finalFrame.type() << std::endl;
        // Write the frame into the file 'outcpp.avi'
        video.write(finalFrame); // or video << frame;
        // video << finalFrame;

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or Esc key
            break;
        }
    }

    // cv::waitKey(0);
    capture.release();  // Release the webcam
    video.release();    // Release the Video File
    cv::destroyAllWindows();  // Close any OpenCV windows


    // Create the X, Y meshgrid (just like MATLAB's meshgrid)
    /*cv::Mat X = cv::Mat::zeros(depth_map.size(), CV_64F);
    cv::Mat Y = cv::Mat::zeros(depth_map.size(), CV_64F);
    for(int i = 0; i < depth_map.rows; i++) {
        for(int j = 0; j < depth_map.cols; j++) {
            X.at<double>(i,j) = static_cast<double>(j);
            Y.at<double>(i,j) = static_cast<double>(i);
        }
    }

    // Convert X, Y, depth_map to point cloud
    std::vector<cv::Point3f> points;
    for(int i = 0; i < depth_map.rows; i++) {
        for(int j = 0; j < depth_map.cols; j++) {
            double depth_value = -depth_map.at<double>(i, j)*50; // Ensure this matches the data type of your depth map
            points.push_back(cv::Point3f(X.at<double>(i,j), Y.at<double>(i,j), depth_value));
        }
    }

    // Create a viz window
    cv::viz::Viz3d window("3D Depth Map Visualization");
    window.setBackgroundColor(cv::viz::Color::white());

    // Display the point cloud in the viz window
    cv::viz::WCloud cloud_widget(points, cv::viz::Color::blue());
    window.showWidget("Point Cloud", cloud_widget);

    // Spin and show the viz window
    window.spin();*/

    return 0;
}
