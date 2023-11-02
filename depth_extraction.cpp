#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat ProcessImage(const cv::Mat& lambda, double t1, double L, double e, cv::Mat& slope_detected, std::vector<double>& mean_lambda_line) {

    for (int row = 0; row < lambda.rows; ++row) {
        cv::Mat line_lambda = lambda.row(row);

        cv::Mat initial_mask = (line_lambda > (cv::mean(line_lambda)[0] - t1)) & (line_lambda < (cv::mean(line_lambda)[0] + t1));
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

    return slope_detected;
}


int main() {
    //First part
    // Load the input image from 'output_0001.png'
    cv::Mat input_image = cv::imread("../dataset/Images_3/image-0001.png");
    int zei = 1;
    cv::resize(input_image, input_image, input_image.size()/zei);

    // Split the image into its RGB channels
    std::vector<cv::Mat> channels(3);
    cv::split(input_image, channels);
    cv::Mat& R = channels[2]; // Remember OpenCV is BGR
    cv::Mat& B = channels[0];

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

    // Load the second image 'output_0100.png'
    cv::Mat input_image_2 = cv::imread("../dataset/Images_3/image-0400.png");
    cv::resize(input_image_2, input_image_2, input_image_2.size()/zei);
    input_image_2.convertTo(input_image_2, CV_64FC3);

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
    double focal_length = 1.53903578e+03*2.5e-03;  // in mm
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
    cv::split(input_image_2, channels2);
    cv::Mat& R2 = channels2[2];
    cv::Mat& B2 = channels2[0];

    // Calculate normalized images using the second image channels
    cv::Mat I_n1, I_n2;
    cv::divide(R2, I_bn, I_n1);
    cv::divide(B2, I_rn, I_n2);

    // Calculate the gradient of the image as the ratio of I_n2 to I_n1
    cv::Mat gradient_image;
    cv::divide(I_n1, I_n2, gradient_image);

    // Apply a 5x5 Gaussian filter to the gradient image
    cv::GaussianBlur(gradient_image, gradient_image, cv::Size(5, 5), 2);

    // lambda is equivalent to gradient_image in this context
    cv::Mat& lambda = gradient_image;
    ProcessImage(lambda, t1, L, e, temp_slope_detected, mean_lambda_line);

    // Calculate depth for each pixel in the lambda image
    cv::Mat depth_pixel;
    depth_pixel = lambda * objectSize_pixel;

    // Process the image in two directions, left to right and right to left

    // Depth accumulation - left to right
    for (int row = 0; row < num_rows; ++row)
    {
        for (int col = 1; col < num_cols; ++col)
        {
            bool accumulate_flag = false;
            if (gradient_image.at<double>(row, col) > 1.1)  // Adjust the data type 'float' if your gradient_image has a different type
            {
                accumulate_flag = true;
            }

            if (slope_detected.at<uchar>(row, col) == 0 && accumulate_flag)  // Assuming slope_detected is a binary image
            {
                depth_map.at<double>(row, col) = depth_map.at<double>(row, col-1) + depth_pixel.at<double>(row, col);  // Adjust the data type 'float' if your depth_map or depth_pixel has a different type
            }
        }
    }

    // Inverse of lambda
    cv::Mat inverse_lambda = 1.0 / lambda;
    ProcessImage(inverse_lambda, t1, L, e, slope_detected, mean_lambda_line);

    slope_detected = (255 - slope_detected) | (255 - temp_slope_detected);
    cv::imshow("slope_detected", slope_detected);

    // Calculate depth for each pixel in the lambda image
    depth_pixel = inverse_lambda * objectSize_pixel;  // Ensure objectSize_pixel is of type float or double

    // Inverse of gradient_image
    cv::Mat& inverse_gradient_image = inverse_lambda;

    // Depth accumulation - right to left
    for (int row = 0; row < num_rows; ++row)
    {
        bool accumulate_flag = false;  // Initialize the flag as false
        for (int col = num_cols - 2; col >= 0; --col)
        {
            if (inverse_gradient_image.at<double>(row, col) > 1.1)  // Adjust the data type 'float' if your gradient_image has a different type
            {
                accumulate_flag = true;
            }
            else
            {
                accumulate_flag = false;
            }

            if (slope_detected.at<uchar>(row, col) == 255 && accumulate_flag)  // Assuming slope_detected is a binary image
            {
                depth_map.at<double>(row, col) = depth_map.at<double>(row, col+1) + depth_pixel.at<double>(row, col);  // Adjust the data type 'float' if your depth_map or depth_pixel has a different type
            }
        }
    }

    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Time taken: " << 1/duration << " FPS" << std::endl;

    // Find the minimum and maximum values in the depth map
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(depth_map, &min_val, &max_val, &min_loc, &max_loc);

    // Print the maximum value
    std::cout << "Maximum depth value: " << max_val << std::endl;

    // Create the X, Y meshgrid (just like MATLAB's meshgrid)
    cv::Mat X = cv::Mat::zeros(depth_map.size(), CV_64F);
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
    window.spin();

    cv::imshow("depth_map", depth_map);
    cv::waitKey(0);

    return 0;
}
