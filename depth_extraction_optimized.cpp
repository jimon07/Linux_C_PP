#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/viz.hpp>

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

int main() {
    //First part
    // Load the input image from 'output_0001.png'
    cv::Mat input_image = cv::imread("../dataset/Images_3/image-0001.png");

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
    input_image_2.convertTo(input_image_2, CV_64FC3);

    //Define constants
    cv::setNumThreads(24);
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

    for(int iteration = 0; iteration<10; iteration++)
    {
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
    }

    // Find the minimum and maximum values in the depth map
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(depth_map, &min_val, &max_val, &min_loc, &max_loc);

    // Print the maximum value
    std::cout << "Maximum depth value: " << max_val << std::endl;

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

    cv::Mat normalized_depth_map;
    cv::normalize(depth_map, normalized_depth_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_heatmap;
    cv::applyColorMap(normalized_depth_map, depth_heatmap, cv::COLORMAP_JET);

    int scale_height = depth_heatmap.rows;
    int scale_width = 50; // Width of the scale bar
    cv::Mat scale_bar(scale_height, scale_width, CV_8UC3);

    // Create a gradient for the scale bar
    for (int i = 0; i < scale_height; ++i) {
        // Map the loop variable to 0-255 and create a color from the colormap
        uchar value = static_cast<uchar>((255 * i) / scale_height);
        cv::Mat single_color_row(1, 1, CV_8U, cv::Scalar(value));
        
        // Apply the colormap to the single pixel
        cv::Mat colored_row;
        cv::applyColorMap(single_color_row, colored_row, cv::COLORMAP_JET);
        
        // Repeat the colored pixel across the width of the scale bar
        cv::Mat color_row;
        cv::repeat(colored_row.reshape(3,1), 1, scale_width, color_row);
        
        // Assign the colored row to the correct position in the scale bar
        color_row.copyTo(scale_bar.row(i));
    }

    // Draw scale bar with scale
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    cv::Point textPosMin(1, scale_height - 15);
    cv::Point textPosMax(1, 20);

    std::ostringstream minStream, maxStream;
    minStream << std::fixed << std::setprecision(2) << max_val;
    maxStream << std::fixed << std::setprecision(2) << min_val;
    std::string minText = minStream.str();
    std::string maxText = maxStream.str();

    cv::putText(scale_bar, minText, textPosMin, fontFace, fontScale, cv::Scalar(255,255,255), thickness);
    cv::putText(scale_bar, maxText, textPosMax, fontFace, fontScale, cv::Scalar(255,255,255), thickness);

    cv::Mat combined_image;
    cv::hconcat(depth_heatmap, scale_bar, combined_image);

    cv::imshow("Depth Heatmap with Scale", combined_image);
    cv::waitKey(0);

    return 0;
}
