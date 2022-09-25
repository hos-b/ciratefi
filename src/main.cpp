#include <cstdlib>
#include <filters/circ_sample_filter.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char ** argv)
{
    int chosen_scale = 0;
    if (argc > 1)
        chosen_scale = std::stoi(argv[1]);

    cv::Mat image = cv::imread("/home/hosein/image.png", cv::IMREAD_GRAYSCALE);
    cv::Mat templ =
        cv::imread("/home/hosein/template.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_org = image.clone();

    auto circ_radii = {2, 4, 10, 12, 20, 30, 40};
    auto scales = {0.6, 0.8, 1.0, 1.2, 1.3, 1.4};

    image.convertTo(image, CV_64FC1, 1.0 / 255);
    templ.convertTo(templ, CV_64FC1, 1.0 / 255);

    auto img_f = ciratefi::circle::compute_image_features(image, circ_radii);
    auto tmp_f =
        ciratefi::circle::compute_template_features(templ, scales, circ_radii);

    auto [corr, scale] =
        ciratefi::circle::compute_circular_correlation(img_f, tmp_f);

    cv::Mat correlation(image.rows, image.cols, CV_64FC1, corr.data_ptr());
    cv::threshold(correlation, correlation, 0.0, 1.0, cv::THRESH_TOZERO);

    cv::Mat viz;
    cv::cvtColor(image_org, viz, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < viz.rows; ++i) {
        for (int j = 0; j < viz.cols; ++j) {
            double corr_num = correlation.at<double>(i, j);
            if (scale.index({i, j}).item().to<int32_t>() != chosen_scale || corr_num < 0.7)
                continue;
            viz.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
        }
    }
    cv::imshow("win", viz);
    cv::waitKey(0);
    return 0;
}