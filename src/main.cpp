#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filters/circ_filter.h>
#include <filters/radi_filter.h>
#include <filters/tmpl_filter.h>

#include <fmt/format.h>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    cv::Mat image = cv::imread("/home/hosein/image.png", cv::IMREAD_GRAYSCALE);
    cv::Mat templ =
        cv::imread("/home/hosein/template.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_org = image.clone();

    auto circ_radii = {2, 4, 10, 12, 14, 16, 18, 20, 22, 24, 26};
    auto scales = {0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2};
    std::vector<double> angles;
    int lambda = *(circ_radii.end() - 1);
    for (int i = 0; i < 360; i += 10)
        angles.emplace_back(i * M_PI / 180.0);
    const double thresh_fg = 0.9;
    const double thresh_sg = 0.9;

    image.convertTo(image, CV_64FC1, 1.0 / 255);
    templ.convertTo(templ, CV_64FC1, 1.0 / 255);

    // circular filter
    auto img_cifi = ciratefi::circle::compute_image_features(image, circ_radii);
    auto tmp_cifi =
        ciratefi::circle::compute_template_features(templ, scales, circ_radii);
    auto [cifi_corr, cifi_argmax] =
        ciratefi::circle::compute_correlation(img_cifi, tmp_cifi);
    // compute first grade mask
    at::Tensor fg_mask = (cifi_corr > thresh_fg).to(at::kByte);

    // radial filter
    auto img_rafi = ciratefi::radial::compute_image_features(image, angles,
                                                             lambda, fg_mask);
    auto tmp_rafi =
        ciratefi::radial::compute_template_features(templ, lambda, angles);
    auto [rafi_corr, rafi_argmax] =
        ciratefi::radial::compute_correlation(img_rafi, tmp_rafi);
    // compute second grade mask
    at::Tensor sg_mask =
        (rafi_corr > thresh_sg).to(at::kByte).bitwise_and(fg_mask);

    // visualization
    cv::Mat correlation(image.rows, image.cols, CV_64FC1, rafi_corr.data_ptr());
    cv::threshold(correlation, correlation, 0.0, 1.0, cv::THRESH_TOZERO);

    cv::Mat viz;
    cv::cvtColor(image_org, viz, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < viz.rows; ++i) {
        for (int j = 0; j < viz.cols; ++j) {
            double corr_num = correlation.at<double>(i, j);
            if (corr_num > thresh_sg)
                viz.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255 * corr_num);
        }
    }
    cv::imshow("win", viz);
    cv::waitKey(0);

    auto results = ciratefi::templ::match(image, templ, scales, angles,
                                          cifi_argmax, rafi_argmax, sg_mask);
    
    for (auto res : results)
        std::cout << res << std::endl;

    return 0;
}