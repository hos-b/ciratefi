#include <cstdint>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

namespace ciratefi::circle
{

cv::Mat calculate_template_features(
    cv::Mat raw_template, const std::initializer_list<double> &scales,
    const std::initializer_list<int> &radi,
    int polar_inter_flag = cv::INTER_LINEAR | cv::WARP_POLAR_LINEAR |
                           cv::WARP_FILL_OUTLIERS,
    int resize_inter_flag = cv::INTER_LINEAR);

Eigen::Tensor<double, 3>
compute_image_features(cv::Mat image, const std::initializer_list<int> &radi);

std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<long int, 2>>
compute_circular_correlation(const Eigen::Tensor<double, 3> &image_features,
                             const Eigen::Tensor<double, 2> &template_features);

} // namespace ciratefi::circle