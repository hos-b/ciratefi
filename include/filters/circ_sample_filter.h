#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

namespace ciratefi::circle
{

Eigen::Tensor<double, 2> compute_template_features(
    cv::Mat raw_template, const std::vector<double> &scales,
    const std::vector<int> &radi, int resize_inter_flag = cv::INTER_LINEAR);

Eigen::Tensor<double, 3> compute_image_features(cv::Mat image,
                                                const std::vector<int> &radi);

std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<long int, 2>>
compute_circular_correlation(const Eigen::Tensor<double, 3> &image_features,
                             const Eigen::Tensor<double, 2> &template_features);

} // namespace ciratefi::circle