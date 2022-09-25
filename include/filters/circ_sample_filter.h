#include <ATen/ATen.h>
#include <cstdint>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ciratefi::circle
{

at::Tensor compute_template_features(cv::Mat raw_template,
                                     const std::vector<double> &scales,
                                     const std::vector<int> &radi,
                                     int resize_inter_flag = cv::INTER_LINEAR);

at::Tensor compute_image_features(cv::Mat image, const std::vector<int> &radi);

std::tuple<at::Tensor, at::Tensor>
compute_circular_correlation(const at::Tensor &image_features,
                             const at::Tensor &template_features);

} // namespace ciratefi::circle