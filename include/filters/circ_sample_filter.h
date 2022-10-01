#ifndef __CIFI_H__
#define __CIFI_H__

#include <ATen/ATen.h>
#include <cstdint>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "debug_trap.h"

namespace ciratefi::circle
{
at::Tensor compute_template_features(cv::Mat raw_template,
                                     const std::vector<double> &scales,
                                     const std::vector<int> &radii,
                                     int resize_inter_flag = cv::INTER_LINEAR);

at::Tensor compute_image_features(cv::Mat image, const std::vector<int> &radii);

std::tuple<at::Tensor, at::Tensor>
compute_correlation(const at::Tensor &image_features,
                    const at::Tensor &template_features);

} // namespace ciratefi::circle

#endif