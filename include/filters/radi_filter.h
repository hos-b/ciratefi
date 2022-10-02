#ifndef __RAFI_H__
#define __RAFI_H__

#include <cstdint>
#include <initializer_list>

#include <ATen/ATen.h>
#include <opencv2/core.hpp>

#include "debug_trap.h"

namespace ciratefi::radial
{

at::Tensor compute_template_features(cv::Mat raw_template, const int lambda,
                                     const std::vector<double> &angles);

at::Tensor compute_image_features(const cv::Mat image,
                                  const std::vector<double> &angles,
                                  const int lambda, const at::Tensor &fg_mask);

std::tuple<at::Tensor, at::Tensor>
compute_correlation(const at::Tensor &image_features,
                    const at::Tensor &template_features);

} // namespace ciratefi::radial

#endif