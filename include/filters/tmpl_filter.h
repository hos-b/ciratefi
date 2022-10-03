#ifndef __TMATCHER_H__
#define __TMATCHER_H__

#include <ATen/ATen.h>
#include <opencv2/core.hpp>

namespace ciratefi::templ
{

struct MatchResult
{
    double normalized_ccorr;
    double angle;
    double scale;
    int x;
    int y;
};

inline std::ostream &operator<<(std::ostream &os, const MatchResult &result)
{
    os << "x: " << result.x << ", y: " << result.y << ", scaled "
       << result.scale << "x"
       << " @" << std::round(result.angle * 180 / M_PI) << " = "
       << result.normalized_ccorr;
    return os;
}

std::vector<MatchResult> match(const cv::Mat image, const cv::Mat raw_template,
                               const std::vector<double> &scales,
                               const std::vector<double> &angles,
                               const at::Tensor &cifi_argmax,
                               const at::Tensor &rafi_argmax,
                               const at::Tensor &sg_mask);

} // namespace ciratefi::templ

#endif