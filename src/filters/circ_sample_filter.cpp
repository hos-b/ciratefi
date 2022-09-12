#include "filters/circ_sample_filter.h"
#include <opencv2/imgproc.hpp>

namespace ciratefi
{

CircularSamplingFilter::CircularSamplingFilter(cv::Mat image_mat, double max_circle_distance, cv::Size quantization,
                                               int inter_flag)
    : _img(image_mat.clone())
{

    // perform the circular mapping
    _img_polar = cv::Mat::zeros(quantization, _img.type());
    cv::linearPolar(_img, _img_polar, {_img.cols / 2.0f, _img.rows / 2.0f}, max_circle_distance, inter_flag);
}

void CircularSamplingFilter::set_template(cv::Mat template_mat, double max_circle_distance, cv::Size quantization,
                                          int inter_flag)
{
    _templ = template_mat.clone();
    _templ_polar = cv::Mat::zeros(quantization, _templ.type());
    cv::linearPolar(_templ, _templ_polar, {_templ.cols / 2.0f, _templ.rows / 2.0f}, max_circle_distance, inter_flag);
}

cv::Mat CircularSamplingFilter::calculate_filter(cv::Mat input)
{

    return input;
}

} // namespace ciratefi