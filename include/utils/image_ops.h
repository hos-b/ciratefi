#include <cmath>

#include <ATen/ATen.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>


namespace ciratefi::ops
{
/**
 * @brief returns the subpixel value in an image
 *
 * @param img input image (assumed to be double)
 * @param pt floating point coordinates
 * @return double subpixel value
 */
inline double get_subpixel_value(const cv::Mat &img, cv::Point2f pt)
{
    int x = static_cast<int>(pt.x);
    int y = static_cast<int>(pt.y);

    int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

    double a = pt.x - static_cast<double>(x);
    double c = pt.y - static_cast<double>(y);

    return (img.at<double>(y0, x0) * (1.0 - a) + img.at<double>(y0, x1) * a) *
               (1.0 - c) +
           (img.at<double>(y1, x0) * (1.0 - a) + img.at<double>(y1, x1) * a) *
               c;
}

/**
 * @brief returns a copy of the input image as a tensor
 *
 * @param mat input image (assumed to be single channel)
 * @return at::Tensor
 */
inline at::Tensor mat_to_tensor(const cv::Mat mat)
{
    at::Tensor tensor = at::empty({mat.rows, mat.cols}, at::kDouble);
    std::memcpy(tensor.data_ptr(), mat.data, mat.total() * mat.elemSize());
    return tensor;
}

} // namespace ciratefi::ops