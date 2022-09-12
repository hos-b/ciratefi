#include <opencv2/core.hpp>

inline uint8_t get_subpixel_value(const cv::Mat &img, cv::Point2f pt) {
  int x = static_cast<int>(pt.x);
  int y = static_cast<int>(pt.y);

  int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
  int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
  int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
  int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

  double a = pt.x - static_cast<double>(x);
  double c = pt.y - static_cast<double>(y);

  return static_cast<uint8_t>(cvRound(
      (img.at<uint8_t>(y0, x0) * (1.f - a) + img.at<uint8_t>(y0, x1) * a) *
          (1.f - c) +
      (img.at<uint8_t>(y1, x0) * (1.f - a) + img.at<uint8_t>(y1, x1) * a) * c));
}