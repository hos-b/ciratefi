
#include <cstdint>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ciratefi
{

class CircularSamplingFilter
{
public:
  CircularSamplingFilter (cv::Mat image_mat, double max_circle_distance,
                          cv::Size quantization,
                          int inter_flag = cv::INTER_LINEAR);
  void set_template (cv::Mat template_mat, double max_circle_distance,
                     cv::Size quantization, int inter_flag);

  template <typename T>
  std::vector<T>
  get_filter (std::initializer_list<size_t> radi)
  {
  }

private:
  cv::Mat calculate_filter (cv::Mat input);

  cv::Mat _img, _img_polar;
  cv::Mat _templ, _templ_polar;

  std::vector<float> _img_filter, templ_filter;
};

} // namespace ciratefi