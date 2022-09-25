#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <filters/circ_sample_filter.h>

namespace ciratefi::circle
{

/**
 * @brief uses Bresenham’s circle drawing algorithm, to sample a circle in an
          image
 * @param image 8-bit grayscale image
 * @param xc center x
 * @param yc center y
 * @param r radius
 * @return std::vector<cv::Point2i>
 */
static double sample_circle(cv::Mat image, int xc, int yc, int r)
{
    double sum = 0.0f;
    size_t count = 0;
    auto sample = [&image, &sum, &count, w = image.cols,
                   h = image.rows](int x, int y) {
        count += 1;
        if (x >= 0 && x < w && y >= 0 && y < h)
            sum += static_cast<double>(image.at<double>(y, x));
    };
    // make sure there is no realloc
    int x = 0;
    int y = r;
    int d = 3 - (2 * r);
    sample(xc + x, yc + y);
    sample(xc - x, yc + y);
    sample(xc + x, yc - y);
    sample(xc - x, yc - y);
    sample(xc + y, yc + x);
    sample(xc - y, yc + x);
    sample(xc + y, yc - x);
    sample(xc - y, yc - x);
    while (y >= x) {
        x += 1;
        if (d > 0) {
            y -= 1;
            d += 4 * (x - y) + 10;
        } else {
            d += 4 * x + 6;
        }
        sample(xc + x, yc + y);
        sample(xc - x, yc + y);
        sample(xc + x, yc - y);
        sample(xc - x, yc - y);
        sample(xc + y, yc + x);
        sample(xc - y, yc + x);
        sample(xc + y, yc - x);
        sample(xc - y, yc - x);
    }
    return sum / count;
}

/**
 * @brief calcualte multi-scale rotation invariant features for template
 *
 * @param raw_template tempalte image
 * @param scales scales
 * @param radi radii
 * @param resize_inter_flag interpolation flag for resizing
 * @return cv::Mat feature matrix
 */
at::Tensor compute_template_features(cv::Mat raw_template,
                                     const std::vector<double> &scales,
                                     const std::vector<int> &radii,
                                     int resize_inter_flag)
{
    at::Tensor cq = at::empty(
        {static_cast<long>(scales.size()), static_cast<long>(radii.size())},
        at::kDouble);
    auto accessor = cq.accessor<double, 2>();
    for (size_t s = 0; s < scales.size(); ++s) {
        cv::Mat resized_template;
        if (scales[s] == 1) {
            resized_template = raw_template.clone();
        } else {
            cv::resize(raw_template, resized_template, cv::Size(), scales[s],
                       scales[s], resize_inter_flag);
        }
        int xc = resized_template.cols / 2;
        int yc = resized_template.rows / 2;
        // after this radius, circle falls completely outside the image
        int max_radius = std::sqrt(xc * xc + yc * yc);
        for (size_t r = 0; r < radii.size(); ++r) {
            // circle sampling
            accessor[s][r] =
                radii[r] <= max_radius
                    ? sample_circle(resized_template, xc, yc, radii[r])
                    : 0.0;
        }
    }
    std::cout << "template samples:\n" << cq << "\n";
    return cq;
}

at::Tensor compute_image_features(cv::Mat image, const std::vector<int> &radii)
{
    at::Tensor circ_features = at::empty(
        {static_cast<long>(radii.size()), image.rows, image.cols}, at::kDouble);
    auto accessor = circ_features.accessor<double, 3>();
    for (size_t r = 0; r < radii.size(); ++r) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                accessor[r][i][j] = sample_circle(image, j, i, radii[r]);
            }
        }
    }
    return circ_features;
}

/**
 * @brief calculate cross correlation across different
 *
 * @param image_features per pixel circular projection features [R, H, W]
 * @param template_features per scale circular projection features [S, R]
 * @return max cross-corelation and the arg-max for scale detection
 */
std::tuple<at::Tensor, at::Tensor>
compute_circular_correlation(const at::Tensor &image_features,
                             const at::Tensor &template_features)
{
    using namespace at::indexing;
    const int scale_count = template_features.size(0);
    const int radius_count = image_features.size(0);
    const int image_h = image_features.size(1);
    const int image_w = image_features.size(2);
    // cross correlation between the image and the template [H, W]
    at::Tensor correlation = at::empty({image_h, image_w}, at::kDouble);
    // argmax for determining the scale [H, W]
    at::Tensor corr_argmax = at::empty({image_h, image_w}, at::kInt);
    // calculate image mean & std across different radii [H, W]
    const at::Tensor img_radius_mean = image_features.mean({0});

    // intermediate results: image_features - mean [R, H, W]
    at::Tensor img_std_interm =
        at::empty({radius_count, image_h, image_w}, at::kDouble);
    for (int r = 0; r < radius_count; r += 1) {
        img_std_interm.index_put_(
            {r, "..."}, (image_features.index({r, "..."}) - img_radius_mean));
    }
    const at::Tensor image_std = img_std_interm.square().sum({0}).sqrt();

    // calculate template std across different radii
    at::Tensor tmp_radius_mean = template_features.mean({1});
    at::Tensor tmp_std_interm =
        at::empty({scale_count, radius_count}, at::kDouble);
    for (int s = 0; s < scale_count; s += 1) {
        tmp_std_interm.index_put_(
            {s, "..."},
            (template_features.index({s, "..."}) - tmp_radius_mean.index({s})));
    }
    const at::Tensor template_std = tmp_std_interm.square().sum({0}).sqrt();
    // [S, H, W]
    at::Tensor cross_corr =
        at::empty({scale_count, image_h, image_w}, at::kDouble);
    auto t = torch::zeros({});
    for (int s = 0; s < scale_count; ++s) {
        for (int i = 0; i < image_h; ++i) {
            for (int j = 0; j < image_w; ++j) {
                at::Scalar numerator = (img_std_interm.index({"...", i, j}) *
                                        tmp_std_interm.index({s, "..."}))
                                           .sum()
                                           .item();
                cross_corr.index_put_({s, i, j},
                                      numerator / (image_std.index({i, j}) *
                                                   template_std.index({s})));
            }
        }
    }
    return cross_corr.max(0);
}

} // namespace ciratefi::circle