#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

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
        if (x >= 0 && x < w && y >= 0 && y < h)
            sum += image.at<double>(y, x);
        count += 1;
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
 * @param polar_inter_flag interpolation flag for polar projection
 * @param resize_inter_flag interpolation flag for resizing
 * @return cv::Mat feature matrix
 */
Eigen::Tensor<double, 2>
compute_template_features(cv::Mat raw_template,
                          const std::vector<double> &scales,
                          const std::vector<int> &radi, int resize_inter_flag)
{
    Eigen::Tensor<double, 2> cq(scales.size(), radi.size());
    for (size_t s = 0; s < scales.size(); ++s) {
        cv::Mat resized_template;
        if (scales[s] == 1) {
            resized_template = raw_template;
        } else {
            cv::resize(raw_template, resized_template, cv::Size(), scales[s],
                       scales[s], resize_inter_flag);
        }
        int xc = resized_template.cols / 2;
        int yc = resized_template.rows / 2;
        int max_radius = std::sqrt(xc * xc + yc * yc);
        for (size_t r = 0; r < radi.size(); ++r) {
            cq(s, r) = radi[r] <= max_radius
                           ? sample_circle(resized_template, xc, yc, radi[r])
                           : 0;
        }
    }
    return cq;
}

Eigen::Tensor<double, 3> compute_image_features(cv::Mat image,
                                                const std::vector<int> &radi)
{
    Eigen::Tensor<double, 3> circ_features(radi.size(), image.rows, image.cols);
    for (size_t r = 0; r < radi.size(); ++r) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                circ_features(r, i, j) = sample_circle(image, j, i, radi[r]);
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
std::pair<Eigen::Tensor<double, 2>, Eigen::Tensor<long int, 2>>
compute_circular_correlation(const Eigen::Tensor<double, 3> &image_features,
                             const Eigen::Tensor<double, 2> &template_features)
{
    const int scale_count = template_features.dimension(0);
    const int radius_count = image_features.dimension(0);
    const int image_h = image_features.dimension(1);
    const int image_w = image_features.dimension(2);
    // cross correlation between the image and the template [H, W]
    Eigen::Tensor<double, 2> correlation(image_h, image_w);
    // argmax for determining the scale [H, W]
    Eigen::Tensor<int, 2> corr_argmax(image_h, image_w);
    // calculate image mean & std across different radii
    const Eigen::Tensor<double, 2> img_radius_mean =
        image_features.mean(Eigen::array<int, 1>({0}));
    Eigen::Tensor<double, 3> img_std_interm(radius_count, image_h, image_w);
    for (int r = 0; r < radius_count; r += 1) {
        img_std_interm.chip(r, 0) =
            (image_features.chip(r, 0) - img_radius_mean);
    }
    const Eigen::Tensor<double, 2> image_std =
        img_std_interm.square().sum(Eigen::array<Eigen::Index, 1>({0})).sqrt();

    // calculate template std across different radii
    const Eigen::Tensor<double, 1> tmp_radius_mean =
        template_features.mean(Eigen::array<Eigen::Index, 1>({1}));
    Eigen::Tensor<double, 2> tmp_std_interm(scale_count, radius_count);
    for (int s = 0; s < scale_count; s += 1) {
        tmp_std_interm.chip(s, 0) =
            (template_features.chip(s, 0) - tmp_radius_mean(s));
    }
    const Eigen::Tensor<double, 1> template_std =
        tmp_std_interm.square().sum(Eigen::array<int, 1>({0})).sqrt();
    // [S, H, W]
    Eigen::Tensor<double, 3> cross_corr(scale_count, image_h, image_w);
    for (int s = 0; s < scale_count; ++s) {
        for (int i = 0; i < image_h; ++i) {
            for (int j = 0; j < image_w; ++j) {
                const Eigen::array<Eigen::Index, 3> offsets = {0, i, j};
                const Eigen::array<Eigen::Index, 3> extents = {radius_count, 1,
                                                               1};
                const Eigen::array<Eigen::Index, 1> new_shape = {radius_count};
                Eigen::Tensor<double, 0> numerator =
                    (img_std_interm.slice(offsets, extents).reshape(new_shape) *
                     tmp_std_interm.chip(s, 0))
                        .sum();
                cross_corr(s, i, j) =
                    numerator(0) / (image_std(i, j) * template_std(s));
            }
        }
    }
    Eigen::Tensor<double, 2> corr =
        cross_corr.maximum(Eigen::array<Eigen::Index, 1>({0}));
    Eigen::Tensor<long int, 2> scale_idx = cross_corr.argmax(0);
    return std::make_pair(corr, scale_idx);
}

} // namespace ciratefi::circle