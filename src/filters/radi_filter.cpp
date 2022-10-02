#include <cmath>
#include <cstdint>
#include <initializer_list>

#include <ATen/ATen.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace ciratefi::radial
{
/**
 * @brief samples along a line on the image using Bresenheim's line
 *        algorithm given the starting point, slope angle, and line
 *        length (circle radius).
 *
 * @param image image to be samples
 * @param xc starting x of the line (circle center)
 * @param yc starting y of the line (circle center)
 * @param lambda length of the line to sample
 * @param angle slope angle
 * @return double
 */
static double sample_line(cv::Mat image, const int xc, const int yc,
                          const int lambda, const double angle)
{
    double sum = 0.0;
    int count = 0;
    int h = image.rows;
    int w = image.cols;
    int x0, y0, x1, y1;
    x0 = xc;
    y0 = yc;
    x1 = xc + std::round(std::cos(angle) * lambda);
    y1 = yc - std::round(std::sin(angle) * lambda);
    int dx = x1 - x0;
    int dy = y1 - y0;
    if (std::abs(dy) < std::abs(dx)) {
        if (dx < 0) {
            std::swap(x0, x1);
            std::swap(y0, y1);
            dx = -dx;
            dy = -dy;
        }
        int yi = +1;
        if (dy < 0) {
            yi = -1;
            dy = -dy;
        }
        int D = (2 * dy) - dx;
        int y = y0;
        count = dx + 1;
        for (int x = x0; x <= x1; ++x) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                sum += image.at<double>(y, x);
            }
            if (D > 0) {
                y = y + yi;
                D = D + (2 * (dy - dx));
            } else {
                D = D + 2 * dy;
            }
        }
    } else {
        if (dy < 0) {
            std::swap(x0, x1);
            std::swap(y0, y1);
            dx = -dx;
            dy = -dy;
        }
        int xi = +1;
        if (dx < 0) {
            xi = -1;
            dx = -dx;
        }
        int D = (2 * dx) - dy;
        int x = x0;
        count = dy + 1;
        for (int y = y0; y <= y1; ++y) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                sum += image.at<double>(y, x);
            }
            if (D > 0) {
                x = x + xi;
                D = D + (2 * (dx - dy));
            } else {
                D = D + 2 * dx;
            }
        }
    }
    return sum / count;
}

/**
 * @brief computes radial features given the template image
 *
 * @param raw_template template image [H', W']
 * @param lambda length of the sampling lines (largest radius in cifi)
 * @param angles vector of angles to sample of size N [A]
 * @return at::Tensor tensor for angle i, shift j [A, A]
 */
at::Tensor compute_template_features(cv::Mat raw_template, const int lambda,
                                     const std::vector<double> &angles)
{
    const int angle_count = angles.size();
    // the feature vector and all its N possible shifts
    at::Tensor rq = at::empty(
        {static_cast<long>(angle_count), static_cast<long>(angle_count)},
        at::kDouble);
    const int xc = raw_template.cols / 2;
    const int yc = raw_template.rows / 2;

    auto accessor = rq.accessor<double, 2>();
#pragma omp parallel for
    for (int a = 0; a < angle_count; ++a) {
        double sample = sample_line(raw_template, xc, yc, lambda, angles[a]);
        for (int j = 0; j < angle_count; ++j) {
            accessor[j][(j + a) % angle_count] = sample;
        }
    }
    return rq;
}

/**
 * @brief computes the radial image features for the first grade candidates
 *
 * @param image input image [H, W]
 * @param angles angles for radial sampling [A]
 * @param lambda length of the sampling lines (largest radius in cifi)
 * @param fg_mask first grade pixels mask [H, W]
 * @return at::Tensor radial features [H, W, A]
 */
at::Tensor compute_image_features(const cv::Mat image,
                                  const std::vector<double> &angles,
                                  const int lambda, const at::Tensor &fg_mask)
{
    const int image_h = image.rows;
    const int image_w = image.cols;
    const int angle_count = angles.size();
    // intialize feature tensor with zeros
    at::Tensor radial_features =
        at::zeros({image_h, image_w, angle_count}, at::kDouble);
    auto rafi_accessor = radial_features.accessor<double, 3>();
    auto mask_accessor = fg_mask.accessor<uint8_t, 2>();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < image_h; ++i) {
        for (int j = 0; j < image_w; ++j) {
            // discard irrelevant pixels
            if (mask_accessor[i][j]) {
                for (int a = 0; a < angle_count; ++a) {
                    rafi_accessor[i][j][a] =
                        sample_line(image, j, i, lambda, angles[a]);
                }
            }
        }
    }
    return radial_features;
}

/**
 * @brief calculate cross correlation across different angles
 *
 * @param image_features per pixel radial projection features [H, W, A]
 * @param template_features per angle shift radial projection features [A, A]
 * @param fg_mask first grade pixels mask [H, W]
 * @return max cross-corelation and the arg-max for angle detection
 */
std::tuple<at::Tensor, at::Tensor>
compute_correlation(const at::Tensor &image_features,
                    const at::Tensor &template_features)
{
    using namespace at::indexing;
    const int angle_count = template_features.size(0);
    const int image_h = image_features.size(0);
    const int image_w = image_features.size(1);

    // cross correlation tensor, per pixel per angle shift
    // at::Tensor cross_corr =
    //     at::zeros({image_h, image_w, angle_count}, at::kDouble);

    // image angular mean [H, W]
    const at::Tensor img_angle_mean = image_features.mean({2});
    // intermediate results: image_features - mean [H, W, A]
    at::Tensor img_std_interm =
        image_features -
        img_angle_mean.unsqueeze(2).expand({image_h, image_w, angle_count});
    // std across angles: [H, W]
    const at::Tensor image_std = img_std_interm.square().sum({2}).sqrt();

    // template angular mean (same for all)
    const double tmp_angle_mean =
        template_features.mean({1}).index({0}).item().toDouble();
    // intermediate results: template_features - mean [A, A]
    at::Tensor tmp_std_interm = template_features - tmp_angle_mean;
    // template std across different angles (same for all shifts)
    const double template_std =
        tmp_std_interm.square().sum({1}).sqrt().index({0}).item().toDouble();

    at::Tensor cross_corr =
        (tmp_std_interm.reshape({angle_count, 1, 1, angle_count})
             .expand({angle_count, image_h, image_w, angle_count}) *
         img_std_interm)
            .sum({3}) /
        (template_std *
         image_std.unsqueeze(0).expand({angle_count, image_h, image_w}));

    // accessors
    // auto mask_accessor = fg_mask.accessor<uint8_t, 2>();
    // auto corr_accessor = cross_corr.accessor<double, 3>();
    // auto img_istd_accessor = img_std_interm.accessor<double, 3>();
    // auto img_std_accessor = image_std.accessor<double, 2>();
    // auto tmp_istd_accessor = tmp_std_interm.accessor<double, 2>();
    // #pragma omp parallel for collapse(2)
    //     for (int i = 0; i < image_h; ++i) {
    //         for (int j = 0; j < image_w; ++j) {
    //             // discard irrelevant pixels
    //             if (mask_accessor[i][j]) {
    //                 double highest_cc = -1.0;
    //                 int shift_index = -1;
    //                 double image_std = img_std_accessor[i][j];
    //                 for (int a = 0; a < angle_count; ++a) {
    //                     double cross_corr =
    //                         (img_std_interm.index({i, j, Ellipsis})
    //                             .dot(tmp_std_interm.index({a, Ellipsis}))
    //                             /
    //                         (image_std *
    //                         template_std)).item().to<double>();
    //                     if (cross_corr > highest_cc) {
    //                         highest_cc = cross_corr;
    //                         shift_index = a;
    //                     }
    //                 }

    //             }
    //         }
    //     }

    return cross_corr.max(0);
}

} // namespace ciratefi::radial