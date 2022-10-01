#include <ATen/ATen.h>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define debug_trap __asm__ volatile("int3")

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
 * @return std::pair<double, double> [sample(angle + pi), sample(angle)]
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
 * @param raw_template template image
 * @param lambda line length (max circle radius)
 * @param angles vector of angles to sample of size N
 * @return at::Tensor [N, 1] tensor
 */
at::Tensor compute_template_features(cv::Mat raw_template, const int lambda,
                                     const std::vector<double> &angles)
{
    at::Tensor rq = at::empty({static_cast<long>(angles.size())}, at::kDouble);
    auto accessor = rq.accessor<double, 1>();
    const int xc = raw_template.cols / 2;
    const int yc = raw_template.rows / 2;

#pragma omp parallel for
    for (size_t a = 0; a < angles.size(); ++a) {
        accessor[a] = sample_line(raw_template, xc, yc, lambda, angles[a]);
    }
    return rq;
}

} // namespace ciratefi::circle