#include <cmath>
#include <cstdint>

#include <ATen/ATen.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <filters/tmpl_filter.h>
#include <utils/image_ops.h>
#include <debug_trap.h>

namespace ciratefi::templ
{

/**
 * @brief calcualtes the correlation coefficients (x - mu) / std
 *
 * @param tensor input image patch
 * @return at::Tensor
 */
static at::Tensor get_corr_coeff(const at::Tensor &tensor)
{
    at::Tensor mu = tensor.mean();
    at::Tensor coeff = tensor - mu;
    at::Tensor std = coeff.square().sum().sqrt();
    return coeff / std;
}

/**
 * @brief returns a vector of (normalized cross-correlation) template
 * matching results given the mask of second grade pixels and the argmax
 * tensor from the previous two filters.
 *
 * @param image input image
 * @param cifi_argmax argmax tensor from circular filter
 * @param rafi_argmax argmax tensor from radial filter
 * @param sg_mask second grade pixels
 * @return std::vector<MatchResult>
 */
std::vector<MatchResult> match(const cv::Mat image, const cv::Mat raw_template,
                               const std::vector<double> &scales,
                               const std::vector<double> &angles,
                               const at::Tensor &cifi_argmax,
                               const at::Tensor &rafi_argmax,
                               const at::Tensor &sg_mask)
{
    using namespace at::indexing;

    std::vector<MatchResult> results;
    std::vector<at::Tensor> coords = at::where(sg_mask != 0);
    const int point_count = coords[0].size(0);
    auto u = coords[0].accessor<int64_t, 1>();
    auto v = coords[1].accessor<int64_t, 1>();

    auto cifi_accessor = cifi_argmax.accessor<int64_t, 2>();
    auto rafi_accessor = rafi_argmax.accessor<int64_t, 2>();

    at::Tensor image_tensor = ops::mat_to_tensor(image);

// #pragma omp parallel for
    for (int p = 0; p < point_count; ++p) {
        const int i = u[p];
        const int j = v[p];
        const double scale = scales[cifi_accessor[i][j]];
        const double angle = angles[rafi_accessor[i][j]];

        // scale & rotate template
        cv::Mat sr_templ;
        cv::resize(raw_template, sr_templ, cv::Size(), scale, scale,
                   cv::INTER_LINEAR);
        cv::Point2f center((sr_templ.cols - 1) / 2.0,
                           (sr_templ.rows - 1) / 2.0);
        cv::Mat rotation_matix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(sr_templ, sr_templ, rotation_matix, sr_templ.size());
        const at::Tensor templ_tensor = ops::mat_to_tensor(sr_templ);

        // do boundary checks
        cv::Rect templ_rect(j - sr_templ.cols / 2, i - sr_templ.rows / 2,
                            sr_templ.cols, sr_templ.rows);
        cv::Rect intr = cv::Rect(0, 0, image.cols, image.rows) & templ_rect;
        int tx = templ_rect.x < 0 ? -templ_rect.x : 0;
        int ty = templ_rect.y < 0 ? -templ_rect.y : 0;

        auto sliced_templ = templ_tensor.index(
            {Slice(ty, ty + intr.height), Slice(tx, tx + intr.width)});
        auto sliced_image =
            image_tensor.index({Slice(intr.y, intr.y + intr.height),
                                Slice(intr.x, intr.x + intr.width)});

        auto ccorr =
            (get_corr_coeff(sliced_image) * get_corr_coeff(sliced_templ)).sum();
        
        if (ccorr.isnan().any().item().toBool())
        {
            debug_trap("nan detected");
        }
        MatchResult result{.normalized_ccorr = ccorr.item().toDouble(),
                           .angle = angle,
                           .scale = scale,
                           .x = j,
                           .y = i};
        results.emplace_back(std::move(result));
    }

    return results;
}

} // namespace ciratefi::templ
