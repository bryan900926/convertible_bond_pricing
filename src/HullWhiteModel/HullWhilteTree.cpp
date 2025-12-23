#include <Eigen/Dense>
#include <utility>

#include "HullWhiteModel.h"



HullWhiteTreeResult HullWhiteTree(const double kappa, const double sigma_r,
                                  const Eigen::ArrayXd &zero_rates, const double dt_first, const double dt_other)
{
    const int j_max_first = std::ceil(0.1835 / (kappa * dt_first));
    const int j_max_other = std::ceil(0.1835 / (kappa * dt_other));

    const int n = zero_rates.size();

    const int non_first = 2 * j_max_first + 1;
    const int non_other = 2 * j_max_other + 1;

    Eigen::ArrayXXd r_tree = HullWhiteFirstStage(n, sigma_r, j_max_first, j_max_other, dt_first, dt_other);

    HullWhiteAlphaResult alphaResult = HullWhiteTreeAlpha(
        n, non_first, non_other,
        dt_first, dt_other, sigma_r,
        kappa, zero_rates,
        j_max_first, j_max_other);

    Eigen::ArrayXXd short_rate_tree = HullWhiteTreeShortRate(
        n, j_max_first, j_max_other,
        alphaResult.alphas, r_tree);

    Eigen::ArrayXXd short_rate = short_rate_tree.colwise().reverse();

    return {std::move(short_rate), std::move(alphaResult), j_max_first, j_max_other};
}