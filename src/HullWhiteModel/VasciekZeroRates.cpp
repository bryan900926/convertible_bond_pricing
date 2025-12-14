#include "HullWhiteModel.h"
#include <Eigen/Dense>
#include <array>

Eigen::ArrayXd VasciekZeroRates(const VasciekParas &params, const Eigen::ArrayXd &maturities)
{
    const double kappa = params.kappa;
    const double r_bar = params.r_bar;
    const double sigma_r = params.sigma_r;
    const double r0 = params.r0;
    const double sigma_r_sq = sigma_r * sigma_r;
    const double kappa_sq = kappa * kappa;

    Eigen::ArrayXd bb = (1 - Eigen::exp(-kappa * maturities)) / kappa;
    Eigen::ArrayXd first_aa = (bb - maturities) * (kappa_sq * r_bar - sigma_r_sq * 0.5) / kappa_sq;
    Eigen::ArrayXd second_aa = bb.square() * sigma_r_sq / (4 * kappa);
    Eigen::ArrayXd aa = Eigen::exp(first_aa - second_aa);

    return (bb * r0 - Eigen::log(aa)) / maturities;
}