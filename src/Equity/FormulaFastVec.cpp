#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>
#include "EquityModel.h"

Eigen::ArrayXXd FormulaFastVec(
    const double x,
    const EquityContext &ctx) // Renamed for brevity
{

    const double SQRT_2 = 1.41421356237309504880;
    const double SQRT_1_2 = 0.70710678118654752440;
    const double z = x * SQRT_2;

    const double c_xx = ctx.c_xx;
    const double c_yy = ctx.c_yy;
    const double c_xy = ctx.c_xy;
    const double c_xx_sqrt = std::sqrt(c_xx);
    const double c_yy_sqrt = std::sqrt(c_yy);

    auto normal_cdf = [&](const Eigen::ArrayXXd &val) -> Eigen::ArrayXXd
    {
        return 0.5 * (1.0 + (val * SQRT_1_2).erf());
    };

    if (std::abs(ctx.corr_xy) < 1e-10)
    {
        const auto prob_term = normal_cdf(-ctx.n_arr / c_yy_sqrt);

        const auto exp1 = Eigen::exp(ctx.m_arr + c_xx / 2.0);
        const auto exp2 = Eigen::exp(ctx.m_arr + ctx.n_arr + c_yy_sqrt * z + (c_xx / 2.0));

        return prob_term * (exp1 - exp2);
    }

    const Eigen::ArrayXXd g_arr = -ctx.n_arr / c_yy_sqrt / ctx.corr_xy;
    const double h = std::sqrt(1.0 - ctx.corr_xy * ctx.corr_xy) / std::sqrt(c_xy);
    const double c_yy_cond_sqrt = std::sqrt(c_yy * (1.0 - c_xy * c_xy));

    Eigen::ArrayXXd d1_arr, d2_arr;

    if (ctx.corr_xy > 0)
    {
        d1_arr = g_arr - c_xx - h * z;
        d2_arr = g_arr - h * z - (c_xx_sqrt + c_yy_sqrt * c_xy);
    }
    else // corr_xy < 0
    {
        d1_arr = c_xx_sqrt - (g_arr - h * z);
        d2_arr = c_xx_sqrt + c_yy_sqrt * c_xy - (g_arr - h * z);
    }

    const Eigen::ArrayXXd Phi_d1 = normal_cdf(d1_arr);
    const Eigen::ArrayXXd Phi_d2 = normal_cdf(d2_arr);

    const double term_squared = (c_xx_sqrt + c_yy_sqrt * c_xy);
    const double half_term_sq = 0.5 * term_squared * term_squared;

    const auto part1 = Eigen::exp(ctx.m_arr + c_xx / 2.0) * Phi_d1;

    const auto part2 = Eigen::exp(
                           ctx.m_arr + ctx.n_arr +
                           c_yy_cond_sqrt * z +
                           half_term_sq) *
                       Phi_d2;

    return part1 - part2;
}