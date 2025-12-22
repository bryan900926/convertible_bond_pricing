#include <Eigen/Dense>
#include <iostream>
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

    const Eigen::ArrayXXd g_arr = -ctx.n_arr / c_yy_sqrt / ctx.corr_xy;
    const double h = std::sqrt(1.0 - ctx.corr_xy * ctx.corr_xy) / ctx.corr_xy;

    Eigen::ArrayXXd d1_arr, d2_arr;

    if (ctx.corr_xy > 0)
    {
        d1_arr = g_arr - c_xx_sqrt - h * z;
        d2_arr = g_arr - h * z - (c_xx_sqrt + c_yy_sqrt * ctx.corr_xy);
    }
    else
    {
        d1_arr = c_xx_sqrt - (g_arr - h * z);
        d2_arr = c_xx_sqrt + c_yy_sqrt * ctx.corr_xy - (g_arr - h * z);
    }

    const Eigen::ArrayXXd Phi_d1 = normal_cdf(d1_arr);
    const Eigen::ArrayXXd Phi_d2 = normal_cdf(d2_arr);

    const double term_squared = (c_xx_sqrt + c_yy_sqrt * ctx.corr_xy);
    const double half_term_sq = 0.5 * term_squared * term_squared;

    const auto part1 = Eigen::exp(ctx.m_arr + c_xx / 2.0) * Phi_d1;
    const double term_z = std::sqrt(c_yy * (1.0 - ctx.corr_xy * ctx.corr_xy)) * z;
    const auto part2 = Eigen::exp(
                           ctx.m_arr + 
                           ctx.n_arr +
                           term_z +
                           half_term_sq) *
                       Phi_d2;
    return part1 - part2;
}