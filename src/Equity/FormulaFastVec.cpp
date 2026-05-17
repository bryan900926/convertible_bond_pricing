#include "EquityModel.h"
#include <Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>


Eigen::ArrayXXd FormulaFastVec(const double x,
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

  auto normal_cdf_hastings = [](const Eigen::ArrayXXd& x) -> Eigen::ArrayXXd {
      // Hastings constants
      const double p  = 0.2316419;
      const double a1 = 0.319381530;
      const double a2 = -0.356563782;
      const double a3 = 1.781477937;
      const double a4 = -1.821255978;
      const double a5 = 1.330274429;
      const double INV_SQRT_2PI = 0.3989422804014327; // 1.0 / sqrt(2 * pi)

      // 1. Calculate the 't' parameter
      Eigen::ArrayXXd abs_x = x.abs();
      Eigen::ArrayXXd t = 1.0 / (1.0 + p * abs_x);

      // 2. Evaluate the polynomial using Horner's Method for max speed
      Eigen::ArrayXXd poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));

      // 3. Calculate the PDF
      Eigen::ArrayXXd pdf = INV_SQRT_2PI * (-0.5 * x.square()).exp();

      // 4. Calculate the CDF for positive x
      Eigen::ArrayXXd cdf_approx = 1.0 - pdf * poly;

      // 5. Use Eigen's .select() to handle x < 0 without branching (SIMD safe)
      return (x >= 0.0).select(cdf_approx, 1.0 - cdf_approx);
  };


  const Eigen::ArrayXXd g_arr = -ctx.n_arr / c_yy_sqrt / ctx.corr_xy;
  const double h = std::sqrt(1.0 - ctx.corr_xy * ctx.corr_xy) / ctx.corr_xy;

  Eigen::ArrayXXd d1_arr, d2_arr;

  if (ctx.corr_xy > 0) {
    d1_arr = g_arr - c_xx_sqrt - h * z;
    d2_arr = g_arr - h * z - (c_xx_sqrt + c_yy_sqrt * ctx.corr_xy);
  } else {
    d1_arr = c_xx_sqrt - (g_arr - h * z);
    d2_arr = c_xx_sqrt + c_yy_sqrt * ctx.corr_xy - (g_arr - h * z);
  }

  const Eigen::ArrayXXd Phi_d1 = normal_cdf_hastings(d1_arr);
  const Eigen::ArrayXXd Phi_d2 = normal_cdf_hastings(d2_arr);

  const double term_squared = (c_xx_sqrt + c_yy_sqrt * ctx.corr_xy);
  const double half_term_sq = 0.5 * term_squared * term_squared;

  const auto part1 = Eigen::exp(ctx.m_arr + c_xx / 2.0) * Phi_d1;
  const double term_z = std::sqrt(c_yy * (1.0 - ctx.corr_xy * ctx.corr_xy)) * z;
  const auto part2 =
      Eigen::exp(ctx.m_arr + ctx.n_arr + term_z + half_term_sq) * Phi_d2;
  return part1 - part2;
}