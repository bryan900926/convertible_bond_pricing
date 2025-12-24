#include "Eigen/Dense"
#include <unsupported/Eigen/SpecialFunctions>

#include "..\Pricing\CbModel.h"
#include "EquityModel.h"

Eigen::ArrayXXd EquityFunVec(const CbParas &cb_paras, const EquityContext &ctx,
                             const Eigen::ArrayXd &v_t_arr) {
  // Constants
  const double SQRT_1_2 = 0.70710678118654752440;
  const double SQRT_2 = 1.41421356237309504880;
  const double INV_SQRT_PI = 0.564189583547756286948;

  // --- CASE 1: Non-Zero Correlation ---
  if (std::abs(ctx.corr_xy) > 1e-10) {
    GaussHermiteResult gh_result = compute_gauss_hermite_rule(cb_paras.qdt);

    const double c_xx = ctx.c_xx;
    const double c_xx_sqrt = std::sqrt(c_xx);
    const double c_yy_sqrt = std::sqrt(ctx.c_yy);

    const Eigen::ArrayXXd g_arr = -ctx.n_arr / c_yy_sqrt / ctx.corr_xy;
    const double h = std::sqrt(1.0 - ctx.corr_xy * ctx.corr_xy) / ctx.corr_xy;

    const Eigen::ArrayXXd exp_part1_base = Eigen::exp(ctx.m_arr + c_xx / 2.0);

    const double term_sq_val = c_xx_sqrt + c_yy_sqrt * ctx.corr_xy;
    const double part_2_fourth = 0.5 * term_sq_val * term_sq_val;

    Eigen::ArrayXXd integral_arr =
        Eigen::ArrayXXd::Zero(ctx.m_arr.rows(), ctx.n_arr.cols());
    Eigen::ArrayXXd d1_arr, d2_arr;

    for (size_t k = 0; k < cb_paras.qdt; ++k) {
      double z = gh_result.x(k) * SQRT_2;

      if (ctx.corr_xy > 0) {
        d1_arr = g_arr - c_xx_sqrt - h * z;
        d2_arr = g_arr - h * z - term_sq_val;
      } else {
        d1_arr = c_xx_sqrt - (g_arr - h * z);
        d2_arr = d1_arr + c_yy_sqrt * ctx.corr_xy;
      }

      const Eigen::ArrayXXd Phi_d1 = 0.5 * (1.0 + (d1_arr * SQRT_1_2).erf());
      const Eigen::ArrayXXd Phi_d2 = 0.5 * (1.0 + (d2_arr * SQRT_1_2).erf());

      const Eigen::ArrayXXd part1 = Phi_d1 * exp_part1_base;

      const Eigen::ArrayXXd part2 =
          Eigen::exp(ctx.n_arr + ctx.m_arr + c_yy_sqrt * z + part_2_fourth) *
          Phi_d2;

      integral_arr += gh_result.w(k) * (part1 - part2);
    }
    return (integral_arr * INV_SQRT_PI).colwise() * (ctx.pt_arr * v_t_arr);
  }
  // --- CASE 2: Zero Correlation ---
  else {
    Eigen::ArrayXXd term_A = ctx.m_arr + ctx.c_xx / 2.0;
    Eigen::ArrayXXd term_B = ctx.n_arr + ctx.c_yy / 2.0;

    Eigen::ArrayXXd first_arr = (1.0 - term_B.exp()) * (term_A.exp().colwise() *
                                                        (v_t_arr * ctx.pt_arr));

    Eigen::ArrayXXd x_arr =
        -ctx.n_arr / std::sqrt(ctx.c_yy) - std::sqrt(ctx.c_yy);
    Eigen::ArrayXXd second_arr = 0.5 * (1.0 + (x_arr * SQRT_1_2).erf());

    return first_arr * second_arr;
  }
}