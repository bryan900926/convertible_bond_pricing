#include "Eigen/Dense"
#include <unsupported/Eigen/SpecialFunctions>
#include <omp.h>

#include "..\Pricing\CbModel.h"
#include "EquityModel.h"

Eigen::ArrayXXd EquityFunVec(const CbParas &cb_paras, const EquityContext &ctx,
                             const Eigen::ArrayXd &v_t_arr)
{
  // Constants
  const double SQRT_1_2 = 0.70710678118654752440;
  const double SQRT_2 = 1.41421356237309504880;
  const double INV_SQRT_PI = 0.564189583547756286948;

  // --- CASE 1: Non-Zero Correlation ---
  if (std::abs(ctx.corr_xy) > 1e-5)
  {
    GaussHermiteResult gh_result = compute_gauss_hermite_rule(cb_paras.qdt);

    const double c_xx = ctx.c_xx;
    const double c_xx_sqrt = std::sqrt(c_xx);
    const double c_yy_sqrt = std::sqrt(ctx.c_yy);
    
    const Eigen::ArrayXXd g_arr = -ctx.n_arr / (c_yy_sqrt * ctx.corr_xy);
    const double h = std::sqrt(1.0 - ctx.corr_xy * ctx.corr_xy) / ctx.corr_xy;

    const Eigen::ArrayXXd exp_part1_base = Eigen::exp(ctx.m_arr + c_xx / 2.0);

    const double term_sq_val = c_xx_sqrt + c_yy_sqrt * ctx.corr_xy;
    const double part_2_fourth = 0.5 * term_sq_val * term_sq_val;

    const Eigen::ArrayXXd exp_part2_base =
        (ctx.n_arr + ctx.m_arr + part_2_fourth).exp();

    Eigen::ArrayXXd integral_arr = Eigen::ArrayXXd::Zero(ctx.m_arr.rows(), ctx.n_arr.cols());
    Eigen::ArrayXXd d1_arr(ctx.m_arr.rows(), ctx.n_arr.cols());
    Eigen::ArrayXXd d2_arr(ctx.m_arr.rows(), ctx.n_arr.cols());

    Eigen::ArrayXXd d1_static, d2_static;
    double d1_sign_h, d2_sign_h; // Multipliers for z

    if (ctx.corr_xy > 0)
    {
      d1_static = g_arr - c_xx_sqrt;
      d2_static = g_arr - term_sq_val; // Derived from original d2 eq
      d1_sign_h = -h;
      d2_sign_h = -h;
    }
    else
    {
      d1_static = c_xx_sqrt - g_arr;
      d2_static = d1_static + c_yy_sqrt * ctx.corr_xy;
      d1_sign_h = h; // Note the sign flip from logic: c_xx_sqrt - (g - h*z) = (c-g) + h*z
      d2_sign_h = h;
    }

    for (size_t k = 0; k < cb_paras.qdt; ++k)
    {
      const double z = gh_result.x(k) * SQRT_2;
      const double w = gh_result.w(k);

      d1_arr = d1_static + (d1_sign_h * z);
      d2_arr = d2_static + (d2_sign_h * z); // Optimized derivation

      d1_arr = (1.0 + (-1.702 * d1_arr).exp()).inverse();
      d2_arr = (1.0 + (-1.702 * d2_arr).exp()).inverse();
      double scalar_exp_z = std::exp(c_yy_sqrt * z); // Scalar exponent

      integral_arr += w * ((d1_arr * exp_part1_base) -
                           (d2_arr * (exp_part2_base * scalar_exp_z)));
    }
    return (integral_arr * INV_SQRT_PI).colwise() * (ctx.pt_arr * v_t_arr);
  }
  // --- CASE 2: Zero Correlation ---
  else
  {
    Eigen::ArrayXXd term_A = ctx.m_arr + ctx.c_xx / 2.0;
    Eigen::ArrayXXd term_B = ctx.n_arr + ctx.c_yy / 2.0;

    Eigen::ArrayXXd first_arr = (1.0 - term_B.exp()) * (term_A.exp().colwise() *
                                                        (v_t_arr * ctx.pt_arr));

    const double safe_c_yy_sqrt = std::max(std::sqrt(ctx.c_yy), 1e-12);
    Eigen::ArrayXXd x_arr =
        -ctx.n_arr / safe_c_yy_sqrt - safe_c_yy_sqrt;
    Eigen::ArrayXXd second_arr = (1.0 + (-1.702 * x_arr).exp()).inverse();    

    return first_arr * second_arr;
  }
}
