#include "Eigen/Dense"


#include "..\HullWhiteModel\HullWhiteModel.h"
#include "..\Pricing\CbModel.h"
#include "EquityModel.h"


EquityContext EquityContextVec(const double dt, const Eigen::ArrayXXd &l_t_arr,
                               const Eigen::ArrayXd &r_t_arr,
                               const Eigen::ArrayXd &theta_t_arr,
                               const Eigen::ArrayXd &theta_t1_arr,
                               const CbParas &cb_paras,
                               const CdgParas &cdg_paras,
                               const VasciekParas &vp) {
  const double B_kappa_dt = Calc_Bk(vp.kappa, dt);
  const double B_2kappa_dt = Calc_Bk(2.0 * vp.kappa, dt);

  const double B_lambda_dt = Calc_Bk(cdg_paras.lamda, dt); // B(ld)
  const double B_lambda_minus_kappa_dt =
      Calc_Bk(cdg_paras.lamda - vp.kappa, dt); // B(ld - K)
  const double B_lambda_plus_kappa_dt =
      Calc_Bk(cdg_paras.lamda + vp.kappa, dt);                    // B(ld + K)
  const double B_2lambda_dt = Calc_Bk(2.0 * cdg_paras.lamda, dt); // B(2ld)

  const double a =
      (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2.0) /
          cdg_paras.lamda -
      cdg_paras.v;
  const double b = -1.0 / cdg_paras.lamda - cdg_paras.phi;

  const Eigen::ArrayXd bond_log_arr =
      (vp.r_bar - vp.sigma_r * vp.sigma_r / (2.0 * vp.kappa * vp.kappa)) *
          (B_kappa_dt - dt) -
      (vp.sigma_r * vp.sigma_r / (4.0 * vp.kappa)) * (B_kappa_dt * B_kappa_dt) -
      B_kappa_dt * r_t_arr;

  const Eigen::ArrayXd m_t_arr =
      (vp.sigma_r * vp.sigma_r / (2.0 * vp.kappa * vp.kappa)) *
          (dt - 2.0 * B_kappa_dt + B_2kappa_dt) -
      bond_log_arr;

  const Eigen::ArrayXXd h_t_arr =
      (l_t_arr * std::exp(-cdg_paras.lamda * dt)).colwise() +
      ((cdg_paras.lamda * cdg_paras.phi / vp.kappa) *
           ((theta_t1_arr + std::exp(-cdg_paras.lamda * dt) * theta_t_arr) /
            2.0) *
           dt +
       a * (1.0 - std::exp(-cdg_paras.lamda * dt)) +
       cdg_paras.lamda * b *
           (std::exp(-cdg_paras.lamda * dt) * B_lambda_minus_kappa_dt *
                r_t_arr +
            std::exp(-vp.kappa * dt) * B_lambda_minus_kappa_dt * theta_t_arr *
                dt / 2.0));

  const Eigen::ArrayXXd m_arr =
      (m_t_arr - cdg_paras.delta * dt -
       cb_paras.sigma_V * cdg_paras.sigma_v / 2.0 * dt -
       vp.sigma_r * vp.sigma_r / (vp.kappa * vp.kappa) *
           (dt - 2 * B_kappa_dt + B_2kappa_dt) -
       cb_paras.sigma_V * vp.sigma_r * cb_paras.rho / vp.kappa *
           (dt - B_kappa_dt))
          .replicate(1, l_t_arr.cols());

  const Eigen::ArrayXXd n_arr =
      h_t_arr -
      (cdg_paras.lamda * b * vp.sigma_r * vp.sigma_r) /
          (vp.kappa * (cdg_paras.lamda - vp.kappa)) *
          (B_kappa_dt - B_lambda_dt - B_2kappa_dt + B_lambda_plus_kappa_dt) +
      cdg_paras.sigma_v * vp.sigma_r * cb_paras.rho / vp.kappa *
          (B_lambda_dt - B_lambda_plus_kappa_dt);

  const double c_xx = ((vp.sigma_r * vp.sigma_r) / (vp.kappa * vp.kappa)) *
                          (dt - 2.0 * B_kappa_dt + B_2kappa_dt) +
                      2 * cdg_paras.sigma_v * vp.sigma_r * cb_paras.rho /
                          vp.kappa * (dt - B_kappa_dt) +
                      cdg_paras.sigma_v * cdg_paras.sigma_v * dt;

  const double c_yy =
      std::pow(cdg_paras.lamda * b * vp.sigma_r, 2) /
          std::pow(cdg_paras.lamda - vp.kappa, 2) *
          (B_2kappa_dt - 2 * B_lambda_plus_kappa_dt + B_2lambda_dt) -
      2 * cdg_paras.lamda * b * vp.sigma_r * cdg_paras.sigma_v * cb_paras.rho /
          (cdg_paras.lamda - vp.kappa) *
          (B_lambda_plus_kappa_dt - B_2lambda_dt) +
      cdg_paras.sigma_v * cdg_paras.sigma_v * B_2lambda_dt;

  const double c_xy =
      cdg_paras.lamda * b * std::pow(vp.sigma_r, 2) /
          (vp.kappa * (cdg_paras.lamda - vp.kappa)) *
          (B_kappa_dt - B_lambda_dt - B_2kappa_dt + B_lambda_plus_kappa_dt) -
      cdg_paras.lamda * b * cdg_paras.sigma_v * cb_paras.rho /
          (cdg_paras.lamda - vp.kappa) *
          (B_lambda_plus_kappa_dt - B_2lambda_dt) -
      cdg_paras.sigma_v * vp.sigma_r * cb_paras.rho / vp.kappa *
          (B_lambda_dt - B_lambda_plus_kappa_dt) // Uses B(ld)
      + cdg_paras.lamda * b * cdg_paras.sigma_v * vp.sigma_r * cb_paras.rho /
            (cdg_paras.lamda - vp.kappa) *
            (B_kappa_dt - B_lambda_dt) // Uses B(ld), NOT B(ld-K)
      - cdg_paras.sigma_v * cdg_paras.sigma_v *
            B_lambda_dt; // Uses B(ld), NOT B(ld+K)

  if (c_yy < 0 || c_xx < 0)
    throw std::runtime_error("Negative variance");

  const double corr_xy = c_xy / std::sqrt(c_xx * c_yy);
  if (std::abs(corr_xy) > 1.0)
    throw std::runtime_error("Correlation out of bounds");

  const double b_k_square_integral =
      (2 * vp.kappa * dt - std::exp(-2 * vp.kappa * dt) +
       4 * std::exp(-vp.kappa * dt) - 3) /
      (2 * std::pow(vp.kappa, 3));

  const Eigen::ArrayXd pt_arr = Eigen::exp(
      -m_t_arr + 0.5 * vp.sigma_r * vp.sigma_r * b_k_square_integral);
  return EquityContext{c_xx,
                       c_yy,
                       c_xy,
                       corr_xy,
                       std::move(m_arr),
                       std::move(n_arr),
                       std::move(pt_arr)};
}