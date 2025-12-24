#include <Eigen/Dense>
#include <cmath>
#include <utility>


#include "HullWhiteModel.h"

HullWhiteAlphaResult
HullWhiteTreeAlpha(int n, int non_first, int non_other, double dt_first,
                   double dt_other, double sigma_r, double kappa,
                   const Eigen::ArrayXd &observed_zero_rates, int jmax_first,
                   int jmax_other) {
  const Eigen::ArrayX3d prob_first =
      ProbCalc(non_first, jmax_first, dt_first, kappa);
  const Eigen::ArrayX3d prob_other =
      ProbCalc(non_other, jmax_other, dt_other, kappa);

  Eigen::ArrayXd alphas(n);
  const int q_rows = std::max(non_other, non_first);

  Eigen::ArrayXXd q_matrix = Eigen::ArrayXXd::Zero(q_rows, n + 2);
  Eigen::ArrayXd b_price_vec = Eigen::ArrayXXd::Zero(n + 2, 1);

  alphas(0, 0) = observed_zero_rates(0, 0);
  b_price_vec(0, 0) = 1.0;

  Eigen::ArrayXd dt_vec(n);
  dt_vec.fill(dt_other);
  dt_vec(0) = dt_first;

  Eigen::ArrayXd t_grid(n + 1);
  t_grid(0) = 0.0;
  for (int i = 1; i <= n; ++i) {
    t_grid(i) = t_grid(i - 1) + dt_vec(i - 1);
  }

  for (int i = 1; i <= n; ++i) {
    b_price_vec(i) = std::exp(-observed_zero_rates(i - 1) * t_grid(i));
  }

  const int center = jmax_first;
  const double exp_neg_alpha_dt = std::exp(-alphas(0) * dt_vec(0));
  q_matrix(center + 1, 1) = 1.0 / 6.0 * exp_neg_alpha_dt;
  q_matrix(center, 1) = 2.0 / 3.0 * exp_neg_alpha_dt;
  q_matrix(center - 1, 1) = 1.0 / 6.0 * exp_neg_alpha_dt;

  Eigen::ArrayXd temp_qsum(q_rows);
  Eigen::ArrayXd j_vec_active, q_slice, discount_factors, q_moving;

  const double dr_first = sigma_r * std::sqrt(3.0 * dt_first);
  const double dr_other = sigma_r * std::sqrt(3.0 * dt_other);

  for (int i = 1; i <= n - 1; ++i) {
    temp_qsum.setZero();
    const double dt = dt_vec(i - 1);
    const double dr = (i == 1) ? dr_first : dr_other;

    const Eigen::ArrayX3d &prob = (i == 1) ? prob_first : prob_other;
    const int jmax_local = (i == 1) ? jmax_first : jmax_other;

    const int j_lo = std::max(-jmax_local, -i);
    const int j_hi = std::min(jmax_local, i);

    double temp_sum = 0;
    for (int j = j_lo; j <= j_hi; ++j) {
      int row_idx = j + jmax_local;
      temp_sum += q_matrix(row_idx, i) * std::exp(-j * dr * dt);
    }

    if (b_price_vec(i + 1) <= 0 || temp_sum <= 0) {
      alphas(i) = 0.0;
    } else {
      alphas(i) = std::log(temp_sum / b_price_vec(i + 1)) / dt;
    }

    for (int j = j_lo; j <= j_hi; ++j) {
      const int row_from = j + jmax_local;
      const double exp_neg_alpha_j_dr_dt = std::exp(-(alphas(i) + j * dr) * dt);
      for (int k = 1; k >= -1; --k) {
        int row_to = j + k + jmax_local;
        if (row_to < 0) {
          row_to = 0;
        } else if (row_to >= q_rows) {
          row_to = q_rows - 1;
        }
        double probkj = prob(row_from, 1 - k);
        temp_qsum(row_to) +=
            q_matrix(row_from, i) * probkj * exp_neg_alpha_j_dr_dt;
      }

      if (i >= jmax_local) {

        for (int k = j + 1; k >= j - 1; --k) {
          const int row_to = k + jmax_local;
          if (row_to >= 0 && row_to < q_rows) {
            q_matrix(row_to, i + 1) = temp_qsum(row_to);
          }
        }
      }
    }
    if (i < jmax_local) {
      for (int k = i + 1; k >= -i - 1; --k) {
        const int row_to = k + jmax_local;
        q_matrix(row_to, i + 1) = temp_qsum(row_to);
      }
    }
  }

  Eigen::ArrayX3d a_first = prob_first.colwise().reverse();
  Eigen::ArrayX3d a_other = prob_other.colwise().reverse();
  const Eigen::ArrayX3d prob_first_final =
      ProbCalcV2(non_first, jmax_first, n, a_first);
  const Eigen::ArrayX3d prob_other_final =
      ProbCalcV2(non_other, jmax_other, n, a_other);

  Eigen::ArrayXd thetas = Eigen::ArrayXd::Zero(n);
  if (n == 1) {
    thetas(0) = -kappa * alphas(0);
  } else {
    thetas(0) = (alphas(1) - alphas(0)) / dt_vec(0) - kappa * alphas(0);
    for (int i = 1; i < n - 1; ++i) {
      const double d_alpha_dt =
          (alphas(i + 1) - alphas(i - 1)) / (dt_vec(i - 1) + dt_vec(i));
      thetas(i) = d_alpha_dt - kappa * alphas(i);
    }
    thetas(n - 1) =
        (alphas(n - 1) - alphas(n - 2)) / dt_vec(n - 2) - kappa * alphas(n - 1);
  }
  return {std::move(prob_first_final), std::move(prob_other_final),
          std::move(alphas), std::move(thetas)};
}
