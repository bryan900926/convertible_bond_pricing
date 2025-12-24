#include "Eigen/Dense"
#include <iostream>
#include <vector>

#include "..\Pricing\CbModel.h"
#include "EquityModel.h"

EquityTreeBuildResult EquityTreeBuild(const std::vector<LNode> &l_data,
                                      const HullWhiteTreeResult &tree_result,
                                      const CbParas &cb_paras,
                                      const CdgParas &cdg_paras,
                                      const std::vector<PNode> &next_p_data,
                                      const std::vector<MNode> &next_m_data,
                                      const VasciekParas &vasciek_paras,
                                      const CouponPaidInfo &coupon_info) {
  const Eigen::ArrayXd ratio_min_vec =
      Eigen::ArrayXd::LinSpaced(cb_paras.partition, 1.0, 0.0);
  const Eigen::ArrayXd ratio_max_vec =
      Eigen::ArrayXd::LinSpaced(cb_paras.partition, 0.0, 1.0);

  Eigen::ArrayX3i idx_vec = Eigen::ArrayX3i::Zero(l_data.size(), 3);
  Eigen::ArrayXi nxt_m(next_m_data.size());
  Eigen::Array<double, Eigen::Dynamic, 9> nxt_p(next_p_data.size(), 9);

  const double x0 = std::log(cdg_paras.V0);
  const double y0 = std::log(cdg_paras.V0);
  const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;
  const double sigma_x =
      cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
  const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
  const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);

  Eigen::ArrayXXd l_data_partition =
      Eigen::ArrayXXd::Zero(l_data.size(), cb_paras.partition);
  Eigen::ArrayXd r_data_partition = Eigen::ArrayXd::Zero(l_data.size());
  Eigen::ArrayXd theta_data_partition = Eigen::ArrayXd::Zero(l_data.size());
  Eigen::ArrayXd theta1_data_partition = Eigen::ArrayXd::Zero(l_data.size());
  Eigen::ArrayXd v_data = Eigen::ArrayXd::Zero(l_data.size());

  // other steps
  for (int h = 0; h < l_data.size(); ++h) {
    idx_vec(h, 0) = l_data[h].step - 1;
    idx_vec(h, 1) = l_data[h].m;
    idx_vec(h, 2) = l_data[h].k;

    const size_t t = (l_data[h].step > 2) ? l_data[h].step - 2 : 0;
    const Eigen::ArrayXd l_vec =
        l_data[h].l_min * ratio_min_vec + l_data[h].l_max * ratio_max_vec;

    const double r_t = tree_result.short_rate_tree(
        static_cast<int>(l_data[h].k), static_cast<int>(l_data[h].step) - 1);
    const double x_t = x0 + l_data[h].m * jump_other;
    const double v_t =
        std::exp(y0 + (x_t - x0) +
                 cdg_paras.sigma_v * cb_paras.rho * (r_t - vasciek_paras.r0) /
                     vasciek_paras.sigma_r);

    l_data_partition.row(h).segment(0, cb_paras.partition) = l_vec.transpose();

    r_data_partition(h) = r_t;
    theta_data_partition(h) = thetas(t);
    theta1_data_partition(h) = thetas(t + 1);
    v_data(h) = v_t;

    if (h < next_m_data.size()) {
      nxt_m(h, 0) = next_m_data[h].nxt_m;
      for (int j = 0; j < 9; ++j) {
        nxt_p(h, j) = next_p_data[h].prob_matrix[j / 3][j % 3];
      }
    }
  }
  const EquityContext ctx = EquityContextVec(
      cb_paras.dt_other, l_data_partition, r_data_partition, theta_data_partition * vasciek_paras.kappa,
      theta1_data_partition * vasciek_paras.kappa, cb_paras, cdg_paras, vasciek_paras);

  Eigen::ArrayXXd equity_tree =
      EquityFunVec(cb_paras, ctx, v_data) / cb_paras.NS;

  // first step
  {
    Eigen::ArrayXd r_data_first = Eigen::ArrayXd::Zero(1);
    Eigen::ArrayXXd l_data_first = Eigen::ArrayXXd::Zero(1, cb_paras.partition);
    Eigen::ArrayXd theta_data_first = Eigen::ArrayXd::Zero(1);
    Eigen::ArrayXd theta1_data_first = Eigen::ArrayXd::Zero(1);
    Eigen::ArrayXd v_data_first = Eigen::ArrayXd::Zero(1);

    const Eigen::ArrayXd l_vec =
        l_data[0].l_min * ratio_min_vec + l_data[0].l_max * ratio_max_vec;

    const double r_t = tree_result.short_rate_tree(
        static_cast<int>(l_data[0].k), static_cast<int>(l_data[0].step) - 1);
    const double x_t = x0 + l_data[0].m * jump_first;
    const double v_t =
        std::exp(y0 + (x_t - x0) +
                 cdg_paras.sigma_v * cb_paras.rho * (r_t - vasciek_paras.r0) /
                     vasciek_paras.sigma_r);

    l_data_first.row(0).segment(0, cb_paras.partition) = l_vec.transpose();

    r_data_first(0) = r_t;
    theta_data_first(0) = 0;
    theta1_data_first(0) = 0;
    v_data_first(0) = v_t;
    equity_tree(0, Eigen::all) =
        EquityFunVec(cb_paras,
                     EquityContextVec(coupon_info.dt_first, l_data_first, r_data_first,
                                      theta_data_first, theta1_data_first,
                                      cb_paras, cdg_paras, vasciek_paras),
                     v_data_first) /
        cb_paras.NS;
  }

  const auto bankruptcy_check = (l_data_partition >= 0.0);
  equity_tree = bankruptcy_check.select(0.0, equity_tree);
  return {equity_tree, idx_vec, nxt_m, nxt_p, l_data_partition};
}