// #define DEBUG
#include "..\HullWhiteModel\HullWhiteModel.h"
#include "..\Util\Util.h"
#include "..\Util\TreeManager.hpp"
#include "CbModel.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <thread>


FinalResultMemoSave CbTreePricingMemoSave(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas& vasciek_paras, const std::string& ticker)
{
  CouponPaidInfo coupon_info =
      CouponPaidCalc(cb_paras.T, cb_paras.dt_other, cb_paras.paid_cycle);

  int n = coupon_info.total_steps;

  double t_end = coupon_info.dt_first + n * cb_paras.dt_other;

  const Eigen::ArrayXd time_grid =
      Eigen::ArrayXd::LinSpaced(n + 1, coupon_info.dt_first, t_end);

  Eigen::ArrayXd zero_rates = VasciekZeroRates(vasciek_paras, time_grid);

  HullWhiteTreeResult tree_result =
      HullWhiteTree(vasciek_paras.kappa, vasciek_paras.sigma_r, zero_rates,
                    coupon_info.dt_first, cb_paras.dt_other);

  PzTreeResult pz_result = PzTreeBuild(coupon_info.total_steps, tree_result);

  const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;
  const double y0 = std::log(cdg_paras.V0);
  const double x0 = std::log(cdg_paras.V0);

  const double sigma_x =
      cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
  const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
  const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);
  if (cb_paras.if_const_r) {
    tree_result.short_rate_tree = Eigen::ArrayXXd::Constant(
        tree_result.short_rate_tree.rows(), tree_result.short_rate_tree.cols(),
        0);
  }

  std::vector<PackedNode> data_cur;
  std::vector<PackedNode> data_next;

  data_cur.emplace_back(PackedNode{1, static_cast<size_t>(pz_result.start_h), 0, cdg_paras.l0,
                            cdg_paras.l0, 0, 0, 0, 0});

  const std::array<int, 9> dm_vec = {-2, 0, 2, -2, 0, 2, -2, 0, 2};

  const double NaN = std::numeric_limits<double>::quiet_NaN();

  const Eigen::ArrayXd range_vec =
      Eigen::ArrayXd::LinSpaced(cb_paras.partition, 1, cb_paras.partition);
  const Eigen::ArrayXd ratio_min =
      (cb_paras.partition - range_vec) / (cb_paras.partition - 1);
  const Eigen::ArrayXd ratio_max = (range_vec - 1) / (cb_paras.partition - 1);

  Eigen::ArrayXd scratch_root(cb_paras.partition);
  Eigen::ArrayXd scratch_next(cb_paras.partition);

  const double miu_y_second =
      cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2;
  int m_idx_offset = 0;

  TreeManager tree_manager("./temp_data/" + ticker + ".bin", 10.0); // Set max size to 10 GB for testing

  const int max_k = tree_result.short_rate_tree.rows();

  // 1. Calculate the Global Maximum Bound for m
  int global_max_m = 0;
  int current_max_m = 0;
  for (int i = 1; i <= n; ++i)
  {
    double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
    double jump = (i == 1) ? jump_first : jump_other;
    double r_min = tree_result.short_rate_tree.col(i - 1).minCoeff();
    double r_max = tree_result.short_rate_tree.col(i - 1).maxCoeff();

    auto calc_nxt_m = [&](double r_val, int m_val)
    {
      double miu_y = r_val - miu_y_second;
      double miu_x = miu_y - cb_paras.sigma_V * cb_paras.rho *
                                 (vasciek_paras.kappa * (thetas(i - 1) - r_val)) /
                                 vasciek_paras.sigma_r;
      double x_val = x0 + m_val * jump;
      return static_cast<int>(std::round(((x_val + miu_x * dt) - x0) / jump));
    };

    current_max_m = std::max(
        std::abs(calc_nxt_m(r_max, current_max_m) + 2),
        std::abs(calc_nxt_m(r_min, -current_max_m) - 2));
    global_max_m = std::max(global_max_m, current_max_m);
  }

  // Add a small safety buffer just in case of floating-point rounding quirks
  global_max_m += 5;

  // 2. Allocate the Global Lookup Matrix ONCE (Flattened 2D: [m][k])
  // Size is (2 * max_m + 1) rows by (max_k) columns.
  std::vector<int> lookup_matrix((global_max_m * 2 + 1) * max_k, -1);

  for (int i = 1; i <= n; ++i)
  {
    const double l_hat_first =
      (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
          cdg_paras.lamda -
      cdg_paras.v + cdg_paras.phi * thetas(i - 1);
    double jump = (i == 1) ? jump_first : jump_other;
    double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
    const Eigen::ArrayX3d &prob_hw = (i == 1)
                                         ? tree_result.alpha_result.prob_first
                                         : tree_result.alpha_result.prob_other;

    const double S1 = sigma_x * sigma_x * dt;

    for (int count = 0; count < data_cur.size(); ++count) {
      PackedNode& node = data_cur[count];
      const int m_now = node.m;
      const size_t k_now = node.k;
      const double r_now = tree_result.short_rate_tree(k_now, i - 1);
      const double l_min_now = node.l_min;
      const double l_max_now = node.l_max;
      const double l_hat =
          l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);
      const double p_u_hw = prob_hw(k_now, 0);
      const double p_m_hw = prob_hw(k_now, 1);
      const double p_d_hw = prob_hw(k_now, 2);
      double hw_sum = p_u_hw + p_m_hw + p_d_hw;
      const double x_now = x0 + m_now * jump;
      const double y_now = y0 + (x_now - x0) +
                           cdg_paras.sigma_v * cb_paras.rho *
                               (r_now - vasciek_paras.r0) /
                               vasciek_paras.sigma_r;
      const double miu_y = r_now - miu_y_second;
      const double miu_x =
          miu_y - cb_paras.sigma_V * cb_paras.rho *
                      (vasciek_paras.kappa * (thetas(i - 1) - r_now)) /
                      vasciek_paras.sigma_r;
      const double expect_x = x_now + miu_x * dt;
      const int nxt_m = std::round((expect_x - x0) / jump);
      const double b = expect_x - x0 - nxt_m * jump;
      const double a = b + 2 * jump;
      const double c = b - 2 * jump;
      const double D = (a - b) * (a - c) * (b - c);
      const double p_d = (S1 * (b - c) + c * b * (b - c)) / D;
      const double p_m =
          (S1 * (c - a) + a * c * (c - a)) / ((b - c) * (b - a) * (c - a));
      const double p_u = 1 - p_d - p_m;
      node.nxt_middle_m = nxt_m;
      node.p_x_up= p_u;
      node.p_x_mid = p_m;
      node.p_x_down = p_d;
      for (int j = 0; j < 9; ++j)
      {
        const int m_next = nxt_m + dm_vec[j];
        m_idx_offset = std::max(m_idx_offset, std::abs(m_next));
        const size_t k_next = pz_result.nxt_r_idx(k_now, j / 3);
        if (l_min_now > 0)
        {
          continue;
        }

        const double x_nxt = x0 + m_next * jump;
        const double y_nxt =
            y0 + (x_nxt - x0) +
            cdg_paras.sigma_v * cb_paras.rho *
                (tree_result.short_rate_tree(k_next, i) - vasciek_paras.r0) /
                vasciek_paras.sigma_r;
        const double scalar_offset = (y_nxt - y_now - miu_y * dt);
        scratch_root = l_min_now * ratio_min + l_max_now * ratio_max;
        scratch_next = scratch_root +
                       cdg_paras.lamda * (l_hat - scratch_root) * dt -
                       scalar_offset;
        const double l_next_min = scratch_next.minCoeff();
        const double l_next_max = scratch_next.maxCoeff();

        const int map_idx = (m_next + global_max_m) * max_k + k_next;
        int existing_idx = lookup_matrix[map_idx];
        if (existing_idx == -1) {
          PackedNode new_node = {i + 1, k_next, m_next, l_next_min, l_next_max, 0, 0, 0, 0};
          data_next.emplace_back(new_node);
          lookup_matrix[map_idx] = data_next.size() - 1;
        }
        else
        {
          PackedNode &exist_node = data_next[existing_idx];
          exist_node.l_min = std::min(exist_node.l_min, l_next_min);
          exist_node.l_max = std::max(exist_node.l_max, l_next_max);
        }
      }
    }
    for (const auto &next_node : data_next)
    {
      const int reset_idx = (next_node.m + global_max_m) * max_k + next_node.k;
      lookup_matrix[reset_idx] = -1;
    }
    tree_manager.append_tree(std::move(data_cur));
    std::printf("Completed forward iteration %d, generated %zu nodes, m_idx_offset=%d\n", i, data_next.size(), m_idx_offset);
    std::swap(data_cur, data_next);
    data_next.clear();
  }
  tree_manager.append_tree(std::move(data_cur));
  return CbTreeBuildMemoSave(cb_paras, cdg_paras, vasciek_paras, tree_result, coupon_info, pz_result, m_idx_offset, tree_manager);
}