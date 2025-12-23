#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "..\Equity\EquityModel.h"
#include "CbModel.h"
#include "Eigen/src/Core/Array.h"
#include "Eigen/src/Core/util/Constants.h"

void CbTreeBuild(const CbParas &cb_paras, const CdgParas &cdg_paras,
                 const VasciekParas &vasciek_paras,
                 const EquityTreeBuildResult &equity_tree_result,
                 const HullWhiteTreeResult &tree_result,
                 const CouponPaidInfo &coupon_info,
                 const PzTreeResult &pz_result,
                 const std::vector<int> &num_node_steps) {

  const Eigen::ArrayXXd &equity_tree = equity_tree_result.equity_tree;
  const Eigen::ArrayX3i &idx_vec =
      equity_tree_result.idx_vec; // [[step,m,k],...]
  const Eigen::ArrayXi &nxt_m = equity_tree_result.nxt_m;
  const Eigen::Array<double, Eigen::Dynamic, 9> &nxt_p =
      equity_tree_result.nxt_p;
  const Eigen::ArrayXXd &l_data = equity_tree_result.l_data_partition;
  const Eigen::ArrayX3i &nxt_r_idx = pz_result.nxt_r_idx;

  const double sigma_x =
      cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
  const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first),
               jump_other = sigma_x * std::sqrt(cb_paras.dt_other);

  const double NaN = std::numeric_limits<double>::quiet_NaN();
  const int n = coupon_info.total_steps;
  const double y0 = cdg_paras.V0, x0 = cdg_paras.V0;

  std::vector<int> cum_node_steps(num_node_steps.size(), 0);
  cum_node_steps[0] = num_node_steps[0];

  for (size_t i = 1; i < num_node_steps.size(); ++i) {
    cum_node_steps[i] =
        cum_node_steps[i - 1] + num_node_steps[i]; // [1, 3, 7,...]
  }

  const int rows = (n + 1) * 4 + 2;
  const int cols = tree_result.short_rate_tree.rows();
  const int m_idx_offset = n * 2;

  Eigen::ArrayXi map_now = Eigen::ArrayXi::Constant(rows * cols, -1);
  Eigen::ArrayXi map_next = Eigen::ArrayXi::Constant(rows * cols, -1);

  auto fill_map = [&](Eigen::ArrayXi &map, const int step) {
    const auto m_arr = idx_vec.col(1).middleRows(cum_node_steps[step - 1],
                                                 num_node_steps[step]);
    const auto k_arr = idx_vec.col(2).middleRows(cum_node_steps[step - 1],
                                                 num_node_steps[step]);

    const auto idxs = (m_arr + m_idx_offset) * cols + k_arr;
    const Eigen::ArrayXi target_idx =
        Eigen::ArrayXi::LinSpaced(m_arr.size(), 0, m_arr.size() - 1);
    map(idxs) = target_idx;
  };

  fill_map(map_next, n);

  Eigen::ArrayXXd l_next = l_data.middleRows(
      cum_node_steps[cum_node_steps.size() - 2], num_node_steps[n]);
  Eigen::ArrayXXd s_next = equity_tree.middleRows(
      cum_node_steps[cum_node_steps.size() - 2], num_node_steps[n]);

  Eigen::ArrayXXd b_next =
      Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
  Eigen::ArrayXXd cb_next =
      Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
  Eigen::ArrayXXd dil_s_next =
      Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
  Eigen::ArrayXXd equity_next = s_next;

  const auto is_nan = l_next.isNaN();
  const auto is_pos = l_next >= 0.0;
  const auto is_neg = !is_pos;
  const auto is_valid = !(is_nan || is_pos);
  const auto low_f = cb_paras.coupon_rate * s_next < cb_paras.F;
  const auto no_convert = is_valid && low_f;
  const auto convert = is_valid && !low_f;

  dil_s_next = is_nan.select(NaN, dil_s_next);
  b_next = is_pos.select(cb_paras.F * cb_paras.rr, b_next);
  cb_next = is_pos.select(cb_paras.F * cb_paras.rr * -l_next.exp(), cb_next);
  cb_next = no_convert.select(cb_paras.F, cb_next);
  dil_s_next = no_convert.select(s_next, dil_s_next);

  const Eigen::ArrayXXd conv_val =
      (s_next * cb_paras.NS + cb_paras.NC * cb_paras.F) /
      (cb_paras.CR * cb_paras.NC + cb_paras.NS);
  dil_s_next = convert.select(s_next.min(conv_val), dil_s_next);
  cb_next = convert.select((dil_s_next * cb_paras.CR).max(cb_paras.F), cb_next);
  b_next = is_neg.select(cb_paras.F * (1 + cb_paras.coupon_rate), b_next);

  Eigen::ArrayXd dm_vec(9);

  dm_vec << -2, -2, -2, 0, 0, 0, 2, 2, 2;

  auto do_interp = [](const Eigen::ArrayXd &arr1, const Eigen::ArrayXd &arr2,
                      const Eigen::ArrayXd &weights) {
    return arr1 + (arr2 - arr1) * weights;
  };

  const double l_hat_first =
      (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
          cdg_paras.lamda -
      cdg_paras.v + cdg_paras.phi * vasciek_paras.r_bar;

  for (size_t i = n; i >= 1; --i) {

    const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
    const double jump = (i == 1) ? jump_first : jump_other;
    const int idx_start = (i > 1) ? cum_node_steps[i - 2] : 0;
    const int num_nodes = num_node_steps[i - 1];

    const Eigen::ArrayXi h_range = Eigen::ArrayXi::LinSpaced(
        num_nodes, idx_start, idx_start + num_nodes - 1);
    const auto &l_now = l_data.middleRows(idx_start, num_nodes);
    const auto &s_now = equity_tree.middleRows(idx_start, num_nodes);

    Eigen::ArrayXXd b_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd cb_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd dil_s_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd equity_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);

    Eigen::ArrayXd m_now_arr =
        idx_vec.col(1).middleRows(idx_start, num_nodes).cast<double>();
    Eigen::ArrayXi k_now_arr = idx_vec.col(2).middleRows(idx_start, num_nodes);
    Eigen::ArrayXd nxt_m_arr =
        nxt_m.col(0).middleRows(idx_start, num_nodes).cast<double>();

    Eigen::ArrayXd r_now_arr =
        tree_result.short_rate_tree.col(i - 1)(k_now_arr);
    Eigen::ArrayXd x_now_arr = x0 + m_now_arr * jump;
    Eigen::ArrayXd y_now_arr = y0 + (x_now_arr - x0) +
                               cdg_paras.sigma_v * cb_paras.rho *
                                   (r_now_arr - vasciek_paras.r0) /
                                   vasciek_paras.sigma_r;
    Eigen::ArrayXd l_hat_arr =
        l_hat_first - r_now_arr * (1 / cdg_paras.lamda + cdg_paras.phi);
    Eigen::ArrayXd miu_y_arr =
        r_now_arr -
        (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2);

    if (i > 1) {
      fill_map(map_now, i - 1);
    }
    Eigen::ArrayXXi k_next_arr =
        nxt_r_idx(k_now_arr, Eigen::all).replicate(1, 3); // h x 9
    Eigen::ArrayXi flat_k_next = k_next_arr.reshaped<Eigen::RowMajor>();
    Eigen::ArrayXd next_rates_col = tree_result.short_rate_tree.col(i - 1);
    Eigen::ArrayXd flat_rates = next_rates_col(flat_k_next);
    Eigen::ArrayXXd r_next_vals =
        flat_rates.reshaped(k_next_arr.rows(), k_next_arr.cols());
    Eigen::ArrayXXd m_next_arr =
        nxt_m_arr.replicate(1, 9).rowwise() + dm_vec.transpose(); // h x 9
    Eigen::ArrayXi flat_m_next =
        m_next_arr.reshaped<Eigen::RowMajor>().cast<int>();
    Eigen::ArrayXXd y_next_arr = y0 + (x0 + m_next_arr * jump - x0) +
                                 cdg_paras.sigma_v * cb_paras.rho *
                                     (r_next_vals - vasciek_paras.r0) /
                                     vasciek_paras.sigma_r; // h x 9
                                                            //
    const Eigen::ArrayXi nxt_idx =
        (flat_m_next + m_idx_offset) * cols + flat_k_next; // h x 9

    Eigen::ArrayXXd simga_term =
        y_next_arr.colwise() - (y_now_arr - miu_y_arr * dt); // h x 9

    int bool_flag = (i > 1) && coupon_info.is_coupon_paid[i - 1];
    Eigen::ArrayXd discount_factor = (-r_now_arr * dt).exp();

    for (size_t p = 0; p < cb_paras.partition; ++p) {
      const Eigen::ArrayXd l_now_vec = l_now.col(p); // h x 1
      const Eigen::ArrayXd s_now_vec = s_now.col(p); // h x 1

      const auto is_nan = l_now_vec.isNaN();
      const auto is_pos = l_now_vec >= 0.0;
      const auto no_convert =
          !is_nan || !is_pos && (cb_paras.CR * s_now_vec < cb_paras.F);
      const auto convert = !(is_nan || is_pos || no_convert);

      dil_s_now.col(p) = no_convert.select(s_now_vec, dil_s_now.col(p));
      dil_s_now.col(p) = convert.select(
          s_now_vec.min((s_now_vec * cb_paras.NS + cb_paras.NC * cb_paras.F) /
                        (cb_paras.CR * cb_paras.NC + cb_paras.NS)),
          dil_s_now.col(p));

      b_now.col(p) = is_pos.select(cb_paras.F * cb_paras.rr, b_now.col(p));
      dil_s_now.col(p) = is_pos.select(0, dil_s_now.col(p));
      equity_now.col(p) = is_pos.select(0, equity_now.col(p));
      cb_now.col(p) = is_pos.select(cb_paras.F * cb_paras.rr * -l_now_vec.exp(),
                                    cb_now.col(p));

      const auto interp = !(is_nan || is_pos); // num_interp x 1
      const int num_interp = interp.count();

      if (num_interp == 0)
        continue;

      Eigen::ArrayXi valid_idx = find_indices(interp);

      Eigen::ArrayXd l_curr = l_now_vec(valid_idx);     // num_interp x 1
      Eigen::ArrayXd l_hat_curr = l_hat_arr(valid_idx); // num_interp x 1
      Eigen::ArrayXd df = discount_factor(valid_idx);   // num_interp x 1
      Eigen::ArrayXd l_curr_to_next =
          ((-simga_term(interp, Eigen::all)).colwise() +
           (l_curr + cdg_paras.lamda * (l_hat_curr - l_curr) * dt))
              .reshaped<Eigen::RowMajor>(); // (num_interp x 9) x 1

      Eigen::ArrayXi flat_idx = map_next(nxt_idx); // num_interp x 9
      Eigen::ArrayXXd l_sub =
          l_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd b_sub =
          b_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd cb_sub =
          cb_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd equity_sub =
          equity_next(flat_idx, Eigen::all); // (num_interp x 9) x partition

      Eigen::ArrayXd l_first = l_sub.col(0);               // num_interp x 1
      Eigen::ArrayXd l_last = l_sub.col(l_sub.cols() - 1); // num_interp x 1

      const auto mask_flat =
          (l_first - l_last).abs() < 1e-8;             //(num_interp x 9) x 1
      const auto mask_low = l_curr_to_next <= l_first; // (num_interp x 9) x 1
      const auto mask_high = l_curr_to_next >= l_last; // (num_interp x 9) x 1

      auto temp = l_sub <= l_curr_to_next.replicate(1, l_sub.cols());
      Eigen::ArrayXi nxt_l_idx = temp.cast<int>().rowwise().sum().min(
          l_sub.cols() - 2); // num_interp x 1

      Eigen::ArrayXd l_1(l_sub.rows());
      Eigen::ArrayXd l_2(l_sub.rows());
      Eigen::ArrayXd cb_interp_1(l_sub.rows());
      Eigen::ArrayXd cb_interp_2(l_sub.rows());
      Eigen::ArrayXd b_interp_1(l_sub.rows());
      Eigen::ArrayXd b_interp_2(l_sub.rows());
      Eigen::ArrayXd equity_interp_1(l_sub.rows());
      Eigen::ArrayXd equity_interp_2(l_sub.rows());

      for (int i = 0; i < l_sub.rows(); ++i) {
        l_1(i) = l_sub(i, nxt_l_idx(i));
        l_2(i) = l_sub(i, nxt_l_idx(i) + 1);
        cb_interp_1(i) = cb_sub(i, nxt_l_idx(i));
        cb_interp_2(i) = cb_sub(i, nxt_l_idx(i) + 1);
        b_interp_1(i) = b_sub(i, nxt_l_idx(i));
        b_interp_2(i) = b_sub(i, nxt_l_idx(i) + 1);
        equity_interp_1(i) = equity_sub(i, nxt_l_idx(i));
        equity_interp_2(i) = equity_sub(i, nxt_l_idx(i) + 1);
      }
      Eigen::ArrayXd weight =
          (l_curr_to_next - l_1) / (l_2 - l_1); // (num_interp x 9) x 1
      weight = mask_flat.select(0, weight);

      auto b_next_flat = do_interp(b_interp_1, b_interp_2,
                                   weight); // (num_interp x 9) x 1

      auto cb_next_flat = do_interp(cb_interp_1, cb_interp_2,
                                    weight); // (num_interp x 9) x 1
      auto equity_next_flat = do_interp(equity_interp_1, equity_interp_2,
                                        weight); // (num_interp x 9) x 1

      Eigen::ArrayXXd b_reshape_next = b_next_flat.reshaped(num_interp, 9);
      Eigen::ArrayXXd cb_reshape_next = cb_next_flat.reshaped(num_interp, 9);
      Eigen::ArrayXXd equity_reshape_next =
          equity_next_flat.reshaped(num_interp, 9);
      Eigen::ArrayXXd next_p_subs =
          nxt_p(h_range(valid_idx).cast<int>(), Eigen::all);

      Eigen::ArrayXd b_res =
          (b_reshape_next * next_p_subs).rowwise().sum() * df;
      Eigen::ArrayXd cb_res =
          (cb_reshape_next * next_p_subs).rowwise().sum() * df;
      Eigen::ArrayXd equity_res =
          (equity_reshape_next * next_p_subs).rowwise().sum() * df;
      b_res += cb_paras.coupon_rate * cb_paras.F * bool_flag;
      cb_res += cb_paras.coupon_rate * cb_paras.F * bool_flag;
      cb_res =
          cb_res.min(cb_paras.CP).max(cb_paras.CR * dil_s_now(valid_idx, p));
      b_now(valid_idx, p) = b_res;
      cb_now(valid_idx, p) = cb_res;
      equity_now(valid_idx, p) = equity_res;
    }
    std::swap(b_now, b_next);
    std::swap(cb_now, cb_next);
    std::swap(equity_now, equity_next);
    std::swap(map_now, map_next);

    b_now.setZero();
    cb_now.setZero();
    equity_now.setZero();
    map_now.setConstant(-1);
  }
  std::cout << cb_next(0, 0) << std::endl;
}