#include <Eigen/Dense>
#include <cmath>
#include <vector>


#include "../Equity/EquityModel.h"
#include "../Util/Timer.h"
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
  Timer time("Cb Tree Build Backward");
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

  const int m_idx_offset = idx_vec.col(1).abs().maxCoeff();
  const int rows = (m_idx_offset + 1) * 2;
  const int cols = tree_result.short_rate_tree.rows() + 1;

  Eigen::ArrayXi map_now = Eigen::ArrayXi::Constant(rows * cols, -1);
  Eigen::ArrayXi map_next = Eigen::ArrayXi::Constant(rows * cols, -1);

  for (int h = 0; h < num_node_steps[n]; ++h) {
    const int idx = cum_node_steps[n - 1] + h;
    const int m = idx_vec(idx, 1);
    const int k = idx_vec(idx, 2);

    const int map_idx = (m + m_idx_offset) * cols + k;
    map_next(map_idx) = h;
  }
  const double miu_y_second =
      cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2;

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
  const auto is_pos = l_next > 0.0;
  const auto is_neg = !is_pos;
  const auto is_valid = !is_nan && !is_pos;
  const auto low_f = (cb_paras.CR * s_next) < cb_paras.F;
  const auto no_convert = is_valid && low_f;
  const auto convert = is_valid && !no_convert;

  dil_s_next = is_nan.select(NaN, dil_s_next);
  b_next = is_pos.select(cb_paras.F * cb_paras.rr, b_next);
  cb_next = is_pos.select(cb_paras.F * cb_paras.rr * (-l_next).exp(), cb_next);
  cb_next = no_convert.select(cb_paras.F, cb_next);
  dil_s_next = no_convert.select(s_next, dil_s_next);

  const Eigen::ArrayXXd conv_val =
      (s_next * cb_paras.NS + cb_paras.NC * cb_paras.F) /
      (cb_paras.CR * cb_paras.NC + cb_paras.NS);
  dil_s_next = convert.select(s_next.min(conv_val), dil_s_next);
  cb_next = convert.select((dil_s_next * cb_paras.CR).max(cb_paras.F), cb_next);
  b_next = is_neg.select(cb_paras.F * (1 + cb_paras.coupon_rate), b_next);

  Eigen::ArrayXd dm_vec(9);
  dm_vec << 2, 2, 2, 0, 0, 0, -2, -2, -2;
  for (size_t i = n; i >= 1; --i) {
    const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
    const double jump = (i == 1) ? jump_first : jump_other;
    const int idx_start = (i > 1) ? cum_node_steps[i - 2] : 0;
    const int num_nodes = num_node_steps[i - 1];

    const auto l_now = l_data.middleRows(idx_start, num_nodes);
    const auto s_now = equity_tree.middleRows(idx_start, num_nodes);

    Eigen::ArrayXXd b_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd cb_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd dil_s_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);
    Eigen::ArrayXXd equity_now =
        Eigen::ArrayXXd::Zero(l_now.rows(), cb_paras.partition);

    int coupont_flag = (i > 1) && coupon_info.is_coupon_paid[i - 1];

    const double l_hat_first =
        (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 1) /
            cdg_paras.lamda -
        cdg_paras.v + cdg_paras.phi * tree_result.alpha_result.thetas(i - 1);

    for (size_t p = 0; p < cb_paras.partition; ++p) {
      const auto l_now_vec = l_now.col(p); // h x 1
      const auto s_now_vec = s_now.col(p); // h x 1

      const auto is_nan = l_now_vec.isNaN();
      const auto is_pos = l_now_vec > 0.0;
      const auto no_convert =
          !is_nan && !is_pos && (cb_paras.CR * s_now_vec < cb_paras.F);
      const auto convert = !(is_nan || is_pos || no_convert);

      dil_s_now.col(p) = no_convert.select(s_now_vec, dil_s_now.col(p));
      dil_s_now.col(p) = convert.select(
          s_now_vec.min((s_now_vec * cb_paras.NS + cb_paras.NC * cb_paras.F) /
                        (cb_paras.CR * cb_paras.NC + cb_paras.NS)),
          dil_s_now.col(p));

      b_now.col(p) = is_pos.select(cb_paras.F * cb_paras.rr, b_now.col(p));
      dil_s_now.col(p) = is_pos.select(0, dil_s_now.col(p));
      equity_now.col(p) = is_pos.select(0, equity_now.col(p));
      cb_now.col(p) = is_pos.select(
          cb_paras.F * cb_paras.rr * (-l_now_vec).exp(), cb_now.col(p));

      const auto interp = !(is_nan || is_pos); // num_interp x 1

      for (int k = 0; k < interp.size(); ++k) {

        int survival_flag = interp(k);

        const int l_idx = idx_start + k;
        const int m_now = idx_vec(l_idx, 1);
        const int k_now = idx_vec(l_idx, 2);
        const int next_m_middle = nxt_m(l_idx);

        map_now((m_now + m_idx_offset) * cols + k_now) = k;

        const double r_now = tree_result.short_rate_tree(k_now, i - 1);
        const double x_now = x0 + m_now * jump;
        const double y_now = y0 + (x_now - x0) +
                             cdg_paras.sigma_v * cb_paras.rho *
                                 (r_now - vasciek_paras.r0) /
                                 vasciek_paras.sigma_r;
        const double miu_y = r_now - miu_y_second;
        const double l_hat =
            l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);
        const double s_now = s_now_vec(k);
        const double df = std::exp(-r_now * dt);
        const double l_curr = l_now_vec(k);

        double b_expected = 0.0;
        double cb_expected = 0.0;
        double eq_expected = 0.0;

        for (int j = 0; j < 9; ++j) {
          const int nxt_m = next_m_middle + dm_vec(j);
          const int nxt_k = nxt_r_idx(k_now, j % 3);
          const double r_next = tree_result.short_rate_tree(nxt_k, i - 1);
          const double y_next = y0 + (x0 + nxt_m * jump - x0) +
                                cdg_paras.sigma_v * cb_paras.rho *
                                    (r_next - vasciek_paras.r0) /
                                    vasciek_paras.sigma_r;
          const double sigma_y = y_next - (y_now + miu_y * dt); // h x 9
          const double l_curr_to_next =
              l_curr + (dt * cdg_paras.lamda * (l_hat - l_curr)) - sigma_y;

          const int l_idx = map_next((nxt_m + m_idx_offset) * cols + nxt_k);

          long long idx_low = (l_next.row(l_idx) <= l_curr_to_next).count() - 1;
          idx_low = std::max(0LL, std::min(idx_low, cb_next.cols() - 2));
          const int idx_high = idx_low + 1;
          const double l_low = l_next(l_idx, idx_low);
          const double l_high = l_next(l_idx, idx_high);
          double weight = (l_curr_to_next - l_low) / (l_high - l_low);
          weight = std::max(0.0, std::min(1.0, weight));
          const double b_val =
              b_next(l_idx, idx_low) +
              weight * (b_next(l_idx, idx_high) - b_next(l_idx, idx_low));
          double cb_val =
              cb_next(l_idx, idx_low) +
              weight * (cb_next(l_idx, idx_high) - cb_next(l_idx, idx_low));
          const double equity_val = equity_next(l_idx, idx_low) +
                                    weight * (equity_next(l_idx, idx_high) -
                                              equity_next(l_idx, idx_low));
          b_expected += b_val * nxt_p(k, j);
          cb_expected += cb_val * nxt_p(k, j);
          eq_expected += equity_val * nxt_p(k, j);
        }

        b_expected *= df;
        cb_expected *= df;
        eq_expected *= df;
        double conversion_val = dil_s_now(k, p) * cb_paras.CR;
        double call_val = 1.0 * cb_paras.CP;
        double final_cb = cb_expected;

        b_expected += cb_paras.coupon_rate * cb_paras.F * coupont_flag;

        if (final_cb > call_val)
          final_cb = call_val;

        if (final_cb < conversion_val)
          final_cb = conversion_val;

        b_now(k, p) = b_expected * survival_flag;
        cb_now(k, p) = final_cb * survival_flag;
        equity_now(k, p) = eq_expected * survival_flag;
      }
    }
    std::swap(b_now, b_next);
    std::swap(cb_now, cb_next);
    std::swap(equity_now, equity_next);
    std::swap(map_now, map_next);
  }
  // std::cout << b_next << std::endl;
  std::printf("Cb Tree Build Completed.\n");
  std::printf("Cb: %.6f\n", cb_next(0, 0));
  std::printf("Bond: %.6f\n", b_next(0, 0));
  std::printf("Equity: %.6f\n", equity_next(0, 0));
}