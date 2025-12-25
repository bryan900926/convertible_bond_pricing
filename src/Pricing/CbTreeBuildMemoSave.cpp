#include <Eigen/Dense>
#include <vector>

#include "..\Equity\EquityModel.h"
#include "CbModel.h"
#include "../Util/Timer.h"
#include "Eigen/src/Core/Array.h"
#include "Eigen/src/Core/util/Constants.h"

void CbTreeBuildMemoSave(const CbParas &cb_paras, const CdgParas &cdg_paras,
                         const VasciekParas &vasciek_paras,
                         const EquityTreeBuildResult &equity_tree_result,
                         const HullWhiteTreeResult &tree_result,
                         const CouponPaidInfo &coupon_info,
                         const PzTreeResult &pz_result,
                         const std::vector<int> &num_node_steps) {

  const double EPSILON = 1e-12;
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

  int max_nodes = 0;
  for (size_t i = 1; i < num_node_steps.size(); ++i) {
    cum_node_steps[i] =
        cum_node_steps[i - 1] + num_node_steps[i]; // [1, 3, 7,...]
    max_nodes = std::max(max_nodes, num_node_steps[i]);
  }

  const int m_idx_offset = idx_vec.col(1).abs().maxCoeff();
  const int rows = (m_idx_offset + 1) * 2;
  const int cols = tree_result.short_rate_tree.rows() + 1;

  Eigen::ArrayXi map_now = Eigen::ArrayXi::Constant(rows * cols, -1);
  Eigen::ArrayXi map_next = Eigen::ArrayXi::Constant(rows * cols, -1);

  FillMap(map_next, n, idx_vec, cum_node_steps, num_node_steps, m_idx_offset,
          cols);

  Eigen::ArrayXd dm_vec(9);
  dm_vec << -2, -2, -2, 0, 0, 0, 2, 2, 2;

  const double l_hat_first =
      (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
          cdg_paras.lamda -
      cdg_paras.v + cdg_paras.phi * vasciek_paras.r_bar;

  Eigen::ArrayXd m_now_arr_buf(max_nodes);
  Eigen::ArrayXi k_now_arr_buf(max_nodes);

  Eigen::ArrayXXd s_now_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd b_now_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd cb_now_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd dil_s_now_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd equity_now_buf(max_nodes, cb_paras.partition);

  Eigen::ArrayXXd b_next_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd cb_next_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd dil_s_next_buf(max_nodes, cb_paras.partition);
  Eigen::ArrayXXd equity_next_buf(max_nodes, cb_paras.partition);

  Eigen::ArrayXXd *p_s_now = &s_now_buf;
  Eigen::ArrayXXd *p_b_now = &b_now_buf;
  Eigen::ArrayXXd *p_cb_now = &cb_now_buf;
  Eigen::ArrayXXd *p_dil_s_now = &dil_s_now_buf;
  Eigen::ArrayXXd *p_equity_now = &equity_now_buf;

  Eigen::ArrayXXd *p_b_next = &b_next_buf;
  Eigen::ArrayXXd *p_cb_next = &cb_next_buf;
  Eigen::ArrayXXd *p_dil_s_next = &dil_s_next_buf;
  Eigen::ArrayXXd *p_equity_next = &equity_next_buf;
  {
    // A. Identify dimensions for Maturity
    int node_term = num_node_steps[n];
    int idx_start_term = cum_node_steps[cum_node_steps.size() - 2];

    // B. Create VIEWS into the global input data (No Copying)
    const auto l_next_final = l_data.middleRows(idx_start_term, node_term);
    const auto s_next_final =
        equity_tree.middleRows(idx_start_term, node_term);

    // C. Create WRITEABLE VIEWS into your "Next" buffers
    // These act like references; writing to them fills the buffer.
    auto b_next_view = p_b_next->topRows(node_term);
    auto cb_next_view = p_cb_next->topRows(node_term);
    auto dil_s_next_view = p_dil_s_next->topRows(node_term);
    auto equity_next_view = p_equity_next->topRows(node_term);

    // D. Initialize Defaults (Equivalent to ::Zero)
    b_next_view.setZero();
    cb_next_view.setZero();
    dil_s_next_view.setZero();
    equity_next_view = s_next_final; // Initial Equity is Stock Price

    // E. Masks (Calculated on the global read-only data)
    const auto is_nan = l_next_final.isNaN();
    const auto is_pos = l_next_final > 0.0;
    const auto is_neg = !is_pos;
    const auto is_valid = !is_nan && !is_pos;
    const auto low_f = (cb_paras.CR * s_next_final) < cb_paras.F;
    const auto no_convert = is_valid && low_f;
    const auto convert = is_valid && !no_convert;

    // F. Apply Payoffs (Writing into the Views)
    dil_s_next_view = is_nan.select(NaN, dil_s_next_view);

    b_next_view = is_pos.select(cb_paras.F * cb_paras.rr, b_next_view);

    cb_next_view = is_pos.select(
        cb_paras.F * cb_paras.rr * (-l_next_final).exp(), cb_next_view);
    cb_next_view = no_convert.select(cb_paras.F, cb_next_view);

    dil_s_next_view = no_convert.select(s_next_final, dil_s_next_view);

    const Eigen::ArrayXXd conv_val =
        (s_next_final * cb_paras.NS + cb_paras.NC * cb_paras.F) /
        (cb_paras.CR * cb_paras.NC + cb_paras.NS);

    dil_s_next_view =
        convert.select(s_next_final.min(conv_val), dil_s_next_view);

    cb_next_view = convert.select(
        (dil_s_next_view * cb_paras.CR).max(cb_paras.F), cb_next_view);

    b_next_view =
        is_neg.select(cb_paras.F * (1 + cb_paras.coupon_rate), b_next_view);
  }
  Eigen::ArrayXi valid_idx_buf(max_nodes * 9);
    Eigen::ArrayXd l_1_buf(max_nodes * 9);
    Eigen::ArrayXd l_2_buf(max_nodes * 9);
    Eigen::ArrayXd cb_interp_1_buf(max_nodes * 9);
    Eigen::ArrayXd cb_interp_2_buf(max_nodes * 9);
    Eigen::ArrayXd b_interp_1_buf(max_nodes * 9);
    Eigen::ArrayXd b_interp_2_buf(max_nodes * 9);
    Eigen::ArrayXd equity_interp_1_buf(max_nodes * 9);
    Eigen::ArrayXd equity_interp_2_buf(max_nodes * 9);

  for (size_t i = n; i >= 1; --i) {
    const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
    const double jump = (i == 1) ? jump_first : jump_other;
    const int idx_start = (i > 1) ? cum_node_steps[i - 2] : 0;
    const int idx_start_next = cum_node_steps[i - 1];
    const int num_nodes = num_node_steps[i - 1];
    const int num_nodes_next = num_node_steps[i];

    const Eigen::ArrayXi h_range = Eigen::ArrayXi::LinSpaced(
        num_nodes, idx_start, idx_start + num_nodes - 1);

    auto l_now = l_data.middleRows(idx_start, num_nodes);
    auto l_next = l_data.middleRows(idx_start_next, num_nodes_next);

    auto s_now = equity_tree.middleRows(idx_start, num_nodes);
    auto b_now = p_b_now->topRows(num_nodes);
    auto cb_now = p_cb_now->topRows(num_nodes);
    auto dil_s_now = p_dil_s_now->topRows(num_nodes);
    auto equity_now = p_equity_now->topRows(num_nodes);

    auto b_next = p_b_next->topRows(num_nodes_next);
    auto cb_next = p_cb_next->topRows(num_nodes_next);
    auto dil_s_next = p_dil_s_next->topRows(num_nodes_next);
    auto equity_next = p_equity_next->topRows(num_nodes_next);

    auto m_now_arr = m_now_arr_buf.head(num_nodes);
    auto k_now_arr = k_now_arr_buf.head(num_nodes);
    m_now_arr = idx_vec.col(1).middleRows(idx_start, num_nodes).cast<double>();
    k_now_arr = idx_vec.col(2).middleRows(idx_start, num_nodes);

    Eigen::ArrayXd nxt_m_arr =
        nxt_m.middleRows(idx_start, num_nodes).cast<double>();

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
      FillMap(map_now, i - 1, idx_vec, cum_node_steps, num_node_steps,
              m_idx_offset, cols);
    }
    Eigen::ArrayXXi k_next_arr =
        nxt_r_idx(k_now_arr, Eigen::all).replicate(1, 3); // h x 9
    Eigen::ArrayXi flat_k_next = k_next_arr.reshaped<Eigen::RowMajor>();
    Eigen::ArrayXd next_rates_col = tree_result.short_rate_tree.col(i - 1);
    Eigen::ArrayXd flat_rates = next_rates_col(flat_k_next);
    Eigen::ArrayXXd r_next_vals = flat_rates.reshaped<Eigen::RowMajor>(
        k_next_arr.rows(), k_next_arr.cols());
    Eigen::ArrayXXd m_next_arr =
        nxt_m_arr.replicate(1, 9).rowwise() + dm_vec.transpose(); // h x 9
    Eigen::ArrayXi flat_m_next =
        m_next_arr.reshaped<Eigen::RowMajor>().cast<int>();
    Eigen::ArrayXXd y_next_arr = y0 + (x0 + m_next_arr * jump - x0) +
                                 cdg_paras.sigma_v * cb_paras.rho *
                                     (r_next_vals - vasciek_paras.r0) /
                                     vasciek_paras.sigma_r; // h x 9
                                                            //
    const Eigen::ArrayXi hash_nxt_idx =
        (flat_m_next + m_idx_offset) * cols + flat_k_next; // h x 9
                                                           //
    const Eigen::ArrayXXi nxt_idx =
        map_next(hash_nxt_idx).reshaped<Eigen::RowMajor>(num_nodes, 9); // h x 9

    Eigen::ArrayXXd sigma_term =
        y_next_arr.colwise() - (y_now_arr + miu_y_arr * dt); // h x 9

    int bool_flag = (i > 1) && coupon_info.is_coupon_paid[i - 1];
    Eigen::ArrayXd discount_factor = (-r_now_arr * dt).exp();

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
          (cb_paras.F * cb_paras.rr * (-l_now_vec)).exp(), cb_now.col(p));

      const auto interp = !(is_nan || is_pos); // num_interp x 1
      const int num_interp = interp.count();

      if (num_interp == 0)
        continue;
      
      int cnt = 0;
      for (int j = 0; j < interp.size(); ++j) {
          if (interp(j)) valid_idx_buf(cnt++) = j;
      }

      auto valid_idx = valid_idx_buf.head(num_interp);

      Eigen::ArrayXd l_curr = l_now_vec(valid_idx);     // num_interp x 1
      Eigen::ArrayXd l_hat_curr = l_hat_arr(valid_idx); // num_interp x 1
      Eigen::ArrayXd df = discount_factor(valid_idx);   // num_interp x 1
      Eigen::ArrayXd l_curr_to_next =
          ((-sigma_term(valid_idx, Eigen::all)).colwise() +
           (l_curr + dt * cdg_paras.lamda * (l_hat_curr - l_curr)))
              .reshaped<Eigen::ColMajor>(); // (num_interp x 9) x 1

      const auto flat_idx =
          nxt_idx(valid_idx, Eigen::all)
              .reshaped<Eigen::ColMajor>(); // num_interp x 9
      Eigen::ArrayXXd l_sub =
          l_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd b_sub =
          b_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd cb_sub =
          cb_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      Eigen::ArrayXXd equity_sub =
          equity_next(flat_idx, Eigen::all); // (num_interp x 9) x partition
      const auto l_first = l_sub.col(0); // num_interp x 1
      const auto l_last = l_sub.col(l_sub.cols() - 1); // num_interp x 1

      Eigen::ArrayXi count =
          (l_sub <= l_curr_to_next.replicate(1, l_sub.cols()))
              .rowwise()
              .count()
              .cast<int>();
      Eigen::ArrayXi nxt_l_idx = (count - 1).max(0).min(l_sub.cols() - 2);

auto l_1 = l_1_buf.head(l_sub.rows());
auto l_2 = l_2_buf.head(l_sub.rows());
auto cb_interp_1 = cb_interp_1_buf.head(l_sub.rows());
auto cb_interp_2 = cb_interp_2_buf.head(l_sub.rows());
auto b_interp_1 = b_interp_1_buf.head(l_sub.rows());
auto b_interp_2 = b_interp_2_buf.head(l_sub.rows());
auto equity_interp_1 = equity_interp_1_buf.head(l_sub.rows());
auto equity_interp_2 = equity_interp_2_buf.head(l_sub.rows());

      for (int j = 0; j < l_sub.rows(); ++j) {
        l_1(j) = l_sub(j, nxt_l_idx(j));
        l_2(j) = l_sub(j, nxt_l_idx(j) + 1);
        cb_interp_1(j) = cb_sub(j, nxt_l_idx(j));
        cb_interp_2(j) = cb_sub(j, nxt_l_idx(j) + 1);
        b_interp_1(j) = b_sub(j, nxt_l_idx(j));
        b_interp_2(j) = b_sub(j, nxt_l_idx(j) + 1);
        equity_interp_1(j) = equity_sub(j, nxt_l_idx(j));
        equity_interp_2(j) = equity_sub(j, nxt_l_idx(j) + 1);
      }

      Eigen::ArrayXd denom = l_2 - l_1;
      const auto is_flat = (denom.abs() < EPSILON);
      Eigen::ArrayXd safe_denom = is_flat.select(1.0, denom);
      Eigen::ArrayXd weight = (l_curr_to_next - l_1) / safe_denom;
      weight = is_flat.select(0.0, weight);

      const auto b_next_flat = DoInterp(b_interp_1, b_interp_2,
                                  weight); // (num_interp x 9) x 1

      const auto cb_next_flat = DoInterp(cb_interp_1, cb_interp_2,
                                   weight); // (num_interp x 9) x 1
      const auto equity_next_flat = DoInterp(equity_interp_1, equity_interp_2,
                                       weight); // (num_interp x 9) x 1

      const auto b_reshape_next =
          b_next_flat.reshaped<Eigen::ColMajor>(num_interp, 9);
      const auto cb_reshape_next =
          cb_next_flat.reshaped<Eigen::ColMajor>(num_interp, 9);
      const auto equity_reshape_next =
          equity_next_flat.reshaped<Eigen::ColMajor>(num_interp, 9);
      Eigen::ArrayXXd next_p_subs = nxt_p(h_range(valid_idx), Eigen::all);

      Eigen::ArrayXd b_res =
          (b_reshape_next * next_p_subs).rowwise().sum() * df;
      Eigen::ArrayXd cb_res =
          (cb_reshape_next * next_p_subs).rowwise().sum() * df;
      Eigen::ArrayXd equity_res =
          (equity_reshape_next * next_p_subs).rowwise().sum() * df;

      b_res += cb_paras.coupon_rate * cb_paras.F * bool_flag;
      cb_res =
          cb_res.min(cb_paras.CP).max(cb_paras.CR * dil_s_now(valid_idx, p));
      b_now(valid_idx, p) = b_res;
      cb_now(valid_idx, p) = cb_res;
      equity_now(valid_idx, p) = equity_res;
    }
    std::swap(p_b_next, p_b_now);
    std::swap(p_cb_next, p_cb_now);
    std::swap(p_equity_next, p_equity_now);
    std::swap(map_now, map_next);
    map_now.setConstant(-1);
  }
  std::printf("Cb Tree Build Completed.\n");
  std::printf("Cb: %.6f\n", (*p_cb_next)(0, 0));
  std::printf("Bond: %.6f\n", (*p_b_next)(0, 0));
  std::printf("Equity: %.6f\n", (*p_equity_next)(0, 0));
}