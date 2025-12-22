#include <iostream>
#include <cmath>

#include "EquityModel.h"
#include "CbModel.h"

void CbTreeBuild(const CbParas &cb_paras,
                 const CdgParas &cdg_paras,
                 const VasciekParas &vasciek_paras,
                 const EquityTreeBuildResult &equity_tree_result,
                 const HullWhiteTreeResult &tree_result,
                 CouponPaidInfo coupon_info,
                 const std::vector<int> &num_node_steps)
{
    const Eigen::ArrayXXd &equity_tree = equity_tree_result.equity_tree;
    const Eigen::ArrayX3i &idx_vec = equity_tree_result.idx_vec;
    const Eigen::ArrayXi &nxt_m = equity_tree_result.nxt_m;
    const Eigen::Array<int, Eigen::Dynamic, 9> &nxt_p = equity_tree_result.nxt_p;
    const Eigen::ArrayXXd &l_data = equity_tree_result.l_data_partition;

    const double sigma_x = cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
    const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const int n = coupon_info.total_steps;
    const double y0 = cdg_paras.V0;
    const double x0 = cdg_paras.V0;

    std::vector<int> cum_node_steps(num_node_steps.size(), 0);
    cum_node_steps[0] = num_node_steps[0];

    for (size_t i = 1; i < num_node_steps.size(); ++i)
    {
        cum_node_steps[i] = cum_node_steps[i - 1] + num_node_steps[i];
    }

    const int rows = (n + 1) * 4 + 2;
    const int cols = tree_result.short_rate_tree.rows();
    const int m_idx_offset = n * 2;

    std::vector<int> map_now(rows * cols, -1);
    std::vector<int> map_next(rows * cols, -1);

    auto fill_map = [&](
                        std::vector<int> &map,
                        const int step)
    {
        const auto m_arr = idx_vec.col(2).middleRows(
            cum_node_steps[step - 1],
            num_node_steps[step]);
        const auto k_arr = idx_vec.col(1).middleRows(
            cum_node_steps[step - 1],
            num_node_steps[step]);
        const int *m_ptr = m_arr.data();
        const int *k_ptr = k_arr.data();
        for (int i = 0; i < m_arr.size(); ++i)
        {
            int m_idx = m_ptr[i] + m_idx_offset;
            int k_idx = k_ptr[i];

            map[m_idx * cols + k_idx] = i;
        }
    };

    fill_map(map_now, n);
    const auto &l_next = l_data.middleRows(
        cum_node_steps[cum_node_steps.size() - 2],
        num_node_steps[n]);
    const auto &s_next = equity_tree.middleRows(
        cum_node_steps[cum_node_steps.size() - 2],
        num_node_steps[n]);

    Eigen::ArrayXXd b_next = Eigen::ArrayXXd::Zero(l_next.rows(), cols);
    Eigen::ArrayXXd cb_next = Eigen::ArrayXXd::Zero(l_next.rows(), cols);
    Eigen::ArrayXXd dil_s_next = Eigen::ArrayXXd::Zero(l_next.rows(), cols);
    Eigen::ArrayXXd cb_next = Eigen::ArrayXXd::Zero(l_next.rows(), cols);
    Eigen::ArrayXd equity_now = s_next;

    const auto is_nan = l_data.isNaN();
    const auto is_pos = l_data >= 0.0;
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
        (s_next * cb_paras.NS + cb_paras.NC * cb_paras.F) / (cb_paras.CR * cb_paras.NC + cb_paras.NS);
    dil_s_next = convert.select(s_next.min(conv_val), dil_s_next);
    cb_next = convert.select(
        (dil_s_next * cb_paras.CR).max(cb_paras.F),
        cb_next);
    b_next = is_neg.select(cb_paras.F * (1 + cb_paras.coupon_rate), b_next);

    const std::array<int, 9> dm_vec = {-2, 0, 2, -2, 0, 2, -2, 0, 2};

    Eigen::ArrayXXd b_next;
    Eigen::ArrayXXd cb_next;
    Eigen::ArrayXXd dil_s_next;
    Eigen::ArrayXXd cb_next;
    Eigen::ArrayXd m_now_arr;
    Eigen::ArrayXd k_now_arr;
    Eigen::ArrayXd nxt_m_arr;
    Eigen::ArrayXd r_now_arr;
    Eigen::ArrayXd x_now_arr;
    Eigen::ArrayXd y_now_arr;
    Eigen::ArrayXd l_hat_arr;
    Eigen::ArrayXd miu_y_arr;
    const double l_hat_first = (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) / cdg_paras.lamda -
                               cdg_paras.v + cdg_paras.phi * vasciek_paras.r_bar;

    for (size_t i = n; i >= 1; --i)
    {
        const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
        const double jump = (i == 1) ? jump_first : jump_other;
        const int idx_start = (i > 1) ? cum_node_steps[i - 2] : 0;
        const int num_nodes = num_node_steps[i - 1];

        const auto &l_now = l_data.middleRows(
            idx_start,
            num_nodes);
        const auto &s_now = equity_tree.middleRows(
            idx_start,
            num_nodes);

        b_next = Eigen::ArrayXXd::Zero(l_now.rows(), cols);
        cb_next = Eigen::ArrayXXd::Zero(l_now.rows(), cols);
        dil_s_next = Eigen::ArrayXXd::Zero(l_now.rows(), cols);
        cb_next = Eigen::ArrayXXd::Zero(l_now.rows(), cols);

        m_now_arr = idx_vec.col(2).middleRows(
            idx_start,
            num_nodes);
        k_now_arr = idx_vec.col(1).middleRows(
            idx_start,
            num_nodes);
        nxt_m_arr = nxt_m.col(0).middleRows(
            idx_start,
            num_nodes);
        r_now_arr = tree_result.short_rate_tree.col(i - 1)(k_now_arr);
        x_now_arr = x0 + m_now_arr * jump;
        y_now_arr = y0 + (x_now_arr - x0) + cdg_paras.sigma_v * cb_paras.rho * (r_now_arr - vasciek_paras.r0) / vasciek_paras.sigma_r;
        l_hat_arr = l_hat_first - r_now_arr * (1 / cdg_paras.lamda + cdg_paras.phi);
        miu_y_arr = r_now_arr - (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2);
        if (i > 1)
        {
            fill_map(map_now, i - 1);
        }
    }
}