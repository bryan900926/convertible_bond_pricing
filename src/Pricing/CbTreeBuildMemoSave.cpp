#include <Eigen/Dense>
#include <float.h>
#include <vector>

#include "..\Equity\EquityModel.h"
#include "..\Util\Util.h"
#include "..\Util\TreeManager.hpp"
#include "CbModel.h" 
#include "Eigen/src/Core/Array.h"

FinalResultMemoSave CbTreeBuildMemoSave(
    const CbParas &cb_paras, const CdgParas &cdg_paras,
    const VasciekParas &vasciek_paras, const HullWhiteTreeResult &tree_result,
    const CouponPaidInfo &coupon_info, const PzTreeResult &pz_result,
    const int m_idx_offset, TreeManager &tree_manager) {
    tree_manager.prepare_for_reading();
    const double sigma_x =
    cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first),
    jump_other = sigma_x * std::sqrt(cb_paras.dt_other);
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const int n = coupon_info.total_steps;
    const double y0 = cdg_paras.V0, x0 = cdg_paras.V0;
    
    const auto& thetas = tree_result.alpha_result.thetas;
    const Eigen::ArrayX3i &nxt_r_idx = pz_result.nxt_r_idx;
    std::vector<PackedNode> final_data = tree_manager.get_period(n + 1);

    const int rows = (m_idx_offset + 1) * 2;
    const int cols = tree_result.short_rate_tree.rows() + 1;
    Eigen::ArrayXi map_now = Eigen::ArrayXi::Constant(rows * cols, -1);
    Eigen::ArrayXi map_next = Eigen::ArrayXi::Constant(rows * cols, -1);

    for (int h = 0; h < final_data.size(); ++h) {
        const int m = final_data[h].m;
        const int k = final_data[h].k;
        const int map_idx =
            (m + m_idx_offset) * cols + k;
        map_next(map_idx) = h;
    }
    const double miu_y_second =
        cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2;

    EquityTreeBuildResultMemoSave final_result = EquityTreeBuildMemoSave(final_data, tree_result, cb_paras, cdg_paras, vasciek_paras, coupon_info);
    auto &l_next = final_result.l_data_partition;
    Eigen::ArrayXXd &equity_next = final_result.equity_tree;
    
    Eigen::ArrayXXd cb_next =
        Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
    const auto is_nan = l_next.isNaN();
    const auto is_pos = l_next > 0.0;
    const auto is_neg = !is_pos && !is_nan;
    const auto is_valid = !is_nan && !is_pos;
    const auto low_f = (cb_paras.CR * final_result.equity_tree) < cb_paras.F;
    const auto no_convert = is_valid && low_f;
    const auto convert = is_valid && !no_convert;

    cb_next = is_pos.select(cb_paras.F * cb_paras.rr, cb_next);
    cb_next = no_convert.select(cb_paras.F, cb_next);
    equity_next = is_pos.select(0.0, equity_next);

    Eigen::ArrayXXd dil_s_next =
        Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
    dil_s_next = no_convert.select(equity_next, dil_s_next);
    
    const Eigen::ArrayXXd diluted_stock_price = 
    (equity_next * cb_paras.NS + cb_paras.NC * cb_paras.F) / 
        (cb_paras.CR * cb_paras.NC + cb_paras.NS);
    dil_s_next = convert.select(equity_next.min(diluted_stock_price), dil_s_next);
    cb_next = convert.select((dil_s_next * cb_paras.CR).max(cb_paras.F), cb_next);

    Eigen::ArrayXXd surv_next = Eigen::ArrayXXd::Ones(l_next.rows(), cb_paras.partition);
    surv_next = is_pos.select(0.0, surv_next);
    Eigen::ArrayXd dm_vec(9);
    dm_vec << -2, 0, 2, -2, 0, 2, -2, 0, 2;

    for (int i = n; i >= 1; --i)
    {
        const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
        const double jump = (i == 1) ? jump_first : jump_other;
        const double call_price = cb_paras.call_schedule.GetActiveCallOneTime(
            coupon_info.dt_first + (i - 1) * cb_paras.dt_other);

        const Eigen::ArrayX3d &prob_hw = (i == 1)
                                         ? tree_result.alpha_result.prob_first
                                         : tree_result.alpha_result.prob_other;

        int bool_flag = (i > 1) && coupon_info.is_coupon_paid[i - 1];

        const double l_hat_first =
            (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
                cdg_paras.lamda -
            cdg_paras.v +
            cdg_paras.phi * tree_result.alpha_result.thetas(i - 1);

        const std::vector<PackedNode>& final_data = tree_manager.get_period(i);

        Eigen::ArrayXXd cb_now(final_data.size(), cb_paras.partition);
        Eigen::ArrayXXd surv_now(final_data.size(), cb_paras.partition);

        EquityTreeBuildResultMemoSave tree_build_result = EquityTreeBuildMemoSave(final_data, tree_result, cb_paras, cdg_paras, vasciek_paras, coupon_info);
        auto &l_data_partition = tree_build_result.l_data_partition;
        const auto &equity_tree = tree_build_result.equity_tree;
        #pragma omp parallel 
        {
            _clearfp();
            _controlfp(_MCW_EM, _MCW_EM);
            #pragma omp for
            for (int k = 0; k < final_data.size(); ++k)
            {

                const PackedNode &node = final_data[k];
                const int m_now = node.m;
                const int k_now = node.k;
                const int next_m_middle = node.nxt_middle_m;

                map_now((m_now + m_idx_offset) * cols + k_now) = k;

                const double r_now = tree_result.short_rate_tree(k_now, i - 1);
                const double x_now = x0 + m_now * jump;
                const double y_now = y0 + (x_now - x0) +
                                     cdg_paras.sigma_v * cb_paras.rho *
                                         (r_now - vasciek_paras.r0) /
                                         vasciek_paras.sigma_r;
                const double miu_y = r_now - miu_y_second;
                const double l_hat = l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);
                double df = fast_safe_exp(-r_now * dt);
                const double p_u_hw = prob_hw(k_now, 0);
                const double p_m_hw = prob_hw(k_now, 1);
                const double p_d_hw = prob_hw(k_now, 2);
                double nxt_p[9] = {
                    node.p_x_down * p_u_hw, node.p_x_mid * p_u_hw, node.p_x_up * p_u_hw,
                    node.p_x_down * p_m_hw, node.p_x_mid * p_m_hw, node.p_x_up * p_m_hw,
                    node.p_x_down * p_d_hw, node.p_x_mid * p_d_hw, node.p_x_up * p_d_hw};
                for (size_t p = 0; p < cb_paras.partition; ++p)
                {
                    const double l_curr = l_data_partition(k, p);
                    const bool survival_flag = l_curr <= 0;
                    const double s_now = equity_tree(k, p);
                    const bool no_convert_flag =
                        survival_flag && ((cb_paras.CR * s_now) < cb_paras.F);
                    double current_dil_s = 0.0;
                    if (no_convert_flag)
                    {
                        current_dil_s = s_now;
                    }
                    else
                    {
                        double val_conv = (s_now * cb_paras.NS + cb_paras.NC * cb_paras.F) /
                                          (cb_paras.CR * cb_paras.NC + cb_paras.NS);
                        current_dil_s = std::min(s_now, val_conv);
                    }

                    if (!survival_flag)
                    {
                        cb_now(k, p) = cb_paras.F * cb_paras.rr;
                        surv_now(k, p) = 0.0;
                        continue;
                    }
                    double cb_expected = 0.0;
                    double surv_expected = 0.0;
                    for (int j = 0; j < 9; ++j)
                    {
                        const int nxt_m = next_m_middle + dm_vec(j);
                        const int nxt_k = nxt_r_idx(k_now, j / 3);
                        const double r_next = tree_result.short_rate_tree(nxt_k, i - 1);
                        const double y_next = y0 + (x0 + nxt_m * jump - x0) +
                                              cdg_paras.sigma_v * cb_paras.rho *
                                                  (r_next - vasciek_paras.r0) /
                                                  vasciek_paras.sigma_r;
                        const double sigma_y =
                            y_next - (y_now + miu_y * dt); // h x 9
                        const double l_curr_to_next =
                            l_curr + (dt * cdg_paras.lamda * (l_hat - l_curr)) -
                            sigma_y;
                        double cb_val = 0.0;
                        double surv_val = 0.0;
                        if (l_curr_to_next > 0)
                        {
                            cb_val = cb_paras.F * cb_paras.rr;
                        }
                        else
                        {
                            const int l_idx =
                                map_next((nxt_m + m_idx_offset) * cols + nxt_k);
                            double *l_row_p = &l_next(l_idx, 0);
                            double *next_end_p = l_row_p + cb_paras.partition;
                            double *hi_p = std::upper_bound(l_row_p,
                                                            next_end_p,
                                                            l_curr_to_next);
                            if (hi_p == next_end_p)
                            {
                                hi_p = next_end_p - 1;
                            }
                            double *lo_p = hi_p - 1;
                            if (lo_p < &l_next(l_idx, 0))
                            {
                                lo_p = &l_next(l_idx, 0);
                            }
                            const double l_low = *lo_p;
                            const double l_high = *hi_p;
                            if (lo_p == hi_p || l_high - l_low < 1e-8)
                            {
                                cb_val = cb_next(l_idx, lo_p - l_row_p);
                                surv_val = surv_next(l_idx, lo_p - l_row_p);
                            }
                            else
                            {
                                double weight = std::max(0.0, std::min(1.0, (l_curr_to_next - l_low) / (l_high - l_low)));
                                cb_val = cb_next(l_idx, lo_p - l_row_p) +
                                         weight * (cb_next(l_idx, hi_p - l_row_p) -
                                                   cb_next(l_idx, lo_p - l_row_p));
                                surv_val = surv_next(l_idx, lo_p - l_row_p) +
                                           weight * (surv_next(l_idx, hi_p - l_row_p) -
                                                     surv_next(l_idx, lo_p - l_row_p));
                            }
                        }
                        const double prob = nxt_p[j];
                        cb_expected += cb_val * prob;
                        surv_expected += surv_val * prob;
                    }
                    cb_expected *= df;
                    cb_expected += cb_paras.coupon_rate * cb_paras.F * bool_flag;
                    double conversion_val = current_dil_s * cb_paras.CR;
                    double final_cb = cb_expected;
                    if (final_cb > call_price)
                        final_cb = call_price;

                    if (final_cb < conversion_val)
                        final_cb = conversion_val;
                    surv_now(k, p) = surv_expected;
                    cb_now(k, p) = final_cb;
                }
            }
        }
        std::printf("Completed backward iteration %d\n", i);
        std::swap(cb_now, cb_next);
        std::swap(surv_now, surv_next);
        std::swap(map_now, map_next);
        std::swap(l_next, l_data_partition);
        map_now.setConstant(-1);
    }
    tree_manager.close();
    return FinalResultMemoSave{cb_next(0, 0), 1 - surv_next(0, 0)};
}
