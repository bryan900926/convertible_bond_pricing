#include <Eigen/Dense>
#include <float.h>
#include <vector>

#include "..\Equity\EquityModel.h"
#include "..\Util\Util.h"
#include "..\Util\TreeManager.hpp"
#include "..\Equity\EquityManager.hpp"
#include "CbModel.h" 
#include "Eigen/src/Core/Array.h"

/// @brief This function do the backward induction to calculate the price of the convertible bond, 
/// and save the tree to a file to avoid memory issue
/// @param cb_paras 
/// @param cdg_paras 
/// @param vasciek_paras 
/// @param tree_result 
/// @param coupon_info this class determine which period has coupon paid, and the total steps of the tree
/// @param pz_result 
/// @param m_idx_offset 
/// @param tree_manager 
/// @return 
FinalResultMemoSave CbTreeBuildMemoSave(
    const CbParas &cb_paras, const CdgParas &cdg_paras,
    const VasciekParas &vasciek_paras, const HullWhiteTreeResult &tree_result,
    const CouponPaidInfo &coupon_info, const PzTreeResult &pz_result,
    const int m_idx_offset, TreeManager &tree_manager) {

    // Prepare the tree manager for reading
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

    // load the last period data from the tree manager
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

    EquityManager equity_manager(cb_paras, cdg_paras, vasciek_paras, final_data.size());
    equity_manager.step_update(cb_paras.dt_other, final_data.size());
    EquityTreeBuildMemoSave(final_data, tree_result, cb_paras, cdg_paras, vasciek_paras, coupon_info, equity_manager);

    // ln(debt / company value) at time t, for each node in the tree, this is used to determine if the company is bankrupt or not
    Eigen::ArrayXXd l_next = equity_manager.l_t_arr;
    Eigen::ArrayXXd l_now = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    
    // convertible bond price
    Eigen::ArrayXXd cb_next = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    Eigen::ArrayXXd cb_now = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    
    // survival probability
    Eigen::ArrayXXd surv_next = Eigen::ArrayXXd::Ones(final_data.size(), cb_paras.partition);
    Eigen::ArrayXXd surv_now = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    
    // vanilla bond price
    Eigen::ArrayXXd b_next = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    Eigen::ArrayXXd b_now = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    
    // equity price
    Eigen::ArrayXXd equity_next = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);
    Eigen::ArrayXXd equity_now = Eigen::ArrayXXd::Zero(final_data.size(), cb_paras.partition);

    equity_manager.update_equity(equity_next);

    const auto is_nan = l_next.isNaN();
    const auto is_pos = l_next > 0.0; // bankruptcy, the company value is less than the debt
    const auto is_neg = !is_pos && !is_nan; // normal state
    const auto is_valid = !is_nan && !is_pos;
    const auto low_f = (cb_paras.CR * equity_next) < cb_paras.F; // conversion value is less that face value so no conversion
    const auto no_convert = is_valid && low_f; 
    const auto convert = is_valid && !no_convert; // we do the conversion if the conversion value is greater than the face value


    cb_next = is_pos.select(cb_paras.F * cb_paras.rr, cb_next);
    b_next = is_pos.select(cb_paras.F * cb_paras.rr, b_next);
    b_next = is_neg.select(cb_paras.F, b_next);   
    cb_next = no_convert.select(cb_paras.F, cb_next);
    equity_next = is_pos.select(0.0, equity_next);

    Eigen::ArrayXXd dil_s_next =
        Eigen::ArrayXXd::Zero(l_next.rows(), cb_paras.partition);
    dil_s_next = no_convert.select(equity_next, dil_s_next); // no convert so the diluted stock price is the same as the equity price
    
    const Eigen::ArrayXXd diluted_stock_price = 
    (equity_next * cb_paras.NS + cb_paras.NC * cb_paras.F) / 
        (cb_paras.CR * cb_paras.NC + cb_paras.NS); // diluted stock price if we do the conversion
    dil_s_next = convert.select(equity_next.min(diluted_stock_price), dil_s_next); // if we do the conversion, the diluted stock price is the minimum of the equity price and the diluted stock price
    cb_next = convert.select((dil_s_next * cb_paras.CR).max(cb_paras.F), cb_next);

    surv_next = is_pos.select(0.0, surv_next);
    Eigen::ArrayXd dm_vec(9);
    dm_vec << -2, 0, 2, -2, 0, 2, -2, 0, 2; // x tree transition

    FinalResultMemoSave res;

    for (int i = n; i >= 1; --i)
    {
        const double dt = (i == 1) ? coupon_info.dt_first : cb_paras.dt_other;
        const double jump = (i == 1) ? jump_first : jump_other;
        // company might have different call schedules at different periods, so we need to get the call price at each period
        const double call_price = cb_paras.call_schedule.GetActiveCallOneTime(
            coupon_info.dt_first + (i - 1) * cb_paras.dt_other
        );

        const Eigen::ArrayX3d &prob_hw = (i == 1)
        ? tree_result.alpha_result.prob_first
        : tree_result.alpha_result.prob_other;
        
        int bool_flag = (i > 1) && coupon_info.is_coupon_paid[i - 1];

        // decouple the l_bar part where it is only related to the parameters, so we don't need to calculate it every time
        const double l_hat_first =
        (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
        cdg_paras.lamda -
        cdg_paras.v +
        cdg_paras.phi * tree_result.alpha_result.thetas(i - 1);
            
        const std::vector<PackedNode>& final_data = tree_manager.get_period(i);
        equity_manager.step_update(dt, final_data.size());

        EquityTreeBuildMemoSave(final_data, tree_result, cb_paras, cdg_paras, vasciek_paras, coupon_info, equity_manager);
        // update the equity price for the current period, this method just directly update equity_now.
        equity_manager.update_equity(equity_now);
        l_now = equity_manager.l_t_arr;

        // trim the matrices to only include the active nodes for the current period
        auto cb_active = cb_now.topRows(final_data.size());
        auto surv_active = surv_now.topRows(final_data.size());
        auto b_active = b_now.topRows(final_data.size());
        auto l_active = l_now.topRows(final_data.size());

        #pragma omp parallel for
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
            double df = std::exp(-r_now * dt);
            const double p_u_hw = prob_hw(k_now, 0);
            const double p_m_hw = prob_hw(k_now, 1);
            const double p_d_hw = prob_hw(k_now, 2);
            const double nxt_p[9] = {
                node.p_x_down * p_u_hw, node.p_x_mid * p_u_hw, node.p_x_up * p_u_hw,
                node.p_x_down * p_m_hw, node.p_x_mid * p_m_hw, node.p_x_up * p_m_hw,
                node.p_x_down * p_d_hw, node.p_x_mid * p_d_hw, node.p_x_up * p_d_hw
            };
            
            for (size_t p = 0; p < cb_paras.partition; ++p)
            {
                const double l_curr = l_now(k, p);
                const bool survival_flag = l_curr <= 0;
                const double s_now = i > 1 ? equity_now(k, p) : cb_paras.s0;
                const bool no_convert_flag =
                    survival_flag && ((cb_paras.CR * s_now) < cb_paras.F);
                double current_dil_s = 0.0;

                if (no_convert_flag)
                {
                    current_dil_s = s_now;
                }
                else // stock conversion occurs, we need to calculate the diluted stock price
                {
                    double val_conv = (s_now * cb_paras.NS + cb_paras.NC * cb_paras.F) /
                                        (cb_paras.CR * cb_paras.NC + cb_paras.NS);
                    current_dil_s = std::min(s_now, val_conv);
                    if (i == 1) res.convert_at_t0 = true;
                }
                // bankruptcy case, the company value is less than the debt, so the convertible bond price and vanilla bond price are both equal to the face value times the recovery rate
                if (!survival_flag)
                {
                    cb_active(k, p) = cb_paras.F * cb_paras.rr;
                    b_active(k, p) = cb_paras.F * cb_paras.rr;
                    surv_active(k, p) = 0.0;
                    continue;
                }

                double cb_expected = 0.0;
                double b_expected = 0.0;
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
                    double b_val = 0.0;
                    // bankruptcy happen at next period, so the convertible bond price and vanilla bond price are both equal to the face value times the recovery rate
                    if (l_curr_to_next > 0)
                    {
                        cb_val = cb_paras.F * cb_paras.rr;
                        b_val = cb_paras.F * cb_paras.rr;
                    }
                    else
                    {
                        const int l_idx = map_next((nxt_m + m_idx_offset) * cols + nxt_k);

                        const double l_min = l_next(l_idx, 0);
                        const double l_max = l_next(l_idx, cb_paras.partition - 1);
                        const double dl = (l_max - l_min) / (cb_paras.partition - 1);

                        // if dl is too small, we can just use the first partition value
                        if (dl < 1e-9)
                        {
                            cb_val = cb_next(l_idx, 0);
                            surv_val = surv_next(l_idx, 0);
                            b_val = b_next(l_idx, 0);
                        }
                        else
                        {
                            int p_idx = static_cast<int>((l_curr_to_next - l_min) / dl);

                            if (p_idx < 0)
                                p_idx = 0;
                            if (p_idx >= cb_paras.partition - 1)
                                p_idx = cb_paras.partition - 2;

                            const double l_low = l_next(l_idx, p_idx);
                            const double l_high = l_next(l_idx, p_idx + 1);
                            double weight = (l_curr_to_next - l_low) / (l_high - l_low);

                            if (weight < 0.0)
                                weight = 0.0;
                            if (weight > 1.0)
                                weight = 1.0;

                            cb_val = cb_next(l_idx, p_idx) + weight * (cb_next(l_idx, p_idx + 1) - cb_next(l_idx, p_idx));
                            surv_val = surv_next(l_idx, p_idx) + weight * (surv_next(l_idx, p_idx + 1) - surv_next(l_idx, p_idx));
                            b_val = b_next(l_idx, p_idx) + weight * (b_next(l_idx, p_idx + 1) - b_next(l_idx, p_idx));
                        }
                    }
                    double prob = nxt_p[j];
                    cb_expected += cb_val * prob;
                    surv_expected += surv_val * prob;
                    b_expected += b_val * prob; 
                }
                cb_expected *= df;
                b_expected *= df;
                cb_expected += cb_paras.coupon_rate * cb_paras.F * bool_flag;
                double conversion_val = current_dil_s * cb_paras.CR;
                double final_cb = cb_expected;
                // company would call the convertible bond if the call price is less than the convertible bond price, so we need to take the minimum of the two
                if (final_cb > call_price)
                    final_cb = call_price;
                if (final_cb < conversion_val)
                    final_cb = conversion_val;
                surv_active(k, p) = surv_expected;
                cb_active(k, p) = final_cb;
                b_active(k, p) = b_expected;
            }
        }
        std::cout << "Completed period in backward induction: " << i << "\n";
        cb_now.swap(cb_next);
        surv_now.swap(surv_next);
        b_now.swap(b_next);
        l_now.swap(l_next);
        equity_now.swap(equity_next);
        map_now.swap(map_next);
        map_now.setConstant(-1);
    }
    tree_manager.close();
    res.cb_price = cb_next(0, 0);
    res.default_prob = 1.0 - surv_next(0, 0);
    res.zcb_price = b_next(0, 0);
    return res;
}
