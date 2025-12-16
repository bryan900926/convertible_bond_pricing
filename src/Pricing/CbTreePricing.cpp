#include <limits>
#include <iostream>
#include "CbModel.h"
#include "..\HullWhiteModel\HullWhiteModel.h"
#include "..\Equity\EquityModel.h"

void CbTreePricing(const CbParas &cb_paras, const CdgParas &cdg_paras, const VasciekParas vasciek_paras)
{
    CouponPaidInfo coupon_info = CouponPaidCalc(
        cb_paras.T,
        cb_paras.dt_other,
        cb_paras.paid_cycle);

    int n = coupon_info.total_steps;

    double t_end = coupon_info.dt_first + n * cb_paras.dt_other;

    const Eigen::ArrayXd time_grid = Eigen::ArrayXd::LinSpaced(
        n + 1,
        coupon_info.dt_first,
        t_end);

    Eigen::ArrayXd zero_rates = VasciekZeroRates(vasciek_paras, time_grid);

    HullWhiteTreeResult tree_result = HullWhiteTree(
        vasciek_paras.kappa,
        vasciek_paras.sigma_r,
        zero_rates,
        coupon_info.dt_first,
        cb_paras.dt_other);

        PzTreeResult pz_result = PzTreeBuild(
        coupon_info.total_steps,
        tree_result);

    const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;

    const double y0 = std::log(cdg_paras.V0);
    const double x0 = std::log(cdg_paras.V0);

    const double sigma_x = cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
    const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);

    std::vector<int> num_node_steps = std::vector<int>(coupon_info.total_steps + 1, 0);
    num_node_steps[0] = 1;

    std::vector<std::array<double, 5>> l_data;
    l_data.reserve(10000);
    l_data.emplace_back(
        std::array<double, 5>{
            1.0, 0.0, static_cast<double>(pz_result.start_h), cdg_paras.l0, cdg_paras.l0});

    const std::array<int, 9> dm_vec = {-2, 0, 2, -2, 0, 2, -2, 0, 2};

    std::vector<std::array<double, 12>> next_p_data;
    next_p_data.reserve(10000);
    std::vector<std::array<double, 4>> next_m_data;
    next_m_data.reserve(10000);
    int idx_nxt_m = 0, idx_total = 0, num_latest_nodes = 1;

    double jump;
    double dt;
    std::unordered_map<uint64_t, int> l_map;

    const double NaN = std::numeric_limits<double>::quiet_NaN();

    const Eigen::ArrayXd range_vec = Eigen::ArrayXd::LinSpaced(
        cb_paras.partition, 1, cb_paras.partition);
    const Eigen::ArrayXd ratio_min = (cb_paras.partition - range_vec) / (cb_paras.partition - 1);
    const Eigen::ArrayXd ratio_max = (range_vec - 1) / (cb_paras.partition - 1);

    Eigen::ArrayXd scratch_root(cb_paras.partition);
    Eigen::ArrayXd scratch_next(cb_paras.partition);

    const double l_hat_first = (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) / cdg_paras.lamda -
                               cdg_paras.v + cdg_paras.phi * vasciek_paras.r_bar;
    const double miu_y_second = cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2;

    for (int i = 1; i <= coupon_info.total_steps; ++i)
    {
        l_map.reserve(num_node_steps[i - 1] * 3);
        if (i == 1)
        {
            jump = jump_first;
            dt = coupon_info.dt_first;
        }
        else
        {
            jump = jump_other;
            dt = cb_paras.dt_other;
        }

        const Eigen::ArrayX3d &prob_hw = (i == 1)
                                             ? tree_result.alpha_result.prob_first
                                             : tree_result.alpha_result.prob_other;

        const int level_start_idx = idx_total - num_latest_nodes + 1;
        const int level_end_idx = idx_total;
        const double S1 = cdg_paras.sigma_v * cdg_paras.sigma_v * dt;

        for (int count = level_start_idx; count <= level_end_idx; ++count)
        {
            const double m_now = l_data[count][1];
            const int k_now = l_data[count][2];
            const double r_now = tree_result.short_rate_tree(static_cast<int>(k_now), i - 1);
            const double l_min_now = l_data[count][3];
            const double l_max_now = l_data[count][4];
            const double l_hat = l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);

            const double p_u_hw = prob_hw(k_now, 0);
            const double p_m_hw = prob_hw(k_now, 1);
            const double p_d_hw = prob_hw(k_now, 2);

            const double x_now = x0 + m_now * jump;
            const double y_now = y0 + (x_now - x0) + cdg_paras.sigma_v * cb_paras.rho * (r_now - vasciek_paras.r0) / vasciek_paras.sigma_r;
            const double miu_y = r_now - miu_y_second;
            const double miu_x = miu_y - cb_paras.sigma_V * cb_paras.rho * (vasciek_paras.kappa * (thetas(i - 1) - r_now)) / vasciek_paras.sigma_r;
            const double expect_x = x_now + miu_x * dt;
            const double nxt_m = std::round((expect_x - x0) / jump);

            const double b = expect_x - x0 - nxt_m * jump;
            const double a = b + 2 * jump;
            const double c = b - 2 * jump;
            const double D = (a - b) * (a - c) * (b - c);

            const double p_u = (S1 * (b - c) + c * b * (b - c)) / D;
            const double p_m = (S1 * (c - a) + a * c * (c - a)) / ((b - c) * (b - a) * (c - a));
            const double p_d = 1 - p_u - p_m;

            std::array<double, 12> p_node = {static_cast<double>(i), m_now, static_cast<double>(k_now),
                                             p_u * p_u_hw, p_u * p_m_hw, p_u * p_d_hw,
                                             p_m * p_u_hw, p_m * p_m_hw, p_m * p_d_hw,
                                             p_d * p_u_hw, p_d * p_m_hw, p_d * p_d_hw};

            next_p_data.emplace_back(p_node);
            std::array<double, 4> m_node = {static_cast<double>(i), m_now, static_cast<double>(k_now), nxt_m};
            next_m_data.emplace_back(m_node);

            for (int j = 0; j < 9; ++j)
            {
                const double m_next = nxt_m + dm_vec[j];
                const double k_next = pz_result.nxt_r_idx(k_now, (j / 3));
                uint64_t key =
                    (static_cast<uint64_t>(static_cast<uint32_t>(m_next)) << 32) |
                    static_cast<uint32_t>(k_next);
                auto search = l_map.find(key);

                if (l_min_now > 0 && l_max_now > 0)
                {
                    if (search == l_map.end())
                    {
                        l_data.emplace_back(
                            std::array<double, 5>{
                                static_cast<double>(i + 1), m_next, k_next,
                                NaN,
                                NaN});
                        l_map[key] = ++idx_total;
                    }
                    continue;
                }

                const double x_nxt = x0 + m_next * jump;
                const double y_nxt = y0 + (x_nxt - x0) + cdg_paras.sigma_v * cb_paras.rho * (tree_result.short_rate_tree(static_cast<int>(k_next), i) - vasciek_paras.r0) / vasciek_paras.sigma_r;

                scratch_root = l_min_now * ratio_min + l_max_now * ratio_max;
                scratch_next = scratch_root + cdg_paras.lamda * (l_hat - scratch_root) * dt - (y_nxt - y_now - miu_y * dt);
                const double l_next_min = scratch_next.minCoeff();
                const double l_next_max = scratch_next.maxCoeff();

                if (search == l_map.end())
                {
                    l_data.emplace_back(
                        std::array<double, 5>{
                            static_cast<double>(i + 1), m_next, k_next,
                            l_next_min,
                            l_next_max});
                    l_map[key] = ++idx_total;
                }
                else
                {
                    std::array<double, 5> &exist_node = l_data[search->second];
                    exist_node[3] = std::min(exist_node[3], l_next_min);
                    exist_node[4] = std::max(exist_node[4], l_next_max);
                }
            }
        }
        num_latest_nodes = idx_total - level_end_idx;
        num_node_steps[i] = num_latest_nodes;
        l_map.clear();
    }

    std::sort(next_m_data.begin(), next_m_data.end(),
              [](const std::array<double, 4> &a, const std::array<double, 4> &b)
              {
                  if (a[0] != b[0])
                      return a[0] < b[0];
                  else if (a[2] != b[2])
                      return a[2] < b[2];
                  else
                      return a[1] < b[1];
              });

    std::sort(next_p_data.begin(), next_p_data.end(),
              [](const std::array<double, 12> &a, const std::array<double, 12> &b)
              {
                  if (a[0] != b[0])
                      return a[0] < b[0];
                  else if (a[2] != b[2])
                      return a[2] < b[2];
                  else
                      return a[1] < b[1];
              });

    std::sort(l_data.begin(), l_data.end(),
              [](const std::array<double, 5> &a, const std::array<double, 5> &b)
              {
                  if (a[0] != b[0])
                      return a[0] < b[0];
                  else if (a[2] != b[2])
                      return a[2] < b[2];
                  else
                      return a[1] < b[1];
              });
    EquityTreeBuild(
        l_data, tree_result, cb_paras, cdg_paras, vasciek_paras, coupon_info);
}
