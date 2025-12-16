#include <iostream>
#include "Eigen/Dense"

#include "..\Pricing\CbModel.h"
#include "EquityModel.h"

void EquityTreeBuild(const std::vector<std::array<double, 5>> &l_data,
                     const HullWhiteTreeResult &tree_result,
                     const CbParas &cb_paras,
                     const CdgParas &cdg_paras,
                     const VasciekParas &vasciek_paras,
                     const CouponPaidInfo &coupon_info)
{
    Eigen::ArrayXXd lt_tree = Eigen::ArrayXXd::Zero(l_data.size(), cb_paras.partition);

    const GaussHermiteResult gh_rule = compute_gauss_hermite_rule(cb_paras.qdt);

    const int cols = l_data[0].size() + cb_paras.partition - 2;
    Eigen::ArrayXXd equity_tree = Eigen::ArrayXXd::Zero(l_data.size(), cols);
    const Eigen::ArrayXd ratio_min_vec = Eigen::ArrayXd::LinSpaced(
        cb_paras.partition,
        1.0,
        0.0);
    const Eigen::ArrayXd ratio_max_vec = Eigen::ArrayXd::LinSpaced(
        cb_paras.partition,
        0.0,
        1.0);
    double dt, jump, theta_t, theta_t1;
    const double x0 = std::log(cdg_paras.V0);
    const double y0 = std::log(cdg_paras.V0);
    const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;
    const double sigma_x = cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
    const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);
    Eigen::ArrayXXd l_data_partition = Eigen::ArrayXXd::Zero(l_data.size(), equity_tree.cols());
    for (int h = 0; h < l_data.size(); ++h)
    {
        if (h == 0)
        {
            dt = coupon_info.dt_first;
            jump = jump_first;
            theta_t = 0;
            theta_t1 = 0;
        }
        else
        {
            const int t = l_data[h][0] - 2;
            dt = cb_paras.dt_other;
            jump = jump_other;
            theta_t = thetas(t);
            theta_t1 = thetas(t + 1);
        }
        l_data_partition(h, 0) = l_data[h][0];
        l_data_partition(h, 1) = l_data[h][1];
        l_data_partition(h, 2) = l_data[h][2];
        const double l_min = l_data[h][3];
        const double l_max = l_data[h][4];
        const Eigen::ArrayXd l_vec = l_min * ratio_min_vec + l_max * ratio_max_vec;
        l_data_partition.row(h).segment(3, cb_paras.partition) = l_vec.transpose();

        const double r_t = tree_result.short_rate_tree(static_cast<int>(l_data[h][2]), static_cast<int>(l_data[h][0]) - 1);
        const double x_t = x0 + l_data[h][1] * jump;
        const double v_t = std::exp(y0 + (x_t - x0) + cdg_paras.sigma_v * cb_paras.rho * (r_t - vasciek_paras.r0) / vasciek_paras.sigma_r);
        lt_tree(h, Eigen::all) = l_vec.transpose();
        for (int p = 0; p < cb_paras.partition; ++p)
        {
            if (l_vec(p) >= 0)
            {
                equity_tree(h, p) = 0.0;
                continue;
            }
            // Calculate the equity value for each partition
            equity_tree(h, p) = CalculateEquityNode(v_t, l_vec(p), r_t, dt, theta_t * vasciek_paras.kappa, theta_t1 * vasciek_paras.kappa, cb_paras, cdg_paras, vasciek_paras, gh_rule) / cb_paras.NS;
        }
        if (h < 30)
        {
            std::cout << "Step: " << l_data[h][0] - 1 << ", Node: " << h
                      << ", l_t: " << l_min
                      << ", r_t: " << r_t
                      << ", v_t: " << v_t << std::endl
                      << ", Equitys: " << equity_tree(h, 0) << std::endl;
        }
    } // tree is for later calculation
}
