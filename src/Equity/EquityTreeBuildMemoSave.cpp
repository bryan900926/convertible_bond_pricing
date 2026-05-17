#include "Eigen/Dense"
#include <float.h>
#include <cmath>
#include <omp.h>
#include <vector>

#include "..\Pricing\CbModel.h"
#include "EquityModel.h"


EquityTreeBuildResultMemoSave EquityTreeBuildMemoSave(
                                      const std::vector<PackedNode> &data,
                                      const HullWhiteTreeResult &tree_result,
                                      const CbParas &cb_paras,
                                      const CdgParas &cdg_paras,
                                      const VasciekParas &vasciek_paras,
                                      const CouponPaidInfo &coupon_info)
{
    const Eigen::ArrayXd ratio_min_vec =
        Eigen::ArrayXd::LinSpaced(cb_paras.partition, 1.0, 0.0);
    const Eigen::ArrayXd ratio_max_vec =
        Eigen::ArrayXd::LinSpaced(cb_paras.partition, 0.0, 1.0);

    const double x0 = std::log(cdg_paras.V0);
    const double y0 = std::log(cdg_paras.V0);
    const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;
    const double sigma_x =
        cb_paras.sigma_V * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
    const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);

    const int num_nodes = data.size();

    ArrayXXdRowMajor l_data_partition(num_nodes, cb_paras.partition);
    Eigen::ArrayXd r_data_partition(num_nodes);
    Eigen::ArrayXd theta_data_partition(num_nodes);
    Eigen::ArrayXd theta1_data_partition(num_nodes);
    Eigen::ArrayXd v_data(num_nodes);
    Eigen::ArrayXd t_vec(num_nodes);
    Eigen::ArrayXd alpha_vec(num_nodes);

    #ifdef _WIN32
        unsigned int current_word;
        // _EM_OVERFLOW: The specific bit for overflow
        // _MCW_EM: The mask for all exception control bits
        _controlfp_s(&current_word, _MCW_EM, _MCW_EM);
    #endif

    for (int h = 0; h < data.size(); ++h)
    {
        const size_t t = (data[h].step > 2) ? data[h].step - 2 : 0;
        t_vec(h) = t * cb_paras.dt_other + coupon_info.dt_first;
        alpha_vec(h) = tree_result.alpha_result.alphas(t);
        const Eigen::ArrayXd l_vec =
            data[h].l_min * ratio_min_vec + data[h].l_max * ratio_max_vec;

        const double r_t = tree_result.short_rate_tree(
            static_cast<int>(data[h].k), data[h].step - 1);
        const double x_t = x0 + data[h].m * jump_other;
        double exponent = y0 + (x_t - x0) +
                        cdg_paras.sigma_v * cb_paras.rho * (r_t - vasciek_paras.r0) /
                        vasciek_paras.sigma_r;

        double v_t = 0.0;
        // 1. Catch NaN and Infinity immediately
        if (!std::isfinite(exponent)) {
            v_t = 0.0; 
        } 
        else if (exponent <= 20.0 && exponent >= -20.0) {
            v_t = std::exp(exponent);
        } 
        else if (exponent > 20.0) {
            if (exponent > 700.0) {
                v_t = std::numeric_limits<double>::max();
            } else {
                int parts = static_cast<int>(exponent / 15.0) + 1;
                double safe_base = std::exp(exponent / parts);
                
                v_t = safe_base;
                for (int i = 1; i < parts; ++i) {
                    v_t *= safe_base; // Standard multiplication does not trigger the FPU trap
                }
            }
        } 
        else {
            v_t = 0.0;
        }
        l_data_partition.row(h).segment(0, cb_paras.partition) = l_vec.transpose();
        r_data_partition(h) = r_t;
        theta_data_partition(h) = thetas(t);
        theta1_data_partition(h) = thetas(t + 1);
        v_data(h) = v_t;
    }
    const EquityContext ctx = EquityContextVec(
        cb_paras.dt_other, l_data_partition, r_data_partition, theta_data_partition * vasciek_paras.kappa,
        theta1_data_partition * vasciek_paras.kappa, cb_paras, cdg_paras, vasciek_paras, alpha_vec, t_vec);
    Eigen::ArrayXXd equity_tree =
        EquityFunVec(cb_paras, ctx, v_data) / cb_paras.NS;
    return {equity_tree, l_data_partition};
}