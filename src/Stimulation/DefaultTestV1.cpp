#include <random>
#include "../HullWhiteModel/HullWhiteModel.h"
#include "../Util/Util.h"


DefaultTestV1Result DefaultTestV1(const CbParas &cb_paras,
                 const CdgParas &cdg_paras,
                 const VasciekParas &vasciek_paras,
                 int num_stimulations) {
    const int rep = 20;
    CouponPaidInfo coupon_info =
        CouponPaidCalc(cb_paras.T, cb_paras.dt_other, cb_paras.paid_cycle);
    const int n = coupon_info.total_steps;
    double t_end = coupon_info.dt_first + n * cb_paras.dt_other;
    const Eigen::ArrayXd time_grid =
        Eigen::ArrayXd::LinSpaced(n + 1, coupon_info.dt_first, t_end);
    Eigen::ArrayXd zero_rates = VasciekZeroRates(vasciek_paras, time_grid);

    HullWhiteTreeResult tree_result =
        HullWhiteTree(vasciek_paras.kappa, vasciek_paras.sigma_r, zero_rates,
                        coupon_info.dt_first, cb_paras.dt_other);
    const Eigen::ArrayXd &thetas = tree_result.alpha_result.thetas;
    PzTreeResult pz_result = PzTreeBuild(coupon_info.total_steps, tree_result);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> randn(0, 1);

    double sum_of_default_probabilities = 0.0;
    const double sigma_x =
        cdg_paras.sigma_v * std::sqrt(1 - cb_paras.rho * cb_paras.rho);
    const double jump_first = sigma_x * std::sqrt(coupon_info.dt_first);
    const double jump_other = sigma_x * std::sqrt(cb_paras.dt_other);
    const double miu_y_second =
        cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2;

    double total_default_periods = 0.0;
    for (int rep_idx = 0; rep_idx < rep; ++rep_idx) {
        int default_count = 0;
        double sum_of_default_periods_iter = 0;
        const double y0 = std::log(cdg_paras.V0);
        for (int stimulation_idx = 0; stimulation_idx < num_stimulations; ++stimulation_idx) {
            double l_now = cdg_paras.l0;
            int m_now = 0, k_now = pz_result.start_h;
            double hw_cum_prob[3] = {0.0, 0.0, 0.0};
            double x_cum_prob[3] = {0.0, 0.0, 0.0};
            for (int t = 1; t <= n; ++t) {
                double jump = (t == 1) ? jump_first : jump_other;
                double dt = (t == 1) ? coupon_info.dt_first : cb_paras.dt_other;
                const Eigen::ArrayX3d &prob_hw =
                    (t == 1) ? tree_result.alpha_result.prob_first
                            : tree_result.alpha_result.prob_other;
                
                hw_cum_prob[0] = prob_hw(k_now, 0); // Up
                hw_cum_prob[1] = prob_hw(k_now, 1) + hw_cum_prob[0]; // Mid
                hw_cum_prob[2] = 1; // Down
                double r_now = tree_result.short_rate_tree(k_now, t - 1);
                
                const double l_hat_first =
                    (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
                        cdg_paras.lamda -
                    cdg_paras.v + cdg_paras.phi * thetas(t - 1);
                const double l_hat =
                    l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);
                double x_now = y0 + m_now * jump;
                double y_now = y0 + (x_now - y0) +
                            cdg_paras.sigma_v * cb_paras.rho *
                                (r_now - vasciek_paras.r0) /
                                vasciek_paras.sigma_r;
                const double miu_y = r_now - miu_y_second;
                const double miu_x =
                    miu_y - cb_paras.sigma_V * cb_paras.rho *
                                (vasciek_paras.kappa * (thetas(t - 1) - r_now)) /
                                vasciek_paras.sigma_r;
                const double expect_x = x_now + miu_x * dt;
                const int nxt_m = std::round((expect_x - y0) / jump);
                const double b = expect_x - y0 - nxt_m * jump;
                const double a = b + 2 * jump;
                const double c = b - 2 * jump;
                const double D = (a - b) * (a - c) * (b - c);
                const double p_d = (sigma_x * sigma_x * dt * (b - c) + c * b * (b - c)) / D;
                const double p_m =
                    (sigma_x * sigma_x * dt * (c - a) + a * c * (c - a)) / ((b - c) * (b - a) * (c - a));
                const double p_u = 1 - p_d - p_m;

                x_cum_prob[0] = p_u;
                x_cum_prob[1] = p_m + x_cum_prob[0]; // Cumulative probability for up and mid
                x_cum_prob[2] = 1; // Cumulative probability for all;

                double x_random = randn(gen);
                double r_random = randn(gen);

                for (int k = 0; k < 3; ++k) {
                    if (r_random <= hw_cum_prob[k]) {
                        k_now = pz_result.nxt_r_idx(k_now, k);
                        break;
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    if (x_random <= x_cum_prob[k]) {
                        m_now = nxt_m - (k - 1) * 2; // k=0->up, k=1->mid, k=2->down
                        break;
                    } 
                }
                const double x_nxt = y0 + m_now * jump;
                const double y_nxt =
                    y0 + (x_nxt - y0) +
                    cdg_paras.sigma_v * cb_paras.rho *
                        (tree_result.short_rate_tree(k_now, t) - vasciek_paras.r0) /
                        vasciek_paras.sigma_r;
                l_now = l_now + (l_hat - l_now) * dt * cdg_paras.lamda -
                        (y_nxt - y_now - miu_y * dt);
                if (l_now > 0) {
                    default_count++;
                    sum_of_default_periods_iter += (coupon_info.dt_first + (t - 1) * cb_paras.dt_other);
                break;
                }
            }
        }
    if (default_count > 0) {
        total_default_periods += sum_of_default_periods_iter / default_count;
    }
    double default_probability =
        static_cast<double>(default_count) / num_stimulations;
    sum_of_default_probabilities += default_probability;
  }
  return {sum_of_default_probabilities / rep, total_default_periods / rep};
}