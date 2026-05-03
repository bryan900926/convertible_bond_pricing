#include <random>
#include "../HullWhiteModel/HullWhiteModel.h"
#include "../Util/Util.h"

double DefaultTest(const CbParas &cb_paras,
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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> randn(0, 1);
  double sum_of_default_probabilities = 0.0;
  for (int _ = 0; _ < rep; ++_) {
    int default_count = 0;
    for (int i = 0; i < num_stimulations; ++i) {
        bool bankruptcy = false;
        double r_now = vasciek_paras.r0;
        double l_now = cdg_paras.l0;
        double y_now = std::log(cdg_paras.V0);
        for (int t = 1; t <= n; ++t) {
            double dt = (t == 1) ? coupon_info.dt_first : cb_paras.dt_other;
            const double l_hat_first =
                (cdg_paras.delta + cdg_paras.sigma_v * cdg_paras.sigma_v / 2) /
                    cdg_paras.lamda -
                cdg_paras.v + cdg_paras.phi * thetas(t - 1);
            const double z_r = randn(gen);
            const double z_v = cb_paras.rho * z_r + std::sqrt(1 - cb_paras.rho * cb_paras.rho) * randn(gen);
            r_now = r_now + vasciek_paras.kappa * (thetas(t - 1) - r_now) * dt +
                    vasciek_paras.sigma_r * std::sqrt(dt) * z_r;
            const double l_hat =
                l_hat_first - r_now * (1 / cdg_paras.lamda + cdg_paras.phi);
            // y_now = y_now + (r_now - cdg_paras.delta - cdg_paras.sigma_v * cdg_paras.sigma_v / 2) * dt + cdg_paras.sigma_v * z_v * dt;
            l_now = l_now + (l_hat - l_now) * dt * cdg_paras.lamda -
                    cdg_paras.sigma_v * std::sqrt(dt) * z_v;
            if (l_now > 0) {
              default_count++;
              break;
            }
        }
    }
    double default_probability =
        static_cast<double>(default_count) / num_stimulations;
    sum_of_default_probabilities += default_probability;
  }
  std::cout << "Average Default Probability over " << rep
            << " repetitions: " << sum_of_default_probabilities / rep
            << std::endl;
  return sum_of_default_probabilities / rep;
}