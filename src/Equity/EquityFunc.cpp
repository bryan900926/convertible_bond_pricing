#include <cmath>
#include <iostream>

#include "..\Pricing\CbModel.h"
#include "..\HullWhiteModel\HullWhiteModel.h"
#include "EquityModel.h"

double Calc_Bk(double k, double t)
{
    if (std::abs(k) < 1e-6)
        return t;
    return (1.0 - std::exp(-k * t)) / k;
}

double CalculateEquityNode(
    double v_t, double l_t, double r_t, double dt,
    double theta_t, double theta_t1,
    const CbParas &cb_paras,
    const CdgParas &cdg_paras,
    const VasciekParas &vp,
    const GaussHermiteResult &gh_rule)
{
    const double K = vp.kappa;
    const double sigma_r = vp.sigma_r;
    const double sigma_r_sq = sigma_r * sigma_r;
    const double ld = cdg_paras.lamda;
    const double phi = cdg_paras.phi;
    const double r_bar = vp.r_bar;
    const double sigma_v = cdg_paras.sigma_v;
    const double sigma_v_sq = sigma_v * sigma_v;
    const double rho = cb_paras.rho;
    const double delta = cdg_paras.delta;
    const double v_param = cdg_paras.v;

    const double Bk_dt = Calc_Bk(K, dt);
    const double Bk_2dt = Calc_Bk(2.0 * K, dt);
    const double Bk_ld_dt = Calc_Bk(ld - K, dt);
    const double B_ld_dt = Calc_Bk(ld, dt);
    const double B_ld_plusK_dt = Calc_Bk(ld + K, dt);
    const double B_2ld_dt = Calc_Bk(2.0 * ld, dt);

    const double a = (delta + sigma_v_sq / 2.0) / ld - v_param;
    const double b = -1.0 / ld - phi;

    const double bond_term_1 = (r_bar - sigma_r_sq / (2.0 * K * K)) * (Bk_dt - dt);
    const double bond_term_2 = (sigma_r_sq / (4.0 * K)) * (Bk_dt * Bk_dt);
    const double bond_log = bond_term_1 - bond_term_2 - Bk_dt * r_t;

    const double term_sr_K2 = sigma_r_sq / (2.0 * K * K);
    const double dt_combo = dt - 2.0 * Bk_dt + Bk_2dt;

    const double Mt = term_sr_K2 * dt_combo - bond_log;

    const double exp_ld_dt = std::exp(-ld * dt);
    const double exp_K_dt = std::exp(-K * dt);

    const double term_ht_1 = l_t * exp_ld_dt;
    const double term_ht_2 = a * (1.0 - exp_ld_dt);
    const double term_ht_3 = (ld * phi / K) * ((theta_t1 + exp_ld_dt * theta_t) / 2.0) * dt;
    const double term_ht_4 = ld * b * (exp_ld_dt * Bk_ld_dt * r_t + exp_K_dt * Bk_ld_dt * theta_t * dt / 2.0);

    const double ht = term_ht_1 + term_ht_2 + term_ht_3 + term_ht_4;

    const double M = Mt - delta * dt - sigma_v_sq / 2.0 * dt - (sigma_r_sq / (K * K)) * dt_combo - sigma_v * sigma_r * rho / K * (dt - Bk_dt);

    const double term_N_1 = (ld * b * sigma_r_sq) / (K * (ld - K)) * (Bk_dt - B_ld_dt - Bk_2dt + B_ld_plusK_dt);
    const double term_N_2 = sigma_v * sigma_r * rho / K * (B_ld_dt - B_ld_plusK_dt);

    const double N = ht - term_N_1 + term_N_2;

    const double Cxx = (sigma_r_sq / (K * K)) * dt_combo + 2.0 * sigma_v * sigma_r * rho / K * (dt - Bk_dt) + sigma_v_sq * dt;
    if (Cxx <= 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double term_Cyy_coeff = (ld * b * sigma_r) / (ld - K);
    const double term_Cyy_1 = term_Cyy_coeff * term_Cyy_coeff * (Bk_2dt - 2.0 * B_ld_plusK_dt + B_2ld_dt);
    const double term_Cyy_2 = 2.0 * ld * b * sigma_v * sigma_r * rho / (ld - K) * (B_ld_plusK_dt - B_2ld_dt);
    const double term_Cyy_3 = sigma_v_sq * B_2ld_dt;

    const double Cyy = term_Cyy_1 - term_Cyy_2 + term_Cyy_3;
    if (Cyy <= 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // --- 10. Cxy ---
    const double term_Cxy_1 = ld * b * sigma_r_sq / ((ld - K) * K) * (Bk_dt - B_ld_dt - Bk_2dt + B_ld_plusK_dt);
    const double term_Cxy_2 = sigma_v * sigma_r * rho / K * (B_ld_dt - B_ld_plusK_dt);
    const double term_Cxy_3 = ld * b * sigma_v * sigma_r * rho / (ld - K) * (Bk_dt - Bk_ld_dt);
    const double term_Cxy_4 = sigma_v_sq * Bk_ld_dt;

    const double Cxy = term_Cxy_1 - term_Cxy_2 + term_Cxy_3 - term_Cxy_4;

    // --- 11. Correlation ---
    double corr_xy = Cxy / std::sqrt(Cxx * Cyy);
    // Clamp correlation to [-1, 1]
    if (std::abs(corr_xy) > 1.0)
    {
        corr_xy = std::max(-1.0, std::min(corr_xy, 1.0));
    }

    // --- 12. Pt ---
    const double Bk_sq_integral = (2.0 * K * dt - std::exp(-2.0 * K * dt) + 4.0 * std::exp(-K * dt) - 3.0) / (2.0 * K * K * K);
    const double Pt = std::exp(-Mt + 0.5 * sigma_r_sq * Bk_sq_integral);

    // --- 13. Integration or Closed Form ---
    double equity_val = 0.0;
    const double PI_CONST = 3.14159265358979323846;

    if (std::abs(corr_xy) > 1e-9) // Non-zero correlation
    {
        // Gauss-Hermite Integration
        double integral_sum = FormulaFastScalar(M, N, Cxx, Cyy, corr_xy, gh_rule.x, gh_rule.w);

        equity_val = Pt * v_t * (1.0 / std::sqrt(PI_CONST)) * integral_sum;
    }
    else // Zero Correlation (Closed Form)
    {
        const double sqrt_Cyy = std::sqrt(Cyy);
        const double sqrt2 = std::sqrt(2.0);
        const double z = -N / sqrt_Cyy - sqrt_Cyy;

        const double normcdf_z = 0.5 * (1.0 + std::erf(z / sqrt2));

        equity_val = Pt * v_t * std::exp(M + Cxx / 2.0) * (1.0 - std::exp(N + Cyy / 2.0)) * normcdf_z;
    }

    if (std::isnan(equity_val) || std::isinf(equity_val))
    {
        equity_val = std::numeric_limits<double>::quiet_NaN();
    }
    return equity_val;
}