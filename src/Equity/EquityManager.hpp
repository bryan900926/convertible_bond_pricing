#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

#include "../Pricing/CbModel.h"
#include "../HullWhiteModel/HullWhiteModel.h"
#include "../Equity/EquityModel.h"

/// @brief EquityTimeStepCache is a structure that caches precomputed values for a given time step in the equity model.
struct EquityTimeStepCache
{
    double B_kappa_dt;
    double B_2kappa_dt;
    double B_lambda_dt;
    double B_lambda_minus_kappa_dt;
    double B_lambda_plus_kappa_dt;
    double B_2lambda_dt;
    double a;
    double b;
    double c_xx;
    double c_yy;
    double c_xy;
    double corr_xy;
    double cur_dt = 0;
    double b_t_t_plus_dt = 0;
    double b_k_square_integral = 0;

    void update(const double dt, const CbParas &cb, const CdgParas &cdg, const VasciekParas &vp)
    {
        if (dt == cur_dt)
        {
            return;
        }
        cur_dt = dt;

        B_kappa_dt = Calc_Bk(vp.kappa, dt);
        B_2kappa_dt = Calc_Bk(2.0 * vp.kappa, dt);
        B_lambda_dt = Calc_Bk(cdg.lamda, dt);
        B_lambda_minus_kappa_dt = Calc_Bk(cdg.lamda - vp.kappa, dt);
        B_lambda_plus_kappa_dt = Calc_Bk(cdg.lamda + vp.kappa, dt);
        B_2lambda_dt = Calc_Bk(2.0 * cdg.lamda, dt);

        a = (cdg.delta + cdg.sigma_v * cdg.sigma_v / 2.0) / cdg.lamda - cdg.v;
        b = -1.0 / cdg.lamda - cdg.phi;

        c_xx = ((vp.sigma_r * vp.sigma_r) / (vp.kappa * vp.kappa)) *
                   (dt - 2.0 * B_kappa_dt + B_2kappa_dt) +
               2 * cdg.sigma_v * vp.sigma_r * cb.rho /
                   vp.kappa * (dt - B_kappa_dt) +
               cdg.sigma_v * cdg.sigma_v * dt;

        c_yy = std::pow(cdg.lamda * b * vp.sigma_r, 2) /
                   std::pow(cdg.lamda - vp.kappa, 2) *
                   (B_2kappa_dt - 2 * B_lambda_plus_kappa_dt + B_2lambda_dt) -
               2 * cdg.lamda * b * vp.sigma_r * cdg.sigma_v * cb.rho /
                   (cdg.lamda - vp.kappa) *
                   (B_lambda_plus_kappa_dt - B_2lambda_dt) +
               cdg.sigma_v * cdg.sigma_v * B_2lambda_dt;

        c_xy = cdg.lamda * b * std::pow(vp.sigma_r, 2) /
                   (vp.kappa * (cdg.lamda - vp.kappa)) *
                   (B_kappa_dt - B_lambda_dt - B_2kappa_dt + B_lambda_plus_kappa_dt) -
               cdg.lamda * b * cdg.sigma_v * cb.rho /
                   (cdg.lamda - vp.kappa) *
                   (B_lambda_plus_kappa_dt - B_2lambda_dt) -
               cdg.sigma_v * vp.sigma_r * cb.rho / vp.kappa *
                   (B_lambda_dt - B_lambda_plus_kappa_dt) +
               cdg.lamda * b * cdg.sigma_v * vp.sigma_r * cb.rho /
                   (cdg.lamda - vp.kappa) *
                   (B_kappa_dt - B_lambda_dt) -
               cdg.sigma_v * cdg.sigma_v *
                   B_lambda_dt;

        b_t_t_plus_dt = (1 - std::exp(-vp.kappa * dt)) / vp.kappa;

        if (c_yy <= 1e-12 || c_xx <= 1e-12)
        {
            throw std::runtime_error("Variance is zero or negative (division by zero risk).");
        }

        corr_xy = c_xy / std::sqrt(c_xx * c_yy);
        corr_xy = std::max(-1.0, std::min(1.0, corr_xy));

        b_k_square_integral =
            (2 * vp.kappa * dt - std::exp(-2 * vp.kappa * dt) +
             4 * std::exp(-vp.kappa * dt) - 3) /
            (2 * std::pow(vp.kappa, 3));
    }
};

class EquityManager
{
private:
    CbParas cb_;
    CdgParas cdg_;
    VasciekParas vp_;
    GaussHermiteResult gh_rule_;
    EquityTimeStepCache cache_;

    int current_rows_; // TRACKS THE SHRINKING SIZE

    // N x P Matrices
    Eigen::ArrayXXd h_t_arr;
    Eigen::ArrayXXd m_arr;
    Eigen::ArrayXXd n_arr;
    Eigen::ArrayXXd exp_part1_base;
    Eigen::ArrayXXd exp_part2_base;
    Eigen::ArrayXXd integral_arr;
    Eigen::ArrayXXd d1_arr;
    Eigen::ArrayXXd d2_arr;
    Eigen::ArrayXXd d1_static;
    Eigen::ArrayXXd d2_static;
    Eigen::ArrayXXd g_arr;

    // N x 1 Vectors

    Eigen::ArrayXd p_0_t_plus_dt;
    Eigen::ArrayXd p_0_t;
    Eigen::ArrayXd a_mid_part;
    Eigen::ArrayXd a_t_t_plus_dt;
    Eigen::ArrayXd bond_log_arr;
    Eigen::ArrayXd m_t_arr;
    Eigen::ArrayXd pt_arr;

    // Zero-correlation temp buffers
    Eigen::ArrayXXd term_A;
    Eigen::ArrayXXd term_B;
    Eigen::ArrayXXd x_arr;
    Eigen::ArrayXXd first_arr;
    Eigen::ArrayXXd second_arr;

    const double SQRT_1_2 = 0.70710678118654752440;
    const double SQRT_2 = 1.41421356237309504880;
    const double INV_SQRT_PI = 0.564189583547756286948;

    Eigen::ArrayXXd int_part1_arr;
    Eigen::ArrayXXd int_part2_arr;

public:
    Eigen::ArrayXXd l_t_arr;
    Eigen::ArrayXd r_t_arr;
    Eigen::ArrayXd t_vec;
    Eigen::ArrayXd v_t_arr;
    Eigen::ArrayXd theta_t_arr;
    Eigen::ArrayXd theta_t1_arr;
    Eigen::ArrayXd alpha_vec;
    // ALLOCATE MAXIMUM MEMORY ONLY ONCE IN THE CONSTRUCTOR
    EquityManager(const CbParas &cb, const CdgParas &cdg, const VasciekParas &vp, const int max_num_data)
        : cb_(cb), cdg_(cdg), vp_(vp), current_rows_(max_num_data)
    {
        gh_rule_ = compute_gauss_hermite_rule(cb_.qdt);

        // 2D Matrices
        l_t_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        h_t_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        m_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        n_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        exp_part1_base = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        exp_part2_base = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        integral_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        d1_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        d2_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        d1_static = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        d2_static = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        g_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);

        term_A = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        term_B = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        x_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        first_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        second_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);

        // 1D Vectors
        r_t_arr = Eigen::ArrayXd::Zero(max_num_data);
        theta_t_arr = Eigen::ArrayXd::Zero(max_num_data);
        theta_t1_arr = Eigen::ArrayXd::Zero(max_num_data);
        alpha_vec = Eigen::ArrayXd::Zero(max_num_data);
        p_0_t_plus_dt = Eigen::ArrayXd::Zero(max_num_data);
        p_0_t = Eigen::ArrayXd::Zero(max_num_data);
        a_mid_part = Eigen::ArrayXd::Zero(max_num_data);
        a_t_t_plus_dt = Eigen::ArrayXd::Zero(max_num_data);
        bond_log_arr = Eigen::ArrayXd::Zero(max_num_data);
        m_t_arr = Eigen::ArrayXd::Zero(max_num_data);
        t_vec = Eigen::ArrayXd::Zero(max_num_data);
        pt_arr = Eigen::ArrayXd::Zero(max_num_data);
        v_t_arr = Eigen::ArrayXd::Zero(max_num_data);

        int_part1_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
        int_part2_arr = Eigen::ArrayXXd::Zero(max_num_data, cb_.partition);
    }

    // Call this at the start of your backward induction loop step to update the cached values for the current time step.
    void step_update(const double dt, const int active_paths)
    {
        cache_.update(dt, cb_, cdg_, vp_);
        current_rows_ = active_paths;
    }

    void update_context()
    {
        // 1. Create fast views into the active data window
        auto l_active = l_t_arr.topRows(current_rows_);
        auto h_active = h_t_arr.topRows(current_rows_);
        auto m_active = m_arr.topRows(current_rows_);
        auto n_active = n_arr.topRows(current_rows_);

        auto r_active = r_t_arr.head(current_rows_);
        auto t_active = t_vec.head(current_rows_);
        auto theta_active = theta_t_arr.head(current_rows_);
        auto theta1_active = theta_t1_arr.head(current_rows_);
        auto alpha_active = alpha_vec.head(current_rows_);

        auto p0_t_plus_active = p_0_t_plus_dt.head(current_rows_);
        auto p0_t_active = p_0_t.head(current_rows_);
        auto a_mid_active = a_mid_part.head(current_rows_);
        auto a_t_t_plus_active = a_t_t_plus_dt.head(current_rows_);
        auto bond_log_active = bond_log_arr.head(current_rows_);
        auto mt_active = m_t_arr.head(current_rows_);
        auto pt_active = pt_arr.head(current_rows_);

        p0_t_plus_active = Eigen::exp(-(r_active * (cache_.cur_dt + t_active)));
        p0_t_active = Eigen::exp(-r_active * t_active);

        a_mid_active = 1.0 / (4.0 * std::pow(vp_.kappa, 3)) * vp_.sigma_r * vp_.sigma_r *
                       (Eigen::exp(-vp_.kappa * (cache_.cur_dt + t_active)) * -Eigen::exp(-vp_.kappa * t_active))
                           .pow(2) *
                       (Eigen::exp(2 * vp_.kappa * t_active) - 1.0);

        a_t_t_plus_active = Eigen::exp(
            Eigen::log(p0_t_plus_active / p0_t_active) +
            (cache_.b_t_t_plus_dt * alpha_active - a_mid_active));

        bond_log_active = a_t_t_plus_active * Eigen::exp(-cache_.b_t_t_plus_dt * r_active);

        mt_active = (vp_.sigma_r * vp_.sigma_r / (2.0 * vp_.kappa * vp_.kappa)) *
                        (cache_.cur_dt - 2.0 * cache_.B_kappa_dt + cache_.B_2kappa_dt) -
                    bond_log_active;

        Eigen::ArrayXd h_vector_part =
            ((cdg_.lamda * cdg_.phi / vp_.kappa) * ((theta1_active + std::exp(-cdg_.lamda * cache_.cur_dt) * theta_active) / 2.0) * cache_.cur_dt +
             cache_.a * (1.0 - std::exp(-cdg_.lamda * cache_.cur_dt))) +
            (r_active * (cdg_.lamda * cache_.b * std::exp(-cdg_.lamda * cache_.cur_dt) * cache_.B_lambda_minus_kappa_dt)) +
            (cdg_.lamda * cache_.b * std::exp(-vp_.kappa * cache_.cur_dt) * cache_.B_lambda_minus_kappa_dt * theta_active * cache_.cur_dt / 2.0);

        // Broadcast the complete 1D vector onto the 2D matrix in one shot
        h_active = (l_active * std::exp(-cdg_.lamda * cache_.cur_dt)).colwise() + h_vector_part;

        // Vector broadcast directly into N x P matrix
        m_active.colwise() = mt_active - cdg_.delta * cache_.cur_dt - cdg_.sigma_v * cdg_.sigma_v / 2.0 * cache_.cur_dt -
                             vp_.sigma_r * vp_.sigma_r / (vp_.kappa * vp_.kappa) * (cache_.cur_dt - 2 * cache_.B_kappa_dt + cache_.B_2kappa_dt) -
                             cdg_.sigma_v * vp_.sigma_r * cb_.rho / vp_.kappa * (cache_.cur_dt - cache_.B_kappa_dt);

        n_active = h_active -
                   (cdg_.lamda * cache_.b * vp_.sigma_r * vp_.sigma_r) /
                       (vp_.kappa * (cdg_.lamda - vp_.kappa)) *
                       (cache_.B_kappa_dt - cache_.B_lambda_dt - cache_.B_2kappa_dt + cache_.B_lambda_plus_kappa_dt) +
                   cdg_.sigma_v * vp_.sigma_r * cb_.rho / vp_.kappa *
                       (cache_.B_lambda_dt - cache_.B_lambda_plus_kappa_dt);

        pt_active = Eigen::exp(-mt_active + 0.5 * vp_.sigma_r * vp_.sigma_r * cache_.b_k_square_integral);
    }

    /// @brief We use current_rows_ to determine the active data window, and we update the equity_result matrix in place.
    /// @param equity_result 
    void update_equity(Eigen::ArrayXXd &equity_result)
    {
        // 1. Map views
        auto m_act = m_arr.topRows(current_rows_);
        auto n_act = n_arr.topRows(current_rows_);
        auto res_act = equity_result.topRows(current_rows_);
        auto pt_act = pt_arr.head(current_rows_);
        auto v_act = v_t_arr.head(current_rows_);

        if (std::abs(cache_.corr_xy) < 1e-12)
        {
            // --- CASE 1: Zero Correlation ---
            auto t_A_act = term_A.topRows(current_rows_);
            auto t_B_act = term_B.topRows(current_rows_);
            auto x_act = x_arr.topRows(current_rows_);
            auto f_act = first_arr.topRows(current_rows_);
            auto s_act = second_arr.topRows(current_rows_);

            t_A_act = m_act + cache_.c_xx / 2.0;
            t_B_act = n_act + cache_.c_yy / 2.0;

            f_act = (1.0 - t_B_act.exp()) * (t_A_act.exp().colwise() * (v_act * pt_act));

            const double safe_c_yy_sqrt = std::max(std::sqrt(cache_.c_yy), 1e-12);
            x_act = n_act * (-1.0 / safe_c_yy_sqrt) - safe_c_yy_sqrt;
            s_act = (1.0 + (-1.702 * x_act).exp()).inverse();

            res_act = f_act * s_act;
            return;
        }

        // --- CASE 2: Non-Zero Correlation ---
        const double c_xx_sqrt = std::sqrt(cache_.c_xx);
        const double c_yy_sqrt = std::sqrt(cache_.c_yy);

        auto g_act = g_arr.topRows(current_rows_);
        auto e1_act = exp_part1_base.topRows(current_rows_);
        auto e2_act = exp_part2_base.topRows(current_rows_);
        auto d1s_act = d1_static.topRows(current_rows_);
        auto d2s_act = d2_static.topRows(current_rows_);
        auto int_act = integral_arr.topRows(current_rows_);
        auto d1_act = d1_arr.topRows(current_rows_);
        auto d2_act = d2_arr.topRows(current_rows_);

        g_act = -n_act / (c_yy_sqrt * cache_.corr_xy);
        const double h = std::sqrt(1.0 - cache_.corr_xy * cache_.corr_xy) / cache_.corr_xy;

        e1_act = Eigen::exp(m_act + cache_.c_xx / 2.0);

        const double term_sq_val = c_xx_sqrt + c_yy_sqrt * cache_.corr_xy;
        const double part_2_fourth = 0.5 * term_sq_val;
        
        e2_act = (n_act + m_act + part_2_fourth).exp();
        
        double d1_sign_h, d2_sign_h;
        if (cache_.corr_xy > 0)
        {
            d1s_act = g_act - c_xx_sqrt;
            d2s_act = g_act - term_sq_val;
            d1_sign_h = -h;
            d2_sign_h = -h;
        }
        else
        {
            d1s_act = c_xx_sqrt - g_act;
            d2s_act = d1s_act + c_yy_sqrt * cache_.corr_xy;
            d1_sign_h = h;
            d2_sign_h = h;
        }
        auto exp_base_d1 = term_A.topRows(current_rows_);
        auto exp_base_d2 = term_B.topRows(current_rows_);
        
        exp_base_d1 = Eigen::exp(-1.702 * d1s_act);
        exp_base_d2 = Eigen::exp(-1.702 * d2s_act);
        
        auto int_p1 = int_part1_arr.topRows(current_rows_);
        auto int_p2 = int_part2_arr.topRows(current_rows_);
        
        int_p1.setZero();
        int_p2.setZero();
        int_act.setZero();

        for (int i = 0; i < gh_rule_.x.size(); ++i)
        {
            const double z = gh_rule_.x(i) * SQRT_2;
            const double w = gh_rule_.w(i);

            // 1. Calculate the SCALAR exponentials (extremely fast)
            double scalar_d1_z = std::exp(-1.702 * d1_sign_h * z);
            double scalar_d2_z = std::exp(-1.702 * d2_sign_h * z);
            double scalar_exp_z = std::exp(c_yy_sqrt * z);

            // 2. Multiply the pre-calculated matrix by the scalar, and add 1.0
            // Notice: NO .exp() ON MATRICES INSIDE THE LOOP!
            // d1_act = (1.0 + (exp_base_d1 * scalar_d1_z)).inverse();
            // d2_act = (1.0 + (exp_base_d2 * scalar_d2_z)).inverse();

            int_p1 += w * (1.0 + (exp_base_d1 * scalar_d1_z)).inverse();
            int_p2 += (w * scalar_exp_z) * (1.0 + (exp_base_d2 * scalar_d2_z)).inverse();
        }
        int_act = (int_p1 * e1_act) - (int_p2 * e2_act);
        equity_result = int_act.colwise() * ((pt_act * v_act) / cb_.NS * INV_SQRT_PI);
    }
};