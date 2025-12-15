#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

#include "EquityModel.h"

double FormulaFastScalar(double M, double N, double Cxx, double Cyy, double corr_xy,
                         const Eigen::ArrayXd &e_arr,
                         const Eigen::ArrayXd &w_arr)
{
    const double SQRT2 = 1.4142135623730951;

    const Eigen::ArrayXd z_arr = e_arr * SQRT2;

    double sqrt_Cxx = std::sqrt(Cxx);
    double sqrt_Cyy = std::sqrt(Cyy);

    if (std::abs(corr_xy) < 1e-9)
    {
        double term_erf_arg = -N / sqrt_Cyy / SQRT2;
        double term_cdf = 0.5 * (1.0 + std::erf(term_erf_arg)); // 0.5 * (1 + erf) is normcdf equivalent

        double part1 = std::exp(M + Cxx / 2.0) * term_cdf;

        const Eigen::ArrayXd exponent_part2_arr = M + N + sqrt_Cyy * z_arr + (Cxx / 2.0);
        const Eigen::ArrayXd part2 = exponent_part2_arr.array().exp() * term_cdf;

        return ((part1 - part2) * w_arr).sum();
    }

    double g = (-N / sqrt_Cyy) / corr_xy;
    double h = std::sqrt(1.0 - corr_xy * corr_xy) / corr_xy;

    Eigen::ArrayXd d1, d2;

    if (corr_xy > 0)
    {
        const Eigen::ArrayXd common_term = g - h * z_arr;
        d1 = common_term - sqrt_Cxx;
        d2 = common_term - (sqrt_Cxx + sqrt_Cyy * corr_xy);
    }
    else // corr_xy < 0
    {
        const Eigen::ArrayXd common_term = g - h * z_arr;
        d1 = sqrt_Cxx - common_term;
        d2 = sqrt_Cxx + sqrt_Cyy * corr_xy - common_term;
    }

    Eigen::ArrayXd Phi_d1 = 0.5 * (1.0 + (d1 / SQRT2).array().erf());
    Eigen::ArrayXd Phi_d2 = 0.5 * (1.0 + (d2 / SQRT2).array().erf());

    Eigen::ArrayXd part1 = std::exp(M + Cxx / 2.0) * Phi_d1;

    Eigen::ArrayXd term_z = std::sqrt(Cyy * (1.0 - corr_xy * corr_xy)) * z_arr;
    double term_sq = (sqrt_Cxx + sqrt_Cyy * corr_xy);
    Eigen::ArrayXd exponent_part2 = M + N + term_z + (term_sq * term_sq) / 2.0;

    Eigen::ArrayXd part2 = Eigen::exp(exponent_part2) * Phi_d2;

    return ((part1 - part2) * w_arr).sum();
}