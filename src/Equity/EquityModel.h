#pragma once

#include <vector>
#include <Eigen/Dense>
#include "..\Pricing\CbModel.h"

double FormulaFastScalar(double M, double N, double Cxx, double Cyy, double corr_xy, const Eigen::ArrayXd &e_arr, const Eigen::ArrayXd &w_arr);

inline double Calc_Bk(double k, double t);

struct GaussHermiteResult
{
    Eigen::ArrayXd x; // Nodes (abscissas)
    Eigen::ArrayXd w; // Weights
};

GaussHermiteResult compute_gauss_hermite_rule(int n);

double CalculateEquityNode(
    double v_t, double l_t, double r_t, double dt,
    double theta_t, double theta_t1,
    const CbParas &cb_paras,
    const CdgParas &cdg_paras,
    const VasciekParas &vp,
    const GaussHermiteResult &gh_rule);

void EquityTreeBuild(const std::vector<std::array<double, 5>> &l_data,
                     const HullWhiteTreeResult &tree_result,
                     const CbParas &cb_paras,
                     const CdgParas &cdg_paras,
                     const VasciekParas &vasciek_paras,
                     const CouponPaidInfo &coupon_info);