#pragma once // Prevents this file from being included twice
#include <Eigen/Dense>
#include <array>

Eigen::ArrayX3d ProbCalc(int non, int j_max, double dt, double kappa);

struct VasciekParas
{
    double kappa;
    double r_bar;
    double sigma_r;
    double r0;
};

Eigen::ArrayXd VasciekZeroRates(const VasciekParas &params, const Eigen::ArrayXd &maturities);

Eigen::ArrayX3d ProbCalcV2(int non, int j_max, int n, const Eigen::ArrayX3d &prob_mat);

struct HullWhiteAlphaResult
{
    const Eigen::ArrayX3d prob_first;
    const Eigen::ArrayX3d prob_other;
    const Eigen::ArrayXd alphas;
    const Eigen::ArrayXd thetas;
};

Eigen::ArrayXXd HullWhiteFirstStage(
    const int n, const double sigma_r, const int j_max_first,
    const int j_max_other, const double dt_first, const double dt_other);

HullWhiteAlphaResult HullWhiteTreeAlpha(
    int n, int non_first, int non_other,
    double dt_first, double dt_other, double sigma_r,
    double kappa, const Eigen::ArrayXd &observed_zero_rates,
    int jmax_first, int jmax_other);

struct HullWhiteTreeResult
{
    const Eigen::ArrayXXd short_rate_tree;
    const HullWhiteAlphaResult alpha_result;
    const int j_max_first;
    const int j_max_other;
};

HullWhiteTreeResult HullWhiteTree(const double kappa, const double sigma_r,
                                  const Eigen::ArrayXd &zero_rates, const double dt_first, const double dt_other);

Eigen::ArrayXXd HullWhiteTreeShortRate(const int n, const int j_max_first, const int j_max_other,
                                       const Eigen::ArrayXd &alphas, const Eigen::ArrayXXd &r_tree);

struct PzTreeResult
{
    Eigen::ArrayX3i nxt_r_idx; // [Time] -> (Node -> {Down, Mid, Up})
    int start_h = -1;          // Initialize with invalid value to detect errors
};

PzTreeResult PzTreeBuild(
    int n,
    HullWhiteTreeResult &tree_result);
