#pragma once

#include <vector>
#include <Eigen/Dense>
#include "..\Pricing\CbModel.h"

typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ArrayXXdRowMajor;

double FormulaFastScalar(double M, double N, double Cxx, double Cyy, double corr_xy, const Eigen::ArrayXd &e_arr, const Eigen::ArrayXd &w_arr);

double Calc_Bk(double k, double t);

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
struct EquityContext
{
    const double c_xx;
    const double c_yy;
    const double c_xy;
    const double corr_xy;
    const Eigen::ArrayXXd m_arr;
    const Eigen::ArrayXXd n_arr;
    const Eigen::ArrayXd pt_arr;
};

EquityContext EquityContextVec(const double dt, const Eigen::ArrayXXd &l_t_arr,
                               const Eigen::ArrayXd &r_t_arr,
                               const Eigen::ArrayXd &theta_t_arr,
                               const Eigen::ArrayXd &theta_t1_arr,
                               const CbParas &cb_paras,
                               const CdgParas &cdg_paras,
                               const VasciekParas &vp,
                               const Eigen::ArrayXd &alpha_vec,
                               const Eigen::ArrayXd &t_vec);

Eigen::ArrayXXd EquityFunVec(
    const CbParas &cb_paras,
    const EquityContext &ctx,
    const Eigen::ArrayXd &v_t_arr);

struct EquityTreeBuildResult
{
    Eigen::ArrayXXd equity_tree;
    const Eigen::ArrayX3i idx_vec;
    const Eigen::ArrayXi nxt_m;
    const Eigen::Array<double, Eigen::Dynamic, 9> nxt_p;
    Eigen::ArrayXXd l_data_partition;
};

struct EquityTreeBuildResultMemoSave
{
    Eigen::ArrayXXd equity_tree;
    ArrayXXdRowMajor l_data_partition;
};

EquityTreeBuildResult EquityTreeBuild(const std::vector<LNode> &l_data,
                                      const HullWhiteTreeResult &tree_result,
                                      const CbParas &cb_paras,
                                      const CdgParas &cdg_paras,
                                      const std::vector<PNode> &next_p_data,
                                      const std::vector<MNode> &next_m_data,
                                      const VasciekParas &vasciek_paras,
                                      const CouponPaidInfo &coupon_info
                                      );

EquityTreeBuildResultMemoSave EquityTreeBuildMemoSave(
                                      const std::vector<PackedNode> &data,
                                      const HullWhiteTreeResult &tree_result,
                                      const CbParas &cb_paras,
                                      const CdgParas &cdg_paras,
                                      const VasciekParas &vasciek_paras,
                                      const CouponPaidInfo &coupon_info);